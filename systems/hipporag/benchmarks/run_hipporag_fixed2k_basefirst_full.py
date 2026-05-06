#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
import sys

BENCHMARK_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from evaluation.evaluator import Evaluator
from evaluation.llm_evaluator import LLMEvaluator
from init.config import Config
from memos_stats import sanitize_trace_record_for_export

from probe_hipporag_fixed2k_base_reuse import (
    base_cache_dataset_name,
    derive_item_cache_from_base,
    ensure_base_cache,
    run_q1_with_judge,
    sync_openai_env_aliases,
)
from run_hipporag_longmemeval import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FIXED2K_BASE_SOURCE_ROOT,
    DEFAULT_FIXED2K_MANIFEST,
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_LIST_ROOT,
    TRACE_FIELDS,
    build_trace_row,
    close_trace_writer,
    ensure_parent_trace,
    get_hipporag_runtime_knobs,
    load_corpus,
    load_question,
    make_dataset_dir,
    q0_ready,
    save_json,
    validate_clean_contract,
    validate_record_contract,
    normalize_clean_config,
    redact_answer_field_in_results,
    dedupe_preserve_order,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "runs" / "hipporag_fixed2k_main3m_basefirst_native_gpt5mini_judge4omini"


def parse_args() -> argparse.Namespace:
    default_dataset_lists = [
        str(DEFAULT_LIST_ROOT / "s0_all.txt"),
        str(DEFAULT_LIST_ROOT / "s100_all.txt"),
        str(DEFAULT_LIST_ROOT / "s200_all.txt"),
        str(DEFAULT_LIST_ROOT / "s300_all.txt"),
        str(DEFAULT_LIST_ROOT / "s400_all.txt"),
    ]
    parser = argparse.ArgumentParser(
        description="Run full fixed2k LongMemEval with HippoRAG base-first canonical q0."
    )
    parser.add_argument(
        "--dataset-lists",
        action="append",
        nargs="+",
        default=None,
        help="Canonical fixed2k list files. Defaults to s0/s100/s200/s300/s400 all.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--manifest-path", default=str(DEFAULT_FIXED2K_MANIFEST))
    parser.add_argument("--base-source-root", default=str(DEFAULT_FIXED2K_BASE_SOURCE_ROOT))
    parser.add_argument("--judge-config", default=str(DEFAULT_JUDGE_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument(
        "--q0-cache-llm-model",
        default="",
        help="Optional model label used only for q0/base+derived cache lookup/build. Defaults to --llm-model.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--top-session-limit", type=int, default=20)
    parser.add_argument("--enable-react-multihop", action="store_true")
    parser.add_argument("--react-max-steps", type=int, default=3)
    parser.add_argument("--react-max-context-chunks", type=int, default=12)
    parser.add_argument("--react-agent-max-tokens", type=int, default=512)
    parser.add_argument("--react-agent-temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Requested item-level concurrency per completed base.")
    parser.add_argument(
        "--disk-reserve-gb",
        type=float,
        default=8.0,
        help="Keep at least this many GB free while scheduling derived caches in parallel.",
    )
    parser.add_argument("--force-base-rebuild", action="store_true")
    parser.add_argument("--force-item-rerun", action="store_true")
    parser.add_argument("--retain-base-caches", action="store_true")
    parser.add_argument("--retain-derived-caches", action="store_true")
    parser.add_argument("--keep-answer-in-results", action="store_true")
    args = parser.parse_args()
    if args.dataset_lists:
        flattened: List[str] = []
        for group in args.dataset_lists:
            flattened.extend(group)
        args.dataset_lists = flattened
    else:
        args.dataset_lists = default_dataset_lists
    return args


def load_dataset_names(list_paths: List[Path]) -> List[str]:
    all_names: List[str] = []
    for path in list_paths:
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_names.extend(rows)
    return dedupe_preserve_order(all_names)


def load_manifest_index(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dataset_name = str(row.get("dataset_name", "") or "").strip()
        if dataset_name:
            index[dataset_name] = row
    return index


def group_datasets_by_base(manifest_index: Dict[str, Dict[str, Any]], dataset_names: List[str]) -> "OrderedDict[str, List[str]]":
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for dataset_name in dataset_names:
        row = manifest_index.get(dataset_name)
        if row is None:
            raise KeyError(f"dataset_name not found in manifest: {dataset_name}")
        base_rel_path = str(row["base_rel_path"])
        grouped.setdefault(base_rel_path, []).append(dataset_name)
    return grouped


def mark_summary_cache_removed(summary_path: Path, *, reason: str) -> None:
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    summary["cache_retained"] = False
    summary["cache_cleanup_reason"] = reason
    save_json(summary_path, summary)


def dataset_completed(dataset_dir: Path) -> bool:
    results_path = dataset_dir / "results" / "results.json"
    metrics_path = dataset_dir / "results" / "metrics.json"
    q0_summary_path = dataset_dir / "q0_index_summary.json"
    q0_cost_path = dataset_dir / "q0_cost_summary.json"
    return results_path.exists() and metrics_path.exists() and q0_summary_path.exists() and q0_cost_path.exists()


def dataset_q0_completed(dataset_dir: Path, *, llm_model: str, embedding_model: str) -> bool:
    q0_summary_path = dataset_dir / "q0_index_summary.json"
    q0_cost_path = dataset_dir / "q0_cost_summary.json"
    cache_dir = dataset_dir / "hipporag_q0_cache"
    return q0_summary_path.exists() and q0_cost_path.exists() and q0_ready(cache_dir, llm_model, embedding_model)


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                continue
    return total


def compute_safe_parallel_workers(
    *,
    requested_workers: int,
    base_cache_dir: Path,
    results_root: Path,
    disk_reserve_gb: float,
) -> Dict[str, Any]:
    requested = max(1, int(requested_workers or 1))
    reserve_bytes = int(max(0.0, float(disk_reserve_gb)) * (1024 ** 3))
    free_bytes = shutil.disk_usage(results_root).free
    base_cache_bytes = max(dir_size_bytes(base_cache_dir), 1)
    estimated_per_item_bytes = int(base_cache_bytes * 1.15)
    usable_bytes = max(0, free_bytes - reserve_bytes)
    safe_by_disk = max(1, usable_bytes // max(estimated_per_item_bytes, 1))
    effective = max(1, min(requested, safe_by_disk))
    return {
        "requested_workers": requested,
        "effective_workers": int(effective),
        "free_bytes": int(free_bytes),
        "reserve_bytes": int(reserve_bytes),
        "base_cache_bytes": int(base_cache_bytes),
        "estimated_per_item_bytes": int(estimated_per_item_bytes),
        "safe_workers_by_disk": int(safe_by_disk),
    }


def process_item_task(
    *,
    dataset_name: str,
    base_rel_path: str,
    data_root: Path,
    base_rows: List[Dict[str, Any]] | None,
    base_cache_dir: Path | None,
    results_root: Path,
    llm_model: str,
    q0_cache_llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    top_session_limit: int,
    enable_react_multihop: bool,
    react_max_steps: int,
    react_max_context_chunks: int,
    react_agent_max_tokens: int,
    react_agent_temperature: float,
    judge_config_path: Path,
    force_item_rerun: bool,
    keep_answer_in_results: bool,
    retain_derived_caches: bool,
) -> Dict[str, Any]:
    dataset_dir = make_dataset_dir(results_root / "derived", dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    derived_cache_dir = dataset_dir / "hipporag_q0_cache"
    q0_summary_path = dataset_dir / "q0_index_summary.json"
    q0_cost_path = dataset_dir / "q0_cost_summary.json"
    if dataset_completed(dataset_dir) and not force_item_rerun:
        return {
            "dataset_name": dataset_name,
            "status": "skipped_existing",
            "completion_entry": {
                "mode": "q0q1_basefirst",
                "dataset_name": dataset_name,
                "results_path": str(dataset_dir / "results" / "results.json"),
                "metrics_path": str(dataset_dir / "results" / "metrics.json"),
                "base_rel_path": base_rel_path,
                "cache_retained": bool(retain_derived_caches),
            },
        }

    target_rows = load_corpus(data_root / dataset_name / "Corpus.json")
    question_row = load_question(data_root / dataset_name / "Question.json")
    existing_q0_ready = dataset_q0_completed(
        dataset_dir,
        llm_model=q0_cache_llm_model,
        embedding_model=embedding_model,
    )
    if existing_q0_ready and not force_item_rerun:
        derived_summary = json.loads(q0_summary_path.read_text(encoding="utf-8"))
    else:
        if base_rows is None or base_cache_dir is None:
            raise RuntimeError(
                f"base cache unavailable for dataset requiring q0 derivation: {dataset_name}"
            )
        derived_summary = derive_item_cache_from_base(
            dataset_name=dataset_name,
            target_rows=target_rows,
            base_rows=base_rows,
            base_cache_dir=base_cache_dir,
            results_root=results_root,
            llm_model=q0_cache_llm_model,
            embedding_model=embedding_model,
            qa_top_k=qa_top_k,
            force=force_item_rerun,
        )

    clean_cfg = normalize_clean_config(Config.parse(judge_config_path))
    validate_clean_contract(clean_cfg)
    llm_evaluator = LLMEvaluator(clean_cfg, "", dataset_name.replace("/", "_"))
    record = run_q1_with_judge(
        dataset_name=dataset_name,
        question_row=question_row,
        cache_root=results_root / "derived",
        llm_model=llm_model,
        cache_llm_model=q0_cache_llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        top_session_limit=top_session_limit,
        enable_react_multihop=enable_react_multihop,
        react_max_steps=react_max_steps,
        react_max_context_chunks=react_max_context_chunks,
        react_agent_max_tokens=react_agent_max_tokens,
        react_agent_temperature=react_agent_temperature,
        judge_cfg=clean_cfg,
        llm_evaluator=llm_evaluator,
    )
    record = validate_record_contract(
        record,
        dataset_name=dataset_name,
        expected_judge_runs=clean_cfg.evaluation.eval_num_runs,
        require_judgments=True,
    )

    results_dir = dataset_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    metrics_path = results_dir / "metrics.json"
    save_json(results_path, [sanitize_trace_record_for_export(record)])
    metrics = asyncio.run(Evaluator(str(results_path), dataset_name, clean_cfg).evaluate())
    save_json(metrics_path, metrics)
    if not keep_answer_in_results:
        redact_answer_field_in_results(results_path)

    trace_row = build_trace_row(dataset_name, record)
    completion_entry = {
        "mode": "q0q1_basefirst",
        "dataset_name": dataset_name,
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "success": bool(record.get("success", False)),
        "llm_judge": bool(record.get("llm_judge", False)),
        "base_rel_path": base_rel_path,
        "q0_total_cost_usd": float(derived_summary["preprocessing_cost"]["total_cost_usd"]),
        "q1_total_cost_usd": float(record.get("total_cost_usd", 0.0) or 0.0),
        "cache_retained": bool(retain_derived_caches),
    }

    if not retain_derived_caches:
        derived_cache_dir = Path(derived_summary["cache_dir"])
        if derived_cache_dir.exists():
            shutil.rmtree(derived_cache_dir)
        mark_summary_cache_removed(
            dataset_dir / "q0_index_summary.json",
            reason="full_run_storage_control_derived_cache_removed_after_q1",
        )

    return {
        "dataset_name": dataset_name,
        "status": "completed",
        "trace_row": trace_row,
        "completion_entry": completion_entry,
        "record_judge": bool(record.get("llm_judge", False)),
        "record_cost_usd": float(record.get("total_cost_usd", 0.0) or 0.0),
    }


def main() -> None:
    args = parse_args()
    sync_openai_env_aliases()
    q0_cache_llm_model = str(args.q0_cache_llm_model or args.llm_model)

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    list_paths = [Path(p).resolve() for p in args.dataset_lists]
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    base_source_root = Path(args.base_source_root).resolve()
    judge_config_path = Path(args.judge_config).resolve()

    dataset_names = load_dataset_names(list_paths)
    manifest_index = load_manifest_index(manifest_path)
    grouped = group_datasets_by_base(manifest_index, dataset_names)

    checkpoint_path = results_root / "checkpoint_q1_full.json"
    trace_path = results_root / "trace_q1_full.csv"
    run_summary_path = results_root / "summary_q1_full.json"

    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            checkpoint = {"completed": {}, "failed": {}, "completed_bases": {}, "updated_at": None}
    else:
        checkpoint = {"completed": {}, "failed": {}, "completed_bases": {}, "updated_at": None}
    checkpoint.setdefault("completed", {})
    checkpoint.setdefault("failed", {})
    checkpoint.setdefault("completed_bases", {})

    clean_cfg = normalize_clean_config(Config.parse(judge_config_path))
    validate_clean_contract(clean_cfg)

    trace_writer = ensure_parent_trace(trace_path)
    completed = checkpoint["completed"]
    failed = checkpoint["failed"]
    completed_bases = checkpoint["completed_bases"]

    started_at = time.time()
    processed_items = 0

    for base_idx, (base_rel_path, base_dataset_names) in enumerate(grouped.items(), start=1):
        remaining_items = [
            name
            for name in base_dataset_names
            if args.force_item_rerun or not dataset_completed(make_dataset_dir(results_root / "derived", name))
        ]
        if not remaining_items and base_rel_path in completed_bases:
            print(f"[base {base_idx}/{len(grouped)}] skip completed base {base_rel_path}")
            continue

        print(f"[base {base_idx}/{len(grouped)}] building/reusing base {base_rel_path} for {len(base_dataset_names)} items")
        items_needing_q0 = []
        items_q1_only_retry = []
        for dataset_name in remaining_items:
            dataset_dir = make_dataset_dir(results_root / "derived", dataset_name)
            if dataset_q0_completed(dataset_dir, llm_model=args.llm_model, embedding_model=args.embedding_model) and not args.force_item_rerun:
                items_q1_only_retry.append(dataset_name)
            elif dataset_q0_completed(dataset_dir, llm_model=q0_cache_llm_model, embedding_model=args.embedding_model) and not args.force_item_rerun:
                items_q1_only_retry.append(dataset_name)
            else:
                items_needing_q0.append(dataset_name)

        base_rows: List[Dict[str, Any]] | None = None
        base_summary: Dict[str, Any] | None = None
        base_cache_dir: Path | None = None
        base_summary_path = (
            results_root
            / "base"
            / base_cache_dataset_name(base_rel_path)
            / "q0_index_summary.json"
        )

        if items_needing_q0:
            base_rows = load_corpus(base_source_root / base_rel_path / "Corpus.json")
            base_summary = ensure_base_cache(
                base_rel_path=base_rel_path,
                base_rows=base_rows,
                results_root=results_root,
                seed_results_root=results_root,
                seed_dataset_name="",
                llm_model=q0_cache_llm_model,
                embedding_model=args.embedding_model,
                qa_top_k=args.qa_top_k,
                force=args.force_base_rebuild,
            )
            base_cache_dir = Path(base_summary["cache_dir"])
            parallel_info = compute_safe_parallel_workers(
                requested_workers=args.workers,
                base_cache_dir=base_cache_dir,
                results_root=results_root,
                disk_reserve_gb=args.disk_reserve_gb,
            )
        else:
            print(f"  q1-only retry mode for {len(items_q1_only_retry)} items with existing derived q0 caches")
            parallel_info = {
                "requested_workers": int(args.workers),
                "effective_workers": int(max(1, args.workers)),
                "safe_workers_by_disk": int(max(1, args.workers)),
                "base_cache_bytes": 0,
                "free_bytes": int(shutil.disk_usage(results_root).free),
            }
        print(
            "  item concurrency "
            f"requested={parallel_info['requested_workers']} "
            f"effective={parallel_info['effective_workers']} "
            f"safe_by_disk={parallel_info['safe_workers_by_disk']} "
            f"base_cache_mb={parallel_info['base_cache_bytes'] / (1024 ** 2):.1f} "
            f"free_gb={parallel_info['free_bytes'] / (1024 ** 3):.2f}"
        )

        futures = {}
        with ThreadPoolExecutor(max_workers=parallel_info["effective_workers"]) as executor:
            for item_idx, dataset_name in enumerate(base_dataset_names, start=1):
                dataset_dir = make_dataset_dir(results_root / "derived", dataset_name)
                dataset_dir.mkdir(parents=True, exist_ok=True)
                if dataset_completed(dataset_dir) and not args.force_item_rerun:
                    processed_items += 1
                    completed[dataset_name] = {
                        "mode": "q0q1_basefirst",
                        "dataset_name": dataset_name,
                        "results_path": str(dataset_dir / "results" / "results.json"),
                        "metrics_path": str(dataset_dir / "results" / "metrics.json"),
                        "base_rel_path": base_rel_path,
                        "cache_retained": bool(args.retain_derived_caches),
                    }
                    failed.pop(dataset_name, None)
                    continue

                print(f"  queue [{item_idx}/{len(base_dataset_names)}] {dataset_name}")
                future = executor.submit(
                    process_item_task,
                    dataset_name=dataset_name,
                    base_rel_path=base_rel_path,
                    data_root=data_root,
                    base_rows=base_rows,
                    base_cache_dir=base_cache_dir,
                    results_root=results_root,
                    llm_model=args.llm_model,
                    q0_cache_llm_model=q0_cache_llm_model,
                    embedding_model=args.embedding_model,
                    qa_top_k=args.qa_top_k,
                    top_session_limit=args.top_session_limit,
                    enable_react_multihop=bool(args.enable_react_multihop),
                    react_max_steps=int(args.react_max_steps),
                    react_max_context_chunks=int(args.react_max_context_chunks),
                    react_agent_max_tokens=int(args.react_agent_max_tokens),
                    react_agent_temperature=float(args.react_agent_temperature),
                    judge_config_path=judge_config_path,
                    force_item_rerun=bool(args.force_item_rerun),
                    keep_answer_in_results=bool(args.keep_answer_in_results),
                    retain_derived_caches=bool(args.retain_derived_caches),
                )
                futures[future] = dataset_name

            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    failed[dataset_name] = {"error": str(exc), "dataset_name": dataset_name, "base_rel_path": base_rel_path}
                    checkpoint["updated_at"] = time.time()
                    save_json(checkpoint_path, checkpoint)
                    print(f"    failed: {dataset_name} -> {exc}")
                    continue

                processed_items += 1
                status = str(payload.get("status", ""))
                if status in {"completed", "skipped_existing"}:
                    completed[dataset_name] = dict(payload["completion_entry"])
                    failed.pop(dataset_name, None)
                    trace_row = payload.get("trace_row")
                    if trace_row:
                        trace_writer.writerow(trace_row)
                        getattr(trace_writer, "_codex_file_handle").flush()  # type: ignore[attr-defined]
                    checkpoint["updated_at"] = time.time()
                    save_json(checkpoint_path, checkpoint)
                    if status == "completed":
                        print(
                            f"    done {dataset_name} judge={int(bool(payload.get('record_judge', False)))} "
                            f"q1_cost={float(payload.get('record_cost_usd', 0.0) or 0.0):.6f}"
                        )
                else:
                    failed[dataset_name] = {
                        "error": str(payload.get("error", "unknown_error")),
                        "dataset_name": dataset_name,
                        "base_rel_path": base_rel_path,
                    }
                    checkpoint["updated_at"] = time.time()
                    save_json(checkpoint_path, checkpoint)
                    print(f"    failed: {dataset_name} -> {payload.get('error', 'unknown_error')}")

        remaining_failed_items = [name for name in base_dataset_names if name in failed]
        retain_base_cache_for_retry = bool(args.retain_base_caches) or bool(remaining_failed_items)
        base_q0_total_cost_usd = 0.0
        if base_summary is not None:
            base_q0_total_cost_usd = float(base_summary["preprocessing_cost"]["total_cost_usd"])
        else:
            base_q0_total_cost_usd = float(completed_bases.get(base_rel_path, {}).get("base_q0_total_cost_usd", 0.0) or 0.0)

        completed_bases[base_rel_path] = {
            "base_rel_path": base_rel_path,
            "item_count": len(base_dataset_names),
            "remaining_failed_items": remaining_failed_items,
            "base_q0_total_cost_usd": base_q0_total_cost_usd,
            "cache_retained": retain_base_cache_for_retry,
        }
        checkpoint["updated_at"] = time.time()
        save_json(checkpoint_path, checkpoint)

        if not retain_base_cache_for_retry and base_cache_dir is not None:
            if base_cache_dir.exists():
                shutil.rmtree(base_cache_dir)
            mark_summary_cache_removed(
                base_summary_path,
                reason="full_run_storage_control_base_cache_removed_after_all_items_finished",
            )

    finished_at = time.time()
    close_trace_writer(trace_writer)

    run_summary = {
        "dataset_lists": [str(p) for p in list_paths],
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "base_source_root": str(base_source_root),
        "results_root": str(results_root),
        "judge_config": str(judge_config_path),
        "judge_model": clean_cfg.evaluation.eval_model,
        "judge_prompt_style": clean_cfg.evaluation.eval_prompt_style,
        "judge_num_runs": clean_cfg.evaluation.eval_num_runs,
        "llm_model": args.llm_model,
        "q0_cache_llm_model": q0_cache_llm_model,
        "embedding_model": args.embedding_model,
        "requested_workers": int(args.workers),
        "disk_reserve_gb": float(args.disk_reserve_gb),
        "hipporag_runtime": get_hipporag_runtime_knobs(
            qa_top_k=args.qa_top_k,
            top_session_limit=args.top_session_limit,
        ),
        "canonical_data_root": str(DEFAULT_DATA_ROOT),
        "canonical_list_root": str(DEFAULT_LIST_ROOT),
        "canonical_manifest_path": str(DEFAULT_FIXED2K_MANIFEST),
        "canonical_base_source_root": str(DEFAULT_FIXED2K_BASE_SOURCE_ROOT),
        "q0_definition": "base_first_cache_derivation_canonical_q0",
        "retain_base_caches": bool(args.retain_base_caches),
        "retain_derived_caches": bool(args.retain_derived_caches),
        "trace_path": str(trace_path),
        "checkpoint_path": str(checkpoint_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": round(finished_at - started_at, 3),
        "processed_items": processed_items,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_bases": completed_bases,
        "failed_datasets": failed,
    }
    save_json(run_summary_path, run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
