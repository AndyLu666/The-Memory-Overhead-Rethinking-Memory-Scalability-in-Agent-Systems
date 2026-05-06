#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

BENCHMARK_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

Evaluator = None
LLMEvaluator = None
Config = None
sanitize_trace_record_for_export = None


def _load_runtime_dependencies() -> None:
    global Evaluator
    global LLMEvaluator
    global Config
    global sanitize_trace_record_for_export
    if Config is not None:
        return

    from evaluation.evaluator import Evaluator as _Evaluator
    from evaluation.llm_evaluator import LLMEvaluator as _LLMEvaluator
    from init.config import Config as _Config
    from memos_stats import sanitize_trace_record_for_export as _sanitize_trace_record_for_export

    Evaluator = _Evaluator
    LLMEvaluator = _LLMEvaluator
    Config = _Config
    sanitize_trace_record_for_export = _sanitize_trace_record_for_export

from run_hipporag_longmemeval import (
    TRACE_FIELDS,
    _load_runtime_dependencies as load_longmemeval_runtime_dependencies,
    build_trace_row,
    close_trace_writer,
    ensure_parent_trace,
    load_corpus,
    load_doc_metadata,
    load_question,
    normalize_clean_config,
    q0_ready,
    query_with_hipporag,
    redact_answer_field_in_results,
    run_q0_for_dataset,
    save_json,
    validate_clean_contract,
    validate_record_contract,
    judge_record,
)


DEFAULT_DATA_ROOT = (
    REPO_ROOT / "data" / "locomo_multihop282_fixedgroup_sbins_lmealign_20260315"
)
DEFAULT_LIST_ROOT = (
    LICOMEMORY_ROOT / "scripts" / "dataset_lists" / "locomo_multihop282_fixedgroup_sbins_lmealign_20260315"
)
DEFAULT_MANIFEST_PATH = DEFAULT_LIST_ROOT / "manifest.jsonl"
DEFAULT_JUDGE_CONFIG = (
    SYSTEM_ROOT / "config" / "hipporag_locomo_eval_only_gpt4omini_memos_20260403.yaml"
)
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "runs"
    / "hipporag_locomo_multihop282_fixedgroup_lmealign"
)

DEFAULT_FORMAL_LISTS = [
    str(DEFAULT_LIST_ROOT / "s000_r01_all.txt"),
    str(DEFAULT_LIST_ROOT / "s100_r01_all.txt"),
    str(DEFAULT_LIST_ROOT / "s200_r01_all.txt"),
    str(DEFAULT_LIST_ROOT / "s300_r01_all.txt"),
    str(DEFAULT_LIST_ROOT / "s400_r01_all.txt"),
]


_Q1_THREAD_STATE = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HippoRAG on canonical LoCoMo fixedgroup sbins with representative q0 reuse and MemOS-aligned q1."
    )
    parser.add_argument(
        "--dataset-lists",
        action="append",
        nargs="+",
        default=None,
        help="Question list files. Defaults to all five canonical stage all lists.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--list-root", default=str(DEFAULT_LIST_ROOT))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--judge-config", default=str(DEFAULT_JUDGE_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--mode", choices=["q0", "q1", "q0q1"], default="q0q1")
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-32B",
        help="Question-time HippoRAG QA model. q0 still uses --q0-cache-llm-model.",
    )
    parser.add_argument(
        "--q0-cache-llm-model",
        default="gpt-5-mini",
        help="Representative q0 build model label.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--top-session-limit", type=int, default=20)
    parser.add_argument(
        "--q0-workers",
        type=int,
        default=1,
        help="Representative q0 build concurrency. Default keeps q0 conservative.",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--force-base-rebuild", action="store_true")
    parser.add_argument("--force-item-rerun", action="store_true")
    parser.add_argument("--keep-answer-in-results", action="store_true")
    args = parser.parse_args()
    if args.dataset_lists:
        flattened: List[str] = []
        for group in args.dataset_lists:
            flattened.extend(group)
        args.dataset_lists = flattened
    else:
        args.dataset_lists = list(DEFAULT_FORMAL_LISTS)
    return args


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_manifest_index(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(manifest_path):
        rel_path = str(row.get("rel_path", "") or "").strip()
        if rel_path:
            index[rel_path] = row
    return index


def load_dataset_names(list_paths: Iterable[Path]) -> List[str]:
    seen = set()
    out: List[str] = []
    for path in list_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            dataset_name = str(line or "").strip()
            if not dataset_name or dataset_name in seen:
                continue
            seen.add(dataset_name)
            out.append(dataset_name)
    return out


def locomo_stage(dataset_name: str) -> str:
    parts = [part for part in str(dataset_name or "").split("/") if part]
    return parts[0] if parts else ""


def validate_locomo_dataset_name(dataset_name: str, manifest_index: Mapping[str, Dict[str, Any]]) -> None:
    row = manifest_index.get(dataset_name)
    if row is None:
        raise RuntimeError(f"dataset_name_not_in_locomo_manifest: {dataset_name}")
    if str(row.get("question_type", "") or "").strip() != "locomo-multi-hop":
        raise RuntimeError(f"non_locomo_multihop_dataset: {dataset_name}")
    if str(row.get("memory_unit", "") or "").strip() != "group":
        raise RuntimeError(f"non_group_memory_unit_dataset: {dataset_name}")
    if str(row.get("noise_mode", "") or "").strip() != "outdomain_longmemeval":
        raise RuntimeError(f"noncanonical_noise_mode_dataset: {dataset_name}")
    if str(row.get("alignment_target", "") or "").strip() != "longmemeval_fixed2k":
        raise RuntimeError(f"noncanonical_alignment_target_dataset: {dataset_name}")


def stage_to_representative_list(list_root: Path, stage: str) -> Path:
    return list_root / f"{stage}_representatives.txt"


def group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("stage", "") or ""), str(row.get("group_id", "") or ""))


def ordered_stage_groups(
    dataset_names: Iterable[str],
    manifest_index: Mapping[str, Dict[str, Any]],
) -> "OrderedDict[tuple[str, str], Dict[str, Any]]":
    grouped: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
    for dataset_name in dataset_names:
        row = manifest_index[dataset_name]
        key = group_key(row)
        entry = grouped.setdefault(
            key,
            {
                "stage": str(row.get("stage", "") or ""),
                "group_id": str(row.get("group_id", "") or ""),
                "representative_rel_path": str(row.get("representative_rel_path", "") or ""),
                "dataset_names": [],
            },
        )
        rep_rel_path = str(row.get("representative_rel_path", "") or "")
        if rep_rel_path and not entry["representative_rel_path"]:
            entry["representative_rel_path"] = rep_rel_path
        entry["dataset_names"].append(dataset_name)
    return grouped


def representative_groups_for_stages(
    *,
    stages: Iterable[str],
    list_root: Path,
    manifest_index: Mapping[str, Dict[str, Any]],
    dataset_names: Iterable[str],
) -> "OrderedDict[tuple[str, str], Dict[str, Any]]":
    wanted = set(str(stage or "").strip() for stage in stages if str(stage or "").strip())
    selected = set(dataset_names)
    grouped: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
    for stage in sorted(wanted):
        rep_list_path = stage_to_representative_list(list_root, stage)
        rep_names = [
            line.strip()
            for line in rep_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for rep_name in rep_names:
            row = manifest_index[rep_name]
            key = group_key(row)
            selected_names = [
                name
                for name in selected
                if group_key(manifest_index[name]) == key
            ]
            if not selected_names:
                continue
            grouped[key] = {
                "stage": str(row.get("stage", "") or ""),
                "group_id": str(row.get("group_id", "") or ""),
                "representative_rel_path": rep_name,
                "dataset_names": selected_names,
            }
    return grouped


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return "symlink"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return "copy"


def ensure_q0_links_for_group(
    *,
    results_root: Path,
    representative_rel_path: str,
    dataset_names: Iterable[str],
) -> Dict[str, Any]:
    rep_dir = results_root / representative_rel_path
    rep_cache_dir = rep_dir / "hipporag_q0_cache"
    rep_summary_path = rep_dir / "q0_index_summary.json"
    rep_cost_path = rep_dir / "q0_cost_summary.json"
    link_modes: List[str] = []
    linked = 0
    for dataset_name in dataset_names:
        dataset_dir = results_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if dataset_name == representative_rel_path:
            continue
        link_modes.append(link_or_copy(rep_cache_dir, dataset_dir / "hipporag_q0_cache"))
        link_modes.append(link_or_copy(rep_summary_path, dataset_dir / "q0_index_summary.json"))
        link_modes.append(link_or_copy(rep_cost_path, dataset_dir / "q0_cost_summary.json"))
        linked += 1
    return {
        "representative_rel_path": representative_rel_path,
        "linked_question_count": linked,
        "link_modes": sorted(set(link_modes)),
    }


def q0_assets_ready(dataset_dir: Path, q0_cache_llm_model: str, embedding_model: str) -> bool:
    return (
        q0_ready(dataset_dir / "hipporag_q0_cache", q0_cache_llm_model, embedding_model)
        and (dataset_dir / "q0_index_summary.json").exists()
        and (dataset_dir / "q0_cost_summary.json").exists()
    )


def sync_openai_env_aliases() -> None:
    if "GPT_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_API_KEY"]
    if "OPENAI_API_KEY" in os.environ and "GPT_API_KEY" not in os.environ:
        os.environ["GPT_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if "GPT_BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_BASE_URL"]
    if "OPENAI_BASE_URL" in os.environ and "GPT_BASE_URL" not in os.environ:
        os.environ["GPT_BASE_URL"] = os.environ["OPENAI_BASE_URL"]


@contextmanager
def scoped_env(overrides: Mapping[str, Optional[str]]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def require_qwen_env() -> Dict[str, str]:
    api_key = (
        os.getenv("QWEN_API", "").strip()
        or os.getenv("HIPPORAG_Q1_API_KEY", "").strip()
        or os.getenv("OTHER_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("OTHER_BASE_URL", "").strip()
        or os.getenv("HIPPORAG_Q1_BASE_URL", "").strip()
        or os.getenv("HIPPORAG_LLM_BASE_URL", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing QWEN_API/OTHER_API_KEY for q1 model.")
    if not base_url:
        raise RuntimeError("Missing OTHER_BASE_URL/HIPPORAG_Q1_BASE_URL for q1 model.")
    return {"api_key": api_key, "base_url": base_url}


def require_gpt_env() -> Dict[str, str]:
    api_key = (
        os.getenv("GPT_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("GPT_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    if not api_key:
        raise RuntimeError("Missing GPT_API_KEY/OPENAI_API_KEY for q0 build.")
    if not base_url:
        raise RuntimeError("Missing GPT_BASE_URL/OPENAI_BASE_URL for q0 build.")
    return {"api_key": api_key, "base_url": base_url}


def build_q0_prepare_summary(
    *,
    results_root: Path,
    dataset_names: Iterable[str],
    q0_cache_llm_model: str,
    embedding_model: str,
    manifest_index: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    ready = 0
    linked = 0
    missing_or_bad: Dict[str, str] = {}
    reps = set()
    for dataset_name in dataset_names:
        row = manifest_index[dataset_name]
        rep_rel_path = str(row.get("representative_rel_path", "") or "")
        if rep_rel_path:
            reps.add(rep_rel_path)
        dataset_dir = results_root / dataset_name
        if not q0_assets_ready(dataset_dir, q0_cache_llm_model, embedding_model):
            missing_or_bad[dataset_name] = "q0_assets_missing_or_incomplete"
            continue
        ready += 1
        if dataset_name != rep_rel_path:
            linked += 1
    summary = {
        "results_root": str(results_root),
        "dataset_count": len(list(dataset_names)),
        "ready_count": ready,
        "linked_count": linked,
        "representative_count": len(reps),
        "missing_or_bad": missing_or_bad,
        "q0_cache_llm_model": q0_cache_llm_model,
        "embedding_model": embedding_model,
    }
    save_json(results_root / "base_reuse_prepare_summary.json", summary)
    return summary


def stage_completion_count(
    *,
    dataset_names: Iterable[str],
    completed: Mapping[str, Any],
) -> int:
    by_stage: Dict[str, set[str]] = {}
    for dataset_name in dataset_names:
        by_stage.setdefault(locomo_stage(dataset_name), set()).add(dataset_name)
    count = 0
    for names in by_stage.values():
        if all(name in completed for name in names):
            count += 1
    return count


def enrich_locomo_record(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    question_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> Dict[str, Any]:
    enriched = dict(record)
    enriched["id"] = str(question_row.get("question_id", "") or "")
    for key in [
        "sample_id",
        "bin_s",
        "group_id",
        "group_index",
        "question_index",
        "question_type_name",
        "category_id",
        "category_name",
        "answer_source",
        "memory_unit",
        "noise_mode",
        "alignment_target",
        "native_group_session_count",
        "native_nonorigin_count",
        "injected_session_count",
        "total_session_count",
        "external_filler_root",
        "external_filler_unique_session_count",
        "replica",
        "stage",
        "evidence",
    ]:
        if key in question_row:
            enriched[key] = question_row.get(key)
    enriched["representative_rel_path"] = str(
        manifest_row.get("representative_rel_path", "")
        or question_row.get("group_representative_rel_path", "")
        or ""
    )
    enriched["group_representative_rel_path"] = str(
        question_row.get("group_representative_rel_path", "")
        or manifest_row.get("representative_rel_path", "")
        or ""
    )
    if enriched.get("s") in (None, "") and enriched.get("bin_s") not in (None, ""):
        try:
            enriched["s"] = int(enriched["bin_s"])
        except Exception:
            enriched["s"] = enriched["bin_s"]
    return enriched


def dataset_completed(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / "results" / "results.json").exists()
        and (dataset_dir / "results" / "metrics.json").exists()
    )


def get_thread_local_llm_evaluator(judge_cfg: Config) -> LLMEvaluator:
    evaluator = getattr(_Q1_THREAD_STATE, "locomo_llm_evaluator", None)
    if evaluator is None:
        evaluator = LLMEvaluator(judge_cfg, "", "locomo_full")
        _Q1_THREAD_STATE.locomo_llm_evaluator = evaluator
    return evaluator


def process_q1_item(
    *,
    dataset_name: str,
    results_root: Path,
    data_root: Path,
    q1_model: str,
    q0_cache_llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    top_session_limit: int,
    judge_cfg: Config,
    manifest_row: Mapping[str, Any],
    keep_answer_in_results: bool,
    force_item_rerun: bool,
) -> Dict[str, Any]:
    dataset_dir = results_root / dataset_name
    if dataset_completed(dataset_dir) and not force_item_rerun:
        return {
            "dataset_name": dataset_name,
            "status": "skipped_existing",
            "record": None,
        }

    question_row = load_question(data_root / dataset_name / "Question.json")
    doc_metadata = load_doc_metadata(dataset_dir / "hipporag_q0_cache" / "doc_metadata.json")
    record = query_with_hipporag(
        dataset_name=dataset_name,
        question_row=question_row,
        doc_metadata=doc_metadata,
        results_root=results_root,
        llm_model=q1_model,
        cache_llm_model=q0_cache_llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        top_session_limit=top_session_limit,
        enable_react_multihop=False,
    )
    record = enrich_locomo_record(
        record,
        dataset_name=dataset_name,
        question_row=question_row,
        manifest_row=manifest_row,
    )
    record = validate_record_contract(
        record,
        dataset_name=dataset_name,
        expected_judge_runs=judge_cfg.evaluation.eval_num_runs,
        require_judgments=False,
    )
    llm_evaluator = get_thread_local_llm_evaluator(judge_cfg)
    record = asyncio.run(judge_record(llm_evaluator, record))
    record = validate_record_contract(
        record,
        dataset_name=dataset_name,
        expected_judge_runs=judge_cfg.evaluation.eval_num_runs,
        require_judgments=True,
    )

    results_dir = dataset_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    metrics_path = results_dir / "metrics.json"
    save_json(results_path, [sanitize_trace_record_for_export(record)])
    metrics = asyncio.run(Evaluator(str(results_path), dataset_name, judge_cfg).evaluate())
    save_json(metrics_path, metrics)
    if not keep_answer_in_results:
        redact_answer_field_in_results(results_path)

    return {
        "dataset_name": dataset_name,
        "status": "completed",
        "record": record,
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
    }


def process_q0_group(
    *,
    entry: Mapping[str, Any],
    results_root: Path,
    data_root: Path,
    q0_cache_llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    force_base_rebuild: bool,
) -> Dict[str, Any]:
    representative_rel_path = str(entry["representative_rel_path"])
    corpus_rows = load_corpus(data_root / representative_rel_path / "Corpus.json")
    q0_summary = run_q0_for_dataset(
        dataset_name=representative_rel_path,
        corpus_rows=corpus_rows,
        results_root=results_root,
        llm_model=q0_cache_llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        force_reindex=bool(force_base_rebuild),
    )
    link_summary = ensure_q0_links_for_group(
        results_root=results_root,
        representative_rel_path=representative_rel_path,
        dataset_names=entry["dataset_names"],
    )
    return {
        "entry": dict(entry),
        "q0_summary": q0_summary,
        "link_summary": link_summary,
    }


def main() -> None:
    args = parse_args()
    load_longmemeval_runtime_dependencies()
    _load_runtime_dependencies()
    sync_openai_env_aliases()

    data_root = Path(args.data_root).resolve()
    list_root = Path(args.list_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    results_root = Path(args.results_root).resolve()
    judge_config_path = Path(args.judge_config).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    list_paths = [Path(path).resolve() for path in args.dataset_lists]
    dataset_names = load_dataset_names(list_paths)
    if not dataset_names:
        raise RuntimeError("No LoCoMo dataset names selected.")

    manifest_index = load_manifest_index(manifest_path)
    for dataset_name in dataset_names:
        validate_locomo_dataset_name(dataset_name, manifest_index)

    stages = sorted({locomo_stage(name) for name in dataset_names})
    rep_groups = representative_groups_for_stages(
        stages=stages,
        list_root=list_root,
        manifest_index=manifest_index,
        dataset_names=dataset_names,
    )

    q0_checkpoint_path = results_root / "checkpoint_q0_groups.json"
    q0_summary_path = results_root / "summary_q0_groups.json"
    q1_checkpoint_path = results_root / "checkpoint_q1_full.json"
    q1_summary_path = results_root / "summary_q1_full.json"
    q1_trace_path = results_root / "trace_q1_full.csv"

    if q0_checkpoint_path.exists():
        try:
            q0_checkpoint = json.loads(q0_checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            q0_checkpoint = {"completed": {}, "failed": {}, "linked_datasets": {}, "updated_at": None}
    else:
        q0_checkpoint = {"completed": {}, "failed": {}, "linked_datasets": {}, "updated_at": None}
    q0_checkpoint.setdefault("completed", {})
    q0_checkpoint.setdefault("failed", {})
    q0_checkpoint.setdefault("linked_datasets", {})

    if q1_checkpoint_path.exists():
        try:
            q1_checkpoint = json.loads(q1_checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            q1_checkpoint = {"completed": {}, "failed": {}, "completed_stages": 0, "updated_at": None}
    else:
        q1_checkpoint = {"completed": {}, "failed": {}, "completed_stages": 0, "updated_at": None}
    q1_checkpoint.setdefault("completed", {})
    q1_checkpoint.setdefault("failed", {})
    q1_checkpoint.setdefault("completed_stages", 0)

    q0_started_at = time.time()
    if args.mode in {"q0", "q0q1"}:
        gpt_env = require_gpt_env()
        with scoped_env(
            {
                "HIPPORAG_LLM_API_KEY": gpt_env["api_key"],
                "HIPPORAG_LLM_BASE_URL": gpt_env["base_url"],
                "HIPPORAG_Q1_API_KEY": None,
                "HIPPORAG_Q1_BASE_URL": None,
            }
        ):
            with ThreadPoolExecutor(max_workers=max(1, int(args.q0_workers))) as pool:
                futures = {
                    pool.submit(
                        process_q0_group,
                        entry=entry,
                        results_root=results_root,
                        data_root=data_root,
                        q0_cache_llm_model=args.q0_cache_llm_model,
                        embedding_model=args.embedding_model,
                        qa_top_k=args.qa_top_k,
                        force_base_rebuild=bool(args.force_base_rebuild),
                    ): (key, entry)
                    for key, entry in rep_groups.items()
                }
                for future in as_completed(futures):
                    _, entry = futures[future]
                    representative_rel_path = str(entry["representative_rel_path"])
                    dataset_dir = results_root / representative_rel_path
                    try:
                        outcome = future.result()
                        q0_summary = outcome["q0_summary"]
                        link_summary = outcome["link_summary"]
                        entry = outcome["entry"]
                    except Exception as exc:
                        q0_checkpoint["failed"][representative_rel_path] = {
                            "stage": entry["stage"],
                            "group_id": entry["group_id"],
                            "error": str(exc),
                        }
                        q0_checkpoint["updated_at"] = time.time()
                        save_json(q0_checkpoint_path, q0_checkpoint)
                        raise
                    q0_checkpoint["completed"][representative_rel_path] = {
                        "stage": entry["stage"],
                        "group_id": entry["group_id"],
                        "representative_rel_path": representative_rel_path,
                        "linked_question_count": int(link_summary["linked_question_count"]),
                        "cache_reused": bool(q0_summary.get("cache_reused", False)),
                        "q0_cost_summary_path": str(dataset_dir / "q0_cost_summary.json"),
                    }
                    for dataset_name in entry["dataset_names"]:
                        q0_checkpoint["linked_datasets"][dataset_name] = {
                            "representative_rel_path": representative_rel_path,
                            "stage": entry["stage"],
                            "group_id": entry["group_id"],
                        }
                    q0_checkpoint["failed"].pop(representative_rel_path, None)
                    q0_checkpoint["updated_at"] = time.time()
                    save_json(q0_checkpoint_path, q0_checkpoint)

        q0_prepare_summary = build_q0_prepare_summary(
            results_root=results_root,
            dataset_names=dataset_names,
            q0_cache_llm_model=args.q0_cache_llm_model,
            embedding_model=args.embedding_model,
            manifest_index=manifest_index,
        )
        q0_summary = {
            "mode": "q0",
            "results_root": str(results_root),
            "dataset_count": len(dataset_names),
            "stage_count": len(stages),
            "representative_count": len(rep_groups),
            "completed_representatives": len(q0_checkpoint["completed"]),
            "failed_representatives": dict(q0_checkpoint["failed"]),
            "q0_cache_llm_model": args.q0_cache_llm_model,
            "embedding_model": args.embedding_model,
            "started_at": q0_started_at,
            "finished_at": time.time(),
            "duration_sec": round(time.time() - q0_started_at, 3),
            "base_reuse_prepare_summary": q0_prepare_summary,
        }
        save_json(q0_summary_path, q0_summary)
        if args.mode == "q0":
            print(json.dumps(q0_summary, ensure_ascii=False, indent=2))
            return

    q0_prepare_summary = build_q0_prepare_summary(
        results_root=results_root,
        dataset_names=dataset_names,
        q0_cache_llm_model=args.q0_cache_llm_model,
        embedding_model=args.embedding_model,
        manifest_index=manifest_index,
    )
    if q0_prepare_summary["missing_or_bad"]:
        raise RuntimeError(
            f"LoCoMo q0 reuse not ready for q1: {len(q0_prepare_summary['missing_or_bad'])} datasets missing or bad"
        )

    qwen_env = require_qwen_env()
    judge_cfg = normalize_clean_config(Config.parse(judge_config_path))
    validate_clean_contract(judge_cfg)
    trace_writer = ensure_parent_trace(q1_trace_path)
    pending_dataset_names = [
        dataset_name
        for dataset_name in dataset_names
        if bool(args.force_item_rerun) or not dataset_completed(results_root / dataset_name)
    ]

    q1_started_at = time.time()
    with scoped_env(
        {
            "HIPPORAG_LLM_API_KEY": qwen_env["api_key"],
            "HIPPORAG_LLM_BASE_URL": qwen_env["base_url"],
        }
    ):
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {
                pool.submit(
                    process_q1_item,
                    dataset_name=dataset_name,
                    results_root=results_root,
                    data_root=data_root,
                    q1_model=args.llm_model,
                    q0_cache_llm_model=args.q0_cache_llm_model,
                    embedding_model=args.embedding_model,
                    qa_top_k=args.qa_top_k,
                    top_session_limit=args.top_session_limit,
                    judge_cfg=judge_cfg,
                    manifest_row=manifest_index[dataset_name],
                    keep_answer_in_results=bool(args.keep_answer_in_results),
                    force_item_rerun=bool(args.force_item_rerun),
                ): dataset_name
                for dataset_name in pending_dataset_names
            }
            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    outcome = future.result()
                    if outcome["status"] == "completed":
                        record = dict(outcome["record"])
                        trace_row = build_trace_row(dataset_name, record)
                        trace_writer.writerow(trace_row)
                        getattr(trace_writer, "_codex_file_handle").flush()  # type: ignore[attr-defined]
                        q1_checkpoint["completed"][dataset_name] = {
                            "dataset_name": dataset_name,
                            "results_path": outcome["results_path"],
                            "metrics_path": outcome["metrics_path"],
                            "success": bool(record.get("success", False)),
                            "llm_judge": bool(record.get("llm_judge", False)),
                            "stage": str(record.get("stage", "") or locomo_stage(dataset_name)),
                        }
                    elif outcome["status"] == "skipped_existing":
                        q1_checkpoint["completed"].setdefault(
                            dataset_name,
                            {
                                "dataset_name": dataset_name,
                                "results_path": str(results_root / dataset_name / "results" / "results.json"),
                                "metrics_path": str(results_root / dataset_name / "results" / "metrics.json"),
                                "stage": locomo_stage(dataset_name),
                            },
                        )
                    q1_checkpoint["failed"].pop(dataset_name, None)
                    q1_checkpoint["completed_stages"] = stage_completion_count(
                        dataset_names=dataset_names,
                        completed=q1_checkpoint["completed"],
                    )
                    q1_checkpoint["updated_at"] = time.time()
                    save_json(q1_checkpoint_path, q1_checkpoint)
                except Exception as exc:
                    q1_checkpoint["failed"][dataset_name] = {
                        "dataset_name": dataset_name,
                        "error": str(exc),
                        "stage": locomo_stage(dataset_name),
                    }
                    q1_checkpoint["completed_stages"] = stage_completion_count(
                        dataset_names=dataset_names,
                        completed=q1_checkpoint["completed"],
                    )
                    q1_checkpoint["updated_at"] = time.time()
                    save_json(q1_checkpoint_path, q1_checkpoint)

    close_trace_writer(trace_writer)
    finished_at = time.time()
    q1_summary = {
        "mode": "q1",
        "results_root": str(results_root),
        "dataset_count": len(dataset_names),
        "stage_count": len(stages),
        "completed_count": len(q1_checkpoint["completed"]),
        "failed_count": len(q1_checkpoint["failed"]),
        "completed_stages": int(q1_checkpoint["completed_stages"]),
        "failed_datasets": dict(q1_checkpoint["failed"]),
        "llm_model": args.llm_model,
        "q0_cache_llm_model": args.q0_cache_llm_model,
        "embedding_model": args.embedding_model,
        "judge_model": judge_cfg.evaluation.eval_model,
        "judge_prompt_style": judge_cfg.evaluation.eval_prompt_style,
        "judge_num_runs": judge_cfg.evaluation.eval_num_runs,
        "trace_path": str(q1_trace_path),
        "checkpoint_path": str(q1_checkpoint_path),
        "started_at": q1_started_at,
        "finished_at": finished_at,
        "duration_sec": round(finished_at - q1_started_at, 3),
    }
    save_json(q1_summary_path, q1_summary)
    print(json.dumps(q1_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
