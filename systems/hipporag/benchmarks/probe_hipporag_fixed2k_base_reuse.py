#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import igraph as ig

BENCHMARK_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from evaluation.llm_evaluator import LLMEvaluator
from init.config import Config
from run_hipporag_longmemeval import (
    DEFAULT_FIXED2K_BASE_SOURCE_ROOT,
    DEFAULT_FIXED2K_MANIFEST,
    DEFAULT_JUDGE_CONFIG,
    build_doc_metadata,
    build_q0_artifact_paths,
    collect_hipporag_usage_cost,
    load_corpus,
    load_doc_metadata,
    load_question,
    make_hipporag_config,
    multiset_doc_difference,
    normalize_clean_config,
    q0_ready,
    query_with_hipporag,
    refresh_hipporag_graph_from_current_cache,
    run_q0_for_dataset,
    save_json,
    validate_clean_contract,
    validate_record_contract,
    judge_record,
    redact_answer_field_in_results,
)
from hipporag.HippoRAG import HippoRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe HippoRAG fixed2k base-cache reuse against a fresh direct build."
    )
    parser.add_argument("--dataset-name", required=True, help="Fixed2k dataset_name, e.g. s100/longmem_ssp/lm_27_a0001")
    parser.add_argument(
        "--seed-dataset-name",
        default="",
        help="Optional fixed2k item cache to seed the base cache from. Must share the same base_rel_path. Leave empty for canonical fresh base build.",
    )
    parser.add_argument(
        "--seed-results-root",
        default=str(REPO_ROOT / "runs" / "hipporag_fixed2k_main3m_native_gpt5mini_judge4omini"),
        help="Results root containing the seed item cache.",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "fixed2k_sbins_fixed2k_main3m_20260224_102211"),
        help="Fixed2k data root.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_FIXED2K_MANIFEST),
        help="Fixed2k manifest.jsonl path.",
    )
    parser.add_argument(
        "--base-source-root",
        default=str(DEFAULT_FIXED2K_BASE_SOURCE_ROOT),
        help="Root for base_rel_path source corpora.",
    )
    parser.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "runs" / "hipporag_fixed2k_base_reuse_probe"),
        help="Output root for this probe.",
    )
    parser.add_argument(
        "--judge-config",
        default=str(DEFAULT_JUDGE_CONFIG),
        help="Judge config for MemOS-aligned evaluation.",
    )
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument(
        "--q0-cache-llm-model",
        default="",
        help="Optional model label used only for q0 cache/openie working-dir lookup. Defaults to --llm-model.",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--qa-top-k", type=int, default=5)
    parser.add_argument("--top-session-limit", type=int, default=20)
    parser.add_argument("--enable-react-multihop", action="store_true")
    parser.add_argument("--react-max-steps", type=int, default=3)
    parser.add_argument("--react-max-context-chunks", type=int, default=12)
    parser.add_argument("--react-agent-max-tokens", type=int, default=512)
    parser.add_argument("--react-agent-temperature", type=float, default=0.0)
    parser.add_argument("--skip-direct", action="store_true", help="Skip the direct fresh-build control arm and only run base-first derived evaluation.")
    parser.add_argument("--force", action="store_true", help="Rebuild base/derived/direct artifacts even if present.")
    return parser.parse_args()


def load_manifest_row(manifest_path: Path, dataset_name: str) -> Dict[str, Any]:
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row.get("dataset_name", "") or "") == dataset_name:
            return row
    raise KeyError(f"dataset_name not found in manifest: {dataset_name}")


def maybe_load_manifest_row(manifest_path: Path, dataset_name: str) -> Dict[str, Any] | None:
    if not str(dataset_name or "").strip():
        return None
    return load_manifest_row(manifest_path, dataset_name)


def sync_openai_env_aliases() -> None:
    if "GPT_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_API_KEY"]
    if "OPENAI_API_KEY" in os.environ and "GPT_API_KEY" not in os.environ:
        os.environ["GPT_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if "GPT_BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_BASE_URL"]
    if "OPENAI_BASE_URL" in os.environ and "GPT_BASE_URL" not in os.environ:
        os.environ["GPT_BASE_URL"] = os.environ["OPENAI_BASE_URL"]


def make_stage_dir(results_root: Path, stage: str, dataset_name: str) -> Path:
    return results_root / stage / dataset_name


def base_cache_dataset_name(base_rel_path: str) -> str:
    return f"_base_caches/{base_rel_path}"


def load_openie_docs(cache_dir: Path, llm_model: str) -> List[Dict[str, Any]]:
    openie_path = cache_dir / f"openie_results_ner_{llm_model.replace('/', '_')}.json"
    payload = json.loads(openie_path.read_text(encoding="utf-8"))
    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        raise TypeError(f"Unexpected OpenIE payload at {openie_path}")
    return docs


def summarize_openie_docs(cache_dir: Path, llm_model: str) -> Dict[str, Any]:
    docs = load_openie_docs(cache_dir, llm_model)
    return {
        "docs": len(docs),
        "entities_total": sum(len((row.get("extracted_entities") or [])) for row in docs),
        "triples_total": sum(len((row.get("extracted_triples") or [])) for row in docs),
        "empty_entity_docs": sum(1 for row in docs if not (row.get("extracted_entities") or [])),
        "empty_triple_docs": sum(1 for row in docs if not (row.get("extracted_triples") or [])),
    }


def compare_doc_metadata_sets(cache_dir_a: Path, cache_dir_b: Path) -> Dict[str, Any]:
    docs_a = load_doc_metadata(cache_dir_a / "doc_metadata.json")
    docs_b = load_doc_metadata(cache_dir_b / "doc_metadata.json")

    def key(row: Dict[str, Any]) -> Any:
        return (
            str(row.get("session_id", "") or ""),
            str(row.get("session_time", "") or ""),
            str(row.get("raw_context", "") or ""),
        )

    set_a = {key(row) for row in docs_a}
    set_b = {key(row) for row in docs_b}
    return {
        "count_a": len(docs_a),
        "count_b": len(docs_b),
        "exact_set_equal": set_a == set_b,
        "a_minus_b_count": len(set_a - set_b),
        "b_minus_a_count": len(set_b - set_a),
    }


def origin_rank(top_session_ids: List[str], origin_session_id: str) -> int | None:
    if origin_session_id in top_session_ids:
        return top_session_ids.index(origin_session_id) + 1
    return None


def ensure_base_cache(
    *,
    base_rel_path: str,
    base_rows: List[Dict[str, Any]],
    results_root: Path,
    seed_results_root: Path,
    seed_dataset_name: str,
    llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    force: bool,
) -> Dict[str, Any]:
    base_dataset_name = base_cache_dataset_name(base_rel_path)
    base_dataset_dir = make_stage_dir(results_root, "base", base_dataset_name)
    base_cache_dir = base_dataset_dir / "hipporag_q0_cache"
    q0_summary_path = base_dataset_dir / "q0_index_summary.json"
    q0_cost_path = base_dataset_dir / "q0_cost_summary.json"

    base_docs, base_doc_meta = build_doc_metadata(base_rows)
    if q0_ready(base_cache_dir, llm_model, embedding_model) and not force:
        summary = json.loads(q0_summary_path.read_text(encoding="utf-8"))
        return summary

    # When a caller intentionally reuses base caches from another results root via
    # per-base symlinks, never mutate the symlink target in-place. If a symlinked
    # base is not q0-ready, the correct fix is to repair the source cache first,
    # not to silently delete and rebuild through the alias path.
    if base_dataset_dir.is_symlink() and (force or not q0_ready(base_cache_dir, llm_model, embedding_model)):
        raise RuntimeError(
            f"symlinked base cache is not q0-ready and cannot be rebuilt in-place: {base_dataset_dir}"
        )

    if base_dataset_dir.exists() and (force or not q0_ready(base_cache_dir, llm_model, embedding_model)):
        shutil.rmtree(base_dataset_dir)
    base_dataset_dir.mkdir(parents=True, exist_ok=True)

    seed_cache_dir = Path(seed_results_root) / seed_dataset_name / "hipporag_q0_cache" if seed_dataset_name else None
    seeded = bool(seed_cache_dir and seed_cache_dir.exists())
    if seeded:
        shutil.copytree(seed_cache_dir, base_cache_dir, dirs_exist_ok=True)

    cfg = make_hipporag_config(
        save_dir=base_cache_dir,
        llm_model=llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        force_index_from_scratch=False,
        corpus_len=len(base_rows),
    )
    hippo = HippoRAG(global_config=cfg)

    start = time.perf_counter()
    hippo.index(base_docs)
    refresh_summary = refresh_hipporag_graph_from_current_cache(hippo)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    preprocessing_cost = collect_hipporag_usage_cost(hippo, llm_model=llm_model, embedding_model=embedding_model)

    save_json(base_cache_dir / "doc_metadata.json", base_doc_meta)
    save_json(q0_cost_path, preprocessing_cost)
    summary = {
        "dataset_name": base_dataset_name,
        "base_rel_path": base_rel_path,
        "seed_dataset_name": seed_dataset_name if seeded else "",
        "seed_cache_used": bool(seeded),
        "cache_reused": False,
        "num_docs": len(base_docs),
        "duration_ms": elapsed_ms,
        "graph_info": refresh_summary["graph_info"],
        "graph_pickle": {
            "vcount": refresh_summary["graph_vcount"],
            "ecount": refresh_summary["graph_ecount"],
        },
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "cache_dir": str(base_cache_dir),
        "artifact_paths": build_q0_artifact_paths(base_cache_dir, llm_model, embedding_model),
        "q0_cost_summary_path": str(q0_cost_path),
        "preprocessing_cost": preprocessing_cost,
    }
    save_json(q0_summary_path, summary)
    return summary


def derive_item_cache_from_base(
    *,
    dataset_name: str,
    target_rows: List[Dict[str, Any]],
    base_rows: List[Dict[str, Any]],
    base_cache_dir: Path,
    results_root: Path,
    llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    force: bool,
) -> Dict[str, Any]:
    derived_dataset_dir = make_stage_dir(results_root, "derived", dataset_name)
    derived_cache_dir = derived_dataset_dir / "hipporag_q0_cache"
    q0_summary_path = derived_dataset_dir / "q0_index_summary.json"
    q0_cost_path = derived_dataset_dir / "q0_cost_summary.json"

    target_docs, target_doc_meta = build_doc_metadata(target_rows)
    base_docs, _ = build_doc_metadata(base_rows)
    docs_to_delete = multiset_doc_difference(base_docs, target_docs)

    cache_ready = q0_ready(derived_cache_dir, llm_model, embedding_model)
    summary_ready = q0_summary_path.exists() and q0_cost_path.exists()

    if cache_ready and summary_ready and not force:
        return json.loads(q0_summary_path.read_text(encoding="utf-8"))

    # Full-run resumes can leave behind partially derived item caches if a base was
    # interrupted after the HippoRAG cache finished materializing but before the q0
    # summary/cost JSONs were written. Those stale directories should be rebuilt
    # rather than treated as reusable caches.
    if derived_dataset_dir.exists() and (force or cache_ready or not summary_ready):
        shutil.rmtree(derived_dataset_dir)

    derived_dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_cache_dir, derived_cache_dir, dirs_exist_ok=True)

    cfg = make_hipporag_config(
        save_dir=derived_cache_dir,
        llm_model=llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        force_index_from_scratch=False,
        corpus_len=len(target_rows),
    )
    hippo = HippoRAG(global_config=cfg)

    start = time.perf_counter()
    hippo.prepare_retrieval_objects()
    hippo.delete(docs_to_delete)
    refresh_summary = refresh_hipporag_graph_from_current_cache(hippo)
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    derivation_cost = collect_hipporag_usage_cost(hippo, llm_model=llm_model, embedding_model=embedding_model)

    save_json(derived_cache_dir / "doc_metadata.json", target_doc_meta)
    save_json(q0_cost_path, derivation_cost)
    summary = {
        "dataset_name": dataset_name,
        "derived_from_base_cache": str(base_cache_dir),
        "cache_reused": False,
        "num_docs": len(target_docs),
        "docs_deleted_from_base": len(docs_to_delete),
        "duration_ms": elapsed_ms,
        "graph_info": refresh_summary["graph_info"],
        "graph_pickle": {
            "vcount": refresh_summary["graph_vcount"],
            "ecount": refresh_summary["graph_ecount"],
        },
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "cache_dir": str(derived_cache_dir),
        "artifact_paths": build_q0_artifact_paths(derived_cache_dir, llm_model, embedding_model),
        "q0_cost_summary_path": str(q0_cost_path),
        "preprocessing_cost": derivation_cost,
    }
    save_json(q0_summary_path, summary)
    return summary


def run_q1_with_judge(
    *,
    dataset_name: str,
    question_row: Dict[str, Any],
    cache_root: Path,
    llm_model: str,
    cache_llm_model: str | None,
    embedding_model: str,
    qa_top_k: int,
    top_session_limit: int,
    enable_react_multihop: bool,
    react_max_steps: int,
    react_max_context_chunks: int,
    react_agent_max_tokens: int,
    react_agent_temperature: float,
    judge_cfg: Config,
    llm_evaluator: LLMEvaluator,
) -> Dict[str, Any]:
    doc_meta = load_doc_metadata(cache_root / dataset_name / "hipporag_q0_cache" / "doc_metadata.json")
    record = query_with_hipporag(
        dataset_name=dataset_name,
        question_row=question_row,
        doc_metadata=doc_meta,
        results_root=cache_root,
        llm_model=llm_model,
        cache_llm_model=cache_llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        top_session_limit=top_session_limit,
        enable_react_multihop=enable_react_multihop,
        react_max_steps=react_max_steps,
        react_max_context_chunks=react_max_context_chunks,
        react_agent_max_tokens=react_agent_max_tokens,
        react_agent_temperature=react_agent_temperature,
    )
    record = validate_record_contract(
        record,
        dataset_name=dataset_name,
        expected_judge_runs=judge_cfg.evaluation.eval_num_runs,
        require_judgments=False,
    )
    record = asyncio.run(judge_record(llm_evaluator, record))
    record = validate_record_contract(
        record,
        dataset_name=dataset_name,
        expected_judge_runs=judge_cfg.evaluation.eval_num_runs,
        require_judgments=True,
    )
    return record


def main() -> None:
    args = parse_args()
    sync_openai_env_aliases()

    manifest_path = Path(args.manifest_path).resolve()
    base_source_root = Path(args.base_source_root).resolve()
    data_root = Path(args.data_root).resolve()
    results_root = Path(args.results_root).resolve()
    seed_results_root = Path(args.seed_results_root).resolve()

    target_manifest_row = load_manifest_row(manifest_path, args.dataset_name)
    seed_manifest_row = maybe_load_manifest_row(manifest_path, args.seed_dataset_name)
    if seed_manifest_row and target_manifest_row["base_rel_path"] != seed_manifest_row["base_rel_path"]:
        raise RuntimeError("seed_dataset_name must share the same base_rel_path as dataset_name")

    base_rel_path = str(target_manifest_row["base_rel_path"])
    base_rows = load_corpus(base_source_root / base_rel_path / "Corpus.json")
    target_rows = load_corpus(data_root / args.dataset_name / "Corpus.json")
    question_row = load_question(data_root / args.dataset_name / "Question.json")
    q0_cache_llm_model = str(args.q0_cache_llm_model or args.llm_model)

    judge_cfg = normalize_clean_config(Config.parse(Path(args.judge_config)))
    validate_clean_contract(judge_cfg)
    llm_evaluator = LLMEvaluator(judge_cfg, "", args.dataset_name.replace("/", "_"))

    base_summary = ensure_base_cache(
        base_rel_path=base_rel_path,
        base_rows=base_rows,
        results_root=results_root,
        seed_results_root=seed_results_root,
        seed_dataset_name=str(args.seed_dataset_name or "").strip(),
        llm_model=q0_cache_llm_model,
        embedding_model=args.embedding_model,
        qa_top_k=args.qa_top_k,
        force=args.force,
    )
    base_cache_dir = Path(base_summary["cache_dir"])

    derived_summary = derive_item_cache_from_base(
        dataset_name=args.dataset_name,
        target_rows=target_rows,
        base_rows=base_rows,
        base_cache_dir=base_cache_dir,
        results_root=results_root,
        llm_model=q0_cache_llm_model,
        embedding_model=args.embedding_model,
        qa_top_k=args.qa_top_k,
        force=args.force,
    )

    derived_record = run_q1_with_judge(
        dataset_name=args.dataset_name,
        question_row=question_row,
        cache_root=results_root / "derived",
        llm_model=args.llm_model,
        cache_llm_model=q0_cache_llm_model,
        embedding_model=args.embedding_model,
        qa_top_k=args.qa_top_k,
        top_session_limit=args.top_session_limit,
        enable_react_multihop=bool(args.enable_react_multihop),
        react_max_steps=int(args.react_max_steps),
        react_max_context_chunks=int(args.react_max_context_chunks),
        react_agent_max_tokens=int(args.react_agent_max_tokens),
        react_agent_temperature=float(args.react_agent_temperature),
        judge_cfg=judge_cfg,
        llm_evaluator=llm_evaluator,
    )

    save_json(results_root / "derived" / args.dataset_name / "results" / "results.json", [derived_record])
    redact_answer_field_in_results(results_root / "derived" / args.dataset_name / "results" / "results.json")

    derived_cache_dir = Path(derived_summary["cache_dir"])
    graph_derived = ig.Graph.Read_Pickle(
        str(derived_cache_dir / f"{q0_cache_llm_model.replace('/', '_')}_{args.embedding_model.replace('/', '_')}" / "graph.pickle")
    )
    derived_openie = summarize_openie_docs(derived_cache_dir, q0_cache_llm_model)

    derived_q1_summary = {
        "output": derived_record.get("output", ""),
        "top_session_ids": derived_record.get("top_session_ids", []),
        "origin_rank_1based": origin_rank(derived_record.get("top_session_ids", []), str(question_row.get("origin", "") or "")),
        "llm_judge": derived_record.get("llm_judge"),
        "llm_judgments": derived_record.get("llm_judgments"),
        "memos_stats": derived_record.get("memos_stats"),
        "total_cost_usd": derived_record.get("total_cost_usd"),
    }

    direct_summary = None
    direct_record = None
    direct_q1_summary = None
    doc_set_checks = None
    direct_openie = None
    derived_vs_direct_graph_pickle = None
    cost_comparison = {
        "base_q0_total_cost_usd": float(base_summary["preprocessing_cost"]["total_cost_usd"]),
        "derived_q0_total_cost_usd": float(derived_summary["preprocessing_cost"]["total_cost_usd"]),
        "derived_q1_total_cost_usd": float(derived_q1_summary["total_cost_usd"] or 0.0),
    }
    time_comparison = {
        "base_q0_duration_ms": float(base_summary["duration_ms"]),
        "derived_q0_duration_ms": float(derived_summary["duration_ms"]),
        "derived_q1_total_duration_ms": float((derived_q1_summary["memos_stats"] or {}).get("total_duration_ms") or 0.0),
    }

    if not args.skip_direct:
        direct_root = results_root / "direct"
        direct_summary = run_q0_for_dataset(
            dataset_name=args.dataset_name,
            corpus_rows=target_rows,
            results_root=direct_root,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            qa_top_k=args.qa_top_k,
            force_reindex=True,
        )
        direct_record = run_q1_with_judge(
            dataset_name=args.dataset_name,
            question_row=question_row,
            cache_root=direct_root,
            llm_model=args.llm_model,
            cache_llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            qa_top_k=args.qa_top_k,
            top_session_limit=args.top_session_limit,
            enable_react_multihop=bool(args.enable_react_multihop),
            react_max_steps=int(args.react_max_steps),
            react_max_context_chunks=int(args.react_max_context_chunks),
            react_agent_max_tokens=int(args.react_agent_max_tokens),
            react_agent_temperature=float(args.react_agent_temperature),
            judge_cfg=judge_cfg,
            llm_evaluator=llm_evaluator,
        )
        save_json(results_root / "direct" / args.dataset_name / "results" / "results.json", [direct_record])
        redact_answer_field_in_results(results_root / "direct" / args.dataset_name / "results" / "results.json")

        direct_cache_dir = Path(direct_summary["cache_dir"])
        graph_direct = ig.Graph.Read_Pickle(
            str(direct_cache_dir / f"{args.llm_model.replace('/', '_')}_{args.embedding_model.replace('/', '_')}" / "graph.pickle")
        )
        doc_set_checks = compare_doc_metadata_sets(derived_cache_dir, direct_cache_dir)
        direct_openie = summarize_openie_docs(direct_cache_dir, args.llm_model)
        direct_q1_summary = {
            "output": direct_record.get("output", ""),
            "top_session_ids": direct_record.get("top_session_ids", []),
            "origin_rank_1based": origin_rank(direct_record.get("top_session_ids", []), str(question_row.get("origin", "") or "")),
            "llm_judge": direct_record.get("llm_judge"),
            "llm_judgments": direct_record.get("llm_judgments"),
            "memos_stats": direct_record.get("memos_stats"),
            "total_cost_usd": direct_record.get("total_cost_usd"),
        }
        derived_vs_direct_graph_pickle = {
            "derived": {"vcount": graph_derived.vcount(), "ecount": graph_derived.ecount()},
            "direct": {"vcount": graph_direct.vcount(), "ecount": graph_direct.ecount()},
        }
        cost_comparison["direct_q0_total_cost_usd"] = float(direct_summary["preprocessing_cost"]["total_cost_usd"])
        cost_comparison["direct_q1_total_cost_usd"] = float(direct_q1_summary["total_cost_usd"] or 0.0)
        time_comparison["direct_q0_duration_ms"] = float(direct_summary["duration_ms"])
        time_comparison["direct_q1_total_duration_ms"] = float((direct_q1_summary["memos_stats"] or {}).get("total_duration_ms") or 0.0)

    comparison = {
        "dataset_name": args.dataset_name,
        "base_rel_path": base_rel_path,
        "seed_dataset_name": args.seed_dataset_name,
        "fixed2k_manifest_row": target_manifest_row,
        "base_summary": base_summary,
        "derived_summary": derived_summary,
        "direct_summary": direct_summary,
        "derived_vs_direct_graph_pickle": derived_vs_direct_graph_pickle,
        "doc_set_checks": doc_set_checks,
        "openie_comparison": {
            "derived": derived_openie,
            "direct": direct_openie,
        },
        "cost_comparison": cost_comparison,
        "time_comparison": time_comparison,
        "derived_q1": derived_q1_summary,
        "direct_q1": direct_q1_summary,
        "notes": [
            "Base build uses the same HippoRAG config as direct q0 and refreshes the graph from current stores after indexing missing docs.",
            "Derived item cache is created via base-cache copy, upstream delete(), and same-config graph refresh.",
            "No controller or ReAct wrapper is introduced in this probe.",
            "direct_summary/direct_q1 may be null when --skip-direct is used for cheaper smoke coverage.",
        ],
    }
    comparison_path = results_root / "comparison" / args.dataset_name / "base_reuse_comparison.json"
    save_json(comparison_path, comparison)
    print(comparison_path)


if __name__ == "__main__":
    main()
