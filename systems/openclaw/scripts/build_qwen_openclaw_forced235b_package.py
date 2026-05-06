#!/usr/bin/env python3
"""Build the OpenClaw LongMemEval package with forced-retrieve Qwen 235B.

This script is intentionally offline: it only reads existing traces/results and
does not call any model or embedding API. It starts from the 20260427 OpenClaw
package, replaces the original Qwen 235B slice with the final forced-retrieve
rollup, then rebuilds combined traces, metrics, reconstructed costs, manifests,
figures, and the tarball.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(1024 * 1024 * 1024)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", SCRIPT_DIR.parents[2])).expanduser().resolve()

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(ARTIFACT_ROOT / "results"))).expanduser().resolve()
OLD_PACKAGE = Path(
    os.environ.get("OPENCLAW_20260427_PACKAGE", str(RESULTS_ROOT / "qwen_openclaw_longmemeval_20260427_package"))
).expanduser().resolve()
FORCED_PACKAGE = Path(
    os.environ.get(
        "OPENCLAW_235B_FORCED_PACKAGE",
        str(RESULTS_ROOT / "qwen_openclaw_235b_forced_retrieve_final_20260503_package"),
    )
).expanduser().resolve()
NEW_PACKAGE = Path(
    os.environ.get(
        "OPENCLAW_OUTPUT_PACKAGE",
        str(RESULTS_ROOT / "qwen_openclaw_longmemeval_20260503_forced235b_package"),
    )
).expanduser().resolve()
OPENCLAW_RESULTS = Path(os.environ.get("OPENCLAW_RESULTS_BASE", str(RESULTS_ROOT / "openclaw"))).expanduser().resolve()
DATASET_ROOT = Path(
    os.environ.get(
        "LONGMEMEVAL_DATASET_ROOT",
        str(ARTIFACT_ROOT / "data/fixed2k_sbins_fixed2k_main3m_20260224_102211"),
    )
).expanduser().resolve()
DATASET_LIST = Path(
    os.environ.get(
        "OPENCLAW_TARGET_LIST",
        str(
            ARTIFACT_ROOT
            / "systems/openclaw/benchmarks/memory_mvp/dataset_lists/longmemeval_fixed2k_fidelity_balanced50x4x5_seed20260421.txt"
        ),
    )
).expanduser().resolve()

TRACE_BASE_FIELDS = [
    "task_id",
    "dataset_name",
    "model",
    "memory",
    "s",
    "question_type",
    "trial",
    "retrieval_calls",
    "retrieved_sessions",
    "react_steps",
    "success",
    "f1",
    "llm_judge",
    "response_duration_ms",
    "search_duration_ms",
    "total_duration_ms",
    "context_tokens",
    "total_cost_usd",
    "context_ok",
    "agent_output",
    "trace_json",
]

COMBINED_TRACE_FIELDS = TRACE_BASE_FIELDS + ["model_label", "run_name"]

RUNS = [
    ("qwen8b", "Qwen 8B", "qwen3-8b", "qwen8b_trace_q1_full.csv"),
    ("qwen32b", "Qwen 32B", "qwen3-32b", "qwen32b_trace_q1_full.csv"),
    ("qwen235b", "Qwen 235B", "qwen3-235b-a22b-instruct-2507", "qwen235b_trace_q1_full.csv"),
]

QWEN_PRICE_CNY_PER_1M = {
    "qwen3-8b": {"input": 0.5, "output": 2.0},
    "qwen3-32b": {"input": 2.0, "output": 8.0},
    "qwen3-235b-a22b-instruct-2507": {"input": 2.0, "output": 8.0},
}

FX_USD_CNY = 6.8348
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_USD_PER_1M = 0.02
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_USD_PER_1M_PROMPT = 0.15
JUDGE_USD_PER_1M_COMPLETION = 0.60
COST_NOTE = (
    "OpenClaw official q0/index embedding cost is not exposed by the official bridge; "
    "Qwen DashScope chat pricing is reconstructed from saved usage tokens. "
    "recorded_total_cost_usd is runner-recorded judge telemetry, while reconstructed "
    "cost files include Qwen chat + estimated q0/q1 embedding + judge."
)

COST_ITEMIZED_FIELDS = [
    "row_id",
    "model_label",
    "qwen_model_id",
    "dataset_name",
    "s",
    "family",
    "question_type_short",
    "success",
    "llm_judge",
    "f1",
    "retrieval_calls",
    "react_steps",
    "context_tokens",
    "qwen_input_price_cny_per_1m",
    "qwen_output_price_cny_per_1m",
    "qwen_chat_prompt_tokens",
    "qwen_chat_completion_tokens",
    "qwen_chat_calls",
    "qwen_chat_cost_cny",
    "qwen_chat_cost_usd_report",
    "q0_corpus_rows",
    "q0_memory_files_indexed_est",
    "q0_embedding_chunks_est",
    "q0_embedding_prompt_tokens_est",
    "q0_embedding_cost_usd_est",
    "q1_query_embedding_prompt_tokens_est",
    "q1_query_embedding_calls_est",
    "q1_query_embedding_cost_usd_est",
    "embedding_cost_usd_est",
    "embedding_model",
    "embedding_cost_basis",
    "judge_model",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "judge_calls",
    "judge_cost_usd",
    "total_cost_cny_report",
    "total_cost_usd_report",
    "original_runner_total_cost_usd",
    "result_path",
    "source_root",
]

COST_GROUP_FIELDS = [
    "model_label",
    "qwen_model_id",
    "items",
    "success_sum",
    "success_rate",
    "qwen_chat_prompt_tokens",
    "qwen_chat_completion_tokens",
    "qwen_chat_calls",
    "qwen_chat_cost_cny",
    "q0_embedding_prompt_tokens_est",
    "q0_embedding_chunks_est",
    "q0_embedding_cost_usd_est",
    "q1_query_embedding_prompt_tokens_est",
    "q1_query_embedding_calls_est",
    "q1_query_embedding_cost_usd_est",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "judge_calls",
    "judge_cost_usd",
    "total_cost_cny_report",
    "total_cost_usd_report",
    "retrieval_calls_sum",
    "retrieval_calls_mean",
    "react_steps_mean",
    "context_tokens_mean",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def fnum(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def inum(value: Any, default: int = 0) -> int:
    return int(round(fnum(value, float(default))))


def mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(fnum(r.get(field)) for r in rows) / len(rows)


def short_family(dataset_name: str) -> tuple[str, str]:
    parts = dataset_name.split("/")
    family = parts[1] if len(parts) > 1 else ""
    mapping = {
        "longmem_ssa": "SSA",
        "longmem_ssp": "SSP",
        "longmem_ssu": "SSU",
        "longmem_tr": "TR",
    }
    return family, mapping.get(family, family)


def normalize_s(row: dict[str, Any]) -> str:
    val = str(row.get("s", "")).strip()
    if val:
        return val
    ds = str(row.get("dataset_name", ""))
    if ds.startswith("s") and "/" in ds:
        return ds.split("/", 1)[0][1:]
    return ""


def stats_row(rows: list[dict[str, Any]], extras: dict[str, Any]) -> dict[str, Any]:
    return {
        **extras,
        "rows": len(rows),
        "success_rate": mean(rows, "success"),
        "llm_judge_rate": mean(rows, "llm_judge"),
        "f1_mean": mean(rows, "f1"),
        "retrieval_calls_mean": mean(rows, "retrieval_calls"),
        "retrieved_sessions_mean": mean(rows, "retrieved_sessions"),
        "react_steps_mean": mean(rows, "react_steps"),
        "context_tokens_mean": mean(rows, "context_tokens"),
        "total_cost_usd_sum": sum(fnum(r.get("total_cost_usd")) for r in rows),
        "total_cost_usd_mean": mean(rows, "total_cost_usd"),
        "response_duration_ms_mean": mean(rows, "response_duration_ms"),
        "search_duration_ms_mean": mean(rows, "search_duration_ms"),
        "total_duration_ms_mean": mean(rows, "total_duration_ms"),
    }


def aggregate_trace_json(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = Counter()
    for row in rows:
        raw = {}
        try:
            raw = json.loads(row.get("trace_json") or "{}")
        except json.JSONDecodeError:
            pass
        qa = raw.get("llm_qa_metadata") or {}
        ev = raw.get("llm_eval_metadata") or {}
        out["qa_chat_prompt_tokens"] += inum(qa.get("chat_prompt_tokens"))
        out["qa_chat_completion_tokens"] += inum(qa.get("chat_completion_tokens"))
        out["qa_chat_calls"] += inum(qa.get("chat_calls"))
        out["eval_prompt_tokens"] += inum(ev.get("eval_prompt_tokens"))
        out["eval_completion_tokens"] += inum(ev.get("eval_completion_tokens"))
        out["eval_calls"] += inum(ev.get("eval_calls"))
        out["eval_total_cost_usd"] += fnum(ev.get("eval_total_cost_usd"))
    return dict(out)


def build_results_and_search_jsonl() -> None:
    trace_path = NEW_PACKAGE / "01_traces/qwen235b_trace_q1_full.csv"
    rows = read_csv(trace_path)
    results = []
    for row in rows:
        try:
            results.append(json.loads(row["trace_json"]))
        except json.JSONDecodeError:
            results.append({"dataset_name": row.get("dataset_name"), "parse_error": True})
    write_jsonl(NEW_PACKAGE / "01_traces/qwen235b_results_full.jsonl", results)

    manifest_rows = load_jsonl(FORCED_PACKAGE / "06_manifests/qwen235b_forced_retrieve_manifest.jsonl")
    manifest = {r.get("dataset_name"): r for r in manifest_rows}
    old_search_by_dataset = {
        r.get("dataset_name"): r
        for r in load_jsonl(OLD_PACKAGE / "01_traces/qwen235b_search_results_full.jsonl")
    }

    search_rows = []
    for row in rows:
        dataset = row["dataset_name"]
        raw = json.loads(row["trace_json"])
        source = manifest.get(dataset, {})
        root_name = source.get("root", "")
        source_root = ""
        search_results_path = ""
        search_results: Any = None

        if root_name == OLD_PACKAGE.name and dataset in old_search_by_dataset:
            old_row = old_search_by_dataset[dataset]
            source_root = str(OLD_PACKAGE)
            search_results_path = str(old_row.get("search_results_path") or "")
            search_results = old_row.get("search_results")
        else:
            candidate_root = OPENCLAW_RESULTS / root_name if root_name else None
            if candidate_root and candidate_root.exists():
                source_root = str(candidate_root)
                candidate_search = candidate_root / "derived" / dataset / "search_results.json"
                if candidate_search.exists():
                    search_results_path = str(candidate_search)
                    search_results = load_json(candidate_search, {})

        if search_results is None:
            calls = []
            for step in raw.get("react_trace") or []:
                if step.get("action") == "memory_search":
                    calls.append(
                        {
                            "query": step.get("query"),
                            "requested_max_results": step.get("requested_max_results"),
                            "requested_min_score": step.get("requested_min_score"),
                            "results": step.get("results") or [],
                            "top_session_ids": step.get("top_session_ids") or [],
                            "retrieved_session_count": step.get("retrieved_session_count", 0),
                            "status": "reconstructed_from_trace_json",
                        }
                    )
            search_results = {
                "sources": ["memory"],
                "agent_mode": "memory_tools",
                "memory_backend": "official",
                "memory_agent_profile": "openclaw_fidelity",
                "memory_search_calls": calls,
                "memory_search_results": raw.get("memory_search_results") or [],
                "reconstruction_note": "Original search_results.json was unavailable in package roots; reconstructed from trace_json.",
            }

        search_rows.append(
            {
                "dataset_name": dataset,
                "model_label": "Qwen 235B",
                "run_name": "qwen235b",
                "source_root": source_root or root_name,
                "search_results_path": search_results_path,
                "search_results": search_results,
            }
        )

    write_jsonl(NEW_PACKAGE / "01_traces/qwen235b_search_results_full.jsonl", search_rows)


def rebuild_combined_traces() -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    for slug, label, _model_id, trace_file in RUNS:
        rows = read_csv(NEW_PACKAGE / "01_traces" / trace_file)
        for row in rows:
            out = {field: row.get(field, "") for field in TRACE_BASE_FIELDS}
            out["model_label"] = label
            out["run_name"] = slug
            combined.append(out)
    write_csv(NEW_PACKAGE / "01_traces/qwen_all_models_combined_trace.csv", combined, COMBINED_TRACE_FIELDS)

    result_paths = [
        NEW_PACKAGE / "01_traces/qwen8b_results_full.jsonl",
        NEW_PACKAGE / "01_traces/qwen32b_results_full.jsonl",
        NEW_PACKAGE / "01_traces/qwen235b_results_full.jsonl",
    ]
    with (NEW_PACKAGE / "01_traces/qwen_all_models_combined_results.jsonl").open("w") as out:
        for p in result_paths:
            with p.open() as f:
                shutil.copyfileobj(f, out)

    search_paths = [
        NEW_PACKAGE / "01_traces/qwen8b_search_results_full.jsonl",
        NEW_PACKAGE / "01_traces/qwen32b_search_results_full.jsonl",
        NEW_PACKAGE / "01_traces/qwen235b_search_results_full.jsonl",
    ]
    with (NEW_PACKAGE / "01_traces/qwen_all_models_combined_search_results.jsonl").open("w") as out:
        for p in search_paths:
            with p.open() as f:
                shutil.copyfileobj(f, out)

    return combined


def rebuild_metrics(combined: list[dict[str, str]]) -> None:
    metrics_dir = NEW_PACKAGE / "02_metrics"
    by_run = defaultdict(list)
    by_s = defaultdict(list)
    by_family = defaultdict(list)
    by_s_family = defaultdict(list)
    for row in combined:
        run = (row["model_label"], row["run_name"])
        s_val = normalize_s(row)
        family, qtype = short_family(row["dataset_name"])
        by_run[run].append(row)
        by_s[run + (s_val,)].append(row)
        by_family[run + (family, qtype)].append(row)
        by_s_family[run + (s_val, family, qtype)].append(row)

    model_rows = [
        stats_row(rows, {"model_label": k[0], "run_name": k[1]})
        for k, rows in sorted(by_run.items(), key=lambda x: x[0][1])
    ]
    write_csv(
        metrics_dir / "model_metrics.csv",
        model_rows,
        [
            "model_label",
            "run_name",
            "rows",
            "success_rate",
            "llm_judge_rate",
            "f1_mean",
            "retrieval_calls_mean",
            "retrieved_sessions_mean",
            "react_steps_mean",
            "context_tokens_mean",
            "total_cost_usd_sum",
            "total_cost_usd_mean",
            "response_duration_ms_mean",
            "search_duration_ms_mean",
            "total_duration_ms_mean",
        ],
    )

    s_rows = [
        stats_row(rows, {"model_label": k[0], "run_name": k[1], "s": k[2]})
        for k, rows in sorted(by_s.items(), key=lambda x: (x[0][1], int(x[0][2] or 0)))
    ]
    write_csv(
        metrics_dir / "metrics_by_s_models.csv",
        s_rows,
        [
            "model_label",
            "run_name",
            "s",
            "rows",
            "success_rate",
            "llm_judge_rate",
            "f1_mean",
            "retrieval_calls_mean",
            "retrieved_sessions_mean",
            "react_steps_mean",
            "context_tokens_mean",
            "total_cost_usd_mean",
            "response_duration_ms_mean",
            "search_duration_ms_mean",
            "total_duration_ms_mean",
            "total_cost_usd_sum",
        ],
    )

    family_rows = [
        stats_row(rows, {"model_label": k[0], "run_name": k[1], "family": k[2], "question_type_short": k[3]})
        for k, rows in sorted(by_family.items(), key=lambda x: (x[0][1], x[0][3]))
    ]
    write_csv(
        metrics_dir / "metrics_by_family_models.csv",
        family_rows,
        [
            "model_label",
            "run_name",
            "family",
            "question_type_short",
            "rows",
            "success_rate",
            "llm_judge_rate",
            "f1_mean",
            "retrieval_calls_mean",
            "retrieved_sessions_mean",
            "react_steps_mean",
            "context_tokens_mean",
            "total_cost_usd_mean",
            "response_duration_ms_mean",
            "search_duration_ms_mean",
            "total_duration_ms_mean",
            "total_cost_usd_sum",
        ],
    )

    sf_rows = [
        stats_row(rows, {"s": k[2], "family": k[3], "question_type_short": k[4], "model_label": k[0], "run_name": k[1]})
        for k, rows in sorted(by_s_family.items(), key=lambda x: (x[0][1], int(x[0][2] or 0), x[0][4]))
    ]
    write_csv(
        metrics_dir / "metrics_by_s_family_models.csv",
        sf_rows,
        [
            "model_label",
            "run_name",
            "rows",
            "success_rate",
            "llm_judge_rate",
            "f1_mean",
            "retrieval_calls_mean",
            "retrieved_sessions_mean",
            "react_steps_mean",
            "context_tokens_mean",
            "total_cost_usd_sum",
            "total_cost_usd_mean",
            "response_duration_ms_mean",
            "search_duration_ms_mean",
            "total_duration_ms_mean",
            "s",
            "family",
            "question_type_short",
        ],
    )

    retrieval_hist = []
    react_hist = []
    for (label, slug), rows in sorted(by_run.items(), key=lambda x: x[0][1]):
        rh = Counter(inum(r.get("retrieval_calls")) for r in rows)
        sh = Counter(inum(r.get("react_steps")) for r in rows)
        for val, n in sorted(rh.items()):
            retrieval_hist.append({"model_label": label, "run_name": slug, "retrieval_calls": val, "rows": n, "fraction": n / len(rows)})
        for val, n in sorted(sh.items()):
            react_hist.append({"model_label": label, "run_name": slug, "react_steps": val, "rows": n, "fraction": n / len(rows)})
    write_csv(metrics_dir / "retrieval_hist_by_model.csv", retrieval_hist, ["model_label", "run_name", "retrieval_calls", "rows", "fraction"])
    write_csv(metrics_dir / "react_steps_hist_by_model.csv", react_hist, ["model_label", "run_name", "react_steps", "rows", "fraction"])

    cost_rows = []
    cost_s_rows = []
    for (label, slug), rows in sorted(by_run.items(), key=lambda x: x[0][1]):
        meta = aggregate_trace_json(rows)
        recorded_sum = sum(fnum(r.get("total_cost_usd")) for r in rows)
        cost_rows.append(
            {
                "model_label": label,
                "run_name": slug,
                "group": "ALL",
                "rows": len(rows),
                "recorded_total_cost_usd_sum": recorded_sum,
                "recorded_total_cost_usd_mean": recorded_sum / len(rows),
                "eval_total_cost_usd_sum": meta.get("eval_total_cost_usd", 0.0),
                "tracked_non_eval_cost_usd_sum": max(0.0, recorded_sum - meta.get("eval_total_cost_usd", 0.0)),
                **meta,
                "cost_observability_note": COST_NOTE,
            }
        )
    for key, rows in sorted(by_s.items(), key=lambda x: (x[0][1], int(x[0][2] or 0))):
        label, slug, s_val = key
        meta = aggregate_trace_json(rows)
        recorded_sum = sum(fnum(r.get("total_cost_usd")) for r in rows)
        cost_s_rows.append(
            {
                "model_label": label,
                "run_name": slug,
                "group": f"s{s_val}",
                "rows": len(rows),
                "recorded_total_cost_usd_sum": recorded_sum,
                "recorded_total_cost_usd_mean": recorded_sum / len(rows),
                "eval_total_cost_usd_sum": meta.get("eval_total_cost_usd", 0.0),
                "tracked_non_eval_cost_usd_sum": max(0.0, recorded_sum - meta.get("eval_total_cost_usd", 0.0)),
                **meta,
                "cost_observability_note": COST_NOTE,
            }
        )
    fields = [
        "model_label",
        "run_name",
        "group",
        "rows",
        "recorded_total_cost_usd_sum",
        "recorded_total_cost_usd_mean",
        "eval_total_cost_usd_sum",
        "tracked_non_eval_cost_usd_sum",
        "qa_chat_prompt_tokens",
        "qa_chat_completion_tokens",
        "qa_chat_calls",
        "eval_prompt_tokens",
        "eval_completion_tokens",
        "eval_calls",
        "cost_observability_note",
    ]
    write_csv(metrics_dir / "cost_by_model.csv", cost_rows, fields)
    write_csv(metrics_dir / "cost_by_s_models.csv", cost_s_rows, fields)


def adapt_forced_cost_rows() -> list[dict[str, Any]]:
    forced_rows = read_csv(FORCED_PACKAGE / "02_metrics/cost_reconstructed_aliyun_qwen235b_forced_itemized.csv")
    adapted = []
    for row in forced_rows:
        out = {field: row.get(field, "") for field in COST_ITEMIZED_FIELDS}
        out["model_label"] = "qwen235b"
        out["qwen_model_id"] = "qwen3-235b-a22b-instruct-2507"
        out["result_path"] = ""
        adapted.append(out)
    return adapted


def group_cost_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(f, "") for f in group_fields)
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items()):
        base = {f: v for f, v in zip(group_fields, key)}
        base.update(
            {
                "items": len(items),
                "success_sum": sum(inum(r.get("success")) for r in items),
                "success_rate": sum(inum(r.get("success")) for r in items) / len(items),
                "qwen_chat_prompt_tokens": sum(inum(r.get("qwen_chat_prompt_tokens")) for r in items),
                "qwen_chat_completion_tokens": sum(inum(r.get("qwen_chat_completion_tokens")) for r in items),
                "qwen_chat_calls": sum(inum(r.get("qwen_chat_calls")) for r in items),
                "qwen_chat_cost_cny": sum(fnum(r.get("qwen_chat_cost_cny")) for r in items),
                "q0_embedding_prompt_tokens_est": sum(inum(r.get("q0_embedding_prompt_tokens_est")) for r in items),
                "q0_embedding_chunks_est": sum(inum(r.get("q0_embedding_chunks_est")) for r in items),
                "q0_embedding_cost_usd_est": sum(fnum(r.get("q0_embedding_cost_usd_est")) for r in items),
                "q1_query_embedding_prompt_tokens_est": sum(inum(r.get("q1_query_embedding_prompt_tokens_est")) for r in items),
                "q1_query_embedding_calls_est": sum(inum(r.get("q1_query_embedding_calls_est")) for r in items),
                "q1_query_embedding_cost_usd_est": sum(fnum(r.get("q1_query_embedding_cost_usd_est")) for r in items),
                "judge_prompt_tokens": sum(inum(r.get("judge_prompt_tokens")) for r in items),
                "judge_completion_tokens": sum(inum(r.get("judge_completion_tokens")) for r in items),
                "judge_calls": sum(inum(r.get("judge_calls")) for r in items),
                "judge_cost_usd": sum(fnum(r.get("judge_cost_usd")) for r in items),
                "total_cost_cny_report": sum(fnum(r.get("total_cost_cny_report")) for r in items),
                "total_cost_usd_report": sum(fnum(r.get("total_cost_usd_report")) for r in items),
                "retrieval_calls_sum": sum(inum(r.get("retrieval_calls")) for r in items),
                "retrieval_calls_mean": sum(inum(r.get("retrieval_calls")) for r in items) / len(items),
                "react_steps_mean": sum(fnum(r.get("react_steps")) for r in items) / len(items),
                "context_tokens_mean": sum(fnum(r.get("context_tokens")) for r in items) / len(items),
            }
        )
        out.append(base)
    return out


def rebuild_reconstructed_costs(combined: list[dict[str, str]]) -> dict[str, Any]:
    old_itemized = read_csv(OLD_PACKAGE / "02_metrics/cost_reconstructed_aliyun_itemized.csv")
    kept = [r for r in old_itemized if r.get("model_label") in {"qwen8b", "qwen32b"}]
    forced = adapt_forced_cost_rows()
    itemized = kept + forced
    for idx, row in enumerate(itemized, start=1):
        row["row_id"] = idx
        for field in COST_ITEMIZED_FIELDS:
            row.setdefault(field, "")
    write_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_itemized.csv", itemized, COST_ITEMIZED_FIELDS)

    by_model = group_cost_rows(itemized, ["model_label", "qwen_model_id"])
    by_s = group_cost_rows(itemized, ["s", "model_label", "qwen_model_id"])
    by_family = group_cost_rows(itemized, ["family", "question_type_short", "model_label", "qwen_model_id"])
    write_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_by_model.csv", by_model, COST_GROUP_FIELDS)
    write_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_by_s_models.csv", by_s, ["s"] + COST_GROUP_FIELDS)
    write_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_by_family_models.csv", by_family, ["family", "question_type_short"] + COST_GROUP_FIELDS)

    totals = group_cost_rows(itemized, [])[0]
    totals["success_sum"] = int(totals["success_sum"])
    totals["items"] = int(totals["items"])
    totals["success_rate"] = totals["success_sum"] / totals["items"]
    totals["qwen_chat_cost_usd_report"] = totals["qwen_chat_cost_cny"] / FX_USD_CNY
    totals["embedding_cost_usd_est"] = totals["q0_embedding_cost_usd_est"] + totals["q1_query_embedding_cost_usd_est"]
    totals["retrieval_calls_mean"] = totals["retrieval_calls_sum"] / totals["items"]

    old_cost = load_json(OLD_PACKAGE / "00_summary/cost_reconstruction_aliyun_qwen.json", {})
    pricing_basis = old_cost.get("pricing_basis", {})
    pricing_basis.update(
        {
            "qwen_price_cny_per_1m": QWEN_PRICE_CNY_PER_1M,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_usd_per_1m_tokens": EMBEDDING_USD_PER_1M,
            "judge_model": JUDGE_MODEL,
            "judge_usd_per_1m_prompt": JUDGE_USD_PER_1M_PROMPT,
            "judge_usd_per_1m_completion": JUDGE_USD_PER_1M_COMPLETION,
            "fx_usd_cny_for_reporting": FX_USD_CNY,
        }
    )

    summary = {
        "package_root": str(NEW_PACKAGE),
        "dataset_root": str(DATASET_ROOT),
        "items": len(itemized),
        "unique_datasets": len({r["dataset_name"] for r in itemized}),
        "models": sorted({r["qwen_model_id"] for r in itemized}),
        "pricing_basis": pricing_basis,
        "fidelity_notes": [
            "This package replaces the original 20260427 Qwen 235B slice with the final forced-retrieve Qwen 235B rollup.",
            "No API calls were made for this cost reconstruction; all costs are computed from saved traces and reconstructed q0/q1 embedding estimates.",
            "Qwen q1 chat cost uses saved provider usage tokens from llm_qa_metadata.",
            "Judge cost uses saved llm_eval_metadata.eval_total_cost_usd.",
            "OpenClaw q0 index embedding and q1 memory_search query embedding usage were not exposed by the official JS bridge, so they are reconstructed estimates from Corpus.json, the official markdown renderer, and OpenClaw chunking settings tokens=400 overlap=80.",
            "q0 cost is counted per model run/item because the run contract used per-item transient q0 builds with cleanup-q0-after-item and no shared durable q0 cache.",
        ],
        "totals": totals,
        "by_model": by_model,
        "by_s_model": by_s,
        "by_family_model": by_family,
    }
    write_json(NEW_PACKAGE / "00_summary/cost_reconstruction_aliyun_qwen.json", summary)

    validation = validate_costs(itemized, combined, summary)
    write_json(NEW_PACKAGE / "00_summary/cost_reconstruction_validation.json", validation)
    write_cost_markdown(summary, validation)
    return summary


def validate_costs(itemized: list[dict[str, Any]], combined: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
    trace_by_model = Counter(r["run_name"] for r in combined)
    cost_by_model = Counter(r["model_label"] for r in itemized)
    total_usd = sum(fnum(r.get("total_cost_usd_report")) for r in itemized)
    total_cny = sum(fnum(r.get("total_cost_cny_report")) for r in itemized)
    q235_rows = [r for r in itemized if r.get("model_label") == "qwen235b"]
    return {
        "created_at_utc": now_utc(),
        "checks": {
            "trace_rows": len(combined),
            "cost_itemized_rows": len(itemized),
            "unique_trace_datasets": len({r["dataset_name"] for r in combined}),
            "trace_by_run": dict(trace_by_model),
            "cost_by_model_label": dict(cost_by_model),
            "zero_qwen_cost_rows": sum(1 for r in itemized if fnum(r.get("qwen_chat_cost_cny")) <= 0),
            "zero_q0_embedding_rows": sum(1 for r in itemized if fnum(r.get("q0_embedding_cost_usd_est")) <= 0),
            "zero_judge_cost_rows": sum(1 for r in itemized if fnum(r.get("judge_cost_usd")) <= 0),
            "zero_total_cost_rows": sum(1 for r in itemized if fnum(r.get("total_cost_usd_report")) <= 0),
            "qwen235b_forced_rows": len(q235_rows),
            "qwen235b_forced_retrieval_zero_rows": sum(1 for r in q235_rows if inum(r.get("retrieval_calls")) < 1),
            "qwen235b_forced_success_rate": sum(inum(r.get("success")) for r in q235_rows) / len(q235_rows),
            "summary_total_cost_usd_matches_itemized": abs(total_usd - fnum(summary["totals"]["total_cost_usd_report"])) < 1e-9,
            "summary_total_cost_cny_matches_itemized": abs(total_cny - fnum(summary["totals"]["total_cost_cny_report"])) < 1e-9,
            "qwen235b_cost_usd_gt_qwen32b": sum(
                fnum(r.get("total_cost_usd_report")) for r in q235_rows
            )
            > sum(fnum(r.get("total_cost_usd_report")) for r in itemized if r.get("model_label") == "qwen32b"),
        },
        "totals_from_itemized": {
            "total_cost_usd_report": total_usd,
            "total_cost_cny_report": total_cny,
        },
    }


def write_cost_markdown(summary: dict[str, Any], validation: dict[str, Any]) -> None:
    lines = [
        "# OpenClaw Qwen Cost Reconstruction",
        "",
        "This package uses the 20260427 OpenClaw 8B/32B results and replaces the 235B slice with the final forced-retrieve rollup.",
        "No API calls were made during cost reconstruction.",
        "",
        "## Total",
        "",
        f"- Items: {summary['totals']['items']}",
        f"- Success: {summary['totals']['success_sum']} / {summary['totals']['items']} ({summary['totals']['success_rate']:.3f})",
        f"- Total cost: {summary['totals']['total_cost_cny_report']:.6f} CNY / {summary['totals']['total_cost_usd_report']:.6f} USD",
        f"- Qwen chat: {summary['totals']['qwen_chat_cost_cny']:.6f} CNY / {summary['totals']['qwen_chat_cost_usd_report']:.6f} USD",
        f"- q0 embedding estimate: {summary['totals']['q0_embedding_cost_usd_est']:.6f} USD",
        f"- q1 query embedding estimate: {summary['totals']['q1_query_embedding_cost_usd_est']:.6f} USD",
        f"- Judge: {summary['totals']['judge_cost_usd']:.6f} USD",
        "",
        "## By Model",
        "",
        "| model | success | total USD | Qwen CNY | q0 emb USD | judge USD | retrieval mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["by_model"], key=lambda r: r["model_label"]):
        lines.append(
            f"| {row['model_label']} | {int(row['success_sum'])}/{int(row['items'])} ({row['success_rate']:.3f}) "
            f"| {row['total_cost_usd_report']:.6f} | {row['qwen_chat_cost_cny']:.6f} "
            f"| {row['q0_embedding_cost_usd_est']:.6f} | {row['judge_cost_usd']:.6f} "
            f"| {row['retrieval_calls_mean']:.3f} |"
        )
    lines += [
        "",
        "## Validation",
        "",
        "```json",
        json.dumps(validation["checks"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (NEW_PACKAGE / "00_summary/cost_reconstruction_aliyun_qwen.md").write_text("\n".join(lines))
    (NEW_PACKAGE / "00_summary/cost_reconstruction_validation.md").write_text(
        "# Cost Reconstruction Validation\n\n```json\n"
        + json.dumps(validation, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def rebuild_overview(cost_summary: dict[str, Any], combined: list[dict[str, str]]) -> None:
    old_overview = load_json(OLD_PACKAGE / "00_summary/overview.json", {})
    forced_summary = load_json(FORCED_PACKAGE / "00_summary/qwen235b_forced_retrieve_final_summary.json", {})
    old_source_roots = {m["slug"]: m.get("selected_source_roots", {}) for m in old_overview.get("models", [])}
    old_source_roots["qwen235b"] = forced_summary.get("by_root", {})

    models = []
    by_run = defaultdict(list)
    for row in combined:
        by_run[row["run_name"]].append(row)
    for slug, label, _mid, _file in RUNS:
        rows = by_run[slug]
        models.append(
            {
                "slug": slug,
                "label": label,
                "completed": len(rows),
                "failed": 0,
                "success_rate": mean(rows, "success"),
                "mean_recorded_cost_usd": mean(rows, "total_cost_usd"),
                "total_recorded_cost_usd": sum(fnum(r.get("total_cost_usd")) for r in rows),
                "retrieval_calls_mean": mean(rows, "retrieval_calls"),
                "retrieval_calls_max": max(inum(r.get("retrieval_calls")) for r in rows),
                "context_tokens_mean": mean(rows, "context_tokens"),
                "total_duration_ms_mean": mean(rows, "total_duration_ms"),
                "selected_source_roots": old_source_roots.get(slug, {}),
                "replacement_note": "Final forced-retrieve rollup replaces original 20260427 qwen235b slice."
                if slug == "qwen235b"
                else "",
            }
        )

    overview = {
        "package_root": str(NEW_PACKAGE),
        "created_at_utc": now_utc(),
        "benchmark": old_overview.get("benchmark", "LongMemEval fixed2k balanced50x4x5"),
        "dataset_root": str(DATASET_ROOT),
        "dataset_list": str(DATASET_LIST),
        "expected_items_per_model": 1000,
        "models": models,
        "cost_summary": {
            "total_cost_cny_report": cost_summary["totals"]["total_cost_cny_report"],
            "total_cost_usd_report": cost_summary["totals"]["total_cost_usd_report"],
            "qwen_chat_cost_cny": cost_summary["totals"]["qwen_chat_cost_cny"],
            "q0_embedding_cost_usd_est": cost_summary["totals"]["q0_embedding_cost_usd_est"],
            "q1_query_embedding_cost_usd_est": cost_summary["totals"]["q1_query_embedding_cost_usd_est"],
            "judge_cost_usd": cost_summary["totals"]["judge_cost_usd"],
        },
        "important_notes": [
            "This is the same OpenClaw LongMemEval package layout as qwen_openclaw_longmemeval_20260427_package.",
            "The original qwen235b results are replaced with qwen_openclaw_235b_forced_retrieve_final_20260503_package.",
            "The forced 235B slice requires at least one memory retrieval for each accepted row; qwen235b retrieval_zero_rows=0.",
            "Costs are reconstructed offline from saved traces and q0/q1 embedding estimates; no API calls were made during packaging.",
        ],
    }
    write_json(NEW_PACKAGE / "00_summary/overview.json", overview)

    source_roots = {
        "package_root": str(NEW_PACKAGE),
        "source_packages": {
            "qwen8b": str(OLD_PACKAGE),
            "qwen32b": str(OLD_PACKAGE),
            "qwen235b": str(FORCED_PACKAGE),
        },
        "selected_source_roots_by_model": old_source_roots,
    }
    write_json(NEW_PACKAGE / "00_summary/source_roots.json", source_roots)


def rebuild_quality_notes(combined: list[dict[str, str]]) -> None:
    forced_summary = load_json(FORCED_PACKAGE / "00_summary/qwen235b_forced_retrieve_final_summary.json", {})
    rows = []
    q235 = [r for r in combined if r["run_name"] == "qwen235b"]
    empty = [r for r in q235 if not str(r.get("agent_output", "")).strip()]
    for row in empty:
        rows.append(
            {
                "model": "qwen235b",
                "dataset_name": row["dataset_name"],
                "issue": "empty_agent_output",
                "success": row.get("success", ""),
                "retrieval_calls": row.get("retrieval_calls", ""),
                "note": "Real raw output failure retained in final forced-retrieve rollup, not a packaging error.",
            }
        )
    for dirty in forced_summary.get("dirty_rows_excluded", []):
        rows.append(
            {
                "model": "qwen235b",
                "dataset_name": dirty.get("dataset_name", ""),
                "issue": dirty.get("issue", "excluded_dirty_row"),
                "success": "",
                "retrieval_calls": "",
                "note": f"Excluded from final rollup: {dirty.get('root', '')}",
            }
        )
    write_csv(NEW_PACKAGE / "00_summary/data_quality_exceptions.csv", rows, ["model", "dataset_name", "issue", "success", "retrieval_calls", "note"])
    write_json(NEW_PACKAGE / "00_summary/data_quality_exceptions.json", rows)
    md = [
        "# Data Quality Exceptions",
        "",
        f"- qwen235b forced-retrieve accepted rows: {len(q235)}",
        f"- qwen235b retrieval_zero_rows in accepted trace: {sum(1 for r in q235 if inum(r.get('retrieval_calls')) < 1)}",
        f"- qwen235b empty agent outputs retained: {len(empty)}",
        f"- qwen235b dirty/excluded rows from rollup manifest: {len(forced_summary.get('dirty_rows_excluded', []))}",
        "",
        "The retained empty output is a real failed raw run, not a packaging failure.",
        "",
    ]
    (NEW_PACKAGE / "00_summary/empty_output_diagnosis.md").write_text("\n".join(md))


def rebuild_run_contract() -> None:
    contract = {
        "runner": "systems/openclaw/benchmarks/memory_mvp/run_openclaw_benchmark.py",
        "probe": "systems/openclaw/benchmarks/memory_mvp/probe_dataset_item.py",
        "official_backend_wrapper": "systems/openclaw/benchmarks/memory_mvp/official_openclaw_memory.py",
        "benchmark": "LongMemEval fixed2k balanced50x4x5",
        "dataset_root": str(DATASET_ROOT),
        "dataset_list": str(DATASET_LIST),
        "shared_base_settings": {
            "memory_backend": "official",
            "memory_agent_profile": "openclaw_fidelity",
            "agent_mode": "memory_tools",
            "sources": ["memory"],
            "top_k": 6,
            "max_agent_steps": 6,
            "chunk_tokens": 400,
            "chunk_overlap": 80,
            "candidate_multiplier": 4,
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "eval_model": "gpt-4o-mini",
            "eval_prompt_style": "memos_json",
            "eval_num_runs": 3,
            "embedding_model": "text-embedding-3-small",
            "cleanup_q0_after_item": True,
            "continue_on_error": True,
        },
        "qwen8b_and_qwen32b_contract": {
            "description": "Copied from qwen_openclaw_longmemeval_20260427_package without rerun.",
            "force_min_memory_searches": 0,
            "important_profile_checks": {
                "searchable_highlights": False,
                "rewrite_memory_search_payload": False,
                "force_first_memory_search": False,
                "extra_benchmark_hints": False,
                "tool_choice_first_step": "auto",
            },
            "core_command_template": (
                "run_openclaw_benchmark.py --memory-backend official "
                "--memory-agent-profile openclaw_fidelity --top-k 6 --max-agent-steps 6 "
                "--sources memory --cleanup-q0-after-item --continue-on-error "
                "--eval-model gpt-4o-mini --eval-prompt-style memos_json --eval-num-runs 3 "
                "--embedding-model text-embedding-3-small --chat-model <qwen3-8b|qwen3-32b>"
            ),
        },
        "qwen235b_replacement_contract": {
            "description": "Final forced-retrieve replacement rollup from qwen_openclaw_235b_forced_retrieve_final_20260503_package.",
            "chat_model": "qwen/qwen3-235b-a22b-2507",
            "priced_as": "qwen3-235b-a22b-instruct-2507",
            "chat_endpoint_env": "OPENROUTER_BASE_URL / OPENROUTER_API_KEY",
            "embedding_endpoint_env": "OTHER_BASE_URL / OTHER_API_KEY or GPT_BASE_URL / GPT_API_KEY depending on source root",
            "eval_endpoint_env": "OTHER_BASE_URL / OTHER_API_KEY",
            "force_min_memory_searches": 1,
            "accepted_rows_require_retrieval_calls_at_least": 1,
            "important_profile_checks": {
                "searchable_highlights": False,
                "rewrite_memory_search_payload": False,
                "force_first_memory_search": False,
                "extra_benchmark_hints": False,
                "tool_choice_first_step": "auto until harness enforces the minimum memory_search count",
            },
            "embedding_stability_knobs": {
                "OPENCLAW_MEMORY_EMBEDDING_BATCH_MAX_TOKENS": 8000,
                "OPENCLAW_MEMORY_EMBEDDING_BATCH_TIMEOUT_REMOTE_MS": 300000,
                "ITEM_TIMEOUT_SECONDS": 3600,
            },
            "core_command_template": (
                "run_openclaw_benchmark.py --memory-backend official "
                "--memory-agent-profile openclaw_fidelity --agent-mode memory_tools "
                "--top-k 6 --max-agent-steps 6 --force-min-memory-searches 1 "
                "--sources memory --chunk-tokens 400 --chunk-overlap 80 "
                "--candidate-multiplier 4 --vector-weight 0.7 --text-weight 0.3 "
                "--cleanup-q0-after-item --continue-on-error --eval-model gpt-4o-mini "
                "--eval-prompt-style memos_json --eval-num-runs 3 "
                "--embedding-model text-embedding-3-small --chat-model qwen/qwen3-235b-a22b-2507"
            ),
        },
        "q0_policy": "per-item transient q0/index build, then cleanup-q0-after-item; no shared durable q0 cache",
        "parallelism_history": {
            "qwen8b": "multiple resumable roots from the 20260427 package",
            "qwen32b": "multiple resumable roots from the 20260427 package",
            "qwen235b": "forced-retrieve replacement rollup, mixed stable roots with lower embedding batch sizes/timeouts",
        },
        "offline_packaging_note": "This package was assembled without API calls from already-completed traces/results.",
    }
    write_json(NEW_PACKAGE / "00_summary/run_contract.json", contract)


def rebuild_samples(combined: list[dict[str, str]]) -> None:
    sample_dir = NEW_PACKAGE / "03_samples"
    for slug, _label, _mid, _file in RUNS:
        rows = [r for r in combined if r["run_name"] == slug][:50]
        trace_samples = [{k: v for k, v in r.items() if k != "trace_json"} for r in rows]
        result_samples = []
        for r in rows:
            try:
                result_samples.append(json.loads(r["trace_json"]))
            except json.JSONDecodeError:
                result_samples.append({"dataset_name": r.get("dataset_name"), "parse_error": True})
        write_json(sample_dir / f"{slug}_trace_samples_50.json", trace_samples)
        write_json(sample_dir / f"{slug}_result_samples_50.json", result_samples)
    sample_md = [
        "# Sample Review",
        "",
        "Each model has first-50 trace/result samples regenerated from the package traces.",
        "The qwen235b samples come from the final forced-retrieve replacement trace.",
        "",
    ]
    (sample_dir / "sample_review.md").write_text("\n".join(sample_md))


def rebuild_audit_summary(cost_summary: dict[str, Any], validation: dict[str, Any], combined: list[dict[str, str]]) -> None:
    checks = validation["checks"]
    audit = {
        "created_at_utc": now_utc(),
        "package_root": str(NEW_PACKAGE),
        "row_counts": {
            "combined_trace": len(combined),
            "combined_results_jsonl": sum(1 for _ in (NEW_PACKAGE / "01_traces/qwen_all_models_combined_results.jsonl").open()),
            "combined_search_jsonl": sum(1 for _ in (NEW_PACKAGE / "01_traces/qwen_all_models_combined_search_results.jsonl").open()),
        },
        "cost_totals": cost_summary["totals"],
        "validation_checks": checks,
        "status": "PASS"
        if all(
            [
                checks["trace_rows"] == 3000,
                checks["cost_itemized_rows"] == 3000,
                checks["qwen235b_forced_rows"] == 1000,
                checks["qwen235b_forced_retrieval_zero_rows"] == 0,
                checks["zero_total_cost_rows"] == 0,
                checks["summary_total_cost_usd_matches_itemized"],
                checks["qwen235b_cost_usd_gt_qwen32b"],
            ]
        )
        else "CHECK",
    }
    write_json(NEW_PACKAGE / "00_summary/audit_summary.json", audit)
    write_json(NEW_PACKAGE / "00_summary/cost_sanity_check.json", audit)
    (NEW_PACKAGE / "00_summary/cost_sanity_check.md").write_text(
        "# Cost Sanity Check\n\n"
        f"- Status: {audit['status']}\n"
        f"- Combined rows: {audit['row_counts']['combined_trace']}\n"
        f"- Itemized cost rows: {checks['cost_itemized_rows']}\n"
        f"- qwen235b forced rows: {checks['qwen235b_forced_rows']}\n"
        f"- qwen235b retrieval-zero rows: {checks['qwen235b_forced_retrieval_zero_rows']}\n"
        f"- Total cost: {cost_summary['totals']['total_cost_cny_report']:.6f} CNY / "
        f"{cost_summary['totals']['total_cost_usd_report']:.6f} USD\n"
        f"- qwen235b cost > qwen32b: {checks['qwen235b_cost_usd_gt_qwen32b']}\n"
    )


def maybe_generate_figures() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - only for missing optional dependency.
        (NEW_PACKAGE / "05_figures/FIGURE_GENERATION_SKIPPED.txt").write_text(str(exc))
        return

    fig_dir = NEW_PACKAGE / "05_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_metrics = read_csv(NEW_PACKAGE / "02_metrics/model_metrics.csv")
    by_s = read_csv(NEW_PACKAGE / "02_metrics/metrics_by_s_models.csv")
    by_family = read_csv(NEW_PACKAGE / "02_metrics/metrics_by_family_models.csv")
    cost_by_s = read_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_by_s_models.csv")

    def save_bar(rows: list[dict[str, str]], x_field: str, y_field: str, group_field: str, title: str, path: Path) -> None:
        groups = sorted({r[group_field] for r in rows})
        xs = sorted({r[x_field] for r in rows}, key=lambda v: int(v) if str(v).isdigit() else str(v))
        width = 0.8 / max(1, len(groups))
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = range(len(xs))
        for idx, g in enumerate(groups):
            vals = [fnum(next((r[y_field] for r in rows if r[group_field] == g and r[x_field] == x), 0)) for x in xs]
            offset = (idx - (len(groups) - 1) / 2) * width
            ax.bar([p + offset for p in positions], vals, width=width, label=g)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(xs)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    save_bar(model_metrics, "run_name", "success_rate", "model_label", "Success Rate by Model", fig_dir / "qwen_models_overview_panel.png")
    save_bar(by_s, "s", "success_rate", "model_label", "Success Rate by s", fig_dir / "success_rate_by_s_models.png")
    save_bar(by_s, "s", "f1_mean", "model_label", "F1 by s", fig_dir / "f1_by_s_models.png")
    save_bar(by_s, "s", "retrieval_calls_mean", "model_label", "Retrieval Calls by s", fig_dir / "retrieval_calls_by_s_models.png")
    save_bar(by_s, "s", "context_tokens_mean", "model_label", "Context Tokens by s", fig_dir / "context_tokens_by_s_models.png")
    save_bar(by_s, "s", "total_duration_ms_mean", "model_label", "Total Duration by s", fig_dir / "total_duration_by_s_models.png")
    save_bar(by_family, "question_type_short", "success_rate", "model_label", "Success Rate by Family", fig_dir / "success_rate_by_family_models.png")
    save_bar(by_family, "question_type_short", "retrieval_calls_mean", "model_label", "Retrieval Calls by Family", fig_dir / "retrieval_calls_by_family_models.png")
    cost_plot_rows = [
        {"s": r["s"], "model_label": r["model_label"], "total_cost_usd_report": r["total_cost_usd_report"]}
        for r in cost_by_s
    ]
    save_bar(cost_plot_rows, "s", "total_cost_usd_report", "model_label", "Total Reconstructed Cost by s", fig_dir / "total_cost_by_s_models.png")


def write_readme_and_index(cost_summary: dict[str, Any], validation: dict[str, Any]) -> None:
    by_model = {r["model_label"]: r for r in cost_summary["by_model"]}
    readme = [
        "# Qwen OpenClaw LongMemEval Package with Forced-Retrieve 235B",
        "",
        "This package mirrors the previous OpenClaw LongMemEval package layout, while replacing the original Qwen 235B slice with the final forced-retrieve Qwen 235B rollup.",
        "",
        "## What Changed",
        "",
        "- Qwen 8B and Qwen 32B are copied from `qwen_openclaw_longmemeval_20260427_package`.",
        "- Qwen 235B is replaced by `qwen_openclaw_235b_forced_retrieve_final_20260503_package`.",
        "- Combined traces/results/search JSONL, metrics, reconstructed costs, figures, samples, summaries, and manifests were rebuilt offline.",
        "- No API calls were made during packaging or cost reconstruction.",
        "",
        "## Headline Metrics",
        "",
        "| model | success | retrieval mean | reconstructed USD | reconstructed CNY |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["qwen8b", "qwen32b", "qwen235b"]:
        row = by_model[label]
        readme.append(
            f"| {label} | {int(row['success_sum'])}/{int(row['items'])} ({row['success_rate']:.3f}) "
            f"| {row['retrieval_calls_mean']:.3f} | {row['total_cost_usd_report']:.6f} "
            f"| {row['total_cost_cny_report']:.6f} |"
        )
    readme += [
        "",
        "## Total Cost",
        "",
        f"- Total: {cost_summary['totals']['total_cost_cny_report']:.6f} CNY / {cost_summary['totals']['total_cost_usd_report']:.6f} USD",
        f"- Qwen chat: {cost_summary['totals']['qwen_chat_cost_cny']:.6f} CNY / {cost_summary['totals']['qwen_chat_cost_usd_report']:.6f} USD",
        f"- q0 embedding estimate: {cost_summary['totals']['q0_embedding_cost_usd_est']:.6f} USD",
        f"- q1 query embedding estimate: {cost_summary['totals']['q1_query_embedding_cost_usd_est']:.6f} USD",
        f"- Judge: {cost_summary['totals']['judge_cost_usd']:.6f} USD",
        "",
        "## Validation",
        "",
        f"- Combined trace rows: {validation['checks']['trace_rows']}",
        f"- Cost itemized rows: {validation['checks']['cost_itemized_rows']}",
        f"- qwen235b forced rows: {validation['checks']['qwen235b_forced_rows']}",
        f"- qwen235b retrieval-zero rows: {validation['checks']['qwen235b_forced_retrieval_zero_rows']}",
        f"- zero total-cost rows: {validation['checks']['zero_total_cost_rows']}",
        f"- qwen235b reconstructed cost > qwen32b: {validation['checks']['qwen235b_cost_usd_gt_qwen32b']}",
        "",
        "## Key Files",
        "",
        "- `00_summary/overview.json`",
        "- `00_summary/cost_reconstruction_aliyun_qwen.json`",
        "- `00_summary/cost_reconstruction_validation.json`",
        "- `01_traces/qwen_all_models_combined_trace.csv`",
        "- `01_traces/qwen_all_models_combined_results.jsonl`",
        "- `01_traces/qwen_all_models_combined_search_results.jsonl`",
        "- `02_metrics/cost_reconstructed_aliyun_itemized.csv`",
        "- `02_metrics/model_metrics.csv`",
        "- `06_manifests/package_file_manifest.json`",
        "",
    ]
    (NEW_PACKAGE / "README.md").write_text("\n".join(readme))
    mentor = [
        "# Mentor Index",
        "",
        "Use this package for the OpenClaw LongMemEval analysis where Qwen 235B must have at least one retrieval.",
        "",
        "Start with:",
        "",
        "- `README.md`",
        "- `00_summary/overview.json`",
        "- `00_summary/cost_reconstruction_aliyun_qwen.md`",
        "- `00_summary/cost_reconstruction_validation.md`",
        "- `01_traces/qwen_all_models_combined_trace.csv`",
        "- `02_metrics/cost_reconstructed_aliyun_itemized.csv`",
        "",
    ]
    (NEW_PACKAGE / "MENTOR_INDEX.md").write_text("\n".join(mentor))


def rebuild_manifests() -> None:
    manifest_dir = NEW_PACKAGE / "06_manifests"
    # The manifest records itself, so write until the self-referential file sizes
    # stabilize. In practice this converges in two or three passes.
    previous: list[dict[str, Any]] | None = None
    files: list[dict[str, Any]] = []
    for _ in range(10):
        files = []
        for path in sorted(NEW_PACKAGE.rglob("*")):
            if path.is_file():
                rel = path.relative_to(NEW_PACKAGE)
                stat = path.stat()
                files.append({"path": str(rel), "bytes": stat.st_size})
        write_json(manifest_dir / "package_file_manifest.json", {"package_root": str(NEW_PACKAGE), "files": files, "file_count": len(files)})
        write_csv(manifest_dir / "package_file_manifest.csv", files, ["path", "bytes"])
        (manifest_dir / "package_file_manifest.txt").write_text("\n".join(f["path"] for f in files) + "\n")
        if previous == files:
            break
        previous = files


def validate_package() -> dict[str, Any]:
    checks = {}
    for slug, _label, _mid, _file in RUNS:
        checks[f"{slug}_trace_rows"] = len(read_csv(NEW_PACKAGE / f"01_traces/{slug}_trace_q1_full.csv"))
        checks[f"{slug}_results_rows"] = sum(1 for _ in (NEW_PACKAGE / f"01_traces/{slug}_results_full.jsonl").open())
        checks[f"{slug}_search_rows"] = sum(1 for _ in (NEW_PACKAGE / f"01_traces/{slug}_search_results_full.jsonl").open())
    combined = read_csv(NEW_PACKAGE / "01_traces/qwen_all_models_combined_trace.csv")
    cost_rows = read_csv(NEW_PACKAGE / "02_metrics/cost_reconstructed_aliyun_itemized.csv")
    q235 = [r for r in combined if r["run_name"] == "qwen235b"]
    checks.update(
        {
            "combined_trace_rows": len(combined),
            "combined_results_rows": sum(1 for _ in (NEW_PACKAGE / "01_traces/qwen_all_models_combined_results.jsonl").open()),
            "combined_search_rows": sum(1 for _ in (NEW_PACKAGE / "01_traces/qwen_all_models_combined_search_results.jsonl").open()),
            "cost_itemized_rows": len(cost_rows),
            "qwen235b_retrieval_zero_rows": sum(1 for r in q235 if inum(r.get("retrieval_calls")) < 1),
            "qwen235b_success_rate": mean(q235, "success"),
            "qwen235b_empty_agent_output_rows": sum(1 for r in q235 if not str(r.get("agent_output", "")).strip()),
        }
    )
    checks["status"] = "PASS" if all(
        checks[k] == 1000 for k in [
            "qwen8b_trace_rows",
            "qwen8b_results_rows",
            "qwen8b_search_rows",
            "qwen32b_trace_rows",
            "qwen32b_results_rows",
            "qwen32b_search_rows",
            "qwen235b_trace_rows",
            "qwen235b_results_rows",
            "qwen235b_search_rows",
        ]
    ) and checks["combined_trace_rows"] == 3000 and checks["combined_results_rows"] == 3000 and checks["combined_search_rows"] == 3000 and checks["cost_itemized_rows"] == 3000 and checks["qwen235b_retrieval_zero_rows"] == 0 else "CHECK"
    write_json(NEW_PACKAGE / "00_summary/package_validation.json", checks)
    return checks


def make_tarball() -> Path:
    tar_path = NEW_PACKAGE.with_suffix("").with_name(NEW_PACKAGE.name + ".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(NEW_PACKAGE, arcname=NEW_PACKAGE.name)
    return tar_path


def main() -> None:
    if not OLD_PACKAGE.exists():
        raise FileNotFoundError(OLD_PACKAGE)
    if not FORCED_PACKAGE.exists():
        raise FileNotFoundError(FORCED_PACKAGE)
    if NEW_PACKAGE.exists():
        shutil.rmtree(NEW_PACKAGE)
    shutil.copytree(OLD_PACKAGE, NEW_PACKAGE)

    forced_trace = FORCED_PACKAGE / "01_traces/qwen235b_forced_retrieve_trace_q1_full.csv"
    shutil.copy2(forced_trace, NEW_PACKAGE / "01_traces/qwen235b_trace_q1_full.csv")
    scripts_dir = NEW_PACKAGE / "07_scripts"
    scripts_dir.mkdir(exist_ok=True)
    forced_scripts = FORCED_PACKAGE / "07_scripts"
    if forced_scripts.exists():
        for script in sorted(forced_scripts.iterdir()):
            if script.is_file() and script.suffix in {".py", ".sh"}:
                shutil.copy2(script, scripts_dir / script.name)
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)

    build_results_and_search_jsonl()
    combined = rebuild_combined_traces()
    rebuild_metrics(combined)
    cost_summary = rebuild_reconstructed_costs(combined)
    validation = load_json(NEW_PACKAGE / "00_summary/cost_reconstruction_validation.json")
    rebuild_overview(cost_summary, combined)
    rebuild_quality_notes(combined)
    rebuild_run_contract()
    rebuild_samples(combined)
    rebuild_audit_summary(cost_summary, validation, combined)
    write_readme_and_index(cost_summary, validation)
    maybe_generate_figures()
    package_validation = validate_package()
    rebuild_manifests()
    tar_path = make_tarball()

    print(json.dumps({
        "package": str(NEW_PACKAGE),
        "tarball": str(tar_path),
        "package_validation": package_validation,
        "total_cost_cny": cost_summary["totals"]["total_cost_cny_report"],
        "total_cost_usd": cost_summary["totals"]["total_cost_usd_report"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
