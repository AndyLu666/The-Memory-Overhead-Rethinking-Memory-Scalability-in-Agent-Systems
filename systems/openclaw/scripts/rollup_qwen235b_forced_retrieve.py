#!/usr/bin/env python3
"""Build the current Qwen-235B forced-retrieve OpenClaw rollup.

The recovery run is intentionally spread across several roots because large
OpenClaw q0 indexing jobs are safer at low parallelism. This script merges the
validated prior manifest with subsequent clean forced-retrieve rows and writes
the current trace/manifest/summary without marking unfinished items complete.
"""

from __future__ import annotations

import collections
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_OPENCLAW_DIR = SCRIPT_DIR.parent

BASE = Path(os.environ.get("OPENCLAW_RESULTS_BASE", "results/openclaw")).expanduser().resolve()
RECOVERY = Path(
    os.environ.get(
        "OPENCLAW_RECOVERY_DIR",
        str(BASE / "qwen235b_forced_retrieve_20260430_recovery"),
    )
)
PACKAGE = Path(
    os.environ.get(
        "OPENCLAW_20260427_PACKAGE",
        "results/qwen_openclaw_longmemeval_20260427_package",
    )
).expanduser().resolve()
TARGET_LIST = Path(
    os.environ.get(
        "OPENCLAW_TARGET_LIST",
        str(
            ARTIFACT_OPENCLAW_DIR
            / "benchmarks"
            / "memory_mvp"
            / "dataset_lists"
            / "longmemeval_fixed2k_fidelity_balanced50x4x5_seed20260421.txt"
        ),
    )
).expanduser().resolve()

TRACE_FIELDS = [
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

OUT_PREFIX = RECOVERY / "current_rollup_qwen235b_forced_retrieve_until_20260502"
OUT_TRACE = Path(os.environ.get("OPENCLAW_ROLLUP_TRACE", str(OUT_PREFIX) + "_trace.csv"))
OUT_MANIFEST = Path(os.environ.get("OPENCLAW_ROLLUP_MANIFEST", str(OUT_PREFIX) + "_manifest.jsonl"))
OUT_SUMMARY = Path(os.environ.get("OPENCLAW_ROLLUP_SUMMARY", str(OUT_PREFIX) + "_summary.json"))
OUT_REMAINING = Path(os.environ.get("OPENCLAW_ROLLUP_REMAINING", str(OUT_PREFIX) + "_remaining.txt"))
OUT_BY_S = Path(os.environ.get("OPENCLAW_ROLLUP_BY_S", str(OUT_PREFIX) + "_by_s.csv"))
OUT_BY_TYPE = Path(os.environ.get("OPENCLAW_ROLLUP_BY_TYPE", str(OUT_PREFIX) + "_by_type.csv"))
OUT_LIVE70 = Path(os.environ.get("OPENCLAW_LIVE70_SUMMARY", str(RECOVERY / "current_remaining70_live_progress_20260502_summary.json")))


def load_target_order() -> list[str]:
    return [line.strip() for line in TARGET_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_traces() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    by_root_dataset: dict[tuple[str, str], dict[str, str]] = {}
    by_dataset_any: dict[str, dict[str, str]] = {}

    def load_trace(root_name: str, path: Path) -> None:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset = (row.get("dataset_name") or "").strip()
                if not dataset:
                    continue
                clean = {field: row.get(field, "") for field in TRACE_FIELDS}
                by_root_dataset[(root_name, dataset)] = clean
                by_dataset_any.setdefault(dataset, clean)

    for trace_path in sorted(BASE.glob("qwen235b_forced_retrieve_openrouter_*/trace_q1_full.csv")):
        load_trace(trace_path.parent.name, trace_path)
    load_trace(
        "qwen_openclaw_longmemeval_20260427_package",
        PACKAGE / "01_traces/qwen235b_trace_q1_full.csv",
    )
    return by_root_dataset, by_dataset_any


def count_search_hits(root_name: str, dataset: str) -> int | None:
    path = BASE / root_name / "derived" / dataset / "search_results.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    calls = payload.get("memory_search_calls") or []
    return sum(len(call.get("results") or []) for call in calls if isinstance(call, dict))


def main() -> None:
    target_order = load_target_order()
    target_set = set(target_order)
    by_root_dataset, by_dataset_any = load_traces()
    accepted: dict[str, dict] = {}
    missing_rows: list[dict] = []
    dirty_rows: list[dict] = []

    def add_row(dataset: str, source: str, root_name: str, priority: int, meta: dict | None = None) -> None:
        meta = meta or {}
        dataset = (dataset or "").strip()
        if dataset not in target_set:
            return
        row = by_root_dataset.get((root_name, dataset)) or by_dataset_any.get(dataset)
        if row is None:
            missing_rows.append({"dataset_name": dataset, "source": source, "root": root_name})
            return
        retrieval_calls = str(row.get("retrieval_calls") or meta.get("retrieval_calls") or "").strip()
        if retrieval_calls in ("", "0", "0.0"):
            dirty_rows.append(
                {
                    "dataset_name": dataset,
                    "source": source,
                    "root": root_name,
                    "issue": "retrieval_calls_zero_or_missing",
                }
            )
            return
        if dataset in accepted and priority < accepted[dataset]["priority"]:
            return
        accepted[dataset] = {
            "dataset_name": dataset,
            "source": source,
            "root": root_name,
            "priority": priority,
            "success": str(row.get("success") or meta.get("success") or ""),
            "retrieval_calls": retrieval_calls,
            "hits": meta.get("hits") if meta.get("hits") is not None else count_search_hits(root_name, dataset),
            "model": row.get("model") or meta.get("model") or "qwen/qwen3-235b-a22b-2507",
            "row": row,
        }

    base_manifest = RECOVERY / "accepted_forced_plus_prev_retrieved_ge1_manifest_20260501.jsonl"
    for line in base_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        add_row(entry["dataset_name"], entry.get("source", "base_manifest"), entry.get("root", ""), 10, entry)

    # These nine rows are the clean rows that moved the accepted count from 919
    # to 928 before the final p1/batchfix split.
    for root_name in (
        "qwen235b_forced_retrieve_openrouter_key1_gptembed_remaining41_reuseprev_p5_20260501_v1",
        "qwen235b_forced_retrieve_openrouter_key2_otherembed_remaining40_reuseprev_p5_20260501_v1",
    ):
        path = BASE / root_name / "trace_q1_full.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    add_row(row.get("dataset_name", ""), "forced_current_reuseprev_p5_clean", root_name, 20)

    for root_name in (
        "qwen235b_forced_retrieve_openrouter_key1_gptembed_remaining36_reuseprev_p1_20260501_v1",
        "qwen235b_forced_retrieve_openrouter_key2_otherembed_remaining36_reuseprev_p1_20260501_v1",
    ):
        path = BASE / root_name / "trace_q1_full.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    add_row(row.get("dataset_name", ""), "forced_current_p1_pre_batchfix", root_name, 30)

    for root in sorted(BASE.glob("qwen235b_forced_retrieve_openrouter_key*_otherembed_remaining35_stable_p2_20260502_v1")):
        path = root / "trace_q1_full.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    add_row(row.get("dataset_name", ""), "forced_current_stable_p2_remaining70", root.name, 40)

    for root in sorted(BASE.glob("qwen235b_forced_retrieve_openrouter_key*_otherembed_autoresume_p2_20260502_round*")):
        path = root / "trace_q1_full.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    add_row(row.get("dataset_name", ""), "forced_current_autoresume_p2_remaining70", root.name, 50)

    ordered = [accepted[dataset] for dataset in target_order if dataset in accepted]
    remaining = [dataset for dataset in target_order if dataset not in accepted]

    OUT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TRACE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for entry in ordered:
            writer.writerow({field: entry["row"].get(field, "") for field in TRACE_FIELDS})

    with OUT_MANIFEST.open("w", encoding="utf-8") as handle:
        for entry in ordered:
            manifest_entry = {
                key: entry[key]
                for key in ("dataset_name", "source", "root", "success", "retrieval_calls", "hits", "model")
            }
            handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    OUT_REMAINING.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")

    by_s = collections.defaultdict(lambda: {"n": 0, "success": 0, "remaining": 0})
    by_type = collections.defaultdict(lambda: {"n": 0, "success": 0})
    by_source = collections.Counter()
    by_root = collections.Counter()
    retrieval_hist = collections.Counter()
    cost_total = 0.0
    cost_missing = 0
    empty_agent_output = 0
    for entry in ordered:
        dataset = entry["dataset_name"]
        row = entry["row"]
        sbin = dataset.split("/", 1)[0]
        qtype = dataset.split("/")[1] if "/" in dataset else ""
        success = str(row.get("success") or "") == "1"
        by_s[sbin]["n"] += 1
        by_s[sbin]["success"] += int(success)
        by_type[qtype]["n"] += 1
        by_type[qtype]["success"] += int(success)
        by_source[entry["source"]] += 1
        by_root[entry["root"]] += 1
        retrieval_hist[str(row.get("retrieval_calls") or "")] += 1
        empty_agent_output += int(not row.get("agent_output"))
        try:
            cost_total += float(row.get("total_cost_usd") or 0.0)
        except Exception:
            cost_missing += 1
    for dataset in remaining:
        by_s[dataset.split("/", 1)[0]]["remaining"] += 1

    with OUT_BY_S.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["s", "n", "success", "success_rate", "remaining"])
        writer.writeheader()
        for key, value in sorted(by_s.items()):
            writer.writerow(
                {
                    "s": key,
                    "n": value["n"],
                    "success": value["success"],
                    "success_rate": round(value["success"] / value["n"], 6) if value["n"] else None,
                    "remaining": value.get("remaining", 0),
                }
            )

    with OUT_BY_TYPE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_type", "n", "success", "success_rate"])
        writer.writeheader()
        for key, value in sorted(by_type.items()):
            writer.writerow(
                {
                    "question_type": key,
                    "n": value["n"],
                    "success": value["success"],
                    "success_rate": round(value["success"] / value["n"], 6) if value["n"] else None,
                }
            )

    remaining70_path = RECOVERY / "remaining_after_p1_batchfix_20260501_batchfix_p5_all70.txt"
    remaining70 = [line.strip() for line in remaining70_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    remaining70_set = set(remaining70)
    live70 = [entry for entry in ordered if entry["dataset_name"] in remaining70_set]
    live70_completed = {entry["dataset_name"] for entry in live70}
    live70_remaining = [dataset for dataset in remaining70 if dataset not in live70_completed]
    live70_by_s = collections.defaultdict(lambda: {"n": 0, "success": 0})
    for entry in live70:
        sbin = entry["dataset_name"].split("/", 1)[0]
        live70_by_s[sbin]["n"] += 1
        live70_by_s[sbin]["success"] += int(str(entry["row"].get("success") or "") == "1")

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    success_count = sum(1 for entry in ordered if str(entry["row"].get("success") or "") == "1")
    summary = {
        "created_at_utc": created_at,
        "policy": (
            "rollup = prior accepted manifest (919) + p5 clean rows (+9) + "
            "p1 pre-batchfix rows (+2) + current stable/autoresume forced-retrieve rows; "
            "newer forced rows override same dataset; requires retrieval_calls>=1"
        ),
        "target_items": len(target_order),
        "accepted": len(ordered),
        "remaining": len(remaining),
        "success": success_count,
        "success_rate": round(success_count / len(ordered), 6) if ordered else None,
        "retrieval_zero_rows": sum(
            1 for entry in ordered if str(entry["row"].get("retrieval_calls") or "") in ("", "0", "0.0")
        ),
        "empty_agent_output_rows": empty_agent_output,
        "dirty_rows_excluded": dirty_rows,
        "missing_rows": missing_rows,
        "by_s": {
            key: {**value, "success_rate": round(value["success"] / value["n"], 6) if value["n"] else None}
            for key, value in sorted(by_s.items())
        },
        "by_type": {
            key: {**value, "success_rate": round(value["success"] / value["n"], 6) if value["n"] else None}
            for key, value in sorted(by_type.items())
        },
        "by_source": dict(by_source),
        "by_root": dict(by_root),
        "retrieval_hist": dict(retrieval_hist),
        "cost_total_usd_trace_sum": round(cost_total, 6),
        "cost_missing_or_unparseable_rows": cost_missing,
        "paths": {
            "trace": str(OUT_TRACE),
            "manifest": str(OUT_MANIFEST),
            "remaining": str(OUT_REMAINING),
            "by_s": str(OUT_BY_S),
            "by_type": str(OUT_BY_TYPE),
            "summary": str(OUT_SUMMARY),
            "live70_summary": str(OUT_LIVE70),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    live70_success = sum(1 for entry in live70 if str(entry["row"].get("success") or "") == "1")
    live70_summary = {
        "created_at_utc": created_at,
        "target_count": len(remaining70),
        "accepted": len(live70),
        "remaining": len(live70_remaining),
        "success": live70_success,
        "success_rate": round(live70_success / len(live70), 6) if live70 else None,
        "retrieval_zero_rows": sum(
            1 for entry in live70 if str(entry["row"].get("retrieval_calls") or "") in ("", "0", "0.0")
        ),
        "by_s": {
            key: {**value, "success_rate": round(value["success"] / value["n"], 6)}
            for key, value in sorted(live70_by_s.items())
        },
        "remaining_path": str(RECOVERY / "supervisor_qwen235b_forced_remaining70_remaining.txt"),
        "roots": dict(collections.Counter(entry["root"] for entry in live70)),
    }
    OUT_LIVE70.write_text(json.dumps(live70_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "live70": live70_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
