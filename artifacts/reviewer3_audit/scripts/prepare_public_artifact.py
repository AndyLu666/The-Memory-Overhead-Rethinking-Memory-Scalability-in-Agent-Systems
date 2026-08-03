#!/usr/bin/env python3
"""Build the public reviewer audit package from local result exports.

The public package keeps every trajectory-level field needed to recompute the
reported diagnostics. It intentionally excludes benchmark conversation text,
retrieved passages, provider payloads, and private filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)

BUDGETS = range(1, 7)
COMMON_FIELDS = [
    "record_set",
    "source_package",
    "benchmark",
    "memory_system",
    "scope",
    "model_label",
    "model",
    "scale",
    "scale_label",
    "trajectory_id",
    "task_id",
    "question_type",
    "trial",
    "retrieval_calls",
    "correctness_source",
    "success",
    "judge_success",
    "context_ok",
]

DERIVED_AGGREGATES = [
    "breakdown_by_budget.csv",
    "cleaning_audit_summary.json",
    "curve_metrics_by_s_budget.csv",
    "curve_metrics_complete_only_by_s_budget.csv",
    "data_quality_checks.csv",
    "failure_decomposition_by_s_budget.csv",
    "main_table_by_budget.csv",
    "main_table_complete_only_by_budget.csv",
    "metric_reconciliation.csv",
    "optional_parameter_coverage.csv",
    "run_summary.json",
    "source_file_inventory.csv",
    "stage_coverage_by_model_s.csv",
    "table_r1_longmemeval_complete_only_by_budget.csv",
    "table_r2_locomo_complete_only_by_budget.csv",
    "trace_integrity_by_model.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Workspace containing the local result packages.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def bool_int(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return 1
    if text in {"0", "false", "f", "no", ""}:
        return 0
    return int(float(text))


def int_value(value: object) -> int:
    return int(float(str(value).strip()))


def clean_model_label(model: str, filename: str = "") -> str:
    text = f"{model} {filename}".lower()
    if "235" in text:
        return "Qwen 235B"
    if "32" in text:
        return "Qwen 32B"
    if "8" in text and "qwen" in text:
        return "Qwen 8B"
    if "gpt" in text:
        return "GPT-5-mini"
    return model


def normalize_scale(value: object) -> int:
    text = str(value).strip().lower().replace("$", "")
    if text.startswith("s_"):
        text = text[2:]
    elif text.startswith("s"):
        text = text[1:]
    return int(float(text))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_published_text(artifact_root: Path) -> None:
    """Make copied text artifacts stable across source platforms."""
    for path in artifact_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if path.suffix.lower() == ".svg":
            lines = [line.rstrip() for line in lines]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_hierarchical(source_root: Path, artifact_root: Path) -> list[dict[str, object]]:
    source_dir = source_root / "derived_pass_at_b_20260414"
    source_path = source_dir / "annotated_traces_wide.csv"
    output_path = artifact_root / "data/hierarchical/trajectory_metrics.csv"

    public_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {source_path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            row["source_trace_file"] = Path(row.get("source_trace_file", "")).name
            public_rows.append(row)
            scale = normalize_scale(row["s"])
            common_rows.append(
                {
                    "record_set": "hierarchical",
                    "source_package": row["source_package"],
                    "benchmark": row["benchmark"],
                    "memory_system": row["memory_system"],
                    "scope": row["scope"],
                    "model_label": row["model_label"],
                    "model": row["model"],
                    "scale": scale,
                    "scale_label": f"s{scale}",
                    "trajectory_id": row["dataset_name"],
                    "task_id": row["task_id"],
                    "question_type": row["question_type"],
                    "trial": int_value(row["trial"]),
                    "retrieval_calls": int_value(row["retrieval_calls"]),
                    "correctness_source": "archived_llm_judge",
                    "success": bool_int(row["success"]),
                    "judge_success": bool_int(row["llm_judge"]),
                    "context_ok": bool_int(row["context_ok"]),
                }
            )
    write_csv(output_path, fieldnames, public_rows)

    aggregate_root = artifact_root / "aggregates/hierarchical"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    for filename in DERIVED_AGGREGATES:
        source_file = source_dir / filename
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        if filename == "run_summary.json":
            with source_file.open(encoding="utf-8") as source:
                summary = json.load(source)
            summary["output_dir"] = "artifacts/reviewer3_audit/aggregates/hierarchical"
            with (aggregate_root / filename).open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
            continue
        if filename != "source_file_inventory.csv":
            shutil.copy2(source_file, aggregate_root / filename)
            continue
        with source_file.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = []
            for row in reader:
                row["source_path"] = Path(row.get("source_path", "")).name
                rows.append(row)
        write_csv(aggregate_root / filename, list(reader.fieldnames or []), rows)
    return common_rows


def copy_additional_interfaces(
    source_root: Path, artifact_root: Path
) -> list[dict[str, object]]:
    source_dir = source_root / "memory_scale-paper-update/memory_system_scaling_assets"
    output_dir = artifact_root / "aggregates/additional_interfaces"
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(source_dir.iterdir()):
        if source_file.name in {"README.md", "memory_system_row_level.csv"}:
            continue
        target = output_dir / source_file.name
        if source_file.name != "memory_system_input_inventory.csv":
            shutil.copy2(source_file, target)
            continue
        with source_file.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = []
            for row in reader:
                for field in ("prediction_file", "eval_file", "retrieval_file"):
                    row[field] = Path(row.get(field, "")).name
                rows.append(row)
        write_csv(target, list(reader.fieldnames or []), rows)

    source_path = source_dir / "memory_system_row_level.csv"
    output_path = artifact_root / "data/additional_interfaces/trajectory_metrics.csv"
    shutil.copy2(source_path, output_path)

    common_rows: list[dict[str, object]] = []
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            scale = normalize_scale(row["s"])
            common_rows.append(
                {
                    "record_set": "additional_interfaces",
                    "source_package": "memory_system_scaling_assets",
                    "benchmark": "LongMemEval",
                    "memory_system": row["memory_system"],
                    "scope": "additional_interface_coverage",
                    "model_label": row["model_label"],
                    "model": row["model"],
                    "scale": scale,
                    "scale_label": f"s{scale}",
                    "trajectory_id": row["key"],
                    "task_id": row["question_id"],
                    "question_type": row["question_type"],
                    "trial": 1,
                    "retrieval_calls": int_value(row["retrieval_calls"]),
                    "correctness_source": "joined_evaluator_result",
                    "success": bool_int(row["success"]),
                    "judge_success": "",
                    "context_ok": "",
                }
            )
    return common_rows


def compact_trace_rows(
    trace_files: list[Path],
    record_set: str,
    source_package: str,
    benchmark: str,
    memory_system: str,
    scope: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rich_fields = [
        "source_package",
        "benchmark",
        "memory_system",
        "scope",
        "model_label",
        "model",
        "s",
        "s_label",
        "dataset_name",
        "task_id",
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
        "source_trace_file",
    ]
    for budget in BUDGETS:
        rich_fields.extend(
            [
                f"pass_at_b{budget}",
                f"budget_exhaustion_b{budget}",
                f"wrong_within_budget_b{budget}",
                f"failure_mode_b{budget}",
            ]
        )

    rich_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    for trace_file in trace_files:
        with trace_file.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                scale = normalize_scale(row["s"])
                calls = int_value(row["retrieval_calls"])
                success = bool_int(row["success"])
                judge_success = bool_int(row["llm_judge"])
                model = row["model"]
                public = {
                    "source_package": source_package,
                    "benchmark": benchmark,
                    "memory_system": memory_system,
                    "scope": scope,
                    "model_label": clean_model_label(model, trace_file.name),
                    "model": model,
                    "s": scale,
                    "s_label": f"s{scale}",
                    "dataset_name": row["dataset_name"],
                    "task_id": row["task_id"],
                    "question_type": row["question_type"],
                    "trial": int_value(row["trial"]),
                    "retrieval_calls": calls,
                    "retrieved_sessions": row.get("retrieved_sessions", ""),
                    "react_steps": row.get("react_steps", ""),
                    "success": success,
                    "f1": row.get("f1", ""),
                    "llm_judge": judge_success,
                    "response_duration_ms": row.get("response_duration_ms", ""),
                    "search_duration_ms": row.get("search_duration_ms", ""),
                    "total_duration_ms": row.get("total_duration_ms", ""),
                    "context_tokens": row.get("context_tokens", ""),
                    "total_cost_usd": row.get("total_cost_usd", ""),
                    "context_ok": row.get("context_ok", ""),
                    "source_trace_file": trace_file.name,
                }
                for budget in BUDGETS:
                    passed = int(success == 1 and calls <= budget)
                    exhausted = int(calls > budget)
                    wrong = int(success == 0 and calls <= budget)
                    if passed:
                        failure_mode = "pass"
                    elif exhausted:
                        failure_mode = "budget_exhausted"
                    else:
                        failure_mode = "wrong_within_budget"
                    public[f"pass_at_b{budget}"] = passed
                    public[f"budget_exhaustion_b{budget}"] = exhausted
                    public[f"wrong_within_budget_b{budget}"] = wrong
                    public[f"failure_mode_b{budget}"] = failure_mode
                rich_rows.append(public)
                common_rows.append(
                    {
                        "record_set": record_set,
                        "source_package": source_package,
                        "benchmark": benchmark,
                        "memory_system": memory_system,
                        "scope": scope,
                        "model_label": public["model_label"],
                        "model": model,
                        "scale": scale,
                        "scale_label": f"s{scale}",
                        "trajectory_id": row["dataset_name"],
                        "task_id": row["task_id"],
                        "question_type": row["question_type"],
                        "trial": int_value(row["trial"]),
                        "retrieval_calls": calls,
                        "correctness_source": "archived_llm_judge",
                        "success": success,
                        "judge_success": judge_success,
                        "context_ok": row.get("context_ok", ""),
                    }
                )
    return rich_rows, common_rows


def copy_openclaw(source_root: Path, artifact_root: Path) -> list[dict[str, object]]:
    source_dir = source_root / "qwen_openclaw_longmemeval"
    trace_files = sorted((source_dir / "01_traces").glob("qwen*b_trace_q1_full.csv"))
    rich_rows, common_rows = compact_trace_rows(
        trace_files,
        record_set="openclaw",
        source_package="qwen_openclaw_longmemeval",
        benchmark="LongMemEval",
        memory_system="OpenClaw",
        scope="additional_interface_coverage",
    )
    write_csv(
        artifact_root / "data/openclaw/trajectory_metrics.csv",
        list(rich_rows[0].keys()),
        rich_rows,
    )

    output_dir = artifact_root / "aggregates/openclaw"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_file in sorted((source_dir / "02_metrics").glob("*.csv")):
        if source_file.name != "cost_reconstructed_aliyun_itemized.csv":
            shutil.copy2(source_file, output_dir / source_file.name)
            continue
        with source_file.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = []
            for row in reader:
                row["result_path"] = Path(row.get("result_path", "")).name
                row["source_root"] = Path(row.get("source_root", "")).name
                rows.append(row)
        write_csv(output_dir / source_file.name, list(reader.fieldnames or []), rows)
    for source_file in sorted((source_dir / "05_figures").glob("*.png")):
        shutil.copy2(source_file, output_dir / source_file.name)

    with (source_dir / "00_summary/overview.json").open(encoding="utf-8") as handle:
        overview = json.load(handle)
    public_overview = {
        "benchmark": overview["benchmark"],
        "expected_items_per_model": overview["expected_items_per_model"],
        "models": [
            {
                "slug": item["slug"],
                "label": item["label"],
                "completed": item["completed"],
                "failed": item["failed"],
                "success_rate": item["success_rate"],
                "retrieval_calls_mean": item["retrieval_calls_mean"],
                "retrieval_calls_max": item["retrieval_calls_max"],
            }
            for item in overview["models"]
        ],
        "memory_system": overview["memory_system"],
        "models_and_services": overview["models_and_services"],
        "trace_authority": overview["trace_authority"],
    }
    with (output_dir / "run_overview.json").open("w", encoding="utf-8") as handle:
        json.dump(public_overview, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return common_rows


def copy_supplementary_gpt(source_root: Path, artifact_root: Path) -> list[dict[str, object]]:
    trace_file = (
        source_root
        / "Licomemory_result/licomemory_locomo_20260410_package/01_traces/gpt_trace_q1_full.csv"
    )
    rich_rows, common_rows = compact_trace_rows(
        [trace_file],
        record_set="supplementary_gpt_locomo",
        source_package="licomemory_locomo_20260410_package",
        benchmark="LoCoMo",
        memory_system="LiCoMemory",
        scope="supplementary_model",
    )
    write_csv(
        artifact_root / "data/hierarchical/gpt5mini_locomo_trajectory_metrics.csv",
        list(rich_rows[0].keys()),
        rich_rows,
    )
    return common_rows


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    artifact_root = args.artifact_root.resolve()

    common_rows: list[dict[str, object]] = []
    common_rows.extend(copy_hierarchical(source_root, artifact_root))
    common_rows.extend(copy_additional_interfaces(source_root, artifact_root))
    common_rows.extend(copy_openclaw(source_root, artifact_root))
    common_rows.extend(copy_supplementary_gpt(source_root, artifact_root))
    common_rows.sort(
        key=lambda row: (
            str(row["record_set"]),
            str(row["benchmark"]),
            str(row["memory_system"]),
            str(row["model_label"]),
            int(row["scale"]),
            str(row["trajectory_id"]),
        )
    )
    write_csv(artifact_root / "data/all_trajectory_outcomes.csv", COMMON_FIELDS, common_rows)
    normalize_published_text(artifact_root)
    print(f"Wrote {len(common_rows):,} public trajectory records to {artifact_root}")


if __name__ == "__main__":
    main()
