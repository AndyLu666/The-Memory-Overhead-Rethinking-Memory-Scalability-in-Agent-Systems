#!/usr/bin/env python3
"""Recompute trajectory diagnostics from the public audit table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BUDGETS = range(1, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rounded_percent(value: float) -> float:
    return round(100.0 * value, 2)


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    input_path = artifact_root / "data/all_trajectory_outcomes.csv"
    output_root = artifact_root / "reproduced"

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    coverage: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    seen_keys: set[tuple[str, ...]] = set()
    duplicates: list[tuple[str, ...]] = []
    total_rows = 0
    independent_judge_rows = 0

    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            key = (
                row["record_set"],
                row["benchmark"],
                row["memory_system"],
                row["model_label"],
                row["scale"],
                row["trajectory_id"],
                row["trial"],
            )
            if key in seen_keys:
                duplicates.append(key)
            seen_keys.add(key)
            if row["judge_success"].strip():
                independent_judge_rows += 1
                if int(row["success"]) != int(row["judge_success"]):
                    raise AssertionError(f"success/judge mismatch: {key}")
            group_key = (
                row["record_set"],
                row["benchmark"],
                row["memory_system"],
                row["scope"],
                row["model_label"],
                row["scale"],
            )
            groups[group_key].append(row)
            coverage[group_key[:5]] += 1

    if duplicates:
        raise AssertionError(f"Found {len(duplicates)} duplicate trajectory keys")
    if total_rows != 116_467:
        raise AssertionError(f"Unexpected trajectory count: {total_rows}")

    metric_rows: list[dict[str, object]] = []
    metric_index: dict[tuple[str, str, str, int, int], dict[str, object]] = {}
    for group_key, rows in sorted(groups.items()):
        record_set, benchmark, memory_system, scope, model_label, scale_text = group_key
        calls = [int(row["retrieval_calls"]) for row in rows]
        successes = [int(row["success"]) for row in rows]
        n = len(rows)
        for budget in BUDGETS:
            pass_count = sum(success and call <= budget for success, call in zip(successes, calls))
            exhausted_count = sum(call > budget for call in calls)
            wrong_count = sum(
                (not success) and call <= budget for success, call in zip(successes, calls)
            )
            if pass_count + exhausted_count + wrong_count != n:
                raise AssertionError(f"Failure partition does not sum to n: {group_key}, B={budget}")
            metric = {
                "record_set": record_set,
                "benchmark": benchmark,
                "memory_system": memory_system,
                "scope": scope,
                "model_label": model_label,
                "scale": int(scale_text),
                "budget": budget,
                "n": n,
                "unbudgeted_success_count": sum(successes),
                "pass_count": pass_count,
                "wrong_within_budget_count": wrong_count,
                "budget_exhausted_count": exhausted_count,
                "unbudgeted_success": sum(successes) / n,
                "pass_at_b": pass_count / n,
                "p_wrong": wrong_count / n,
                "p_exh": exhausted_count / n,
                "mean_calls": sum(calls) / n,
                "median_calls": quantile(calls, 0.5),
                "p90_calls": quantile(calls, 0.9),
            }
            metric_rows.append(metric)
            metric_index[(benchmark, memory_system, model_label, int(scale_text), budget)] = metric

    metric_fields = [
        "record_set",
        "benchmark",
        "memory_system",
        "scope",
        "model_label",
        "scale",
        "budget",
        "n",
        "unbudgeted_success_count",
        "pass_count",
        "wrong_within_budget_count",
        "budget_exhausted_count",
        "unbudgeted_success",
        "pass_at_b",
        "p_wrong",
        "p_exh",
        "mean_calls",
        "median_calls",
        "p90_calls",
    ]
    write_csv(output_root / "metrics_by_cell_and_budget.csv", metric_fields, metric_rows)

    coverage_rows = [
        {
            "record_set": key[0],
            "benchmark": key[1],
            "memory_system": key[2],
            "scope": key[3],
            "model_label": key[4],
            "rows": count,
        }
        for key, count in sorted(coverage.items())
    ]
    write_csv(
        output_root / "coverage.csv",
        ["record_set", "benchmark", "memory_system", "scope", "model_label", "rows"],
        coverage_rows,
    )

    longmemeval_systems = {
        key[2] for key in coverage if key[1] == "LongMemEval"
    }
    locomo_systems = {key[2] for key in coverage if key[1] == "LoCoMo"}
    expected_longmemeval_systems = {
        "HippoRAG",
        "LiCoMemory",
        "MemOS-text",
        "Mem0",
        "MemOS-Tree",
        "OpenClaw",
    }
    expected_locomo_systems = {"HippoRAG", "LiCoMemory"}
    if longmemeval_systems != expected_longmemeval_systems:
        raise AssertionError(
            f"Unexpected LongMemEval coverage: {sorted(longmemeval_systems)}"
        )
    if locomo_systems != expected_locomo_systems:
        raise AssertionError(f"Unexpected LoCoMo coverage: {sorted(locomo_systems)}")

    def get(
        benchmark: str, system: str, model: str, scale: int, budget: int
    ) -> dict[str, object]:
        return metric_index[(benchmark, system, model, scale, budget)]

    hippo_s4 = get("LongMemEval", "HippoRAG", "Qwen 8B", 400, 2)
    lico_s4_b2 = get("LongMemEval", "LiCoMemory", "Qwen 8B", 400, 2)
    lico_s4_b5 = get("LongMemEval", "LiCoMemory", "Qwen 8B", 400, 5)
    locomo_s0 = get("LoCoMo", "LiCoMemory", "Qwen 8B", 0, 2)
    locomo_s4 = get("LoCoMo", "LiCoMemory", "Qwen 8B", 400, 2)

    claim_checks = {
        "total_trajectory_rows": total_rows,
        "rows_with_separately_retained_judge_field": independent_judge_rows,
        "decision_reversal_longmemeval_qwen8b_s400": {
            "hipporag_unbudgeted_success_pct": rounded_percent(
                float(hippo_s4["unbudgeted_success"])
            ),
            "licomemory_unbudgeted_success_pct": rounded_percent(
                float(lico_s4_b2["unbudgeted_success"])
            ),
            "hipporag_pass_at_2_pct": rounded_percent(float(hippo_s4["pass_at_b"])),
            "licomemory_pass_at_2_pct": rounded_percent(float(lico_s4_b2["pass_at_b"])),
            "licomemory_pass_at_5_pct": rounded_percent(float(lico_s4_b5["pass_at_b"])),
        },
        "locomo_licomemory_qwen8b": {
            "s0": {
                "n": locomo_s0["n"],
                "unbudgeted_success_count": locomo_s0["unbudgeted_success_count"],
                "pass_at_2_count": locomo_s0["pass_count"],
                "wrong_within_budget_count": locomo_s0["wrong_within_budget_count"],
                "budget_exhausted_count": locomo_s0["budget_exhausted_count"],
                "unbudgeted_success_pct": rounded_percent(
                    float(locomo_s0["unbudgeted_success"])
                ),
                "pass_at_2_pct": rounded_percent(float(locomo_s0["pass_at_b"])),
                "p_wrong_pct": rounded_percent(float(locomo_s0["p_wrong"])),
                "p_exh_pct": rounded_percent(float(locomo_s0["p_exh"])),
                "p90_calls": locomo_s0["p90_calls"],
            },
            "s400": {
                "n": locomo_s4["n"],
                "unbudgeted_success_count": locomo_s4["unbudgeted_success_count"],
                "pass_at_2_count": locomo_s4["pass_count"],
                "wrong_within_budget_count": locomo_s4["wrong_within_budget_count"],
                "budget_exhausted_count": locomo_s4["budget_exhausted_count"],
                "unbudgeted_success_pct": rounded_percent(
                    float(locomo_s4["unbudgeted_success"])
                ),
                "pass_at_2_pct": rounded_percent(float(locomo_s4["pass_at_b"])),
                "p_wrong_pct": rounded_percent(float(locomo_s4["p_wrong"])),
                "p_exh_pct": rounded_percent(float(locomo_s4["p_exh"])),
                "p90_calls": locomo_s4["p90_calls"],
            },
        },
        "additional_interface_examples_s400_b2": {
            "mem0_qwen8b_p_wrong_pct": rounded_percent(
                float(get("LongMemEval", "Mem0", "Qwen 8B", 400, 2)["p_wrong"])
            ),
            "mem0_qwen32b_p_exh_pct": rounded_percent(
                float(get("LongMemEval", "Mem0", "Qwen 32B", 400, 2)["p_exh"])
            ),
        },
        "additional_interface_recovery_s400_b2_to_b6_pct_points": {
            system: {
                model: round(
                    100.0
                    * (
                        float(get("LongMemEval", system, model, 400, 6)["pass_at_b"])
                        - float(get("LongMemEval", system, model, 400, 2)["pass_at_b"])
                    ),
                    2,
                )
                for model in ("Qwen 8B", "Qwen 32B", "Qwen 235B")
            }
            for system in ("MemOS-text", "Mem0", "MemOS-Tree")
        },
        "benchmark_system_coverage": {
            "LongMemEval": sorted(longmemeval_systems),
            "LoCoMo": sorted(locomo_systems),
        },
    }

    expected = {
        "hipporag_unbudgeted_success_pct": 58.15,
        "licomemory_unbudgeted_success_pct": 63.10,
        "hipporag_pass_at_2_pct": 58.15,
        "licomemory_pass_at_2_pct": 52.20,
        "licomemory_pass_at_5_pct": 63.10,
    }
    observed = claim_checks["decision_reversal_longmemeval_qwen8b_s400"]
    for name, expected_value in expected.items():
        if abs(float(observed[name]) - expected_value) > 0.01:
            raise AssertionError(f"Headline check failed for {name}: {observed[name]}")

    locomo_expected = {
        "s0": {
            "unbudgeted_success_pct": 41.49,
            "pass_at_2_pct": 7.09,
            "p_exh_pct": 91.84,
            "p90_calls": 5.0,
        },
        "s400": {
            "unbudgeted_success_pct": 41.49,
            "pass_at_2_pct": 5.67,
            "p_exh_pct": 93.97,
            "p90_calls": 5.0,
        },
    }
    locomo_observed = claim_checks["locomo_licomemory_qwen8b"]
    for scale, expected_values in locomo_expected.items():
        for name, expected_value in expected_values.items():
            if abs(float(locomo_observed[scale][name]) - expected_value) > 0.01:
                raise AssertionError(
                    f"LoCoMo check failed for {scale}.{name}: "
                    f"{locomo_observed[scale][name]}"
                )

    interface_expected = {
        "mem0_qwen8b_p_wrong_pct": 85.60,
        "mem0_qwen32b_p_exh_pct": 83.60,
    }
    interface_observed = claim_checks["additional_interface_examples_s400_b2"]
    for name, expected_value in interface_expected.items():
        if abs(float(interface_observed[name]) - expected_value) > 0.01:
            raise AssertionError(
                f"Additional-interface check failed for {name}: {interface_observed[name]}"
            )

    with (output_root / "claim_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(claim_checks, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Verified {total_rows:,} trajectories with no duplicate keys.")
    print("Reproduced the decision reversal, LoCoMo diagnosis, and interface checks.")


if __name__ == "__main__":
    main()
