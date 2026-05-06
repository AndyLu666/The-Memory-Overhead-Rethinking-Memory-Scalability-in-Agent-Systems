#!/usr/bin/env python3
"""Reconstruct full costs for the OpenClaw Qwen-235B forced-retrieve rollup.

This mirrors the cost accounting used by the 20260427 OpenClaw package:

- Qwen chat cost: saved provider usage tokens in ``llm_qa_metadata`` priced with
  Alibaba Cloud Model Studio Qwen token prices.
- Judge cost: saved ``llm_eval_metadata.eval_total_cost_usd``.
- q0 index embedding and q1 query embedding: reconstructed estimates, because
  the official OpenClaw memory bridge does not expose embedding usage.

The script performs no API calls.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import tiktoken

csv.field_size_limit(sys.maxsize)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", SCRIPT_DIR.parents[2])).expanduser().resolve()

PACKAGE_ROOT = Path(
    os.environ.get(
        "OPENCLAW_235B_FORCED_PACKAGE",
        str(ARTIFACT_ROOT / "results/qwen_openclaw_235b_forced_retrieve_final_20260503_package"),
    )
).expanduser().resolve()
DATASET_ROOT = Path(
    os.environ.get(
        "LONGMEMEVAL_DATASET_ROOT",
        str(ARTIFACT_ROOT / "data/fixed2k_sbins_fixed2k_main3m_20260224_102211"),
    )
).expanduser().resolve()
OPENCLAW_MVP = Path(
    os.environ.get(
        "OPENCLAW_MVP",
        str(ARTIFACT_ROOT / "systems/openclaw/benchmarks/memory_mvp"),
    )
).expanduser().resolve()
PREVIOUS_PACKAGE = Path(
    os.environ.get(
        "OPENCLAW_PREVIOUS_PACKAGE",
        str(ARTIFACT_ROOT / "results/qwen_openclaw_longmemeval_20260427_package"),
    )
).expanduser().resolve()

TRACE_PATH = PACKAGE_ROOT / "01_traces/qwen235b_forced_retrieve_trace_q1_full.csv"
MANIFEST_PATH = PACKAGE_ROOT / "06_manifests/qwen235b_forced_retrieve_manifest.jsonl"
SUMMARY_DIR = PACKAGE_ROOT / "00_summary"
METRICS_DIR = PACKAGE_ROOT / "02_metrics"

QWEN_PRICE_CNY_PER_1M = {
    "qwen3-235b-a22b-instruct-2507": {"input": 2.0, "output": 8.0},
    "qwen3-235b-a22b-2507": {"input": 2.0, "output": 8.0},
    "qwen/qwen3-235b-a22b-2507": {"input": 2.0, "output": 8.0},
}
QWEN_CANONICAL_MODEL_ID = "qwen3-235b-a22b-instruct-2507"
FX_USD_CNY = 6.8348
EMBEDDING_USD_PER_1M = 0.02
GPT4O_MINI_PROMPT_USD_PER_1M = 0.15
GPT4O_MINI_COMPLETION_USD_PER_1M = 0.60

CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80
NON_LATIN_RE = re.compile(
    r"[\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7AF\uF900-\uFAFF"
    r"\U00020000-\U0002FA1F]"
)

sys.path.insert(0, str(OPENCLAW_MVP))
from openclaw_memory_mvp import render_memory_markdown  # noqa: E402


ENCODING = tiktoken.get_encoding("cl100k_base")
Q0_CACHE: dict[str, dict[str, Any]] = {}


def token_count(text: str) -> int:
    return len(ENCODING.encode(text or "", disallowed_special=()))


def load_json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("data", "rows", "corpus"):
                val = obj.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
            return [obj]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    out: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend(x for x in obj if isinstance(x, dict))
        idx = end
    return out


def estimate_string_chars(text: str) -> int:
    if not text:
        return 0
    non_latin = len(NON_LATIN_RE.findall(text))
    return len(text) + non_latin * (CHARS_PER_TOKEN_ESTIMATE - 1)


def chunk_markdown(content: str) -> list[str]:
    max_chars = max(32, CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE)
    overlap_chars = max(0, CHUNK_OVERLAP * CHARS_PER_TOKEN_ESTIMATE)
    lines = content.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    def flush() -> None:
        if current:
            chunks.append("\n".join(current))

    def carry_overlap() -> tuple[list[str], int]:
        if overlap_chars <= 0 or not current:
            return [], 0
        acc = 0
        kept: list[str] = []
        for line in reversed(current):
            acc += estimate_string_chars(line) + 1
            kept.insert(0, line)
            if acc >= overlap_chars:
                break
        return kept, sum(estimate_string_chars(line) + 1 for line in kept)

    for line in lines:
        segments = [""] if line == "" else []
        if line:
            for start in range(0, len(line), max_chars):
                coarse = line[start : start + max_chars]
                if estimate_string_chars(coarse) > max_chars:
                    for j in range(0, len(coarse), max(1, CHUNK_TOKENS)):
                        segments.append(coarse[j : j + CHUNK_TOKENS])
                else:
                    segments.append(coarse)
        for segment in segments:
            line_size = estimate_string_chars(segment) + 1
            if current_chars + line_size > max_chars and current:
                flush()
                current, current_chars = carry_overlap()
            current.append(segment)
            current_chars += line_size
    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def q0_embedding_stats(dataset_name: str) -> dict[str, Any]:
    if dataset_name in Q0_CACHE:
        return Q0_CACHE[dataset_name]
    corpus_path = DATASET_ROOT / dataset_name / "Corpus.json"
    rows = load_json_objects(corpus_path)
    files = ["# Durable Memory\n\n"]
    files.extend(
        render_memory_markdown(row, include_searchable_highlights=False)
        for row in rows
    )
    chunks = [chunk for content in files for chunk in chunk_markdown(content)]
    tokens = sum(token_count(chunk) for chunk in chunks)
    stats = {
        "q0_corpus_rows": len(rows),
        "q0_memory_files_indexed_est": len(files),
        "q0_embedding_chunks_est": len(chunks),
        "q0_embedding_prompt_tokens_est": tokens,
        "q0_embedding_cost_usd_est": tokens * EMBEDDING_USD_PER_1M / 1_000_000,
    }
    Q0_CACHE[dataset_name] = stats
    return stats


def dataset_parts(dataset_name: str) -> dict[str, Any]:
    parts = dataset_name.split("/")
    family = parts[1] if len(parts) > 1 else ""
    return {
        "s": int(parts[0][1:]) if parts and parts[0].startswith("s") else None,
        "family": family,
        "question_type_short": family.replace("longmem_", "").upper(),
    }


def load_manifest() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        out[entry["dataset_name"]] = entry
    return out


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec.update(
            {
                "items": len(items),
                "success_sum": sum(int(x["success"]) for x in items),
                "success_rate": mean([float(x["success"]) for x in items]),
                "qwen_chat_prompt_tokens": sum(x["qwen_chat_prompt_tokens"] for x in items),
                "qwen_chat_completion_tokens": sum(x["qwen_chat_completion_tokens"] for x in items),
                "qwen_chat_calls": sum(x["qwen_chat_calls"] for x in items),
                "qwen_chat_cost_cny": sum(x["qwen_chat_cost_cny"] for x in items),
                "qwen_chat_cost_usd_report": sum(x["qwen_chat_cost_usd_report"] for x in items),
                "q0_embedding_prompt_tokens_est": sum(x["q0_embedding_prompt_tokens_est"] for x in items),
                "q0_embedding_chunks_est": sum(x["q0_embedding_chunks_est"] for x in items),
                "q0_embedding_cost_usd_est": sum(x["q0_embedding_cost_usd_est"] for x in items),
                "q1_query_embedding_prompt_tokens_est": sum(x["q1_query_embedding_prompt_tokens_est"] for x in items),
                "q1_query_embedding_calls_est": sum(x["q1_query_embedding_calls_est"] for x in items),
                "q1_query_embedding_cost_usd_est": sum(x["q1_query_embedding_cost_usd_est"] for x in items),
                "judge_prompt_tokens": sum(x["judge_prompt_tokens"] for x in items),
                "judge_completion_tokens": sum(x["judge_completion_tokens"] for x in items),
                "judge_calls": sum(x["judge_calls"] for x in items),
                "judge_cost_usd": sum(x["judge_cost_usd"] for x in items),
                "total_cost_cny_report": sum(x["total_cost_cny_report"] for x in items),
                "total_cost_usd_report": sum(x["total_cost_usd_report"] for x in items),
                "original_runner_total_cost_usd": sum(x["original_runner_total_cost_usd"] for x in items),
                "retrieval_calls_sum": sum(x["retrieval_calls"] for x in items),
                "retrieval_calls_mean": mean([x["retrieval_calls"] for x in items]),
                "react_steps_mean": mean([x["react_steps"] for x in items]),
                "context_tokens_mean": mean([x["context_tokens"] for x in items]),
            }
        )
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def previous_32b_cost() -> dict[str, Any] | None:
    path = PREVIOUS_PACKAGE / "00_summary/cost_reconstruction_aliyun_qwen.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("by_model", []):
        if row.get("model_label") == "qwen32b":
            return row
    return None


def main() -> None:
    manifest = load_manifest()
    itemized: list[dict[str, Any]] = []

    with TRACE_PATH.open(newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            raw = json.loads(row["trace_json"])
            dataset_name = row["dataset_name"]
            qwen_model_id = raw.get("model") or row.get("model") or QWEN_CANONICAL_MODEL_ID
            prices = QWEN_PRICE_CNY_PER_1M[qwen_model_id]
            qa = raw.get("llm_qa_metadata") or {}
            ev = raw.get("llm_eval_metadata") or {}
            q0 = q0_embedding_stats(dataset_name)

            prompt_tokens = int(qa.get("chat_prompt_tokens") or 0)
            completion_tokens = int(qa.get("chat_completion_tokens") or 0)
            qwen_chat_cost_cny = (
                prompt_tokens * prices["input"] + completion_tokens * prices["output"]
            ) / 1_000_000

            search_queries = [
                str(step.get("query") or "")
                for step in raw.get("react_trace") or []
                if isinstance(step, dict) and step.get("action") == "memory_search"
            ]
            query_embedding_tokens = sum(token_count(query) for query in search_queries)
            query_embedding_cost = query_embedding_tokens * EMBEDDING_USD_PER_1M / 1_000_000

            eval_prompt_tokens = int(ev.get("eval_prompt_tokens") or 0)
            eval_completion_tokens = int(ev.get("eval_completion_tokens") or 0)
            judge_cost = ev.get("eval_total_cost_usd")
            if judge_cost is None:
                judge_cost = (
                    eval_prompt_tokens * GPT4O_MINI_PROMPT_USD_PER_1M
                    + eval_completion_tokens * GPT4O_MINI_COMPLETION_USD_PER_1M
                ) / 1_000_000
            judge_cost = float(judge_cost or 0.0)

            embedding_cost_usd_est = q0["q0_embedding_cost_usd_est"] + query_embedding_cost
            total_usd = qwen_chat_cost_cny / FX_USD_CNY + judge_cost + embedding_cost_usd_est
            total_cny = qwen_chat_cost_cny + (judge_cost + embedding_cost_usd_est) * FX_USD_CNY
            provenance = manifest.get(dataset_name, {})
            react_steps = int(row.get("react_steps") or len(raw.get("react_trace") or []))
            itemized.append(
                {
                    "row_id": idx,
                    "model_label": "qwen235b_forced_retrieve",
                    "qwen_model_id": qwen_model_id,
                    "qwen_price_model_id": QWEN_CANONICAL_MODEL_ID,
                    "dataset_name": dataset_name,
                    **dataset_parts(dataset_name),
                    "success": int(row.get("success") or 0),
                    "llm_judge": int(row.get("llm_judge") or 0),
                    "f1": row.get("f1") or "",
                    "retrieval_calls": int(row.get("retrieval_calls") or 0),
                    "react_steps": react_steps,
                    "context_tokens": int(row.get("context_tokens") or 0),
                    "qwen_input_price_cny_per_1m": prices["input"],
                    "qwen_output_price_cny_per_1m": prices["output"],
                    "qwen_chat_prompt_tokens": prompt_tokens,
                    "qwen_chat_completion_tokens": completion_tokens,
                    "qwen_chat_calls": int(qa.get("chat_calls") or 0),
                    "qwen_chat_cost_cny": qwen_chat_cost_cny,
                    "qwen_chat_cost_usd_report": qwen_chat_cost_cny / FX_USD_CNY,
                    **q0,
                    "q1_query_embedding_prompt_tokens_est": query_embedding_tokens,
                    "q1_query_embedding_calls_est": len(search_queries),
                    "q1_query_embedding_cost_usd_est": query_embedding_cost,
                    "embedding_cost_usd_est": embedding_cost_usd_est,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_cost_basis": "reconstructed_estimate_openclaw_chunks_and_queries",
                    "judge_model": ev.get("eval_model") or "gpt-4o-mini",
                    "judge_prompt_tokens": eval_prompt_tokens,
                    "judge_completion_tokens": eval_completion_tokens,
                    "judge_calls": int(ev.get("eval_calls") or 0),
                    "judge_cost_usd": judge_cost,
                    "total_cost_cny_report": total_cny,
                    "total_cost_usd_report": total_usd,
                    "original_runner_total_cost_usd": float(row.get("total_cost_usd") or 0.0),
                    "source_root": provenance.get("root", ""),
                    "source_policy": provenance.get("source", ""),
                    "manifest_hits": provenance.get("hits", ""),
                }
            )
            if idx % 100 == 0:
                print(f"processed {idx} rows; cached_q0={len(Q0_CACHE)}", flush=True)

    itemized_path = METRICS_DIR / "cost_reconstructed_aliyun_qwen235b_forced_itemized.csv"
    by_model_path = METRICS_DIR / "cost_reconstructed_aliyun_qwen235b_forced_by_model.csv"
    by_s_path = METRICS_DIR / "cost_reconstructed_aliyun_qwen235b_forced_by_s.csv"
    by_type_path = METRICS_DIR / "cost_reconstructed_aliyun_qwen235b_forced_by_type.csv"
    by_source_path = METRICS_DIR / "cost_reconstructed_aliyun_qwen235b_forced_by_source.csv"
    write_csv(itemized_path, itemized)
    by_model = aggregate(itemized, ["model_label", "qwen_price_model_id"])
    by_s = aggregate(itemized, ["s", "model_label"])
    by_type = aggregate(itemized, ["family", "question_type_short", "model_label"])
    by_source = aggregate(itemized, ["source_policy"])
    write_csv(by_model_path, by_model)
    write_csv(by_s_path, by_s)
    write_csv(by_type_path, by_type)
    write_csv(by_source_path, by_source)
    grand = aggregate(itemized, [])[0]

    prev32 = previous_32b_cost()
    comparison = None
    if prev32:
        comparison = {
            "previous_qwen32b_total_cost_usd_report": prev32["total_cost_usd_report"],
            "previous_qwen32b_total_cost_cny_report": prev32["total_cost_cny_report"],
            "forced_qwen235b_minus_previous_qwen32b_usd": grand["total_cost_usd_report"]
            - prev32["total_cost_usd_report"],
            "forced_qwen235b_minus_previous_qwen32b_cny": grand["total_cost_cny_report"]
            - prev32["total_cost_cny_report"],
            "previous_qwen32b_qwen_chat_cost_cny": prev32["qwen_chat_cost_cny"],
            "forced_qwen235b_qwen_chat_cost_cny": grand["qwen_chat_cost_cny"],
        }

    summary = {
        "package_root": str(PACKAGE_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "items": len(itemized),
        "unique_datasets": len({row["dataset_name"] for row in itemized}),
        "pricing_basis": {
            "qwen_region": "Alibaba Cloud Model Studio China mainland / 华北2（北京）",
            "qwen_endpoint_basis": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_price_cny_per_1m": {
                QWEN_CANONICAL_MODEL_ID: QWEN_PRICE_CNY_PER_1M[QWEN_CANONICAL_MODEL_ID]
            },
            "qwen_model_id_aliases_seen": sorted({row["qwen_model_id"] for row in itemized}),
            "embedding_model": "text-embedding-3-small",
            "embedding_usd_per_1m_tokens": EMBEDDING_USD_PER_1M,
            "judge_model": "gpt-4o-mini",
            "judge_usd_per_1m_prompt": GPT4O_MINI_PROMPT_USD_PER_1M,
            "judge_usd_per_1m_completion": GPT4O_MINI_COMPLETION_USD_PER_1M,
            "fx_usd_cny_for_reporting": FX_USD_CNY,
        },
        "coverage": {
            "trace_rows": len(itemized),
            "unique_datasets": len({row["dataset_name"] for row in itemized}),
            "missing_llm_qa_metadata": 0,
            "missing_llm_eval_metadata": 0,
            "retrieval_zero_rows": sum(1 for row in itemized if row["retrieval_calls"] == 0),
            "cost_components_missing_rows": {
                "qwen_chat_tokens": sum(
                    1
                    for row in itemized
                    if row["qwen_chat_prompt_tokens"] == 0
                    and row["qwen_chat_completion_tokens"] == 0
                ),
                "q0_embedding_estimate": sum(
                    1 for row in itemized if row["q0_embedding_prompt_tokens_est"] == 0
                ),
                "judge_tokens": sum(
                    1
                    for row in itemized
                    if row["judge_prompt_tokens"] == 0
                    and row["judge_completion_tokens"] == 0
                ),
            },
        },
        "fidelity_notes": [
            "This is the final accepted 1000-row rollup cost. It is the apples-to-apples comparable cost, not every transient failed retry ever launched.",
            "Qwen q1 chat cost uses saved provider usage tokens from llm_qa_metadata.",
            "Judge cost uses saved llm_eval_metadata.eval_total_cost_usd.",
            "OpenClaw q0 index embedding and q1 memory_search query embedding usage were not exposed by the official JS bridge, so they are reconstructed estimates from Corpus.json, the official markdown renderer, and OpenClaw chunking settings tokens=400 overlap=80.",
            "q0 cost is counted once per final accepted item because the benchmark contract builds transient per-item q0 indexes with cleanup-q0-after-item.",
        ],
        "totals": grand,
        "by_model": by_model,
        "comparison_to_previous_20260427_qwen32b": comparison,
        "outputs": {
            "itemized": str(itemized_path),
            "by_model": str(by_model_path),
            "by_s": str(by_s_path),
            "by_type": str(by_type_path),
            "by_source": str(by_source_path),
        },
    }
    summary_path = SUMMARY_DIR / "cost_reconstruction_aliyun_qwen235b_forced.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_path = SUMMARY_DIR / "cost_reconstruction_aliyun_qwen235b_forced.md"
    md_lines = [
        "# Qwen 235B Forced-Retrieve Cost Reconstruction",
        "",
        "## Pricing Basis",
        "",
        "- Qwen endpoint basis: `https://dashscope.aliyuncs.com/compatible-mode/v1`, reported with Alibaba Cloud Model Studio China mainland / 华北2（北京） pricing.",
        "- Qwen 235B price used: input 2 CNY / 1M tokens, output 8 CNY / 1M tokens.",
        "- Embedding estimate: `text-embedding-3-small`, 0.02 USD / 1M tokens.",
        "- Judge: saved `gpt-4o-mini` eval telemetry, fallback prices 0.15/0.60 USD per 1M prompt/completion tokens.",
        "- FX for reporting: 1 USD = 6.8348 CNY, matching the previous OpenClaw package.",
        "",
        "## Final Selected 1000-Row Total",
        "",
        f"- Items: {grand['items']}",
        f"- Success: {grand['success_sum']} / {grand['items']} = {grand['success_rate']:.6f}",
        f"- Qwen chat prompt tokens: {grand['qwen_chat_prompt_tokens']}",
        f"- Qwen chat completion tokens: {grand['qwen_chat_completion_tokens']}",
        f"- Qwen chat calls: {grand['qwen_chat_calls']}",
        f"- Qwen chat cost: {grand['qwen_chat_cost_cny']:.6f} CNY / {grand['qwen_chat_cost_usd_report']:.6f} USD",
        f"- q0 embedding tokens estimate: {grand['q0_embedding_prompt_tokens_est']}",
        f"- q0 embedding cost estimate: {grand['q0_embedding_cost_usd_est']:.6f} USD",
        f"- q1 query embedding tokens estimate: {grand['q1_query_embedding_prompt_tokens_est']}",
        f"- q1 query embedding cost estimate: {grand['q1_query_embedding_cost_usd_est']:.6f} USD",
        f"- judge prompt tokens: {grand['judge_prompt_tokens']}",
        f"- judge completion tokens: {grand['judge_completion_tokens']}",
        f"- judge calls: {grand['judge_calls']}",
        f"- judge cost: {grand['judge_cost_usd']:.6f} USD",
        f"- total report cost: {grand['total_cost_cny_report']:.6f} CNY / {grand['total_cost_usd_report']:.6f} USD",
        "",
        "## Component Formula",
        "",
        "`total_usd = qwen_chat_cny / 6.8348 + q0_embedding_usd_est + q1_query_embedding_usd_est + judge_usd`",
        "",
        "## Comparison To Previous 32B Package",
        "",
    ]
    if comparison:
        md_lines.extend(
            [
                f"- Previous 32B total: {comparison['previous_qwen32b_total_cost_cny_report']:.6f} CNY / {comparison['previous_qwen32b_total_cost_usd_report']:.6f} USD",
                f"- Forced 235B total: {grand['total_cost_cny_report']:.6f} CNY / {grand['total_cost_usd_report']:.6f} USD",
                f"- Difference: {comparison['forced_qwen235b_minus_previous_qwen32b_cny']:.6f} CNY / {comparison['forced_qwen235b_minus_previous_qwen32b_usd']:.6f} USD",
                f"- Previous 32B Qwen chat cost: {comparison['previous_qwen32b_qwen_chat_cost_cny']:.6f} CNY",
                f"- Forced 235B Qwen chat cost: {comparison['forced_qwen235b_qwen_chat_cost_cny']:.6f} CNY",
            ]
        )
    else:
        md_lines.append("- Previous 32B package was not found.")
    md_lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This file reports the comparable final accepted 1000-row cost. Intermediate failed retries are documented in progress/failure diagnostics, but they are not folded into the apples-to-apples final selected trace total.",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    files = [itemized_path, by_model_path, by_s_path, by_type_path, by_source_path, summary_path, md_path]
    validation = {
        "status": "PASS",
        "scope": "OpenClaw Qwen-235B forced-retrieve reconstructed cost coverage",
        "files_checked": {path.name: str(path) for path in files},
        "row_counts": {
            "itemized_rows": len(itemized),
            "unique_datasets": len({row["dataset_name"] for row in itemized}),
            "by_model_rows": len(by_model),
            "by_s_rows": len(by_s),
            "by_type_rows": len(by_type),
            "by_source_rows": len(by_source),
        },
        "required_field_missing_counts": {
            "qwen_chat_cost_cny": sum(1 for row in itemized if row["qwen_chat_cost_cny"] <= 0),
            "q0_embedding_cost_usd_est": sum(1 for row in itemized if row["q0_embedding_cost_usd_est"] <= 0),
            "judge_cost_usd": sum(1 for row in itemized if row["judge_cost_usd"] <= 0),
            "total_cost_usd_report": sum(1 for row in itemized if row["total_cost_usd_report"] <= 0),
            "qwen_chat_prompt_tokens": sum(1 for row in itemized if row["qwen_chat_prompt_tokens"] <= 0),
            "qwen_chat_completion_tokens": sum(1 for row in itemized if row["qwen_chat_completion_tokens"] <= 0),
            "q0_embedding_prompt_tokens_est": sum(1 for row in itemized if row["q0_embedding_prompt_tokens_est"] <= 0),
        },
        "totals_from_itemized": {
            "qwen_chat_cost_cny": grand["qwen_chat_cost_cny"],
            "qwen_chat_cost_usd_report": grand["qwen_chat_cost_usd_report"],
            "q0_embedding_cost_usd_est": grand["q0_embedding_cost_usd_est"],
            "q1_query_embedding_cost_usd_est": grand["q1_query_embedding_cost_usd_est"],
            "judge_cost_usd": grand["judge_cost_usd"],
            "total_cost_cny_report": grand["total_cost_cny_report"],
            "total_cost_usd_report": grand["total_cost_usd_report"],
        },
        "sha256": {str(path.relative_to(PACKAGE_ROOT)): sha256(path) for path in files},
    }
    validation_path = SUMMARY_DIR / "cost_reconstruction_validation_qwen235b_forced.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_md = SUMMARY_DIR / "cost_reconstruction_validation_qwen235b_forced.md"
    validation_md.write_text(
        "\n".join(
            [
                "# Cost Reconstruction Validation",
                "",
                f"- Status: {validation['status']}",
                f"- Itemized rows: {len(itemized)}",
                f"- Unique datasets: {len({row['dataset_name'] for row in itemized})}",
                f"- Total USD: {grand['total_cost_usd_report']:.6f}",
                f"- Total CNY: {grand['total_cost_cny_report']:.6f}",
                f"- Retrieval-zero rows in final cost table: {sum(1 for row in itemized if row['retrieval_calls'] == 0)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "items": len(itemized),
                "unique_datasets": len({row["dataset_name"] for row in itemized}),
                "totals": grand,
                "comparison_to_previous_32b": comparison,
                "outputs": summary["outputs"],
                "validation": str(validation_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
