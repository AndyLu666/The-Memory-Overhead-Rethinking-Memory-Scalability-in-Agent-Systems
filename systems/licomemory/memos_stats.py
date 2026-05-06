from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


_ENCODING = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None

REACT_EVIDENCE_PATTERN = re.compile(
    r"\[Current Aggregated Evidence\]\s*(.*?)\s*\[Conversation History\]",
    flags=re.S,
)
QUERY_CONTEXT_PATTERN = re.compile(
    r"Context:\s*(.*?)(?:\n-Instructions-|\n################)",
    flags=re.S,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _compact_jsonable(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value).strip()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(value, "__fspath__"):
        return str(value)
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def _build_context_from_record(record: Mapping[str, Any]) -> str:
    parts: list[str] = []

    summaries = record.get("summaries") or []
    triples = record.get("triples") or []
    chunks = record.get("chunks") or []

    if summaries:
        rendered = "\n".join(
            f"{idx}. {_compact_jsonable(item)}"
            for idx, item in enumerate(summaries, 1)
            if _compact_jsonable(item)
        )
        if rendered:
            parts.append("Session Summaries:\n" + rendered)

    if triples:
        rendered = "\n".join(
            f"{idx}. {_compact_jsonable(item)}"
            for idx, item in enumerate(triples, 1)
            if _compact_jsonable(item)
        )
        if rendered:
            parts.append("Triples:\n" + rendered)

    if chunks:
        rendered = "\n".join(
            f"{idx}. {_compact_jsonable(item)}"
            for idx, item in enumerate(chunks, 1)
            if _compact_jsonable(item)
        )
        if rendered:
            parts.append("Text Chunks:\n" + rendered)

    return "\n\n".join(parts).strip()


def extract_search_context(record: Mapping[str, Any]) -> str:
    prompt = str(record.get("formatted_prompt") or "")
    if prompt:
        match = REACT_EVIDENCE_PATTERN.search(prompt)
        if match:
            context = match.group(1).strip()
            if context:
                return context

        match = QUERY_CONTEXT_PATTERN.search(prompt)
        if match:
            context = match.group(1).strip()
            if context:
                return context

    return _build_context_from_record(record)


def count_context_tokens(context: str) -> int:
    if not context:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(context))
        except Exception:
            pass
    return max(1, len(context) // 4)


def derive_memos_metrics(
    *,
    row: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = row or {}
    record = record or {}

    context = str(row.get("search_context") or "").strip()
    if not context:
        context = extract_search_context(record)

    existing_memos = record.get("memos_stats", {}) or {}
    if not existing_memos and isinstance(row, Mapping):
        existing_memos = row.get("memos_stats", {}) or {}

    response_duration_ms = _safe_float(existing_memos.get("response_duration_ms"), -1.0)
    if response_duration_ms < 0:
        response_duration_ms = _safe_float(row.get("response_duration_ms"), -1.0)

    search_duration_ms = _safe_float(existing_memos.get("search_duration_ms"), -1.0)
    if search_duration_ms < 0:
        search_duration_ms = _safe_float(row.get("search_duration_ms"), -1.0)

    total_duration_ms = _safe_float(existing_memos.get("total_duration_ms"), -1.0)
    if total_duration_ms < 0:
        total_duration_ms = _safe_float(row.get("total_duration_ms"), -1.0)
    if total_duration_ms < 0:
        total_duration_ms = response_duration_ms + search_duration_ms

    context_tokens = _safe_int(row.get("context_tokens"), -1)
    if context_tokens < 0:
        context_tokens = _safe_int(existing_memos.get("context_tokens"), -1)
    if context_tokens < 0:
        context_tokens = count_context_tokens(context)

    return {
        "search_context": context,
        "context_tokens": int(context_tokens),
        "response_duration_ms": round(float(response_duration_ms), 3),
        "search_duration_ms": round(float(search_duration_ms), 3),
        "total_duration_ms": round(float(total_duration_ms), 3),
    }


def sanitize_trace_record_for_export(
    record: Mapping[str, Any] | None,
    *,
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = record or {}
    row = row or {}
    sanitized = _to_jsonable(record)
    memos_metrics = derive_memos_metrics(row=row, record=record)

    sanitized["memos_stats"] = {
        "context_tokens": int(memos_metrics["context_tokens"]),
        "response_duration_ms": float(memos_metrics["response_duration_ms"]),
        "search_duration_ms": float(memos_metrics["search_duration_ms"]),
        "total_duration_ms": float(memos_metrics["total_duration_ms"]),
    }
    sanitized["context_tokens"] = int(memos_metrics["context_tokens"])
    sanitized["response_duration_ms"] = float(memos_metrics["response_duration_ms"])
    sanitized["search_duration_ms"] = float(memos_metrics["search_duration_ms"])
    sanitized["total_duration_ms"] = float(memos_metrics["total_duration_ms"])

    total_cost_usd = _safe_float(record.get("total_cost_usd"), -1.0)
    if total_cost_usd < 0:
        total_cost_usd = _safe_float(row.get("total_cost_usd"), -1.0)
    if total_cost_usd >= 0:
        sanitized["total_cost_usd"] = round(float(total_cost_usd), 4)

    final_answer = str(record.get("final_answer") or "").strip()
    if not final_answer:
        react_trace = sanitized.get("react_trace")
        if isinstance(react_trace, list):
            for step in reversed(react_trace):
                if isinstance(step, dict) and str(step.get("action", "")).strip() == "finish":
                    final_answer = str(step.get("final_answer") or "").strip()
                    if final_answer:
                        break
    if not final_answer:
        final_answer = str(record.get("output") or row.get("agent_output") or "").strip()
    if final_answer:
        sanitized["final_answer"] = final_answer

    sanitized.pop("response_duration_sec", None)
    sanitized.pop("response_duration", None)
    for legacy_time_key in ("wallclock" + "_duration_ms", "start_ts", "end_ts"):
        sanitized.pop(legacy_time_key, None)
    for legacy_nested_key in ("query" + "_summary", "cost" + "_summary"):
        sanitized.pop(legacy_nested_key, None)

    react_trace = sanitized.get("react_trace")
    if isinstance(react_trace, list):
        for step in react_trace:
            if isinstance(step, dict):
                step.pop("llm_prompt_tokens_delta", None)
                step.pop("llm_completion_tokens_delta", None)

    return sanitized


def aggregate_memos_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [derive_memos_metrics(record=record or {}) for record in records or []]
    n = len(rows)
    if n == 0:
        return {
            "queries": 0,
            "context_tokens_total": 0,
            "context_tokens_avg": 0.0,
            "response_duration_ms_avg": 0.0,
            "search_duration_ms_avg": 0.0,
            "total_duration_ms_avg": 0.0,
        }

    context_tokens_total = sum(int(row["context_tokens"]) for row in rows)
    response_duration_ms_total = sum(float(row["response_duration_ms"]) for row in rows)
    search_duration_ms_total = sum(float(row["search_duration_ms"]) for row in rows)
    total_duration_ms_total = sum(float(row["total_duration_ms"]) for row in rows)

    return {
        "queries": n,
        "context_tokens_total": int(context_tokens_total),
        "context_tokens_avg": round(context_tokens_total / n, 3),
        "response_duration_ms_avg": round(response_duration_ms_total / n, 3),
        "search_duration_ms_avg": round(search_duration_ms_total / n, 3),
        "total_duration_ms_avg": round(total_duration_ms_total / n, 3),
    }
