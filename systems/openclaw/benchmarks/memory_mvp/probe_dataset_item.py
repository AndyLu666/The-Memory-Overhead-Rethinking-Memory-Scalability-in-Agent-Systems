#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from openai import AsyncOpenAI

from official_openclaw_memory import OfficialOpenClawMemoryBackend
from openclaw_memory_mvp import (
    OpenClawMemoryMVP,
    SearchResult,
    SOURCE_MEMORY,
    count_tokens,
    dedupe_preserve_order,
)


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

TOKEN_COSTS = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
}

BENCHMARK_TUNED_PROFILE = "benchmark_tuned"
OPENCLAW_FIDELITY_PROFILE = "openclaw_fidelity"

MEMORY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Mandatory recall step: semantically search MEMORY.md + memory/*.md "
            "(and optional session transcripts) before answering questions about prior work, "
            "decisions, dates, people, preferences, or todos; returns top snippets with path + lines. "
            "If response has disabled=true, memory retrieval is unavailable and should be surfaced to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "maxResults": {"type": "number"},
                "minScore": {"type": "number"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

MEMORY_GET_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_get",
        "description": (
            "Safe snippet read from MEMORY.md or memory/*.md with optional from/lines; "
            "use after memory_search to pull only the needed lines and keep context small."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "from": {"type": "number"},
                "lines": {"type": "number"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OpenClaw document memory on one benchmark item."
    )
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--question-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--chat-model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-max-tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--top-session-limit", type=int, default=20)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--sources", default="memory")
    parser.add_argument("--memory-backend", default="official")
    parser.add_argument("--agent-mode", default="memory_tools")
    parser.add_argument("--memory-agent-profile", default=BENCHMARK_TUNED_PROFILE)
    parser.add_argument("--max-agent-steps", type=int, default=6)
    parser.add_argument(
        "--force-min-memory-searches",
        type=int,
        default=0,
        help=(
            "Force at least this many memory_search tool calls before allowing "
            "the agent to finish. Default 0 preserves the selected profile's "
            "native tool-choice behavior."
        ),
    )
    parser.add_argument("--chat-base-url", default="")
    parser.add_argument("--chat-base-url-env", default="")
    parser.add_argument("--chat-api-key-env", default="")
    parser.add_argument("--chat-extra-body-json", default="")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-base-url-env", default="")
    parser.add_argument("--embedding-api-key-env", default="")
    parser.add_argument(
        "--openclaw-repo-root",
        default="",
        help=(
            "OpenClaw repository root. Defaults to the repo containing this "
            "probe script, so packaged bundles can run from a relocated artifact."
        ),
    )
    parser.add_argument("--openclaw-node-bin", default="")
    parser.add_argument("--excerpt-before-lines", type=int, default=2)
    parser.add_argument("--excerpt-after-lines", type=int, default=10)
    parser.add_argument("--enable-mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--enable-temporal-decay", action="store_true")
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--reuse-existing-q0", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_openai_env_aliases() -> None:
    if "GPT_BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_BASE_URL"]
    if "OPENAI_BASE_URL" in os.environ and "GPT_BASE_URL" not in os.environ:
        os.environ["GPT_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
    if "GPT_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_API_KEY"]
    if "OPENAI_API_KEY" in os.environ and "GPT_API_KEY" not in os.environ:
        os.environ["GPT_API_KEY"] = os.environ["OPENAI_API_KEY"]


def maybe_load_env_file(path: Path) -> None:
    if not path.exists():
        return
    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_optional_json_object(raw: str, *, field_name: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{field_name} must be a valid JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{field_name} must decode to a JSON object")
    return payload


def _resolve_env_value(
    *,
    explicit: str,
    env_name: str,
    fallback_envs: Sequence[str],
) -> str:
    explicit_value = str(explicit or "").strip()
    if explicit_value:
        return explicit_value
    named_env = str(env_name or "").strip()
    if named_env:
        return str(os.environ.get(named_env, "") or "").strip()
    for candidate in fallback_envs:
        value = str(os.environ.get(candidate, "") or "").strip()
        if value:
            return value
    return ""


def resolve_memory_agent_profile(raw: str) -> str:
    profile = str(raw or "").strip().lower()
    if profile in {"fidelity", OPENCLAW_FIDELITY_PROFILE}:
        return OPENCLAW_FIDELITY_PROFILE
    return BENCHMARK_TUNED_PROFILE


def profile_uses_searchable_highlights(profile: str) -> bool:
    return resolve_memory_agent_profile(profile) != OPENCLAW_FIDELITY_PROFILE


def profile_rewrites_memory_search_payload(profile: str) -> bool:
    return resolve_memory_agent_profile(profile) != OPENCLAW_FIDELITY_PROFILE


def profile_forces_first_search(profile: str) -> bool:
    return resolve_memory_agent_profile(profile) != OPENCLAW_FIDELITY_PROFILE


def profile_adds_benchmark_hints(profile: str) -> bool:
    return resolve_memory_agent_profile(profile) != OPENCLAW_FIDELITY_PROFILE


def load_jsonl(path: Path, *, max_attempts: int = 3, retry_delay_s: float = 0.25) -> List[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for lineno, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        except json.JSONDecodeError as exc:
            preview = ""
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for current_lineno, raw_line in enumerate(handle, start=1):
                        if current_lineno == exc.lineno:
                            preview = raw_line[:160]
                            break
            except Exception:
                preview = ""
            message = (
                f"invalid jsonl in {path} line {exc.lineno}: {exc.msg}"
                + (f" | preview={preview!r}" if preview else "")
            )
            last_error = ValueError(message)
            if attempt >= max(1, int(max_attempts)):
                raise last_error
            time.sleep(max(0.0, float(retry_delay_s)) * attempt)
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, int(max_attempts)):
                raise
            time.sleep(max(0.0, float(retry_delay_s)) * attempt)
    if last_error is not None:
        raise last_error
    return []


def load_question(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def q0_reuse_matches(
    summary: Dict[str, Any],
    *,
    corpus_sha256: str,
    memory_backend: str,
    embedding_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
    sources: Sequence[str],
    include_searchable_highlights: bool,
) -> bool:
    summary_sources = list(summary.get("sources") or [])
    expected_sources = list(sources)
    return (
        str(summary.get("corpus_sha256", "") or "").strip() == corpus_sha256
        and str(summary.get("memory_backend", "") or "").strip().lower() == memory_backend
        and str(summary.get("embedding_model", "") or "").strip() == str(embedding_model or "").strip()
        and int(summary.get("chunk_tokens", 0) or 0) == int(chunk_tokens or 0)
        and int(summary.get("chunk_overlap", 0) or 0) == int(chunk_overlap or 0)
        and bool(summary.get("include_searchable_highlights", True)) == bool(include_searchable_highlights)
        and summary_sources == expected_sources
    )


def infer_dataset_name(corpus_path: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    parts = corpus_path.resolve().parts
    if "fixed2k_sbins_fixed2k_main3m_20260224_102211" in parts:
        idx = parts.index("fixed2k_sbins_fixed2k_main3m_20260224_102211")
        rel = parts[idx + 1 : -1]
        return "/".join(rel)
    return corpus_path.parent.name


def infer_bin_s(dataset_name: str) -> int | str:
    import re

    match = re.search(r"(?:^|/)s(\d+)(?:/|$)", str(dataset_name))
    if match:
        return int(match.group(1))
    return ""


def parse_sources(raw: str) -> List[str]:
    allowed = {"memory", "sessions"}
    values = [
        token.strip().lower()
        for token in str(raw or "").replace("+", ",").split(",")
        if token.strip()
    ]
    ordered: List[str] = []
    for value in values:
        if value in allowed and value not in ordered:
            ordered.append(value)
    return ordered or ["sessions"]


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    table = TOKEN_COSTS.get(model)
    if not table:
        return 0.0
    return (
        prompt_tokens * float(table["prompt"]) + completion_tokens * float(table["completion"])
    ) / 1000.0


def normalize_text(text: str) -> List[str]:
    import re

    raw = str(text or "").lower()
    raw = re.sub(r"[^\w\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.split() if raw else []


def f1_score(pred: str, gold: str) -> float:
    from collections import Counter

    pred_tokens = normalize_text(pred)
    gold_tokens = normalize_text(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())
    if num_same <= 0:
        return 0.0
    precision = num_same / max(1, sum(pred_counter.values()))
    recall = num_same / max(1, sum(gold_counter.values()))
    return (2.0 * precision * recall) / (precision + recall)


def ensure_trace_writer(path: Path) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
    if not csv_exists:
        writer.writeheader()
    writer._file_handle = handle  # type: ignore[attr-defined]
    return writer


def close_trace_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_file_handle", None)
    if handle is not None:
        handle.close()


class OpenAICompatRunner:
    def __init__(
        self,
        *,
        model: str,
        timeout: int = 300,
        api_key: str = "",
        base_url: str = "",
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        api_key = str(api_key or "").strip() or os.getenv("OPENAI_API_KEY", "") or os.getenv("GPT_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY/GPT_API_KEY in the environment.")
        self.model = model
        self.base_url = str(base_url or "").strip() or os.getenv("OPENAI_BASE_URL", "") or os.getenv("GPT_BASE_URL", "")
        self.extra_body = dict(extra_body or {})
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=(self.base_url or None),
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self.client.close()

    async def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> tuple[List[List[float]], UsageTotals]:
        usage = UsageTotals()
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if not batch:
                continue
            response = await self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([list(row.embedding) for row in response.data])
            resp_usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(resp_usage, "prompt_tokens", 0) or 0)
            total_tokens = int(getattr(resp_usage, "total_tokens", prompt_tokens) or 0)
            usage.prompt_tokens += prompt_tokens
            usage.total_tokens += total_tokens
            usage.total_cost_usd += estimate_cost_usd(self.model, prompt_tokens, 0)
            usage.calls += 1
        return all_embeddings, usage

    @staticmethod
    def extract_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item.get("text")))
                    elif "content" in item and item.get("content"):
                        parts.append(str(item.get("content")))
                elif item:
                    parts.append(str(item))
            return "\n".join(part.strip() for part in parts if str(part).strip()).strip()
        if content is None:
            return ""
        return str(content).strip()

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
    ) -> tuple[Any, UsageTotals]:
        usage = UsageTotals()
        request_kwargs: Dict[str, Any] = {}
        if self.model.lower().startswith("gpt-5"):
            request_kwargs["reasoning_effort"] = "minimal"
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        extra_body = dict(self.extra_body)
        if (
            "dashscope.aliyuncs.com" in self.base_url.lower()
            and self.model.lower().startswith("qwen3")
            and "enable_thinking" not in extra_body
        ):
            extra_body["enable_thinking"] = False
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **request_kwargs,
        )
        message = response.choices[0].message
        resp_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(resp_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(resp_usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(resp_usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = total_tokens
        usage.total_cost_usd = estimate_cost_usd(self.model, prompt_tokens, completion_tokens)
        usage.calls = 1
        return message, usage

    async def chat(self, messages: List[Dict[str, str]], *, max_tokens: int = 256, temperature: float = 0.0) -> tuple[str, UsageTotals]:
        message, usage = await self.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = self.extract_text_from_content(getattr(message, "content", ""))
        return text, usage


def build_prompt_messages(
    *,
    question: str,
    question_time: str,
    question_type: str,
    search_results: List[SearchResult],
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are answering a long-term memory benchmark question. "
        "Use only the retrieved memory snippets. "
        "If the snippets are insufficient, say so briefly. "
        "When facts conflict, prefer snippets from session times closer to the question time."
    )
    blocks: List[str] = []
    for idx, result in enumerate(search_results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] source={result.source} session_id={result.session_id} session_time={result.session_time}",
                    f"path={result.path} lines={result.start_line}-{result.end_line} score={result.score:.4f}",
                    result.text.strip(),
                ]
            ).strip()
        )
    user_prompt = "\n\n".join(
        [
            f"Question Type: {question_type or 'unknown'}",
            f"Question Time: {question_time or 'unknown'}",
            "Retrieved Memory Snippets:",
            "\n\n".join(blocks) if blocks else "(none)",
            f"Question: {question}",
            "Answer briefly without extra explanation.",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_chat_messages(messages: List[Dict[str, str]]) -> str:
    rows: List[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        body_parts: List[str] = []
        if message.get("tool_calls"):
            body_parts.append(
                json.dumps(message.get("tool_calls", []), ensure_ascii=False, indent=2)
            )
        if message.get("tool_call_id"):
            body_parts.append(f"tool_call_id={message['tool_call_id']}")
        content = message.get("content", "")
        if content not in (None, ""):
            if isinstance(content, str):
                body_parts.append(content)
            else:
                body_parts.append(json.dumps(content, ensure_ascii=False, indent=2))
        rows.append(f"{role}:\n" + ("\n".join(body_parts).strip() or "(empty)"))
    return "\n\n".join(rows).strip()


def build_memory_agent_messages(
    *,
    question: str,
    question_time: str,
    question_type: str,
    memory_prompt_lines: Optional[Sequence[str]] = None,
    agent_profile: str = BENCHMARK_TUNED_PROFILE,
) -> List[Dict[str, str]]:
    prompt_lines = [str(line).strip() for line in list(memory_prompt_lines or []) if str(line).strip()]
    if not prompt_lines:
        prompt_lines = [
            "## Memory Recall",
            "Before answering anything about prior work, decisions, dates, people, preferences, or todos: run memory_search on MEMORY.md + memory/*.md; then use memory_get to pull only the needed lines. If low confidence after search, say you checked.",
            "Citations: include Source: <path#line> when it helps the user verify memory snippets.",
        ]
    if profile_adds_benchmark_hints(agent_profile):
        prompt_lines.extend(
            [
                "For benchmark questions, prefer specific remembered facts over generic advice.",
                "If the first memory_search results are broad, truncated, or only partially relevant, reformulate the query and search again before finishing.",
                "If a result looks promising but the key fact may be outside the snippet, call memory_get on that path before saying the memory is insufficient.",
                "Do not say there is no evidence if a retrieved snippet already contains a plausible answer.",
            ]
        )
        system_prompt = "\n".join(
            [
                "You are a personal assistant running inside OpenClaw.",
                "",
                *prompt_lines,
                "",
                "Answer briefly and do not add extra explanation.",
            ]
        ).strip()
    else:
        system_prompt = "\n".join(prompt_lines).strip()
    user_prompt = "\n".join(
        [
            f"Question Type: {question_type or 'unknown'}",
            f"Question Time: {question_time or 'unknown'}",
            f"Question: {question}",
        ]
    ).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_memory_tools(*, max_results_cap: int) -> List[Dict[str, Any]]:
    max_results_cap = max(1, int(max_results_cap))
    memory_search_tool = json.loads(json.dumps(MEMORY_SEARCH_TOOL))
    memory_search_tool["function"]["description"] = (
        "Mandatory recall step: semantically search MEMORY.md + memory/*.md "
        "(and optional session transcripts) before answering questions about prior work, "
        "decisions, dates, people, preferences, or todos; returns top snippets with path + lines. "
        f"Keep maxResults at or below {max_results_cap}. "
        "If response has disabled=true, memory retrieval is unavailable and should be surfaced to the user."
    )
    memory_search_tool["function"]["parameters"]["properties"]["maxResults"]["maximum"] = max_results_cap
    memory_search_tool["function"]["parameters"]["properties"]["maxResults"]["minimum"] = 1
    return [memory_search_tool, json.loads(json.dumps(MEMORY_GET_TOOL))]


def _line_ranges_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> bool:
    return not (end_a < start_b or end_b < start_a)


def _expand_result_excerpt(
    *,
    workspace_dir: Path,
    result: SearchResult,
    before_lines: int,
    after_lines: int,
) -> SearchResult:
    abs_path = workspace_dir / result.path
    if not abs_path.exists():
        return result
    lines = abs_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return result
    start_idx = max(0, int(result.start_line) - 1 - max(0, before_lines))
    end_idx = min(len(lines), int(result.end_line) + max(0, after_lines))
    excerpt = "\n".join(lines[start_idx:end_idx]).strip()
    if not excerpt:
        return result
    return SearchResult(
        id=result.id,
        path=result.path,
        source=result.source,
        session_id=result.session_id,
        session_time=result.session_time,
        start_line=start_idx + 1,
        end_line=end_idx,
        text=excerpt,
        score=result.score,
        vector_score=result.vector_score,
        text_score=result.text_score,
    )


async def _maybe_read_memory_match(
    *,
    engine: Any,
    result: SearchResult,
) -> SearchResult:
    if result.source != SOURCE_MEMORY:
        return result
    read = engine.read_file(
        rel_path=result.path,
        from_line=result.start_line,
        lines=max(1, int(result.end_line) - int(result.start_line) + 1),
    )
    if asyncio.iscoroutine(read):
        read = await read
    text = str(read.get("text", "") or "").strip()
    if not text:
        return result
    return SearchResult(
        id=result.id,
        path=result.path,
        source=result.source,
        session_id=result.session_id,
        session_time=result.session_time,
        start_line=result.start_line,
        end_line=result.end_line,
        text=text,
        score=result.score,
        vector_score=result.vector_score,
        text_score=result.text_score,
    )


async def _prepare_prompt_results(
    *,
    engine: Any,
    selected_results: List[SearchResult],
) -> List[SearchResult]:
    prepared: List[SearchResult] = []
    for result in selected_results:
        hydrated = await _maybe_read_memory_match(engine=engine, result=result)
        if any(
            hydrated.path == kept.path
            and _line_ranges_overlap(
                hydrated.start_line,
                hydrated.end_line,
                kept.start_line,
                kept.end_line,
            )
            for kept in prepared
        ):
            continue
        prepared.append(hydrated)
    return prepared


def sanitize_record_for_export(record: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(record, ensure_ascii=False))


def merge_usage(total: UsageTotals, delta: UsageTotals) -> None:
    total.prompt_tokens += delta.prompt_tokens
    total.completion_tokens += delta.completion_tokens
    total.total_tokens += delta.total_tokens
    total.total_cost_usd += delta.total_cost_usd
    total.calls += delta.calls


def infer_provider_name(model: str) -> str:
    lowered = str(model or "").strip().lower()
    if lowered.startswith("text-embedding-3") or lowered.startswith("text-embedding-ada"):
        return "openai"
    if "gemini" in lowered:
        return "gemini"
    if "voyage" in lowered:
        return "voyage"
    if "mistral" in lowered:
        return "mistral"
    if "ollama" in lowered:
        return "ollama"
    return "custom"


def embedding_base_url_fallback_envs(model: str) -> tuple[str, ...]:
    provider = infer_provider_name(model)
    if provider == "openai":
        return ("OPENROUTER_BASE_URL", "OPENAI_BASE_URL", "GPT_BASE_URL")
    if provider == "ollama":
        return ("OLLAMA_BASE_URL",)
    return ("OPENAI_BASE_URL", "GPT_BASE_URL")


def embedding_api_key_fallback_envs(model: str) -> tuple[str, ...]:
    provider = infer_provider_name(model)
    if provider == "openai":
        return (
            "OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY2",
            "OPENAI_API_KEY",
            "GPT_API_KEY",
        )
    if provider == "ollama":
        return ("OLLAMA_API_KEY",)
    return ("OPENAI_API_KEY", "GPT_API_KEY")


def resolve_official_remote_config(
    model: str,
    *,
    base_url: str = "",
    api_key: str = "",
) -> tuple[str, str, str]:
    provider = infer_provider_name(model)
    if provider == "openai":
        return (
            provider,
            str(
                base_url
                or _resolve_env_value(
                    explicit="",
                    env_name="",
                    fallback_envs=embedding_base_url_fallback_envs(model),
                )
                or ""
            ),
            str(
                api_key
                or _resolve_env_value(
                    explicit="",
                    env_name="",
                    fallback_envs=embedding_api_key_fallback_envs(model),
                )
                or ""
            ),
        )
    if provider == "gemini":
        return provider, "", str(os.environ.get("GEMINI_API_KEY", "") or "")
    if provider == "voyage":
        return provider, "", str(os.environ.get("VOYAGE_API_KEY", "") or "")
    if provider == "mistral":
        return provider, "", str(os.environ.get("MISTRAL_API_KEY", "") or "")
    if provider == "ollama":
        return provider, "", str(os.environ.get("OLLAMA_API_KEY", "") or "")
    return provider, "", ""


def result_key(result: SearchResult) -> tuple[str, str, int, int]:
    return (result.source, result.path, int(result.start_line), int(result.end_line))


def serialize_search_result(result: SearchResult) -> Dict[str, Any]:
    return {
        "source": result.source,
        "path": result.path,
        "session_id": result.session_id,
        "session_time": result.session_time,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "score": round(result.score, 6),
        "vector_score": round(result.vector_score, 6),
        "text_score": round(result.text_score, 6),
        "text": result.text,
    }


def build_memory_search_tool_payload(
    *,
    results: List[SearchResult],
    embedding_model: str,
    search_mode: str,
) -> Dict[str, Any]:
    return {
        "results": [
            {
                "path": result.path,
                "startLine": result.start_line,
                "endLine": result.end_line,
                "score": round(result.score, 6),
                "snippet": result.text,
                "source": result.source,
            }
            for result in results
        ],
        "provider": infer_provider_name(embedding_model),
        "model": embedding_model,
        "fallback": None,
        "citations": "off",
        "mode": search_mode,
    }


def build_model_memory_search_payload(
    *,
    prepared_results: List[SearchResult],
    embedding_model: str,
    search_mode: str,
    raw_payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload = build_memory_search_tool_payload(
        results=prepared_results,
        embedding_model=embedding_model,
        search_mode=search_mode,
    )
    if "provider" in raw_payload:
        payload["provider"] = raw_payload.get("provider")
    if "model" in raw_payload:
        payload["model"] = raw_payload.get("model")
    if "fallback" in raw_payload:
        payload["fallback"] = raw_payload.get("fallback")
    if "citations" in raw_payload:
        payload["citations"] = raw_payload.get("citations")
    if "mode" in raw_payload:
        payload["mode"] = raw_payload.get("mode")
    if raw_payload.get("disabled"):
        payload["disabled"] = True
    if raw_payload.get("unavailable"):
        payload["unavailable"] = True
    if raw_payload.get("warning") is not None:
        payload["warning"] = raw_payload.get("warning")
    if raw_payload.get("action") is not None:
        payload["action"] = raw_payload.get("action")
    if raw_payload.get("error") is not None:
        payload["error"] = raw_payload.get("error")
    return payload


def normalize_tool_call(tool_call: Any) -> Dict[str, Any]:
    function = getattr(tool_call, "function", None)
    return {
        "id": str(getattr(tool_call, "id", "") or ""),
        "type": "function",
        "function": {
            "name": str(getattr(function, "name", "") or ""),
            "arguments": str(getattr(function, "arguments", "") or "{}"),
        },
    }


async def execute_memory_search_call(
    *,
    engine: Any,
    embedder: OpenAICompatRunner,
    embedding_model: str,
    question_time: str,
    query: str,
    top_k_default: int,
    min_score_default: float,
    candidate_multiplier: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    search_started = time.perf_counter()
    if isinstance(engine, OfficialOpenClawMemoryBackend):
        requested_k = max(1, int(top_k_default))
        backend_result = await engine.search(
            query=query,
            max_results=requested_k,
            min_score=float(min_score_default),
        )
        selected_results = list(backend_result.get("results") or [])
        payload = dict(backend_result.get("payload") or {})
        return payload, {
            "selected_results": selected_results,
            "selected_results_payload": [
                serialize_search_result(result) for result in selected_results
            ],
            "candidate_limit": None,
            "vector_result_count": None,
            "keyword_result_count": None,
            "duration_ms": round((time.perf_counter() - search_started) * 1000.0, 3),
            "usage": UsageTotals(),
            "search_mode": backend_result.get("mode") or "hybrid",
            "status": backend_result.get("status") or {},
            "disabled": bool(backend_result.get("disabled")),
            "unavailable": bool(backend_result.get("unavailable")),
            "warning": backend_result.get("warning"),
            "action": backend_result.get("action"),
            "error": backend_result.get("error"),
        }

    query_embeddings, embed_usage = await embedder.embed_texts([query])
    query_embedding = query_embeddings[0] if query_embeddings else []
    requested_k = max(1, int(top_k_default))
    candidate_limit = min(200, requested_k * max(1, int(candidate_multiplier)))
    vector_results = engine.search_vector(query_embedding, candidate_limit) if query_embedding else []
    keyword_results = engine.search_keyword(query, candidate_limit)
    merged_results = engine.merge_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
        reference_time=question_time,
    )
    selected_results = engine.select_results(
        merged_results=merged_results,
        keyword_results=keyword_results,
        max_results=requested_k,
        min_score=float(min_score_default),
    )
    search_mode = "hybrid" if query_embedding else "fts-only"
    payload = build_memory_search_tool_payload(
        results=selected_results,
        embedding_model=embedding_model,
        search_mode=search_mode,
    )
    meta = {
        "selected_results": selected_results,
        "selected_results_payload": [serialize_search_result(result) for result in selected_results],
        "candidate_limit": candidate_limit,
        "vector_result_count": len(vector_results),
        "keyword_result_count": len(keyword_results),
        "duration_ms": round((time.perf_counter() - search_started) * 1000.0, 3),
        "usage": embed_usage,
        "search_mode": search_mode,
    }
    return payload, meta


def execute_memory_get_call(
    *,
    engine: Any,
    path: str,
    from_line: Optional[int],
    line_count: Optional[int],
) -> Any:
    try:
        result = engine.read_file(
            rel_path=path,
            from_line=from_line,
            lines=line_count,
        )
        if asyncio.iscoroutine(result):
            async def _await_result() -> Dict[str, Any]:
                resolved = await result
                return {"path": resolved.get("path", path), "text": resolved.get("text", "")}

            return _await_result()
        return {"path": result.get("path", path), "text": result.get("text", "")}
    except Exception as exc:
        return {
            "path": path,
            "text": "",
            "disabled": True,
            "error": str(exc),
        }


def build_trace_row(dataset_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    output = str(record.get("output", "") or "")
    gold = str(record.get("answer", "") or "")
    formatted_prompt = str(record.get("formatted_prompt", "") or "")
    chunks = list(record.get("chunks", []) or [])
    summaries = list(record.get("summaries", []) or [])
    triples = list(record.get("triples", []) or [])
    context_ok = bool(formatted_prompt) and (bool(chunks) or bool(summaries) or bool(triples))
    trace_json = json.dumps(sanitize_record_for_export(record), ensure_ascii=False)
    s_value = record.get("bin_s", "")
    try:
        s_value = int(s_value)
    except Exception:
        s_value = ""
    return {
        "task_id": record.get("question_id", ""),
        "dataset_name": dataset_name,
        "model": record.get("model", ""),
        "memory": record.get("memory_system", "OpenClawMemoryMVP"),
        "s": s_value,
        "question_type": record.get("question_type", ""),
        "trial": record.get("trial", 1),
        "retrieval_calls": record.get("retrieval_calls", 0),
        "retrieved_sessions": record.get("retrieved_sessions", len(record.get("top_session_ids", []) or [])),
        "react_steps": len(record.get("react_trace", []) or []),
        "success": int(bool(record.get("success"))) if record.get("success") is not None and record.get("success") != "" else "",
        "f1": round(f1_score(output, gold), 6),
        "llm_judge": int(bool(record.get("llm_judge"))) if record.get("llm_judge") is not None and record.get("llm_judge") != "" else "",
        "response_duration_ms": record.get("response_duration_ms", 0.0),
        "search_duration_ms": record.get("search_duration_ms", 0.0),
        "total_duration_ms": record.get("total_duration_ms", 0.0),
        "context_tokens": record.get("context_tokens", 0),
        "total_cost_usd": record.get("total_cost_usd", 0.0),
        "context_ok": int(context_ok),
        "agent_output": output,
        "trace_json": trace_json,
    }


async def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    corpus_path = Path(args.corpus_json).resolve()
    question_path = Path(args.question_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    reuse_existing_q0 = bool(getattr(args, "reuse_existing_q0", False))
    if args.force and output_dir.exists() and not reuse_existing_q0:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = infer_dataset_name(corpus_path, args.dataset_name)
    corpus_rows = load_jsonl(corpus_path)
    corpus_sha256 = sha256_file(corpus_path)
    question_row = load_question(question_path)
    sources = parse_sources(getattr(args, "sources", "memory"))
    memory_backend = str(getattr(args, "memory_backend", "official") or "official").strip().lower()
    memory_agent_profile = resolve_memory_agent_profile(
        getattr(args, "memory_agent_profile", BENCHMARK_TUNED_PROFILE)
    )
    include_searchable_highlights = profile_uses_searchable_highlights(memory_agent_profile)
    chat_base_url = _resolve_env_value(
        explicit=getattr(args, "chat_base_url", ""),
        env_name=getattr(args, "chat_base_url_env", ""),
        fallback_envs=("OPENAI_BASE_URL", "GPT_BASE_URL"),
    )
    chat_api_key = _resolve_env_value(
        explicit="",
        env_name=getattr(args, "chat_api_key_env", ""),
        fallback_envs=("OPENAI_API_KEY", "GPT_API_KEY"),
    )
    chat_extra_body = _resolve_optional_json_object(
        getattr(args, "chat_extra_body_json", ""),
        field_name="chat_extra_body_json",
    )
    embedding_base_url = _resolve_env_value(
        explicit=getattr(args, "embedding_base_url", ""),
        env_name=getattr(args, "embedding_base_url_env", ""),
        fallback_envs=embedding_base_url_fallback_envs(args.embedding_model),
    )
    embedding_api_key = _resolve_env_value(
        explicit="",
        env_name=getattr(args, "embedding_api_key_env", ""),
        fallback_envs=embedding_api_key_fallback_envs(args.embedding_model),
    )

    embedder: OpenAICompatRunner | None = OpenAICompatRunner(
        model=args.embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
    )
    chatter: OpenAICompatRunner | None = OpenAICompatRunner(
        model=args.chat_model,
        base_url=chat_base_url,
        api_key=chat_api_key,
        extra_body=chat_extra_body,
    )
    if memory_backend == "official":
        provider, remote_base_url, remote_api_key = resolve_official_remote_config(
            args.embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )
        engine: Any = OfficialOpenClawMemoryBackend(
            repo_root=(
                Path(str(getattr(args, "openclaw_repo_root", "") or "")).expanduser().resolve()
                if str(getattr(args, "openclaw_repo_root", "") or "").strip()
                else Path(__file__).resolve().parents[2]
            ),
            work_root=output_dir,
            embedding_model=args.embedding_model,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            vector_weight=args.vector_weight,
            text_weight=args.text_weight,
            candidate_multiplier=args.candidate_multiplier,
            enable_mmr=args.enable_mmr,
            mmr_lambda=args.mmr_lambda,
            enable_temporal_decay=args.enable_temporal_decay,
            half_life_days=args.half_life_days,
            sources=sources,
            node_bin=str(getattr(args, "openclaw_node_bin", "") or "").strip() or None,
            provider=provider,
            remote_base_url=remote_base_url,
            remote_api_key=remote_api_key,
            include_searchable_highlights=include_searchable_highlights,
        )
    else:
        engine = OpenClawMemoryMVP(
            workspace_dir=output_dir / "workspace",
            index_db_path=output_dir / "openclaw_memory.sqlite",
            token_model=args.chat_model,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            vector_weight=args.vector_weight,
            text_weight=args.text_weight,
            candidate_multiplier=args.candidate_multiplier,
            enable_mmr=args.enable_mmr,
            mmr_lambda=args.mmr_lambda,
            enable_temporal_decay=args.enable_temporal_decay,
            half_life_days=args.half_life_days,
            include_searchable_highlights=include_searchable_highlights,
        )

    try:
        q0_summary_path = output_dir / "q0_index_summary.json"
        q0_cost_summary_path = output_dir / "q0_cost_summary.json"
        can_reuse_q0 = (
            reuse_existing_q0
            and q0_summary_path.exists()
            and (output_dir / "workspace").exists()
            and (output_dir / "openclaw_state").exists()
        )
        if can_reuse_q0:
            existing_q0_summary = json.loads(q0_summary_path.read_text(encoding="utf-8"))
            can_reuse_q0 = q0_reuse_matches(
                existing_q0_summary,
                corpus_sha256=corpus_sha256,
                memory_backend=memory_backend,
                embedding_model=args.embedding_model,
                chunk_tokens=int(args.chunk_tokens),
                chunk_overlap=int(args.chunk_overlap),
                sources=sources,
                include_searchable_highlights=include_searchable_highlights,
            )
        if can_reuse_q0:
            q0_summary = dict(existing_q0_summary)
            embed_usage = UsageTotals()
            q0_duration_ms = float(q0_summary.get("duration_ms", 0.0) or 0.0)
            q0_cost_summary = (
                json.loads(q0_cost_summary_path.read_text(encoding="utf-8"))
                if q0_cost_summary_path.exists()
                else {
                    "embedding_prompt_tokens": 0,
                    "embedding_total_tokens": 0,
                    "embedding_calls": 0,
                    "total_cost_usd": 0.0,
                    "duration_ms": q0_duration_ms,
                    "memory_backend": memory_backend,
                    "cost_observability": "reused_existing_q0",
                }
            )
            q0_summary["reused_existing_q0"] = True
        else:
            q0_started = time.perf_counter()
            if memory_backend == "official":
                q0_summary = await engine.build_index_from_corpus(
                    corpus_rows=corpus_rows,
                    sources=sources,
                )
                embed_usage = UsageTotals()
            else:
                preview_files = engine.build_corpus_files(corpus_rows, sources=sources)
                preview_chunks = engine.build_chunks(preview_files)
                embeddings, embed_usage = await embedder.embed_texts(
                    [chunk.text for chunk in preview_chunks]
                )
                q0_summary = engine.build_index_from_corpus(
                    corpus_rows=corpus_rows,
                    embeddings=embeddings,
                    sources=sources,
                )
            q0_duration_ms = round((time.perf_counter() - q0_started) * 1000.0, 3)

            q0_cost_summary = {
                "embedding_prompt_tokens": embed_usage.prompt_tokens,
                "embedding_total_tokens": embed_usage.total_tokens,
                "embedding_calls": embed_usage.calls,
                "total_cost_usd": round(embed_usage.total_cost_usd, 6),
                "duration_ms": q0_duration_ms,
                "memory_backend": memory_backend,
                "cost_observability": (
                    "not_exposed_by_openclaw" if memory_backend == "official" else "tracked_in_python"
                ),
            }
            q0_summary["corpus_sha256"] = corpus_sha256
            q0_summary["memory_backend"] = memory_backend
            q0_summary["embedding_model"] = str(args.embedding_model or "").strip()
            q0_summary["memory_agent_profile"] = memory_agent_profile
            q0_summary["include_searchable_highlights"] = include_searchable_highlights
            q0_summary["duration_ms"] = q0_duration_ms
            save_json(q0_summary_path, q0_summary)
            save_json(q0_cost_summary_path, q0_cost_summary)

        question = str(question_row.get("question", "") or "")
        question_time = str(question_row.get("question_time", "") or "")
        question_type = str(question_row.get("question_type", "") or "")
        agent_mode = str(getattr(args, "agent_mode", "memory_tools") or "memory_tools").strip().lower()
        answer_text = ""
        chat_usage = UsageTotals()
        query_embed_usage = UsageTotals()
        response_duration_ms = 0.0
        search_duration_ms = 0.0
        prompt_messages: List[Dict[str, Any]] = []
        formatted_prompt = ""
        selected_results_payload: List[Dict[str, Any]] = []
        prompt_context_payload: List[Dict[str, Any]] = []
        react_trace: List[Dict[str, Any]] = []
        react_round_chunk_counts: List[int] = []
        search_results_export: Dict[str, Any] = {
            "sources": sources,
            "agent_mode": agent_mode,
            "memory_backend": memory_backend,
            "memory_agent_profile": memory_agent_profile,
            "force_min_memory_searches": max(
                0,
                int(getattr(args, "force_min_memory_searches", 0) or 0),
            ),
        }

        if agent_mode == "memory_tools":
            forced_min_memory_searches = max(
                0,
                int(getattr(args, "force_min_memory_searches", 0) or 0),
            )
            memory_prompt_lines: Optional[List[str]] = None
            if isinstance(engine, OfficialOpenClawMemoryBackend):
                try:
                    memory_prompt_lines = await engine.prompt_section(
                        citations_mode="auto",
                        available_tools=["memory_search", "memory_get"],
                    )
                except Exception:
                    memory_prompt_lines = None
            prompt_messages = build_memory_agent_messages(
                question=question,
                question_time=question_time,
                question_type=question_type,
                memory_prompt_lines=memory_prompt_lines,
                agent_profile=memory_agent_profile,
            )
            context_results: Dict[tuple[str, str, int, int], SearchResult] = {}
            context_order: List[tuple[str, str, int, int]] = []
            search_call_logs: List[Dict[str, Any]] = []
            tools = build_memory_tools(max_results_cap=max(1, int(args.top_k)))

            for loop_idx in range(max(1, int(getattr(args, "max_agent_steps", 6) or 6))):
                memory_searches_so_far = sum(
                    1 for entry in react_trace if entry.get("action") == "memory_search"
                )
                tool_choice: Any
                if memory_searches_so_far < forced_min_memory_searches:
                    tool_choice = {"type": "function", "function": {"name": "memory_search"}}
                elif loop_idx == 0 and profile_forces_first_search(memory_agent_profile):
                    tool_choice = {"type": "function", "function": {"name": "memory_search"}}
                else:
                    tool_choice = "auto"
                response_started = time.perf_counter()
                assistant_message, step_chat_usage = await chatter.chat_completion(
                    prompt_messages,
                    max_tokens=max(1, int(getattr(args, "chat_max_tokens", 512) or 512)),
                    temperature=0.0,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                response_duration_ms += (time.perf_counter() - response_started) * 1000.0
                merge_usage(chat_usage, step_chat_usage)

                assistant_text = OpenAICompatRunner.extract_text_from_content(
                    getattr(assistant_message, "content", "")
                )
                normalized_tool_calls = [
                    normalize_tool_call(tool_call)
                    for tool_call in list(getattr(assistant_message, "tool_calls", []) or [])
                ]
                assistant_payload: Dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_text,
                }
                if normalized_tool_calls:
                    assistant_payload["tool_calls"] = normalized_tool_calls
                prompt_messages.append(assistant_payload)

                if not normalized_tool_calls:
                    answer_text = assistant_text
                    react_trace.append(
                        {
                            "step": len(react_trace) + 1,
                            "action": "finish",
                            "final_answer": answer_text,
                        }
                    )
                    break

                for tool_call in normalized_tool_calls:
                    function_name = str(tool_call["function"].get("name", "") or "").strip()
                    raw_arguments = str(tool_call["function"].get("arguments", "") or "{}")
                    try:
                        tool_args = json.loads(raw_arguments) if raw_arguments else {}
                    except json.JSONDecodeError:
                        tool_args = {}

                    if function_name == "memory_search":
                        requested_query = str(tool_args.get("query", "") or "").strip() or question
                        requested_top_k = max(
                            1,
                            min(
                                int(tool_args.get("maxResults", args.top_k) or args.top_k),
                                int(args.top_k),
                            ),
                        )
                        requested_min_score = float(
                            tool_args.get("minScore", args.min_score) or args.min_score
                        )
                        payload, meta = await execute_memory_search_call(
                            engine=engine,
                            embedder=embedder,
                            embedding_model=args.embedding_model,
                            question_time=question_time,
                            query=requested_query,
                            top_k_default=requested_top_k,
                            min_score_default=requested_min_score,
                            candidate_multiplier=args.candidate_multiplier,
                        )
                        merge_usage(query_embed_usage, meta["usage"])
                        search_duration_ms += float(meta["duration_ms"])
                        if profile_rewrites_memory_search_payload(memory_agent_profile):
                            model_visible_results = await _prepare_prompt_results(
                                engine=engine,
                                selected_results=list(meta["selected_results"]),
                            )
                            payload_for_model = build_model_memory_search_payload(
                                prepared_results=model_visible_results,
                                embedding_model=args.embedding_model,
                                search_mode=meta["search_mode"],
                                raw_payload=payload,
                            )
                        else:
                            model_visible_results = list(meta["selected_results"])
                            payload_for_model = payload
                        for result in model_visible_results:
                            key = result_key(result)
                            if key not in context_results:
                                context_order.append(key)
                            context_results[key] = result
                        search_call_logs.append(
                            {
                                "query": requested_query,
                                "requested_max_results": requested_top_k,
                                "requested_min_score": requested_min_score,
                                "candidate_limit": meta["candidate_limit"],
                                "vector_result_count": meta["vector_result_count"],
                                "keyword_result_count": meta["keyword_result_count"],
                                "duration_ms": meta["duration_ms"],
                                "mode": meta["search_mode"],
                                "status": meta.get("status", {}),
                                "disabled": bool(meta.get("disabled")),
                                "unavailable": bool(meta.get("unavailable")),
                                "warning": meta.get("warning"),
                                "action_hint": meta.get("action"),
                                "error": meta.get("error"),
                                "results": meta["selected_results_payload"],
                                "tool_payload": payload,
                                "llm_tool_payload": payload_for_model,
                            }
                        )
                        react_round_chunk_counts.append(len(model_visible_results))
                        top_call_session_ids = dedupe_preserve_order(
                            result.session_id for result in model_visible_results
                        )
                        react_trace.append(
                            {
                                "step": len(react_trace) + 1,
                                "action": "memory_search",
                                "query": requested_query,
                                "sources": sources,
                                "requested_max_results": requested_top_k,
                                "requested_min_score": requested_min_score,
                                "disabled": bool(meta.get("disabled")),
                                "top_session_ids": top_call_session_ids,
                                "retrieved_session_count": len(top_call_session_ids),
                                "results": meta["selected_results_payload"],
                            }
                        )
                    elif function_name == "memory_get":
                        requested_path = str(tool_args.get("path", "") or "").strip()
                        requested_from = (
                            int(tool_args["from"])
                            if tool_args.get("from") is not None
                            else None
                        )
                        requested_lines = (
                            int(tool_args["lines"])
                            if tool_args.get("lines") is not None
                            else None
                        )
                        payload = execute_memory_get_call(
                            engine=engine,
                            path=requested_path,
                            from_line=requested_from,
                            line_count=requested_lines,
                        )
                        if asyncio.iscoroutine(payload):
                            payload = await payload
                        if payload.get("text"):
                            for key in list(context_order):
                                existing = context_results.get(key)
                                if existing is None or existing.path != requested_path:
                                    continue
                                updated_start = requested_from or existing.start_line
                                updated_end = (
                                    updated_start + max(1, int(requested_lines)) - 1
                                    if requested_lines is not None
                                    else existing.end_line
                                )
                                if not _line_ranges_overlap(
                                    existing.start_line,
                                    existing.end_line,
                                    updated_start,
                                    updated_end,
                                ):
                                    continue
                                updated = SearchResult(
                                    id=existing.id,
                                    path=existing.path,
                                    source=existing.source,
                                    session_id=existing.session_id,
                                    session_time=existing.session_time,
                                    start_line=updated_start,
                                    end_line=updated_end,
                                    text=str(payload.get("text", "") or ""),
                                    score=existing.score,
                                    vector_score=existing.vector_score,
                                    text_score=existing.text_score,
                                )
                                context_results[key] = updated
                                break
                        react_trace.append(
                            {
                                "step": len(react_trace) + 1,
                                "action": "memory_get",
                                "path": requested_path,
                                "from": requested_from,
                                "lines": requested_lines,
                                "disabled": bool(payload.get("disabled")),
                            }
                        )
                    else:
                        payload = {
                            "disabled": True,
                            "error": f"unsupported tool: {function_name}",
                        }

                    prompt_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                payload_for_model if function_name == "memory_search" else payload,
                                ensure_ascii=False,
                            ),
                        }
                    )

                if args.skip_answer:
                    break

            if not answer_text and not args.skip_answer:
                response_started = time.perf_counter()
                final_message, final_usage = await chatter.chat_completion(
                    prompt_messages,
                    max_tokens=max(1, int(getattr(args, "chat_max_tokens", 512) or 512)),
                    temperature=0.0,
                )
                response_duration_ms += (time.perf_counter() - response_started) * 1000.0
                merge_usage(chat_usage, final_usage)
                answer_text = OpenAICompatRunner.extract_text_from_content(
                    getattr(final_message, "content", "")
                )
                prompt_messages.append({"role": "assistant", "content": answer_text})
                react_trace.append(
                    {
                        "step": len(react_trace) + 1,
                        "action": "finish",
                        "final_answer": answer_text,
                        "forced_after_max_steps": True,
                    }
                )

            search_duration_ms = round(search_duration_ms, 3)
            response_duration_ms = round(response_duration_ms, 3)
            aggregated_results = [context_results[key] for key in context_order if key in context_results]
            selected_results_payload = [
                result
                for call in search_call_logs
                for result in list(call.get("results", []) or [])
            ]
            prompt_context_payload = [
                serialize_search_result(result) for result in aggregated_results
            ]
            formatted_prompt = render_chat_messages(prompt_messages)
            search_results_export.update(
                {
                    "memory_search_calls": search_call_logs,
                    "memory_search_results": selected_results_payload,
                    "prompt_context_results": prompt_context_payload,
                }
            )
        else:
            search_started = time.perf_counter()
            if memory_backend == "official":
                official_search = await engine.search(
                    query=question,
                    max_results=max(1, int(args.top_k)),
                    min_score=float(args.min_score),
                )
                selected_results = list(official_search.get("results") or [])
                if profile_rewrites_memory_search_payload(memory_agent_profile):
                    search_results = await _prepare_prompt_results(
                        engine=engine,
                        selected_results=selected_results,
                    )
                else:
                    search_results = list(selected_results)
                search_duration_ms = round((time.perf_counter() - search_started) * 1000.0, 3)

                selected_results_payload = [
                    serialize_search_result(result) for result in selected_results
                ]
                prompt_context_payload = [
                    serialize_search_result(result) for result in search_results
                ]
                search_results_export.update(
                    {
                        "candidate_limit": None,
                        "vector_result_count": None,
                        "keyword_result_count": None,
                        "provider": official_search.get("provider"),
                        "model": official_search.get("model"),
                        "mode": official_search.get("mode"),
                        "status": official_search.get("status") or {},
                        "memory_search_results": selected_results_payload,
                        "prompt_context_results": prompt_context_payload,
                    }
                )
            else:
                query_embeddings, direct_query_usage = await embedder.embed_texts([question])
                merge_usage(query_embed_usage, direct_query_usage)
                query_embedding = query_embeddings[0] if query_embeddings else []
                candidate_limit = min(
                    200,
                    max(1, int(args.top_k)) * max(1, int(args.candidate_multiplier)),
                )
                vector_results = engine.search_vector(query_embedding, candidate_limit) if query_embedding else []
                keyword_results = engine.search_keyword(question, candidate_limit)
                merged_results = engine.merge_results(
                    vector_results=vector_results,
                    keyword_results=keyword_results,
                    reference_time=question_time,
                )
                selected_results = engine.select_results(
                    merged_results=merged_results,
                    keyword_results=keyword_results,
                    max_results=max(1, int(args.top_k)),
                    min_score=float(args.min_score),
                )
                if profile_rewrites_memory_search_payload(memory_agent_profile):
                    search_results = await _prepare_prompt_results(
                        engine=engine,
                        selected_results=selected_results,
                    )
                else:
                    search_results = list(selected_results)
                search_duration_ms = round((time.perf_counter() - search_started) * 1000.0, 3)

                selected_results_payload = [
                    serialize_search_result(result) for result in selected_results
                ]
                prompt_context_payload = [
                    serialize_search_result(result) for result in search_results
                ]
                search_results_export.update(
                    {
                        "candidate_limit": candidate_limit,
                        "vector_result_count": len(vector_results),
                        "keyword_result_count": len(keyword_results),
                        "memory_search_results": selected_results_payload,
                        "prompt_context_results": prompt_context_payload,
                    }
                )

            prompt_messages = build_prompt_messages(
                question=question,
                question_time=question_time,
                question_type=question_type,
                search_results=search_results,
            )
            formatted_prompt = render_chat_messages(prompt_messages)
            if not args.skip_answer:
                response_started = time.perf_counter()
                answer_text, step_chat_usage = await chatter.chat(
                    prompt_messages,
                    max_tokens=max(1, int(getattr(args, "chat_max_tokens", 512) or 512)),
                    temperature=0.0,
                )
                response_duration_ms = round((time.perf_counter() - response_started) * 1000.0, 3)
                merge_usage(chat_usage, step_chat_usage)
            react_trace = [
                {
                    "step": 1,
                    "action": "memory_search",
                    "query": question,
                    "sources": sources,
                    "results": selected_results_payload,
                },
                {
                    "step": 2,
                    "action": "finish",
                    "final_answer": answer_text,
                },
            ]
            react_round_chunk_counts = [len(search_results)]

        save_json(output_dir / "search_results.json", search_results_export)

        top_session_ids = dedupe_preserve_order(
            item.get("session_id", "") for item in prompt_context_payload
        )[: max(1, int(args.top_session_limit))]
        total_duration_ms = round(search_duration_ms + response_duration_ms, 3)
        total_cost_usd = round(query_embed_usage.total_cost_usd + chat_usage.total_cost_usd, 6)
        context_tokens = count_tokens(formatted_prompt, args.chat_model)

        record: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "question_id": str(question_row.get("question_id", "") or ""),
            "question": question,
            "answer": str(question_row.get("answer", "") or ""),
            "label": question_row.get("label", ""),
            "question_type": question_type,
            "question_time": question_time,
            "origin": question_row.get("origin", ""),
            "bin_s": question_row.get("bin_s", "") or infer_bin_s(dataset_name),
            "output": answer_text,
            "model": args.chat_model,
            "memory_system": "OpenClawOfficialMemory" if memory_backend == "official" else "OpenClawMemoryMVP",
            "trial": 1,
            "retrieval_calls": sum(1 for entry in react_trace if entry.get("action") == "memory_search"),
            "retrieved_sessions": len(top_session_ids),
            "top_session_ids": top_session_ids,
            "react_trace": react_trace,
            "react_round_chunk_counts": react_round_chunk_counts,
            "final_context_chunk_count": len(prompt_context_payload),
            "final_context_chunk_alloc_per_turn": react_round_chunk_counts,
            "formatted_prompt": formatted_prompt,
            "formatted_prompt_messages": prompt_messages,
            "qa_prompt_template_dataset": (
                (
                    "openclaw_memory_tools_fidelity"
                    if agent_mode == "memory_tools"
                    and memory_agent_profile == OPENCLAW_FIDELITY_PROFILE
                    else "openclaw_memory_tools"
                )
                if agent_mode == "memory_tools"
                else "openclaw_memory_mvp"
            ),
            "triples": [],
            "chunks": prompt_context_payload,
            "memory_search_results": selected_results_payload,
            "summaries": [],
            "llm_qa_raw_response": answer_text,
            "llm_qa_raw_answer": answer_text,
            "llm_qa_metadata": {
                "query_embedding_prompt_tokens": query_embed_usage.prompt_tokens,
                "query_embedding_calls": query_embed_usage.calls,
                "chat_prompt_tokens": chat_usage.prompt_tokens,
                "chat_completion_tokens": chat_usage.completion_tokens,
                "chat_calls": chat_usage.calls,
                "agent_mode": agent_mode,
                "memory_backend": memory_backend,
                "memory_agent_profile": memory_agent_profile,
                "force_min_memory_searches": max(
                    0,
                    int(getattr(args, "force_min_memory_searches", 0) or 0),
                ),
            },
            "memos_stats": {
                "context_tokens": int(context_tokens),
                "response_duration_ms": float(response_duration_ms),
                "search_duration_ms": float(search_duration_ms),
                "total_duration_ms": float(total_duration_ms),
            },
            "context_tokens": int(context_tokens),
            "response_duration_ms": float(response_duration_ms),
            "search_duration_ms": float(search_duration_ms),
            "total_duration_ms": float(total_duration_ms),
            "total_cost_usd": total_cost_usd,
            "llm_judge": "",
            "success": "",
            "memory_agent_profile": memory_agent_profile,
        }

        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "results.json"
        save_json(results_path, [sanitize_record_for_export(record)])

        trace_writer = ensure_trace_writer(output_dir / "trace_probe.csv")
        trace_writer.writerow(build_trace_row(dataset_name, record))
        getattr(trace_writer, "_file_handle").flush()  # type: ignore[attr-defined]
        close_trace_writer(trace_writer)

        probe_summary = {
            "dataset_name": dataset_name,
            "corpus_json": str(corpus_path),
            "question_json": str(question_path),
            "output_dir": str(output_dir),
            "chat_model": args.chat_model,
            "embedding_model": args.embedding_model,
            "memory_backend": memory_backend,
            "memory_agent_profile": memory_agent_profile,
            "skip_answer": bool(args.skip_answer),
            "q0_summary_path": str(output_dir / "q0_index_summary.json"),
            "q0_cost_summary_path": str(output_dir / "q0_cost_summary.json"),
            "results_path": str(results_path),
            "search_results_path": str(output_dir / "search_results.json"),
            "trace_csv_path": str(output_dir / "trace_probe.csv"),
            "top_session_ids": top_session_ids,
            "answer_preview": answer_text[:300],
            "memory_knobs": {
                "chunk_tokens": args.chunk_tokens,
                "chunk_overlap": args.chunk_overlap,
                "top_k": args.top_k,
                "top_session_limit": args.top_session_limit,
                "candidate_multiplier": args.candidate_multiplier,
                "vector_weight": args.vector_weight,
                "text_weight": args.text_weight,
                "min_score": args.min_score,
                "sources": sources,
                "memory_backend": memory_backend,
                "memory_agent_profile": memory_agent_profile,
                "force_min_memory_searches": max(
                    0,
                    int(getattr(args, "force_min_memory_searches", 0) or 0),
                ),
                "excerpt_before_lines": args.excerpt_before_lines,
                "excerpt_after_lines": args.excerpt_after_lines,
                "enable_mmr": args.enable_mmr,
                "mmr_lambda": args.mmr_lambda,
                "enable_temporal_decay": args.enable_temporal_decay,
                "half_life_days": args.half_life_days,
            },
        }
        save_json(output_dir / "probe_summary.json", probe_summary)
        return probe_summary
    finally:
        if chatter is not None:
            await chatter.aclose()
        if embedder is not None:
            await embedder.aclose()
        engine.close()


def main() -> None:
    args = parse_args()
    if args.env_file:
        maybe_load_env_file(Path(args.env_file))
    ensure_openai_env_aliases()
    summary = asyncio.run(run_probe(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
