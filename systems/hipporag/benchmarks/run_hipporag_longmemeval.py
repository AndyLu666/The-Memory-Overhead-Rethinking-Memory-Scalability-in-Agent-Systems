#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import json
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
LICOMEMORY_ROOT = REPO_ROOT / "systems" / "licomemory"
HIPPORAG_ROOT = SYSTEM_ROOT / "upstream"
HIPPORAG_SRC = HIPPORAG_ROOT / "src"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "fixed2k_sbins_fixed2k_main3m_20260224_102211"
DEFAULT_LIST_ROOT = LICOMEMORY_ROOT / "scripts" / "dataset_lists" / "fixed2k_sbins_fixed2k_main3m_20260224_102211"
DEFAULT_FIXED2K_MANIFEST = DEFAULT_LIST_ROOT / "manifest.jsonl"
DEFAULT_FIXED2K_BASE_SOURCE_ROOT = REPO_ROOT / "data" / "longmemeval_m_subset_oracle_success100_20260205"
DEFAULT_JUDGE_CONFIG = SYSTEM_ROOT / "config" / "hipporag_longmemeval_eval_only_gpt4omini_memos_20260324.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "runs" / "hipporag_fixed2k_main3m_native_gpt5mini_judge4omini"

if str(LICOMEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(LICOMEMORY_ROOT))
if str(HIPPORAG_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPORAG_SRC))

ig = None
Evaluator = None
LLMEvaluator = None
Config = None
derive_memos_metrics = None
sanitize_trace_record_for_export = None
TOKEN_COSTS = None
HippoRAG = None
answer_looks_like_instruction_fragment = None
build_react_conversation_messages = None
clip_text = None
dedupe_chunks_preserve_order = None
get_question_type_guidance = None
is_meta_completion_answer = None
parse_react_action = None
postprocess_answer = None
select_multihop_chunks_for_prompt = None
_get_llm_class = None
BaseConfig = None
NerRawOutput = None
QuerySolution = None
TripleRawOutput = None
extract_entity_nodes = None
reformat_openie_results = None
text_processing = None


def _load_runtime_dependencies() -> None:
    global ig
    global Evaluator
    global LLMEvaluator
    global Config
    global derive_memos_metrics
    global sanitize_trace_record_for_export
    global TOKEN_COSTS
    global HippoRAG
    global answer_looks_like_instruction_fragment
    global build_react_conversation_messages
    global clip_text
    global dedupe_chunks_preserve_order
    global get_question_type_guidance
    global is_meta_completion_answer
    global parse_react_action
    global postprocess_answer
    global select_multihop_chunks_for_prompt
    global _get_llm_class
    global BaseConfig
    global NerRawOutput
    global QuerySolution
    global TripleRawOutput
    global extract_entity_nodes
    global reformat_openie_results
    global text_processing
    if Config is not None:
        return

    import igraph as _ig
    from evaluation.evaluator import Evaluator as _Evaluator
    from evaluation.llm_evaluator import LLMEvaluator as _LLMEvaluator
    from init.config import Config as _Config
    from memos_stats import (
        derive_memos_metrics as _derive_memos_metrics,
        sanitize_trace_record_for_export as _sanitize_trace_record_for_export,
    )
    from utils.token_counter import TOKEN_COSTS as _TOKEN_COSTS

    from hipporag.HippoRAG import HippoRAG as _HippoRAG
    from hipporag.agentic import (
        answer_looks_like_instruction_fragment as _answer_looks_like_instruction_fragment,
        build_react_conversation_messages as _build_react_conversation_messages,
        clip_text as _clip_text,
        dedupe_chunks_preserve_order as _dedupe_chunks_preserve_order,
        get_question_type_guidance as _get_question_type_guidance,
        is_meta_completion_answer as _is_meta_completion_answer,
        parse_react_action as _parse_react_action,
        postprocess_answer as _postprocess_answer,
        select_multihop_chunks_for_prompt as _select_multihop_chunks_for_prompt,
    )
    from hipporag.llm import _get_llm_class as __get_llm_class
    from hipporag.utils.config_utils import BaseConfig as _BaseConfig
    from hipporag.utils.misc_utils import (
        NerRawOutput as _NerRawOutput,
        QuerySolution as _QuerySolution,
        TripleRawOutput as _TripleRawOutput,
        extract_entity_nodes as _extract_entity_nodes,
        reformat_openie_results as _reformat_openie_results,
        text_processing as _text_processing,
    )

    ig = _ig
    Evaluator = _Evaluator
    LLMEvaluator = _LLMEvaluator
    Config = _Config
    derive_memos_metrics = _derive_memos_metrics
    sanitize_trace_record_for_export = _sanitize_trace_record_for_export
    TOKEN_COSTS = _TOKEN_COSTS
    HippoRAG = _HippoRAG
    answer_looks_like_instruction_fragment = _answer_looks_like_instruction_fragment
    build_react_conversation_messages = _build_react_conversation_messages
    clip_text = _clip_text
    dedupe_chunks_preserve_order = _dedupe_chunks_preserve_order
    get_question_type_guidance = _get_question_type_guidance
    is_meta_completion_answer = _is_meta_completion_answer
    parse_react_action = _parse_react_action
    postprocess_answer = _postprocess_answer
    select_multihop_chunks_for_prompt = _select_multihop_chunks_for_prompt
    _get_llm_class = __get_llm_class
    BaseConfig = _BaseConfig
    NerRawOutput = _NerRawOutput
    QuerySolution = _QuerySolution
    TripleRawOutput = _TripleRawOutput
    extract_entity_nodes = _extract_entity_nodes
    reformat_openie_results = _reformat_openie_results
    text_processing = _text_processing


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

EXPECTED_JUDGE_MODEL = "gpt-4o-mini"
EXPECTED_JUDGE_PROMPT_STYLE = "memos_json"
EXPECTED_JUDGE_NUM_RUNS = 3
ALLOWED_BIN_VALUES = {0, 100, 200, 300, 400}
ALLOWED_TASK_FAMILIES = {"longmem_ssp", "longmem_ssu", "longmem_ssa", "longmem_tr"}
HIPPORAG_OPENIE_MODE = "online"
HIPPORAG_RETRIEVAL_TOP_K = 200
HIPPORAG_LINKING_TOP_K = 5
HIPPORAG_MAX_QA_STEPS = 1
HIPPORAG_GRAPH_TYPE = "facts_and_sim_passage_node_unidirectional"
HIPPORAG_EMBEDDING_BATCH_SIZE = 16
HIPPORAG_EMBEDDING_MAX_SEQ_LEN = 8191
HIPPORAG_TEMPERATURE = 0.0
HIPPORAG_MAX_RETRY_ATTEMPTS = int(os.getenv("HIPPORAG_MAX_RETRY_ATTEMPTS_OVERRIDE", "12") or 12)
DEFAULT_REACT_MAX_STEPS = 3
DEFAULT_REACT_MAX_CONTEXT_CHUNKS = 12
DEFAULT_REACT_AGENT_MAX_TOKENS = 512
DEFAULT_REACT_AGENT_TEMPERATURE = 0.0


def disable_hipporag_progress_bars() -> None:
    try:
        from tqdm import tqdm as real_tqdm
    except ModuleNotFoundError:
        return

    def quiet_tqdm(*args: Any, **kwargs: Any):
        kwargs["disable"] = True
        return real_tqdm(*args, **kwargs)

    module_names = [
        "hipporag.HippoRAG",
        "hipporag.StandardRAG",
        "hipporag.embedding_model.OpenAI",
        "hipporag.information_extraction.openie_openai",
        "hipporag.utils.embed_utils",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "tqdm"):
            setattr(module, "tqdm", quiet_tqdm)


disable_hipporag_progress_bars()


def get_llm_base_url() -> str | None:
    return (
        os.getenv("HIPPORAG_LLM_BASE_URL", "").strip()
        or os.getenv("HIPPORAG_Q1_BASE_URL", "").strip()
        or os.getenv("OTHER_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("GPT_BASE_URL", "").strip()
        or None
    )


def get_embedding_base_url() -> str | None:
    return (
        os.getenv("HIPPORAG_EMBEDDING_BASE_URL", "").strip()
        or os.getenv("EMBEDDING_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("GPT_BASE_URL", "").strip()
        or None
    )


def get_llm_api_key() -> str | None:
    return (
        os.getenv("HIPPORAG_LLM_API_KEY", "").strip()
        or os.getenv("HIPPORAG_Q1_API_KEY", "").strip()
        or os.getenv("OTHER_API_KEY", "").strip()
        or os.getenv("QWEN_API", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GPT_API_KEY", "").strip()
        or None
    )


def get_eval_api_key() -> str | None:
    return (
        os.getenv("HIPPORAG_EVAL_API_KEY", "").strip()
        or os.getenv("GPT_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or None
    )


def get_eval_base_url() -> str | None:
    return (
        os.getenv("HIPPORAG_EVAL_BASE_URL", "").strip()
        or os.getenv("GPT_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or None
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HippoRAG on LiCoMemory's fixed2k LongMemEval datasets with the clean MemOS judge/trace contract."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Fixed2k LongMemEval dataset root.",
    )
    parser.add_argument(
        "--dataset-list",
        required=True,
        help="Text file containing dataset_name entries relative to data_root.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Output root for HippoRAG q0/q1 artifacts and traces.",
    )
    parser.add_argument(
        "--judge-config",
        default=str(DEFAULT_JUDGE_CONFIG),
        help="LiCoMemory clean q1 config that defines the MemOS judge contract.",
    )
    parser.add_argument(
        "--mode",
        choices=["q0", "q1", "q0q1"],
        default="q0q1",
        help="Run preprocessing/index only, query only, or both.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5-mini",
        help="HippoRAG LLM model name.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="HippoRAG embedding model name.",
    )
    parser.add_argument(
        "--qa-top-k",
        type=int,
        default=5,
        help="How many retrieved passages to feed into HippoRAG QA.",
    )
    parser.add_argument(
        "--top-session-limit",
        type=int,
        default=20,
        help="Max number of unique session ids stored in trace output.",
    )
    parser.add_argument(
        "--enable-react-multihop",
        action="store_true",
        help="Enable no-controller ReAct-style multi-hop q1 over reused HippoRAG q0 graphs.",
    )
    parser.add_argument(
        "--react-max-steps",
        type=int,
        default=DEFAULT_REACT_MAX_STEPS,
        help="Maximum retrieve/finish turns for HippoRAG multi-hop q1.",
    )
    parser.add_argument(
        "--react-max-context-chunks",
        type=int,
        default=DEFAULT_REACT_MAX_CONTEXT_CHUNKS,
        help="Maximum accumulated chunks exposed to the multi-hop agent prompt.",
    )
    parser.add_argument(
        "--react-agent-max-tokens",
        type=int,
        default=DEFAULT_REACT_AGENT_MAX_TOKENS,
        help="Max completion tokens for the multi-hop agent decision call.",
    )
    parser.add_argument(
        "--react-agent-temperature",
        type=float,
        default=DEFAULT_REACT_AGENT_TEMPERATURE,
        help="Temperature for the multi-hop agent decision call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on how many dataset names to run from dataset-list.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Rebuild HippoRAG q0 caches from scratch.",
    )
    parser.add_argument(
        "--keep-answer-in-results",
        action="store_true",
        help="Do not redact the gold answer from results.json after metrics are computed.",
    )
    return parser.parse_args()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _json_load_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    rows = []
    idx = 0
    text_len = len(text)
    while idx < text_len:
        while idx < text_len and text[idx].isspace():
            idx += 1
        if idx >= text_len:
            break
        try:
            obj, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        rows.append(obj)
        idx = next_idx
    return rows


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    raw = _json_load_any(corpus_path)
    if isinstance(raw, dict):
        raw = [raw]
    rows: List[Dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            rows.append(
                {
                    "session_id": str(item.get("session_id", "") or ""),
                    "session_time": str(item.get("session_time", "") or ""),
                    "context": str(item.get("context", "") or ""),
                }
            )
        else:
            rows.append({"session_id": "", "session_time": "", "context": str(item or "")})
    return rows


def load_question(question_path: Path) -> Dict[str, Any]:
    raw = _json_load_any(question_path)
    if isinstance(raw, list):
        return dict(raw[0]) if raw else {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def format_doc_for_hipporag(row: Dict[str, Any]) -> str:
    session_time = str(row.get("session_time", "") or "").strip()
    context = str(row.get("context", "") or "").strip()
    if session_time:
        return f"Session Date: {session_time}\n{context}"
    return context


_FOCUS_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "ask",
    "asking",
    "been",
    "before",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "for",
    "from",
    "get",
    "give",
    "had",
    "have",
    "help",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "let",
    "like",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "please",
    "remind",
    "should",
    "some",
    "suggest",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "up",
    "use",
    "using",
    "want",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def _focus_terms(query: str) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for term in re.findall(r"[a-z0-9][a-z0-9_/-]{2,}", str(query or "").lower()):
        if term in _FOCUS_STOPWORDS:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out[:8]


def _excerpt_context_for_query(
    context: str,
    *,
    query: str,
    max_chars: int = 2400,
    window_chars: int = 900,
    max_windows: int = 3,
) -> str:
    text = str(context or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    terms = _focus_terms(query)
    if not terms:
        return clip_text(text, max_chars)

    lower = text.lower()
    spans: List[Tuple[int, int]] = []
    per_term_limit = 2
    for term in terms:
        matches = list(re.finditer(re.escape(term), lower))
        for match in matches[:per_term_limit]:
            start = max(0, match.start() - (window_chars // 2))
            end = min(len(text), match.end() + (window_chars // 2))
            spans.append((start, end))
        if len(spans) >= max_windows * per_term_limit:
            break

    if not spans:
        return clip_text(text, max_chars)

    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1] + 120:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: List[str] = []
    for start, end in merged[:max_windows]:
        snippet = text[start:end].strip()
        if snippet:
            parts.append(snippet)

    if not parts:
        return clip_text(text, max_chars)

    excerpt = "\n...\n".join(parts)
    return clip_text(excerpt, max_chars)


def render_context_block(meta: Dict[str, Any], *, focus_query: str = "") -> str:
    session_id = str(meta.get("session_id", "") or "").strip()
    session_time = str(meta.get("session_time", "") or "").strip()
    context = str(meta.get("raw_context", meta.get("rendered_context", "")) or "").strip()
    rendered_context = _excerpt_context_for_query(context, query=focus_query)
    header_parts = []
    if session_id:
        header_parts.append(f"session_id={session_id}")
    if session_time:
        header_parts.append(f"session_time={session_time}")
    if header_parts:
        return f"[{', '.join(header_parts)}]\n{rendered_context}"
    return rendered_context


def make_dataset_dir(results_root: Path, dataset_name: str) -> Path:
    return results_root / dataset_name


def make_hipporag_save_dir(results_root: Path, dataset_name: str) -> Path:
    return make_dataset_dir(results_root, dataset_name) / "hipporag_q0_cache"


def make_hipporag_working_dir(save_dir: Path, llm_model: str, embedding_model: str) -> Path:
    return save_dir / f"{llm_model.replace('/', '_')}_{embedding_model.replace('/', '_')}"


def build_q0_artifact_paths(save_dir: Path, llm_model: str, embedding_model: str) -> Dict[str, str]:
    working_dir = make_hipporag_working_dir(save_dir, llm_model, embedding_model)
    return {
        "cache_dir": str(save_dir),
        "working_dir": str(working_dir),
        "graph_pickle": str(working_dir / "graph.pickle"),
        "chunk_embeddings_parquet": str(working_dir / "chunk_embeddings" / "vdb_chunk.parquet"),
        "entity_embeddings_parquet": str(working_dir / "entity_embeddings" / "vdb_entity.parquet"),
        "fact_embeddings_parquet": str(working_dir / "fact_embeddings" / "vdb_fact.parquet"),
        "llm_cache_sqlite": str(save_dir / "llm_cache" / f"{llm_model.replace('/', '_')}_cache.sqlite"),
        "openie_results_json": str(save_dir / f"openie_results_ner_{llm_model.replace('/', '_')}.json"),
        "doc_metadata_json": str(save_dir / "doc_metadata.json"),
    }


def estimate_model_cost_usd(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    rates = TOKEN_COSTS.get(str(model or ""), {})
    prompt_rate = float(rates.get("prompt", 0.0) or 0.0)
    completion_rate = float(rates.get("completion", 0.0) or 0.0)
    return round(
        ((int(prompt_tokens or 0) * prompt_rate) + (int(completion_tokens or 0) * completion_rate)) / 1000.0,
        6,
    )


def estimate_q1_runtime_cost_usd(
    runtime_cost: Dict[str, Any],
    qa_metadata: Dict[str, Any],
    llm_model: str,
) -> float:
    total_cost = float(runtime_cost.get("total_cost_usd", 0.0) or 0.0)
    if total_cost > 0.0:
        return round(total_cost, 6)
    prompt_tokens = int(qa_metadata.get("prompt_tokens", 0) or 0)
    completion_tokens = int(qa_metadata.get("completion_tokens", 0) or 0)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0
    return estimate_model_cost_usd(prompt_tokens, completion_tokens, llm_model)


def get_usage_totals_safe(obj: Any) -> Dict[str, Any]:
    if obj is None or not hasattr(obj, "get_usage_totals"):
        return {}
    try:
        usage = obj.get_usage_totals()
    except Exception:
        return {}
    return dict(usage or {})


def collect_hipporag_usage_cost(hippo: HippoRAG, llm_model: str, embedding_model: str) -> Dict[str, Any]:
    llm_usage = get_usage_totals_safe(getattr(hippo, "llm_model", None))
    embedding_usage = get_usage_totals_safe(getattr(hippo, "embedding_model", None))

    llm_prompt_tokens = int(llm_usage.get("prompt_tokens", 0) or 0)
    llm_completion_tokens = int(llm_usage.get("completion_tokens", 0) or 0)
    embedding_prompt_tokens = int(embedding_usage.get("prompt_tokens", 0) or 0)

    llm_cost_usd = estimate_model_cost_usd(llm_prompt_tokens, llm_completion_tokens, llm_model)
    embedding_cost_usd = estimate_model_cost_usd(embedding_prompt_tokens, 0, embedding_model)

    return {
        "pricing_source": "LiCoMemory.utils.token_counter.TOKEN_COSTS",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "llm_usage": {
            "prompt_tokens": llm_prompt_tokens,
            "completion_tokens": llm_completion_tokens,
            "total_tokens": int(llm_usage.get("total_tokens", llm_prompt_tokens + llm_completion_tokens) or 0),
            "api_call_count": int(llm_usage.get("api_call_count", 0) or 0),
            "cache_hit_count": int(llm_usage.get("cache_hit_count", 0) or 0),
        },
        "embedding_usage": {
            "prompt_tokens": embedding_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": int(embedding_usage.get("total_tokens", embedding_prompt_tokens) or 0),
            "api_call_count": int(embedding_usage.get("api_call_count", 0) or 0),
            "encoded_text_count": int(embedding_usage.get("encoded_text_count", 0) or 0),
        },
        "llm_cost_usd": llm_cost_usd,
        "embedding_cost_usd": embedding_cost_usd,
        "total_cost_usd": round(llm_cost_usd + embedding_cost_usd, 6),
    }


def q0_ready(save_dir: Path, llm_model: str, embedding_model: str) -> bool:
    working_dir = make_hipporag_working_dir(save_dir, llm_model, embedding_model)
    return (
        (working_dir / "graph.pickle").exists()
        and (working_dir / "chunk_embeddings" / "vdb_chunk.parquet").exists()
        and (working_dir / "entity_embeddings" / "vdb_entity.parquet").exists()
        and (working_dir / "fact_embeddings" / "vdb_fact.parquet").exists()
        and (save_dir / "doc_metadata.json").exists()
    )


def build_doc_metadata(corpus_rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    docs: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for row in corpus_rows:
        rendered = format_doc_for_hipporag(row)
        docs.append(rendered)
        metadata.append(
            {
                "session_id": str(row.get("session_id", "") or ""),
                "session_time": str(row.get("session_time", "") or ""),
                "raw_context": str(row.get("context", "") or ""),
                "rendered_context": rendered,
            }
        )
    return docs, metadata


def build_content_index(doc_metadata: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    content_to_meta: Dict[str, List[Dict[str, Any]]] = {}
    for meta in doc_metadata:
        content_to_meta.setdefault(meta["rendered_context"], []).append(meta)
    return content_to_meta


def resolve_doc_metas(
    retrieved_docs: Iterable[str],
    doc_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content_index = build_content_index(doc_metadata)
    doc_metas: List[Dict[str, Any]] = []
    for doc in retrieved_docs:
        queue = content_index.get(str(doc), [])
        if queue:
            doc_metas.append(queue[0])
        else:
            doc_metas.append(
                {
                    "session_id": "",
                    "session_time": "",
                    "raw_context": str(doc or ""),
                    "rendered_context": str(doc or ""),
                }
            )
    return doc_metas


def multiset_doc_difference(source_docs: List[str], retained_docs: List[str]) -> List[str]:
    retained_counter = Counter(retained_docs)
    docs_to_delete: List[str] = []
    for doc in source_docs:
        if retained_counter[doc] > 0:
            retained_counter[doc] -= 1
        else:
            docs_to_delete.append(doc)
    return docs_to_delete


def refresh_hipporag_graph_from_current_cache(hippo: HippoRAG) -> Dict[str, Any]:
    all_openie_info, _ = hippo.load_existing_openie([])
    hippo.graph = ig.Graph(directed=hippo.global_config.is_directed_graph)
    hippo.node_to_node_stats = {}
    hippo.ent_node_to_chunk_ids = {}

    current_chunk_ids = list(hippo.chunk_embedding_store.get_all_ids())
    ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

    for chunk_id in current_chunk_ids:
        if chunk_id not in ner_results_dict:
            ner_results_dict[chunk_id] = NerRawOutput(
                chunk_id=chunk_id,
                response=None,
                metadata={},
                unique_entities=[],
            )
        if chunk_id not in triple_results_dict:
            triple_results_dict[chunk_id] = TripleRawOutput(
                chunk_id=chunk_id,
                response=None,
                metadata={},
                triples=[],
            )

    chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in current_chunk_ids]
    _, chunk_triple_entities = extract_entity_nodes(chunk_triples)
    hippo.add_fact_edges(current_chunk_ids, chunk_triples)
    hippo.add_passage_edges(current_chunk_ids, chunk_triple_entities)
    if hippo.entity_embedding_store.get_all_ids():
        hippo.add_synonymy_edges()
    hippo.augment_graph()
    hippo.save_igraph()
    hippo.ready_to_retrieve = False
    hippo.prepare_retrieval_objects()
    return {
        "graph_vcount": int(hippo.graph.vcount()),
        "graph_ecount": int(hippo.graph.ecount()),
        "graph_info": hippo.get_graph_info(),
    }


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def infer_bin_s(dataset_name: str) -> int | str:
    parts = [part for part in str(dataset_name or "").split("/") if part]
    if not parts:
        return ""
    stage = parts[0].strip().lower()
    if stage.startswith("s") and stage[1:].isdigit():
        value = int(stage[1:])
        if value in ALLOWED_BIN_VALUES:
            return value
    return ""


def preserve_or_infer_bin_s(value: Any, dataset_name: str) -> int | str:
    if value is None:
        return infer_bin_s(dataset_name)
    if isinstance(value, str) and not value.strip():
        return infer_bin_s(dataset_name)
    return value


def validate_dataset_name(dataset_name: str) -> None:
    parts = [part for part in str(dataset_name or "").split("/") if part]
    if len(parts) < 3:
        raise RuntimeError(f"invalid_dataset_name: {dataset_name}")
    bin_s = infer_bin_s(dataset_name)
    if bin_s == "":
        raise RuntimeError(f"invalid_bin_stage: {dataset_name}")
    if parts[1] not in ALLOWED_TASK_FAMILIES:
        raise RuntimeError(f"invalid_task_family: {dataset_name}")


def maybe_warn_noncanonical_inputs(data_root: Path, dataset_list_path: Path) -> None:
    warnings: List[str] = []
    if data_root.resolve() != DEFAULT_DATA_ROOT.resolve():
        warnings.append(
            f"data_root is not the canonical fixed2k data root: {data_root}"
        )
    if not _is_relative_to(dataset_list_path, DEFAULT_LIST_ROOT):
        warnings.append(
            f"dataset_list is not under the canonical fixed2k list root: {dataset_list_path}"
        )
    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")


def build_native_hipporag_prompt(
    question: str,
    retrieved_docs: List[str],
    *,
    question_type: str = "",
    question_time: str = "",
) -> str:
    prompt_user = ""
    for passage in retrieved_docs:
        prompt_user += f"Wikipedia Title: {passage}\n\n"
    if question_time:
        prompt_user += f"Question Time: {question_time}\n"
    if question_type:
        prompt_user += f"Question Type: {question_type}\n"
        prompt_user += (
            "Question-Type Guidance: "
            f"{get_question_type_guidance(question_type)}\n"
        )
    prompt_user += (
        "Answer Guidance: "
        "Use only the retrieved memory evidence. Answer concisely and directly. "
        "For user-preference or recommendation questions, give one or two concrete memory-grounded suggestions only; "
        "do not give broad brainstorming or long lists. "
        "If the memory includes explicit dislikes, exclusions, or 'instead of / beyond' constraints, preserve them. "
        "If the evidence is insufficient, answer exactly: Insufficient information from context.\n"
    )
    prompt_user += "Question: " + question + "\nThought: "
    return prompt_user


def render_chat_messages(messages: List[Dict[str, Any]]) -> str:
    rendered_blocks: List[str] = []
    for message in messages:
        role = str(message.get("role", "") or "").strip().upper() or "UNKNOWN"
        content = str(message.get("content", "") or "")
        rendered_blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(rendered_blocks)


def build_native_hipporag_qa_messages(
    hippo: HippoRAG,
    *,
    question: str,
    retrieved_docs: List[str],
    question_type: str = "",
    question_time: str = "",
) -> Tuple[str, List[Dict[str, Any]], str]:
    prompt_user = build_native_hipporag_prompt(
        question=question,
        retrieved_docs=retrieved_docs,
        question_type=question_type,
        question_time=question_time,
    )
    if hippo.prompt_template_manager.is_template_name_valid(name=f"rag_qa_{hippo.global_config.dataset}"):
        prompt_dataset_name = str(hippo.global_config.dataset)
    else:
        prompt_dataset_name = "musique"
    rendered_messages = hippo.prompt_template_manager.render(
        name=f"rag_qa_{prompt_dataset_name}",
        prompt_user=prompt_user,
    )
    if not isinstance(rendered_messages, list):
        raise RuntimeError("hipporag_qa_prompt_not_chat_messages")
    return prompt_user, rendered_messages, prompt_dataset_name


def get_hipporag_runtime_knobs(qa_top_k: int, top_session_limit: int) -> Dict[str, Any]:
    return {
        "openie_mode": HIPPORAG_OPENIE_MODE,
        "retrieval_top_k": HIPPORAG_RETRIEVAL_TOP_K,
        "linking_top_k": HIPPORAG_LINKING_TOP_K,
        "qa_top_k": max(1, int(qa_top_k)),
        "max_qa_steps": HIPPORAG_MAX_QA_STEPS,
        "graph_type": HIPPORAG_GRAPH_TYPE,
        "embedding_batch_size": HIPPORAG_EMBEDDING_BATCH_SIZE,
        "embedding_max_seq_len": HIPPORAG_EMBEDDING_MAX_SEQ_LEN,
        "temperature": HIPPORAG_TEMPERATURE,
        "top_session_export_limit": max(1, int(top_session_limit)),
    }


def llm_infer_with_cache_flag(
    llm: Any,
    messages: List[Dict[str, Any]],
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any], bool]:
    result = llm.infer(messages, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        text, metadata, cache_hit = result
        return str(text or ""), dict(metadata or {}), bool(cache_hit)
    if isinstance(result, tuple) and len(result) == 2:
        text, metadata = result
        return str(text or ""), dict(metadata or {}), False
    raise RuntimeError("unexpected_llm_infer_result_shape")
def redact_answer_field_in_results(path: Path) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return
    changed = False
    for row in data:
        if isinstance(row, dict) and "answer" in row:
            row.pop("answer", None)
            changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def f1_score(pred: str, gold: str) -> float:
    def normalize(text: str) -> List[str]:
        import re

        text = str(text or "").lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    pred_tokens = normalize(pred)
    gold_tokens = normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, sum(pred_counter.values()))
    recall = num_same / max(1, sum(gold_counter.values()))
    return (2 * precision * recall) / (precision + recall)


def build_trace_row(dataset_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    output = str(record.get("output", "") or "")
    gold = str(record.get("answer", "") or "")
    question_type = str(record.get("question_type", "") or "")
    memos_metrics = derive_memos_metrics(row=record, record=record)
    sanitized_record = sanitize_trace_record_for_export(record)
    trace_json = json.dumps(sanitized_record, ensure_ascii=False)

    question_json_s = record.get("bin_s", "")
    try:
        s_value: int | str = int(question_json_s)
    except Exception:
        s_value = ""

    chunks = record.get("chunks", []) or []
    summaries = record.get("summaries", []) or []
    triples = record.get("triples", []) or []
    formatted_prompt = str(record.get("formatted_prompt", "") or "")
    context_ok = bool(formatted_prompt) and (bool(chunks) or bool(summaries) or bool(triples))

    return {
        "task_id": record.get("question_id", ""),
        "dataset_name": dataset_name,
        "model": record.get("model", ""),
        "memory": record.get("memory_system", "HippoRAG"),
        "s": s_value,
        "question_type": question_type,
        "trial": record.get("trial", 1),
        "retrieval_calls": record.get("retrieval_calls", 0),
        "retrieved_sessions": record.get("retrieved_sessions", len(record.get("top_session_ids", []) or [])),
        "react_steps": len(record.get("react_trace", []) or []),
        "success": int(bool(record.get("success", False))),
        "f1": round(f1_score(output, gold), 6),
        "llm_judge": int(bool(record.get("llm_judge"))) if record.get("llm_judge") is not None else "",
        "response_duration_ms": memos_metrics["response_duration_ms"],
        "search_duration_ms": memos_metrics["search_duration_ms"],
        "total_duration_ms": memos_metrics["total_duration_ms"],
        "context_tokens": memos_metrics["context_tokens"],
        "total_cost_usd": record.get("total_cost_usd", ""),
        "context_ok": int(context_ok),
        "agent_output": output,
        "trace_json": trace_json,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def validate_clean_contract(cfg: Config) -> None:
    errors: List[str] = []
    if str(cfg.evaluation.eval_model or "").strip() != EXPECTED_JUDGE_MODEL:
        errors.append(f"eval_model must be {EXPECTED_JUDGE_MODEL}")
    if str(cfg.evaluation.eval_prompt_style or "").strip().lower() != EXPECTED_JUDGE_PROMPT_STYLE:
        errors.append(f"eval_prompt_style must be {EXPECTED_JUDGE_PROMPT_STYLE}")
    if int(getattr(cfg.evaluation, "eval_num_runs", 0) or 0) != EXPECTED_JUDGE_NUM_RUNS:
        errors.append(f"eval_num_runs must be {EXPECTED_JUDGE_NUM_RUNS}")
    if errors:
        raise RuntimeError("clean_contract_violation: " + "; ".join(errors))


def _contains_query_error_text(text: Any) -> bool:
    lower = str(text or "").strip().lower()
    if not lower:
        return False
    bad_patterns = (
        "error processing query",
        "traceback",
        "react_agent_action_parse_failed",
        "react_agent_missing_retrieve_query",
        "react_agent_missing_finish_answer",
    )
    return any(pattern in lower for pattern in bad_patterns)


def validate_record_contract(
    record: Dict[str, Any],
    *,
    dataset_name: str,
    expected_judge_runs: int,
    require_judgments: bool,
) -> Dict[str, Any]:
    errors: List[str] = []

    record["bin_s"] = preserve_or_infer_bin_s(record.get("bin_s", ""), dataset_name)

    react_trace = list(record.get("react_trace", []) or [])
    retrieve_steps = [step for step in react_trace if str(step.get("action", "")).strip() == "retrieve"]
    finish_steps = [step for step in react_trace if str(step.get("action", "")).strip() == "finish"]
    output = str(record.get("output", "") or "").strip()
    raw_response = str(record.get("llm_qa_raw_response", "") or "").strip()
    top_session_ids = list(record.get("top_session_ids", []) or [])
    memos_stats = dict(record.get("memos_stats", {}) or {})

    if not output:
        errors.append("empty_agent_output")
    if output.lower() == "complete":
        errors.append("meta_completion_output")
    if _contains_query_error_text(output) or _contains_query_error_text(raw_response):
        errors.append("query_error_text_in_output")
    if int(record.get("retrieval_calls", 0) or 0) != len(retrieve_steps):
        errors.append("retrieval_calls_mismatch")
    if len(finish_steps) != 1:
        errors.append("finish_count_mismatch")
    else:
        final_answer = str(finish_steps[0].get("final_answer", "") or "").strip()
        if not final_answer:
            errors.append("empty_finish_answer")
        elif final_answer != output:
            errors.append("final_answer_output_mismatch")
    if len(top_session_ids) != len(dedupe_preserve_order(top_session_ids)):
        errors.append("duplicate_top_session_ids")
    if int(record.get("retrieved_sessions", 0) or 0) != len(top_session_ids):
        errors.append("retrieved_sessions_mismatch")

    expected_memos_keys = {
        "context_tokens",
        "response_duration_ms",
        "search_duration_ms",
        "total_duration_ms",
    }
    if set(memos_stats.keys()) != expected_memos_keys:
        errors.append("memos_stats_key_mismatch")

    if require_judgments:
        llm_judgments = record.get("llm_judgments", {})
        expected_keys = {f"judgment_{idx}" for idx in range(1, expected_judge_runs + 1)}
        if set((llm_judgments or {}).keys()) != expected_keys:
            errors.append("llm_judgments_key_mismatch")

    if errors:
        raise RuntimeError(
            "hipporag_record_contract_violation: "
            + ", ".join(errors)
            + f" [dataset={dataset_name}]"
        )
    return record


def _looks_like_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith("${") and text.endswith("}")


def normalize_clean_config(cfg: Config) -> Config:
    llm_api_key = get_llm_api_key() or ""
    eval_api_key = get_eval_api_key() or ""
    actual_base_url = get_llm_base_url() or ""
    eval_base_url = get_eval_base_url() or ""
    actual_embedding_base_url = get_embedding_base_url() or ""
    force_llm_api = bool(
        os.getenv("HIPPORAG_LLM_API_KEY", "").strip()
        or os.getenv("HIPPORAG_Q1_API_KEY", "").strip()
        or os.getenv("OTHER_API_KEY", "").strip()
        or os.getenv("QWEN_API", "").strip()
    )
    force_llm_base = bool(
        os.getenv("HIPPORAG_LLM_BASE_URL", "").strip()
        or os.getenv("HIPPORAG_Q1_BASE_URL", "").strip()
        or os.getenv("OTHER_BASE_URL", "").strip()
    )
    force_eval_api = bool(os.getenv("HIPPORAG_EVAL_API_KEY", "").strip())
    force_eval_base = bool(os.getenv("HIPPORAG_EVAL_BASE_URL", "").strip())

    if llm_api_key:
        if force_llm_api or not str(cfg.llm.api_key or "").strip() or _looks_like_placeholder(cfg.llm.api_key):
            cfg.llm.api_key = llm_api_key
        if force_llm_api or not str(cfg.query_llm.api_key or "").strip() or _looks_like_placeholder(cfg.query_llm.api_key):
            cfg.query_llm.api_key = llm_api_key

    if eval_api_key:
        if force_eval_api or not str(cfg.evaluation.eval_api_key or "").strip() or _looks_like_placeholder(cfg.evaluation.eval_api_key):
            cfg.evaluation.eval_api_key = eval_api_key

    if actual_base_url:
        if force_llm_base or not str(cfg.llm.base_url or "").strip() or _looks_like_placeholder(cfg.llm.base_url):
            cfg.llm.base_url = actual_base_url
        if force_llm_base or not str(cfg.query_llm.base_url or "").strip() or _looks_like_placeholder(cfg.query_llm.base_url):
            cfg.query_llm.base_url = actual_base_url
    if eval_base_url:
        if force_eval_base or not str(cfg.evaluation.eval_base_url or "").strip() or _looks_like_placeholder(cfg.evaluation.eval_base_url):
            cfg.evaluation.eval_base_url = eval_base_url
    if actual_embedding_base_url:
        if not str(cfg.embedding.base_url or "").strip() or _looks_like_placeholder(cfg.embedding.base_url):
            cfg.embedding.base_url = actual_embedding_base_url

    return cfg


def load_doc_metadata(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else []


def make_hipporag_config(
    save_dir: Path,
    llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    force_index_from_scratch: bool,
    corpus_len: int | None = None,
) -> BaseConfig:
    cfg = BaseConfig()
    cfg.save_dir = str(save_dir)
    cfg.llm_name = llm_model
    cfg.embedding_model_name = embedding_model
    cfg.llm_base_url = get_llm_base_url()
    cfg.embedding_base_url = get_embedding_base_url()
    cfg.openie_mode = HIPPORAG_OPENIE_MODE
    cfg.force_index_from_scratch = force_index_from_scratch
    cfg.force_openie_from_scratch = force_index_from_scratch
    cfg.rerank_dspy_file_path = str(
        HIPPORAG_ROOT / "src" / "hipporag" / "prompts" / "dspy_prompts" / "filter_llama3.3-70B-Instruct.json"
    )
    cfg.qa_top_k = max(1, int(qa_top_k))
    cfg.retrieval_top_k = HIPPORAG_RETRIEVAL_TOP_K
    cfg.linking_top_k = HIPPORAG_LINKING_TOP_K
    cfg.max_qa_steps = HIPPORAG_MAX_QA_STEPS
    cfg.graph_type = HIPPORAG_GRAPH_TYPE
    cfg.embedding_batch_size = HIPPORAG_EMBEDDING_BATCH_SIZE
    cfg.embedding_max_seq_len = HIPPORAG_EMBEDDING_MAX_SEQ_LEN
    cfg.corpus_len = corpus_len
    cfg.temperature = HIPPORAG_TEMPERATURE
    cfg.max_retry_attempts = HIPPORAG_MAX_RETRY_ATTEMPTS
    return cfg


def make_hipporag_qa_llm(base_cfg: BaseConfig, llm_model: str):
    qa_cfg = deepcopy(base_cfg)
    qa_cfg.llm_name = llm_model
    qa_cfg.llm_base_url = get_llm_base_url()
    return _get_llm_class(qa_cfg)


def run_q0_for_dataset(
    *,
    dataset_name: str,
    corpus_rows: List[Dict[str, Any]],
    results_root: Path,
    llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    force_reindex: bool,
) -> Dict[str, Any]:
    dataset_dir = make_dataset_dir(results_root, dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hippo_save_dir = make_hipporag_save_dir(results_root, dataset_name)
    artifact_paths = build_q0_artifact_paths(hippo_save_dir, llm_model, embedding_model)
    docs, doc_metadata = build_doc_metadata(corpus_rows)

    q0_summary_path = dataset_dir / "q0_index_summary.json"
    q0_cost_path = dataset_dir / "q0_cost_summary.json"
    doc_meta_path = hippo_save_dir / "doc_metadata.json"

    if q0_ready(hippo_save_dir, llm_model, embedding_model) and not force_reindex:
        existing = {}
        if q0_summary_path.exists():
            try:
                existing = json.loads(q0_summary_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        if not doc_meta_path.exists():
            save_json(doc_meta_path, doc_metadata)
        existing_cost = {}
        if q0_cost_path.exists():
            try:
                existing_cost = json.loads(q0_cost_path.read_text(encoding="utf-8"))
            except Exception:
                existing_cost = {}
        existing["cache_reused"] = True
        existing["dataset_name"] = dataset_name
        existing["num_docs"] = len(docs)
        existing["cache_dir"] = str(hippo_save_dir)
        existing["working_dir"] = str(make_hipporag_working_dir(hippo_save_dir, llm_model, embedding_model))
        existing["artifact_paths"] = artifact_paths
        existing["q0_cost_summary_path"] = str(q0_cost_path)
        if existing_cost:
            existing["preprocessing_cost"] = existing_cost
        save_json(q0_summary_path, existing)
        return existing

    cfg = make_hipporag_config(
        save_dir=hippo_save_dir,
        llm_model=llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        force_index_from_scratch=force_reindex,
        corpus_len=len(corpus_rows),
    )
    hippo = HippoRAG(global_config=cfg)

    start = time.perf_counter()
    hippo.index(docs)
    hippo.prepare_retrieval_objects()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    graph_info = hippo.get_graph_info()
    preprocessing_cost = collect_hipporag_usage_cost(hippo, llm_model=llm_model, embedding_model=embedding_model)

    save_json(doc_meta_path, doc_metadata)
    save_json(q0_cost_path, preprocessing_cost)
    summary = {
        "dataset_name": dataset_name,
        "cache_reused": False,
        "num_docs": len(docs),
        "duration_ms": elapsed_ms,
        "graph_info": graph_info,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "cache_dir": str(hippo_save_dir),
        "working_dir": str(make_hipporag_working_dir(hippo_save_dir, llm_model, embedding_model)),
        "artifact_paths": artifact_paths,
        "q0_cost_summary_path": str(q0_cost_path),
        "preprocessing_cost": preprocessing_cost,
    }
    save_json(q0_summary_path, summary)
    return summary


def run_hipporag_retrieval_round(
    *,
    hippo: HippoRAG,
    query: str,
    doc_metadata: List[Dict[str, Any]],
    qa_top_k: int,
    top_session_limit: int,
) -> Dict[str, Any]:
    search_start = time.perf_counter()
    retrieval_results = hippo.retrieve(queries=[query])
    search_duration_ms = round((time.perf_counter() - search_start) * 1000.0, 3)

    retrieval_solution = retrieval_results[0]
    top_ranked_docs = [str(doc or "") for doc in retrieval_solution.docs[:top_session_limit]]
    top_ranked_scores = [
        float(score) for score in list(retrieval_solution.doc_scores[: len(top_ranked_docs)])
    ]
    top_ranked_doc_metas = resolve_doc_metas(top_ranked_docs, doc_metadata)
    top_session_ids = dedupe_preserve_order(
        meta.get("session_id", "") for meta in top_ranked_doc_metas
    )[:top_session_limit]
    prompt_doc_metas = top_ranked_doc_metas[: max(1, qa_top_k)]
    prompt_chunks = [render_context_block(meta, focus_query=query) for meta in prompt_doc_metas]
    return {
        "retrieval_solution": retrieval_solution,
        "top_ranked_docs": top_ranked_docs,
        "top_ranked_scores": top_ranked_scores,
        "top_ranked_doc_metas": top_ranked_doc_metas,
        "top_session_ids": top_session_ids,
        "prompt_doc_metas": prompt_doc_metas,
        "prompt_chunks": prompt_chunks,
        "search_duration_ms": search_duration_ms,
    }


def run_native_hipporag_qa_from_docs(
    *,
    hippo: HippoRAG,
    question: str,
    retrieved_docs: List[str],
    question_type: str = "",
    question_time: str = "",
) -> Dict[str, Any]:
    query_solution = QuerySolution(question=question, docs=list(retrieved_docs))
    response_start = time.perf_counter()
    qa_solutions, all_response_messages, all_metadata = hippo.qa([query_solution])
    response_duration_ms = round((time.perf_counter() - response_start) * 1000.0, 3)

    qa_solution = qa_solutions[0]
    raw_response = str(all_response_messages[0] or "")
    qa_metadata = dict(all_metadata[0] or {})
    parsed_answer = postprocess_answer(str(qa_solution.answer or ""))
    raw_answer = postprocess_answer(raw_response)
    answer = parsed_answer or raw_answer
    if answer_looks_like_instruction_fragment(answer) and raw_answer and raw_answer != answer:
        answer = raw_answer

    _, rendered_qa_messages, prompt_template_dataset = build_native_hipporag_qa_messages(
        hippo,
        question=question,
        retrieved_docs=list(retrieved_docs),
        question_type=question_type,
        question_time=question_time,
    )
    formatted_prompt = render_chat_messages(rendered_qa_messages)
    return {
        "answer": answer,
        "raw_response": raw_response,
        "raw_answer": str(qa_solution.answer or ""),
        "qa_metadata": qa_metadata,
        "response_duration_ms": response_duration_ms,
        "formatted_prompt": formatted_prompt,
        "formatted_prompt_messages": rendered_qa_messages,
        "prompt_template_dataset": prompt_template_dataset,
    }


def run_single_round_hipporag_query(
    *,
    hippo: HippoRAG,
    dataset_name: str,
    question_row: Dict[str, Any],
    doc_metadata: List[Dict[str, Any]],
    llm_model: str,
    qa_top_k: int,
    top_session_limit: int,
) -> Dict[str, Any]:
    question = str(question_row.get("question", "") or "")
    question_type = str(question_row.get("question_type", "") or "")

    round_data = run_hipporag_retrieval_round(
        hippo=hippo,
        query=question,
        doc_metadata=doc_metadata,
        qa_top_k=qa_top_k,
        top_session_limit=top_session_limit,
    )

    top_ranked_docs = list(round_data["top_ranked_docs"])
    top_ranked_scores = list(round_data["top_ranked_scores"])
    top_session_ids = list(round_data["top_session_ids"])
    qa_docs = top_ranked_docs[: max(1, qa_top_k)]
    qa_doc_metas = list(round_data["prompt_doc_metas"])

    qa_result = run_native_hipporag_qa_from_docs(
        hippo=hippo,
        question=question,
        retrieved_docs=qa_docs,
        question_type=question_type,
        question_time=str(question_row.get("question_time", "") or ""),
    )
    response_duration_ms = float(qa_result["response_duration_ms"])
    raw_response = str(qa_result["raw_response"])
    qa_metadata = dict(qa_result["qa_metadata"])
    answer = str(qa_result["answer"] or "")
    total_duration_ms = round(float(round_data["search_duration_ms"]) + response_duration_ms, 3)
    runtime_cost = collect_hipporag_usage_cost(hippo, llm_model=llm_model, embedding_model=hippo.global_config.embedding_model_name)
    rendered_qa_messages = list(qa_result["formatted_prompt_messages"])
    prompt_template_dataset = str(qa_result["prompt_template_dataset"])
    formatted_prompt = str(qa_result["formatted_prompt"])
    record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "question_id": str(question_row.get("question_id", "") or ""),
        "question": question,
        "answer": str(question_row.get("answer", "") or ""),
        "label": question_row.get("label", ""),
        "question_type": question_type,
        "question_time": str(question_row.get("question_time", "") or ""),
        "origin": question_row.get("origin", ""),
        "bin_s": preserve_or_infer_bin_s(question_row.get("bin_s", ""), dataset_name),
        "output": answer,
        "model": llm_model,
        "memory_system": "HippoRAG",
        "trial": 1,
        "retrieval_calls": 1,
        "retrieved_sessions": len(top_session_ids),
        "top_session_ids": top_session_ids,
        "react_trace": [
            {
                "step": 1,
                "action": "retrieve",
                "query": question,
                "top_session_ids": top_session_ids,
                "retrieved_session_count": len(top_session_ids),
            },
            {
                "step": 2,
                "action": "finish",
                "final_answer": answer,
            },
        ],
        "react_round_chunk_counts": [len(qa_docs)],
        "final_context_chunk_count": len(qa_docs),
        "final_context_chunk_alloc_per_turn": [len(qa_docs)],
        "formatted_prompt": formatted_prompt,
        "formatted_prompt_messages": rendered_qa_messages,
        "qa_prompt_template_dataset": prompt_template_dataset,
        "triples": [],
        "chunks": [render_context_block(meta, focus_query=question) for meta in qa_doc_metas],
        "summaries": [],
        "llm_qa_raw_response": raw_response,
        "llm_qa_raw_answer": str(qa_result["raw_answer"]),
        "llm_qa_metadata": qa_metadata,
        "hipporag_doc_scores": top_ranked_scores,
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": float(round_data["search_duration_ms"]),
        "total_duration_ms": total_duration_ms,
        "total_cost_usd": estimate_q1_runtime_cost_usd(runtime_cost, qa_metadata, llm_model),
    }

    memos_metrics = derive_memos_metrics(row=record, record=record)
    record["context_tokens"] = int(memos_metrics["context_tokens"])
    record["memos_stats"] = {
        "context_tokens": int(memos_metrics["context_tokens"]),
        "response_duration_ms": float(memos_metrics["response_duration_ms"]),
        "search_duration_ms": float(memos_metrics["search_duration_ms"]),
        "total_duration_ms": float(memos_metrics["total_duration_ms"]),
    }
    return record


def run_react_multihop_hipporag_query(
    *,
    hippo: HippoRAG,
    dataset_name: str,
    question_row: Dict[str, Any],
    doc_metadata: List[Dict[str, Any]],
    llm_model: str,
    qa_top_k: int,
    top_session_limit: int,
    react_max_steps: int,
    react_max_context_chunks: int,
    react_agent_max_tokens: int,
    react_agent_temperature: float,
) -> Dict[str, Any]:
    question = str(question_row.get("question", "") or "")
    question_type = str(question_row.get("question_type", "") or "")
    question_time = str(question_row.get("question_time", "") or "")

    ordered_sessions: List[str] = []
    seen_sessions: Set[str] = set()
    seen_chunk_keys: Set[str] = set()
    prior_queries: Set[str] = set()
    react_trace: List[Dict[str, Any]] = []
    round_chunks_for_prompt: List[List[str]] = []
    retrieval_actions = 0
    response_duration_ms = 0.0
    search_duration_ms = 0.0
    answer = ""
    formatted_prompt = ""
    formatted_prompt_messages: List[Dict[str, Any]] = []
    final_agent_raw_response = ""
    final_agent_answer = ""
    final_agent_metadata: Dict[str, Any] = {}
    qa_prompt_template_dataset = "react_multihop"
    last_doc_scores: List[float] = []
    final_chunks_for_prompt: List[str] = []
    chunk_alloc_per_turn: List[int] = []
    top_session_ids: List[str] = []

    aggregated_doc_best_scores: Dict[str, float] = {}
    aggregated_doc_metas: Dict[str, Dict[str, Any]] = {}
    aggregated_doc_first_seen: Dict[str, int] = {}
    doc_order_counter = 0

    def get_aggregated_docs_for_native_qa() -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        ranked = sorted(
            aggregated_doc_best_scores.items(),
            key=lambda item: (-float(item[1]), int(aggregated_doc_first_seen.get(item[0], 10**9))),
        )
        ordered_docs = [doc for doc, _ in ranked[: max(1, int(top_session_limit))]]
        ordered_scores = [float(score) for _, score in ranked[: len(ordered_docs)]]
        ordered_metas = [aggregated_doc_metas[doc] for doc in ordered_docs if doc in aggregated_doc_metas]
        return ordered_docs, ordered_scores, ordered_metas

    def finalize_with_native_qa() -> None:
        nonlocal answer
        nonlocal response_duration_ms
        nonlocal formatted_prompt
        nonlocal formatted_prompt_messages
        nonlocal final_agent_raw_response
        nonlocal final_agent_answer
        nonlocal final_agent_metadata
        nonlocal qa_prompt_template_dataset
        nonlocal last_doc_scores
        nonlocal final_chunks_for_prompt
        nonlocal chunk_alloc_per_turn
        nonlocal top_session_ids

        native_docs, native_scores, native_doc_metas = get_aggregated_docs_for_native_qa()
        if not native_docs:
            if not answer:
                answer = "Insufficient information from context."
            return

        qa_docs = native_docs[: max(1, int(qa_top_k))]
        qa_doc_metas = list(native_doc_metas[: len(qa_docs)])
        qa_result = run_native_hipporag_qa_from_docs(
            hippo=hippo,
            question=question,
            retrieved_docs=qa_docs,
            question_type=question_type,
            question_time=question_time,
        )
        response_duration_ms += float(qa_result["response_duration_ms"])
        answer = str(qa_result["answer"] or "")
        if answer_looks_like_instruction_fragment(answer) or is_meta_completion_answer(answer):
            answer = "Insufficient information from context."
        final_agent_raw_response = str(qa_result["raw_response"])
        final_agent_answer = str(qa_result["raw_answer"] or answer)
        final_agent_metadata = {
            **dict(qa_result["qa_metadata"]),
            "mode": "react_multihop",
            "react_max_steps": int(react_max_steps),
            "react_max_context_chunks": int(react_max_context_chunks),
            "finish_mode": "hipporag_native_qa",
        }
        formatted_prompt = str(qa_result["formatted_prompt"])
        formatted_prompt_messages = list(qa_result["formatted_prompt_messages"])
        qa_prompt_template_dataset = str(qa_result["prompt_template_dataset"])
        final_chunks_for_prompt = [render_context_block(meta, focus_query=question) for meta in qa_doc_metas]
        chunk_alloc_per_turn = [len(chunks or []) for chunks in round_chunks_for_prompt]
        top_session_ids = dedupe_preserve_order(
            meta.get("session_id", "") for meta in native_doc_metas
        )[: max(1, int(top_session_limit))]
        last_doc_scores = list(native_scores)

    for step_idx in range(1, max(2, int(react_max_steps or DEFAULT_REACT_MAX_STEPS)) + 1):
        current_chunks_for_prompt, _ = select_multihop_chunks_for_prompt(
            round_chunks_for_prompt,
            max(1, int(react_max_context_chunks or DEFAULT_REACT_MAX_CONTEXT_CHUNKS)),
        )
        formatted_prompt, formatted_prompt_messages = build_react_conversation_messages(
            question=question,
            question_time=question_time,
            question_type=question_type,
            turn=step_idx,
            max_turns=max(2, int(react_max_steps or DEFAULT_REACT_MAX_STEPS)),
            chunks=current_chunks_for_prompt,
            history_steps=react_trace,
            max_ctx_chunks=max(1, int(react_max_context_chunks or DEFAULT_REACT_MAX_CONTEXT_CHUNKS)),
        )

        llm_start = time.perf_counter()
        decision_raw, decision_metadata, decision_cache_hit = llm_infer_with_cache_flag(
            hippo.llm_model,
            formatted_prompt_messages,
            response_format={"type": "json_object"},
            temperature=float(react_agent_temperature),
            max_completion_tokens=max(128, int(react_agent_max_tokens)),
        )
        turn_llm_duration_ms = round((time.perf_counter() - llm_start) * 1000.0, 3)
        decision = parse_react_action(decision_raw, question)

        if decision["action"] == "finish":
            response_duration_ms += turn_llm_duration_ms
            final_agent_answer = str(decision.get("final_answer", "") or "")
            finalize_with_native_qa()
            react_trace.append(
                {
                    "step": step_idx,
                    "action": "finish",
                    "query": "",
                    "thought": str(decision.get("thought", "") or ""),
                    "llm_prompt_tokens_delta": int(decision_metadata.get("prompt_tokens", 0) or 0),
                    "llm_completion_tokens_delta": int(decision_metadata.get("completion_tokens", 0) or 0),
                    "turn_llm_duration_ms": turn_llm_duration_ms,
                    "cache_hit": bool(decision_cache_hit),
                    "final_answer": answer,
                    "note": "finish",
                }
            )
            break

        response_query = str(decision.get("query", "") or "").strip() or question
        search_duration_ms += turn_llm_duration_ms
        round_data = run_hipporag_retrieval_round(
            hippo=hippo,
            query=response_query,
            doc_metadata=doc_metadata,
            qa_top_k=qa_top_k,
            top_session_limit=top_session_limit,
        )
        search_duration_ms += float(round_data["search_duration_ms"])
        retrieval_actions += 1
        last_doc_scores = list(round_data["top_ranked_scores"])

        for doc, score, meta in zip(
            list(round_data["top_ranked_docs"]),
            list(round_data["top_ranked_scores"]),
            list(round_data["top_ranked_doc_metas"]),
        ):
            doc_text = str(doc or "")
            if not doc_text:
                continue
            doc_score = float(score or 0.0)
            prev_score = aggregated_doc_best_scores.get(doc_text)
            if prev_score is None or doc_score > prev_score:
                aggregated_doc_best_scores[doc_text] = doc_score
                if isinstance(meta, dict):
                    aggregated_doc_metas[doc_text] = dict(meta)
            if doc_text not in aggregated_doc_first_seen:
                aggregated_doc_first_seen[doc_text] = doc_order_counter
                doc_order_counter += 1

        new_sessions = 0
        for session_id in round_data["top_session_ids"]:
            sid = str(session_id or "").strip()
            if not sid or sid in seen_sessions:
                continue
            seen_sessions.add(sid)
            ordered_sessions.append(sid)
            new_sessions += 1

        new_unique_chunks = 0
        round_prompt_chunks: List[str] = []
        for chunk in list(round_data["prompt_chunks"]):
            text = str(chunk or "").strip()
            if not text:
                continue
            chunk_key = text[:512]
            round_prompt_chunks.append(text)
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)
                new_unique_chunks += 1

        round_chunks_for_prompt.append(round_prompt_chunks)
        normalized_query = " ".join(response_query.lower().split())
        repeated_query = normalized_query in prior_queries
        prior_queries.add(normalized_query)
        if new_sessions > 0 or new_unique_chunks > 0:
            note = "ok"
        elif repeated_query:
            note = "repeat_no_progress"
        else:
            note = "no_progress"
        react_trace.append(
            {
                "step": step_idx,
                "action": "retrieve",
                "query": response_query,
                "thought": str(decision.get("thought", "") or ""),
                "top_session_ids": list(round_data["top_session_ids"]),
                "retrieved_session_count": len(round_data["top_session_ids"]),
                "new_sessions": new_sessions,
                "chunks_added": int(new_unique_chunks),
                "llm_prompt_tokens_delta": int(decision_metadata.get("prompt_tokens", 0) or 0),
                "llm_completion_tokens_delta": int(decision_metadata.get("completion_tokens", 0) or 0),
                "turn_llm_duration_ms": turn_llm_duration_ms,
                "search_duration_ms": float(round_data["search_duration_ms"]),
                "cache_hit": bool(decision_cache_hit),
                "note": note,
            }
        )

    if not any(str(step.get("action", "")) == "finish" for step in react_trace):
        finalize_with_native_qa()
        if not answer:
            answer = "Insufficient information from context."
        react_trace.append(
            {
                "step": len(react_trace) + 1,
                "action": "finish",
                "query": "",
                "thought": "The retrieval budget ended before the agent produced a grounded final answer.",
                "llm_prompt_tokens_delta": 0,
                "llm_completion_tokens_delta": 0,
                "turn_llm_duration_ms": 0.0,
                "cache_hit": False,
                "final_answer": answer,
                "note": "budget_stop",
            }
        )

    if not final_chunks_for_prompt:
        final_chunks_for_prompt, chunk_alloc_per_turn = select_multihop_chunks_for_prompt(
            round_chunks_for_prompt,
            max(1, int(react_max_context_chunks or DEFAULT_REACT_MAX_CONTEXT_CHUNKS)),
        )
    runtime_cost = collect_hipporag_usage_cost(
        hippo,
        llm_model=llm_model,
        embedding_model=hippo.global_config.embedding_model_name,
    )
    total_duration_ms = round(float(search_duration_ms) + float(response_duration_ms), 3)
    if not top_session_ids:
        top_session_ids = ordered_sessions[: max(1, int(top_session_limit))]

    record = {
        "dataset_name": dataset_name,
        "question_id": str(question_row.get("question_id", "") or ""),
        "question": question,
        "answer": str(question_row.get("answer", "") or ""),
        "label": question_row.get("label", ""),
        "question_type": question_type,
        "question_time": question_time,
        "origin": question_row.get("origin", ""),
        "bin_s": preserve_or_infer_bin_s(question_row.get("bin_s", ""), dataset_name),
        "output": answer,
        "model": llm_model,
        "memory_system": "HippoRAG",
        "trial": 1,
        "retrieval_calls": int(retrieval_actions),
        "retrieved_sessions": len(top_session_ids),
        "top_session_ids": top_session_ids,
        "react_trace": react_trace,
        "react_round_chunk_counts": [len(chunks or []) for chunks in round_chunks_for_prompt],
        "final_context_chunk_count": len(final_chunks_for_prompt),
        "final_context_chunk_alloc_per_turn": chunk_alloc_per_turn,
        "formatted_prompt": formatted_prompt,
        "formatted_prompt_messages": formatted_prompt_messages,
        "qa_prompt_template_dataset": qa_prompt_template_dataset,
        "triples": [],
        "chunks": final_chunks_for_prompt,
        "summaries": [],
        "llm_qa_raw_response": final_agent_raw_response,
        "llm_qa_raw_answer": final_agent_answer or answer,
        "llm_qa_metadata": {
            **final_agent_metadata,
            "mode": str(final_agent_metadata.get("mode", "react_multihop")),
            "react_max_steps": int(react_max_steps),
            "react_max_context_chunks": int(react_max_context_chunks),
        },
        "hipporag_doc_scores": last_doc_scores,
        "response_duration_ms": round(float(response_duration_ms), 3),
        "search_duration_ms": round(float(search_duration_ms), 3),
        "total_duration_ms": total_duration_ms,
        "total_cost_usd": estimate_q1_runtime_cost_usd(runtime_cost, final_agent_metadata, llm_model),
    }

    memos_metrics = derive_memos_metrics(row=record, record=record)
    record["context_tokens"] = int(memos_metrics["context_tokens"])
    record["memos_stats"] = {
        "context_tokens": int(memos_metrics["context_tokens"]),
        "response_duration_ms": float(memos_metrics["response_duration_ms"]),
        "search_duration_ms": float(memos_metrics["search_duration_ms"]),
        "total_duration_ms": float(memos_metrics["total_duration_ms"]),
    }
    return record


def query_with_hipporag(
    *,
    dataset_name: str,
    question_row: Dict[str, Any],
    doc_metadata: List[Dict[str, Any]],
    results_root: Path,
    llm_model: str,
    embedding_model: str,
    qa_top_k: int,
    top_session_limit: int,
    cache_llm_model: str | None = None,
    enable_react_multihop: bool = False,
    react_max_steps: int = DEFAULT_REACT_MAX_STEPS,
    react_max_context_chunks: int = DEFAULT_REACT_MAX_CONTEXT_CHUNKS,
    react_agent_max_tokens: int = DEFAULT_REACT_AGENT_MAX_TOKENS,
    react_agent_temperature: float = DEFAULT_REACT_AGENT_TEMPERATURE,
) -> Dict[str, Any]:
    cache_llm_model = str(cache_llm_model or llm_model)
    hippo_save_dir = make_hipporag_save_dir(results_root, dataset_name)
    cfg = make_hipporag_config(
        save_dir=hippo_save_dir,
        llm_model=cache_llm_model,
        embedding_model=embedding_model,
        qa_top_k=qa_top_k,
        force_index_from_scratch=False,
        corpus_len=len(doc_metadata),
    )
    hippo = HippoRAG(global_config=cfg)
    hippo.prepare_retrieval_objects()
    if llm_model != cache_llm_model:
        hippo.llm_model = make_hipporag_qa_llm(cfg, llm_model)
    if enable_react_multihop:
        return run_react_multihop_hipporag_query(
            hippo=hippo,
            dataset_name=dataset_name,
            question_row=question_row,
            doc_metadata=doc_metadata,
            llm_model=llm_model,
            qa_top_k=qa_top_k,
            top_session_limit=top_session_limit,
            react_max_steps=react_max_steps,
            react_max_context_chunks=react_max_context_chunks,
            react_agent_max_tokens=react_agent_max_tokens,
            react_agent_temperature=react_agent_temperature,
        )
    return run_single_round_hipporag_query(
        hippo=hippo,
        dataset_name=dataset_name,
        question_row=question_row,
        doc_metadata=doc_metadata,
        llm_model=llm_model,
        qa_top_k=qa_top_k,
        top_session_limit=top_session_limit,
    )


async def judge_record(
    evaluator: LLMEvaluator,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    bundle = await evaluator.evaluate_with_llm_bundle(
        question=str(record.get("question", "") or ""),
        answer=str(record.get("answer", "") or ""),
        response=str(record.get("output", "") or ""),
        question_type=str(record.get("question_type", "") or ""),
    )
    record["llm_judge"] = bool(bundle["majority_label"])
    record["success"] = bool(bundle["majority_label"])
    record["llm_judgments"] = dict(bundle["judgments"])
    return record


def ensure_parent_trace(trace_path: Path) -> csv.DictWriter:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = trace_path.exists() and trace_path.stat().st_size > 0
    f = trace_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
    if not csv_exists:
        writer.writeheader()
    writer._codex_file_handle = f  # type: ignore[attr-defined]
    return writer


def close_trace_writer(writer: csv.DictWriter) -> None:
    handle = getattr(writer, "_codex_file_handle", None)
    if handle is not None:
        handle.close()


def main() -> None:
    args = parse_args()
    _load_runtime_dependencies()

    data_root = Path(args.data_root).resolve()
    dataset_list_path = Path(args.dataset_list).resolve()
    results_root = Path(args.results_root).resolve()
    judge_config_path = Path(args.judge_config).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    maybe_warn_noncanonical_inputs(data_root=data_root, dataset_list_path=dataset_list_path)

    if "GPT_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing GPT_API_KEY/OPENAI_API_KEY in the environment.")
    if "GPT_BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_BASE_URL"]
    if "GPT_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_API_KEY"]
    if "OPENAI_BASE_URL" in os.environ and "GPT_BASE_URL" not in os.environ:
        os.environ["GPT_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
    if "OPENAI_API_KEY" in os.environ and "GPT_API_KEY" not in os.environ:
        os.environ["GPT_API_KEY"] = os.environ["OPENAI_API_KEY"]

    dataset_names = [
        line.strip()
        for line in dataset_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit and args.limit > 0:
        dataset_names = dataset_names[: args.limit]
    if not dataset_names:
        raise RuntimeError(f"No dataset names found in {dataset_list_path}")
    if len(dataset_names) != len(set(dataset_names)):
        raise RuntimeError(f"Duplicate dataset names found in {dataset_list_path}")
    for dataset_name in dataset_names:
        validate_dataset_name(dataset_name)

    trace_path = results_root / f"trace_{dataset_list_path.stem}_{args.mode}.csv"
    checkpoint_path = results_root / f"checkpoint_{dataset_list_path.stem}_{args.mode}.json"
    run_summary_path = results_root / f"summary_{dataset_list_path.stem}_{args.mode}.json"

    checkpoint: Dict[str, Any]
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            checkpoint = {"completed": {}, "failed": {}, "updated_at": None}
    else:
        checkpoint = {"completed": {}, "failed": {}, "updated_at": None}
    checkpoint.setdefault("completed", {})
    checkpoint.setdefault("failed", {})

    clean_cfg = normalize_clean_config(Config.parse(judge_config_path))
    validate_clean_contract(clean_cfg)
    llm_evaluator = LLMEvaluator(clean_cfg, "", dataset_list_path.stem)

    trace_writer = ensure_parent_trace(trace_path)
    completed = checkpoint.get("completed", {})
    failed = checkpoint.get("failed", {})

    started_at = time.time()
    for idx, dataset_name in enumerate(dataset_names, start=1):
        print(f"[{idx}/{len(dataset_names)}] {dataset_name}")
        dataset_data_dir = data_root / dataset_name
        dataset_results_dir = make_dataset_dir(results_root, dataset_name)
        dataset_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            corpus_rows = load_corpus(dataset_data_dir / "Corpus.json")
            question_row = load_question(dataset_data_dir / "Question.json")

            if args.mode in {"q0", "q0q1"}:
                q0_summary = run_q0_for_dataset(
                    dataset_name=dataset_name,
                    corpus_rows=corpus_rows,
                    results_root=results_root,
                    llm_model=args.llm_model,
                    embedding_model=args.embedding_model,
                    qa_top_k=args.qa_top_k,
                    force_reindex=bool(args.force_reindex),
                )
                print(
                    f"  q0 docs={q0_summary.get('num_docs', 0)} cache_reused={q0_summary.get('cache_reused', False)}"
                )

            if args.mode == "q0":
                completed[dataset_name] = {
                    "mode": "q0",
                    "dataset_name": dataset_name,
                    "results_dir": str(dataset_results_dir),
                }
                failed.pop(dataset_name, None)
                checkpoint["updated_at"] = time.time()
                save_json(checkpoint_path, checkpoint)
                continue

            hippo_doc_meta = load_doc_metadata(make_hipporag_save_dir(results_root, dataset_name) / "doc_metadata.json")
            record = query_with_hipporag(
                dataset_name=dataset_name,
                question_row=question_row,
                doc_metadata=hippo_doc_meta,
                results_root=results_root,
                llm_model=args.llm_model,
                embedding_model=args.embedding_model,
                qa_top_k=args.qa_top_k,
                top_session_limit=args.top_session_limit,
                enable_react_multihop=bool(args.enable_react_multihop),
                react_max_steps=int(args.react_max_steps),
                react_max_context_chunks=int(args.react_max_context_chunks),
                react_agent_max_tokens=int(args.react_agent_max_tokens),
                react_agent_temperature=float(args.react_agent_temperature),
            )
            record = validate_record_contract(
                record,
                dataset_name=dataset_name,
                expected_judge_runs=clean_cfg.evaluation.eval_num_runs,
                require_judgments=False,
            )
            record = asyncio.run(judge_record(llm_evaluator, record))
            record = validate_record_contract(
                record,
                dataset_name=dataset_name,
                expected_judge_runs=clean_cfg.evaluation.eval_num_runs,
                require_judgments=True,
            )

            results_dir = dataset_results_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / "results.json"
            metrics_path = results_dir / "metrics.json"
            save_json(results_path, [sanitize_trace_record_for_export(record)])

            metrics = asyncio.run(Evaluator(str(results_path), dataset_name, clean_cfg).evaluate())
            save_json(metrics_path, metrics)
            if not args.keep_answer_in_results:
                redact_answer_field_in_results(results_path)

            trace_row = build_trace_row(dataset_name, record)
            trace_writer.writerow(trace_row)
            getattr(trace_writer, "_codex_file_handle").flush()  # type: ignore[attr-defined]

            completed[dataset_name] = {
                "mode": args.mode,
                "dataset_name": dataset_name,
                "results_path": str(results_path),
                "metrics_path": str(metrics_path),
                "success": bool(record.get("success", False)),
                "llm_judge": bool(record.get("llm_judge", False)),
            }
            failed.pop(dataset_name, None)
            checkpoint["updated_at"] = time.time()
            save_json(checkpoint_path, checkpoint)
            print(
                f"  q1 judge={record.get('llm_judge')} output={str(record.get('output', '') or '')[:120]}"
            )
        except Exception as exc:
            failed[dataset_name] = {"error": str(exc), "dataset_name": dataset_name}
            checkpoint["updated_at"] = time.time()
            save_json(checkpoint_path, checkpoint)
            print(f"  failed: {exc}")

    finished_at = time.time()
    close_trace_writer(trace_writer)

    run_summary = {
        "dataset_list": str(dataset_list_path),
        "data_root": str(data_root),
        "mode": args.mode,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "judge_config": str(judge_config_path),
        "judge_model": clean_cfg.evaluation.eval_model,
        "judge_prompt_style": clean_cfg.evaluation.eval_prompt_style,
        "judge_num_runs": clean_cfg.evaluation.eval_num_runs,
        "hipporag_runtime": get_hipporag_runtime_knobs(
            qa_top_k=args.qa_top_k,
            top_session_limit=args.top_session_limit,
        ),
        "canonical_data_root": str(DEFAULT_DATA_ROOT),
        "canonical_list_root": str(DEFAULT_LIST_ROOT),
        "results_root": str(results_root),
        "trace_path": str(trace_path),
        "checkpoint_path": str(checkpoint_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": round(finished_at - started_at, 3),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_datasets": sorted(completed.keys()),
        "failed_datasets": failed,
    }
    save_json(run_summary_path, run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
