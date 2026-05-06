#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List

from probe_dataset_item import (
    TRACE_FIELDS,
    OpenAICompatRunner,
    UsageTotals,
    build_trace_row,
    ensure_openai_env_aliases,
    ensure_trace_writer,
    close_trace_writer,
    maybe_load_env_file,
    _resolve_env_value,
    _resolve_optional_json_object,
    run_probe,
    sanitize_record_for_export,
    save_json,
)


EXPECTED_EVAL_MODEL = "gpt-4o-mini"
EXPECTED_EVAL_PROMPT_STYLE = "memos_json"
EXPECTED_EVAL_NUM_RUNS = 3

MEMOS_LOCOMO_PROMPT = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:\n"
    "    (1) a question (posed by one user to another user),\n"
    "    (2) a 'gold' (ground truth) answer,\n"
    "    (3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced topic.\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references "
    '(like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, '
    'it should be counted as CORRECT. Even if the format differs, consider it CORRECT if it is the same date.\n\n'
    "Now it's time for the real question:\n"
    "Question: {}\n"
    "Gold answer: {}\n"
    "Generated answer: {}\n\n"
    "First, provide a short explanation of your reasoning, then finish with CORRECT or WRONG.\n"
    "Do NOT include both CORRECT and WRONG in your response.\n\n"
    'Just return the label CORRECT or WRONG in a json format with the key as "label".'
)

MEMOS_LME_PROMPT = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:\n"
    "    (1) a question (posed by one user to another user),\n"
    "    (2) a 'gold' (ground truth) answer,\n"
    "    (3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced topic.\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references "
    '(like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, '
    'it should be counted as CORRECT. Even if the format differs, consider it CORRECT if it is the same date.\n\n'
    "Now it's time for the real question:\n"
    "Question: {}\n"
    "Gold answer: {}\n"
    "Generated answer: {}\n\n"
    "First, provide a short explanation of your reasoning, then finish with CORRECT or WRONG.\n"
    "Do NOT include both CORRECT and WRONG in your response.\n\n"
    'Just return the label CORRECT or WRONG in a json format with the key as "label".'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenClaw document memory on a benchmark dataset list."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-list", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--chat-model", default="gpt-5-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-max-tokens", type=int, default=512)
    parser.add_argument("--eval-model", default=EXPECTED_EVAL_MODEL)
    parser.add_argument("--eval-prompt-style", default=EXPECTED_EVAL_PROMPT_STYLE)
    parser.add_argument("--eval-num-runs", type=int, default=EXPECTED_EVAL_NUM_RUNS)
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
    parser.add_argument("--memory-agent-profile", default="benchmark_tuned")
    parser.add_argument("--max-agent-steps", type=int, default=6)
    parser.add_argument(
        "--force-min-memory-searches",
        type=int,
        default=0,
        help=(
            "Force each item to make at least this many memory_search tool calls "
            "before the agent may finish. Use 1 for the qwen235b forced-retrieve rerun."
        ),
    )
    parser.add_argument("--chat-base-url", default="")
    parser.add_argument("--chat-base-url-env", default="")
    parser.add_argument("--chat-api-key-env", default="")
    parser.add_argument("--chat-extra-body-json", default="")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-base-url-env", default="")
    parser.add_argument("--embedding-api-key-env", default="")
    parser.add_argument("--eval-base-url", default="")
    parser.add_argument("--eval-base-url-env", default="")
    parser.add_argument("--eval-api-key-env", default="")
    parser.add_argument("--openclaw-repo-root", default="")
    parser.add_argument("--openclaw-node-bin", default="")
    parser.add_argument("--excerpt-before-lines", type=int, default=2)
    parser.add_argument("--excerpt-after-lines", type=int, default=10)
    parser.add_argument("--enable-mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--enable-temporal-decay", action="store_true")
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--reuse-existing-q0", action="store_true")
    parser.add_argument("--keep-answer-in-results", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--item-max-attempts", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=float, default=20.0)
    parser.add_argument("--item-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--cleanup-q0-after-item", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def validate_eval_contract(args: argparse.Namespace) -> None:
    errors: List[str] = []
    if str(args.eval_model or "").strip() != EXPECTED_EVAL_MODEL:
        errors.append(f"eval_model must be {EXPECTED_EVAL_MODEL}")
    if str(args.eval_prompt_style or "").strip().lower() != EXPECTED_EVAL_PROMPT_STYLE:
        errors.append(f"eval_prompt_style must be {EXPECTED_EVAL_PROMPT_STYLE}")
    if int(args.eval_num_runs or 0) != EXPECTED_EVAL_NUM_RUNS:
        errors.append(f"eval_num_runs must be {EXPECTED_EVAL_NUM_RUNS}")
    if errors:
        raise RuntimeError("clean_contract_violation: " + "; ".join(errors))


def load_dataset_names(path: Path, limit: int) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        names = names[:limit]
    return names


def build_eval_messages(
    *,
    question: str,
    answer: str,
    response: str,
    question_type: str,
) -> List[Dict[str, str]]:
    template = MEMOS_LOCOMO_PROMPT if str(question_type or "").strip().lower().startswith("locomo-") else MEMOS_LME_PROMPT
    return [
        {
            "role": "system",
            "content": "You are an expert grader that determines if answers to questions match a gold standard answer.",
        },
        {
            "role": "user",
            "content": template.format(question, answer, response),
        },
    ]


def parse_memos_json_label(text: str) -> bool:
    raw_text = str(text or "").strip()
    match = re.search(r'["\']label["\']\s*:\s*["\']([^"\']+)["\']', raw_text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip().lower()
    else:
        upper = raw_text.upper()
        has_correct = bool(re.search(r"\bCORRECT\b", upper))
        has_wrong = bool(re.search(r"\bWRONG\b", upper))
        if has_correct and not has_wrong:
            value = "correct"
        elif has_wrong and not has_correct:
            value = "wrong"
        else:
            raise ValueError("memos_judge_parse_failed")
    if value == "correct":
        return True
    if value == "wrong":
        return False
    raise ValueError("memos_judge_unknown_label")


async def evaluate_record(
    *,
    eval_runner: OpenAICompatRunner,
    record: Dict[str, Any],
    num_runs: int,
) -> Dict[str, Any]:
    judgments: Dict[str, bool] = {}
    raw_responses: List[str] = []
    usage = UsageTotals()
    for idx in range(1, num_runs + 1):
        messages = build_eval_messages(
            question=str(record.get("question", "") or ""),
            answer=str(record.get("answer", "") or ""),
            response=str(record.get("output", "") or ""),
            question_type=str(record.get("question_type", "") or ""),
        )
        text, run_usage = await eval_runner.chat(messages, max_tokens=160, temperature=0.0)
        raw_responses.append(text)
        label = parse_memos_json_label(text)
        judgments[f"judgment_{idx}"] = label
        usage.prompt_tokens += run_usage.prompt_tokens
        usage.completion_tokens += run_usage.completion_tokens
        usage.total_tokens += run_usage.total_tokens
        usage.total_cost_usd += run_usage.total_cost_usd
        usage.calls += run_usage.calls
    num_correct = sum(1 for value in judgments.values() if value)
    majority = num_correct >= ((num_runs // 2) + 1)
    return {
        "llm_judgments": judgments,
        "llm_judge": majority,
        "success": majority,
        "llm_eval_metadata": {
            "eval_model": eval_runner.model,
            "eval_prompt_style": EXPECTED_EVAL_PROMPT_STYLE,
            "eval_num_runs": num_runs,
            "eval_prompt_tokens": usage.prompt_tokens,
            "eval_completion_tokens": usage.completion_tokens,
            "eval_total_tokens": usage.total_tokens,
            "eval_calls": usage.calls,
            "eval_total_cost_usd": round(usage.total_cost_usd, 6),
            "eval_raw_responses": raw_responses,
        },
    }


def overwrite_trace_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def redact_record_for_export(record: Dict[str, Any], *, keep_answer: bool) -> Dict[str, Any]:
    exported = json.loads(json.dumps(record, ensure_ascii=False))
    if not keep_answer:
        exported["answer"] = ""
        if "label" in exported:
            exported["label"] = ""
    return sanitize_record_for_export(exported)


def cleanup_q0_assets(output_dir: Path) -> None:
    for child in ("workspace", "openclaw_state"):
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for child in ("q0_index_summary.json", "q0_cost_summary.json"):
        path = output_dir / child
        if path.exists():
            path.unlink()


def write_run_progress(
    *,
    results_root: Path,
    items_requested: int,
    completed: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    last_event: Dict[str, Any],
) -> None:
    save_json(
        results_root / "run_progress.json",
        {
            "results_root": str(results_root),
            "items_requested": items_requested,
            "items_completed": len(completed),
            "items_failed": len(failures),
            "last_event": last_event,
            "recent_failures": failures[-20:],
        },
    )


def build_probe_namespace(args: argparse.Namespace, *, dataset_name: str, output_dir: Path, corpus_json: Path, question_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        corpus_json=str(corpus_json),
        question_json=str(question_json),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        env_file=args.env_file,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        chat_max_tokens=args.chat_max_tokens,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        top_session_limit=args.top_session_limit,
        candidate_multiplier=args.candidate_multiplier,
        vector_weight=args.vector_weight,
        text_weight=args.text_weight,
        min_score=args.min_score,
        sources=args.sources,
        memory_backend=args.memory_backend,
        agent_mode=args.agent_mode,
        memory_agent_profile=args.memory_agent_profile,
        max_agent_steps=args.max_agent_steps,
        force_min_memory_searches=args.force_min_memory_searches,
        chat_base_url=args.chat_base_url,
        chat_base_url_env=args.chat_base_url_env,
        chat_api_key_env=args.chat_api_key_env,
        chat_extra_body_json=args.chat_extra_body_json,
        embedding_base_url=args.embedding_base_url,
        embedding_base_url_env=args.embedding_base_url_env,
        embedding_api_key_env=args.embedding_api_key_env,
        openclaw_repo_root=args.openclaw_repo_root,
        openclaw_node_bin=args.openclaw_node_bin,
        excerpt_before_lines=args.excerpt_before_lines,
        excerpt_after_lines=args.excerpt_after_lines,
        enable_mmr=args.enable_mmr,
        mmr_lambda=args.mmr_lambda,
        enable_temporal_decay=args.enable_temporal_decay,
        half_life_days=args.half_life_days,
        skip_answer=args.skip_answer,
        reuse_existing_q0=args.reuse_existing_q0,
        force=args.force,
    )


async def process_one(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    data_root: Path,
    results_root: Path,
    trace_writer: csv.DictWriter,
    trace_lock: asyncio.Lock,
    eval_runner: OpenAICompatRunner | None,
) -> Dict[str, Any]:
    corpus_json = data_root / dataset_name / "Corpus.json"
    question_json = data_root / dataset_name / "Question.json"
    if not corpus_json.exists():
        raise FileNotFoundError(f"missing Corpus.json for {dataset_name}: {corpus_json}")
    if not question_json.exists():
        raise FileNotFoundError(f"missing Question.json for {dataset_name}: {question_json}")

    output_dir = results_root / "derived" / dataset_name
    probe_args = build_probe_namespace(
        args,
        dataset_name=dataset_name,
        output_dir=output_dir,
        corpus_json=corpus_json,
        question_json=question_json,
    )
    try:
        summary = await run_probe(probe_args)
        results_path = Path(summary["results_path"])
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"invalid results.json for {dataset_name}")
        record = dict(rows[0])

        if not args.skip_answer and not args.disable_eval and eval_runner is not None:
            eval_bundle = await evaluate_record(
                eval_runner=eval_runner,
                record=record,
                num_runs=args.eval_num_runs,
            )
            record.update(eval_bundle)
            record["total_cost_usd"] = round(
                float(record.get("total_cost_usd", 0.0) or 0.0)
                + float(eval_bundle["llm_eval_metadata"]["eval_total_cost_usd"]),
                6,
            )
        else:
            record["llm_judgments"] = {}
            record["llm_judge"] = ""
            record["success"] = ""
            record["llm_eval_metadata"] = {
                "eval_model": "",
                "eval_prompt_style": EXPECTED_EVAL_PROMPT_STYLE,
                "eval_num_runs": 0,
                "eval_prompt_tokens": 0,
                "eval_completion_tokens": 0,
                "eval_total_tokens": 0,
                "eval_calls": 0,
                "eval_total_cost_usd": 0.0,
                "eval_raw_responses": [],
            }

        full_record_for_metrics = dict(record)
        exported_record = redact_record_for_export(record, keep_answer=args.keep_answer_in_results)
        exported_record["evaluation"] = {
            "eval_model": args.eval_model if not args.disable_eval else "",
            "eval_prompt_style": args.eval_prompt_style,
            "eval_num_runs": args.eval_num_runs if not args.disable_eval else 0,
        }

        trace_row = build_trace_row(dataset_name, full_record_for_metrics)
        trace_row["trace_json"] = json.dumps(exported_record, ensure_ascii=False)
        async with trace_lock:
            trace_writer.writerow(trace_row)
            getattr(trace_writer, "_file_handle").flush()  # type: ignore[attr-defined]

        save_json(results_path, [exported_record])
        overwrite_trace_csv(output_dir / "trace_probe.csv", trace_row)
        return {
            "dataset_name": dataset_name,
            "results_path": str(results_path),
            "success": bool(full_record_for_metrics.get("success")) if full_record_for_metrics.get("success") != "" else None,
            "top_session_ids": list(full_record_for_metrics.get("top_session_ids", []) or []),
        }
    finally:
        if args.cleanup_q0_after_item:
            cleanup_q0_assets(output_dir)


async def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.disable_eval:
        validate_eval_contract(args)

    data_root = Path(args.data_root).resolve()
    dataset_list = Path(args.dataset_list).resolve()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    dataset_names = load_dataset_names(dataset_list, args.limit)

    trace_path = results_root / "trace_q1_full.csv"
    if args.force and trace_path.exists():
        trace_path.unlink()
    trace_writer = ensure_trace_writer(trace_path)
    eval_base_url = _resolve_env_value(
        explicit=getattr(args, "eval_base_url", ""),
        env_name=getattr(args, "eval_base_url_env", ""),
        fallback_envs=("OPENAI_BASE_URL", "GPT_BASE_URL"),
    )
    eval_api_key = _resolve_env_value(
        explicit="",
        env_name=getattr(args, "eval_api_key_env", ""),
        fallback_envs=("OPENAI_API_KEY", "GPT_API_KEY"),
    )
    _resolve_optional_json_object(getattr(args, "chat_extra_body_json", ""), field_name="chat_extra_body_json")
    eval_runner = None if args.disable_eval or args.skip_answer else OpenAICompatRunner(
        model=args.eval_model,
        base_url=eval_base_url,
        api_key=eval_api_key,
    )

    completed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    trace_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, int(args.parallelism or 1)))

    async def worker(idx: int, dataset_name: str) -> None:
        async with semaphore:
            max_attempts = max(1, int(getattr(args, "item_max_attempts", 1) or 1))
            retry_delay_seconds = max(0.0, float(getattr(args, "retry_delay_seconds", 20.0) or 0.0))
            last_failure: Dict[str, Any] | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    item_timeout_seconds = max(
                        0.0, float(getattr(args, "item_timeout_seconds", 0.0) or 0.0)
                    )
                    process_coro = process_one(
                        args=args,
                        dataset_name=dataset_name,
                        data_root=data_root,
                        results_root=results_root,
                        trace_writer=trace_writer,
                        trace_lock=trace_lock,
                        eval_runner=eval_runner,
                    )
                    if item_timeout_seconds > 0:
                        item_summary = await asyncio.wait_for(
                            process_coro,
                            timeout=item_timeout_seconds,
                        )
                    else:
                        item_summary = await process_coro
                    item_summary["index"] = idx
                    item_summary["attempts"] = attempt
                    async with progress_lock:
                        completed.append(item_summary)
                        write_run_progress(
                            results_root=results_root,
                            items_requested=len(dataset_names),
                            completed=completed,
                            failures=failures,
                            last_event={"status": "completed", **item_summary},
                        )
                    return
                except Exception as exc:
                    item_timeout_seconds = max(
                        0.0, float(getattr(args, "item_timeout_seconds", 0.0) or 0.0)
                    )
                    error_message = (
                        f"item_timeout_after_{int(item_timeout_seconds)}s"
                        if isinstance(exc, asyncio.TimeoutError) and item_timeout_seconds > 0
                        else str(exc)
                    )
                    last_failure = {
                        "index": idx,
                        "dataset_name": dataset_name,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error": error_message,
                        "traceback": traceback.format_exc(),
                    }
                    if attempt < max_attempts:
                        async with progress_lock:
                            write_run_progress(
                                results_root=results_root,
                                items_requested=len(dataset_names),
                                completed=completed,
                                failures=failures,
                                last_event={"status": "retrying", **last_failure},
                            )
                        await asyncio.sleep(retry_delay_seconds * attempt)
                        continue
                    async with progress_lock:
                        failures.append(last_failure)
                        write_run_progress(
                            results_root=results_root,
                            items_requested=len(dataset_names),
                            completed=completed,
                            failures=failures,
                            last_event={"status": "failed", **last_failure},
                        )
                    if not args.continue_on_error:
                        raise
                    return

    try:
        await asyncio.gather(
            *(worker(idx, dataset_name) for idx, dataset_name in enumerate(dataset_names, start=1))
        )
    finally:
        close_trace_writer(trace_writer)
        if eval_runner is not None:
            await eval_runner.aclose()

    run_summary = {
        "data_root": str(data_root),
        "dataset_list": str(dataset_list),
        "results_root": str(results_root),
        "items_requested": len(dataset_names),
        "items_completed": len(completed),
        "items_failed": len(failures),
        "chat_model": args.chat_model,
        "embedding_model": args.embedding_model,
        "sources": args.sources,
        "memory_agent_profile": args.memory_agent_profile,
        "force_min_memory_searches": max(0, int(args.force_min_memory_searches or 0)),
        "openclaw_repo_root": args.openclaw_repo_root,
        "parallelism": args.parallelism,
        "item_max_attempts": args.item_max_attempts,
        "retry_delay_seconds": args.retry_delay_seconds,
        "item_timeout_seconds": args.item_timeout_seconds,
        "eval_model": "" if args.disable_eval else args.eval_model,
        "eval_prompt_style": args.eval_prompt_style,
        "eval_num_runs": 0 if args.disable_eval else args.eval_num_runs,
        "completed": completed,
        "failures": failures,
    }
    save_json(results_root / "run_summary.json", run_summary)
    return run_summary


def main() -> None:
    args = parse_args()
    if args.env_file:
        maybe_load_env_file(Path(args.env_file))
    ensure_openai_env_aliases()
    summary = asyncio.run(run_all(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
