#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List
import random

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from memos_stats import derive_memos_metrics, sanitize_trace_record_for_export

def _normalize_text(text: str) -> List[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return []
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def _f1_score(pred: str, gold: str) -> float:
    from collections import Counter
    pred_tokens = _normalize_text(pred)
    gold_tokens = _normalize_text(gold)
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


def _list_items(data_root: Path) -> List[str]:
    items = []
    for p in data_root.rglob("lm_*"):
        if p.is_dir():
            items.append(str(p.relative_to(data_root)))
    return sorted(items)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "failed": {}, "updated_at": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": {}, "failed": {}, "updated_at": None}


def _save_checkpoint(path: Path, checkpoint: Dict[str, Any]) -> None:
    checkpoint["updated_at"] = time.time()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _preserve_graph_copy(
    graph_path: Path,
    dataset_name: str,
    graph_file: str,
    preserve_graph_root: str,
) -> bool:
    if not preserve_graph_root:
        return True
    if not graph_path.exists() or graph_path.stat().st_size <= 0:
        return False
    dst_path = Path(preserve_graph_root) / dataset_name / graph_file
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_path, dst_path)
    return dst_path.exists() and dst_path.stat().st_size > 0


def _run_one(args: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = args["dataset_name"]
    repo_root = args["repo_root"]
    python_bin = args["python_bin"]
    config_path = args["config_path"]
    data_root_override = args.get("data_root")
    root_prefix = args["root_prefix"]
    log_dir = Path(args["log_dir"])
    max_retries = int(args.get("max_retries", 0))
    retry_backoff = float(args.get("retry_backoff", 5))
    timeout_sec = args.get("timeout_sec")
    query_mode = int(args.get("query", 1))
    graph_file = str(args.get("graph_file", "dynamic_memory_graph.pkl"))
    require_graph_nonempty = bool(args.get("require_graph_nonempty", False))
    preserve_graph_root = str(args.get("preserve_graph_root", "") or "")

    root_name = f"{root_prefix}/{dataset_name}"
    effective_config_path = config_path
    tmp_config_path = None
    if data_root_override:
        try:
            base_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            if isinstance(base_cfg, dict):
                base_cfg["data_root"] = str(data_root_override)
                safe_name = dataset_name.replace("/", "__")
                tmp_config_path = log_dir / f".runner_cfg_{safe_name}_{int(time.time() * 1000)}_{os.getpid()}.yaml"
                tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_config_path.write_text(
                    yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=False),
                    encoding="utf-8",
                )
                effective_config_path = str(tmp_config_path)
        except Exception:
            # Fall back to original config path if temporary rewrite fails.
            effective_config_path = config_path
    cmd = [
        python_bin,
        "main.py",
        "-opt",
        effective_config_path,
        "-dataset_name",
        dataset_name,
        "-root",
        root_name,
        "-query",
        str(query_mode),
    ]

    results_path = Path(repo_root) / "results" / root_name / "results" / "results.json"
    graph_path = Path(repo_root) / "results" / root_name / graph_file
    last_returncode = None
    start_ts = None
    end_ts = None
    log_file = None
    success = False
    attempts = 0

    try:
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            log_file = log_dir / f"{dataset_name.replace('/', '_')}_try{attempts}.log"
            if query_mode == 1 and require_graph_nonempty:
                if not _is_graph_nonempty(graph_path):
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(
                            f"[runner] graph integrity check failed before query: {graph_path}\n"
                        )
                    last_returncode = 99
                    start_ts = time.time()
                    end_ts = start_ts
                    if attempt < max_retries:
                        time.sleep(retry_backoff)
                        continue
                    break

            start_ts = time.time()
            try:
                with log_file.open("w", encoding="utf-8") as f:
                    child_env = os.environ.copy()
                    # Query-time action/finish parse failures must surface as item failures,
                    # not be serialized into results and sent to the judge.
                    child_env["LICOMEMORY_FAIL_FAST_QUERY_ERRORS"] = "1"
                    proc = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        stdout=f,
                        stderr=f,
                        text=True,
                        timeout=timeout_sec,
                        env=child_env,
                    )
                end_ts = time.time()
                last_returncode = proc.returncode
            except subprocess.TimeoutExpired:
                end_ts = time.time()
                last_returncode = 124
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(f"\n[runner] Timeout after {timeout_sec}s\n")

            if last_returncode == 0:
                if query_mode == 1 and _wait_for_results(results_path):
                    success = True
                    break
                if query_mode == 0 and _wait_for_nonempty_file(graph_path):
                    if not require_graph_nonempty or _is_graph_nonempty(graph_path):
                        if _preserve_graph_copy(graph_path, dataset_name, graph_file, preserve_graph_root):
                            success = True
                            break
                        with log_file.open("a", encoding="utf-8") as f:
                            f.write(
                                f"[runner] graph preserve copy failed after build: {graph_path} -> {preserve_graph_root}\n"
                            )
                        last_returncode = 97
                        if attempt < max_retries:
                            time.sleep(retry_backoff)
                            continue
                        break
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(
                            f"[runner] graph integrity check failed after build: {graph_path}\n"
                        )
                    last_returncode = 98
                    if attempt < max_retries:
                        time.sleep(retry_backoff)
                        continue
                    break

            if attempt < max_retries:
                time.sleep(retry_backoff)
    finally:
        if tmp_config_path is not None:
            try:
                tmp_config_path.unlink(missing_ok=True)
            except Exception:
                pass

    return {
        "dataset_name": dataset_name,
        "returncode": last_returncode,
        "log_file": str(log_file) if log_file else "",
        "root_name": root_name,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_sec": round(end_ts - start_ts, 6) if start_ts and end_ts else "",
        "success": success,
        "attempts": attempts,
    }


def _count_corpus_sessions(corpus_path: Path) -> int | None:
    if not corpus_path.exists():
        return None
    try:
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except json.JSONDecodeError:
        pass
    except Exception:
        return None
    try:
        with corpus_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return None


def _read_first_json_line(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        return json.loads(first) if first else {}
    except Exception:
        return {}


def _compute_s_value(data_root: Path, dataset_name: str, origin: Any) -> int | str:
    question_path = data_root / dataset_name / "Question.json"
    question_record = _read_first_json_line(question_path)
    explicit_bin_s = question_record.get("bin_s", "")
    if explicit_bin_s not in ("", None):
        try:
            return int(explicit_bin_s)
        except Exception:
            pass

    corpus_path = data_root / dataset_name / "Corpus.json"
    total_sessions = _count_corpus_sessions(corpus_path)
    if total_sessions is None:
        return ""
    if isinstance(origin, list):
        origin_count = len(origin)
    elif isinstance(origin, str):
        origin_count = 1 if origin.strip() else 0
    elif isinstance(origin, tuple):
        origin_count = len(origin)
    else:
        origin_count = 0
    s_val = total_sessions - origin_count
    if s_val < 0:
        return 0
    return s_val


def _extract_row(results_path: Path, dataset_name: str, data_root: Path) -> Dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    record = data[0] if data else {}

    output = record.get("output", "")
    gold = record.get("answer", "")

    f1 = _f1_score(output, gold)

    llm_judge = record.get("llm_judge")
    success = record.get("success")
    if success is None:
        success = bool(llm_judge) if llm_judge is not None else False

    chunks = record.get("chunks", []) or []
    summaries = record.get("summaries", []) or []
    triples = record.get("triples", []) or []
    formatted_prompt = record.get("formatted_prompt", "") or ""
    context_ok = bool(formatted_prompt) and (bool(chunks) or bool(summaries) or bool(triples))

    sanitized_record = sanitize_trace_record_for_export(record)
    trace_json = json.dumps(sanitized_record, ensure_ascii=False)
    s_value = _compute_s_value(data_root, dataset_name, record.get("origin", []))
    memos_metrics = derive_memos_metrics(record=sanitized_record)
    dataset_prefix = dataset_name.split("/", 1)[0] if "/" in dataset_name else dataset_name

    question_type = record.get("question_type", "")
    if not question_type:
        prefix_to_type = {
            "longmem_ssp": "single-session-preference",
            "longmem_ssu": "single-session-user",
            "longmem_ssa": "single-session-assistant",
            "longmem_tr": "temporal-reasoning",
            "longmem_ku": "knowledge-update",
            "longmem_ms": "multi-session",
        }
        question_type = prefix_to_type.get(dataset_prefix, "")

    row = {
        "task_id": record.get("question_id", ""),
        "dataset_name": dataset_name,
        "model": record.get("model", ""),
        "memory": record.get("memory_system", "LiCoMemory"),
        "s": s_value,
        "question_type": question_type,
        "trial": record.get("trial", 1),
        "retrieval_calls": record.get("retrieval_calls", 0),
        "retrieved_sessions": record.get("retrieved_sessions", len(record.get("top_session_ids", []) or [])),
        "react_steps": len(record.get("react_trace", []) or []),
        "success": int(bool(success)),
        "f1": round(f1, 6),
        "llm_judge": int(bool(llm_judge)) if llm_judge is not None else "",
        "response_duration_ms": memos_metrics["response_duration_ms"],
        "search_duration_ms": memos_metrics["search_duration_ms"],
        "total_duration_ms": memos_metrics["total_duration_ms"],
        "context_tokens": memos_metrics["context_tokens"],
        "total_cost_usd": record.get("total_cost_usd", ""),
        "context_ok": int(context_ok),
        "agent_output": output,
        "trace_json": trace_json,
    }
    return row

def _wait_for_results(path: Path, timeout_sec: int = 30, interval_sec: float = 1.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(interval_sec)
    return path.exists() and path.stat().st_size > 0


def _wait_for_nonempty_file(path: Path, timeout_sec: int = 30, interval_sec: float = 1.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(interval_sec)
    return path.exists() and path.stat().st_size > 0


def _is_graph_nonempty(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        data = pickle.load(path.open("rb"))
    except Exception:
        return False

    graph_obj = None
    entity_map = None
    if isinstance(data, dict):
        graph_obj = data.get("graph")
        # Support both legacy and current graph schemas.
        entity_map = (
            data.get("entity_name_to_index")
            or data.get("entities")
            or data.get("entity_map")
        )
    else:
        graph_obj = getattr(data, "graph", None)
        entity_map = (
            getattr(data, "entity_name_to_index", None)
            or getattr(data, "entities", None)
            or getattr(data, "entity_map", None)
        )

    nodes = 0
    try:
        if graph_obj is not None and hasattr(graph_obj, "nodes"):
            nodes = len(graph_obj.nodes())
        elif graph_obj is not None:
            nodes = len(graph_obj)
    except Exception:
        nodes = 0

    entities = 0
    try:
        if entity_map is not None:
            entities = len(entity_map)
    except Exception:
        entities = 0

    return (nodes > 0) and (entities > 0)


def _error_row(dataset_name: str, returncode: int, log_file: str) -> Dict[str, Any]:
    trace_json = json.dumps({
        "error": "results.json missing after run",
        "returncode": returncode,
        "log_file": log_file,
    }, ensure_ascii=False)
    return {
        "task_id": "",
        "dataset_name": dataset_name,
        "model": "",
        "memory": "LiCoMemory",
        "s": "",
        "question_type": "",
        "trial": "",
        "retrieval_calls": 0,
        "retrieved_sessions": 0,
        "react_steps": 0,
        "success": 0,
        "f1": 0.0,
        "llm_judge": "",
        "response_duration_ms": "",
        "search_duration_ms": "",
        "total_duration_ms": "",
        "context_tokens": "",
        "total_cost_usd": "",
        "context_ok": 0,
        "agent_output": "",
        "trace_json": trace_json,
    }


def _log_runner(log_dir: Path, message: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runner.log"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path.open("a", encoding="utf-8").write(f"{ts} - {message}\n")


def _read_llm_max_concurrent(config_path: str) -> int:
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
        return int(llm_cfg.get("max_concurrent", 1))
    except Exception:
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--root-prefix", default="longmemeval_full")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset-list", type=str, default=None, help="Optional file with dataset_name per line to run (relative to data_root).")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=5.0)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--query", type=int, choices=[0, 1], default=1, help="Run query/eval (1) or build/load graph only (0).")
    parser.add_argument("--graph-file", type=str, default="dynamic_memory_graph.pkl", help="Graph filename to validate in query=0 mode.")
    parser.add_argument("--require-graph-nonempty", action="store_true", help="Treat graph as valid only if nodes/entities are non-empty.")
    parser.add_argument("--preserve-graph-root", type=str, default="", help="Optional stable cache root to copy q0 graphs into immediately after detection.")
    parser.add_argument(
        "--fail-on-unfinished",
        action="store_true",
        help="Exit non-zero if any dataset in this run remains unfinished (not in completed set).",
    )
    args = parser.parse_args()

    try:
        repo_root = Path(args.repo_root)
        data_root = Path(args.data_root)
        log_dir = Path(args.log_dir) if args.log_dir else (repo_root / "logs" / f"longmemeval_run_{int(time.time())}")
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = _load_checkpoint(checkpoint_path)
        if "failed" not in checkpoint:
            checkpoint["failed"] = {}
        completed = checkpoint.get("completed", {})
        failed = checkpoint.get("failed", {})

        items = _list_items(data_root)
        if args.dataset_list:
            list_path = Path(args.dataset_list)
            if not list_path.exists():
                raise FileNotFoundError(f"dataset list not found: {list_path}")
            requested = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            available = set(items)
            items = [name for name in requested if name in available]
            _log_runner(log_dir, f"dataset_list={list_path} requested={len(requested)} matched={len(items)}")
        if args.shuffle:
            random.seed(args.seed)
            random.shuffle(items)
        if args.limit and args.limit > 0:
            items = items[: args.limit]

        # Filter completed; failed are retried on next run unless stop-on-fail is used
        pending = [item for item in items if item not in completed]
        _log_runner(log_dir, f"items={len(items)} pending={len(pending)} completed={len(completed)} failed={len(failed)} workers={args.workers}")
        llm_max_concurrent = _read_llm_max_concurrent(args.config)
        effective_llm_parallelism = max(1, args.workers) * max(1, llm_max_concurrent)
        _log_runner(
            log_dir,
            f"config.llm.max_concurrent={llm_max_concurrent} -> "
            f"effective_parallelism=workers({args.workers})*llm.max_concurrent({llm_max_concurrent})={effective_llm_parallelism}"
        )
        if effective_llm_parallelism > 6:
            _log_runner(
                log_dir,
                f"warning: effective_parallelism={effective_llm_parallelism} may trigger provider overload/503; "
                "consider lowering workers or llm.max_concurrent"
            )

        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_exists = csv_path.exists()
        if csv_exists and csv_path.stat().st_size == 0:
            csv_exists = False
        fieldnames = [
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

        with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()

            run_args = []
            for item in pending:
                run_args.append({
                    "dataset_name": item,
                    "repo_root": str(repo_root),
                    "data_root": str(data_root),
                    "python_bin": args.python_bin,
                    "config_path": args.config,
                    "root_prefix": args.root_prefix,
                    "log_dir": str(log_dir),
                    "max_retries": args.max_retries,
                    "retry_backoff": args.retry_backoff,
                    "timeout_sec": args.timeout_sec,
                    "query": args.query,
                    "graph_file": args.graph_file,
                    "require_graph_nonempty": args.require_graph_nonempty,
                    "preserve_graph_root": args.preserve_graph_root,
                })

            if not run_args:
                _log_runner(log_dir, "nothing to run; exiting")
                return

            # Each worker already launches an external `main.py` subprocess.
            # Using a process pool here adds a second layer of worker processes,
            # which has proven fragile under heavier q1 workloads and surfaces
            # as BrokenProcessPool even when the underlying subprocess work is fine.
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_dataset = {}
                for ra in run_args:
                    fut = executor.submit(_run_one, ra)
                    future_to_dataset[fut] = ra["dataset_name"]

                for fut in as_completed(future_to_dataset):
                    dataset_name = future_to_dataset[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        _log_runner(log_dir, f"Worker crash for {dataset_name}: {e!r}")
                        checkpoint["failed"][dataset_name] = {
                            "returncode": -1,
                            "log_file": "",
                            "root_name": f"{args.root_prefix}/{dataset_name}",
                            "start_ts": "",
                            "end_ts": "",
                            "duration_sec": "",
                            "attempts": 0,
                        }
                        checkpoint["completed"].pop(dataset_name, None)
                        _save_checkpoint(checkpoint_path, checkpoint)
                        if args.stop_on_fail:
                            raise
                        continue

                    if result.get("success"):
                        checkpoint["completed"][dataset_name] = {
                            "returncode": result["returncode"],
                            "log_file": result["log_file"],
                            "root_name": result["root_name"],
                            "start_ts": result["start_ts"],
                            "end_ts": result["end_ts"],
                            "duration_sec": result["duration_sec"],
                            "attempts": result.get("attempts", 1),
                        }
                        checkpoint["failed"].pop(dataset_name, None)
                        _save_checkpoint(checkpoint_path, checkpoint)
                        _log_runner(
                            log_dir,
                            f"done dataset={dataset_name} rc={result['returncode']} "
                            f"attempts={result.get('attempts', 1)} "
                            f"completed={len(checkpoint['completed'])}/{len(items)}",
                        )

                        if args.query == 1:
                            results_path = repo_root / "results" / result["root_name"] / "results" / "results.json"
                            row = _extract_row(results_path, dataset_name, data_root)
                            writer.writerow(row)
                            csv_file.flush()
                    else:
                        checkpoint["failed"][dataset_name] = {
                            "returncode": result["returncode"],
                            "log_file": result["log_file"],
                            "root_name": result["root_name"],
                            "start_ts": result["start_ts"],
                            "end_ts": result["end_ts"],
                            "duration_sec": result["duration_sec"],
                            "attempts": result.get("attempts", 1),
                        }
                        checkpoint["completed"].pop(dataset_name, None)
                        _save_checkpoint(checkpoint_path, checkpoint)
                        _log_runner(
                            log_dir,
                            f"failed dataset={dataset_name} rc={result['returncode']} "
                            f"attempts={result.get('attempts', 1)} "
                            f"completed={len(checkpoint['completed'])}/{len(items)} "
                            f"failed_total={len(checkpoint['failed'])}",
                        )
                        if args.stop_on_fail:
                            raise RuntimeError(f"Run failed for {dataset_name}; see {result['log_file']}")

            unfinished = [item for item in items if item not in checkpoint.get("completed", {})]
            unresolved_failed = [
                item
                for item in unfinished
                if item in checkpoint.get("failed", {})
            ]
            _log_runner(
                log_dir,
                "run summary: "
                f"items={len(items)} completed={len(checkpoint.get('completed', {}))} "
                f"unfinished={len(unfinished)} unresolved_failed={len(unresolved_failed)}",
            )
            if args.fail_on_unfinished and unfinished:
                raise RuntimeError(
                    "unfinished datasets remain after run: "
                    f"{len(unfinished)} (unresolved_failed={len(unresolved_failed)})"
                )
    except Exception as e:
        # Best-effort logging for top-level failures
        try:
            _log_runner(Path(args.log_dir) if args.log_dir else Path(args.repo_root) / "logs", f"runner fatal error: {e!r}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
