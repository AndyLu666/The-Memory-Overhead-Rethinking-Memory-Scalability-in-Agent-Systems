#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

from official_openclaw_memory import OfficialOpenClawMemoryBackend
from probe_dataset_item import (
    ensure_openai_env_aliases,
    maybe_load_env_file,
    _resolve_env_value,
    load_jsonl,
    parse_sources,
    q0_reuse_matches,
    resolve_official_remote_config,
    save_json,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prebuild q0 OpenClaw document-memory assets for a dataset list."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-list", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-tokens", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--sources", default="memory")
    parser.add_argument("--memory-backend", default="official")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-base-url-env", default="")
    parser.add_argument("--embedding-api-key-env", default="")
    parser.add_argument("--openclaw-repo-root", default="")
    parser.add_argument("--openclaw-node-bin", default="")
    parser.add_argument("--enable-mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--enable-temporal-decay", action="store_true")
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_dataset_names(path: Path, limit: int) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        names = names[:limit]
    return names


def q0_cache_key(
    *,
    corpus_sha256: str,
    memory_backend: str,
    embedding_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
    sources: Sequence[str],
) -> str:
    payload = {
        "corpus_sha256": corpus_sha256,
        "memory_backend": memory_backend,
        "embedding_model": str(embedding_model or "").strip(),
        "chunk_tokens": int(chunk_tokens or 0),
        "chunk_overlap": int(chunk_overlap or 0),
        "sources": list(sources),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def q0_output_ready(
    *,
    root: Path,
    corpus_sha256: str,
    memory_backend: str,
    embedding_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
    sources: Sequence[str],
) -> bool:
    summary_path = root / "q0_index_summary.json"
    workspace_dir = root / "workspace"
    state_dir = root / "openclaw_state"
    if not summary_path.exists() or not workspace_dir.exists() or not state_dir.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return q0_reuse_matches(
        summary,
        corpus_sha256=corpus_sha256,
        memory_backend=memory_backend,
        embedding_model=embedding_model,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        sources=sources,
    )


def copy_q0_assets(*, src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in ["workspace", "openclaw_state"]:
        src = src_root / name
        dst = dst_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for name in ["q0_index_summary.json", "q0_cost_summary.json"]:
        src = src_root / name
        dst = dst_root / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)


class ProgressTracker:
    def __init__(self, *, results_root: Path, items_requested: int) -> None:
        self.results_root = results_root
        self.items_requested = items_requested
        self.completed = 0
        self.reused = 0
        self.failed = 0
        self.failures: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._jsonl_path = self.results_root / "q0_prebuild_results.jsonl"
        self._progress_path = self.results_root / "q0_prebuild_progress.json"
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, item: Dict[str, Any]) -> None:
        async with self._lock:
            status = str(item.get("status", "") or "").strip().lower()
            if status == "failed":
                self.failed += 1
                self.failures.append(
                    {
                        "dataset_name": item.get("dataset_name", ""),
                        "error": item.get("error", ""),
                        "attempts": item.get("attempts", 0),
                    }
                )
            else:
                self.completed += 1
                if status == "reused":
                    self.reused += 1
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            save_json(
                self._progress_path,
                {
                    "results_root": str(self.results_root),
                    "items_requested": self.items_requested,
                    "items_completed": self.completed,
                    "items_reused": self.reused,
                    "items_failed": self.failed,
                    "last_item": item,
                    "failures": self.failures[-20:],
                },
            )


async def build_one(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    data_root: Path,
    results_root: Path,
    sources: Sequence[str],
    embedding_base_url: str,
    embedding_api_key: str,
    cache_locks: Dict[str, asyncio.Lock],
) -> Dict[str, Any]:
    corpus_json = data_root / dataset_name / "Corpus.json"
    if not corpus_json.exists():
        return {
            "dataset_name": dataset_name,
            "status": "failed",
            "attempts": 0,
            "error": f"missing Corpus.json: {corpus_json}",
        }

    output_dir = results_root / "derived" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    q0_summary_path = output_dir / "q0_index_summary.json"
    q0_cost_summary_path = output_dir / "q0_cost_summary.json"
    corpus_sha256 = sha256_file(corpus_json)
    memory_backend = str(args.memory_backend or "").strip().lower()
    cache_key = q0_cache_key(
        corpus_sha256=corpus_sha256,
        memory_backend=memory_backend,
        embedding_model=args.embedding_model,
        chunk_tokens=int(args.chunk_tokens),
        chunk_overlap=int(args.chunk_overlap),
        sources=sources,
    )
    cache_root = results_root / "_q0_cache" / cache_key

    if (
        not args.force
        and q0_output_ready(
            root=output_dir,
            corpus_sha256=corpus_sha256,
            memory_backend=memory_backend,
            embedding_model=args.embedding_model,
            chunk_tokens=int(args.chunk_tokens),
            chunk_overlap=int(args.chunk_overlap),
            sources=sources,
        )
    ):
        existing = json.loads(q0_summary_path.read_text(encoding="utf-8"))
        existing_cost = (
            json.loads(q0_cost_summary_path.read_text(encoding="utf-8"))
            if q0_cost_summary_path.exists()
            else {}
        )
        return {
            "dataset_name": dataset_name,
            "status": "reused",
            "reuse_scope": "output_dir",
            "attempts": 0,
            "duration_ms": float(existing.get("duration_ms", 0.0) or 0.0),
            "num_files": int(existing.get("num_files", 0) or 0),
            "num_chunks": int(existing.get("num_chunks", 0) or 0),
            "corpus_sha256": corpus_sha256,
            "cache_key": cache_key,
            "output_dir": str(output_dir),
            "total_cost_usd": float(existing_cost.get("total_cost_usd", 0.0) or 0.0),
            "cost_observability": existing_cost.get("cost_observability", ""),
        }

    if memory_backend != "official":
        return {
            "dataset_name": dataset_name,
            "status": "failed",
            "attempts": 0,
            "error": "prebuild_openclaw_q0 currently supports only --memory-backend official",
        }

    try:
        corpus_rows = load_jsonl(corpus_json)
    except Exception as exc:
        return {
            "dataset_name": dataset_name,
            "status": "failed",
            "attempts": 0,
            "error": f"failed to load Corpus.json: {exc}",
            "corpus_sha256": corpus_sha256,
            "cache_key": cache_key,
            "output_dir": str(output_dir),
        }
    provider, remote_base_url, remote_api_key = resolve_official_remote_config(
        args.embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
    )
    cache_lock = cache_locks.setdefault(cache_key, asyncio.Lock())
    async with cache_lock:
        if (
            not args.force
            and q0_output_ready(
                root=cache_root,
                corpus_sha256=corpus_sha256,
                memory_backend=memory_backend,
                embedding_model=args.embedding_model,
                chunk_tokens=int(args.chunk_tokens),
                chunk_overlap=int(args.chunk_overlap),
                sources=sources,
            )
        ):
            copy_q0_assets(src_root=cache_root, dst_root=output_dir)
            cache_summary = json.loads((cache_root / "q0_index_summary.json").read_text(encoding="utf-8"))
            cache_cost = json.loads((cache_root / "q0_cost_summary.json").read_text(encoding="utf-8"))
            return {
                "dataset_name": dataset_name,
                "status": "reused",
                "reuse_scope": "corpus_hash_cache",
                "attempts": 0,
                "duration_ms": float(cache_summary.get("duration_ms", 0.0) or 0.0),
                "num_files": int(cache_summary.get("num_files", 0) or 0),
                "num_chunks": int(cache_summary.get("num_chunks", 0) or 0),
                "corpus_sha256": corpus_sha256,
                "cache_key": cache_key,
                "output_dir": str(output_dir),
                "total_cost_usd": float(cache_cost.get("total_cost_usd", 0.0) or 0.0),
                "cost_observability": cache_cost.get("cost_observability", ""),
            }

        if args.force and cache_root.exists():
            shutil.rmtree(cache_root)

        last_error = ""
        for attempt in range(1, max(1, int(args.max_attempts)) + 1):
            try:
                started = asyncio.get_running_loop().time()
                engine = OfficialOpenClawMemoryBackend(
                    repo_root=(
                        Path(str(getattr(args, "openclaw_repo_root", "") or "")).expanduser().resolve()
                        if str(getattr(args, "openclaw_repo_root", "") or "").strip()
                        else Path(__file__).resolve().parents[2]
                    ),
                    work_root=cache_root,
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
                )
                q0_summary = await engine.build_index_from_corpus(
                    corpus_rows=corpus_rows,
                    sources=sources,
                )
                duration_ms = round((asyncio.get_running_loop().time() - started) * 1000.0, 3)
                q0_summary["corpus_sha256"] = corpus_sha256
                q0_summary["memory_backend"] = "official"
                q0_summary["embedding_model"] = str(args.embedding_model or "").strip()
                q0_summary["duration_ms"] = duration_ms
                q0_cost_summary = {
                    "embedding_prompt_tokens": 0,
                    "embedding_total_tokens": 0,
                    "embedding_calls": 0,
                    "total_cost_usd": 0.0,
                    "duration_ms": duration_ms,
                    "memory_backend": "official",
                    "cost_observability": "not_exposed_by_openclaw",
                }
                save_json(cache_root / "q0_index_summary.json", q0_summary)
                save_json(cache_root / "q0_cost_summary.json", q0_cost_summary)
                copy_q0_assets(src_root=cache_root, dst_root=output_dir)
                return {
                    "dataset_name": dataset_name,
                    "status": "built",
                    "attempts": attempt,
                    "duration_ms": duration_ms,
                    "num_files": int(q0_summary.get("num_files", 0) or 0),
                    "num_chunks": int(q0_summary.get("num_chunks", 0) or 0),
                    "corpus_sha256": corpus_sha256,
                    "cache_key": cache_key,
                    "output_dir": str(output_dir),
                    "total_cost_usd": 0.0,
                    "cost_observability": "not_exposed_by_openclaw",
                }
            except Exception as exc:
                last_error = f"{exc}\n{traceback.format_exc()}"
                if attempt >= max(1, int(args.max_attempts)):
                    break
                await asyncio.sleep(max(0.0, float(args.retry_delay_seconds)) * attempt)

    return {
        "dataset_name": dataset_name,
        "status": "failed",
        "attempts": max(1, int(args.max_attempts)),
        "error": last_error,
        "corpus_sha256": corpus_sha256,
        "cache_key": cache_key,
        "output_dir": str(output_dir),
    }


async def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    dataset_list = Path(args.dataset_list).resolve()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    names = load_dataset_names(dataset_list, int(args.limit or 0))
    sources = parse_sources(getattr(args, "sources", "memory"))
    embedding_base_url = _resolve_env_value(
        explicit=getattr(args, "embedding_base_url", ""),
        env_name=getattr(args, "embedding_base_url_env", ""),
        fallback_envs=("OPENAI_BASE_URL", "GPT_BASE_URL"),
    )
    embedding_api_key = _resolve_env_value(
        explicit="",
        env_name=getattr(args, "embedding_api_key_env", ""),
        fallback_envs=("OPENAI_API_KEY", "GPT_API_KEY"),
    )

    tracker = ProgressTracker(results_root=results_root, items_requested=len(names))
    semaphore = asyncio.Semaphore(max(1, int(args.parallelism or 1)))
    cache_locks: Dict[str, asyncio.Lock] = {}
    collected: List[Dict[str, Any]] = []

    async def worker(dataset_name: str) -> None:
        async with semaphore:
            item = await build_one(
                args=args,
                dataset_name=dataset_name,
                data_root=data_root,
                results_root=results_root,
                sources=sources,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                cache_locks=cache_locks,
            )
            collected.append(item)
            await tracker.record(item)

    await asyncio.gather(*(worker(name) for name in names))
    summary = {
        "data_root": str(data_root),
        "dataset_list": str(dataset_list),
        "results_root": str(results_root),
        "items_requested": len(names),
        "items_completed": sum(1 for item in collected if item.get("status") != "failed"),
        "items_reused": sum(1 for item in collected if item.get("status") == "reused"),
        "unique_builds": sum(1 for item in collected if item.get("status") == "built"),
        "items_failed": sum(1 for item in collected if item.get("status") == "failed"),
        "parallelism": int(args.parallelism or 1),
        "max_attempts": int(args.max_attempts or 1),
        "embedding_model": args.embedding_model,
        "memory_backend": args.memory_backend,
        "sources": list(sources),
        "completed": [item for item in collected if item.get("status") != "failed"],
        "failures": [item for item in collected if item.get("status") == "failed"],
    }
    save_json(results_root / "q0_prebuild_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.env_file:
        maybe_load_env_file(Path(args.env_file))
    ensure_openai_env_aliases()
    summary = asyncio.run(run_all(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
