from __future__ import annotations

import asyncio
import time
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openclaw_memory_mvp import (
    parse_dialog_turns,
    SOURCE_MEMORY,
    SOURCE_SESSIONS,
    SearchResult,
    hash_text,
    normalize_session_date,
    render_memory_markdown,
    safe_slug,
)


DEFAULT_NODE_BIN = "node"


def render_official_session_jsonl(row: Dict[str, Any]) -> str:
    context = str(row.get("context", "") or "")
    turns = parse_dialog_turns(context)
    if not turns:
        payload = {
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": context.strip()}],
            },
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"

    role_map: Dict[str, str] = {}
    ordered_roles = ["user", "assistant"]
    raw_lines: List[str] = []
    for turn in turns:
        speaker = str(turn.get("role", "") or "").strip()
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        lowered = speaker.lower()
        if lowered in {"user", "assistant"}:
            normalized_role = lowered
            text = content
        else:
            if lowered not in role_map:
                role_map[lowered] = ordered_roles[min(len(role_map), len(ordered_roles) - 1)]
            normalized_role = role_map[lowered]
            text = f"{speaker}: {content}" if speaker else content
        payload = {
            "type": "message",
            "message": {
                "role": normalized_role,
                "content": [{"type": "text", "text": text}],
            },
        }
        raw_lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(raw_lines).strip() + "\n"


class OfficialOpenClawMemoryBackend:
    def __init__(
        self,
        *,
        repo_root: Path,
        work_root: Path,
        embedding_model: str,
        chunk_tokens: int,
        chunk_overlap: int,
        vector_weight: float,
        text_weight: float,
        candidate_multiplier: int,
        enable_mmr: bool,
        mmr_lambda: float,
        enable_temporal_decay: bool,
        half_life_days: float,
        sources: Sequence[str],
        agent_id: str = "main",
        node_bin: Optional[str] = None,
        provider: str = "openai",
        remote_base_url: str = "",
        remote_api_key: str = "",
        include_searchable_highlights: bool = True,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.work_root = work_root.resolve()
        self.workspace_dir = self.work_root / "workspace"
        self.state_dir = self.work_root / "openclaw_state"
        self.bridge_path = self.repo_root / "benchmarks" / "memory_mvp" / "official_memory_bridge.mjs"
        self.store_path = self.state_dir / "index-{agentId}.sqlite"
        self.embedding_model = str(embedding_model or "").strip()
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap = int(chunk_overlap)
        self.vector_weight = float(vector_weight)
        self.text_weight = float(text_weight)
        self.candidate_multiplier = int(candidate_multiplier)
        self.enable_mmr = bool(enable_mmr)
        self.mmr_lambda = float(mmr_lambda)
        self.enable_temporal_decay = bool(enable_temporal_decay)
        self.half_life_days = float(half_life_days)
        self.sources = list(dict.fromkeys(str(source).strip() for source in sources if str(source).strip()))
        self.agent_id = str(agent_id or "main")
        self.node_bin = str(node_bin or os.environ.get("OPENCLAW_NODE_BIN") or DEFAULT_NODE_BIN)
        self.provider = str(provider or "openai").strip() or "openai"
        self.remote_base_url = str(remote_base_url or "").strip()
        self.remote_api_key = str(remote_api_key or "").strip()
        self.include_searchable_highlights = bool(include_searchable_highlights)

    def close(self) -> None:
        return None

    def _sanitize_snapshot_value(self, value: Any, *, key: str = "") -> Any:
        lowered = str(key or "").strip().lower()
        if isinstance(value, dict):
            return {
                str(k): self._sanitize_snapshot_value(v, key=str(k))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_snapshot_value(item, key=key) for item in value]
        if any(token in lowered for token in ("api_key", "apikey", "authorization", "token", "secret", "password")):
            text = str(value or "")
            if not text:
                return ""
            if len(text) <= 8:
                return "***"
            return f"{text[:4]}...{text[-4:]}"
        return value

    def _write_bridge_snapshot(
        self,
        *,
        payload: Dict[str, Any],
        reason: str,
        returncode: Optional[int],
        stdout_text: str,
        stderr_text: str,
        error_message: str,
    ) -> Path:
        snapshot_root = self.work_root / "bridge_failures"
        snapshot_root.mkdir(parents=True, exist_ok=True)
        op_name = safe_slug(str(payload.get("op", "bridge") or "bridge"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_dir = snapshot_root / f"{stamp}_{time.time_ns() % 1_000_000_000:09d}_{op_name}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        sanitized_payload = self._sanitize_snapshot_value(payload)
        meta = {
            "reason": reason,
            "error_message": error_message,
            "returncode": returncode,
            "node_bin": self.node_bin,
            "bridge_path": str(self.bridge_path),
            "repo_root": str(self.repo_root),
            "work_root": str(self.work_root),
            "workspace_dir": str(self.workspace_dir),
            "state_dir": str(self.state_dir),
            "provider": self.provider,
            "embedding_model": self.embedding_model,
            "remote_base_url": self.remote_base_url,
            "remote_api_key": self._sanitize_snapshot_value(self.remote_api_key, key="api_key"),
            "env_summary": {
                "OPENCLAW_STATE_DIR": str(self.state_dir),
                "OPENCLAW_LOG_LEVEL": str(os.environ.get("OPENCLAW_LOG_LEVEL", "") or ""),
                "OPENCLAW_DEBUG_MEMORY_EMBEDDINGS": str(
                    os.environ.get("OPENCLAW_DEBUG_MEMORY_EMBEDDINGS", "") or ""
                ),
            },
        }
        (snapshot_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (snapshot_dir / "payload.json").write_text(
            json.dumps(sanitized_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (snapshot_dir / "stdout.log").write_text(stdout_text or "", encoding="utf-8")
        (snapshot_dir / "stderr.log").write_text(stderr_text or "", encoding="utf-8")
        return snapshot_dir

    def _memory_search_config(self) -> Dict[str, Any]:
        query_cfg: Dict[str, Any] = {
            "hybrid": {
                "enabled": True,
                "vectorWeight": self.vector_weight,
                "textWeight": self.text_weight,
                "candidateMultiplier": self.candidate_multiplier,
            }
        }
        if self.enable_mmr:
            query_cfg["hybrid"]["mmr"] = {
                "enabled": True,
                "lambda": self.mmr_lambda,
            }
        if self.enable_temporal_decay:
            query_cfg["hybrid"]["temporalDecay"] = {
                "enabled": True,
                "halfLifeDays": self.half_life_days,
            }
        memory_search: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.embedding_model,
            "store": {
                "path": str(self.store_path),
            },
            "chunking": {
                "tokens": self.chunk_tokens,
                "overlap": self.chunk_overlap,
            },
            "query": query_cfg,
            "sources": list(self.sources),
            "experimental": {
                "sessionMemory": SOURCE_SESSIONS in self.sources,
            },
            "sync": {
                "watch": False,
                "onSessionStart": False,
                # The benchmark harness force-syncs q0 before q1. Keeping the
                # official on-search refresh enabled in this short-lived bridge
                # setup can race with search/close and surface as sqlite
                # "database is not open" even though the index was already built.
                "onSearch": False,
            },
        }
        if self.remote_base_url or self.remote_api_key:
            memory_search["remote"] = {}
            if self.remote_base_url:
                memory_search["remote"]["baseUrl"] = self.remote_base_url
            if self.remote_api_key:
                memory_search["remote"]["apiKey"] = self.remote_api_key
        return memory_search

    def build_config(self) -> Dict[str, Any]:
        return {
            "agents": {
                "defaults": {
                    "workspace": str(self.workspace_dir),
                    "memorySearch": self._memory_search_config(),
                },
                "list": [
                    {
                        "id": self.agent_id,
                        "default": True,
                    }
                ],
            }
        }

    async def _bridge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        command = [self.node_bin, str(self.bridge_path)]
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(self.state_dir)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.repo_root),
                env=env,
            )
        except Exception as exc:
            snapshot_dir = self._write_bridge_snapshot(
                payload=payload,
                reason="spawn_failed",
                returncode=None,
                stdout_text="",
                stderr_text="",
                error_message=str(exc),
            )
            raise RuntimeError(
                f"official_openclaw_bridge_spawn_failed message={exc} snapshot={snapshot_dir}"
            ) from exc
        stdin_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            stdout_bytes, stderr_bytes = await process.communicate(stdin_payload)
        except asyncio.CancelledError as exc:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except Exception:
                    process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except Exception:
                        pass
            snapshot_dir = self._write_bridge_snapshot(
                payload=payload,
                reason="cancelled",
                returncode=process.returncode,
                stdout_text="",
                stderr_text="",
                error_message="official_openclaw_bridge_cancelled",
            )
            raise RuntimeError(
                f"official_openclaw_bridge_cancelled snapshot={snapshot_dir}"
            ) from exc
        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            snapshot_dir = self._write_bridge_snapshot(
                payload=payload,
                reason="empty_output",
                returncode=process.returncode,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                error_message=stderr_text or "official_openclaw_bridge_empty_output",
            )
            raise RuntimeError(
                f"official_openclaw_bridge_empty_output rc={process.returncode} "
                f"stderr={stderr_text} snapshot={snapshot_dir}"
            )
        result: Dict[str, Any] | None = None
        last_exc: json.JSONDecodeError | None = None
        for line in reversed(stdout_text.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
            if isinstance(parsed, dict) and "ok" in parsed:
                result = parsed
                break
        if result is None:
            snapshot_dir = self._write_bridge_snapshot(
                payload=payload,
                reason="invalid_json",
                returncode=process.returncode,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                error_message="official_openclaw_bridge_invalid_json",
            )
            raise RuntimeError(
                f"official_openclaw_bridge_invalid_json rc={process.returncode} "
                f"snapshot={snapshot_dir} stdout={stdout_text} stderr={stderr_text}"
            ) from last_exc
        if process.returncode != 0 or not bool(result.get("ok")):
            error = result.get("error") or {}
            error_message = str(error.get("message") or stderr_text or stdout_text)
            snapshot_dir = self._write_bridge_snapshot(
                payload=payload,
                reason="bridge_failed",
                returncode=process.returncode,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                error_message=error_message,
            )
            raise RuntimeError(
                f"official_openclaw_bridge_failed rc={process.returncode} "
                f"message={error_message} snapshot={snapshot_dir}"
            )
        return result

    def _session_dir(self) -> Path:
        return self.state_dir / "agents" / self.agent_id / "sessions"

    def materialize_corpus_files(
        self,
        corpus_rows: List[Dict[str, Any]],
        *,
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "memory").mkdir(parents=True, exist_ok=True)
        self._session_dir().mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "MEMORY.md").write_text("# Durable Memory\n\n", encoding="utf-8")

        used_paths: set[tuple[str, str]] = set()
        entries: List[Dict[str, Any]] = []
        source_set = {str(source).strip() for source in sources if str(source).strip()}
        if not source_set:
            source_set = {SOURCE_SESSIONS}

        for row in corpus_rows:
            session_id = str(row.get("session_id", "") or "")
            session_time = str(row.get("session_time", "") or "")
            date_stamp = normalize_session_date(session_time)
            for source in source_set:
                if source == SOURCE_MEMORY:
                    base_name = f"{date_stamp}-{safe_slug(session_id)}.md"
                    rel_path = f"memory/{base_name}"
                    abs_path = self.workspace_dir / rel_path
                    raw_content = render_memory_markdown(
                        row,
                        include_searchable_highlights=self.include_searchable_highlights,
                    )
                elif source == SOURCE_SESSIONS:
                    base_name = f"{safe_slug(session_id)}.jsonl"
                    rel_path = f"sessions/{base_name}"
                    abs_path = self._session_dir() / base_name
                    raw_content = render_official_session_jsonl(row)
                else:
                    continue

                suffix = 2
                while (source, rel_path) in used_paths:
                    stem, ext = rel_path.rsplit(".", 1)
                    rel_path = f"{stem}-{suffix:02d}.{ext}"
                    if source == SOURCE_MEMORY:
                        abs_path = self.workspace_dir / rel_path
                    else:
                        abs_path = self._session_dir() / Path(rel_path).name
                    suffix += 1

                used_paths.add((source, rel_path))
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(raw_content, encoding="utf-8")
                entries.append(
                    {
                        "source": source,
                        "rel_path": rel_path,
                        "abs_path": str(abs_path),
                        "session_id": session_id,
                        "session_time": session_time,
                        "hash": hash_text(raw_content),
                    }
                )

        return {
            "entries": entries,
            "num_files": len(entries),
            "workspace_dir": str(self.workspace_dir),
            "state_dir": str(self.state_dir),
            "sources": list(source_set),
        }

    async def build_index_from_corpus(
        self,
        *,
        corpus_rows: List[Dict[str, Any]],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        materialized = self.materialize_corpus_files(corpus_rows, sources=sources)
        sync_result = await self._bridge(
            {
                "op": "sync",
                "stateDir": str(self.state_dir),
                "cfg": self.build_config(),
                "agentId": self.agent_id,
                "force": True,
            }
        )
        status = dict(sync_result.get("status") or {})
        return {
            "num_files": int(status.get("files", 0) or 0),
            "num_chunks": int(status.get("chunks", 0) or 0),
            "workspace_dir": str(self.workspace_dir),
            "state_dir": str(self.state_dir),
            "index_db_path": str(status.get("dbPath", "") or self.store_path),
            "sources": list(self.sources),
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap": self.chunk_overlap,
            "provider": status.get("provider", self.provider),
            "model": status.get("model", self.embedding_model),
            "scan_files_materialized": materialized["num_files"],
        }

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        min_score: float,
    ) -> Dict[str, Any]:
        result = await self._bridge(
            {
                "op": "search",
                "stateDir": str(self.state_dir),
                "cfg": self.build_config(),
                "agentId": self.agent_id,
                "query": query,
                "maxResults": max_results,
                "minScore": min_score,
            }
        )
        payload = dict(result.get("payload") or {})
        raw_results = list(payload.get("results") or [])
        parsed_results: List[SearchResult] = []
        for index, row in enumerate(raw_results, start=1):
            path_value = str(row.get("path", "") or "")
            session_id = ""
            if str(row.get("source", "") or "") == SOURCE_SESSIONS and path_value:
                session_id = Path(path_value).stem
            parsed_results.append(
                SearchResult(
                    id=str(row.get("id", "") or f"official-{index}"),
                    path=path_value,
                    source=str(row.get("source", "") or ""),
                    session_id=session_id,
                    session_time="",
                    start_line=int(row.get("startLine", 1) or 1),
                    end_line=int(row.get("endLine", row.get("startLine", 1)) or row.get("startLine", 1) or 1),
                    text=str(row.get("snippet", "") or ""),
                    score=float(row.get("score", 0.0) or 0.0),
                    vector_score=float(row.get("score", 0.0) or 0.0),
                    text_score=0.0,
                )
            )
        return {
            "payload": payload,
            "results": parsed_results,
            "raw_results": raw_results,
            "status": {},
            "provider": payload.get("provider", self.provider),
            "model": payload.get("model", self.embedding_model),
            "fallback": payload.get("fallback"),
            "mode": payload.get("mode"),
            "citations": payload.get("citations"),
            "disabled": bool(payload.get("disabled")),
            "unavailable": bool(payload.get("unavailable")),
            "warning": payload.get("warning"),
            "action": payload.get("action"),
            "error": payload.get("error"),
        }

    async def read_file(
        self,
        *,
        rel_path: str,
        from_line: Optional[int] = None,
        lines: Optional[int] = None,
    ) -> Dict[str, str]:
        result = await self._bridge(
            {
                "op": "read_file",
                "stateDir": str(self.state_dir),
                "cfg": self.build_config(),
                "agentId": self.agent_id,
                "path": rel_path,
                "fromLine": from_line,
                "lines": lines,
            }
        )
        return dict(result.get("payload") or {})

    async def prompt_section(
        self,
        *,
        citations_mode: str = "off",
        available_tools: Optional[Sequence[str]] = None,
    ) -> List[str]:
        result = await self._bridge(
            {
                "op": "prompt_section",
                "availableTools": list(available_tools or ["memory_search", "memory_get"]),
                "citationsMode": citations_mode,
            }
        )
        payload = dict(result.get("payload") or {})
        return [str(line) for line in list(payload.get("lines") or []) if str(line).strip()]
