from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


SPEAKER_PATTERN = re.compile(r'"([^"\n]+)"\s*:\s*"')
WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)
SNIPPET_MAX_CHARS = 700
SOURCE_MEMORY = "memory"
SOURCE_SESSIONS = "sessions"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text or "").strip())
    slug = slug.strip("-")
    return slug or "item"


def parse_session_datetime(raw: str) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def normalize_session_date(raw: str) -> str:
    dt = parse_session_datetime(raw)
    if dt is not None:
        return dt.strftime("%Y-%m-%d")
    return re.sub(r"[^\d-]+", "-", str(raw or "").strip()).strip("-") or "undated"


def truncate_utf16_safe(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    truncated = raw[: max_chars + 1]
    if truncated and 0xD800 <= ord(truncated[-1]) <= 0xDBFF:
        truncated = truncated[:-1]
    return truncated[:max_chars]


def parse_dialog_turns(context_text: str) -> List[Dict[str, str]]:
    positions: List[Tuple[str, int, int]] = []
    for match in SPEAKER_PATTERN.finditer(context_text):
        speaker = str(match.group(1) or "").strip()
        if speaker:
            positions.append((speaker, match.start(), match.end()))
    positions.sort(key=lambda item: item[1])

    turns: List[Dict[str, str]] = []
    for idx, (speaker, _match_start, start_pos) in enumerate(positions):
        if idx + 1 < len(positions):
            next_start = positions[idx + 1][1]
            content = context_text[start_pos:next_start]
        else:
            content = context_text[start_pos:]
        content = content.rstrip('"')
        content = re.sub(r"\s+", " ", content.strip())
        if content:
            turns.append({"role": speaker, "content": content})
    return turns


def render_memory_markdown(
    row: Dict[str, Any],
    *,
    include_searchable_highlights: bool = True,
) -> str:
    session_id = str(row.get("session_id", "") or "")
    session_time = str(row.get("session_time", "") or "")
    context = str(row.get("context", "") or "")
    turns = parse_dialog_turns(context)

    lines = [
        f"# Session {session_id or 'unknown'}",
        f"Session ID: {session_id or 'unknown'}",
        f"Session Time: {session_time or 'unknown'}",
        "",
    ]
    if turns:
        if include_searchable_highlights:
            lines.append("## Searchable Highlights")
            for idx, turn in enumerate(turns, start=1):
                role = str(turn.get("role", "") or "").strip() or "Speaker"
                content = str(turn.get("content", "") or "").strip()
                if not content:
                    continue
                snippet = truncate_utf16_safe(content, 280).strip()
                if len(content) > len(snippet):
                    snippet = snippet.rstrip() + "..."
                lines.append(f"- Turn {idx} {role}: {snippet}")
            lines.append("")
        for idx, turn in enumerate(turns, start=1):
            role = str(turn.get("role", "") or "").capitalize() or "Speaker"
            lines.append(f"## Turn {idx}")
            lines.append(f"{role}: {str(turn.get('content', '') or '').strip()}")
            lines.append("")
    else:
        lines.extend(["## Transcript", context.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def render_session_jsonl(row: Dict[str, Any]) -> Tuple[str, str, List[int]]:
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
        raw_line = json.dumps(payload, ensure_ascii=False)
        indexed = context.strip()
        if indexed:
            return raw_line + "\n", indexed + "\n", [1]
        return raw_line + "\n", raw_line + "\n", [1]

    raw_lines: List[str] = []
    indexed_lines: List[str] = []
    line_map: List[int] = []
    for turn in turns:
        role = str(turn.get("role", "") or "").strip().lower() or "user"
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        payload = {
            "type": "message",
            "message": {
                "role": role,
                "content": [{"type": "text", "text": content}],
            },
        }
        raw_lines.append(json.dumps(payload, ensure_ascii=False))
        indexed_lines.append(f"{role.capitalize()}: {content}")
        line_map.append(len(raw_lines))
    if not raw_lines:
        raw_lines = [
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": context.strip()}],
                    },
                },
                ensure_ascii=False,
            )
        ]
        indexed_lines = [context.strip() or raw_lines[0]]
        line_map = [1]
    return "\n".join(raw_lines).strip() + "\n", "\n".join(indexed_lines).strip() + "\n", line_map


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    raw = str(text or "")
    if not raw:
        return 0
    if tiktoken is None:
        return max(1, len(raw) // 4)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(raw))


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def chunk_markdown_by_lines(
    text: str,
    *,
    token_model: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return []
    line_tokens = [max(1, count_tokens(line or " ", token_model)) for line in lines]
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end = start
        current_tokens = 0
        while end < len(lines):
            next_tokens = line_tokens[end]
            if end > start and current_tokens + next_tokens > chunk_tokens:
                break
            current_tokens += next_tokens
            end += 1
        if end <= start:
            end = min(len(lines), start + 1)
        chunk_text = "\n".join(lines[start:end]).strip()
        if chunk_text:
            chunks.append((start + 1, end, chunk_text))
        if end >= len(lines):
            break
        overlap_tokens = 0
        next_start = end - 1
        while next_start > start and overlap_tokens < chunk_overlap:
            overlap_tokens += line_tokens[next_start]
            next_start -= 1
        start = max(start + 1, next_start + 1)
        if start >= end:
            start = end
    return chunks


def build_fts_query(raw: str) -> Optional[str]:
    tokens = [token for token in WORD_PATTERN.findall(str(raw or "").lower()) if token.strip()]
    if not tokens:
        return None
    return " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += float(a) * float(b)
        norm_a += float(a) * float(a)
        norm_b += float(b) * float(b)
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def bm25_rank_to_score(rank: float) -> float:
    if not math.isfinite(rank):
        return 1.0 / 1000.0
    if rank < 0:
        relevance = -rank
        return relevance / (1.0 + relevance)
    return 1.0 / (1.0 + rank)


def to_decay_lambda(half_life_days: float) -> float:
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        return 0.0
    return math.log(2.0) / half_life_days


def apply_temporal_decay(
    score: float,
    *,
    session_time: str,
    reference_time: str,
    half_life_days: float,
) -> float:
    session_dt = parse_session_datetime(session_time)
    ref_dt = parse_session_datetime(reference_time)
    if session_dt is None or ref_dt is None:
        return score
    age_days = max(0.0, (ref_dt - session_dt).total_seconds() / 86400.0)
    lam = to_decay_lambda(half_life_days)
    if lam <= 0.0:
        return score
    return score * math.exp(-lam * age_days)


def tokenize_for_mmr(text: str) -> set[str]:
    return set(token for token in WORD_PATTERN.findall(str(text or "").lower()) if token.strip())


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(inter) / float(union or 1)


@dataclass
class MemoryChunk:
    id: str
    path: str
    source: str
    session_id: str
    session_time: str
    start_line: int
    end_line: int
    text: str
    hash: str


@dataclass
class SearchResult:
    id: str
    path: str
    source: str
    session_id: str
    session_time: str
    start_line: int
    end_line: int
    text: str
    score: float
    vector_score: float
    text_score: float


class OpenClawMemoryMVP:
    def __init__(
        self,
        *,
        workspace_dir: Path,
        index_db_path: Path,
        token_model: str = "gpt-4o",
        chunk_tokens: int = 400,
        chunk_overlap: int = 80,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        candidate_multiplier: int = 4,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.7,
        enable_temporal_decay: bool = False,
        half_life_days: float = 30.0,
        include_searchable_highlights: bool = True,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.index_db_path = index_db_path
        self.token_model = token_model
        self.chunk_tokens = max(1, int(chunk_tokens))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.vector_weight = float(vector_weight)
        self.text_weight = float(text_weight)
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.enable_mmr = bool(enable_mmr)
        self.mmr_lambda = float(mmr_lambda)
        self.enable_temporal_decay = bool(enable_temporal_decay)
        self.half_life_days = float(half_life_days)
        self.include_searchable_highlights = bool(include_searchable_highlights)

        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.index_db_path))
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def close(self) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                source TEXT,
                session_id TEXT,
                session_time TEXT,
                hash TEXT,
                updated_at REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT,
                source TEXT,
                session_id TEXT,
                session_time TEXT,
                start_line INTEGER,
                end_line INTEGER,
                hash TEXT,
                text TEXT,
                embedding TEXT,
                updated_at REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                id UNINDEXED,
                path UNINDEXED,
                source UNINDEXED,
                session_id UNINDEXED,
                session_time UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )
        self.conn.commit()

    def reset_index(self) -> None:
        self.conn.execute("DELETE FROM files")
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def build_corpus_files(
        self,
        corpus_rows: List[Dict[str, Any]],
        *,
        sources: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        memory_dir = self.workspace_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = self.workspace_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, Any]] = []
        used_paths: set[tuple[str, str]] = set()
        source_set = {str(source).strip() for source in sources if str(source).strip()}
        if not source_set:
            source_set = {SOURCE_SESSIONS}

        memory_md = self.workspace_dir / "MEMORY.md"
        memory_md.write_text("# Durable Memory\n\n", encoding="utf-8")
        for row in corpus_rows:
            session_id = str(row.get("session_id", "") or "")
            session_time = str(row.get("session_time", "") or "")
            date_stamp = normalize_session_date(session_time)
            for source in source_set:
                if source == SOURCE_MEMORY:
                    base_name = f"{date_stamp}-{safe_slug(session_id)}.md"
                    rel_path = f"memory/{base_name}"
                    raw_content = render_memory_markdown(
                        row,
                        include_searchable_highlights=self.include_searchable_highlights,
                    )
                    indexed_content = raw_content
                    line_map: Optional[List[int]] = None
                elif source == SOURCE_SESSIONS:
                    base_name = f"{safe_slug(session_id)}.jsonl"
                    rel_path = f"sessions/{base_name}"
                    raw_content, indexed_content, line_map = render_session_jsonl(row)
                else:
                    continue
                suffix = 2
                while (source, rel_path) in used_paths:
                    stem, ext = rel_path.rsplit(".", 1)
                    rel_path = f"{stem}-{suffix:02d}.{ext}"
                    suffix += 1
                used_paths.add((source, rel_path))
                abs_path = self.workspace_dir / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(raw_content, encoding="utf-8")
                entries.append(
                    {
                        "source": source,
                        "rel_path": rel_path,
                        "abs_path": abs_path,
                        "session_id": session_id,
                        "session_time": session_time,
                        "content": raw_content,
                        "indexed_content": indexed_content,
                        "line_map": line_map,
                        "hash": hash_text(raw_content),
                    }
                )
        return entries

    def build_chunks(self, file_entries: List[Dict[str, Any]]) -> List[MemoryChunk]:
        chunks: List[MemoryChunk] = []
        for entry in file_entries:
            line_chunks = chunk_markdown_by_lines(
                str(entry.get("indexed_content", entry["content"])),
                token_model=self.token_model,
                chunk_tokens=self.chunk_tokens,
                chunk_overlap=self.chunk_overlap,
            )
            line_map = list(entry.get("line_map") or [])
            for start_line, end_line, chunk_text in line_chunks:
                mapped_start = int(start_line)
                mapped_end = int(end_line)
                if line_map:
                    mapped = line_map[max(0, start_line - 1) : min(len(line_map), end_line)]
                    if mapped:
                        mapped_start = int(mapped[0])
                        mapped_end = int(mapped[-1])
                chunk_hash = hash_text(chunk_text)
                chunk_id = hash_text(
                    f"{entry['rel_path']}:{entry['session_id']}:{mapped_start}:{mapped_end}:{chunk_hash}"
                )
                chunks.append(
                    MemoryChunk(
                        id=chunk_id,
                        path=str(entry["rel_path"]),
                        source=str(entry["source"]),
                        session_id=str(entry["session_id"]),
                        session_time=str(entry["session_time"]),
                        start_line=mapped_start,
                        end_line=mapped_end,
                        text=chunk_text,
                        hash=chunk_hash,
                    )
                )
        return chunks

    def persist_index(
        self,
        *,
        file_entries: List[Dict[str, Any]],
        chunks: List[MemoryChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.reset_index()
        now = datetime.now(tz=timezone.utc).timestamp()
        for entry in file_entries:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO files (path, source, session_id, session_time, hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entry["rel_path"]),
                    str(entry["source"]),
                    str(entry["session_id"]),
                    str(entry["session_time"]),
                    str(entry["hash"]),
                    now,
                ),
            )
        for chunk, embedding in zip(chunks, embeddings):
            self.conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    id, path, source, session_id, session_time, start_line, end_line, hash, text, embedding, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.path,
                    chunk.source,
                    chunk.session_id,
                    chunk.session_time,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.hash,
                    chunk.text,
                    json.dumps(list(embedding)),
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO chunks_fts (text, id, path, source, session_id, session_time, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.text,
                    chunk.id,
                    chunk.path,
                    chunk.source,
                    chunk.session_id,
                    chunk.session_time,
                    chunk.start_line,
                    chunk.end_line,
                ),
            )
        self.conn.commit()

    def build_index_from_corpus(
        self,
        *,
        corpus_rows: List[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        sources: Sequence[str],
    ) -> Dict[str, Any]:
        file_entries = self.build_corpus_files(corpus_rows, sources=sources)
        chunks = self.build_chunks(file_entries)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: chunks={len(chunks)} embeddings={len(embeddings)}"
            )
        self.persist_index(file_entries=file_entries, chunks=chunks, embeddings=embeddings)
        return {
            "num_files": len(file_entries),
            "num_chunks": len(chunks),
            "workspace_dir": str(self.workspace_dir),
            "index_db_path": str(self.index_db_path),
            "sources": list(dict.fromkeys(str(source) for source in sources)),
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap": self.chunk_overlap,
        }

    def load_all_chunks(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT id, path, source, session_id, session_time, start_line, end_line, text, embedding
            FROM chunks
            """
        ).fetchall()

    def read_file(
        self,
        *,
        rel_path: str,
        from_line: int | None = None,
        lines: int | None = None,
    ) -> Dict[str, str]:
        requested = str(rel_path or "").strip().replace("\\", "/")
        if not requested:
            raise ValueError("path required")
        if requested.startswith("/") or requested.startswith("../") or "/../" in requested:
            raise ValueError("path required")
        allowed_roots = ("MEMORY.md", "memory/")
        if requested != "MEMORY.md" and not requested.startswith(allowed_roots[1]):
            raise ValueError("path required")
        abs_path = (self.workspace_dir / requested).resolve()
        workspace_root = self.workspace_dir.resolve()
        if workspace_root != abs_path and workspace_root not in abs_path.parents:
            raise ValueError("path required")
        try:
            text = abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"path": requested, "text": ""}
        if from_line is not None or lines is not None:
            raw_lines = text.splitlines()
            start_idx = max(0, int(from_line or 1) - 1)
            line_count = (
                max(0, int(lines))
                if lines is not None
                else max(0, len(raw_lines) - start_idx)
            )
            selected = raw_lines[start_idx : start_idx + line_count]
            text = "\n".join(selected)
        return {"path": requested, "text": text}

    def search_keyword(self, query: str, limit: int) -> List[SearchResult]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        rows = self.conn.execute(
            """
            SELECT id, path, source, session_id, session_time, start_line, end_line, text, bm25(chunks_fts) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [
            SearchResult(
                id=str(row["id"]),
                path=str(row["path"]),
                source=str(row["source"]),
                session_id=str(row["session_id"]),
                session_time=str(row["session_time"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                text=truncate_utf16_safe(str(row["text"]), SNIPPET_MAX_CHARS),
                score=bm25_rank_to_score(float(row["rank"])),
                vector_score=0.0,
                text_score=bm25_rank_to_score(float(row["rank"])),
            )
            for row in rows
        ]

    def search_vector(self, query_embedding: Sequence[float], limit: int) -> List[SearchResult]:
        rows = self.load_all_chunks()
        scored: List[SearchResult] = []
        for row in rows:
            embedding = json.loads(str(row["embedding"] or "[]"))
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                SearchResult(
                    id=str(row["id"]),
                    path=str(row["path"]),
                    source=str(row["source"]),
                    session_id=str(row["session_id"]),
                    session_time=str(row["session_time"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=truncate_utf16_safe(str(row["text"]), SNIPPET_MAX_CHARS),
                    score=score,
                    vector_score=score,
                    text_score=0.0,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def merge_results(
        self,
        *,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        reference_time: str = "",
    ) -> List[SearchResult]:
        if vector_results and keyword_results:
            merged_by_id: Dict[str, SearchResult] = {result.id: result for result in vector_results}
            for result in keyword_results:
                existing = merged_by_id.get(result.id)
                if existing is None:
                    merged_by_id[result.id] = result
                else:
                    existing.text_score = result.text_score
                    if result.text.strip():
                        existing.text = result.text
            merged: List[SearchResult] = []
            for result in merged_by_id.values():
                score = self.vector_weight * result.vector_score + self.text_weight * result.text_score
                if self.enable_temporal_decay and reference_time:
                    score = apply_temporal_decay(
                        score,
                        session_time=result.session_time,
                        reference_time=reference_time,
                        half_life_days=self.half_life_days,
                    )
                result.score = score
                merged.append(result)
            merged = sorted(merged, key=lambda item: item.score, reverse=True)
        elif vector_results:
            merged = sorted(vector_results, key=lambda item: item.score, reverse=True)
        else:
            merged = sorted(keyword_results, key=lambda item: item.score, reverse=True)
        if self.enable_mmr and len(merged) > 1:
            return self.apply_mmr(merged)
        return merged

    def select_results(
        self,
        *,
        merged_results: List[SearchResult],
        keyword_results: List[SearchResult],
        max_results: int,
        min_score: float,
    ) -> List[SearchResult]:
        strict = [entry for entry in merged_results if entry.score >= float(min_score)]
        if strict or not keyword_results:
            return strict[: max(1, int(max_results))]
        relaxed_min = min(float(min_score), float(self.text_weight))
        keyword_keys = {
            (entry.source, entry.path, entry.start_line, entry.end_line) for entry in keyword_results
        }
        relaxed = [
            entry
            for entry in merged_results
            if entry.score >= relaxed_min
            and (entry.source, entry.path, entry.start_line, entry.end_line) in keyword_keys
        ]
        return relaxed[: max(1, int(max_results))]

    def apply_mmr(self, results: List[SearchResult]) -> List[SearchResult]:
        lam = max(0.0, min(1.0, float(self.mmr_lambda)))
        if lam >= 1.0 or len(results) <= 1:
            return list(results)
        token_map = {result.id: tokenize_for_mmr(result.text) for result in results}
        max_score = max(result.score for result in results)
        min_score = min(result.score for result in results)
        score_range = max_score - min_score

        def normalize(score: float) -> float:
            if score_range <= 0:
                return 1.0
            return (score - min_score) / score_range

        selected: List[SearchResult] = []
        remaining = list(results)
        while remaining:
            best: Optional[SearchResult] = None
            best_score = -float("inf")
            for candidate in remaining:
                relevance = normalize(candidate.score)
                similarity = 0.0
                if selected:
                    similarity = max(
                        jaccard_similarity(token_map[candidate.id], token_map[picked.id])
                        for picked in selected
                    )
                mmr_score = lam * relevance - (1.0 - lam) * similarity
                if best is None or mmr_score > best_score or (
                    math.isclose(mmr_score, best_score) and candidate.score > best.score
                ):
                    best = candidate
                    best_score = mmr_score
            if best is None:
                break
            selected.append(best)
            remaining = [candidate for candidate in remaining if candidate.id != best.id]
        return selected
