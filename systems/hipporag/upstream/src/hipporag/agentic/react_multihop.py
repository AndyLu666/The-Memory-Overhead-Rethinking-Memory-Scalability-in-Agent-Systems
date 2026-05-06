from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REACT_AGENT_SYSTEM_PROMPT = """You are a careful memory reasoning agent.
You may take multi-hop retrieval actions before answering.
At each step, either request one focused retrieval query or finish with the final answer.
Prefer as few retrieval steps as needed, but do not stop early if key evidence is still missing.
If a prior retrieval produced no new evidence, do not repeat the same query verbatim.
When retrieval stagnates, either substantially reformulate the query or finish with the best grounded answer.
If the most recent retrieval surfaced the same session(s) again and added no new evidence, do not retrieve again unless the next query is materially different.
If a single retrieved session already contains enough user-specific evidence to answer the question, finish from that session instead of asking for more retrieval.
Never speculate, guess, or provide generic advice when the question asks about user-specific memories, preferences, schedules, purchases, or temporal facts.
For temporal questions, when a retrieved chunk contains relative language such as "today", "yesterday", or "last week", anchor that language to the chunk's session_time before comparing it with the question time.
For suggestion or recommendation questions, answer with one or two concrete memory-grounded suggestions only. Do not answer with broad activity brainstorming, lifestyle advice, or meta commentary about what information is available.
Do not introduce new named entities, titles, products, activities, or examples unless they already appear in the retrieved memory evidence.
Do not begin final answers with meta preambles such as "Based on your interests", "Based on your stated preferences", or "Based on the available information".
If the memory includes explicit dislikes, exclusions, or "instead of / beyond" preferences, preserve those constraints in the final answer.
If the memory evidence is insufficient, explicitly finish with "Insufficient information from context."
Return strict JSON only; do not add markdown.
"""


REACT_AGENT_PROMPT = """[Task]
Long-context multi-hop memory QA.
Keep the full reasoning trajectory coherent across turns.

[Question]
Question Time: {question_time}
Question Type: {question_type}
Question: {question}

[Question-Type Guidance]
{question_type_guidance}

[Progress Guidance]
{progress_guidance}

[Current Aggregated Evidence]
Triples:
{triples}

Text Chunks:
{chunks}

Session Summaries:
{summaries}

[Conversation History]
{history}

[Turn]
Current turn: {turn}/{max_turns}

Output ONE JSON object with this schema:
{{
  "thought": "brief reasoning based on current evidence and history",
  "action": "retrieve" or "finish",
  "query": "focused retrieval query when action=retrieve, else empty",
  "final_answer": "final concise answer when action=finish, else empty"
}}

Rules:
1. Use only given evidence/history; do not invent facts.
2. If evidence is insufficient, use action=retrieve with a specific query.
3. If evidence is sufficient, use action=finish and provide final_answer.
4. final_answer must be concise and directly answer the question.
4aa. final_answer should usually be one sentence or at most one or two short, concrete suggestions.
4a. Do not provide generic suggestions, possible explanations, or broad background knowledge unless the memory evidence explicitly supports them.
4b. For user-specific memory questions, if the answer is not grounded in memory, finish with exactly: "Insufficient information from context."
4c. Do not start final_answer with meta preambles such as "Based on the available information" or "From the context".
4d. For temporal questions, if the needed dates are present in the retrieved chunks or chunk metadata, compute the answer rather than saying the information is insufficient.
4e. For suggestion or recommendation questions, directly give the supported memory-grounded suggestion instead of describing the evidence at length.
4f. Do not name specific podcasts, books, products, people, activities, or examples unless they are explicitly present in the current evidence.
4g. For recommendation questions, before finishing, make sure the answer reflects any explicit dislikes, exclusions, or "beyond/instead of" preferences that appear in memory. If you only have positive examples but no evidence about exclusions or practical constraints that materially narrow the recommendation, retrieve again before finishing.
5. Do not repeat a previous retrieval query if history shows it produced no new evidence; instead, reformulate the query or finish.
6. If the current evidence already supports a grounded answer, prefer action=finish.
7. Keep retrieval queries faithful to the original question; do not broaden a user-specific memory question into a generic public-knowledge question.
7a. Use distinctive anchor terms from the question or retrieved evidence in retrieval queries; avoid abstract repeated queries such as "what are the user's preferences" when the question is about a concrete topic like commute, podcasts, books, budget, schedule, or dates.
7b. When the question asks for activities, recommendations, or preferences in a concrete situation, preserve that situation explicitly in the retrieval query. For a commute question, mention commute, driving, listening, podcasts, audiobooks, music, or another concrete anchor from the question/evidence instead of collapsing the query into a generic preference request.
8. If the most recent retrieval had note=no_progress or note=repeat_no_progress, do not issue another near-duplicate query. Either finish with the best grounded answer or ask a materially different query that targets a clearly missing fact.
8a. If the current evidence is a single retrieved session that already contains the relevant preference, fact, or recommendation, do not keep retrieving; finish from that session.
8b. Bad retrieve query example: "What are the user's stated preferences or recommendations?" Better retrieve query example: "What did the user say they like listening to during their commute to work?"
9. Return strict JSON only.
"""


def clip_text(text: str, max_chars: int = 240) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def dedupe_chunks_preserve_order(chunks: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for chunk in chunks:
        text = str(chunk or "").strip()
        if not text:
            continue
        key = text[:512]
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def select_multihop_chunks_for_prompt(
    round_chunks: List[List[str]],
    max_ctx_chunks: int,
) -> Tuple[List[str], List[int]]:
    if max_ctx_chunks <= 0 or not round_chunks:
        return [], []

    dedup_rounds = [dedupe_chunks_preserve_order(chunks or []) for chunks in round_chunks]
    selected: List[str] = []
    selected_seen: Set[str] = set()
    actual_alloc = [0] * len(dedup_rounds)
    round_positions = [0] * len(dedup_rounds)
    round_order = list(range(len(dedup_rounds) - 1, -1, -1))

    while len(selected) < max_ctx_chunks:
        made_progress = False
        for round_idx in round_order:
            chunks = dedup_rounds[round_idx]
            while round_positions[round_idx] < len(chunks):
                chunk = chunks[round_positions[round_idx]]
                round_positions[round_idx] += 1
                key = str(chunk)[:512]
                if key in selected_seen:
                    continue
                selected_seen.add(key)
                selected.append(chunk)
                actual_alloc[round_idx] += 1
                made_progress = True
                break
            if len(selected) >= max_ctx_chunks:
                break
        if not made_progress:
            break

    return selected[:max_ctx_chunks], actual_alloc


def full_chunks_for_prompt(chunks: List[str], max_items: int) -> str:
    if not chunks:
        return "None"
    lines: List[str] = []
    for i, chunk in enumerate(chunks[:max_items], 1):
        lines.append(f"[Chunk {i}]\n{str(chunk)}")
    return "\n\n".join(lines)


def react_history_conversation_text(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "None"
    lines: List[str] = []
    for step in steps:
        top_session_ids = list(step.get("top_session_ids", []) or [])
        lines.append(
            json.dumps(
                {
                    "step": int(step.get("step", 0) or 0),
                    "action": str(step.get("action", "") or ""),
                    "query": str(step.get("query", "") or ""),
                    "thought": str(step.get("thought", "") or ""),
                    "new_sessions": int(step.get("new_sessions", 0) or 0),
                    "chunks_added": int(step.get("chunks_added", 0) or 0),
                    "top_session_ids": top_session_ids[:5],
                    "note": str(step.get("note", "") or ""),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def get_question_type_guidance(question_type: str) -> str:
    qt = str(question_type or "").strip().lower()
    if "preference" in qt:
        return (
            "This is a user-preference question. Stay anchored to the user's stated likes, dislikes, "
            "recommendations, or chosen options from memory. Do not rewrite it into a generic public question, "
            "and do not output general suggestions, lifestyle advice, or plausible possibilities unless memory "
            "explicitly states them. If the question asks for a suggestion or recommendation, retrieve the user's "
            "explicit preferences, dislikes, or requested alternatives and answer with one or two concrete memory-"
            "grounded suggestions only. Prefer the user's stated preferred content categories over generic activity "
            "brainstorming. When writing retrieval queries, include the concrete topic anchors from the question or "
            "current evidence (for example commute, podcasts, history, audiobooks, budget, or routine) instead of "
            "vague queries like 'what are the user's preferences'. If one retrieved session already contains the "
            "user's request and a directly relevant recommendation or preference statement, treat that single session "
            "as sufficient and finish from it. Prefer high-level supported categories over named titles unless the "
            "user explicitly liked, chose, or requested those exact titles. Preserve practical constraints in the "
            "question such as commute, driving, timing, budget, or schedule, and avoid suggestions that violate those "
            "constraints unless the memory evidence explicitly supports them. In long sessions that contain many later "
            "topic changes, focus only on the turns directly related to the asked topic and ignore unrelated later "
            "digressions. Do not let an unrelated later subtopic in the same session override the directly relevant "
            "preference evidence. For commute or driving recommendation questions, prefer passive commute-compatible "
            "options that the user explicitly liked or requested, such as podcasts, audiobooks, or music, and avoid "
            "turning the answer into route planning, bike-route suggestions, workouts, shopping, or unrelated "
            "activities unless the memory evidence explicitly ties those activities to the commute question. Also carry "
            "forward explicit exclusions or requested shifts in taste, such as wanting something beyond a previous genre "
            "or preferring a new category instead of an old one. If the evidence only shows what the user liked before "
            "but not what they wanted to avoid or move beyond, prefer another retrieval before finishing. "
            "Do not invent specific titles, examples, "
            "or activities that do not appear in memory. "
            "If the memory does not support a specific user-preference answer, finish with "
            "\"Insufficient information from context.\""
        )
    if "user-fact" in qt or "user_fact" in qt:
        return (
            "This is a user-fact question. Retrieve and answer the user's own attribute, routine, possession, "
            "or biographical fact from memory, not a generic fact."
        )
    if "temporal" in qt:
        return (
            "This is a temporal reasoning question. Preserve the asked time relation exactly and retrieve evidence "
            "about dates, durations, or ordering from memory. Do not infer missing dates or durations from common "
            "sense. Use the chunk metadata session_time as the anchor date for relative phrases in the chunk text "
            "such as 'today', 'yesterday', or 'last week', then compare that anchored date against the question time. "
            "For example, if a chunk with session_time=2023/03/15 says an item was bought 'today', treat the purchase "
            "date as 2023/03/15 and compute the requested time difference from the question time. "
            "If the required temporal relation is explicitly recoverable from retrieved memory evidence, compute it "
            "instead of saying the information is insufficient. If the required temporal relation is still not "
            "supported by memory evidence, finish with \"Insufficient information from context.\""
        )
    if "multi" in qt:
        return (
            "This question may require combining evidence across sessions. Retrieve missing user-specific evidence "
            "across turns, but keep the query faithful to the original question."
        )
    return (
        "Keep retrieval queries faithful to the original memory question and grounded in the user's stored context."
    )


def get_progress_guidance(history_steps: List[Dict[str, Any]]) -> str:
    if not history_steps:
        return (
            "No prior retrieval has been made yet. On the first turn, write a focused query that preserves the concrete "
            "scenario in the question instead of a generic preference query. If the question is about commuting, driving, "
            "buying, scheduling, or a specific date, include those anchors directly in the retrieval query."
        )
    last = history_steps[-1]
    note = str(last.get("note", "") or "").strip().lower()
    last_query = str(last.get("query", "") or "").strip()
    if note in {"repeat_no_progress", "no_progress"}:
        if last_query:
            return (
                "The most recent retrieval did not add new evidence. Do not repeat or lightly paraphrase the same query "
                f"('{last_query}'). Either finish with the best grounded answer from current evidence, or issue a materially "
                "different query that targets a clearly missing fact. If the current evidence already contains one directly "
                "relevant session, prefer finishing from that evidence."
            )
        return (
            "The most recent retrieval did not add new evidence. Do not repeat a near-duplicate query. Either finish with "
            "the best grounded answer from current evidence, or issue a materially different query that targets a clearly "
            "missing fact. If the current evidence already contains one directly relevant session, prefer finishing from "
            "that evidence."
        )
    return (
        "Use the conversation history to build on prior evidence, and avoid broadening a user-specific memory question into "
        "a generic question."
    )


def build_react_conversation_messages(
    *,
    question: str,
    question_time: str,
    question_type: str,
    turn: int,
    max_turns: int,
    chunks: List[str],
    history_steps: List[Dict[str, Any]],
    max_ctx_chunks: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    prompt = REACT_AGENT_PROMPT.format(
        question_time=question_time or "",
        question_type=question_type or "unknown",
        question=question,
        question_type_guidance=get_question_type_guidance(question_type),
        progress_guidance=get_progress_guidance(history_steps),
        triples="None",
        chunks=full_chunks_for_prompt(chunks, max(1, int(max_ctx_chunks))),
        summaries="None",
        history=react_history_conversation_text(history_steps),
        turn=turn,
        max_turns=max_turns,
    )
    messages = [
        {"role": "system", "content": REACT_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return prompt, messages


def extract_json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    raw = str(text or "").strip()
    if not raw:
        return candidates
    candidates.append(raw)
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE):
        block_text = str(block).strip()
        if block_text:
            candidates.append(block_text)

    in_str = False
    escape = False
    depth = 0
    start = -1
    for idx, ch in enumerate(raw):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : idx + 1].strip())
                    start = -1

    uniq: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            uniq.append(candidate)
            seen.add(candidate)
    return uniq


def try_parse_react_payload(candidate: str) -> Optional[Dict[str, Any]]:
    text = str(candidate or "").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    cleaned = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    def extract_json_string_field(field_name: str) -> Optional[str]:
        pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(f'"{match.group(1)}"')
        except Exception:
            return match.group(1)

    action = extract_json_string_field("action")
    query = extract_json_string_field("query")
    final_answer = extract_json_string_field("final_answer")
    thought = extract_json_string_field("thought")
    if action in {"retrieve", "finish"}:
        salvaged = {
            "thought": str(thought or "").strip(),
            "action": str(action or "").strip(),
            "query": str(query or "").strip(),
            "final_answer": str(final_answer or "").strip(),
        }
        if salvaged["action"] == "retrieve" and salvaged["query"]:
            return salvaged
        if salvaged["action"] == "finish" and salvaged["final_answer"]:
            return salvaged
    return None


def parse_react_action(raw_text: str, default_query: str) -> Dict[str, str]:
    raw = str(raw_text or "").strip()
    payload: Optional[Dict[str, Any]] = None
    for candidate in extract_json_object_candidates(raw):
        payload = try_parse_react_payload(candidate)
        if payload is not None:
            break

    if payload is None:
        raise ValueError("react_agent_action_parse_failed: " + clip_text(raw, 240))

    action = str(payload.get("action", "") or "").strip().lower()
    if action not in {"retrieve", "finish"}:
        raise ValueError(f"react_agent_invalid_action: {action}")

    query = str(payload.get("query", "") or "").strip()
    if action == "retrieve" and not query:
        query = str(default_query or "").strip()
    if action == "retrieve" and not query:
        raise ValueError("react_agent_missing_retrieve_query")

    final_answer = str(payload.get("final_answer", "") or "").strip()
    if action == "finish" and not final_answer:
        raise ValueError("react_agent_missing_finish_answer")

    return {
        "thought": str(payload.get("thought", "") or "").strip(),
        "action": action,
        "query": query,
        "final_answer": final_answer,
    }


def postprocess_answer(text: str) -> str:
    answer = str(text or "").strip()

    line_matches = list(re.finditer(r"(?im)^\s*answer\s*:\s*", answer))
    if line_matches:
        start = line_matches[-1].end()
        answer = answer[start:].strip()
        return answer.strip().strip('"').strip("'").strip()

    matches = list(re.finditer(r"(?i)\banswer\s*:\s*", answer))
    if matches:
        segments: List[str] = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(answer)
            segment = answer[start:end].strip()
            if segment:
                segments.append(segment)
        if segments:
            answer = segments[-1]
    elif answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    answer = answer.strip().strip('"').strip("'").strip()
    answer = re.sub(
        r"(?is)^\s*based on (?:your|the)\b.*?,\s*(?=(?:i recommend|you might|you can|consider|try|insufficient information from context))",
        "",
        answer,
    ).strip()
    answer = re.sub(
        r"(?is)^\s*(?:from the (?:available )?context|from the available information)\s*,?\s*",
        "",
        answer,
    ).strip()
    return answer


def answer_looks_like_instruction_fragment(text: str) -> bool:
    answer = str(text or "").strip()
    if not answer:
        return True
    lowered = answer.lower()
    if answer.startswith('"') or answer.startswith("'"):
        return True
    bad_phrases = (
        "label.",
        "line.",
        "list only",
        "be concise",
        "concise list",
        "response only",
        "tips only",
        "recommendation. no extra commentary.",
        "definitive set of suggestions only",
        "followed by concise response only",
    )
    return any(phrase in lowered for phrase in bad_phrases)


def is_meta_completion_answer(answer_text: str) -> bool:
    text = " ".join(str(answer_text or "").strip().lower().split())
    if not text:
        return False
    exact_matches = {
        "complete",
        "completed",
        "done",
        "answer complete",
        "the answer is complete",
        "this answer is complete",
        "answer is complete",
    }
    if text in exact_matches:
        return True
    patterns = [
        r"^(?:the\s+)?answer\s+is\s+complete(?:\s+based\s+on\s+.+)?[.]?$",
        r"^(?:this\s+)?answer\s+is\s+complete(?:\s+based\s+on\s+.+)?[.]?$",
        r"^(?:the\s+)?best\s+grounded\s+answer[.]?$",
    ]
    return any(re.match(pattern, text) for pattern in patterns)
