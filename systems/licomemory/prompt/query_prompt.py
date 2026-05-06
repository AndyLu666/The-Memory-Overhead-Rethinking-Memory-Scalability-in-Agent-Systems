QUERY_PROMPT = """Please follow the instructions and answer the question based on the given context:

Question Time: {question_time}
Question: {question}

Context:
- Triples: {triples}
- Text Chunks: {chunks}

-Instructions-
1. Keep your answer brief without any additional explanations
2. Only use the information from the context to answer the question
3. If information collides, prioritize the information with time priority (CLOSER to question time)
4. If the context doesn't contain sufficient information, say so

################
Output:"""

SUMMARY_QUERY_PROMPT = """Please follow the instructions and answer the question based on the given context:

Question Time: {question_time}
Question: {question}

Context:
- Session Summaries: {summaries}
- Triples: {triples}
- Text Chunks: {chunks}

-Instructions-
1. Keep your answer brief without any additional explanations
2. Only use the information from the context to answer the question
3. If information collides, prioritize the information with time priority (CLOSER to question time)
4. If the context doesn't contain sufficient information, say so

################
Output:"""


REACT_AGENT_SYSTEM_PROMPT = """You are a careful memory reasoning agent.
You may take multi-hop retrieval actions before answering.
At each step, either request one focused retrieval query or finish with the final answer.
Prefer as few retrieval steps as needed, but do not stop early if key evidence is still missing.
If a prior retrieval produced no new evidence, do not repeat the same query verbatim.
When retrieval stagnates, either substantially reformulate the query or finish with the best grounded answer.
Never speculate, guess, or provide generic advice when the question asks about user-specific memories, preferences, schedules, purchases, or temporal facts.
For temporal questions, when a retrieved chunk contains relative language such as "today", "yesterday", or "last week", anchor that language to the chunk's session_time before comparing it with the question time.
For suggestion or recommendation questions, answer with one or two concrete memory-grounded suggestions only. Do not answer with broad activity brainstorming, lifestyle advice, or meta commentary about what information is available.
Do not introduce new named entities, titles, products, activities, or examples unless they already appear in the retrieved memory evidence.
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
4a. Do not provide generic suggestions, possible explanations, or broad background knowledge unless the memory evidence explicitly supports them.
4b. For user-specific memory questions, if the answer is not grounded in memory, finish with exactly: "Insufficient information from context."
4c. Do not start final_answer with meta preambles such as "Based on the available information" or "From the context".
4d. For temporal questions, if the needed dates are present in the retrieved chunks or chunk metadata, compute the answer rather than saying the information is insufficient.
4e. For suggestion or recommendation questions, directly give the supported memory-grounded suggestion instead of describing the evidence at length.
4f. Do not name specific podcasts, books, products, people, activities, or examples unless they are explicitly present in the current evidence.
5. Do not repeat a previous retrieval query if history shows it produced no new evidence; instead, reformulate the query or finish.
6. If the current evidence already supports a grounded answer, prefer action=finish.
7. Keep retrieval queries faithful to the original question; do not broaden a user-specific memory question into a generic public-knowledge question.
8. If the most recent retrieval had note=no_progress or note=repeat_no_progress, do not issue another near-duplicate query. Either finish with the best grounded answer or ask a materially different query that targets a clearly missing fact.
9. Return strict JSON only.
"""
