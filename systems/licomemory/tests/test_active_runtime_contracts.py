import asyncio
import json
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.llm_evaluator import LLMEvaluator
from memos_stats import sanitize_trace_record_for_export
from query.query_processor import QueryProcessor


ACTIVE_Q1_CONFIG = REPO_ROOT / "config" / "licomemory_locomo_multihop282_fixedgroup_lmealign_qwen3_8b_ifopen_query_q1_llmeval_gpt4omini_memos_20260317.yaml"
ACTIVE_DATASET_SUMMARY = REPO_ROOT / "scripts" / "dataset_lists" / "locomo_multihop282_fixedgroup_sbins_lmealign_20260315" / "summary.json"
ACTIVE_RUNNER = REPO_ROOT / "scripts" / "run_longmemeval_runner.py"
ACTIVE_MAIN = REPO_ROOT / "main.py"


class _FakeLLM:
    async def extract_entities(self, question, session_time=""):
        raise RuntimeError("synthetic failure")


class QueryProcessorActiveContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qp = QueryProcessor.__new__(QueryProcessor)
        self.qp.llm = _FakeLLM()

    def test_meta_completion_answer_detection(self) -> None:
        self.assertTrue(QueryProcessor._is_meta_completion_answer("complete"))
        self.assertTrue(
            QueryProcessor._is_meta_completion_answer(
                "The answer is complete based on the available information."
            )
        )
        self.assertFalse(QueryProcessor._is_meta_completion_answer("Rome"))

    def test_postprocess_rejects_meta_completion_answers(self) -> None:
        self.assertEqual(
            QueryProcessor._postprocess_answer(
                "complete",
                "locomo-multi-hop",
                question="What items did John mention having as a child?",
            ),
            "",
        )

    def test_parse_react_action_requires_json(self) -> None:
        with self.assertRaises(ValueError):
            self.qp._parse_react_action("not json", "fallback query")

    def test_parse_react_action_retrieve_requires_query(self) -> None:
        with self.assertRaises(ValueError):
            self.qp._parse_react_action(
                '{"thought":"need more","action":"retrieve","query":"","final_answer":""}',
                "fallback query",
            )

    def test_parse_react_action_finish_requires_final_answer(self) -> None:
        with self.assertRaises(ValueError):
            self.qp._parse_react_action(
                '{"thought":"done","action":"finish","query":"","final_answer":""}',
                "fallback query",
            )

    def test_parse_react_action_salvages_malformed_retrieve_json(self) -> None:
        parsed = self.qp._parse_react_action(
            '{"\n  : "Need more evidence.", "action": "retrieve", "query": "Who is Caroline and what did she research?", "final_answer": ""}',
            "fallback query",
        )
        self.assertEqual(parsed["action"], "retrieve")
        self.assertEqual(parsed["query"], "Who is Caroline and what did she research?")

    def test_parse_react_action_salvages_malformed_finish_json(self) -> None:
        parsed = self.qp._parse_react_action(
            '{"\n  : "Enough evidence.", "action": "finish", "query": "", "final_answer": "Adoption agencies"}',
            "fallback query",
        )
        self.assertEqual(parsed["action"], "finish")
        self.assertEqual(parsed["final_answer"], "Adoption agencies")

    def test_extract_query_entities_no_heuristic_fallback(self) -> None:
        entities = asyncio.run(self.qp._extract_query_entities("Which city did Jean and John both visit?"))
        self.assertEqual(entities, [])


class MemOSJudgeContractTests(unittest.TestCase):
    def test_prompt_selection_matches_task_family(self) -> None:
        evaluator = LLMEvaluator.__new__(LLMEvaluator)
        evaluator.eval_prompt_style = "memos_json"

        locomo_prompt = evaluator.get_anscheck_prompt(
            "locomo-multi-hop",
            "What did I get in Hawaii?",
            "A shell necklace",
            "A shell necklace",
        )
        lme_prompt = evaluator.get_anscheck_prompt(
            "single-session-user",
            "Where did I buy my new tennis racket from?",
            "the sports store downtown",
            "the sports store downtown",
        )

        self.assertIn("Hawaii", locomo_prompt)
        self.assertIn("shell necklace", locomo_prompt)
        self.assertIn("tennis racket", lme_prompt)
        self.assertIn("sports store downtown", lme_prompt)

    def test_memos_system_prompt_is_enabled(self) -> None:
        evaluator = LLMEvaluator.__new__(LLMEvaluator)
        evaluator.eval_prompt_style = "memos_json"
        self.assertEqual(
            evaluator._get_eval_system_prompt(strict=False),
            "You are an expert grader that determines if answers to questions match a gold standard answer",
        )

    def test_parse_memos_json_label(self) -> None:
        evaluator = LLMEvaluator.__new__(LLMEvaluator)
        self.assertTrue(evaluator._parse_memos_json_label('{"label":"CORRECT"}'))
        self.assertFalse(evaluator._parse_memos_json_label('{"label":"WRONG"}'))
        self.assertIsNone(evaluator._parse_memos_json_label('{"label":"MAYBE"}'))


class MemOSStatsSanitizationTests(unittest.TestCase):
    def test_sanitize_trace_record_removes_legacy_query_fields(self) -> None:
        sanitized = sanitize_trace_record_for_export(
            {
                "query_summary": {"retrieval_time": 1.0},
                "cost_summary": {"total_query_tokens": 123},
                "response_duration_sec": 1.5,
                "react_trace": [
                    {
                        "step": 1,
                        "action": "retrieve",
                        "llm_prompt_tokens_delta": 11,
                        "llm_completion_tokens_delta": 7,
                    }
                ],
            },
            row={
                "response_duration_ms": 100.0,
                "search_duration_ms": 25.0,
                "total_duration_ms": 125.0,
                "context_tokens": 42,
            },
        )
        self.assertIn("memos_stats", sanitized)
        self.assertNotIn("query_summary", sanitized)
        self.assertNotIn("cost_summary", sanitized)
        self.assertNotIn("response_duration_sec", sanitized)
        self.assertNotIn("llm_prompt_tokens_delta", sanitized["react_trace"][0])
        self.assertNotIn("llm_completion_tokens_delta", sanitized["react_trace"][0])

    def test_sanitize_trace_record_promotes_finish_answer(self) -> None:
        sanitized = sanitize_trace_record_for_export(
            {
                "output": "Caroline researched adoption agencies.",
                "react_trace": [
                    {"step": 1, "action": "retrieve", "query": "Caroline research"},
                    {
                        "step": 2,
                        "action": "finish",
                        "final_answer": "Caroline researched adoption agencies.",
                    },
                ],
            },
            row={
                "response_duration_ms": 100.0,
                "search_duration_ms": 25.0,
                "total_duration_ms": 125.0,
                "context_tokens": 42,
            },
        )
        self.assertEqual(
            sanitized["final_answer"],
            "Caroline researched adoption agencies.",
        )


class ActiveConfigAndDatasetContractTests(unittest.TestCase):
    def test_active_q1_config_matches_requested_contract(self) -> None:
        cfg = yaml.safe_load(ACTIVE_Q1_CONFIG.read_text(encoding="utf-8"))
        retriever = cfg["retriever"]
        evaluation = cfg["evaluation"]
        self.assertTrue(retriever["enable_react_multihop"])
        self.assertNotIn("react_controller_temperature", retriever)
        self.assertEqual(evaluation["eval_model"], "gpt-4o-mini")
        self.assertEqual(evaluation["eval_prompt_style"], "memos_json")
        self.assertEqual(evaluation["eval_num_runs"], 3)
        self.assertNotIn("enable_global_semantic_fallback", retriever)
        self.assertNotIn("enable_direct_chunk_fallback", retriever)
        self.assertNotIn("lexical_fallback_limit", retriever)

    def test_dataset_summary_matches_irrelevant_session_contract(self) -> None:
        summary = json.loads(ACTIVE_DATASET_SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["memory_unit"], "group")
        self.assertEqual(summary["noise_mode"], "outdomain_longmemeval")
        self.assertEqual(summary["alignment_target"], "longmemeval_fixed2k")
        self.assertEqual(summary["bins"], [0, 100, 200, 300, 400])

    def test_runner_uses_thread_pool_for_external_subprocess_work(self) -> None:
        runner_source = ACTIVE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("ThreadPoolExecutor", runner_source)
        self.assertNotIn("ProcessPoolExecutor", runner_source)

    def test_runner_forces_query_fail_fast_in_child_processes(self) -> None:
        runner_source = ACTIVE_RUNNER.read_text(encoding="utf-8")
        self.assertIn('child_env["LICOMEMORY_FAIL_FAST_QUERY_ERRORS"] = "1"', runner_source)


class QueryErrorOutputGuardTests(unittest.TestCase):
    def test_main_contains_query_error_output_guard(self) -> None:
        main_source = ACTIVE_MAIN.read_text(encoding="utf-8")
        self.assertIn("def _is_query_error_output", main_source)
        self.assertIn("react_agent_action_parse_failed", main_source)
        self.assertIn("Query failed before evaluation", main_source)


if __name__ == "__main__":
    unittest.main()
