import json
import asyncio
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from base.llm import LLMManager

class LLMEvaluator:

    MEMOS_LOCOMO_PROMPT = (
        "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:\n"
        "    (1) a question (posed by one user to another user),\n"
        "    (2) a 'gold' (ground truth) answer,\n"
        "    (3) a generated answer\n"
        "which you will score as CORRECT/WRONG.\n\n"
        "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
        "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
        "Question: Do you remember what I got the last time I went to Hawaii?\n"
        "Gold answer: A shell necklace\n"
        "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
        "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references "
        '(like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, '
        'it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n\n'
        "Now it's time for the real question:\n"
        "Question: {}\n"
        "Gold answer: {}\n"
        "Generated answer: {}\n\n"
        "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.\n"
        "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
        'Just return the label CORRECT or WRONG in a json format with the key as "label".'
    )

    MEMOS_LME_PROMPT = (
        "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:\n"
        "    (1) a question (posed by one user to another user),\n"
        "    (2) a 'gold' (ground truth) answer,\n"
        "    (3) a generated answer\n"
        "which you will score as CORRECT/WRONG.\n\n"
        "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
        "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
        "Question: Where did I buy my new tennis racket from?\n"
        "Gold answer: the sports store downtown\n"
        "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
        "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references "
        '(like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, '
        'it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n\n'
        "Now it's time for the real question:\n"
        "Question: {}\n"
        "Gold answer: {}\n"
        "Generated answer: {}\n\n"
        "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.\n"
        "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
        'Just return the label CORRECT or WRONG in a json format with the key as "label".'
    )

    def __init__(self, config: Config, results_path: str, dataset_name: str):
        self.config = config
        self.results_path = Path(results_path)
        self.dataset_name = dataset_name
        self.enable_llm_eval = config.evaluation.enable_llm_eval
        self.eval_prompt_style = str(
            getattr(config.evaluation, "eval_prompt_style", "licomemory_yesno") or "licomemory_yesno"
        ).strip().lower()
        self.eval_num_runs = max(1, int(getattr(config.evaluation, "eval_num_runs", 1) or 1))
        
        if self.enable_llm_eval:
            eval_api_key = (
                config.evaluation.eval_api_key
                or os.getenv("OPENAI_API_KEY", "").strip()
                or os.getenv("GPT_API_KEY", "").strip()
                or config.llm.api_key
            )
            eval_base_url = (
                config.evaluation.eval_base_url
                or os.getenv("OPENAI_BASE_URL", "").strip()
                or os.getenv("GPT_BASE_URL", "").strip()
                or "https://api.openai.com/v1"
            )
            eval_timeout = (
                config.evaluation.eval_timeout
                if getattr(config.evaluation, "eval_timeout", 0) and config.evaluation.eval_timeout > 0
                else config.llm.timeout
            )
            self.eval_llm = LLMManager(
                api_key=eval_api_key,
                model=config.evaluation.eval_model,
                max_tokens=config.evaluation.eval_max_tokens,
                base_url=eval_base_url,
                enable_concurrent=False,  # Disable concurrent for evaluation stability
                max_concurrent=1,
                timeout=eval_timeout,
                retry_attempts=config.llm.retry_attempts,
                retry_backoff=config.llm.retry_backoff,
                retry_backoff_max=config.llm.retry_backoff_max,
                fail_on_error=True,
            )
            self.eval_temperature = config.evaluation.eval_temperature
            self.eval_max_tokens = config.evaluation.eval_max_tokens
            logger.info(
                "LLM Evaluator initialized with model=%s, base_url=%s, timeout=%s, prompt_style=%s",
                config.evaluation.eval_model,
                eval_base_url,
                eval_timeout,
                self.eval_prompt_style,
            )
        else:
            self.eval_llm = None
            logger.info("LLM evaluation disabled, using exact matching")

    def get_anscheck_prompt(
        self,
        task: str,
        question: str,
        answer: str,
        response: str,
        strict: bool = False,
    ) -> str:
        task_norm = str(task or "").strip().lower()
        if strict:
            if task_norm == 'single-session-preference':
                template = (
                    "I will give you a question, a rubric for the desired personalized response, and a "
                    "response from a model. Apply a strict grading policy. Answer yes only if the model "
                    "response satisfies the rubric without adding contradictory personalization. Answer no "
                    "if the response misses the key requested preference or relies on unsupported personal "
                    "details. Answer yes or no only.\n\n"
                    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                    "Is the model response strictly correct? Answer yes or no only."
                )
                return template.format(question, answer, response)

            template = (
                "I will give you a question, a correct answer, and a response from a model. Apply a strict "
                "grading policy. Answer yes only if the model response gives the correct answer with the same "
                "scope as the gold answer. Minor wording differences or harmless spelling variations are fine, "
                "but do not accept extra entities, extra items, extra cities, extra events, extra reasons, "
                "extra dates, or speculative additions beyond the gold answer. For list/set questions, all and "
                "only the gold items must be present; order does not matter. If the response is broader than the "
                "gold answer, partially correct, only semantically related, or adds unsupported details, answer no."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response strictly correct? Answer yes or no only."
            )
            return template.format(question, answer, response)

        if self.eval_prompt_style == "memos_json":
            if task_norm.startswith("locomo-"):
                template = self.MEMOS_LOCOMO_PROMPT
            else:
                template = self.MEMOS_LME_PROMPT
            return template.format(question, answer, response)

        if task_norm in [
            'single-session-user',
            'single-session-assistant',
            'multi-session',
            'locomo-single-hop',
            'locomo-multi-hop',
            'locomo-commonsense',
            'locomo-adversarial',
        ]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task_norm in ['temporal-reasoning', 'locomo-temporal']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task_norm == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task_norm == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
            logger.warning(f"Unknown task type '{task}', using default prompt template")
        
        return prompt

    def _get_eval_system_prompt(self, strict: bool = False) -> Optional[str]:
        if strict:
            return None
        if self.eval_prompt_style == "memos_json":
            return "You are an expert grader that determines if answers to questions match a gold standard answer"
        return None

    async def _evaluate_with_llm_once(
        self,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        strict: bool = False,
    ) -> bool:
        try:
            prompt = self.get_anscheck_prompt(
                question_type,
                question,
                answer,
                response,
                strict=strict,
            )
            system_prompt = self._get_eval_system_prompt(strict=strict)
            eval_response = await self.eval_llm.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=self.eval_temperature,
                max_tokens=self.eval_max_tokens
            )

            if self.eval_prompt_style == "memos_json" and not strict:
                label = self._parse_memos_json_label(eval_response)
            else:
                label = self._parse_binary_label(eval_response)
            if label is None:
                raise ValueError(
                    "memos_judge_parse_failed"
                    if self.eval_prompt_style == "memos_json" and not strict
                    else "llm_eval_parse_failed"
                )

            logger.debug(f"LLM evaluation - Question: {question[:100]}")
            logger.debug(f"Expected: {answer}")
            logger.debug(f"Response: {response[:100]}")
            logger.debug(f"LLM says: {str(eval_response).strip()} -> {label}")
            
            return label
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            raise

    def _parse_memos_json_label(self, text: str) -> Optional[bool]:
        match = re.search(r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}', text or "", flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).strip().lower()
        if value == "correct":
            return True
        if value == "wrong":
            return False
        return None

    async def evaluate_with_llm_bundle(
        self,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        strict: bool = False,
    ) -> Dict[str, Any]:
        num_runs = 1 if strict else self.eval_num_runs
        judgments: Dict[str, bool] = {}
        for run_idx in range(1, num_runs + 1):
            judgments[f"judgment_{run_idx}"] = await self._evaluate_with_llm_once(
                question=question,
                answer=answer,
                response=response,
                question_type=question_type,
                strict=strict,
            )

        positive = sum(1 for value in judgments.values() if value)
        majority_label = positive > (len(judgments) / 2.0)
        return {
            "majority_label": majority_label,
            "judgments": judgments,
        }

    async def evaluate_with_llm(
        self,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        strict: bool = False,
    ) -> bool:
        bundle = await self.evaluate_with_llm_bundle(
            question=question,
            answer=answer,
            response=response,
            question_type=question_type,
            strict=strict,
        )
        return bool(bundle["majority_label"])

    async def evaluate_with_llm_strict(
        self,
        question: str,
        answer: str,
        response: str,
        question_type: str,
    ) -> bool:
        return await self.evaluate_with_llm(
            question=question,
            answer=answer,
            response=response,
            question_type=question_type,
            strict=True,
        )

    def _parse_binary_label(self, text: str) -> Optional[bool]:
        """Parse yes/no or CORRECT/WRONG labels from evaluator output."""
        s = (text or "").strip().lower()
        if not s:
            return None

        json_match = re.search(r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}', text or "", flags=re.IGNORECASE)
        if json_match:
            value = json_match.group(1).strip().lower()
            if value == "correct":
                return True
            if value == "wrong":
                return False

        first_line = s.splitlines()[0].strip()
        if re.match(r"^yes\b", first_line):
            return True
        if re.match(r"^no\b", first_line):
            return False
        if re.match(r"^correct\b", first_line):
            return True
        if re.match(r"^wrong\b", first_line):
            return False

        # Fallback: inspect earliest standalone yes/no/correct/wrong token.
        tokens = re.findall(r"[a-z]+", s)
        for tok in tokens[:20]:
            if tok == "yes":
                return True
            if tok == "no":
                return False
            if tok == "correct":
                return True
            if tok == "wrong":
                return False
        return None

    def _exact_match_contains(self, expected_answer: str, model_output: str) -> bool:
        if not expected_answer or not model_output:
            return False
        
        expected_lower = expected_answer.lower().strip()
        output_lower = model_output.lower().strip()
        
        return expected_lower in output_lower

    async def evaluate(self) -> Dict[str, Any]:
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            if self.enable_llm_eval and self.eval_llm:
                metrics = await self._calculate_llm_metrics(results)
            else:
                metrics = self._calculate_exact_metrics(results)

            logger.info("Evaluation completed")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

    async def _calculate_llm_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to evaluate"}

        total_questions = len(results)
        correct_answers = 0
        answered_questions = 0
        type_stats = {}

        logger.info(f"Starting LLM-based evaluation of {total_questions} questions...")

        for i, result in enumerate(results):
            question = result.get('question', '')
            expected_answer = str(result.get('answer', '')).strip()
            model_output = str(result.get('output', '')).strip()
            question_type = result.get('question_type', 'default')
            if question_type not in type_stats:
                type_stats[question_type] = {'total': 0, 'correct': 0}
            type_stats[question_type]['total'] += 1
            
            if not expected_answer:
                logger.warning(f"No expected answer for question {i+1}: {question[:50]}...")
                continue
                
            if model_output:
                answered_questions += 1
                judge_bundle = await self.evaluate_with_llm_bundle(
                    question,
                    expected_answer,
                    model_output,
                    question_type,
                )
                is_correct = bool(judge_bundle["majority_label"])
                
                if is_correct:
                    correct_answers += 1
                    type_stats[question_type]['correct'] += 1
                    logger.debug(f"✅ Question {i+1}: Correct")
                else:
                    logger.debug(f"❌ Question {i+1}: Incorrect")
            else:
                logger.debug(f"⚠️ Question {i+1}: No response")

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{total_questions} questions...")

        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        answer_rate = answered_questions / total_questions if total_questions > 0 else 0

        type_accuracy = {}
        for qtype, stats in type_stats.items():
            if stats['total'] > 0:
                type_accuracy[qtype] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'total': stats['total'],
                    'correct': stats['correct']
                }

        metrics = {
            'evaluation_method': 'llm_based',
            'eval_model': self.config.evaluation.eval_model,
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'answer_rate': answer_rate,
            'dataset': self.dataset_name,
            'type_accuracy': type_accuracy
        }

        logger.info(f"LLM Evaluation completed: {correct_answers}/{total_questions} correct (accuracy: {accuracy:.3f})")
        return metrics

    def _calculate_exact_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to evaluate"}

        total_questions = len(results)
        correct_answers = 0
        answered_questions = 0

        for result in results:
            expected_answer = str(result.get('answer', '')).strip()
            model_output = str(result.get('output', '')).strip()
            
            if not expected_answer:
                continue
                
            if model_output:
                answered_questions += 1
                if self._exact_match_contains(expected_answer, model_output):
                    correct_answers += 1

        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        answer_rate = answered_questions / total_questions if total_questions > 0 else 0

        metrics = {
            'evaluation_method': 'exact_matching',
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'answer_rate': answer_rate,
            'dataset': self.dataset_name
        }

        logger.info(f"Exact matching evaluation: {correct_answers}/{total_questions} correct (accuracy: {accuracy:.3f})")
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
