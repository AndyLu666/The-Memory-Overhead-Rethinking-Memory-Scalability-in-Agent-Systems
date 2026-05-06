from typing import Dict, Any, List, Optional
import openai
import asyncio
import sys
import os
import random
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from utils.token_counter import count_input_tokens, count_output_tokens, get_token_cost
from utils.cost_manager import CostManager, TokenCostManager

class LLMManager:
    """Manager for Large Language Model interactions."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_tokens: int = 32768,
                 temperature: float = 0.0, base_url: str = None, enable_concurrent: bool = True,
                 max_concurrent: int = 16, timeout: int = 600, max_budget: float = 100.0,
                 retry_attempts: Optional[int] = None, retry_backoff: Optional[float] = None,
                 retry_backoff_max: Optional[float] = None, fail_on_error: Optional[bool] = None):
        """Initialize LLM manager."""
        self.api_key = api_key
        if not self.api_key:
            for env_name in ("LICOMEMORY_API_KEY", "QWEN_API", "OTHER_API_KEY", "OPENAI_API_KEY", "GPT_API_KEY"):
                env_val = os.getenv(env_name, "").strip()
                if env_val:
                    self.api_key = env_val
                    logger.info(f"LLM api_key loaded from environment variable: {env_name}")
                    break
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Keep config precedence, but allow env fallback when config leaves base_url empty.
        env_base_url = (
            os.getenv("LICOMEMORY_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or os.getenv("GPT_BASE_URL", "").strip()
        )
        self.base_url = base_url or env_base_url or "https://api.openai.com/v1"
        self.enable_concurrent = enable_concurrent
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        env_retry_attempts = int(os.getenv("LICOMEMORY_LLM_RETRY_ATTEMPTS", "6"))
        env_retry_backoff = float(os.getenv("LICOMEMORY_LLM_RETRY_BACKOFF", "1.0"))
        env_retry_backoff_max = float(os.getenv("LICOMEMORY_LLM_RETRY_BACKOFF_MAX", "20.0"))
        self.retry_attempts = max(1, int(retry_attempts if retry_attempts is not None else env_retry_attempts))
        self.retry_backoff = max(0.1, float(retry_backoff if retry_backoff is not None else env_retry_backoff))
        self.retry_backoff_max = max(
            self.retry_backoff,
            float(retry_backoff_max if retry_backoff_max is not None else env_retry_backoff_max),
        )
        # Qwen-compatible gateways frequently return transient busy/openai_error
        # bursts under high concurrency; increase default resilience unless user
        # has explicitly overridden retry env vars.
        if self._is_qwen_model():
            if "LICOMEMORY_LLM_RETRY_ATTEMPTS" not in os.environ:
                self.retry_attempts = max(self.retry_attempts, 10)
            if "LICOMEMORY_LLM_RETRY_BACKOFF_MAX" not in os.environ:
                self.retry_backoff_max = max(self.retry_backoff_max, 30.0)
        env_fail_on_error = os.getenv("LICOMEMORY_LLM_FAIL_ON_ERROR", "0").lower() in {"1", "true", "yes", "on"}
        self.fail_on_error = env_fail_on_error if fail_on_error is None else bool(fail_on_error)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent) if self.enable_concurrent else None

        if api_key == "demo-key" or "open-llm" in model.lower():
            self.cost_manager = TokenCostManager(max_budget=max_budget)
        else:
            self.cost_manager = CostManager(max_budget=max_budget)

        logger.info(
            f"LLM Manager initialized with model: {model}, base_url: {self.base_url}, "
            f"max_tokens: {max_tokens}, temperature: {temperature}, "
            f"concurrent: {enable_concurrent}, max_concurrent: {max_concurrent}, "
            f"retry_attempts: {self.retry_attempts}, fail_on_error: {self.fail_on_error}"
        )

    def _is_qwen_model(self) -> bool:
        model = (self.model or "").lower()
        return "qwen" in model

    def _is_gpt5_model(self) -> bool:
        model = (self.model or "").lower()
        return "gpt-5" in model

    def _is_gpt_oss_model(self) -> bool:
        model = (self.model or "").lower()
        return "gpt-oss" in model

    def _is_retryable_error(self, err: Exception) -> bool:
        status_code = getattr(err, "status_code", None)

        msg = str(err).lower()
        # Some third-party OpenAI-compatible gateways mislabel transient
        # upstream/database failures as 401 AuthenticationError. Treat only
        # these specific provider-side variants as retryable; keep true auth
        # failures non-retryable.
        if status_code == 401:
            transient_401_markers = [
                "数据库查询出错",
                "database query",
                "query_data_error",
                "connect: connection refused",
                "connection refused",
                "请联系管理员",
                "contact administrator",
            ]
            if any(marker in msg for marker in transient_401_markers):
                return True

        # Some OpenAI-compatible Qwen gateways may intermittently return 402
        # "insufficient balance" under bursty concurrency, then recover on retry.
        # Treat this as retryable for Qwen models to reduce false hard-fail turns.
        if status_code == 402 and self._is_qwen_model():
            transient_402_markers = [
                "insufficient balance",
                "unknown_error",
                "temporarily unavailable",
                "rate limit",
                "busy",
            ]
            if any(marker in msg for marker in transient_402_markers):
                return True

        non_retry_markers = [
            "model_not_found",
            "invalid_api_key",
            "authentication",
            "unauthorized",
            "insufficient_quota",
            "quota exceeded",
        ]
        if any(marker in msg for marker in non_retry_markers):
            return False

        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        # Some OpenAI-compatible gateways return transient upstream failures
        # as HTTP 403 with "bad_response_status_code"/"openai_error".
        if status_code == 403:
            transient_403_markers = [
                "bad_response_status_code",
                "openai_error",
                "rate limit",
                "do_request_failed",
                "temporarily unavailable",
                "service unavailable",
                "system memory overloaded",
            ]
            return any(marker in msg for marker in transient_403_markers)

        retry_markers = [
            "system memory overloaded",
            "do_request_failed",
            "rate limit",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
            "error code: 429",
            "error code: 403",
            "bad_response_status_code",
            "openai_error",
        ]
        return any(marker in msg for marker in retry_markers)

    def _get_reasoning_tokens(self, response) -> int:
        usage = getattr(response, "usage", None)
        if not usage:
            return 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is None:
            return 0
        if isinstance(details, dict):
            return int(details.get("reasoning_tokens") or 0)
        return int(getattr(details, "reasoning_tokens", 0) or 0)

    @staticmethod
    def _is_moderation_block_error(err: Exception) -> bool:
        text = str(err or "").lower()
        return "data_inspection_failed" in text or "inappropriate content" in text

    @staticmethod
    def _sanitize_prompt_for_provider_retry(prompt: str) -> str:
        text = str(prompt or "")
        if not text:
            return text
        text = re.sub(
            r"\nText Chunks:\n.*?\n\nSession Summaries:\n",
            "\nText Chunks:\n[omitted for provider-safe retry]\n\nSession Summaries:\n",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\(attached is [^)]+\)", "(attachment omitted)", text, flags=re.IGNORECASE)
        return text

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM with concurrent control."""
        try:
            if self.semaphore:
                async with self.semaphore:
                    return await self._generate_internal(prompt, **kwargs)
            else:
                return await self._generate_internal(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed (model={self.model}, base_url={self.base_url}): {e}")
            if self.fail_on_error:
                raise
            return ""

    async def _generate_internal(self, prompt: str, **kwargs) -> str:
        """Internal generate method without semaphore control."""
        system_prompt = kwargs.pop("system_prompt", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = count_input_tokens(messages, self.model)
        
        # For demo purposes, return mock response
        if self.api_key == "demo-key":
            logger.info(f"Using mock LLM response (demo mode)")
            mock_response = f"Mock response for: {prompt[:50]}..."
            completion_tokens = count_output_tokens(mock_response, self.model)
            self.cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
            return mock_response

        # Real LLM call using OpenAI 1.0+ API
        # Use manager defaults unless caller overrides per request.
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        temperature = kwargs.pop('temperature', self.temperature)
        extra_body = kwargs.pop('extra_body', None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        # Qwen models often return empty `content` while streaming long reasoning.
        # Force concise non-thinking mode by default for stable answer extraction.
        if self._is_qwen_model():
            if extra_body is None:
                extra_body = {"enable_thinking": False}
            elif isinstance(extra_body, dict) and "enable_thinking" not in extra_body:
                extra_body = {**extra_body, "enable_thinking": False}
        # OpenRouter gpt-oss free routes often emit action text into reasoning-only
        # fields, leaving `content` empty or malformed for our ReAct parser.
        # Disable reasoning when possible so the visible content stays aligned with
        # the strict JSON action contract expected by the agent loop.
        request_reasoning = kwargs.pop("reasoning", None)
        # GPT-5 family may spend the entire budget on hidden reasoning unless the
        # effort level is constrained; default to minimal for stable short-form
        # graph build, agent-turn, and judge prompts.
        if self._is_gpt5_model() and reasoning_effort is None:
            reasoning_effort = "minimal"
        response = None
        response_content = ""
        dynamic_max_tokens = max_tokens
        moderation_retry_used = False
        for attempt in range(1, self.retry_attempts + 1):
            try:
                request_kwargs = dict(kwargs)
                if extra_body is not None:
                    request_kwargs["extra_body"] = extra_body
                if request_reasoning is not None:
                    request_kwargs["reasoning"] = request_reasoning
                if reasoning_effort is not None:
                    request_kwargs["reasoning_effort"] = reasoning_effort
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=dynamic_max_tokens,
                    temperature=temperature,
                    **request_kwargs
                )
                message = response.choices[0].message
                response_content = (getattr(message, "content", "") or "").strip()

                # Some local OpenAI-compatible backends may place generated text
                # outside `content`. Prefer structured fallback before treating
                # the call as failed.
                if not response_content:
                    reasoning_text = (
                        getattr(message, "reasoning", None)
                        or getattr(message, "reasoning_content", None)
                        or ""
                    )
                    if isinstance(reasoning_text, str) and reasoning_text.strip():
                        logger.warning(
                            f"LLM returned empty content; falling back to reasoning text "
                            f"(model={self.model}, base_url={self.base_url})"
                        )
                        response_content = reasoning_text.strip()

                if response_content:
                    break

                finish_reason = getattr(response.choices[0], "finish_reason", "") or ""
                reasoning_tokens = self._get_reasoning_tokens(response)
                # GPT-5 family may spend the entire completion budget on hidden reasoning,
                # leaving visible content empty. When that happens, retry immediately
                # with a larger completion budget instead of repeating the same request.
                if (
                    self._is_gpt5_model()
                    and finish_reason == "length"
                    and reasoning_tokens >= dynamic_max_tokens
                    and dynamic_max_tokens < 8192
                ):
                    bumped_tokens = min(max(dynamic_max_tokens * 2, 4096), 8192)
                    logger.warning(
                        f"GPT-5 consumed completion budget in reasoning only; "
                        f"retrying with max_tokens={bumped_tokens} "
                        f"(attempt {attempt}/{self.retry_attempts}, model={self.model}, "
                        f"base_url={self.base_url})"
                    )
                    dynamic_max_tokens = bumped_tokens
                    continue

                is_last_attempt = attempt >= self.retry_attempts
                logger.warning(
                    f"LLM returned empty content (attempt {attempt}/{self.retry_attempts}, "
                    f"model={self.model}, base_url={self.base_url}, max_tokens={dynamic_max_tokens}); retrying"
                )
                if is_last_attempt:
                    if self.fail_on_error:
                        raise RuntimeError("LLM returned empty response content")
                    return ""
                backoff = min(self.retry_backoff * (2 ** (attempt - 1)), self.retry_backoff_max)
                backoff += random.uniform(0, min(1.0, backoff * 0.1))
                await asyncio.sleep(backoff)
                continue
            except Exception as api_error:
                retryable = self._is_retryable_error(api_error)
                is_last_attempt = attempt >= self.retry_attempts
                if (
                    self._is_qwen_model()
                    and not moderation_retry_used
                    and self._is_moderation_block_error(api_error)
                ):
                    sanitized_prompt = self._sanitize_prompt_for_provider_retry(prompt)
                    if sanitized_prompt != prompt:
                        moderation_retry_used = True
                        prompt = sanitized_prompt
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": str(system_prompt)})
                        messages.append({"role": "user", "content": prompt})
                        prompt_tokens = count_input_tokens(messages, self.model)
                        logger.warning(
                            "Provider moderation blocked Qwen request; retrying once with "
                            "provider-safe prompt sanitization "
                            f"(model={self.model}, base_url={self.base_url})"
                        )
                        continue
                logger.error(
                    f"LLM API call failed (attempt {attempt}/{self.retry_attempts}, "
                    f"model={self.model}, base_url={self.base_url}, retryable={retryable}): {api_error}"
                )
                if is_last_attempt or not retryable:
                    raise
                backoff = min(self.retry_backoff * (2 ** (attempt - 1)), self.retry_backoff_max)
                backoff += random.uniform(0, min(1.0, backoff * 0.1))
                await asyncio.sleep(backoff)

        if response is None:
            raise RuntimeError("LLM response is None after retry loop")

        if not response_content:
            logger.warning(
                f"LLM returned empty content and no reasoning fallback "
                f"(model={self.model}, base_url={self.base_url})"
            )
            if self.fail_on_error:
                raise RuntimeError("LLM returned empty response content")
            return ""
        
        if hasattr(response, 'usage') and response.usage:
            completion_tokens = response.usage.completion_tokens
            actual_prompt_tokens = response.usage.prompt_tokens
        else:
            completion_tokens = count_output_tokens(response_content, self.model)
            actual_prompt_tokens = prompt_tokens
        
        self.cost_manager.update_cost(actual_prompt_tokens, completion_tokens, self.model)
        
        return response_content

    async def extract_entities(self, text: str, session_time: str = "") -> List[Dict[str, Any]]:
        """Extract entities from text using professional prompt.
        
        Args:
            text: The text to extract entities from
            session_time: Optional session/query time for temporal entity processing
        """
        try:
            from ..prompt.entity_prompt import QUERY_ENTITY_EXTRACTION_PROMPT
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from prompt.entity_prompt import QUERY_ENTITY_EXTRACTION_PROMPT

        prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(text=text, session_time=session_time)
        logger.info(f"Sending professional entity extraction prompt to LLM (model: {self.model}, session_time: {session_time})")
        response = await self.generate(prompt)
        logger.debug(f"LLM response received (first 200 chars): {response[:200]}...")

        # Parse the custom format from the professional prompt
        entities = self._parse_entity_extraction_response(response)
        logger.info(f"Successfully extracted {len(entities)} entities using professional prompt")
        return entities

    def _parse_entity_extraction_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the custom format from entity extraction prompt."""
        entities = []
        records = response.split("##")
        for record in records:
            record = record.strip()
            if not record or record == "##END##":
                continue
            if record.startswith('("entity"|'):
                try:
                    content = record.strip()
                    if content.endswith(')'):
                        content = content[10:-1]
                    else:
                        content = content[10:]

                    # Split by the pipe delimiter |
                    parts = content.split('|')

                    if len(parts) >= 2:
                        # Clean up any remaining quotes from each part
                        entity_name = parts[0].strip('"').strip()
                        entity_type = parts[1].strip('"').strip()

                        # Validate that we have both name and type
                        if entity_name and entity_type:
                            entity = {
                                'entity': entity_name,
                                'type': entity_type,
                            }
                            entities.append(entity)
                            logger.debug(f"Successfully parsed entity: {entity_name} ({entity_type})")
                    else:
                        logger.warning(f"Insufficient parts in entity record: {record} (got {len(parts)} parts, expected at least 2)")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse entity record: {record}, error: {e}")
                    logger.debug(f"Content was: {content if 'content' in locals() else 'N/A'}")

        return entities

    async def batch_generate(self, prompts: List[str], progress_bar=None, **kwargs) -> List[str]:
        """Batch generate text with concurrent support.
        
        Args:
            prompts: List of prompts to generate
            progress_bar: Optional tqdm progress bar to update as each request completes
            **kwargs: Additional arguments for generate method
        """
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential processing")
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, **kwargs)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)
            return results

        if self.fail_on_error:
            if self.max_concurrent <= 1 or len(prompts) <= 1:
                logger.info("Strict fail-fast enabled, using sequential batch generation")
                results = []
                for prompt in prompts:
                    result = await self.generate(prompt, **kwargs)
                    results.append(result)
                    if progress_bar:
                        progress_bar.update(1)
                return results

            logger.info(
                "Strict fail-fast enabled, using bounded concurrent batch generation "
                f"(max concurrent: {self.max_concurrent})"
            )
            results_list = [None] * len(prompts)
            next_index = 0
            pending: Dict[asyncio.Task, int] = {}

            async def generate_one(index: int):
                result = await self.generate(prompts[index], **kwargs)
                if not result:
                    raise ValueError(f"Batch generation returned empty content at index={index}")
                return index, result

            while next_index < len(prompts) or pending:
                while next_index < len(prompts) and len(pending) < self.max_concurrent:
                    task = asyncio.create_task(generate_one(next_index))
                    pending[task] = next_index
                    next_index += 1

                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    pending.pop(task, None)
                    try:
                        index, result = task.result()
                    except Exception:
                        for other in pending:
                            other.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise

                    results_list[index] = result
                    if progress_bar:
                        progress_bar.update(1)

            return results_list
        
        logger.info(f"Batch concurrent processing {len(prompts)} requests, max concurrent: {self.max_concurrent}")
        
        # Create a list to store results
        results_list = [None] * len(prompts)
        failed_indices = []
        
        async def generate_with_progress(prompt, index):
            """Generate with progress tracking."""
            try:
                result = await self.generate(prompt, **kwargs)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch processing request {index} failed: {e}")
                results_list[index] = ""
                failed_indices.append(index)
            finally:
                # Update progress bar when each request completes
                if progress_bar:
                    progress_bar.update(1)
        
        tasks = [generate_with_progress(prompt, i) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=True)

        if failed_indices and self.fail_on_error:
            preview = failed_indices[:10]
            raise RuntimeError(
                f"Batch generation failed for {len(failed_indices)}/{len(prompts)} requests; "
                f"failed_indices={preview}"
            )
        
        return results_list

    async def batch_extract_entities(self, texts: List[str], progress_bar=None) -> List[List[Dict[str, Any]]]:
        """Batch extract entities with concurrent support.
        
        Args:
            texts: List of texts to extract entities from
            progress_bar: Optional tqdm progress bar to update as each request completes
        """
        if not self.enable_concurrent:
            logger.info("Concurrent not enabled, using sequential entity extraction")
            results = []
            for text in texts:
                result = await self.extract_entities(text)
                results.append(result)
                if progress_bar:
                    progress_bar.update(1)
            return results

        if self.fail_on_error:
            if self.max_concurrent <= 1 or len(texts) <= 1:
                logger.info("Strict fail-fast enabled, using sequential batch entity extraction")
                results = []
                for text in texts:
                    result = await self.extract_entities(text)
                    results.append(result)
                    if progress_bar:
                        progress_bar.update(1)
                return results

            logger.info(
                "Strict fail-fast enabled, using bounded concurrent batch entity extraction "
                f"(max concurrent: {self.max_concurrent})"
            )
            results_list = [None] * len(texts)
            next_index = 0
            pending: Dict[asyncio.Task, int] = {}

            async def extract_one(index: int):
                entities = await self.extract_entities(texts[index])
                if texts[index] and entities is None:
                    raise ValueError(f"Batch entity extraction returned None at index={index}")
                return index, entities

            while next_index < len(texts) or pending:
                while next_index < len(texts) and len(pending) < self.max_concurrent:
                    task = asyncio.create_task(extract_one(next_index))
                    pending[task] = next_index
                    next_index += 1

                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    pending.pop(task, None)
                    try:
                        index, entities = task.result()
                    except Exception:
                        for other in pending:
                            other.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise

                    results_list[index] = entities
                    if progress_bar:
                        progress_bar.update(1)

            return results_list

        logger.info(f"Batch concurrent entity extraction {len(texts)} texts, max concurrent: {self.max_concurrent}")
        
        # Create a list to store results
        results_list = [None] * len(texts)
        failed_indices = []
        
        async def extract_with_progress(text, index):
            """Extract entities with progress tracking."""
            try:
                result = await self.extract_entities(text)
                results_list[index] = result
            except Exception as e:
                logger.error(f"Batch entity extraction request {index} failed: {e}")
                results_list[index] = []
                failed_indices.append(index)
            finally:
                # Update progress bar when each request completes
                if progress_bar:
                    progress_bar.update(1)
        
        tasks = [extract_with_progress(text, i) for i, text in enumerate(texts)]
        await asyncio.gather(*tasks, return_exceptions=True)

        if failed_indices and self.fail_on_error:
            preview = failed_indices[:10]
            raise RuntimeError(
                f"Batch entity extraction failed for {len(failed_indices)}/{len(texts)} requests; "
                f"failed_indices={preview}"
            )
        
        return results_list

    def get_costs(self):
        """Get current cost statistics."""
        return self.cost_manager.get_costs()

    def get_last_stage_cost(self):
        """Get last stage cost statistics."""
        return self.cost_manager.get_last_stage_cost()

    def get_cost_totals(self):
        """Get cost totals."""
        return self.cost_manager.get_cost_totals()

    def check_budget(self):
        """Check budget."""
        return self.cost_manager.check_budget()

    def set_max_budget(self, budget: float):
        """Set maximum budget."""
        self.cost_manager.max_budget = budget
        logger.info(f"💰 Max budget set to: ${budget:.2f}")
