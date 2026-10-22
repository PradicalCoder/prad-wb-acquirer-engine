"""The one LLM client of a run: structured answers, answer cache, retries, token counts.

Created once at startup and passed into every agent (tests pass in a fake), so
all calls share one cache, one concurrency limit and one usage record.

What a call does, in order:
  1. Build the request and its cache key (see cache.py).
  2. Mode allows reading and the answer is stored -> return it (no tokens spent).
  3. Replay mode and nothing stored -> LLMUnavailable (the step falls back).
  4. Call the API, retrying rate limits, server errors and dropped connections,
     but only while the step's deadline allows (see _invoke_with_retries).
  5. The answer doesn't fit the schema -> LLMOutputInvalid (the step repairs).
  6. Store the validated answer (if the mode writes) and return it.

Structured output uses Claude's own structured-output feature
(output_config.format with a JSON schema), not a forced tool call.
"""

import asyncio
import random
import time
from typing import Any, TypeVar

import anthropic
import structlog
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from acquirer_engine.cache import AnswerCache, Mode, cache_key
from acquirer_engine.config import LLM, Retries

T = TypeVar("T", bound=BaseModel)
log = structlog.get_logger()


class LLMUnavailable(Exception):
    """No usable answer: nothing recorded (replay), out of time, retries used up,
    or the API refused the request. The calling step uses its fallback."""


class LLMOutputInvalid(Exception):
    """The model answered, but not in the required shape. The calling step
    sends the error back to the model to repair."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


class CallRecord(BaseModel):
    """Token usage of one call, for the run report."""

    agent: str
    profile: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int   # prompt tokens served from Anthropic's prompt cache
    cache_hit: bool          # answer came from our answer cache: no new tokens spent
    attempts: int
    seconds: float


def is_retryable(error: anthropic.APIError) -> bool:
    """Rate limit (429), server error or overload (5xx, incl. 529), or a dropped
    connection. A bad request (400) or auth error would fail again: not retried."""
    if isinstance(error, anthropic.APIConnectionError):
        return True
    return isinstance(error, anthropic.APIStatusError) and (
        error.status_code == 429 or error.status_code >= 500
    )


def retry_wait(attempt: int, error: anthropic.APIError, cfg: Retries) -> float:
    """Seconds to wait before retry number `attempt` (0 = the first retry).

    If the API sent a retry-after header, wait that long. Otherwise
    base, 2 x base, ... capped at cap_seconds (0.5 s, 1 s, 2 s), scaled by a
    random 0.5-1.0 so parallel calls don't all retry at the same moment.
    """
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return min(cfg.base_seconds * 2**attempt, cfg.cap_seconds) * random.uniform(0.5, 1.0)


class LLMClient:
    def __init__(self, cfg: LLM, mode: Mode, cache: AnswerCache | None, api_key: str | None):
        self.cfg = cfg
        self.mode = mode
        self.cache = cache
        self.api_key = api_key
        self.usage: list[CallRecord] = []
        self._models: dict[str, ChatAnthropic] = {}
        self._slots = asyncio.Semaphore(cfg.max_concurrency)

    async def structured(
        self,
        *,
        agent: str,                 # who is asking, for logs and the run report
        profile: str,               # name of a profile in config.yaml (llm.profiles)
        system: str | list[dict],   # text, or content blocks (e.g. with cache_control)
        messages: list[dict],       # [{"role": "user", "content": ...}, ...]
        schema: type[T],            # the Pydantic model the answer must fit
        prompt_version: str,        # bump when the prompt changes
        deadline: float,            # time.monotonic() value by which we must be done
    ) -> T:
        """Ask for an answer that fits `schema`. See the module docstring for the steps."""
        settings = self.cfg.profiles[profile]
        request = {
            "model": settings.model,
            "effort": settings.effort,
            "thinking": settings.thinking,
            "max_tokens": settings.max_tokens,
            "system": system,
            "messages": messages,
            "output_schema": schema.model_json_schema(),
            "prompt_version": prompt_version,
        }
        key = cache_key(request)
        started = time.monotonic()

        if self.mode.reads and self.cache is not None:
            stored = self.cache.get(key)
            if stored is not None:
                self._record(agent, profile, started, attempts=0, cache_hit=True,
                             usage={"input_tokens": stored.input_tokens,
                                    "output_tokens": stored.output_tokens})
                log.info("llm.cache_hit", agent=agent, profile=profile)
                return schema.model_validate_json(stored.response_json)

        if not self.mode.calls_api:
            raise LLMUnavailable(f"{agent}: no recorded answer for this request (replay mode)")

        result, attempts = await self._invoke_with_retries(agent, profile, system, messages,
                                                           schema, deadline)
        usage = result["raw"].usage_metadata or {}
        record = self._record(agent, profile, started, attempts=attempts, cache_hit=False,
                              usage=usage)
        log.info("llm.done", **record.model_dump())

        if result["parsing_error"] is not None:
            raise LLMOutputInvalid(str(result["parsing_error"]), raw_text=result["raw"].text)

        answer = result["parsed"]
        if self.mode.writes and self.cache is not None:
            self.cache.put(key, request, answer.model_dump_json(),
                           record.input_tokens, record.output_tokens)
        return answer

    async def _invoke_with_retries(self, agent, profile, system, messages, schema, deadline):
        """Call the API; retry retryable errors while the deadline allows.

        Each attempt may only use the time left before the deadline. A retry
        starts only if, after waiting, at least min_attempt_seconds would remain;
        otherwise we give up straight away so the step can fall back in time.
        Returns (result, number of attempts).
        """
        runnable = self._model(profile).with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        chat = [{"role": "system", "content": system}, *messages]
        retries = self.cfg.retries

        for attempt in range(retries.max_retries + 1):
            remaining = deadline - time.monotonic()
            try:
                result = await asyncio.wait_for(self._send(runnable, chat), timeout=remaining)
                return result, attempt + 1
            except TimeoutError:
                raise LLMUnavailable(f"{agent}: out of time") from None
            except anthropic.APIError as error:
                if not is_retryable(error) or attempt == retries.max_retries:
                    raise LLMUnavailable(f"{agent}: {type(error).__name__}: {error}") from error
                wait = retry_wait(attempt, error, retries)
                if time.monotonic() + wait + retries.min_attempt_seconds > deadline:
                    raise LLMUnavailable(
                        f"{agent}: no time left to retry after {type(error).__name__}"
                    ) from error
                log.warning("llm.retry", agent=agent, attempt=attempt + 1,
                            wait_s=round(wait, 2), error=type(error).__name__)
                await asyncio.sleep(wait)

    async def _send(self, runnable, chat: list[dict]) -> dict[str, Any]:
        """The single network call. Waits for a free slot so at most
        max_concurrency calls run at once. Tests replace this method."""
        async with self._slots:
            return await runnable.ainvoke(chat)

    def _model(self, profile: str) -> ChatAnthropic:
        """The chat model for a profile, created on first use (so replay mode
        never needs an API key). The SDK's own retries are off: we retry above."""
        if profile not in self._models:
            settings = self.cfg.profiles[profile]
            self._models[profile] = ChatAnthropic(
                model=settings.model,
                api_key=self.api_key,
                max_tokens=settings.max_tokens,
                thinking={"type": settings.thinking},
                output_config={"effort": settings.effort},
                max_retries=0,
            )
        return self._models[profile]

    def _record(self, agent, profile, started, *, attempts, cache_hit, usage) -> CallRecord:
        record = CallRecord(
            agent=agent,
            profile=profile,
            model=self.cfg.profiles[profile].model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=(usage.get("input_token_details") or {}).get("cache_read") or 0,
            cache_hit=cache_hit,
            attempts=attempts,
            seconds=round(time.monotonic() - started, 3),
        )
        self.usage.append(record)
        return record
