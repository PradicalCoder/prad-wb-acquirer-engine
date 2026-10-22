"""Tests for the LLM client and answer cache. No API key or network: the single
network method (_send) is replaced by scripted answers and errors."""

import asyncio
import time

import anthropic
import httpx2
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from acquirer_engine.cache import AnswerCache, Mode, cache_key
from acquirer_engine.config import load_config
from acquirer_engine.llm import LLMClient, LLMOutputInvalid, LLMUnavailable, retry_wait

CONFIG = load_config()
# Tiny waits so retry tests run fast.
FAST = CONFIG.llm.model_copy(update={"retries": CONFIG.llm.retries.model_copy(update={
    "base_seconds": 0.001, "cap_seconds": 0.002, "min_attempt_seconds": 0.0,
})})
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class Answer(BaseModel):
    value: int


def ok(value=7):
    raw = AIMessage(content=f'{{"value": {value}}}', usage_metadata={
        "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
        "input_token_details": {"cache_read": 100},
    })
    return {"raw": raw, "parsed": Answer(value=value), "parsing_error": None}


def bad_shape():
    raw = AIMessage(content='{"value": "seven"}', usage_metadata={
        "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
    })
    return {"raw": raw, "parsed": None, "parsing_error": ValueError("value is not an int")}


def status_error(cls, code, headers=None):
    return cls("error", response=httpx2.Response(code, request=REQUEST, headers=headers), body=None)


def make_client(tmp_path, mode=Mode.live, script=(), cfg=FAST):
    """A client whose API answers come from `script`, in order (answers or exceptions)."""
    client = LLMClient(cfg, mode, AnswerCache(tmp_path / "cache.sqlite"), api_key="test-key")
    client.sent = 0
    replies = list(script)

    async def fake_send(runnable, chat):
        client.sent += 1
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply == "slow":
            await asyncio.sleep(1)
        return reply

    client._send = fake_send
    return client


async def ask(client, prompt="Pick a number.", seconds=5.0):
    return await client.structured(
        agent="test", profile="analyst_notes", system="You answer with JSON.",
        messages=[{"role": "user", "content": prompt}], schema=Answer,
        prompt_version="v1", deadline=time.monotonic() + seconds,
    )


# --- Cache key ----------------------------------------------------------------

def test_key_ignores_dict_order_but_not_content():
    a = cache_key({"model": "m", "messages": [{"role": "user", "content": "x"}], "prompt_version": "v1"})
    b = cache_key({"prompt_version": "v1", "messages": [{"content": "x", "role": "user"}], "model": "m"})
    c = cache_key({"model": "m", "messages": [{"role": "user", "content": "x"}], "prompt_version": "v2"})
    assert a == b
    assert a != c


def test_mode_flags():
    assert (Mode.live.reads, Mode.live.writes, Mode.live.calls_api) == (True, True, True)
    assert (Mode.replay.reads, Mode.replay.writes, Mode.replay.calls_api) == (True, False, False)
    assert (Mode.refresh.reads, Mode.refresh.writes) == (False, True)
    assert (Mode.off.reads, Mode.off.writes) == (False, False)


# --- Request shape ----------------------------------------------------------------

@pytest.mark.parametrize("profile", ["analyst_notes", "analyst_write"])
def test_request_uses_structured_outputs(tmp_path, profile):
    """Built offline: the payload langchain-anthropic would send for one call."""
    model = make_client(tmp_path)._model(profile)
    bound = model.with_structured_output(Answer, method="json_schema").first
    kwargs = {k: v for k, v in bound.kwargs.items() if not k.startswith("ls_")}  # tracing tags
    payload = model._get_request_payload([{"role": "user", "content": "hi"}], **kwargs)

    settings = CONFIG.llm.profiles[profile]
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["effort"] == settings.effort
    assert payload["thinking"] == {"type": settings.thinking}
    for removed in ("output_format", "temperature", "tool_choice", "tools"):
        assert removed not in payload


# --- Modes ----------------------------------------------------------------------

async def test_live_stores_then_reuses(tmp_path):
    client = make_client(tmp_path, script=[ok(7)])
    assert (await ask(client)).value == 7
    assert (await ask(client)).value == 7      # second time: from the cache
    assert client.sent == 1

    first, second = client.usage
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert first.input_tokens > 0 and first.output_tokens > 0
    assert first.cache_read_tokens == 100


async def test_prompt_change_is_a_new_question(tmp_path):
    client = make_client(tmp_path, script=[ok(1), ok(2)])
    await ask(client, "Pick a number.")
    assert (await ask(client, "Pick another number.")).value == 2
    assert client.sent == 2


async def test_replay_uses_stored_answers_only(tmp_path):
    await ask(make_client(tmp_path, script=[ok(7)]))            # record it
    replay = make_client(tmp_path, mode=Mode.replay)
    assert (await ask(replay)).value == 7
    with pytest.raises(LLMUnavailable, match="replay"):
        await ask(replay, "A question never recorded.")
    assert replay.sent == 0


async def test_refresh_calls_again_and_overwrites(tmp_path):
    await ask(make_client(tmp_path, script=[ok(7)]))
    refresh = make_client(tmp_path, mode=Mode.refresh, script=[ok(8)])
    assert (await ask(refresh)).value == 8
    assert (await ask(make_client(tmp_path, mode=Mode.replay))).value == 8


async def test_off_stores_nothing(tmp_path):
    await ask(make_client(tmp_path, mode=Mode.off, script=[ok(7)]))
    with pytest.raises(LLMUnavailable):
        await ask(make_client(tmp_path, mode=Mode.replay))


async def test_invalid_answer_is_reported_and_not_stored(tmp_path):
    client = make_client(tmp_path, script=[bad_shape()])
    with pytest.raises(LLMOutputInvalid) as caught:
        await ask(client)
    assert "seven" in caught.value.raw_text
    assert client.usage[0].output_tokens == 30          # tokens were still spent
    with pytest.raises(LLMUnavailable):
        await ask(make_client(tmp_path, mode=Mode.replay))


# --- Retries --------------------------------------------------------------------

async def test_rate_limits_are_retried(tmp_path):
    limit = status_error(anthropic.RateLimitError, 429)
    client = make_client(tmp_path, script=[limit, limit, ok(7)])
    assert (await ask(client)).value == 7
    assert client.usage[0].attempts == 3


async def test_overload_and_dropped_connection_are_retried(tmp_path):
    overloaded = status_error(anthropic.InternalServerError, 529)
    dropped = anthropic.APIConnectionError(request=REQUEST)
    client = make_client(tmp_path, script=[overloaded, dropped, ok(7)])
    assert (await ask(client)).value == 7


async def test_bad_request_is_not_retried(tmp_path):
    client = make_client(tmp_path, script=[status_error(anthropic.BadRequestError, 400), ok(7)])
    with pytest.raises(LLMUnavailable, match="BadRequestError"):
        await ask(client)
    assert client.sent == 1


async def test_retries_stop_when_used_up(tmp_path):
    limit = status_error(anthropic.RateLimitError, 429)
    client = make_client(tmp_path, script=[limit, limit, limit, ok(7)])
    with pytest.raises(LLMUnavailable):
        await ask(client)
    assert client.sent == FAST.retries.max_retries + 1


async def test_no_retry_without_time_for_it(tmp_path):
    # 3 s minimum per attempt, but the step has only 1 s in total.
    cfg = CONFIG.llm
    client = make_client(tmp_path, script=[status_error(anthropic.RateLimitError, 429), ok(7)],
                         cfg=cfg)
    with pytest.raises(LLMUnavailable, match="no time left"):
        await ask(client, seconds=1.0)
    assert client.sent == 1


async def test_slow_answer_runs_out_of_time(tmp_path):
    client = make_client(tmp_path, script=["slow"])
    with pytest.raises(LLMUnavailable, match="out of time"):
        await ask(client, seconds=0.05)


def test_retry_after_header_wins():
    error = status_error(anthropic.RateLimitError, 429, headers={"retry-after": "1.5"})
    assert retry_wait(0, error, CONFIG.llm.retries) == 1.5


def test_backoff_doubles_and_is_capped():
    error = anthropic.APIConnectionError(request=REQUEST)
    cfg = CONFIG.llm.retries
    assert cfg.base_seconds * 0.5 <= retry_wait(0, error, cfg) <= cfg.base_seconds
    assert retry_wait(5, error, cfg) <= cfg.cap_seconds
