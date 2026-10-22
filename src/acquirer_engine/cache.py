"""Stored LLM answers, so a run can be repeated exactly (and without an API key).

Every LLM request is reduced to a key: the SHA-256 hash of the request written
as canonical JSON (keys sorted, no spaces). Anything that could change the
answer is part of the request: model, settings, system prompt, messages
(including tool results), tool and output schemas, and a per-agent
prompt_version that is bumped whenever a prompt is edited. Change any of these
and the key changes, so an old answer is never reused for a new question.

The answers live in one SQLite file. Only answers that passed schema
validation are stored.
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Mode(str, Enum):
    """How LLM calls use the stored answers.

    live    - use a stored answer when there is one, otherwise call the API and store it
    replay  - stored answers only; no API key needed (a missing answer is an error)
    refresh - always call the API, and overwrite the stored answer
    off     - always call the API, store nothing (for debugging)
    """

    live = "live"
    replay = "replay"
    refresh = "refresh"
    off = "off"

    @property
    def reads(self) -> bool:
        return self in (Mode.live, Mode.replay)

    @property
    def writes(self) -> bool:
        return self in (Mode.live, Mode.refresh)

    @property
    def calls_api(self) -> bool:
        return self is not Mode.replay


class CachedAnswer(BaseModel):
    response_json: str
    input_tokens: int     # tokens of the original call, kept for the run report
    output_tokens: int


def cache_key(request: dict) -> str:
    """SHA-256 of the request as canonical JSON: same request, same key, whatever the dict order."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AnswerCache:
    """A small SQLite table of answers, keyed by cache_key().

    One connection is shared by every call in the run; a lock makes sure only
    one call writes at a time (SQLite allows a single writer).
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                model TEXT,
                prompt_version TEXT,
                created_at TEXT,
                request_json TEXT,
                response_json TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER
            )"""
        )
        self._db.commit()

    def get(self, key: str) -> CachedAnswer | None:
        with self._lock:
            row = self._db.execute(
                "SELECT response_json, input_tokens, output_tokens FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedAnswer(response_json=row[0], input_tokens=row[1], output_tokens=row[2])

    def put(self, key: str, request: dict, response_json: str, input_tokens: int,
            output_tokens: int) -> None:
        """Store (or overwrite) one answer."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    request.get("model"),
                    request.get("prompt_version"),
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(request, sort_keys=True, default=str),
                    response_json,
                    input_tokens,
                    output_tokens,
                ),
            )
            self._db.commit()
