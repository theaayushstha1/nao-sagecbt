"""FastAPI WebSocket transport — Phase 1 replacement for Flask /turn + /stream_turn.

Endpoints
---------
- ``GET  /health``  Liveness probe (no auth required).
- ``GET  /metrics`` Prometheus exposition (mounted from `server.metrics`).
- ``WS   /ws/{username}`` Long-lived bidirectional voice loop.

Frame envelope is defined in ``docs/PHASE_1_TASK_MAP.md`` and MUST match
exactly. Field names are load-bearing — the NAO client agent depends on them.

Per Phase 1 ownership, this module imports the agent runner, VAD, STT, and
filter helpers from ``server._legacy_helpers`` (verbatim copies of frozen
private helpers in ``server/server.py``). The legacy Flask app is untouched.

The handler streams TTS one sentence at a time:
  1. Run the agent graph in a worker thread (sync API).
  2. Slice the reply into sentence chunks (via ``streaming``).
  3. Synthesize each chunk in a worker thread (OpenAI TTS).
  4. Push one ``audio_chunk`` frame per sentence the moment it's ready.

Body actions accumulated in the agent context are flushed BEFORE the first
audio chunk so the robot can begin moving while it speaks.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from fastapi import (
    FastAPI,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from server import breathing_pacing, config, memory, motion_trigger, openai_tts, safety
from server import _legacy_helpers as legacy
from server.tools import emotion as _emotion_module

# Phase 11.8: ElevenLabs streaming TTS as primary path. Fallback to
# OpenAI tts-1 when EL is missing/unconfigured/erroring.
try:
    from server import elevenlabs_tts as _eleven  # type: ignore
except Exception:  # pragma: no cover -- module always present, keep belt+braces
    _eleven = None  # type: ignore[assignment]


def _synth_for(username: str, text: str) -> bytes | None:
    """Pick TTS provider per-call. Tries ElevenLabs first when enabled
    and available; falls back to OpenAI on any failure / missing key.

    Voice profile resolution order:
      1. Per-user pref via session.get_voice_profile(username)
      2. Server default ELEVENLABS_DEFAULT_PROFILE
      3. OpenAI fallback if neither resolves to an EL voice ID
    """
    use_eleven = (
        _eleven is not None
        and getattr(config, "USE_ELEVENLABS_TTS", True)
        and _eleven.is_available()
    )
    if use_eleven:
        try:
            from server import session as _ses
            profile = _ses.get_voice_profile(username) or \
                      getattr(config, "ELEVENLABS_DEFAULT_PROFILE", "girl")
            voice_id = _eleven._voice_id_for(profile) or \
                       _eleven._resolve_default_voice_id()
            if voice_id:
                bytes_ = _eleven.synthesize(text, voice_id=voice_id)
                if bytes_:
                    return bytes_
            # Resolved profile but EL returned no audio — fall through.
            logger.warning(
                "elevenlabs_synth_returned_none",
                user=username, text_preview=(text or "")[:80],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "elevenlabs_synth_error",
                user=username, error=repr(exc),
            )
    # Default / fallback path.
    return openai_tts.synthesize(text)


# Phase 11.6 — vision refresh policy. Visual-trigger phrases that
# unconditionally force a fresh observation regardless of cache age.
_VISION_REFRESH_TRIGGERS: tuple[str, ...] = (
    # "look at me" family
    "look at me", "look at my", "look at this",
    "do i look", "how do i look", "how am i looking",
    # "see" family
    "can you see", "do you see", "what do you see",
    "see my", "see how i", "see the way",
    "see what i", "watch me", "are you watching",
    "see right now", "see now",
    # affective / posture cues (legacy therapy)
    "i'm crying", "im crying", "i am crying",
    "i'm smiling", "im smiling",
    "i'm tired", "you can tell", "do i seem",
    # what-am-i-wearing family
    "what am i wearing", "what color is my", "what color shirt",
    "what color am i", "describe my outfit", "describe what i am",
    "describe what im", "describe me",
    # describe-the-room family
    "what do you notice", "what's around", "whats around",
    "describe the room", "describe what you see",
    "tell me what you see", "what's in front", "whats in front",
    "what's behind", "whats behind",
    # holding objects
    "what am i holding", "what is this", "what's this",
    "do you recognize this", "identify this",
)

# Vision is LAZY — fires only when the user's transcript matches one
# of the trigger phrases above. Every fire is fresh: there is no cache
# reuse. Caching was removed because if the user asked the same visual
# question minutes later (or a friend asked it in a different setting)
# NAO would replay the prior description, even though the scene was
# completely different. Recomputing per visual question costs ~1.5 s
# per question, but only on questions that actually need vision.


def _should_refresh_vision(sess: Any, transcript: str) -> tuple[bool, str]:
    """Decide whether to run vision this turn (LAZY policy, no cache).

    Vision is OFF by default. It fires only when the user's transcript
    matches a vision trigger phrase ("what color is my shirt", "describe
    me", "look at me", "do you see", etc.). Every trigger gets a fresh
    GPT-4o vision call against the latest image — the previous 5 min
    cache caused a real bug where if a friend asked the same visual
    question a few minutes later in a different setting, NAO replied
    with the cached description of the *previous* user.

    Returns ``(refresh, reason)``. ``reason`` lands in the structured
    log so we can audit when vision actually fired.
    """
    low = (transcript or "").lower()
    matched_trigger = None
    for phrase in _VISION_REFRESH_TRIGGERS:
        if phrase in low:
            matched_trigger = phrase
            break

    # No trigger phrase → never fire.
    if matched_trigger is None:
        return False, "no_visual_question"

    # Trigger phrase fired — always refresh. No cache reuse.
    return True, f"trigger_phrase:{matched_trigger}"


def _cancel_pending_vision(sess) -> None:
    """Drop the pending parallel vision task on a short-circuit path.

    Motion-trigger / crisis / echo-reject all skip the agent run, so
    awaiting the vision result would just burn an API call. Cancel the
    task quietly; ignore CancelledError on next-loop drain.
    """
    task = getattr(sess, "_vision_task", None)
    if task is not None and not task.done():
        try:
            task.cancel()
        except Exception:
            pass
    try:
        sess._vision_task = None  # type: ignore[attr-defined]
    except Exception:
        pass

# ───────── observability adapters ─────────
#
# `server/metrics.py` and `server/logging_setup.py` are owned by other Phase-1
# agents (observability slug). We import lazily and fall back to no-op shims
# so this module boots even if those files don't exist yet — important for
# parallel agent execution where any one agent might land before another.

try:  # pragma: no cover — exercised via tests once observability lands
    from server import metrics as _metrics  # type: ignore[attr-defined]
    PROM_REGISTRY = getattr(_metrics, "PROM_REGISTRY", None)
    _phase_timer = getattr(_metrics, "phase_timer", None)
except Exception:  # noqa: BLE001
    _metrics = None
    PROM_REGISTRY = None
    _phase_timer = None


class _NullPhaseTimer:
    """No-op stand-in for ``metrics.phase_timer`` until the real one ships.

    Records the elapsed milliseconds in the ``phase_ms`` dict so the per-turn
    log event still has timing data even when Prometheus isn't wired.
    """

    __slots__ = ("_label", "_phase_ms", "_t0")

    def __init__(self, label: str, phase_ms: dict[str, float]) -> None:
        self._label = label
        self._phase_ms = phase_ms
        self._t0 = 0.0

    def __enter__(self):  # type: ignore[no-untyped-def]
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        # Always record into the per-turn dict (used by the structlog event).
        self._phase_ms[self._label] = round(elapsed_ms, 2)
        return False


def _phase(label: str, phase_ms: dict[str, float]):
    """Return a context manager that times a phase.

    Prefers ``metrics.phase_timer`` (Prometheus Histogram) when available;
    always also records the elapsed time into the per-turn ``phase_ms`` dict
    so the structured turn log retains timing data.
    """
    if _phase_timer is not None:
        try:
            real = _phase_timer(label)
            return _CombinedTimer(real, label, phase_ms)
        except Exception:  # pragma: no cover — defensive
            pass
    return _NullPhaseTimer(label, phase_ms)


class _CombinedTimer:
    """Chains the real ``metrics.phase_timer`` with our local phase_ms record."""

    __slots__ = ("_inner", "_label", "_phase_ms", "_t0")

    def __init__(self, inner: Any, label: str, phase_ms: dict[str, float]) -> None:
        self._inner = inner
        self._label = label
        self._phase_ms = phase_ms
        self._t0 = 0.0

    def __enter__(self):  # type: ignore[no-untyped-def]
        self._t0 = time.perf_counter()
        try:
            self._inner.__enter__()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        try:
            self._inner.__exit__(exc_type, exc, tb)
        except Exception:
            pass
        self._phase_ms[self._label] = round(
            (time.perf_counter() - self._t0) * 1000.0, 2,
        )
        return False


try:  # pragma: no cover — exercised via tests once observability lands
    from server.logging_setup import logger as _structlog_logger  # type: ignore
except Exception:  # noqa: BLE001
    _structlog_logger = None


class _StdLogger:
    """Tiny adapter mimicking the structlog API surface this module uses."""

    def __init__(self) -> None:
        self._log = logging.getLogger("sage.app_ws")

    def info(self, event: str, **kwargs: Any) -> None:
        try:
            self._log.info("%s %s", event, json.dumps(kwargs, default=str))
        except Exception:
            self._log.info(event)

    def warning(self, event: str, **kwargs: Any) -> None:
        try:
            self._log.warning("%s %s", event, json.dumps(kwargs, default=str))
        except Exception:
            self._log.warning(event)

    def error(self, event: str, **kwargs: Any) -> None:
        try:
            self._log.error("%s %s", event, json.dumps(kwargs, default=str))
        except Exception:
            self._log.error(event)


logger = _structlog_logger if _structlog_logger is not None else _StdLogger()

# Module-level scratch space for mic_trace prints. _Session uses __slots__
# so per-instance trace state needs to live outside the session object.
_MIC_TRACE_STATE: dict[str, dict[str, Any]] = {}

# Session-scoped identification result from the robot's onboarding face
# scan. Keyed by session_id. Read by `format_user_message` (legacy
# helpers) to inject a `[USER name=X returning=true]` block into the
# agent's user-message prefix on the FIRST turn of each new session.
# Cleared on session_close.
_IDENTIFIED_USERS: dict[str, dict[str, Any]] = {}


# ───────── env-driven knobs ─────────

TTS_CHUNK_MIN_CHARS = int(os.environ.get("TTS_CHUNK_MIN_CHARS", "30"))
TTS_CHUNK_TIMEOUT_MS = int(os.environ.get("TTS_CHUNK_TIMEOUT_MS", "400"))

# Phase 2 — EoU arbiter knobs (per docs/PHASE_2_TASK_MAP.md).
#
# `MIN_SILENCE_MS` is the silero-driven silence threshold that finalizes a
# turn outright. The robot-hint-driven branch uses a tighter `150 ms` window
# (since the robot has already declared the user done — we just want silero
# confirmation). The semantic-early branch fires on `200 ms` of silence with
# a complete-thought signal.
#
# The 60 s ceiling matches the PRD ("Allow up to 60 s of legitimate
# continuous speech").
EOU_MIN_SILENCE_MS = int(os.environ.get("EOU_MIN_SILENCE_MS", "500"))
EOU_HINT_CONFIRM_MS = int(os.environ.get("EOU_HINT_CONFIRM_MS", "150"))
EOU_SEMANTIC_SILENCE_MS = int(os.environ.get("EOU_SEMANTIC_SILENCE_MS", "200"))
EOU_HARD_CEILING_MS = int(os.environ.get("EOU_HARD_CEILING_MS", "60_000"))

# Phase 2 — Post-TTS cooldown knob. Bytes received within
# `(MIC_GATE_GRACE_MS + 400) ms` after the last audio_chunk are dropped as
# echo. The 400 ms tail covers reverb that survives the robot's own mic
# unsubscribe.
TTS_COOLDOWN_PADDING_MS = int(os.environ.get("TTS_COOLDOWN_PADDING_MS", "400"))


# ───────── streaming Silero (parallel agent) ─────────
#
# `server.vad_silero.StreamingSilero` is being added by the `server-silero`
# agent in a sibling worktree. Until that lands here we must degrade
# gracefully — the existing behaviour was "always finalize on robot hint",
# so a None instance falls back to that path.

try:  # pragma: no cover — exercised once server-silero lands
    from server.vad_silero import StreamingSilero as _StreamingSilero  # type: ignore
except Exception:  # noqa: BLE001
    _StreamingSilero = None  # type: ignore[assignment]


def _get_streaming_silero() -> Any | None:
    """Return a fresh ``StreamingSilero`` instance, or None if unavailable.

    The factory is wrapped in try/except because Silero pulls in torch which
    can fail at construction time on machines without the model weights.
    """
    if _StreamingSilero is None:
        return None
    try:
        return _StreamingSilero()
    except Exception as e:  # noqa: BLE001
        logging.getLogger("sage.app_ws").warning(
            "StreamingSilero construction failed (%s); arbiter will fall "
            "back to robot-hint-only finalization.", e,
        )
        return None


# ───────── echo cooldown counter ─────────
#
# Tracks how often we drop incoming audio_chunk frames during post-TTS
# cooldown. Defined locally if `server.metrics.echo_cooldown_drops_total`
# isn't published yet (the parallel observability agent will whitelist it).

_echo_cooldown_drops_total: Any | None = None


def _resolve_echo_drop_counter() -> Any | None:
    """Lazily resolve the cooldown counter, preferring the published metric."""
    global _echo_cooldown_drops_total
    if _echo_cooldown_drops_total is not None:
        return _echo_cooldown_drops_total
    if _metrics is not None:
        existing = getattr(_metrics, "echo_cooldown_drops_total", None)
        if existing is not None:
            _echo_cooldown_drops_total = existing
            return existing
    # Build a private Counter on the metrics registry if available, else a
    # truly local one so increments still work in tests / no-metrics mode.
    try:
        from prometheus_client import Counter
        registry = PROM_REGISTRY
        if registry is not None:
            _echo_cooldown_drops_total = Counter(
                "nao_echo_cooldown_drops_total",
                "Inbound audio_chunk frames dropped during post-TTS cooldown",
                registry=registry,
            )
        else:
            _echo_cooldown_drops_total = Counter(
                "nao_echo_cooldown_drops_total",
                "Inbound audio_chunk frames dropped during post-TTS cooldown",
            )
    except Exception:  # noqa: BLE001 — prometheus_client may be missing
        class _LocalCounter:
            __slots__ = ("_n",)

            def __init__(self) -> None:
                self._n = 0

            def inc(self, amount: float = 1.0) -> None:
                self._n += amount

        _echo_cooldown_drops_total = _LocalCounter()
    return _echo_cooldown_drops_total


# ───────── wake_event counter (Phase 3) ─────────
#
# `nao_wake_events_total{gate}` counts AWARE→ENGAGED transitions broken down
# by which engagement gate fired (mutual_gaze, proximity, sustained_face,
# speech, keyword). Defined on `PROM_REGISTRY` if `server.metrics` hasn't
# whitelisted it yet (mirrors the echo-cooldown counter pattern).

_wake_events_total: Any | None = None


def _resolve_wake_events_counter() -> Any | None:
    """Lazily resolve the `wake_events_total{gate}` counter.

    Prefers the published metric if `server.metrics` already exposes it;
    otherwise registers a private Counter on the local registry; otherwise
    returns a truly local counter so increments are no-ops at the test
    layer when prometheus_client isn't installed.
    """
    global _wake_events_total
    if _wake_events_total is not None:
        return _wake_events_total
    if _metrics is not None:
        existing = getattr(_metrics, "wake_events_total", None)
        if existing is not None:
            _wake_events_total = existing
            return existing
    try:
        from prometheus_client import Counter
        registry = PROM_REGISTRY
        if registry is not None:
            _wake_events_total = Counter(
                "nao_wake_events_total",
                "Wake events received by gate type (Phase 3 hybrid wake)",
                ["gate"],
                registry=registry,
            )
        else:
            _wake_events_total = Counter(
                "nao_wake_events_total",
                "Wake events received by gate type (Phase 3 hybrid wake)",
                ["gate"],
            )
    except Exception:  # noqa: BLE001 — prometheus_client may be missing
        class _LocalLabeledCounter:
            __slots__ = ("_n",)

            def __init__(self) -> None:
                self._n: dict[str, float] = {}

            def labels(self, gate: str) -> "_LocalLabeledCounter._Bound":
                return _LocalLabeledCounter._Bound(self, gate)

            class _Bound:
                __slots__ = ("_owner", "_gate")

                def __init__(self, owner: "_LocalLabeledCounter", gate: str) -> None:
                    self._owner = owner
                    self._gate = gate

                def inc(self, amount: float = 1.0) -> None:
                    self._owner._n[self._gate] = (
                        self._owner._n.get(self._gate, 0.0) + amount
                    )

        _wake_events_total = _LocalLabeledCounter()
    return _wake_events_total


# Per-session FSM tag tracked in a module-level dict keyed by ``session_id``.
# Phase 3 only needs a coarse label ("listening" once a wake_event has been
# greeted / acknowledged) so downstream phases can read it without us
# expanding the `_Session` slot list. The dict is intentionally append-only
# during a session's life and pruned on session_close.
_SESSION_FSM_STATE: dict[str, str] = {}


def _set_session_fsm_state(session_id: str, state: str) -> None:
    if not session_id:
        return
    _SESSION_FSM_STATE[session_id] = state


def _clear_session_fsm_state(session_id: str) -> None:
    if not session_id:
        return
    _SESSION_FSM_STATE.pop(session_id, None)


# Wake-greeting recency window — Phase 3 contract: resume the SQLiteSession
# only when the same face_id was seen in the last 24 h. Configurable for
# tests / ops; default per PRD §Phase 3.
WAKE_RESUME_WINDOW_S = float(os.environ.get("WAKE_RESUME_WINDOW_S", str(24 * 3600)))


def _lookup_returning_user(face_id: str) -> tuple[bool, str | None]:
    """Inspect the `users` table (owned by `server.memory`) for a recent visit.

    Returns ``(is_returning, display_name)``. ``is_returning`` is True iff a
    row exists for ``face_id`` and ``updated_at`` is within
    ``WAKE_RESUME_WINDOW_S``. We open a short-lived sqlite connection
    directly on `config.SESSION_DB` rather than mutate `server.memory` —
    `server/app_ws.py` is the only owned file in this slug.
    """
    fid = (face_id or "").strip().lower()
    if not fid:
        return False, None
    try:
        import sqlite3
        conn = sqlite3.connect(config.SESSION_DB)
        try:
            row = conn.execute(
                "SELECT display_name, updated_at FROM users WHERE face_id = ?",
                (fid,),
            ).fetchone()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — DB or schema absent
        logger.warning(
            "wake_user_lookup_failed",
            face_id=fid, error=repr(e),
        )
        return False, None
    if row is None:
        return False, None
    display_name, updated_at = row
    try:
        age_s = time.time() - float(updated_at or 0.0)
    except (TypeError, ValueError):
        age_s = float("inf")
    is_returning = age_s <= WAKE_RESUME_WINDOW_S
    return is_returning, (str(display_name) if display_name else None)


def _last_recap_line(username: str) -> str | None:
    """Best-effort one-line topic-continuity hint from the most recent recap.

    Uses `server.session.load_recent_recaps` (existing public API) so we
    don't reach into the DB ourselves for recap data. The returned line is
    truncated to ~140 chars and stripped of leading/trailing punctuation
    that would read awkwardly when appended after the welcome sentence.
    """
    if not username:
        return None
    try:
        from server import session as _session
        recaps = _session.load_recent_recaps(username, n=1)
    except Exception as e:  # noqa: BLE001 — never break wake on a recap miss
        logger.warning(
            "wake_recap_lookup_failed", user=username, error=repr(e),
        )
        return None
    if not recaps:
        return None
    body = (recaps[0] or "").strip()
    if not body:
        return None
    # Take the first sentence-ish span (keep things to a single line).
    first = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()
    if not first:
        return None
    if len(first) > 140:
        first = first[:137].rstrip() + "..."
    return first


_RECAP_LEAD_STRIPPER = re.compile(
    r"^(we\s+(?:talked|discussed|spoke|chatted)\s+about\s+|we\s+covered\s+|"
    r"the\s+user\s+(?:talked|discussed|asked)\s+about\s+|"
    r"discussed\s+|talked\s+about\s+|covered\s+)",
    re.IGNORECASE,
)


def _build_returning_greeting(display_name: str | None,
                              recap_line: str | None) -> str:
    """Compose the spoken welcome-back greeting per Phase 3 contract."""
    name = (display_name or "").strip() or "friend"
    head = "Welcome back, {0}.".format(name)
    if recap_line:
        # Strip "We talked about ..." / "Talked about ..." style leads so the
        # follow-on doesn't read "Last time we were talking about we talked
        # about ...". Whatever survives is the actual subject.
        tail = _RECAP_LEAD_STRIPPER.sub("", recap_line, count=1).strip()
        # Trim trailing punctuation so the appended period reads cleanly.
        tail = tail.rstrip(" .!?,;:")
        if tail:
            head += " Last time we were talking about {0}.".format(tail)
    return head


# ───────── self-echo guard state (Phase 2 strengthening) ─────────
#
# Per-username rolling window of the last 8 sentences passed to TTS, plus a
# joined snapshot for substring containment checks. These augment the
# bigram-overlap guard in `_legacy_helpers._is_self_echo` (which is left
# untouched per the file-ownership rules).

_LAST_REPLY_CHUNKS: dict[str, list[str]] = {}
_LAST_REPLY_FULL: dict[str, str] = {}
_REPLY_CHUNKS_MAX = 8


def _record_reply_chunk(username: str, sentence: str) -> None:
    """Append a sentence to the per-user rolling chunk buffer.

    Called every time we synthesize a sentence (crisis hotline reply,
    motion-trigger ack, or per-sentence agent stream chunk). Keeps the
    buffer capped at ``_REPLY_CHUNKS_MAX`` and rebuilds the joined
    snapshot used by the substring guard.
    """
    if not sentence:
        return
    text = str(sentence).strip()
    if not text:
        return
    chunks = _LAST_REPLY_CHUNKS.setdefault(username, [])
    chunks.append(text)
    # Trim to the most recent _REPLY_CHUNKS_MAX entries.
    if len(chunks) > _REPLY_CHUNKS_MAX:
        del chunks[: len(chunks) - _REPLY_CHUNKS_MAX]
    _LAST_REPLY_FULL[username] = " ".join(chunks)


def _reset_reply_chunks(username: str, full_reply: str) -> None:
    """Reset the per-user buffer to a single-entry view of the full reply.

    Used for non-streamed replies (crisis, motion) where we synthesize a
    single string in one shot rather than walking sentences.
    """
    text = (full_reply or "").strip()
    if not text:
        _LAST_REPLY_CHUNKS.pop(username, None)
        _LAST_REPLY_FULL.pop(username, None)
        return
    _LAST_REPLY_CHUNKS[username] = [text]
    _LAST_REPLY_FULL[username] = text


def _is_substring_or_sentence_echo(username: str, transcript: str) -> bool:
    """Phase 2 echo guard layer above ``_legacy_helpers._is_self_echo``.

    Two new checks:
    1. ``transcript.lower().strip()`` is a substring of the joined last-reply.
    2. Any single recorded sentence shares >= 70% of its tokens with the
       transcript (Jaccard token overlap on the smaller side).

    Either match returns True — caller emits ``echo_reject`` and skips the
    agent.
    """
    if not transcript:
        return False
    nt = transcript.lower().strip()
    if not nt:
        return False

    full = _LAST_REPLY_FULL.get(username, "").lower()
    if full and nt in full:
        return True

    # Token-overlap against each individual sentence — protects against the
    # case where the transcript echoes one sentence but the joined string
    # is too long for a substring hit.
    nt_tokens = set(re.findall(r"[a-z0-9']+", nt))
    if not nt_tokens:
        return False
    for sent in _LAST_REPLY_CHUNKS.get(username, []):
        sent_tokens = set(re.findall(r"[a-z0-9']+", sent.lower()))
        if not sent_tokens:
            continue
        smaller = min(len(nt_tokens), len(sent_tokens))
        if smaller == 0:
            continue
        overlap = len(nt_tokens & sent_tokens) / float(smaller)
        if overlap >= 0.70:
            return True
    return False


# ───────── auth ─────────

_OPEN_PATHS = {"/health", "/metrics"}


def _check_ws_auth(websocket: WebSocket) -> bool:
    """Validate the shared-secret on the WebSocket upgrade.

    Accepts the secret from either the ``X-NAO-Secret`` header (preferred
    for parity with HTTP) or the ``secret`` query string param (fallback for
    naoqi's WebSocket client which can't always set custom headers).
    """
    expected = config.NAO_SHARED_SECRET
    if not expected:
        return True
    got = websocket.headers.get("x-nao-secret", "")
    if got == expected:
        return True
    qp = websocket.query_params.get("secret", "")
    return qp == expected


# ───────── app factory ─────────

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    if not config.NAO_SHARED_SECRET:
        logging.getLogger("sage.app_ws").warning(
            "NAO_SHARED_SECRET unset — server is OPEN to anyone on the network. "
            "Set it in .env before exposing /ws/{username}.",
        )
    # Phase 6 — apply any pending DB migrations on boot. The runner is
    # idempotent (skips files already recorded in the `migrations` table)
    # and best-effort: a migration failure logs and continues so a bad
    # file in this directory can't take the whole server down.
    try:
        from server.migrations import apply_pending_migrations
        applied = apply_pending_migrations()
        if applied:
            logging.getLogger("sage.app_ws").info(
                "migrations applied on boot: %s", ",".join(applied),
            )
    except Exception as e:  # noqa: BLE001
        logging.getLogger("sage.app_ws").error(
            "migration runner failed on boot: %r", e,
        )
    yield


app = FastAPI(
    title="NAO Morgan Assist — Phase 1 WebSocket transport",
    version="phase-1",
    lifespan=_lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "version": "phase-1"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition endpoint.

    Delegates to ``server.metrics.PROM_REGISTRY`` once the observability
    agent ships that module. Until then, returns 503 so monitoring tools can
    detect the missing dependency cleanly. ``/metrics`` is intentionally in
    ``_OPEN_PATHS`` so Prometheus scrapers don't need the shared secret.
    """
    if PROM_REGISTRY is None:
        return Response(
            content=b"# metrics unavailable: server.metrics module not loaded\n",
            media_type="text/plain; version=0.0.4",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:  # pragma: no cover — exercised once observability lands
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(
            content=generate_latest(PROM_REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    except Exception as e:  # noqa: BLE001
        return Response(
            content="# metrics error: {0}\n".format(e).encode("utf-8"),
            media_type="text/plain; version=0.0.4",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


# ───────── WAV writing ─────────

# WS audio chunk format (per task map): 20 ms PCM16 mono @ 16 kHz, base64.
# 16 kHz × 2 bytes × 0.020 s = 640 bytes per chunk.
_WS_AUDIO_SR = 16_000
_WS_AUDIO_BYTES_PER_FRAME = 2  # PCM16 mono


def _write_pcm_to_wav(pcm: bytes, sr: int = _WS_AUDIO_SR) -> str:
    """Bundle the accumulated PCM bytes into a temp WAV file the legacy
    pipeline can consume.

    The legacy STT / VAD path (``has_voice``, ``transcribe``) is file-based.
    Rather than rewrite those for streaming bytes (Phase 2's job), Phase 1
    wraps the buffered chunks into a one-shot WAV per turn.
    """
    import wave
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="ws_turn_")
    try:
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(_WS_AUDIO_BYTES_PER_FRAME)
            w.setframerate(sr)
            w.writeframes(pcm)
        return path
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise


# ───────── sentence chunker bridge ─────────

async def _stream_reply_sentences(reply: str) -> AsyncIterator[str]:
    """Yield TTS-ready sentence chunks from a finished agent reply.

    Prefers ``server.streaming.chunk_for_tts`` (the `tts-chunker` agent's
    async API contract: it accepts an ``AsyncIterator[str]`` and emits
    sentence-sized chunks) when available. Falls back to the existing
    synchronous ``iter_sentences`` helper otherwise — both are owned by the
    `tts-chunker` agent, so we tolerate either shape during the rollout.
    """
    if not reply:
        return

    chunker = None
    try:
        from server import streaming as _streaming
        chunker = getattr(_streaming, "chunk_for_tts", None)
    except Exception:  # noqa: BLE001
        chunker = None

    if chunker is not None and asyncio.iscoroutinefunction(chunker):
        async def _one_shot() -> AsyncIterator[str]:
            yield reply
        try:
            async for sent in chunker(  # type: ignore[misc]
                _one_shot(),
                min_chars=TTS_CHUNK_MIN_CHARS,
                timeout_ms=TTS_CHUNK_TIMEOUT_MS,
            ):
                if sent and sent.strip():
                    yield sent.strip()
            return
        except TypeError:
            # Signature differs from the documented contract — fall through
            # to the sync chunker.
            pass
        except Exception as e:  # noqa: BLE001 — defensive against in-flight rewrites
            logging.getLogger("sage.app_ws").warning(
                "chunk_for_tts failed (%s); falling back to iter_sentences", e,
            )

    # Synchronous fallback: use the existing iter_sentences generator.
    from server.streaming import iter_sentences

    def _sync_chunks() -> Iterable[str]:
        return iter_sentences(iter([reply]))

    for sent in await asyncio.to_thread(lambda: list(_sync_chunks())):
        if sent and sent.strip():
            yield sent.strip()


# ───────── frame helpers ─────────

async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Send a JSON text frame, swallowing close-related errors."""
    try:
        await ws.send_text(json.dumps(payload, separators=(",", ":")))
    except (WebSocketDisconnect, RuntimeError):
        raise
    except Exception as e:  # noqa: BLE001
        logging.getLogger("sage.app_ws").warning("send_json failed: %s", e)


def _audio_chunk_frame(
    seq: int,
    text: str,
    mp3_bytes: bytes,
    pause_after_ms: int = 0,
) -> dict[str, Any]:
    return {
        "type": "audio_chunk",
        "seq": seq,
        "format": "mp3",
        "text": text,
        "pause_after_ms": max(0, int(pause_after_ms or 0)),
        "data": base64.b64encode(mp3_bytes).decode("ascii"),
    }


def _action_frame(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "action", "name": name, "args": args or {}}


def _control_frame(subtype: str, **data: Any) -> dict[str, Any]:
    return {"type": "control", "subtype": subtype, "data": data}


# ───────── per-session state ─────────

class _Session:
    """Per-WebSocket session state — one per connected user."""

    __slots__ = (
        "username", "session_id", "face_id", "hint",
        "asking_name", "onboarding_retries",
        "audio_buf", "image_b64", "turn_idx",
        "out_seq",
        # Phase 2 additions ─ EoU arbiter + post-TTS cooldown
        "silero", "robot_eou_hint", "utterance_start_ms",
        "tts_active_until_ms", "_finalize_in_flight",
        "had_speech",
        # Phase 10.5: barge-in abort signal. Set by the `barge_in` control
        # handler; checked by the TTS streaming loop between sentence chunks
        # so the player on the robot side can stop without finishing the reply.
        "barge_event",
        # Phase 10.5: in-flight agent turn task. Spawned so the receive loop
        # keeps draining frames (especially barge_in) during long replies.
        "active_turn_task",
        # Phase 11 / Option B: parallel vision call task fired right after
        # STT and awaited inside _emit_agent_turn. Cancelled on short-
        # circuit paths (motion_trigger / crisis / echo_reject).
        "_vision_task",
        # Therapy turn count drives the every-Nth-turn refresh hint
        # for the therapist agent (vision cache itself was removed —
        # every visual trigger now runs fresh).
        "_therapy_turn_count",
        # Phase 11.10 — per-session streaming STT (ElevenLabs Scribe Realtime).
        "_streaming_stt",
    )

    def __init__(self, username: str) -> None:
        self.username = username
        self.session_id = str(uuid.uuid4())
        self.face_id: str | None = None
        self.hint: str | None = None
        self.asking_name: bool = False
        self.onboarding_retries: int = 0
        self.audio_buf = bytearray()
        self.image_b64: str | None = None
        self.turn_idx = 0
        self.out_seq = 0  # monotonic seq for outgoing audio_chunk frames

        # Phase 2: server-side streaming Silero (set lazily on first audio
        # chunk so a missing dependency at import time doesn't kill the
        # session). See `_get_streaming_silero()` for the construction path.
        self.silero: Any | None = None
        # Robot energy-VAD hint flag — set when the client sends
        # `end_of_utterance` control with `robot_eou_hint=True`. Cleared
        # when the turn finalizes.
        self.robot_eou_hint: bool = False
        # Wall-clock ms of the first inbound audio_chunk for the current
        # utterance — used for the 60 s hard ceiling in the arbiter.
        self.utterance_start_ms: float = 0.0
        # Wall-clock ms beyond which inbound audio is ignored as TTS
        # echo (post-TTS cooldown). 0 = cooldown inactive.
        self.tts_active_until_ms: float = 0.0
        # Reentrancy guard so two arbiter triggers (e.g. silero-driven and
        # robot-hint-driven) don't both kick `_process_turn`.
        self._finalize_in_flight: bool = False
        # True once silero has registered any speech inside the current
        # utterance. Silence-only utterances (the user never spoke) wait
        # for the robot's EoU hint or the hard ceiling — we never
        # auto-finalize a buffer that contains no detected voice.
        self.had_speech: bool = False
        # Phase 10.5: a fresh asyncio.Event used to abort the TTS streaming
        # loop on `barge_in`. Re-created at the start of every agent turn so
        # one barge can't preemptively cancel the next reply.
        self.barge_event: asyncio.Event = asyncio.Event()
        # Phase 10.5: tracks the currently-running agent turn so the receive
        # loop can keep handling control frames (especially barge_in) while
        # the turn is mid-flight.
        self.active_turn_task: Any | None = None
        # Phase 11 / Option B: pending vision call. Set right after STT in
        # _process_turn, awaited just-in-time by _emit_agent_turn, cancelled
        # on short-circuit paths.
        self._vision_task: Any | None = None
        self._therapy_turn_count: int = 0
        # Phase 11.10 — streaming STT (ElevenLabs Scribe Realtime).
        # Per-session WS that receives PCM frames as they arrive and
        # emits partial + final transcripts. Used in place of the
        # buffer-then-upload Whisper/Deepgram path when configured.
        self._streaming_stt: Any | None = None

    def reset_turn(self) -> None:
        self.audio_buf = bytearray()
        # Do NOT clear image_b64 here — the robot snaps a fresh image on
        # session_open + every tts_ended, but those snaps may not arrive
        # before the next user utterance triggers vision kickoff. Keeping
        # the previous image means the agent gets at-worst a slightly
        # stale frame instead of `vision_status=skipped`. The image gets
        # overwritten when the next snap arrives at the WS frame handler.
        self.robot_eou_hint = False
        self.utterance_start_ms = 0.0
        self.had_speech = False
        # Reset streaming VAD state so the next utterance doesn't inherit
        # silence already accumulated at the tail of the prior one.
        if self.silero is not None:
            try:
                self.silero.reset()
            except Exception:
                # If reset() throws, drop the instance — it'll be re-created
                # lazily on the next chunk.
                self.silero = None

    def next_seq(self) -> int:
        self.out_seq += 1
        return self.out_seq


# ───────── EoU arbiter ─────────


def _silero_silence_ms(sess: _Session) -> int:
    """Best-effort read of the streaming Silero's silence accumulator.

    Returns 0 if the stream isn't available — the arbiter then degrades to
    robot-hint-only finalization, matching pre-Phase-2 behavior.
    """
    sl = sess.silero
    if sl is None:
        return 0
    try:
        return int(sl.silence_duration_ms())
    except Exception:
        return 0


def _silero_speaking(sess: _Session) -> bool:
    """True if the streaming Silero currently registers speech.

    On any error we conservatively report False so the arbiter doesn't
    block finalization on a failing detector.
    """
    sl = sess.silero
    if sl is None:
        return False
    try:
        return bool(sl.is_speech_now())
    except Exception:
        return False


async def _maybe_run_semantic_endpoint(transcript: str | None) -> bool:
    """Async-tolerant wrapper around ``semantic_endpoint.is_complete_thought``.

    The `semantic-endpoint` agent is upgrading the function to ``async def``
    in a parallel worktree. We accept either a coroutine return value or a
    sync bool so we don't break when only one side has shipped. Empty or
    None transcripts return False — the caller treats that as "wait".
    """
    if not transcript or not transcript.strip():
        return False
    from server import semantic_endpoint  # imported lazily; cheap after first
    try:
        result = semantic_endpoint.is_complete_thought(transcript)
        if asyncio.iscoroutine(result):
            return bool(await result)
        return bool(result)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "semantic_endpoint_call_failed",
            error=repr(e),
            transcript_preview=(transcript or "")[:80],
        )
        return False


async def _should_finalize_turn(sess: _Session,
                                transcript_so_far: str | None,
                                now_ms: float) -> bool:
    """Multi-signal end-of-utterance arbiter.

    Returns True iff the audio buffer should be flushed and ``_process_turn``
    invoked. Combines:

    1. **Silero silence ≥ ``EOU_MIN_SILENCE_MS``** → finalize outright.
    2. **Robot hint** + Silero confirms no-speech for ``EOU_HINT_CONFIRM_MS``
       → finalize.
    3. **Silero silence ≥ ``EOU_SEMANTIC_SILENCE_MS``** + a non-empty
       transcript snapshot that ``is_complete_thought`` says is complete →
       finalize early.
    4. Hard ceiling: utterance duration ≥ ``EOU_HARD_CEILING_MS``
       (default 60 s) → finalize.

    Otherwise returns False (keep buffering). When ``StreamingSilero`` is
    unavailable the function degrades to "finalize on robot hint", matching
    Phase 1 behavior.

    The whole call is timed via ``metrics.phase_timer("eou_arbiter")`` when
    that label exists; if the metrics module hasn't whitelisted it yet, the
    timer is a no-op and the local ``phase_ms`` dict still records the
    elapsed time.
    """
    # If we have no Silero stream, the only signal we can act on is the
    # robot hint — preserve the pre-Phase-2 contract.
    if sess.silero is None:
        if sess.robot_eou_hint:
            return True
        # Hard ceiling still applies even with no detector.
        if (sess.utterance_start_ms
                and (now_ms - sess.utterance_start_ms)
                >= EOU_HARD_CEILING_MS):
            logger.info(
                "eou_arbiter_hard_ceiling",
                user=sess.username, session_id=sess.session_id,
                duration_ms=round(now_ms - sess.utterance_start_ms, 1),
                detector="absent",
            )
            return True
        return False

    silence_ms = _silero_silence_ms(sess)
    speaking = _silero_speaking(sess)

    # 1. Silero says we've been silent long enough on its own — but only
    #    after speech actually happened. A clip that is silence-from-the-
    #    start gets handled by the robot hint or the hard ceiling, not by
    #    the silence accumulator (otherwise we'd auto-finalize before the
    #    user has even started talking).
    if sess.had_speech and silence_ms >= EOU_MIN_SILENCE_MS:
        return True

    # 2. Robot also thinks we're done — only need a brief silero-confirmed
    #    silence window to commit. This branch fires regardless of
    #    `had_speech` because the robot's energy VAD already endpointed
    #    the utterance — we trust it.
    if sess.robot_eou_hint and (silence_ms >= EOU_HINT_CONFIRM_MS or not speaking):
        return True

    # 3. Semantic-early branch — only when we already have a transcript
    #    snapshot to inspect (cheap conditions warrant the LLM call).
    if (sess.had_speech
            and transcript_so_far
            and silence_ms >= EOU_SEMANTIC_SILENCE_MS
            and not speaking
            and await _maybe_run_semantic_endpoint(transcript_so_far)):
        return True

    # 4. Hard ceiling.
    if (sess.utterance_start_ms
            and (now_ms - sess.utterance_start_ms) >= EOU_HARD_CEILING_MS):
        logger.info(
            "eou_arbiter_hard_ceiling",
            user=sess.username, session_id=sess.session_id,
            duration_ms=round(now_ms - sess.utterance_start_ms, 1),
            silence_ms=silence_ms, speaking=speaking,
        )
        return True

    return False


# ───────── crisis path ─────────

def _arm_post_tts_cooldown(sess: _Session) -> None:
    """Set the post-TTS cooldown window after the last audio_chunk fires.

    The receive loop drops inbound audio_chunk frames while
    ``time.time() * 1000 < sess.tts_active_until_ms`` so reverb echoing
    through NAO's mic in the moments after speaker shutoff doesn't fire a
    self-conversation loop. The padding (`TTS_COOLDOWN_PADDING_MS`) is added
    on top of `MIC_GATE_GRACE_MS` because the robot side resubscribes its
    mic after the grace window — frames in flight on the wire still need to
    be discarded server-side.
    """
    grace_ms = int(getattr(config, "MIC_GATE_GRACE_MS", 200) or 0)
    sess.tts_active_until_ms = (time.time() * 1000.0
                                + grace_ms
                                + TTS_COOLDOWN_PADDING_MS)


def _agent_turn_running(sess: _Session) -> bool:
    task = getattr(sess, "active_turn_task", None)
    return task is not None and not task.done()


async def _emit_crisis(ws: WebSocket, sess: _Session, transcript: str,
                       phase_ms: dict[str, float]) -> None:
    """Emit the hardcoded 988-hotline reply with TTS, plus a white-eye action."""
    sess.turn_idx += 1
    await _send_json(ws, _control_frame("crisis_lock",
                                        transcript=transcript,
                                        turn_idx=sess.turn_idx))
    await _send_json(ws, {
        "type": "control",
        "subtype": "transcript",
        "data": {"transcript": transcript,
                 "stt_ms": phase_ms.get("stt", 0)},
    })
    # Action so the robot's eyes shift while it speaks the hotline reply.
    await _send_json(ws, _action_frame("change_eye_color", {"color": "white"}))

    with _phase("tts_synth_first_chunk", phase_ms):
        mp3 = await asyncio.to_thread(
            _synth_for, sess.username, safety.HOTLINE_REPLY,
        )
    # Record the hotline reply for the substring/sentence echo guard before
    # we hand audio to the client — the next inbound transcript may echo it.
    _reset_reply_chunks(sess.username, safety.HOTLINE_REPLY)
    if mp3:
        await _send_json(
            ws,
            _audio_chunk_frame(sess.next_seq(), safety.HOTLINE_REPLY, mp3),
        )
    legacy.LAST_REPLY[sess.username] = safety.HOTLINE_REPLY
    _arm_post_tts_cooldown(sess)
    await _send_json(ws, _control_frame("tts_ended"))

    logger.info(
        "crisis_block",
        user=sess.username, session_id=sess.session_id,
        turn_idx=sess.turn_idx, phase_ms=phase_ms,
        transcript=transcript[:200],
        reply_preview=safety.HOTLINE_REPLY[:80],
        outcome="crisis",
    )


# ───────── motion-trigger short-circuit ─────────

def _persist_voice_profile(sess: _Session, profile: str) -> None:
    """Persist a voice-profile change for this session's user."""
    from server import session as _ses

    clean = (profile or "").strip().lower()
    if clean not in {"girl", "man", "neutral", "my"}:
        logger.warning(
            "voice_profile_invalid",
            user=sess.username, session_id=sess.session_id,
            voice_profile=clean,
        )
        return
    _ses.set_voice_profile(sess.username, clean)
    logger.info(
        "voice_profile_set",
        user=sess.username, session_id=sess.session_id,
        voice_profile=clean,
    )


async def _emit_motion(ws: WebSocket, sess: _Session, transcript: str,
                       motion: motion_trigger.MotionMatch,
                       phase_ms: dict[str, float]) -> None:
    sess.turn_idx += 1
    await _send_json(ws, {
        "type": "control",
        "subtype": "transcript",
        "data": {"transcript": transcript,
                 "stt_ms": phase_ms.get("stt", 0)},
    })

    # Phase 11.8: voice-profile picker is a motion trigger because we
    # want the change to take effect on THIS turn's ack, not the next
    # one. Persist BEFORE we synthesize so the new voice is used for
    # the canonical "Switching to X voice." reply.
    if motion.action == "set_voice_profile":
        try:
            profile = (motion.args or {}).get("profile") or ""
            _persist_voice_profile(sess, profile)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "voice_profile_set_error",
                user=sess.username, error=repr(exc),
            )
    else:
        # `learn_face` teaches the NAOqi face database on the robot — that is
        # what lets NAO's eyes recognise this person later. But recognition
        # only pays off if the server can map that name back to a profile,
        # and `memory.upsert_user` (the only writer of `users`) was never
        # called from the live path. Result: faces got learned on the robot
        # while the users table never heard of them, so every session looked
        # like a brand-new user. Persist BEFORE the action so a robot-side
        # failure can't leave the two stores disagreeing.
        if motion.action == "learn_face":
            learned = ((motion.args or {}).get("name") or "").strip()
            if learned:
                try:
                    await asyncio.to_thread(memory.ensure_user, learned, learned)
                    sess.username = learned.strip().lower()
                    sess.asking_name = False
                    sess.onboarding_retries = 0
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "learn_face_upsert_failed",
                        user=sess.username, name=learned, error=repr(exc),
                    )
        # Action FIRST so the robot can begin the gesture as the ack starts.
        # The voice-profile branch above doesn't have a robot-side action;
        # it's a server-state flip.
        await _send_json(ws, _action_frame(motion.action, motion.args))

    with _phase("tts_synth_first_chunk", phase_ms):
        mp3 = await asyncio.to_thread(_synth_for, sess.username, motion.ack)
    _reset_reply_chunks(sess.username, motion.ack)
    if mp3:
        await _send_json(
            ws,
            _audio_chunk_frame(sess.next_seq(), motion.ack, mp3),
        )
    legacy.LAST_REPLY[sess.username] = motion.ack
    _arm_post_tts_cooldown(sess)
    await _send_json(ws, _control_frame("tts_ended"))

    logger.info(
        "motion_match",
        user=sess.username, session_id=sess.session_id,
        turn_idx=sess.turn_idx, action=motion.action, args=motion.args,
        transcript=transcript[:200], reply_preview=motion.ack,
        phase_ms=phase_ms, outcome="motion_short_circuit",
    )


_ONBOARDING_NAME_PROMPT = "Hi, I'm NAO. What should I call you?"
_ONBOARDING_NAME_RETRY = "Sorry, what name should I call you?"

# Give up asking after this many rejected answers and carry on as `guest`.
# Without a bound, a user whose answer never parses as a name is asked
# forever — and if the robot hears its own prompt, it asks *itself* forever.
_ONBOARDING_NAME_MAX_RETRIES = 3

# Echo markers, as patterns rather than literals. The retry prompt says
# "what NAME should I call you"; a plain "what should i call you" literal
# missed it by one interposed word, so NAO's own retry sailed through the
# guard as if a user had said it — the ask/echo/ask loop. `test_onboarding_
# echo_guard_covers_every_prompt_constant` pins every prompt constant here
# so the two can never drift apart again.
_ONBOARDING_ECHO_PATTERNS = (
    re.compile(r"what\s+(?:name\s+)?should\s+i\s+call\s+you"),
    re.compile(r"\bi(?:'m| am)\s+nao\b"),
    re.compile(r"my camera is on for this conversation"),
    re.compile(r"say stop watching me"),
    re.compile(r"heads up"),
)


def _is_onboarding_prompt_echo(transcript: str) -> bool:
    """True when STT heard NAO's own onboarding/camera prompt.

    During onboarding we intentionally ask for short name answers, so the
    generic echo guard used for normal turns is bypassed. That made a bad
    failure possible: the robot hears "Hi, I'm NAO. What should I call you?"
    from its own speaker, the LLM treats it as user input, then replies as
    if the user introduced themselves as the assistant. Keep this guard
    deterministic and narrow so real names still pass.
    """
    t = re.sub(r"\s+", " ", transcript or "").strip().lower()
    if not t:
        return False
    return any(p.search(t) for p in _ONBOARDING_ECHO_PATTERNS)


async def _emit_onboarding_name_retry(
    ws: WebSocket,
    sess: _Session,
    *,
    reason: str,
) -> None:
    """Retry the name prompt without handing the turn to the LLM."""
    sess.onboarding_retries += 1
    if sess.onboarding_retries > _ONBOARDING_NAME_MAX_RETRIES:
        sess.asking_name = False
        logger.info(
            "onboarding_name_gave_up",
            user=sess.username,
            session_id=sess.session_id,
            reason=reason,
            retries=sess.onboarding_retries - 1,
        )
        return

    text = _ONBOARDING_NAME_RETRY
    phase_ms: dict[str, float] = {}
    with _phase("onboarding_name_retry_synth", phase_ms):
        try:
            mp3 = await asyncio.to_thread(_synth_for, sess.username, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "onboarding_name_retry_tts_failed",
                user=sess.username,
                session_id=sess.session_id,
                error=repr(exc),
            )
            mp3 = b""

    try:
        _reset_reply_chunks(sess.username, text)
    except Exception:
        pass

    if mp3:
        await _send_json(
            ws,
            _audio_chunk_frame(sess.next_seq(), text, mp3),
        )
        try:
            legacy.LAST_REPLY[sess.username] = text
        except Exception:
            pass
        _arm_post_tts_cooldown(sess)
        await _send_json(
            ws,
            _control_frame("tts_ended", sentences=1, asking_name=True),
        )
    else:
        await _send_json(
            ws,
            _control_frame("tts_chunk_skipped", text=text, asking_name=True),
        )

    logger.info(
        "onboarding_name_retry",
        user=sess.username,
        session_id=sess.session_id,
        reason=reason,
        phase_ms=phase_ms,
    )


async def _emit_returning_identity_greeting(
    ws: WebSocket,
    sess: _Session,
    display_name: str,
    *,
    reason: str,
) -> None:
    """Say a deterministic welcome when robot-side face recognition succeeds."""
    name = (display_name or "").strip()
    if not name:
        return
    # Returning identity wins over any pending unknown-face onboarding.
    sess.asking_name = False

    recap_line = await asyncio.to_thread(_last_recap_line, sess.username)
    text = _build_returning_greeting(name, recap_line)
    phase_ms: dict[str, float] = {}
    with _phase("returning_identity_greeting_synth", phase_ms):
        try:
            mp3 = await asyncio.to_thread(_synth_for, sess.username, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "returning_identity_greeting_tts_failed",
                user=sess.username,
                session_id=sess.session_id,
                face_name=name,
                error=repr(exc),
            )
            mp3 = b""

    try:
        _reset_reply_chunks(sess.username, text)
    except Exception:
        pass

    if mp3:
        await _send_json(
            ws,
            _audio_chunk_frame(sess.next_seq(), text, mp3),
        )
        try:
            legacy.LAST_REPLY[sess.username] = text
        except Exception:
            pass
        _arm_post_tts_cooldown(sess)
        await _send_json(ws, _control_frame(
            "tts_ended", sentences=1, returning_user=True,
        ))
    else:
        await _send_json(ws, _control_frame(
            "tts_chunk_skipped", text=text, returning_user=True,
        ))

    logger.info(
        "returning_identity_greeting",
        user=sess.username,
        session_id=sess.session_id,
        face_name=name,
        reason=reason,
        phase_ms=phase_ms,
    )


async def _emit_onboarding_name_prompt(
    ws: WebSocket,
    sess: _Session,
    *,
    reason: str,
) -> None:
    """Ask an unknown visible face for their name via the ElevenLabs path."""
    identity = _IDENTIFIED_USERS.get(sess.session_id) or {}
    if identity.get("recognized") and identity.get("name"):
        sess.asking_name = False
        logger.info(
            "onboarding_name_prompt_skipped",
            user=sess.username,
            session_id=sess.session_id,
            reason="recognized_identity_present",
        )
        return
    if sess.asking_name:
        return
    sess.asking_name = True

    text = _ONBOARDING_NAME_PROMPT
    phase_ms: dict[str, float] = {}
    with _phase("onboarding_name_prompt_synth", phase_ms):
        try:
            mp3 = await asyncio.to_thread(_synth_for, sess.username, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "onboarding_name_prompt_tts_failed",
                user=sess.username,
                session_id=sess.session_id,
                error=repr(exc),
            )
            mp3 = b""

    try:
        _reset_reply_chunks(sess.username, text)
    except Exception:
        pass

    if mp3:
        await _send_json(
            ws,
            _audio_chunk_frame(sess.next_seq(), text, mp3),
        )
        try:
            legacy.LAST_REPLY[sess.username] = text
        except Exception:
            pass
        _arm_post_tts_cooldown(sess)
        await _send_json(
            ws,
            _control_frame("tts_ended", sentences=1, asking_name=True),
        )
    else:
        await _send_json(
            ws,
            _control_frame("tts_chunk_skipped", text=text, asking_name=True),
        )

    logger.info(
        "onboarding_name_prompt",
        user=sess.username,
        session_id=sess.session_id,
        reason=reason,
        phase_ms=phase_ms,
    )


# ───────── full agent path ─────────

async def _emit_agent_turn(ws: WebSocket, sess: _Session,
                           transcript: str, image_b64: str | None,
                           phase_ms: dict[str, float],
                           t_user_done: float) -> None:
    """Run the agent graph, drain actions, stream sentence-by-sentence TTS."""
    sess.turn_idx += 1
    await _send_json(ws, {
        "type": "control",
        "subtype": "transcript",
        "data": {"transcript": transcript,
                 "stt_ms": phase_ms.get("stt", 0)},
    })

    # Phase 11 / Option B + 11.6 cache: await the parallel vision call
    # (kicked off right after STT) just-in-time. Bounded by 4s timeout.
    # If no fresh call was launched but we have a cached observation
    # within TTL, reuse that. Otherwise the prompt sees vision_status=
    # skipped and the safety rule kicks in (no visual claims).
    vision_observation = None
    vision_task = getattr(sess, "_vision_task", None)
    if vision_task is not None:
        try:
            with _phase("vision_call", phase_ms):
                vision_observation = await asyncio.wait_for(
                    vision_task, timeout=4.0,
                )
        except asyncio.TimeoutError:
            vision_task.cancel()
            vision_observation = {
                "vision_status": "failed",
                "vision_model": getattr(config, "VISION_MODEL", None),
                "vision_latency_ms": 4000.0,
                "vision_summary": "",
                "raw": None,
            }
        except Exception as exc:
            vision_observation = {
                "vision_status": "failed",
                "vision_model": getattr(config, "VISION_MODEL", None),
                "vision_latency_ms": None,
                "vision_summary": "",
                "raw": None,
            }
            logger.warning(
                "vision_task_error", user=sess.username,
                session_id=sess.session_id, error=repr(exc),
            )
        finally:
            sess._vision_task = None  # type: ignore[attr-defined]
        # No cache write — every visual question runs fresh vision now.
    else:
        # No vision task fired this turn (no trigger phrase). Skip
        # vision entirely — the prompt's Rule 0 will see status=skipped
        # and won't claim to see anything. We deliberately do NOT fall
        # back to a stale cached observation: the previous behavior
        # caused NAO to reuse a description from a prior user / setting
        # when a friend asked "do you see around me?" minutes later.
        vision_observation = {
            "vision_status": "skipped",
            "vision_model": None,
            "vision_latency_ms": None,
            "vision_summary": "",
            "raw": None,
        }

    # Audit log: every turn shows whether vision actually fired AND
    # whether we used a cached observation.
    logger.info(
        "turn_vision",
        user=sess.username, session_id=sess.session_id,
        turn_idx=sess.turn_idx,
        vision_status=vision_observation.get("vision_status"),
        vision_model=vision_observation.get("vision_model"),
        vision_latency_ms=vision_observation.get("vision_latency_ms"),
        vision_summary=(vision_observation.get("vision_summary") or "")[:200],
        vision_cached=bool(vision_observation.get("vision_cached")),
        vision_age_ms=vision_observation.get("vision_age_ms"),
    )

    # Phase 11.5 — TRUE STREAMING. Drive the agent in streaming mode so
    # token deltas flow through a sentence chunker → parallel TTS synth.
    # We emit audio_chunk frames as soon as each sentence's MP3 is ready,
    # without waiting for the full reply. Crisis check ALREADY ran (it's
    # in _process_turn upstream of here), so it's safe to start TTS.
    #
    # The legacy synchronous path remains as a fallback if the topology
    # is non-passthrough (debate / supervisor_veto / shared_pool need
    # multiple full Runner.run calls and can't stream). Detected on the
    # fly: if the streamer yields a `done` event with no preceding
    # `delta` events, we treat it as the sync fallback.

    # Reset the per-user reply-chunk window before any TTS — the echo
    # guard reads from this buffer and stale state from a prior turn
    # would create false rejections.
    _LAST_REPLY_CHUNKS.pop(sess.username, None)
    _LAST_REPLY_FULL.pop(sess.username, None)
    # Phase 10.5: fresh barge_event per turn.
    sess.barge_event = asyncio.Event()

    # Wire up the agent generator. It runs on its own event loop in a
    # background thread (Runner.run_streamed needs asyncio); we marshal
    # events back through a queue so this coroutine can `await` cleanly.
    import queue as _queue
    import threading as _threading
    _q: "asyncio.Queue[dict]" = asyncio.Queue()
    _loop = asyncio.get_running_loop()

    # Build per-turn identity payload from the most recent
    # user_identified scan + first-turn flag. _IDENTIFIED_USERS is
    # written by _ingest_control when the robot pushes user_identified.
    # `first_turn` is true only for the FIRST agent run after the scan;
    # we mark `greeted=True` after consuming so subsequent turns don't
    # re-prepend the greeting note.
    _identity = None
    _id_state = _IDENTIFIED_USERS.get(sess.session_id)
    if _id_state is not None:
        _identity = {
            "name": _id_state.get("name"),
            "recognized": bool(_id_state.get("recognized")),
            "face_visible": bool(_id_state.get("face_visible")),
            "first_turn": not _id_state.get("greeted", False),
        }
        # Mark greeted so the next turn's identity payload has
        # first_turn=False (prevents re-greeting on every turn).
        _id_state["greeted"] = True

    async def _drive_agent():
        try:
            async for ev in legacy.run_agent_streamed(
                sess.username, sess.hint, transcript, None,
                vision_observation, _identity,
            ):
                await _q.put(ev)
        except Exception as e:
            await _q.put({"type": "error", "error": repr(e)})
        finally:
            await _q.put({"type": "_eos"})

    # Phase 11.12 — pure-chat fast-fallback wrapper. When the user is
    # in hint='chat' and the transcript doesn't ask for embodied
    # actions, route through the safety-valve wrapper that watches for
    # nano stalling past 3.5 s and falls back to gpt-4o-mini with a
    # short audio filler. Other modes (therapy / morgan / skills) keep
    # the plain run_agent_streamed since their long latencies are
    # justified (vision, RAG, multi-tool reasoning).
    is_pure_chat_lane = False
    if (sess.hint or "").lower() == "chat":
        try:
            from server.agents import _wants_embodied
            is_pure_chat_lane = not _wants_embodied(transcript)
        except Exception:
            is_pure_chat_lane = True

    def _bridge():
        # Spawn a fresh asyncio loop in this thread to host the agent
        # generator. Use run_coroutine_threadsafe to push items back to
        # the main loop's queue so the WS handler can await them.
        async def _runner():
            stream_fn = (
                legacy.run_pure_chat_with_fallback
                if is_pure_chat_lane
                else legacy.run_agent_streamed
            )
            try:
                async for ev in stream_fn(
                    sess.username, sess.hint, transcript, None,
                    vision_observation, _identity,
                ):
                    asyncio.run_coroutine_threadsafe(
                        _q.put(ev), _loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    _q.put({"type": "error", "error": repr(e)}), _loop,
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    _q.put({"type": "_eos"}), _loop).result()
        asyncio.run(_runner())

    bridge_thread = _threading.Thread(target=_bridge, daemon=True)
    bridge_thread.start()

    # Sentence chunker: pulls from a `delta` -> str async generator and
    # yields complete-sentence strings as they form.
    delta_q: "asyncio.Queue[str | None]" = asyncio.Queue()

    async def _delta_iter():
        while True:
            d = await delta_q.get()
            if d is None:
                return
            yield d

    # Outputs from the streaming agent (dones / actions / agent handoffs).
    final_reply: dict = {"text": "", "active_agent": "agent",
                          "actions": [], "suppress_image": False,
                          "errored": False}

    async def _consume_events():
        """Pull events from _q, route to delta_q for sentences, capture rest."""
        sent_any_delta = False
        while True:
            ev = await _q.get()
            t = ev.get("type")
            if t == "delta":
                final_reply["text"] += ev.get("text", "")
                sent_any_delta = True
                await delta_q.put(ev.get("text", ""))
            elif t == "filler":
                # Phase 11.12 — pure-chat fallback emitted "One sec."
                # before kicking off the gpt-4o-mini retry. Pipe it
                # through the chunker the same way as a delta so the
                # robot starts speaking immediately while the fallback
                # model is still warming up.
                filler_text = ev.get("text", "One sec.")
                final_reply["text"] += filler_text + " "
                sent_any_delta = True
                logger.info(
                    "chat_fallback_filler",
                    user=sess.username, session_id=sess.session_id,
                    filler=filler_text,
                )
                await delta_q.put(filler_text + " ")
            elif t == "agent":
                final_reply["active_agent"] = ev.get("active_agent",
                                                      final_reply["active_agent"])
            elif t == "done":
                # Sync fallback or end-of-stream. If we never got deltas,
                # the topology fallback returned the full reply here; pump
                # it through the sentence chunker so TTS still happens.
                final_reply["active_agent"] = ev.get("active_agent",
                                                      final_reply["active_agent"])
                final_reply["actions"] = ev.get("actions") or []
                final_reply["suppress_image"] = bool(ev.get("suppress_image"))
                if not sent_any_delta:
                    full = ev.get("reply") or ""
                    final_reply["text"] = full
                    if full:
                        await delta_q.put(full)
            elif t == "error":
                final_reply["errored"] = True
                logger.warning(
                    "agent_stream_error", user=sess.username,
                    session_id=sess.session_id, error=ev.get("error"),
                )
            elif t == "_eos":
                await delta_q.put(None)
                return

    consumer_task = asyncio.create_task(_consume_events())

    from server.streaming import chunk_for_tts

    # Send `tts_started` as soon as we know we'll have audio. Active
    # agent might be wrong here (handoff lands later); send a fresh
    # agent_handoff control if the active_agent changes.
    await _send_json(ws, _control_frame("tts_started",
                                         active_agent=final_reply["active_agent"]))

    first_chunk_emitted = False
    sent_count = 0
    barged = False
    handoff_sent_for: str | None = None
    tts_total_t0 = time.perf_counter()
    # nao-therapy: per-turn dedup of synthesized sentences. The streaming
    # agent occasionally re-yields the same sentence (especially around
    # tool calls — e.g. learn_face / vision — where the runner re-plays
    # accumulated deltas after the tool returns). Without dedup the robot
    # speaks each sentence twice, doubling reply time and stealing the
    # user's next speaking window. Normalize on lower+stripped text and
    # short-circuit before synthesis to also save ElevenLabs cost.
    _sentences_seen_this_turn: set[str] = set()
    try:
        # chunk_for_tts collapses delta tokens into sentence-sized chunks.
        # We then synthesize each one in a background task so the next
        # synth can start while the current one is being emitted.
        async for sentence in chunk_for_tts(_delta_iter()):
            # Per-turn duplicate sentence guard.
            _dedup_key = (sentence or "").strip().lower()
            if _dedup_key and _dedup_key in _sentences_seen_this_turn:
                logger.info(
                    "tts_duplicate_sentence_skipped",
                    user=sess.username, session_id=sess.session_id,
                    sentence_preview=sentence[:80],
                )
                continue
            if _dedup_key:
                _sentences_seen_this_turn.add(_dedup_key)
            paced_chunks = breathing_pacing.expand_tts_pacing(sentence)
            if len(paced_chunks) > 1 or (
                paced_chunks and paced_chunks[0][1] > 0
            ):
                logger.info(
                    "tts_breath_pacing_expanded",
                    user=sess.username, session_id=sess.session_id,
                    chunks=len(paced_chunks),
                    sentence_preview=sentence[:100],
                )
            for tts_text, pause_after_ms in paced_chunks:
                # Barge guard between chunks.
                if sess.barge_event.is_set():
                    barged = True
                    logger.info(
                        "tts_barged", user=sess.username,
                        session_id=sess.session_id, sent_chunks=sent_count,
                    )
                    break
                # If a handoff happened mid-stream, surface it once before
                # the first audio chunk of the new agent.
                if handoff_sent_for != final_reply["active_agent"]:
                    await _send_json(ws, _control_frame(
                        "agent_handoff",
                        active_agent=final_reply["active_agent"],
                        suppress_image=bool(final_reply["suppress_image"]),
                    ))
                    handoff_sent_for = final_reply["active_agent"]
                t_synth = time.perf_counter()
                mp3 = await asyncio.to_thread(_synth_for, sess.username, tts_text)
                # Re-check after synth — barge_in can arrive during the
                # synthesize call (which can take hundreds of ms). Dropping
                # the freshly-synthesized chunk here keeps us within the
                # tight 600 ms barge budget the robot demands.
                if sess.barge_event.is_set():
                    barged = True
                    logger.info(
                        "tts_barged_post_synth", user=sess.username,
                        session_id=sess.session_id, sent_chunks=sent_count,
                    )
                    break
                # Phase 2: record EVERY synthesized chunk into the per-user
                # echo-guard window. The buffer is capped at _REPLY_CHUNKS_MAX
                # so long replies don't grow unbounded.
                _record_reply_chunk(sess.username, tts_text)
                elapsed = (time.perf_counter() - t_synth) * 1000.0
                if not first_chunk_emitted:
                    phase_ms["tts_synth_first_chunk"] = round(elapsed, 2)
                    phase_ms["e2e_user_to_first_audio"] = round(
                        (time.perf_counter() - t_user_done) * 1000.0, 2,
                    )
                    first_chunk_emitted = True
                if not mp3:
                    # TTS failed — emit a sentence-only control so the client can
                    # at least log it; skip the audio chunk.
                    await _send_json(ws, _control_frame(
                        "tts_chunk_skipped", text=tts_text,
                    ))
                    continue
                await _send_json(
                    ws,
                    _audio_chunk_frame(
                        sess.next_seq(), tts_text, mp3,
                        pause_after_ms=pause_after_ms,
                    ),
                )
                sent_count += 1
            if barged:
                break
    finally:
        phase_ms["tts_synth_total"] = round(
            (time.perf_counter() - tts_total_t0) * 1000.0, 2,
        )
        phase_ms["e2e_user_to_complete"] = round(
            (time.perf_counter() - t_user_done) * 1000.0, 2,
        )
        # Wait for the consumer to drain so final_reply is fully populated
        # (we need it for actions + transcript log + agent_handoff if it
        # never fired). 2s is plenty — Runner has long since finished by
        # the time the last sentence got synthesized.
        try:
            await asyncio.wait_for(consumer_task, timeout=2.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()

    # Variables expected by the rest of this function (post-TTS cooldown,
    # action drain, agent_handoff fallback, turn_complete log).
    reply = final_reply["text"]
    active_agent = final_reply["active_agent"]
    actions = final_reply["actions"] or []
    suppress_image = final_reply["suppress_image"]

    # If we never sent an agent_handoff (e.g. no streaming deltas + no
    # active agent change), send it now so the client knows who replied.
    if handoff_sent_for is None:
        await _send_json(ws, _control_frame(
            "agent_handoff",
            active_agent=active_agent,
            suppress_image=bool(suppress_image),
        ))
    # Drain actions AFTER reply (actions populated during streaming).
    # This is a deviation from the pre-streaming contract that emitted
    # actions BEFORE the first audio chunk — with streaming we don't
    # know the action list until the agent finishes. Trade-off:
    # gestures fire ~500ms later than ideal, but first audio fires
    # several seconds earlier. Net positive for the user.
    with _phase("action_dispatch", phase_ms):
        for action in actions:
            name = action.get("name") if isinstance(action, dict) else None
            if not name:
                continue
            args = action.get("args") if isinstance(action, dict) else None
            if name == "set_voice_profile":
                try:
                    _persist_voice_profile(
                        sess,
                        (args or {}).get("profile") if isinstance(args, dict) else "",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "voice_profile_set_error",
                        user=sess.username, error=repr(exc),
                    )
                continue
            await _send_json(ws, _action_frame(name, args or {}))

    # Arm the post-TTS cooldown right before signalling tts_ended. Frames
    # already in flight from the robot will land within the cooldown window
    # and be dropped. (On a barge, we still arm cooldown so the next
    # transcript isn't immediately consumed by mic-in-flight residue.)
    _arm_post_tts_cooldown(sess)
    if barged:
        # Surface the abort so the client can confirm its local stop took
        # effect. Lands BEFORE tts_ended so the order is observable.
        await _send_json(ws, _control_frame("tts_aborted",
                                             active_agent=active_agent,
                                             sent_chunks=sent_count))
    await _send_json(ws, _control_frame("tts_ended",
                                        sentences=sent_count,
                                        suppress_image=bool(suppress_image)))

    if reply:
        legacy.LAST_REPLY[sess.username] = reply

    logger.info(
        "turn_complete",
        user=sess.username, session_id=sess.session_id,
        turn_idx=sess.turn_idx, phase_ms=phase_ms,
        transcript=transcript[:200],
        reply_preview=(reply or "")[:80],
        active_agent=active_agent,
        actions=[(a.get("name") if isinstance(a, dict) else "?") for a in actions],
        outcome="ok",
    )


# ───────── per-turn pipeline ─────────

async def _process_turn(ws: WebSocket, sess: _Session) -> None:
    """Run the same pipeline as Flask /stream_turn:
    validate → has_voice → transcribe → reject → crisis → motion → agent.
    """
    if not sess.audio_buf:
        return

    # Capture the streaming-VAD verdict before reset_turn() clears the
    # per-utterance state. The legacy WAV VAD/STT path can still produce
    # hallucinated text from low-level noise; Silero is the stricter gate
    # that tells us whether the user actually spoke.
    turn_silero_available = sess.silero is not None
    turn_had_speech = bool(sess.had_speech)
    pcm = bytes(sess.audio_buf)
    image_b64 = sess.image_b64
    sess.reset_turn()

    phase_ms: dict[str, float] = {}
    t_user_done = time.perf_counter()

    # Materialize PCM into a WAV for the legacy file-based STT/VAD path.
    wav_path: str | None = None
    try:
        wav_path = _write_pcm_to_wav(pcm)

        with _phase("vad", phase_ms):
            if not legacy.validate_wav(wav_path):
                logger.warning(
                    "turn_complete",
                    user=sess.username, session_id=sess.session_id,
                    turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
                    outcome="rejected", reject_reason="invalid_audio",
                )
                await _send_json(ws, _control_frame(
                    "transcript", transcript="", reject_reason="invalid_audio",
                ))
                return
            if not legacy.has_voice(wav_path):
                phase_ms["e2e_user_to_first_audio"] = round(
                    (time.perf_counter() - t_user_done) * 1000.0, 2,
                )
                logger.info(
                    "turn_complete",
                    user=sess.username, session_id=sess.session_id,
                    turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
                    outcome="rejected", reject_reason="no_voice",
                )
                await _send_json(ws, _control_frame(
                    "transcript", transcript="", reject_reason="no_voice",
                ))
                return

        # Phase 11.10 — prefer the streaming STT's final transcript when
        # available. We've been forwarding live PCM into ElevenLabs Scribe
        # the whole time the user was talking, so by now the model usually
        # has already emitted partials and is just waiting for end_of_audio.
        # Send EoU and wait briefly for the final event; on timeout / no
        # streaming session, fall back to the legacy WAV-then-Whisper path.
        stt_streaming = getattr(sess, "_streaming_stt", None)
        transcript = ""
        used_streaming = False
        if stt_streaming is not None and stt_streaming.opened:
            with _phase("stt", phase_ms):
                try:
                    await stt_streaming.signal_eou()
                    final = await stt_streaming.await_final(timeout_s=1.5)
                except Exception as exc:  # noqa: BLE001
                    final = None
                    logger.warning(
                        "stt_streaming_finalize_error",
                        user=sess.username, error=repr(exc),
                    )
            if final:
                transcript = final
                used_streaming = True
                if stt_streaming.first_partial_ms is not None:
                    phase_ms["stt_first_partial"] = stt_streaming.first_partial_ms
                if stt_streaming.final_ms is not None:
                    phase_ms["stt_final"] = stt_streaming.final_ms
                logger.info(
                    "stt_streaming_final",
                    user=sess.username, session_id=sess.session_id,
                    transcript=transcript[:200],
                    first_partial_ms=stt_streaming.first_partial_ms,
                    final_ms=stt_streaming.final_ms,
                )
            # Reset for the next turn regardless of outcome.
            try:
                await stt_streaming.reset()
            except Exception:
                pass

        if not used_streaming:
            # Legacy buffer-then-upload path (Whisper or Deepgram).
            with _phase("stt", phase_ms):
                transcript = await asyncio.to_thread(
                    legacy.transcribe, wav_path,
                )
            logger.info(
                "stt_legacy",
                user=sess.username, session_id=sess.session_id,
                transcript=(transcript or "")[:200],
            )
    finally:
        if wav_path:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    if turn_silero_available and not turn_had_speech:
        phase_ms["e2e_user_to_first_audio"] = round(
            (time.perf_counter() - t_user_done) * 1000.0, 2,
        )
        logger.info(
            "turn_complete",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=(transcript or "")[:200],
            outcome="rejected", reject_reason="silero_no_speech",
        )
        await _send_json(ws, _control_frame(
            "transcript",
            transcript=transcript,
            reject_reason="silero_no_speech",
        ))
        return

    # Phase 11 / Option B: kick off the vision call IN PARALLEL with the
    # downstream pipeline (echo guard, crisis check, semantic endpoint,
    # motion trigger). The agent turn awaits this task just-in-time.
    # If the user's transcript bounces (rejected / motion-trigger /
    # crisis), the task is cancelled so we don't burn a needless API call.
    # No cache layer: each visual question gets a fresh GPT-4o call so
    # we never hand back a stale description from a prior user/setting.
    #
    # Phase 11.7 — fast-chat lane. When hint='chat' the agent is the
    # nano-model casual chatbot; vision is unused there and would just
    # eat 2 s for nothing. Skip the kickoff entirely so the prompt
    # sees vision_status=skipped (its safety rule kicks in).
    image_b64 = sess.image_b64
    sess._vision_task = None  # type: ignore[attr-defined]
    is_fast_chat = (sess.hint or "").lower() == "chat"
    print("[vision_trace] kickoff decision is_fast_chat={0} image_b64_present={1} hint={2!r}".format(
        is_fast_chat, image_b64 is not None,
        sess.hint), flush=True)
    if is_fast_chat:
        logger.info(
            "vision_decision",
            user=sess.username, session_id=sess.session_id,
            decision="skip", reason="fast_chat_lane",
        )
    elif image_b64:
        refresh, reason = _should_refresh_vision(sess, transcript)
        if refresh:
            sess._vision_task = asyncio.create_task(  # type: ignore[attr-defined]
                asyncio.to_thread(
                    _emotion_module.observe_face_for_turn, image_b64,
                )
            )
            logger.info(
                "vision_decision",
                user=sess.username, session_id=sess.session_id,
                decision="refresh", reason=reason,
            )
        else:
            logger.info(
                "vision_decision",
                user=sess.username, session_id=sess.session_id,
                decision="cache_hit", reason=reason,
            )

    reason = legacy.transcript_reject_reason(
        sess.username, transcript, asking_name=sess.asking_name,
    )
    if reason:
        logger.info(
            "turn_complete",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=(transcript or "")[:200],
            outcome="rejected", reject_reason=reason,
        )
        await _send_json(ws, _control_frame(
            "transcript", transcript=transcript, reject_reason=reason,
        ))
        return

    if sess.asking_name and _is_onboarding_prompt_echo(transcript):
        logger.info(
            "turn_rejected",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=(transcript or "")[:200],
            reason="onboarding_prompt_echo",
        )
        await _send_json(ws, _control_frame(
            "echo_reject",
            transcript=transcript,
            reason="onboarding_prompt_echo",
        ))
        return

    # Phase 2 echo guard — substring containment + per-sentence token
    # overlap. Layered ABOVE the legacy bigram-overlap check so we don't
    # touch `_legacy_helpers.py`. Runs before crisis/agent dispatch.
    if (not sess.asking_name
            and _is_substring_or_sentence_echo(sess.username, transcript)):
        logger.info(
            "turn_rejected",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=(transcript or "")[:200],
            reason="self_echo",
        )
        await _send_json(ws, _control_frame(
            "echo_reject",
            transcript=transcript,
            reason="self_echo",
        ))
        return

    # Crisis FIRST — on the raw clip — so a partial like
    # "I keep thinking about" can't be quietly waited on.
    with _phase("crisis_check", phase_ms):
        crisis = await asyncio.to_thread(safety.crisis_check, transcript)
    if crisis.positive:
        legacy.consume_partial(sess.username, transcript)
        await _emit_crisis(ws, sess, transcript, phase_ms)
        return

    if sess.asking_name:
        name_motion = motion_trigger.detect_name_answer(transcript)
        if name_motion is not None:
            _cancel_pending_vision(sess)
            sess.asking_name = False
            await _emit_motion(ws, sess, transcript, name_motion, phase_ms)
            return
        _cancel_pending_vision(sess)
        logger.info(
            "turn_complete",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=(transcript or "")[:200],
            outcome="rejected", reject_reason="asking_name_not_name",
        )
        await _emit_onboarding_name_retry(
            ws, sess, reason="asking_name_not_name",
        )
        return

    # Semantic endpointing — wait for more audio if the user trailed off.
    from server import semantic_endpoint
    if (semantic_endpoint.USE_SEMANTIC_ENDPOINT
            and not await asyncio.to_thread(
                semantic_endpoint.is_complete_thought, transcript)
            and not legacy.partial_wait_limit_hit(sess.username)):
        legacy.stash_partial(sess.username, transcript)
        await _send_json(ws, _control_frame(
            "transcript",
            transcript=transcript,
            wait=True,
            stt_ms=phase_ms.get("stt", 0),
        ))
        logger.info(
            "turn_complete",
            user=sess.username, session_id=sess.session_id,
            turn_idx=sess.turn_idx + 1, phase_ms=phase_ms,
            transcript=transcript[:200], outcome="rejected",
            reject_reason="wait_more_audio",
        )
        return

    # Stitch any buffered partial onto the current transcript.
    transcript = legacy.consume_partial(sess.username, transcript)

    # Motion-trigger short-circuit — bypass the LLM for clear body commands.
    with _phase("motion_trigger", phase_ms):
        motion = motion_trigger.detect(transcript)
    if motion is not None:
        # Cancel the parallel vision call — motion path doesn't use it
        # and we shouldn't burn an API call we won't read.
        _cancel_pending_vision(sess)
        await _emit_motion(ws, sess, transcript, motion, phase_ms)
        return

    # Phase 10.5: spawn the agent turn as a background task so the WS
    # receive loop keeps draining inbound frames during TTS streaming.
    # Without this, a `barge_in` control frame sent by the robot mid-reply
    # sits in the WS queue until the entire reply finishes — defeating
    # the whole point of barge-in. The task is owned by the session;
    # awaited at session_close (or cancelled if the WS drops).
    task = asyncio.create_task(
        _emit_agent_turn(
            ws, sess, transcript, image_b64, phase_ms, t_user_done,
        )
    )
    sess.active_turn_task = task

    def _clear_active_turn(done_task: asyncio.Task) -> None:
        if sess.active_turn_task is done_task:
            sess.active_turn_task = None
        try:
            exc = done_task.exception()
        except asyncio.CancelledError:
            return
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "agent_turn_task_exception_check_failed",
                user=sess.username, session_id=sess.session_id,
                error=repr(err),
            )
            return
        if exc is not None:
            logger.warning(
                "agent_turn_task_failed",
                user=sess.username, session_id=sess.session_id,
                error=repr(exc),
            )

    task.add_done_callback(_clear_active_turn)


# ───────── frame ingest ─────────

async def _finalize_turn_if_ready(ws: WebSocket, sess: _Session,
                                  *, force: bool = False) -> bool:
    """Run the arbiter (or force) and process the turn if it says so.

    Returns True if a turn was actually processed. The reentrancy guard
    prevents two arbiter triggers (silero-driven from inside an audio
    chunk and robot-hint-driven from an `end_of_utterance` control) from
    both kicking ``_process_turn``.
    """
    if sess._finalize_in_flight:
        return False
    if not sess.audio_buf:
        return False

    decision = force
    if not decision:
        # Time the arbiter via metrics.phase_timer when the label is
        # whitelisted; otherwise fall through to the no-op timer that
        # still records into a local dict.
        local_phase: dict[str, float] = {}
        with _phase("eou_arbiter", local_phase):
            decision = await _should_finalize_turn(
                sess, transcript_so_far=None,
                now_ms=time.time() * 1000.0,
            )

    if not decision:
        return False

    sess._finalize_in_flight = True
    try:
        await _process_turn(ws, sess)
    finally:
        sess._finalize_in_flight = False
    return True


async def _ingest_frame(ws: WebSocket, sess: _Session,
                        frame: dict[str, Any]) -> bool:
    """Apply one inbound frame.

    Returns True to continue the loop, False if the session should close.
    """
    ftype = frame.get("type")

    if ftype == "audio_chunk":
        # Phase 2 — post-TTS cooldown. Drop frames that arrive while the
        # robot's speaker is still ringing in the room. Frames in flight
        # from the robot's own mic land here; without this gate they
        # could echo back through STT and trigger a self-conversation
        # loop.
        now_ms = time.time() * 1000.0
        if now_ms < sess.tts_active_until_ms:
            counter = _resolve_echo_drop_counter()
            if counter is not None:
                try:
                    counter.inc()
                except Exception:
                    pass
            return True

        # While the previous turn is still generating/speaking, do not build
        # a second utterance from room noise or the user's next words. The
        # robot sends explicit barge_in control frames for interruption; raw
        # mic frames during this window were creating duplicate no_voice turns.
        if _agent_turn_running(sess):
            return True

        b64 = frame.get("data") or ""
        if not b64:
            return True
        try:
            pcm = base64.b64decode(b64)
            sess.audio_buf.extend(pcm)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "audio_decode_error",
                user=sess.username, error=repr(e),
            )
            return True

        # mic_trace: visible proof on stderr that audio is reaching the server.
        # _Session uses __slots__, so we can't attach trace state to the
        # instance. Use a module-level dict keyed by session_id instead.
        sid = getattr(sess, "session_id", "?")
        _trace_state = _MIC_TRACE_STATE.setdefault(sid, {"count": 0, "logged": False})
        if not _trace_state["logged"]:
            _trace_state["logged"] = True
            print(
                "[mic_trace] server_audio_chunk_received user={0} bytes={1}".format(
                    sess.username, len(pcm)
                ),
                flush=True,
            )
        _trace_state["count"] += 1
        if _trace_state["count"] % 50 == 0:
            print(
                "[mic_trace] server_audio_chunk_received count={0} buf_bytes={1}".format(
                    _trace_state["count"], len(sess.audio_buf)
                ),
                flush=True,
            )

        # Phase 11.10 — also forward the PCM to the streaming STT WS
        # if it's open. We keep buffering into sess.audio_buf for the
        # legacy fallback path; double-feeding both is cheap (one is
        # in-memory bytearray append, the other is one ws.send()) and
        # gives us a clean automatic fallback if EL fails mid-turn.
        stt_streaming = getattr(sess, "_streaming_stt", None)
        if stt_streaming is not None and stt_streaming.opened:
            try:
                await stt_streaming.feed(pcm)
            except Exception as exc:  # noqa: BLE001 — never block legacy
                logger.warning(
                    "stt_streaming_feed_error",
                    user=sess.username, error=repr(exc),
                )

        # First chunk of a new utterance: stamp the start time and (re)build
        # the streaming silero. We rebuild on each fresh utterance instead of
        # relying solely on `reset()` so a model that crashed mid-utterance
        # doesn't leave the session permanently silero-less.
        if sess.utterance_start_ms == 0.0:
            sess.utterance_start_ms = now_ms
            if sess.silero is None:
                sess.silero = _get_streaming_silero()

        # Feed silero. The detector is permissive on internal errors —
        # we never let a VAD bug kill an in-flight session.
        if sess.silero is not None:
            try:
                sess.silero.feed(pcm)
                # Mark that speech has been heard at least once in this
                # utterance. Silero's `is_speech_now()` flips between True
                # during talking and False during pauses; we just OR it in.
                if not sess.had_speech and _silero_speaking(sess):
                    sess.had_speech = True
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "silero_feed_failed",
                    user=sess.username, error=repr(e),
                )
                # Don't drop the session; just retire the broken instance.
                sess.silero = None

        # Arbiter check — may finalize the turn right here.
        await _finalize_turn_if_ready(ws, sess)
        return True

    if ftype == "image":
        b64 = frame.get("data") or ""
        if b64:
            sess.image_b64 = b64
            print("[vision_trace] image stashed user={0} bytes_b64={1}".format(
                sess.username, len(b64)), flush=True)
        return True

    if ftype == "control":
        return await _ingest_control(ws, sess, frame)

    logger.warning("unknown_frame_type", user=sess.username, ftype=ftype)
    return True


# ───────── wake_event handler (Phase 3) ─────────

async def _handle_wake_event(ws: WebSocket, sess: _Session,
                             data: dict[str, Any]) -> bool:
    """Phase 3 server-wake-event handler.

    Robot transitions AWARE→ENGAGED on its side and signals the server with
    a `wake_event` control frame. We then:

      1. Log + Prometheus-count by gate.
      2. Persist a row to `safety_events` (one source of truth for all
         identity-relevant signals — see PRD §Phase 3).
      3. Decide returning vs new user from the `users` table; if returning
         within ``WAKE_RESUME_WINDOW_S`` (default 24 h), bind the
         SQLiteSession (via the existing `ensure_active_session` helper)
         and synthesize a personalized greeting.
      4. Emit one `audio_chunk` frame with the greeting (returning user
         only). Arm the post-TTS cooldown so the greeting doesn't echo
         back through STT.
      5. Always finish with a `ready_to_listen` control frame and flip the
         per-session FSM tag to ``listening``. New users skip the greeting
         (deferred to Phase 8 onboarding).

    The whole handler is wrapped in `_phase("wake_to_first_audio", ...)`
    which falls back to a no-op timer when the metrics module hasn't
    whitelisted the label yet — same defensive pattern as the EoU arbiter.
    """
    face_id_raw = data.get("face_id")
    gate = str(data.get("gate") or "unknown")
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        distance_m = float(data.get("distance_m") or 0.0)
    except (TypeError, ValueError):
        distance_m = 0.0
    face_id = (str(face_id_raw).strip() if face_id_raw is not None else "")

    phase_ms: dict[str, float] = {}
    with _phase("wake_to_first_audio", phase_ms):
        # 1. Structured log — load-bearing for telemetry dashboards.
        logger.info(
            "wake_event",
            user=sess.username, session_id=sess.session_id,
            face_id=face_id or None, gate=gate,
            confidence=round(confidence, 3),
            distance_m=round(distance_m, 3),
        )

        # 2. Prometheus counter (gate-labelled).
        counter = _resolve_wake_events_counter()
        if counter is not None:
            try:
                counter.labels(gate=gate).inc()
            except Exception:  # noqa: BLE001 — never break wake on a metric error
                pass

        # 3. Persist to safety_events. The table predates Phase 3 — its
        # original purpose was invariant-violation logging, but the PRD
        # explicitly reuses it as the audit trail for wake events too
        # (single source of truth keeps the data ergonomic).
        try:
            from server import session as _session
            payload = json.dumps(
                {"face_id": face_id, "gate": gate,
                 "confidence": round(confidence, 3),
                 "distance_m": round(distance_m, 3)},
                default=str,
            )
            await asyncio.to_thread(
                _session.append_safety_event,
                sess.username,
                sess.turn_idx,
                "wake_event",
                "info",
                payload,
            )
        except Exception as e:  # noqa: BLE001 — best-effort persistence
            logger.warning(
                "wake_event_persist_failed",
                user=sess.username, error=repr(e),
            )

        # 4. Returning-user lookup against the `users` table managed by
        #    `server.memory`. The ``WAKE_RESUME_WINDOW_S`` knob defaults
        #    to 24 h per the PRD.
        is_returning = False
        display_name: str | None = None
        ask_name_after_ready = False
        if face_id:
            is_returning, display_name = await asyncio.to_thread(
                _lookup_returning_user, face_id,
            )
            sess.face_id = face_id
            if is_returning and display_name:
                try:
                    sess.username = str(display_name).strip().lower()
                    sess.asking_name = False
                except Exception:
                    pass

        # 5. Bind / resume the SQLiteSession. `ensure_active_session` is
        #    idempotent — it'll reuse an existing active session id for
        #    this username or create a new one. The Agents SDK's
        #    SQLiteSession itself is keyed by `user:<username>` and lives
        #    in the same DB, so chat history naturally resumes when the
        #    same username walks back up.
        try:
            await asyncio.to_thread(
                legacy.ensure_active_session, sess.username, sess.hint,
            )
        except Exception as e:  # noqa: BLE001 — never block wake on this
            logger.warning(
                "wake_session_resume_failed",
                user=sess.username, error=repr(e),
            )

        # 6. Greeting path.
        if is_returning:
            recap_line = await asyncio.to_thread(
                _last_recap_line, sess.username,
            )
            greeting = _build_returning_greeting(display_name, recap_line)

            with _phase("tts_synth_first_chunk", phase_ms):
                mp3 = await asyncio.to_thread(
                    _synth_for, sess.username, greeting,
                )

            # Record for the substring/sentence echo guard before shipping
            # audio — the next inbound transcript may echo this greeting.
            _reset_reply_chunks(sess.username, greeting)

            if mp3:
                await _send_json(
                    ws,
                    _audio_chunk_frame(sess.next_seq(), greeting, mp3),
                )
            else:
                # TTS failed — emit a transcript-only control so the robot
                # can still surface the greeting visually / in logs.
                await _send_json(ws, _control_frame(
                    "tts_chunk_skipped", text=greeting,
                ))
            legacy.LAST_REPLY[sess.username] = greeting
            _arm_post_tts_cooldown(sess)
            await _send_json(ws, _control_frame(
                "tts_ended", sentences=1,
            ))
            outcome = "greeted_returning"
            _IDENTIFIED_USERS[sess.session_id] = {
                "name": display_name,
                "recognized": True,
                "face_visible": bool(face_id),
                "ts": time.time(),
                "greeted": True,
                "prompted": False,
            }
        else:
            # New user (or unrecognised face) — ask their name here.
            #
            # This used to defer to the Phase 8 flow, which only fires the
            # prompt from a `user_identified` frame carrying an unrecognised
            # face. Waking by touch or proximity resolves no face at all, so
            # nothing ever asked: the robot woke, said nothing, and waited —
            # indistinguishable from broken. Ask on wake instead; the
            # `prompted` flag below keeps the face path from asking twice.
            outcome = "prompted_new_user"
            ask_name_after_ready = True

        # 7. Per-session FSM tag → ``listening``.
        _set_session_fsm_state(sess.session_id, "listening")

        # 8. ``ready_to_listen`` control — robot uses this to flip its
        #    LISTENING-state LEDs and start streaming PCM.
        await _send_json(ws, _control_frame(
            "ready_to_listen",
            face_id=face_id or None,
            gate=gate,
            is_returning_user=is_returning,
            display_name=display_name,
        ))

        # 8b. Ask the name only once the robot is listening, so the prompt
        #     isn't spoken into a mic that hasn't started streaming yet.
        if ask_name_after_ready:
            _IDENTIFIED_USERS[sess.session_id] = {
                "name": None,
                "recognized": False,
                "face_visible": bool(face_id),
                "ts": time.time(),
                "greeted": True,
                "prompted": True,
            }
            await _emit_onboarding_name_prompt(
                ws, sess, reason="wake_{0}".format(gate or "unknown"),
            )

    logger.info(
        "wake_event_handled",
        user=sess.username, session_id=sess.session_id,
        face_id=face_id or None, gate=gate,
        is_returning_user=is_returning,
        outcome=outcome,
        phase_ms=phase_ms,
    )
    return True


# ───────── Phase 7 — Robot-Side Brain cache sync ─────────


async def _maybe_push_brain_sync(ws: WebSocket, sess: _Session,
                                 data: dict[str, Any]) -> None:
    """Phase 7 brain-cache delta push.

    The robot ships a small identity/preferences cache (``~/nao_assist/brain.json``)
    capped at 64 KB. On ``session_open`` it includes its current
    ``brain_version``; if the server has newer data for this ``face_id``, we
    push it back via a ``brain_sync`` control frame so the robot can hydrate
    its cache before the conversation starts.

    Trigger conditions (all must hold):
    * ``data["face_id"]`` is a non-empty string (legacy clients omit it).
    * ``data["brain_version"]`` is a non-negative integer (legacy clients omit it).
    * ``server.session.pull_brain_updates`` returns a non-empty delta.

    The delta is sent as ``control { subtype: "brain_sync", data: {updates: {...}} }``
    BEFORE any greeting / first-turn announce so apply_updates lands before
    those would matter. Never raises: any error is logged and swallowed so
    the rest of session_open still runs.
    """
    face_id_raw = data.get("face_id")
    if not face_id_raw or not isinstance(face_id_raw, str):
        return
    face_id = face_id_raw.strip()
    if not face_id:
        return

    brain_version_raw = data.get("brain_version")
    # Both presence and parsability are required — bool would coerce here so
    # we reject it explicitly. Negative versions are also nonsense.
    if isinstance(brain_version_raw, bool) or brain_version_raw is None:
        return
    try:
        brain_version = int(brain_version_raw)
    except (TypeError, ValueError):
        return
    if brain_version < 0:
        return

    try:
        from server import session as _session
        updates = await asyncio.to_thread(
            _session.pull_brain_updates, face_id, brain_version,
        )
    except Exception as e:  # noqa: BLE001 — never break session_open on this
        logger.warning(
            "brain_sync_pull_failed",
            user=sess.username, session_id=sess.session_id,
            face_id=face_id, brain_version=brain_version, error=repr(e),
        )
        return

    if not updates:
        # No-op for unknown / unchanged users; legacy code-path continues.
        logger.info(
            "brain_sync_skipped",
            user=sess.username, session_id=sess.session_id,
            face_id=face_id, brain_version=brain_version,
        )
        return

    try:
        await _send_json(ws, _control_frame("brain_sync", updates=updates))
    except Exception as e:  # noqa: BLE001 — send failures never propagate
        logger.warning(
            "brain_sync_send_failed",
            user=sess.username, session_id=sess.session_id,
            face_id=face_id, brain_version=brain_version, error=repr(e),
        )
        return

    # Light-weight log for ops — count user keys rather than dump payloads
    # so we don't leak display_name / recap text into structured logs.
    logger.info(
        "brain_sync_pushed",
        user=sess.username, session_id=sess.session_id,
        face_id=face_id, brain_version=brain_version,
        users_updated=len((updates.get("users") or {})),
        prompt_fragments_updated=len(
            (updates.get("system_prompt_fragments") or {})
        ),
    )


# ───────── Phase 6 — first-turn camera-consent heads-up ─────────

# Default copy if the config knob isn't published yet (the `vision-debug`
# slug owns config.py for Phase 6 — fall back so we don't crash if our
# branch lands first).
_CAMERA_ANNOUNCE_FALLBACK = (
    "Heads up — my camera is on for this conversation. "
    "Say 'stop watching me' anytime."
)


async def _maybe_announce_camera_consent(ws: WebSocket, sess: _Session) -> None:
    """Phase 6 first-turn audible heads-up that the camera is on.

    Fires once per WS session (tracked via ``session.is_first_turn``) when
    BOTH of these hold:

    * ``config.CAMERA_DEFAULT_ON`` is True (operator opt-out lives here).
    * The user's persisted ``camera_consent`` is True.

    Synthesizes ``config.CAMERA_ANNOUNCE_TEXT`` via OpenAI TTS and sends
    one ``audio_chunk`` frame followed by a ``tts_ended`` control. The
    post-TTS cooldown is armed so the heads-up doesn't feed back into
    STT, mirroring the wake-event greeting path.

    Never raises — TTS / DB failures degrade silently to "no announce".
    The caller still proceeds with whatever else session_open does.
    """
    # Operator-level kill switch. Defaults to True so the heads-up still
    # plays if the parallel `vision-debug` agent's config edit hasn't
    # landed yet.
    if not bool(getattr(config, "CAMERA_DEFAULT_ON", True)):
        return

    try:
        from server import session as _session
    except Exception:  # noqa: BLE001 — should never happen in practice
        return

    try:
        first_turn = _session.is_first_turn(sess.session_id, sess.username)
    except TypeError:
        # Back-compat for tests/older deployments that still expose the
        # original one-argument helper.
        first_turn = _session.is_first_turn(sess.session_id)
    if not first_turn:
        return

    # Read consent off-loop so a slow SQLite call doesn't stall the
    # event loop. `get_camera_consent` lazily inserts default-1 rows.
    try:
        consent = await asyncio.to_thread(
            _session.get_camera_consent, sess.username,
        )
    except Exception as e:  # noqa: BLE001 — never break session_open on this
        logger.warning(
            "camera_announce_consent_failed",
            user=sess.username, error=repr(e),
        )
        return
    if not consent:
        return

    text = str(getattr(config, "CAMERA_ANNOUNCE_TEXT", _CAMERA_ANNOUNCE_FALLBACK))
    if not text.strip():
        return

    phase_ms: dict[str, float] = {}
    with _phase("camera_announce_synth", phase_ms):
        try:
            mp3 = await asyncio.to_thread(_synth_for, sess.username, text)
        except Exception as e:  # noqa: BLE001 — TTS down → silent skip
            logger.warning(
                "camera_announce_tts_failed",
                user=sess.username, error=repr(e),
            )
            mp3 = b""

    # Mark BEFORE the send so a transient send error doesn't cause us to
    # double-announce on the next session_open. Idempotency wins over a
    # one-time miss.
    _session.mark_first_turn_announced(sess.session_id, sess.username)

    if mp3:
        # Echo-guard hooks — keep the sentence list aligned with the
        # wake-event path so the regression suite stays happy.
        try:
            _reset_reply_chunks(sess.username, text)
        except Exception:
            pass
        try:
            await _send_json(
                ws, _audio_chunk_frame(sess.next_seq(), text, mp3),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "camera_announce_send_failed",
                user=sess.username, error=repr(e),
            )
            return
        try:
            legacy.LAST_REPLY[sess.username] = text
        except Exception:
            pass
        _arm_post_tts_cooldown(sess)
        try:
            await _send_json(ws, _control_frame("tts_ended", sentences=1))
        except Exception:
            pass
    else:
        # TTS unavailable — emit a transcript-only control so the robot
        # can still surface the heads-up visually / in logs.
        try:
            await _send_json(
                ws, _control_frame("tts_chunk_skipped", text=text),
            )
        except Exception:
            pass

    logger.info(
        "camera_announce",
        user=sess.username, session_id=sess.session_id,
        had_audio=bool(mp3),
        phase_ms=phase_ms,
    )


async def _ingest_control(ws: WebSocket, sess: _Session,
                          frame: dict[str, Any]) -> bool:
    sub = frame.get("subtype")
    data = frame.get("data") or {}

    if sub == "session_open":
        sess.face_id = data.get("face_id") or sess.face_id
        sess.hint = data.get("hint") or None
        legacy.ensure_active_session(sess.username, sess.hint)
        await _send_json(ws, _control_frame(
            "session_open_ack",
            session_id=sess.session_id,
            face_id=sess.face_id,
            hint=sess.hint,
        ))
        logger.info(
            "session_open",
            user=sess.username, session_id=sess.session_id,
            face_id=sess.face_id, hint=sess.hint,
        )
        # Phase 11.10 — open the ElevenLabs Scribe Realtime STT WS for
        # this session if the feature flag is set. Live mic frames will
        # be forwarded into it from the audio_chunk handler. Falls back
        # to legacy buffer-then-upload STT if open fails or the EL plan
        # doesn't include Scribe (gated on USE_ELEVENLABS_STT).
        try:
            from server import elevenlabs_stt as _el_stt
            if _el_stt.is_available():
                stt_sess = _el_stt.StreamingSttSession()
                ok = await stt_sess.open(sample_rate=16000, language="en")
                if ok:
                    sess._streaming_stt = stt_sess  # type: ignore[attr-defined]
                    logger.info(
                        "stt_streaming_opened",
                        user=sess.username, session_id=sess.session_id,
                        provider="elevenlabs_scribe",
                    )
                else:
                    logger.warning(
                        "stt_streaming_open_failed",
                        user=sess.username, session_id=sess.session_id,
                        error=stt_sess.error,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "stt_streaming_init_error",
                user=sess.username, error=repr(exc),
            )
        # Phase 7 — Robot-Side Brain cache sync. The robot announces its
        # current ``brain_version`` in the handshake; if the server has
        # newer data for this face_id, push it back as a ``brain_sync``
        # control frame BEFORE any greeting / first-turn announce so the
        # robot can apply the deltas before they'd matter for the next
        # spoken interaction. Both ``face_id`` and ``brain_version`` must
        # be present — older clients that don't carry the brain handshake
        # fields keep the legacy behavior. Never breaks session_open: any
        # error here is logged and swallowed.
        await _maybe_push_brain_sync(ws, sess, data)
        # Phase 6 — first-turn camera-consent heads-up. Runs BEFORE any
        # personalized greeting (wake_event greetings come later in the
        # stream when the engagement gates fire). Gated on the operator
        # knob so a deployment can opt out without code changes.
        await _maybe_announce_camera_consent(ws, sess)
        return True

    if sub == "session_close":
        await asyncio.to_thread(legacy.close_active_session, sess.username)
        _clear_session_fsm_state(sess.session_id)
        # Phase 6 — drop the per-session first-turn flag so the in-memory
        # tracker doesn't leak across long-lived server uptime.
        try:
            from server import session as _session
            _session.forget_session(sess.session_id)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        # Phase 11.10 — close the streaming STT WS for this session.
        stt_streaming = getattr(sess, "_streaming_stt", None)
        if stt_streaming is not None:
            try:
                await stt_streaming.close()
            except Exception:
                pass
            sess._streaming_stt = None  # type: ignore[attr-defined]
        await _send_json(ws, _control_frame(
            "session_end", session_id=sess.session_id,
        ))
        return False

    if sub == "wake_event":
        return await _handle_wake_event(ws, sess, data)

    if sub == "barge_in":
        # Phase 10.5: set the barge_event so the TTS loop in
        # _emit_agent_turn breaks out of its sentence iteration. Robot
        # client is responsible for stopping its local player; the
        # event just stops the server from sending more audio_chunk
        # frames at it.
        logger.info(
            "barge_in", user=sess.username, session_id=sess.session_id,
        )
        try:
            sess.barge_event.set()
        except Exception:
            # Defensive: if the event somehow wasn't initialized (older
            # _Session pickled in, etc.), don't kill the connection.
            pass
        return True

    if sub == "mic_resumed":
        logger.info(
            "mic_resumed", user=sess.username, session_id=sess.session_id,
        )
        return True

    if sub == "user_identified":
        # Robot's onboarding face-recognition scan completed. Payload:
        #   { name: <str|null>, recognized: <bool>,
        #     face_visible: <bool>, source: "face"|"unknown" }
        # Stash on session so the next turn can:
        #   • Greet returning user by name in the agent prompt
        #   • Prompt unknown user for their name + suggest `learn_face`
        face_name = (data.get("name") or "").strip() or None
        recognized = bool(data.get("recognized"))
        if face_name and face_name.lower() in {"guest", "unknown"}:
            face_name = None
            recognized = False
        face_visible = bool(data.get("face_visible"))
        prev_identity = _IDENTIFIED_USERS.get(sess.session_id) or {}
        prev_recognized = bool(
            prev_identity.get("recognized") and prev_identity.get("name")
        )
        prompted = bool(prev_identity.get("prompted"))
        greeted = bool(prev_identity.get("greeted", False))
        if prev_recognized and not (recognized and face_name):
            # Robot-side face recognition can emit a late unknown scan
            # after an earlier confident match. Do not let that overwrite
            # identity or re-open onboarding in the same session.
            sess.asking_name = False
            logger.info(
                "user_identified_ignored",
                session_id=sess.session_id,
                user=sess.username,
                prior_name=prev_identity.get("name"),
                incoming_name=face_name,
                incoming_recognized=recognized,
                reason="recognized_identity_sticky",
            )
            return True
        if recognized and face_name:
            prompted = False
            sess.asking_name = False
        should_greet_returning = bool(
            recognized and face_name and not greeted
        )
        # Store on session — tolerated even though _Session uses __slots__
        # because we go through a module-level dict (same trick the
        # mic_trace counters use).
        _IDENTIFIED_USERS[sess.session_id] = {
            "name": face_name,
            "recognized": recognized,
            "face_visible": face_visible,
            "ts": time.time(),
            "greeted": greeted,
            "prompted": prompted,
        }
        # If we recognized the user, set sess.username so SQLite-backed
        # memory + voice-profile-prefs persist correctly across sessions.
        if face_name and recognized:
            try:
                sess.username = face_name.lower()
            except Exception:
                pass
        logger.info(
            "user_identified",
            session_id=sess.session_id,
            user=sess.username,
            face_name=face_name,
            recognized=recognized,
            face_visible=face_visible,
        )
        if should_greet_returning:
            _IDENTIFIED_USERS[sess.session_id]["greeted"] = True
            await _emit_returning_identity_greeting(
                ws,
                sess,
                face_name,
                reason=str(data.get("source") or "recognized_face"),
            )
        if face_visible and not recognized and not face_name and not prompted:
            _IDENTIFIED_USERS[sess.session_id]["prompted"] = True
            _IDENTIFIED_USERS[sess.session_id]["greeted"] = True
            await _emit_onboarding_name_prompt(
                ws, sess, reason=str(data.get("source") or "unknown_face"),
            )
        return True

    if sub == "end_of_utterance":
        sess.asking_name = bool(data.get("asking_name", sess.asking_name))
        # Record the robot-side hint for the arbiter. The hint may already
        # be enough on its own (no Silero, or Silero confirms silence) —
        # `_finalize_turn_if_ready` runs the multi-signal check.
        sess.robot_eou_hint = bool(data.get("robot_eou_hint", True))
        # If the audio buffer is empty (the robot sent EoU without any
        # audio_chunk frames first), there's nothing to finalize — skip.
        if not sess.audio_buf:
            sess.robot_eou_hint = False
            sess.asking_name = False
            return True

        # Run the arbiter; if it doesn't decide to finalize on its own,
        # force the legacy behavior (Phase 1 contract: EoU control
        # finalizes the turn).
        finalized = await _finalize_turn_if_ready(ws, sess)
        if not finalized:
            await _finalize_turn_if_ready(ws, sess, force=True)
        sess.asking_name = False
        return True

    logger.warning(
        "unknown_control_subtype",
        user=sess.username, subtype=sub,
    )
    return True


# ───────── WS endpoint ─────────

@app.websocket("/ws/{username}")
async def ws_handler(websocket: WebSocket, username: str) -> None:
    if not _check_ws_auth(websocket):
        await websocket.close(code=4401)  # custom: unauthorized
        return

    await websocket.accept()
    sess = _Session(username=username or "guest")

    logger.info(
        "ws_connected", user=sess.username, session_id=sess.session_id,
    )

    # Wait for the FIRST frame and require it to be `session_open` per spec.
    first_raw: str | None = None
    try:
        first_raw = await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(
            "ws_disconnected_pre_handshake",
            user=sess.username, session_id=sess.session_id,
        )
        return

    try:
        first = json.loads(first_raw)
    except Exception:
        await websocket.close(code=4400)  # bad request
        return
    if not (isinstance(first, dict)
            and first.get("type") == "control"
            and first.get("subtype") == "session_open"):
        await websocket.close(code=4400)
        return

    await _ingest_control(websocket, sess, first)

    # Main receive loop.
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info(
                "turn_complete",
                user=sess.username, session_id=sess.session_id,
                turn_idx=sess.turn_idx, outcome="client_dropped",
            )
            return

        try:
            frame = json.loads(raw)
        except Exception:
            logger.warning("malformed_json", user=sess.username)
            continue
        if not isinstance(frame, dict):
            continue

        try:
            keep = await _ingest_frame(websocket, sess, frame)
        except WebSocketDisconnect:
            logger.info(
                "turn_complete",
                user=sess.username, session_id=sess.session_id,
                turn_idx=sess.turn_idx, outcome="client_dropped",
            )
            return
        except Exception as e:  # noqa: BLE001
            logger.error(
                "turn_error",
                user=sess.username, session_id=sess.session_id,
                turn_idx=sess.turn_idx, error=repr(e),
            )
            try:
                await _send_json(websocket, _control_frame(
                    "session_end", reason="server_error", error=repr(e),
                ))
            except Exception:
                pass
            try:
                await websocket.close(code=1011)  # internal error
            except Exception:
                pass
            return

        if not keep:
            try:
                await websocket.close(code=1000)
            except Exception:
                pass
            return


__all__ = ["app"]
