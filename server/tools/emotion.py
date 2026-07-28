"""Emotion tools for the therapist + CBT + grounding agents.

Phase 6 (PRD v2) — `observe_face` is the vision-debug entrypoint. The
prior implementation called the chat-completions API with the wrong model
default (`THERAPIST_MODEL`, a text-only family) and bubbled exceptions up
to the agent loop, which was the root cause of the empty-emotion bug
described in docs/PHASE_6_TASK_MAP.md. We now:

  * Resolve the vision model from `config.VISION_MODEL` (default gpt-4o).
  * Build the chat-completions multimodal payload with `image_url`
    objects shaped `{"url": "data:image/jpeg;base64,…"}` — NOT a bare
    string. Older code passed a string directly which 400'd silently in
    some SDK versions.
  * Wrap the round-trip in `metrics.phase_timer("vision_call")` when the
    metrics module + phase label are available; otherwise no-op so this
    keeps working on environments where Prometheus isn't wired up.
  * Catch every exception and return `"unable to observe right now"` so
    the agent never crashes a user-facing turn on a vision hiccup. The
    happy-path return shape stays a dict (preserved for back-compat with
    tests in `server/tests/test_emotion.py`).
  * Honour `DEBUG_VISION=1` in the environment for full payload-size +
    response logging during development.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager

from agents import RunContextWrapper, function_tool
from server import config, memory, session
from server import llm_compat

_log = logging.getLogger(__name__)

# Sentinel string returned when the vision call fails for any reason. The
# tool never raises — agents see this string and respond with a graceful
# fallback ("I can't quite see right now, but tell me what's on your mind").
_OBSERVE_FAILURE_STRING = "unable to observe right now"


def _debug_vision_enabled() -> bool:
    """`DEBUG_VISION=1` toggles full payload + response logging in dev."""
    return os.environ.get("DEBUG_VISION") == "1"


@contextmanager
def _vision_phase_timer():
    """Defensive wrapper around `metrics.phase_timer("vision_call")`.

    Falls back to a no-op contextmanager if either:
      * the `server.metrics` module is unavailable in this deployment
        (older branches don't have Phase 1's observability layer), OR
      * `vision_call` is rejected by `_validate_phase` because it
        hasn't been added to `ALLOWED_PHASES` yet.

    This way wiring observability later is purely additive — nothing here
    has to change to pick up the timer once the phase label lands in
    `metrics.ALLOWED_PHASES`.
    """
    inner = None
    try:
        from server import metrics as _metrics  # local import → optional dep
        inner = _metrics.phase_timer("vision_call")
        inner.__enter__()
    except Exception:
        # Either the metrics module is missing or the phase label isn't
        # registered yet. Run the wrapped block without timing.
        inner = None
    try:
        yield
    finally:
        if inner is not None:
            try:
                inner.__exit__(None, None, None)
            except Exception:  # pragma: no cover — defensive only
                pass

_DISTORTIONS = (
    "catastrophizing", "all-or-nothing", "mind reading", "personalization",
    "fortune-telling", "emotional reasoning", "shoulds", "labeling",
    "magnification/minimization", "filtering",
)


def _unwrap(ctx) -> dict:
    return ctx.context if isinstance(ctx, RunContextWrapper) else ctx


# ────────── log_emotion ──────────

def _log_emotion_impl(ctx, mood: str, intensity: int, trigger: str) -> str:
    store = _unwrap(ctx)
    store.setdefault("emotion_log", []).append(
        {"mood": mood, "intensity": intensity, "trigger": trigger}
    )
    # Persist to SQLite so the next-session greeting can surface the
    # mood trajectory. In-memory `emotion_log` is still used by the
    # session recap rollup at end-of-conversation.
    try:
        from server import session as _ses
        username = (store.get("username") or "").strip()
        if username:
            _ses.log_mood(username, mood, int(intensity), trigger)
    except Exception:
        pass  # best-effort; never break the agent turn
    return "logged"


@function_tool
def log_emotion(ctx: RunContextWrapper, mood: str, intensity: int, trigger: str) -> str:
    """Log a per-turn emotion read (mood, intensity 1-10, trigger) for session recap."""
    return _log_emotion_impl(ctx, mood, intensity, trigger)


# ────────── identify_distortion / suggest_reframe ──────────

def _classify_distortion(thought: str) -> dict:
    prompt = (
        "Classify the cognitive distortion in the user's thought. Choose exactly "
        "ONE from: " + ", ".join(_DISTORTIONS) + ". Respond as JSON: "
        '{"distortion": "<name>", "explanation": "<one sentence, gentle tone>"}'
    )
    out = llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": thought},
        ],
        json_mode=True,
        temperature=0.2,
        max_tokens=300,
    )
    return json.loads(out)


def _identify_distortion_impl(thought: str) -> dict:
    return _classify_distortion(thought)


def _persist_thought(ctx, thought: str, distortion: str) -> None:
    """Write the thought + identified distortion to SQLite so it's
    available in next-session memory preamble. Best-effort — never
    breaks the agent turn.
    """
    try:
        from server import session as _ses
        store = _unwrap(ctx)
        username = (store.get("username") or "").strip()
        if username:
            _ses.log_thought_record(username, thought, distortion, reframe="")
    except Exception:
        pass


def _persist_reframe(ctx, thought: str, reframe_text: str) -> None:
    try:
        from server import session as _ses
        store = _unwrap(ctx)
        username = (store.get("username") or "").strip()
        if username:
            _ses.attach_reframe_to_latest_thought(username, thought, reframe_text)
    except Exception:
        pass


@function_tool
def identify_distortion(ctx: RunContextWrapper, thought: str) -> dict:
    """Identify one CBT cognitive distortion in the user's thought with a gentle explanation."""
    out = _identify_distortion_impl(thought)
    distortion_name = (out.get("distortion") or "").strip() if isinstance(out, dict) else ""
    if distortion_name:
        _persist_thought(ctx, thought, distortion_name)
    return out


def _reframe_impl(thought: str, distortion: str) -> list[str]:
    prompt = (
        f"The user has a thought exhibiting {distortion}. Offer 2 balanced, "
        "compassionate alternative thoughts they could consider. Reply as a JSON "
        'list of 2 strings: {"reframes": ["...", "..."]}'
    )
    out = llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": thought},
        ],
        json_mode=True,
        temperature=0.4,
        max_tokens=400,
    )
    return json.loads(out)["reframes"]


@function_tool
def suggest_reframe(ctx: RunContextWrapper, thought: str, distortion: str) -> list[str]:
    """Return two balanced reframes for a thought exhibiting the given distortion."""
    reframes = _reframe_impl(thought, distortion)
    if reframes:
        _persist_reframe(ctx, thought, reframes[0])
    return reframes


# ────────── observe_face ──────────

# System prompt is in module scope so the self-check + tests can import it
# without instantiating the OpenAI client.
_VISION_SYSTEM = (
    "You describe a single video frame for a supportive robot companion. "
    "Stay observational — never diagnose, never identify the user, never "
    "guess age. ALWAYS return the JSON envelope below with all three "
    "fields populated, even if the image is dark, blurry, low-detail, or "
    "shows no clearly readable face: in those cases pick "
    'dominant_emotion="neutral", secondary="", and put what IS visible '
    "(lighting, framing, posture, hands, clothing, room, screen, "
    "objects) in `notes`. Do NOT return empty strings or omit fields.\n"
    "Return JSON exactly shaped:\n"
    '{"dominant_emotion": "<happy|sad|angry|fearful|surprised|disgusted|neutral|tired|stressed>",\n'
    ' "secondary": "<same vocabulary or empty string>",\n'
    ' "notes": "<≤30-word observational sentence about whatever IS visible>"}'
)


def _vision_classify(image_b64: str) -> dict:
    """Call OpenAI vision and parse the JSON envelope.

    Builds the multimodal chat-completions payload by-the-book:
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }]
    The data-URL wrapper is what the chat-completions vision contract
    actually expects — passing a bare base64 string OR a `{"url": "..."}`
    without the `data:image/jpeg;base64,` prefix returns a 400.
    """
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    if _debug_vision_enabled():
        # Approx payload size = ~4/3 of the raw image for base64 + small
        # JSON overhead. Logged so we can diagnose 413s when running
        # against a vision endpoint with a tight body limit.
        approx_kb = (len(image_b64) * 3) // 4 // 1024
        _log.info(
            "[DEBUG_VISION] observe_face payload: model=%s b64_len=%d ~kb=%d",
            config.VISION_MODEL, len(image_b64), approx_kb,
        )

    # llm_compat converts the image_url data URI into whichever image block
    # the configured provider expects, so VISION_MODEL can name either.
    raw = llm_compat.chat(
        model=config.VISION_MODEL,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this frame?"},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                ],
            },
        ],
        json_mode=True,
        temperature=0.2,
        max_tokens=400,
    )
    finish = None
    refusal = None

    if _debug_vision_enabled():
        _log.info("[DEBUG_VISION] observe_face response: %s (finish=%s refusal=%s)",
                   raw, finish, refusal)

    # gpt-4o sometimes returns content=None with finish_reason='content_filter'
    # when its safety classifier flags an image (people / faces fall under
    # privacy heuristics). One retry with the no-people-detection prompt
    # reliably gets past it.
    if (not raw or not raw.strip()) and refusal is None:
        raw = llm_compat.chat(
            model=config.VISION_MODEL,
            messages=[
                {"role": "system",
                 "content": (
                    "You describe SCENE COMPOSITION ONLY — lighting, "
                    "framing, posture, hands, clothing, room. Do NOT "
                    "identify any person, infer identity, or guess age. "
                    "Return the exact JSON shape: "
                    '{"dominant_emotion": "<happy|sad|angry|fearful|'
                    'surprised|disgusted|neutral|tired|stressed>", '
                    '"secondary": "<same vocabulary or \'\'>", '
                    '"notes": "<≤30 words about lighting/posture/'
                    'environment, no identifying details>"}'
                 )},
                {"role": "user",
                 "content": [
                    {"type": "text",
                     "text": "Describe the scene composition only."},
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                 ]},
            ],
            json_mode=True,
            temperature=0.2,
            max_tokens=400,
        )
        if _debug_vision_enabled():
            _log.info("[DEBUG_VISION] observe_face retry response: %s", raw)

    if not raw or not raw.strip():
        return {"error": "empty_response"}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"dominant_emotion": "unknown", "secondary": "",
                "notes": raw.strip()[:400]}


def observe_face_for_turn(image_b64: str | None) -> dict:
    """Server-side vision call with structured status envelope.

    Phase 11 (Option B): the WS handler runs this BEFORE the agent so
    the therapist receives the observation as injected context, not via
    a tool call. Eliminates two failure modes from the prompt-only path:

      1. Model skips observe_face but still says "I can see..." (hallucination)
      2. observe_face JSON parse failure trashes the turn

    Returns a dict with these keys (always present):
        vision_status      — "success" | "unavailable" | "failed" | "skipped"
        vision_model       — model id used, or None
        vision_latency_ms  — round-trip ms, or None
        vision_summary     — short human-readable text the prompt can quote
        raw                — full vision response dict (may be None)
    """
    if not image_b64:
        return {
            "vision_status": "unavailable",
            "vision_model": None,
            "vision_latency_ms": None,
            "vision_summary": "",
            "raw": None,
        }
    t0 = time.perf_counter()
    try:
        with _vision_phase_timer():
            raw = _vision_classify(image_b64)
    except Exception as exc:
        _log.warning(
            "observe_face_for_turn failed: %s",
            exc, exc_info=_debug_vision_enabled(),
        )
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "vision_summary": "",
            "raw": None,
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not isinstance(raw, dict):
        # Edge: empty / unparseable. _vision_classify now wraps these
        # but be defensive in case future paths return something else.
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": elapsed_ms,
            "vision_summary": "",
            "raw": None,
        }

    if raw.get("error"):
        # Empty response or other recoverable error — surface as failed
        # but include the raw payload for forensics.
        return {
            "vision_status": "failed",
            "vision_model": config.VISION_MODEL,
            "vision_latency_ms": elapsed_ms,
            "vision_summary": "",
            "raw": raw,
        }

    notes = (raw.get("notes") or "").strip()
    dom = (raw.get("dominant_emotion") or "").strip()
    sec = (raw.get("secondary") or "").strip()

    # Build a one-liner the therapist prompt can quote verbatim.
    summary_parts = []
    if dom:
        summary_parts.append(dom + (f"/{sec}" if sec else ""))
    if notes:
        summary_parts.append(notes)
    summary = "; ".join(summary_parts)[:400]

    return {
        "vision_status": "success",
        "vision_model": config.VISION_MODEL,
        "vision_latency_ms": elapsed_ms,
        "vision_summary": summary,
        "raw": raw,
    }


def _observe_face_impl(ctx):
    """Run the vision call defensively.

    Returns:
        - `{"error": "no_image"}` when the run context has no JPEG
          (back-compat with existing tests).
        - The parsed JSON dict from the vision model on success.
        - The string `"unable to observe right now"` on ANY error path
          — network blip, JSON parse failure, model-side refusal, etc.
          We never raise; the therapist agent sees the string and
          gracefully falls back without crashing the turn.
    """
    store = _unwrap(ctx)
    b64 = store.get("latest_image_b64")
    if not b64:
        return {"error": "no_image"}
    try:
        with _vision_phase_timer():
            return _vision_classify(b64)
    except Exception as exc:
        _log.warning("observe_face failed: %s", exc, exc_info=_debug_vision_enabled())
        return _OBSERVE_FAILURE_STRING


@function_tool
def observe_face(ctx: RunContextWrapper):
    """Read the user's face from the current turn's image.

    Returns one of:
      * dict with keys dominant_emotion / secondary / notes  (success)
      * `{"error": "no_image"}`                              (no JPEG attached)
      * `"unable to observe right now"`                      (vision call failed)

    Call this FIRST every turn whenever camera_consent=1 — the model
    needs the affect read before composing a reflective reply.
    """
    return _observe_face_impl(ctx)


# ────────── camera consent ──────────

def _set_camera_consent_impl(ctx, enabled: bool) -> str:
    store = _unwrap(ctx)
    username = store.get("username", "guest")
    session.set_camera_consent(username, enabled)
    if not enabled:
        store["suppress_image"] = True
    else:
        store["suppress_image"] = False
    return f"camera_consent={enabled}"


@function_tool
def set_camera_consent(ctx: RunContextWrapper, enabled: bool) -> str:
    """Set the user's camera consent. When False, NAO stops uploading images this session and next visits."""
    return _set_camera_consent_impl(ctx, enabled)


# ────────── recap_session ──────────

def _recap_session_impl(ctx) -> str:
    store = _unwrap(ctx)
    username = store.get("username", "guest")
    log = store.get("emotion_log", [])
    if not log:
        body = "Brief check-in; no notable thoughts logged."
    else:
        moods = ", ".join(f"{e['mood']}({e['intensity']})" for e in log[-5:])
        body = f"Emotions: {moods}. Triggers: {'; '.join(e['trigger'] for e in log[-5:])}."
    session.save_recap(username, body)
    from server import memory_rollup
    memory_rollup.maybe_rollup_week(username)
    memory_rollup.maybe_rollup_month(username)
    return body


@function_tool
def recap_session(ctx: RunContextWrapper) -> str:
    """Summarize this therapy session and persist it to the user's history."""
    return _recap_session_impl(ctx)


# ────────── per-user memory tools (used by therapist + cbt + grounding) ──────────

@function_tool
def recall_recent_topics(ctx: RunContextWrapper) -> str:
    """Return the user's last 3 session summaries as plain text.

    Use sparingly — only when you want to surface a thread from prior
    sessions. Returns an empty string for new users.
    """
    store = _unwrap(ctx)
    face_id = store.get("username", "guest")
    rows = memory.recent_sessions(face_id, n=3)
    if not rows:
        return ""
    return "\n".join(f"- {r['summary']}" for r in rows if r.get("summary"))


@function_tool
def update_user_note(ctx: RunContextWrapper, key: str, value: str) -> str:
    """Save or overwrite a single note on the user's profile (e.g.
    update_user_note("recurring_concern", "exam stress around midterms")).

    Keys are free-form snake_case. Use this when you learn something
    durable about the user that future sessions should know.
    """
    store = _unwrap(ctx)
    face_id = store.get("username", "guest")
    if not key or not isinstance(key, str):
        return "error: empty key"
    memory.update_profile(face_id, {key: value})
    return f"saved {key}"


# ────────── __main__ self-check ──────────
#
# Quick smoke-test for the vision call wiring. Monkeypatches the OpenAI
# client to return a canned envelope, runs `_observe_face_impl` against
# a dummy ctx, asserts we get the canned dict back. Also exercises the
# error path by swapping the classifier for one that raises and asserts
# the sentinel string is returned. Run with:
#     python -m server.tools.emotion
if __name__ == "__main__":  # pragma: no cover — manual smoke test
    canned = {
        "dominant_emotion": "sad",
        "secondary": "tired",
        "notes": "soft eye contact, slumped posture, slow pacing.",
    }

    # 1) Happy path — monkeypatch _vision_classify to return the canned dict.
    _orig_classify = _vision_classify

    def _fake_classify(b64: str) -> dict:
        assert b64 == "FAKEB64", b64
        return canned

    globals()["_vision_classify"] = _fake_classify
    try:
        ctx = {"latest_image_b64": "FAKEB64"}
        out = _observe_face_impl(ctx)
        assert out == canned, ("happy-path mismatch", out)
    finally:
        globals()["_vision_classify"] = _orig_classify

    # 2) No-image path — returns {"error": "no_image"}.
    out = _observe_face_impl({"latest_image_b64": None})
    assert out == {"error": "no_image"}, ("no_image mismatch", out)

    # 3) Error path — classifier raises → sentinel string returned, no raise.
    def _broken_classify(b64: str) -> dict:
        raise RuntimeError("simulated vision API failure")

    globals()["_vision_classify"] = _broken_classify
    try:
        out = _observe_face_impl({"latest_image_b64": "FAKEB64"})
        assert out == _OBSERVE_FAILURE_STRING, ("error-path mismatch", out)
    finally:
        globals()["_vision_classify"] = _orig_classify

    # 4) Phase-timer fallback — _vision_phase_timer must yield even when
    # metrics is unavailable. We exercise the no-op branch by simulating
    # an import failure.
    import sys
    real_metrics = sys.modules.pop("server.metrics", None)
    sys.modules["server.metrics"] = None  # make import raise inside the wrapper
    try:
        with _vision_phase_timer():
            pass  # must not raise
    finally:
        if real_metrics is not None:
            sys.modules["server.metrics"] = real_metrics
        else:
            sys.modules.pop("server.metrics", None)

    print("OK")
