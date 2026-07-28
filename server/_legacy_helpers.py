"""Verbatim copies of private helpers from `server/server.py`.

`server/server.py` is FROZEN for the Phase 1 rework — `app_ws.py` (the new
FastAPI WebSocket transport) needs the same WAV validation, voice-gate,
hallucination filters, partial-buffer logic, and agent-runner glue that the
Flask app uses, but importing private (underscored) names directly couples us
to a frozen module. We copy them here so the new transport can evolve without
mutating the legacy module.

If `server/server.py` is ever unfrozen, the right move is to delete this file
and have both transports import from a shared helpers module — but that's a
post-Phase-9 cleanup.

Every symbol here mirrors the implementation in `server/server.py` at commit
f606534. Drift between them is a bug; if you change one, mirror the other.
"""
from __future__ import annotations

import asyncio
import os
import re
import wave

from openai import OpenAI

from server import config, memory, semantic_endpoint, session, vad_silero
from server.agents import pick_initial_agent
from server.topologies import run_topology

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ───────── active-session bookkeeping ─────────

# Active session id per user (face_id). Created lazily on the first turn after
# wake; closed on end_session=true or exit intent.
_ACTIVE_SESSIONS: dict[str, int] = {}


def ensure_active_session(username: str, hint: str | None) -> int:
    fid = (username or "").strip().lower()
    if not fid:
        return 0
    sid = _ACTIVE_SESSIONS.get(fid)
    if sid:
        return sid
    memory.ensure_user(fid, display_name=username)
    sid = memory.start_session(fid, mode=hint)
    _ACTIVE_SESSIONS[fid] = sid
    return sid


def close_active_session(username: str) -> None:
    """End the active session and kick off async summarization."""
    fid = (username or "").strip().lower()
    # Drop any buffered partial — the user is leaving, no point dragging
    # an unfinished sentence into the next session.
    _PARTIAL_BUFFER.pop(fid, None)
    sid = _ACTIVE_SESSIONS.pop(fid, None)
    if not sid:
        return
    lines: list[str] = []
    try:
        sess = session.get_or_create_session(fid)
        items = asyncio.run(sess.get_items())
        for it in items or []:
            role = it.get("role") if isinstance(it, dict) else None
            content = it.get("content") if isinstance(it, dict) else None
            if isinstance(content, list):
                txt = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict)
                    and p.get("type") in ("input_text", "output_text", "text")
                ).strip()
            elif isinstance(content, str):
                txt = content
            else:
                txt = ""
            if txt and role:
                lines.append("{0}: {1}".format(role, txt))
    except Exception:
        pass
    memory.end_session(sid, summary=None)
    if lines:
        memory.summarize_session_async(sid, lines)


# ───────── WAV validation ─────────

def validate_wav(path: str) -> bool:
    """Reject obviously empty clips. The hallucination filter and self-echo
    detector handle noise rejection downstream.
    """
    if os.path.getsize(path) < 1500:
        return False
    try:
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / float(w.getframerate() or 1)
            return dur >= 0.3
    except Exception:
        return False


# ───────── voice-activity gate ─────────

try:
    import webrtcvad  # type: ignore
    _VAD_AVAILABLE = True
except Exception:
    _VAD_AVAILABLE = False


def has_voice(path: str, aggressiveness: int = 2,
              voiced_ratio_min: float = 0.18) -> bool:
    """True if at least `voiced_ratio_min` of 30 ms frames are detected as speech.

    Layered: Silero VAD is the authoritative gate when it loads. webrtcvad is
    the fallback. Either rejecting drops the clip — but both are permissive on
    internal errors (return True) so we never block traffic on a VAD bug.
    """
    try:
        if not vad_silero.has_voice(path):
            return False
    except Exception:
        pass
    if not _VAD_AVAILABLE:
        return True
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if ch != 1 or sw != 2 or sr not in (8000, 16000, 32000, 48000):
            return True
        vad = webrtcvad.Vad(aggressiveness)
        frame_ms = 30
        bytes_per_frame = int(sr * frame_ms / 1000) * sw
        if bytes_per_frame == 0 or len(raw) < bytes_per_frame:
            return False
        n_total = 0
        n_voiced = 0
        for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
            frame = raw[i:i + bytes_per_frame]
            n_total += 1
            try:
                if vad.is_speech(frame, sr):
                    n_voiced += 1
            except Exception:
                continue
        if n_total == 0:
            return False
        return (n_voiced / n_total) >= voiced_ratio_min
    except Exception:
        return True


# ───────── hallucination / echo filters ─────────

_WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "please subscribe",
    "subscribe to my channel",
    "i'm here to listen and support you",
    "i'm here to listen and support you.",
    "i am here to listen and support you",
    "how can i help you",
    "how can i help you?",
    "how can i help you today",
    "how can i help you today?",
    "how are you doing today",
    "how are you doing today?",
    "how's your day going",
    "how's your day going so far",
    "how's your day going so far?",
    "hello",
    "hello.",
    "hi.",
    "hey.",
    "bye.",
    "thank you.",
    "thanks.",
    "you",
    "you.",
    "okay",
    "okay.",
    "ok.",
    ".",
    "",
    "hey there it's great to see you again",
    "hey there, it's great to see you again.",
    "hey there it's great to see you again.",
    "hey there, it's great to see you again",
    "hey there, it's good to see you again.",
    "hey there it's good to see you again.",
    "i remember you were talking about your studies last time.",
    "hey there it's great to see you again how's your day going so far",
    "hey there it's great to see you again hows your day going so far",
    "hey there it's good to see you again i remember you were talking about your studies last time",
    "good afternoon",
    "good afternoon.",
    "good morning",
    "good morning.",
    "good evening",
    "good evening.",
}

_ASR_NOISE_FRAGMENTS = {
    "e",
    "world map",
    "world right now",
    "right now",
    "yip",
}

_ROBOT_ECHO_PHRASES = (
    "hey there",
    "great to see you again",
    "good to see you again",
    "how's your day going",
    "hows your day going",
    "i remember you were talking",
    "i'm here to listen and support you",
    "i am here to listen and support you",
)


def _is_robot_named_echo(text: str) -> bool:
    t = (text or "").strip().lower()
    return "talking to a robot named nao" in t or "to a robot named nao" in t


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _clean_asr_text(text: str) -> str:
    return re.sub(r"\s+", " ", _norm(text)).strip()


def _looks_like_robot_greeting_echo(text: str) -> bool:
    """Reject our own stock greetings when the mic hears NAO's speaker."""
    t = _clean_asr_text(text)
    if not t:
        return False
    hits = sum(1 for phrase in _ROBOT_ECHO_PHRASES if phrase in t)
    return hits >= 2 or (
        t.startswith("hey there")
        and any(phrase in t for phrase in (
            "how's your day", "hows your day", "i remember you were talking",
        ))
    )


# Phrases in _WHISPER_HALLUCINATIONS that a real person genuinely says to a
# robot. They are only "hallucinations" when Whisper invents them from silence;
# with VAD-confirmed speech behind them they are ordinary conversation, and
# dropping them makes NAO stonewall anyone who opens with "Good morning".
# NAO's own scripted lines and Whisper's YouTube artifacts ("thanks for
# watching", "please subscribe", "you") deliberately stay out of this set, and
# _is_self_echo still guards against NAO hearing its own reply.
_HUMAN_GREETINGS = frozenset({
    "good morning", "good afternoon", "good evening",
    "hello", "hi", "hey", "bye",
    "ok", "okay", "thanks", "thank you",
    "how are you doing today",
})


def _looks_like_hallucination(text: str, had_speech: bool = False) -> bool:
    """True if Whisper output matches a known silence-hallucination pattern.

    `had_speech` means the caller has independent evidence that real speech
    was present (the WS path's `no_voice` / `silero_no_speech` gates ran and
    passed). That evidence disarms the short-utterance backstop below, which
    otherwise eats legitimate one-word turns -- most painfully a bare first
    name like "Mia", which is a single word of 3 characters.
    """
    t = (text or "").strip().lower()
    if not t:
        return True
    nt = _clean_asr_text(t)
    # Verified speech turns an "invented on silence" phrase back into a real
    # greeting. Punctuation is already stripped by _clean_asr_text, so this
    # catches "Good morning." and "Ok." alike.
    if had_speech and nt in _HUMAN_GREETINGS:
        return False
    if t in _WHISPER_HALLUCINATIONS:
        return True
    if nt in _WHISPER_HALLUCINATIONS or nt in _ASR_NOISE_FRAGMENTS:
        return True
    if _looks_like_robot_greeting_echo(t):
        return True
    # Length-only backstop for callers with no VAD evidence. When speech is
    # already confirmed upstream, a short word is a short word -- not noise.
    if not had_speech and len(t.split()) <= 1 and len(t) <= 4:
        return True
    return False


# Per-username last-reply cache for self-echo detection. When NAO's mic picks
# up the speaker output, Whisper transcribes our own reply back. We compare
# each new transcript against the last reply and reject if too similar.
LAST_REPLY: dict[str, str] = {}


def _is_self_echo(username: str, transcript: str) -> bool:
    """Reject transcripts that look like our own previous reply (mic feedback)."""
    if not transcript:
        return False
    last = LAST_REPLY.get(username, "")
    if not last:
        return False
    nt = _norm(transcript)
    nl = _norm(last)
    if not nt or not nl:
        return False
    if nt in nl or nl in nt:
        return True
    tt = set(nt.split())
    tl = set(nl.split())
    if not tt or not tl:
        return False
    inter = len(tt & tl)
    union = len(tt | tl)
    return (inter / union) >= 0.6


# Latin-script foreign-language detection.
#
# `transcribe()` pins language="en", but that is only a hint about the input:
# measured 2026-07-28, whisper-1, gpt-4o-transcribe and gpt-4o-mini-transcribe
# all return German text with it set, and whisper-1's verbose_json reports
# language='english' while doing so. So the API cannot flag this for us.
#
# Non-Latin scripts (CJK, Hangul, Cyrillic) already fall out because
# _clean_asr_text strips them to "". These markers cover Latin-script
# languages. Every entry must be a word that essentially never appears in
# spoken English -- no "die"/"man"/"so"/"to", which are German/Polish words
# but also common English ones.
_NON_ENGLISH_MARKERS = frozenset({
    # German
    "ich", "nicht", "bist", "geht", "dir", "schoen", "schön", "danke",
    "moechte", "möchte", "guten", "wiedersehen", "gern", "auch", "sehr",
    "wie geht", "bin hier", "es dir",
    # Polish
    "prosze", "proszę", "czesc", "cześć", "dziekuje", "dziękuję", "jestem",
    "milo", "miło", "slyszec", "słyszeć",
    # Spanish
    "hola", "gracias", "estas", "estás", "como estas", "senor", "señor",
    "buenos", "adios", "adiós",
    # French
    "bonjour", "merci", "comment", "ca va", "ça va", "je suis", "oui",
    "salut", "s'il",
    # Italian / Portuguese / Romanian
    "ciao", "grazie", "prego", "obrigado", "ola", "olá", "buna", "bună",
    "ce mai faci", "multumesc", "mulțumesc",
})

# Letters that are vanishingly rare in English but routine elsewhere. A
# transcript carrying these is almost certainly not English. Deliberately
# excludes bare accented vowels (é, à, ï) so "cafe"/"naive"/"resume" style
# loanwords and names are unaffected.
_FOREIGN_CHARS = frozenset("ßäöüąćęłńśźżğıșțñ¿¡")


def _looks_non_english(text: str) -> bool:
    """True when a transcript is almost certainly not English.

    Conservative by design: a false positive silently drops something the
    user actually said, which is worse than letting one odd phrase through.
    """
    raw = (text or "").strip().lower()
    if not raw:
        return False
    if any(ch in _FOREIGN_CHARS for ch in raw):
        return True
    nt = _clean_asr_text(raw)
    if not nt:
        return False
    if any(phrase in nt for phrase in _NON_ENGLISH_MARKERS if " " in phrase):
        return True
    words = set(nt.split())
    return bool(words & _NON_ENGLISH_MARKERS)


def transcript_reject_reason(username: str, transcript: str,
                             asking_name: bool = False,
                             had_speech: bool = False) -> str | None:
    if asking_name:
        t = (transcript or "").strip()
        if not t:
            return "hallucination_or_noise"
        return None
    if _looks_like_hallucination(transcript, had_speech=had_speech):
        return "hallucination_or_noise"
    if _looks_non_english(transcript):
        return "non_english"
    if _is_self_echo(username, transcript):
        return "self_echo"
    if _is_robot_named_echo(transcript):
        return "robot_named_echo"
    return None


# ───────── partial-transcript buffer ─────────

_PARTIAL_BUFFER: dict[str, str] = {}
_PARTIAL_WAIT_COUNT: dict[str, int] = {}
_PARTIAL_MAX_CHARS = 500
_PARTIAL_MAX_WAIT = 4


def consume_partial(username: str, current: str) -> str:
    """Prepend any buffered partial to the current transcript and clear it."""
    _PARTIAL_WAIT_COUNT.pop(username, None)
    head = _PARTIAL_BUFFER.pop(username, "")
    if not head:
        return current
    joined = (head + " " + current).strip()
    return joined[-_PARTIAL_MAX_CHARS:]


def stash_partial(username: str, text: str) -> None:
    """Stash a partial transcript so the next turn can resume it."""
    if not text or not text.strip():
        return
    _PARTIAL_WAIT_COUNT[username] = _PARTIAL_WAIT_COUNT.get(username, 0) + 1
    _PARTIAL_BUFFER[username] = text.strip()[-_PARTIAL_MAX_CHARS:]


def partial_wait_limit_hit(username: str) -> bool:
    return _PARTIAL_WAIT_COUNT.get(username, 0) >= _PARTIAL_MAX_WAIT


# ───────── ASR ─────────

def transcribe(path: str) -> str:
    if config.USE_DEEPGRAM:
        from server import deepgram_asr
        text = deepgram_asr.transcribe(path)
        if text:
            return text
        print("[transcribe] deepgram returned empty; falling back to whisper",
              flush=True)
    if getattr(config, "USE_ELEVENLABS_STT", False):
        # Batch REST, not the realtime WS -- keys without the realtime
        # entitlement 403 on the handshake.
        from server import elevenlabs_stt
        text, non_english = elevenlabs_stt.transcribe_rest_detailed(path)
        if text:
            return text
        if non_english:
            # Scribe is confident this was not English. Whisper would happily
            # hand the same foreign text back (it ignores our language="en"
            # hint), so stop here and let the empty transcript be rejected.
            return ""
        print("[transcribe] elevenlabs returned empty; falling back to whisper",
              flush=True)
    with open(path, "rb") as f:
        resp = _client.audio.transcriptions.create(
            model=config.WHISPER_MODEL, file=f,
            language="en", temperature=0,
        )
    return resp.text


# ───────── agent runner ─────────

def _build_user_message(transcript: str, image_b64: str | None,
                          vision_observation: dict | None = None,
                          identity: dict | None = None):
    """Build a Runner input.

    When ``vision_observation`` is supplied (Phase 11 Option B path), we
    prepend a short developer note to the transcript so the therapist
    agent reads it as context. The agent prompt enforces the rule that
    visual observations may only be referenced when ``vision_status ==
    "success"`` -- this builder is the *only* place vision data enters
    the conversation under Option B, so the therapist can't hallucinate.

    When ``identity`` is supplied (from the robot's onboarding face
    scan), we ALSO prepend a [USER ...] block so the agent knows who's
    talking and whether to greet them by name on the first turn.

    For text-only, return a string. For multimodal, return the
    Responses-API shape: a single user message with typed content items
    (input_text / input_image).
    """
    identity_prefix = ""
    if isinstance(identity, dict):
        name = (identity.get("name") or "").strip()
        recognized = bool(identity.get("recognized"))
        face_visible = bool(identity.get("face_visible"))
        first_turn = bool(identity.get("first_turn", False))
        if recognized and name:
            identity_prefix = (
                "[USER name=" + name
                + " returning=true first_turn="
                + ("true" if first_turn else "false") + "]\n"
                "(NAO has previously learned this person's face. "
                + ("This is the first turn after engagement — open with a "
                   "warm \"Welcome back, " + name + "\" or similar before "
                   "answering. " if first_turn else "")
                + "Use their name naturally when it fits.)\n"
            )
        elif face_visible and not name and first_turn:
            identity_prefix = (
                "[USER name=unknown returning=false first_turn=true]\n"
                "(A face is visible but NAO has never learned it. On THIS "
                "first turn, briefly introduce yourself ('Hi, I'm NAO') and "
                "ask their name. If they answer with a name, call the "
                "`learn_face(name=...)` tool so future sessions recognize "
                "them. Don't ramble — one short sentence.)\n"
            )

    vision_prefix = ""
    if isinstance(vision_observation, dict):
        status = vision_observation.get("vision_status") or "skipped"
        summary = (vision_observation.get("vision_summary") or "").strip()
        cached = bool(vision_observation.get("vision_cached"))
        age_ms = vision_observation.get("vision_age_ms")
        if status == "success" and summary:
            cache_note = ""
            if cached:
                age_s = (age_ms or 0) / 1000.0
                cache_note = (
                    " vision_cached=true vision_age_s=" + f"{age_s:.0f}"
                )
            vision_prefix = (
                "[NAO_VISION vision_status=success" + cache_note
                + " vision_summary=\""
                + summary.replace("\"", "'") + "\"]\n"
                "(Server-side vision observation"
                + (
                    " from " + f"{(age_ms or 0)/1000.0:.0f}"
                    + " seconds ago — still safe to reference; "
                      "if user has visibly shifted, lead with current emotion "
                      "instead of contradicted details."
                    if cached else ""
                )
                + ". You MAY reference these details in your reply per "
                  "the prompt's Rule 0.)\n"
            )
        else:
            vision_prefix = (
                "[NAO_VISION vision_status=" + status + " vision_summary=\"\"]\n"
                "(No usable vision data this turn. Per the prompt's safety "
                "rule, do NOT reference any visual details about the user "
                "or their environment in your reply.)\n"
            )

    text = identity_prefix + vision_prefix + (transcript or "")
    if not image_b64:
        return text
    return [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image",
             "image_url": "data:image/jpeg;base64,{0}".format(image_b64)},
        ],
    }]


async def run_agent_streamed(
    username: str, hint: str | None, transcript: str,
    image_b64: str | None, vision_observation: dict | None = None,
    identity: dict | None = None,
    *,
    model_override: str | None = None,
    first_token_timeout_s: float | None = None,
):
    """Stream a passthrough agent turn token-by-token.

    Yields one of three event shapes (dict):
        {"type": "delta",    "text": "<chunk>"}
        {"type": "action",   "name": "<tool>", "args": {...}}
        {"type": "agent",    "active_agent": "<name>"}      # on handoff
        {"type": "done",     "reply": "<full>", "active_agent": "<name>",
         "actions": [...], "suppress_image": bool}
        {"type": "error",    "error": "<repr>"}

    Falls back to the sync run_agent if the topology isn't
    'passthrough' (debate / supervisor_veto / shared_pool require
    multiple full Runner.run calls — they don't fit a single token
    stream). The caller can detect the fallback by getting a `done`
    event with no preceding `delta` events and adapt.

    Phase 11.5 (true streaming TTS): the WS handler pipes ``delta``
    events through the sentence chunker → parallel TTS synth, so the
    robot can start speaking before the agent has finished the reply.
    Crisis check has already gated this (it runs in _process_turn
    BEFORE this generator is invoked), so streaming is safe.
    """
    import asyncio
    import os
    from agents import Runner
    try:
        from openai.types.responses import ResponseTextDeltaEvent
    except Exception:  # pragma: no cover
        ResponseTextDeltaEvent = None  # type: ignore[assignment]

    topology = (os.environ.get("SAGE_TOPOLOGY") or "passthrough").lower()
    if topology != "passthrough":
        # Non-passthrough topologies need multiple Runner.run calls
        # (debate, supervisor_veto, shared_pool). They can't be streamed
        # token-by-token cleanly. Fall back to the sync path and emit
        # one synthetic `done` event.
        reply, active, actions, suppress = run_agent(
            username, hint, transcript, image_b64, vision_observation,
            identity=identity,
        )
        yield {"type": "done", "reply": reply, "active_agent": active,
                "actions": actions, "suppress_image": suppress}
        return

    agent = pick_initial_agent(username, hint, transcript)
    # Phase 11.12 — model override (used by the chat-fallback wrapper to
    # rebuild the same agent against gpt-4o-mini after a nano timeout).
    if model_override:
        from agents import Agent as _Agent
        agent = _Agent(
            name=getattr(agent, "name", "agent"),
            instructions=getattr(agent, "instructions", ""),
            model=model_override,
            model_settings=getattr(agent, "model_settings", None),
            tools=list(getattr(agent, "tools", []) or []),
        )
    sess = session.get_or_create_session(username)
    ctx = {
        "username": username,
        "actions_queue": [],
        "emotion_log": [],
        "latest_image_b64": image_b64,
        "suppress_image": False,
        "vision_observation": vision_observation,
    }
    message = _build_user_message(transcript, image_b64, vision_observation,
                                   identity=identity)

    active_agent = getattr(agent, "name", "agent")
    reply_parts: list[str] = []
    first_delta_seen = False
    timeout_task: asyncio.Task | None = None
    if first_token_timeout_s and first_token_timeout_s > 0:
        timeout_task = asyncio.create_task(
            asyncio.sleep(first_token_timeout_s),
        )

    try:
        run = Runner.run_streamed(agent, message, context=ctx, session=sess)
        async for ev in run.stream_events():
            # Phase 11.12 — first-token deadline check between events.
            # Runner emits frequent low-level events even before the
            # model's first text delta, so we get plenty of chances to
            # notice the timer fired without polling.
            if (timeout_task is not None
                    and timeout_task.done()
                    and not first_delta_seen):
                yield {"type": "timeout",
                        "reason": "first_token_timeout",
                        "limit_s": first_token_timeout_s}
                try:
                    run.cancel()
                except Exception:
                    pass
                return
            if ev.type == "raw_response_event":
                data = ev.data
                if ResponseTextDeltaEvent is not None and \
                   isinstance(data, ResponseTextDeltaEvent):
                    delta = data.delta or ""
                    if delta:
                        if not first_delta_seen:
                            first_delta_seen = True
                            if timeout_task is not None:
                                timeout_task.cancel()
                                timeout_task = None
                        reply_parts.append(delta)
                        yield {"type": "delta", "text": delta}
            elif ev.type == "agent_updated_stream_event":
                new_agent = getattr(ev, "new_agent", None)
                if new_agent is not None:
                    active_agent = getattr(new_agent, "name", active_agent)
                    yield {"type": "agent", "active_agent": active_agent}
        try:
            final_text = run.final_output_as(str)
        except Exception:
            final_text = "".join(reply_parts)
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "error": repr(e)}
        return

    if timeout_task is not None:
        timeout_task.cancel()
    yield {
        "type": "done",
        "reply": final_text,
        "active_agent": active_agent,
        "actions": list(ctx["actions_queue"]),
        "suppress_image": bool(ctx["suppress_image"]),
    }


# Phase 11.12 — pure-chat outlier safety valve. Wraps run_agent_streamed
# with a first-token deadline. If gpt-4.1-nano stalls past the limit,
# yield a synthetic "filler" event ("One sec.") so the WS handler can
# emit immediate audio, then start a fallback stream against gpt-4o-mini
# and pipe its events through transparently. Never used in normal turns
# — only kicks in on the rare ~2-3% of nano calls that exceed budget.
async def run_pure_chat_with_fallback(
    username: str, hint: str | None, transcript: str,
    image_b64: str | None, vision_observation: dict | None = None,
    identity: dict | None = None,
    *,
    first_token_timeout_s: float = 3.5,
    fallback_model: str = "gpt-4o-mini",
    filler_text: str = "One sec.",
):
    """Pure-chat lane wrapper with first-token timeout + model fallback.

    Yields:
      Normal nano events ({type:"delta"|"agent"|"done"}), OR
      {"type":"filler","text":"One sec."}      — on nano timeout
      followed by mini events transparently.

    The caller treats `filler` like a `delta` for TTS purposes — it's
    a complete sentence ready to synthesize and play immediately. The
    stream then continues with the fallback model's tokens.
    """
    timed_out = False
    async for ev in run_agent_streamed(
            username, hint, transcript, image_b64, vision_observation,
            identity,
            first_token_timeout_s=first_token_timeout_s):
        if ev.get("type") == "timeout":
            timed_out = True
            break
        yield ev
    if not timed_out:
        return

    # Filler audio — single short sentence, gets TTS'd immediately by
    # the WS handler's existing chunker pipeline.
    yield {"type": "filler", "text": filler_text}

    # Fallback stream on the mini model. No timeout this time —
    # mini's tail latency is bounded around 3.8 s per the bench.
    async for ev in run_agent_streamed(
            username, hint, transcript, image_b64, vision_observation,
            identity,
            model_override=fallback_model):
        yield ev


def run_agent(username: str, hint: str | None, transcript: str,
              image_b64: str | None,
              vision_observation: dict | None = None,
              identity: dict | None = None,
              ) -> tuple[str, str, list[dict], bool]:
    """Run the agent graph synchronously and return
    (reply, active_agent_name, actions_queue, suppress_image).

    Phase 11 / Option B: callers may pass ``vision_observation`` (the
    structured envelope from ``server.tools.emotion.observe_face_for_turn``)
    to inject the result as conversation context. When supplied, the
    therapist agent reads it as part of the user message; it never has
    to call ``observe_face`` itself, so the "skipped tool but said 'I can
    see...'" hallucination path is closed.
    """
    agent = pick_initial_agent(username, hint, transcript)
    sess = session.get_or_create_session(username)
    ctx = {
        "username": username,
        "actions_queue": [],
        "emotion_log": [],
        "latest_image_b64": image_b64,
        "suppress_image": False,
        # Stash for the topology trace + therapist prompt-time inspection.
        "vision_observation": vision_observation,
    }
    message = _build_user_message(transcript, image_b64, vision_observation,
                                   identity=identity)
    reply, active, _verdict, _metadata = run_topology(
        agent, message, context=ctx, session=sess,
    )
    return (
        reply,
        active,
        list(ctx["actions_queue"]),
        bool(ctx["suppress_image"]),
    )


__all__ = [
    "ensure_active_session",
    "close_active_session",
    "validate_wav",
    "has_voice",
    "transcript_reject_reason",
    "consume_partial",
    "stash_partial",
    "partial_wait_limit_hit",
    "transcribe",
    "run_agent",
    "LAST_REPLY",
]
