"""Environment-driven configuration for the server.

Centralizes every env var the server reads. Phase 1 of the v2 rework
(see docs/PRD_v2.md) layers a FastAPI + WebSocket transport on top of
the existing Flask app behind the ``USE_WS`` feature flag — the new
``WS_*``/``LOG_*``/``METRICS_*``/``TTS_CHUNK_*``/``MIC_GATE_*``/
``WS_RECONNECT_*`` exports below are owned by this module per the task
map in docs/PHASE_1_TASK_MAP.md. All legacy exports are preserved so
the Flask path keeps booting unchanged when ``USE_WS=0`` (default).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gpt-4.1-nano")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1-nano")
# The embodied lane carries the 18 NAO action tools, and tool-calling cost
# differs sharply by provider -- measured 2026-07-28, the same tool-heavy turn
# took 7.8s on gpt-4.1-mini and 19.6s on claude-sonnet-5. It gets its own knob
# so the conversational lane can move to a different provider without dragging
# the action lane's latency with it. Defaults to CHAT_MODEL when unset.
CHAT_EMBODIED_MODEL = os.environ.get("CHAT_EMBODIED_MODEL", CHAT_MODEL)
CHATBOT_MODEL = os.environ.get("CHATBOT_MODEL", "gpt-4.1-mini")
THERAPIST_MODEL = os.environ.get("THERAPIST_MODEL", "gpt-4.1-mini")
SKILLS_MODEL = os.environ.get("SKILLS_MODEL", "gpt-4.1-nano")
CRISIS_MODEL = os.environ.get("CRISIS_MODEL", "gpt-4.1")
# NOTE: there is deliberately no CBT_MODEL / GROUNDING_MODEL. Both coaches are
# therapist sub-agents and read THERAPIST_MODEL (cbt_coach.py,
# grounding_coach.py). Separate vars existed here but nothing consumed them, so
# setting them looked like it worked and silently did nothing. If those lanes
# ever need to diverge from the therapist, add the var *and* the read together.

# Per-agent max output tokens. Nano agents are capped tightly to keep replies
# snappy (under 2 short sentences). Mini agents get a bit more headroom for
# clinical reasoning / RAG synthesis but still stay terse.
NANO_MAX_TOKENS = int(os.environ.get("NANO_MAX_TOKENS", "200"))
# Phase 11.7: dedicated cap for the fast-chat lane. Casual replies must
# be 1–2 spoken sentences (~25 words), so 80 tokens is plenty and forces
# the model to stop early on long-tail tangents. Helps the under-3-second
# target for chat mode.
FAST_CHAT_MAX_TOKENS = int(os.environ.get("FAST_CHAT_MAX_TOKENS", "80"))
MINI_MAX_TOKENS = int(os.environ.get("MINI_MAX_TOKENS", "400"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "gpt-4o-mini-transcribe")

# ────────── ElevenLabs streaming TTS (Phase 11.8 — fast first audio) ──────────
# When ELEVENLABS_API_KEY is set, the WS handler routes TTS through
# ElevenLabs Flash + WebSocket (~150-300ms first audio, vs ~1-2s for
# OpenAI tts-1). Falls back to OpenAI TTS when key/voice missing or on
# any failure.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_OUTPUT_FORMAT = os.environ.get(
    "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_64",
)
# Voice slots. User pastes voice IDs from elevenlabs.io into env.
# "MY" is the operator's personal cloned voice — triggered by saying
# "switch to my voice".
ELEVENLABS_VOICE_GIRL    = os.environ.get("ELEVENLABS_VOICE_GIRL", "")
ELEVENLABS_VOICE_MAN     = os.environ.get("ELEVENLABS_VOICE_MAN", "")
ELEVENLABS_VOICE_NEUTRAL = os.environ.get("ELEVENLABS_VOICE_NEUTRAL", "")
ELEVENLABS_VOICE_MY      = os.environ.get("ELEVENLABS_VOICE_MY", "")
ELEVENLABS_DEFAULT_PROFILE = os.environ.get(
    "ELEVENLABS_DEFAULT_PROFILE", "girl",
)
# Force the OpenAI TTS path for testing or hard-disable EL temporarily.
USE_ELEVENLABS_TTS = os.environ.get("USE_ELEVENLABS_TTS", "1") == "1"
# Phase 11.9: candidate ElevenLabs Scribe v2 Realtime STT. Off by
# default — gated behind USE_ELEVENLABS_STT for the A/B benchmark
# (sim/stt_ab.py). Production STT path stays on Deepgram + OpenAI
# Whisper until the bench data says otherwise.
USE_ELEVENLABS_STT = os.environ.get("USE_ELEVENLABS_STT", "0") == "1"
# The Scribe *realtime* WS needs an entitlement a standard key lacks -- it
# 403s on the handshake, once per session, before falling back. The batch
# REST path in elevenlabs_stt.transcribe_rest works on any key, so leave the
# WS off unless the plan actually includes it.
USE_ELEVENLABS_STT_REALTIME = (
    os.environ.get("USE_ELEVENLABS_STT_REALTIME", "0") == "1"
)
ELEVENLABS_STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2_realtime")

# Deepgram Nova-2 streaming/prerecorded ASR. When USE_DEEPGRAM is true and the
# API key is present, /turn and /stream_turn use Deepgram instead of Whisper.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-2")
DEEPGRAM_LANGUAGE = os.environ.get("DEEPGRAM_LANGUAGE", "en-US")
USE_DEEPGRAM = bool(DEEPGRAM_API_KEY) and os.environ.get("USE_DEEPGRAM", "1") == "1"
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime")
REALTIME_VAD_THRESHOLD = float(os.environ.get("REALTIME_VAD_THRESHOLD", "0.30"))
REALTIME_VAD_PREFIX_MS = int(os.environ.get("REALTIME_VAD_PREFIX_MS", "500"))
REALTIME_VAD_SILENCE_MS = int(os.environ.get("REALTIME_VAD_SILENCE_MS", "450"))

# OpenAI TTS. /stream_turn synthesizes each sentence and emits an "audio"
# SSE event (base64 mp3) so NAO plays it via ALAudioPlayer.
# Female voices: nova (warm), shimmer (soft), coral, sage.
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "nova")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
USE_OPENAI_TTS = os.environ.get("USE_OPENAI_TTS", "1") == "1"

# Shared secret required on every HTTP/WS request (X-NAO-Secret header, or
# {"secret": "..."} in the realtime WebSocket handshake). Empty string =
# OPEN mode for local dev — server logs a warning at startup.
NAO_SHARED_SECRET = os.environ.get("NAO_SHARED_SECRET", "")

# Vertex AI Search (Morgan State CS knowledge base)
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "csnavigator-vertex-ai")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us")
VERTEX_DATASTORE_ID = os.environ.get("VERTEX_DATASTORE_ID", "csnavigator-kb-v7")

# Networking
NAO_IP = os.environ.get("NAO_IP", "172.20.95.111")
NAO_PORT = int(os.environ.get("NAO_PORT", "9559"))
SERVER_IP = os.environ.get("SERVER_IP", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))

# Off by default because /greet speaks proactively when a person is detected.
PROACTIVE_GREET_ENABLED = os.environ.get("PROACTIVE_GREET_ENABLED", "0") == "1"

# Persistence
SESSION_DB = os.environ.get("SESSION_DB", "server/nao.db")

# Tracing (SDK reads OPENAI_AGENTS_DISABLE_TRACING; we keep it on by default)
OPENAI_AGENTS_TRACE = os.environ.get("OPENAI_AGENTS_TRACE", "1") == "1"

# ───────── SAGE-CBT research layer ─────────
# Topology dispatcher. "passthrough" = existing router behavior, unchanged.
# Other values activate the SAGE-CBT topology layer for the therapist subgraph.
SAGE_TOPOLOGY = os.environ.get("SAGE_TOPOLOGY", "passthrough")  # passthrough|supervisor_veto|debate|shared_pool

# SafetyAgent provider. Only used when SAGE_TOPOLOGY != "passthrough".
SAGE_SAFETY_PROVIDER = os.environ.get("SAGE_SAFETY_PROVIDER", "openai")  # openai|claude

# Models for the SafetyAgent (each provider picks the right key).
SAFETY_MODEL_OPENAI = os.environ.get("SAFETY_MODEL_OPENAI", "gpt-4o")
SAFETY_MODEL_CLAUDE = os.environ.get("SAFETY_MODEL_CLAUDE", "claude-opus-4-7")

# Anthropic key — optional. If SAGE_SAFETY_PROVIDER=claude, this MUST be set.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Per-session log directory for topology traces (written as JSONL).
SAGE_LOG_DIR = os.environ.get("SAGE_LOG_DIR", "logs")

# ───────── Phase 1: FastAPI + WebSocket transport (PRD v2) ─────────
# Feature flag. When 1, run.sh boots `uvicorn server.app_ws:app` instead of
# the legacy Flask `python -m server.server`. The Flask path stays as-is for
# the deployment week (backwards compat) — flip this only when the FastAPI
# app is actually merged.
USE_WS = os.environ.get("USE_WS", "0") == "1"

# uvicorn bind for the WS server.
WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("WS_PORT", "5050"))

# structlog wiring used by `server/logging_setup.py` (Phase 1 observability).
# `json` is the prod default; `console` switches to a human-readable renderer.
LOG_FORMAT = os.environ.get("LOG_FORMAT", "json")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Toggle the Prometheus exporter exposed at /metrics. Lets dev runs skip the
# metrics middleware entirely without removing the routes.
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "1") == "1"

# Sentence chunker tuning for streaming TTS (server/streaming.py). Emit a
# chunk once we've buffered at least TTS_CHUNK_MIN_CHARS, or after the model
# has paused for TTS_CHUNK_TIMEOUT_MS without a sentence boundary so we don't
# wait forever on a slow tail.
TTS_CHUNK_MIN_CHARS = int(os.environ.get("TTS_CHUNK_MIN_CHARS", "24"))
TTS_CHUNK_TIMEOUT_MS = int(os.environ.get("TTS_CHUNK_TIMEOUT_MS", "250"))

# Robot mic gate timing. After the last TTS audio chunk completes, wait this
# long before resubscribing the mic — catches in-flight buffers + reverb.
MIC_GATE_GRACE_MS = int(os.environ.get("MIC_GATE_GRACE_MS", "200"))

# Robot WS reconnect backoff schedule, in milliseconds. Comma-separated env
# string parsed once at import. The robot walks this list on successive
# reconnect attempts and stays at the last value once exhausted.
WS_RECONNECT_BACKOFF_MS = [
    int(x) for x in os.environ.get(
        "WS_RECONNECT_BACKOFF_MS", "300,600,1200,2400"
    ).split(",")
    if x.strip()
]

# ───────── Phase 5 (PRD v2): CS Navigator integration ─────────
# Replaces the in-tree Pinecone/Vertex RAG with a thin HTTP proxy that calls
# the operator's deployed Cloud Run FastAPI ("CS Navigator") for any Morgan
# State CS knowledge query. See docs/PHASE_5_TASK_MAP.md for the contract.
#
# CS_NAVIGATOR_URL — Cloud Run base URL, no trailing slash. Empty string
#                    means the chatbot agent will short-circuit and apologize.
# CS_NAVIGATOR_TOKEN — optional bearer token. When empty the tool POSTs to
#                    `/chat/guest`; when set it POSTs to `/chat/stream` with
#                    `Authorization: Bearer <TOKEN>`.
# CS_NAVIGATOR_TIMEOUT_S — request timeout (seconds). Float so tests can
#                    bypass with sub-second values; production stays at 30 s
#                    to absorb cold starts on the Cloud Run side.
CS_NAVIGATOR_URL = os.environ.get("CS_NAVIGATOR_URL", "")
CS_NAVIGATOR_TOKEN = os.environ.get("CS_NAVIGATOR_TOKEN", "")
CS_NAVIGATOR_TIMEOUT_S = float(os.environ.get("CS_NAVIGATOR_TIMEOUT_S", "30"))

# ───────── Phase 6 (PRD v2): Therapist Vision-On ─────────
# Vision model used by `server/tools/emotion.py:observe_face` to read the
# user's face from the per-turn JPEG. Default `gpt-4o`; can be flipped to
# `gpt-5` (or whatever's GA at deploy time) without a code change.
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o")

# Default camera-consent for new users. Phase 6 flips this to ON so the
# therapist can call `observe_face` from turn 1; users opt OUT via the
# stop-watching pattern triggers or the explicit `set_camera_consent(false)`
# tool. See docs/PHASE_6_TASK_MAP.md for the full contract.
CAMERA_DEFAULT_ON = os.environ.get("CAMERA_DEFAULT_ON", "1") == "1"

# First-turn audible heads-up emitted by the WS server when a brand-new
# session opens with `camera_consent=1`. Plain text — TTS-only, no SSML.
CAMERA_ANNOUNCE_TEXT = os.environ.get(
    "CAMERA_ANNOUNCE_TEXT",
    "Heads up — my camera is on for this conversation. "
    "Say 'stop watching me' anytime.",
)
