# -*- coding: utf-8 -*-
"""ElevenLabs streaming TTS for the v2 fast-audio path.

Why this exists
---------------
OpenAI TTS (`tts-1`, the current default) takes 1-2 s to return the
first MP3 bytes. ElevenLabs Flash + WebSocket streaming returns first
audio in 100-500 ms in North America. For chat and therapy first-audio
latency, that's a 1-2 s improvement per turn.

Public API
----------
    synthesize(text, voice_id=None, output_format=None) -> bytes | None
        Drop-in replacement for ``openai_tts.synthesize``. Opens a
        single ElevenLabs WS connection, streams the input text,
        collects all audio chunks, returns the concatenated bytes.
        Returns None on failure so the caller can fall back to OpenAI.

    async synthesize_stream(text, voice_id=None, output_format=None)
        Async generator yielding raw audio chunks AS THEY ARRIVE.
        For per-sentence pipelining: WS handler can ship each chunk
        to the robot before the full sentence finishes synthesizing.

    is_available() -> bool
        True iff ELEVENLABS_API_KEY is set + a default voice ID
        resolves. Caller uses this to decide between ElevenLabs and
        OpenAI fallback on the synthesis path.

Output formats
--------------
ElevenLabs supports many; we default to ``mp3_44100_64`` (good quality,
small payload). For lowest latency, use ``pcm_16000`` — but NAO's
ALAudioPlayer wants a wrapped format (MP3 or WAV), so PCM would
require client-side WAV header injection. MP3 is the safe default.

Failure modes
-------------
- API key missing: ``synthesize`` returns None on first call.
- Network error / WS closes early: returns whatever audio arrived
  before the failure (may be None or partial).
- ElevenLabs rate-limited: returns None; caller falls back to OpenAI.
- Voice ID invalid: ElevenLabs returns a 4xx in the WS init event;
  we log and return None.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import AsyncIterator

from server import config

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Loudness
# ──────────────────────────────────────────────────────────────────────────
# openai_tts amplifies every clip by OPENAI_TTS_GAIN_DB before handing it to
# NAO, because raw TTS is far too quiet to carry across a room from the
# robot's small speakers. This module had no equivalent, so switching the
# voice over to ElevenLabs silently dropped that boost and NAO became hard
# to hear. Reuse openai_tts's ffmpeg helper rather than duplicating it --
# it already fails open, returning the audio unchanged on any ffmpeg error
# so a CLI hiccup can never make the robot mute.
#
# Defaults to the OpenAI gain so both providers sound the same out of the
# box; set ELEVENLABS_TTS_GAIN_DB to tune this voice independently.


def _gain_db() -> float:
    raw = os.environ.get("ELEVENLABS_TTS_GAIN_DB")
    if raw is None:
        raw = os.environ.get("OPENAI_TTS_GAIN_DB", "8")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _apply_gain(audio: bytes) -> bytes:
    gain = _gain_db()
    if not audio or gain <= 0:
        return audio
    try:
        from server.openai_tts import _amplify_mp3
    except Exception as e:  # noqa: BLE001
        _log.warning("[elevenlabs_tts] gain helper unavailable: %r", e)
        return audio
    try:
        return _amplify_mp3(audio, gain)
    except Exception as e:  # noqa: BLE001
        _log.warning("[elevenlabs_tts] gain failed, sending unamplified: %r", e)
        return audio


# ──────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────


def is_available() -> bool:
    """Return True iff EL is configured. Cheap; called per turn."""
    if not getattr(config, "ELEVENLABS_API_KEY", ""):
        return False
    if not _resolve_default_voice_id():
        return False
    return True


def _resolve_default_voice_id() -> str:
    """Look up the default voice ID from env (matches DEFAULT_PROFILE)."""
    profile = (getattr(config, "ELEVENLABS_DEFAULT_PROFILE", "girl") or "").lower()
    return _voice_id_for(profile) or ""


def _voice_id_for(profile: str) -> str | None:
    """Map a voice profile name to an env-supplied voice ID."""
    p = (profile or "").lower().strip()
    table = {
        "girl": getattr(config, "ELEVENLABS_VOICE_GIRL", ""),
        "woman": getattr(config, "ELEVENLABS_VOICE_GIRL", ""),
        "f": getattr(config, "ELEVENLABS_VOICE_GIRL", ""),
        "1": getattr(config, "ELEVENLABS_VOICE_GIRL", ""),
        "man": getattr(config, "ELEVENLABS_VOICE_MAN", ""),
        "guy": getattr(config, "ELEVENLABS_VOICE_MAN", ""),
        "m": getattr(config, "ELEVENLABS_VOICE_MAN", ""),
        "2": getattr(config, "ELEVENLABS_VOICE_MAN", ""),
        "neutral": getattr(config, "ELEVENLABS_VOICE_NEUTRAL", ""),
        "n": getattr(config, "ELEVENLABS_VOICE_NEUTRAL", ""),
        "3": getattr(config, "ELEVENLABS_VOICE_NEUTRAL", ""),
        # "my voice" — operator's personal cloned voice
        "my": getattr(config, "ELEVENLABS_VOICE_MY", ""),
        "mine": getattr(config, "ELEVENLABS_VOICE_MY", ""),
        "aayush": getattr(config, "ELEVENLABS_VOICE_MY", ""),
        "operator": getattr(config, "ELEVENLABS_VOICE_MY", ""),
        "4": getattr(config, "ELEVENLABS_VOICE_MY", ""),
    }
    return table.get(p) or None


# ──────────────────────────────────────────────────────────────────────────
# Sync entry point — drop-in for openai_tts.synthesize
# ──────────────────────────────────────────────────────────────────────────


def synthesize(text: str,
               voice_id: str | None = None,
               output_format: str | None = None) -> bytes | None:
    """Synthesize the entire sentence and return all audio bytes.

    Drop-in replacement for ``openai_tts.synthesize``. Internally opens
    a single WS, streams the input, collects audio chunks until the
    'isFinal' event lands, returns the concatenation.

    Returns None on any failure so the caller can fall back to OpenAI.
    """
    if not text or not str(text).strip():
        return None
    if not is_available():
        return None
    voice_id = voice_id or _resolve_default_voice_id()
    if not voice_id:
        return None

    output_format = output_format or getattr(
        config, "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_64",
    )

    async def _go() -> bytes | None:
        chunks: list[bytes] = []
        try:
            async for chunk in synthesize_stream(text, voice_id=voice_id,
                                                   output_format=output_format):
                chunks.append(chunk)
        except Exception as e:  # noqa: BLE001
            _log.warning("elevenlabs synthesize failed: %r", e)
            return None
        if not chunks:
            return None
        return _apply_gain(b"".join(chunks))

    # If we're already inside an event loop (the WS handler is async),
    # run on a worker loop via asyncio.run in a thread. This module is
    # primarily called from `await asyncio.to_thread(synthesize, text)`
    # in app_ws._emit_agent_turn, so we'll be in a thread here — fine.
    try:
        return asyncio.run(_go())
    except RuntimeError:
        # Already in a running loop somewhere (shouldn't happen in our
        # path, but defensive). Run on a fresh loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_go())
        finally:
            loop.close()


# ──────────────────────────────────────────────────────────────────────────
# Async streaming entry point — for true sentence-pipelined WS handler
# ──────────────────────────────────────────────────────────────────────────


async def synthesize_stream(text: str,
                              voice_id: str | None = None,
                              output_format: str | None = None,
                              ) -> AsyncIterator[bytes]:
    """Yield raw audio chunks as they arrive from ElevenLabs.

    This is the latency-optimal path: a 30-character sentence gets
    its first MP3 bytes in ~150-300 ms via Flash + WebSocket. Caller
    can forward each chunk to the robot WS as it arrives.
    """
    if not text or not str(text).strip():
        return
    if not is_available():
        return
    voice_id = voice_id or _resolve_default_voice_id()
    if not voice_id:
        return
    output_format = output_format or getattr(
        config, "ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_64",
    )

    api_key = config.ELEVENLABS_API_KEY
    model_id = getattr(config, "ELEVENLABS_MODEL", "eleven_flash_v2_5")

    # Per ElevenLabs WS docs: connect to
    #   wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input
    # Send a {text:" ", voice_settings:{...}} init frame, then text
    # chunks (each {text}), finally {text:""} to signal end.
    url = (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        f"?model_id={model_id}"
        f"&output_format={output_format}"
        f"&optimize_streaming_latency=4"
    )

    try:
        import websockets  # type: ignore
    except ImportError:
        _log.warning("`websockets` package missing; install for ElevenLabs streaming")
        return

    headers = [("xi-api-key", api_key)]
    try:
        # websockets >= 11 uses additional_headers; older uses extra_headers.
        try:
            ws = await websockets.connect(url, additional_headers=headers)
        except TypeError:
            ws = await websockets.connect(url, extra_headers=headers)
    except Exception as e:  # noqa: BLE001
        _log.warning("elevenlabs WS connect failed: %r", e)
        return

    try:
        # Init frame.
        await ws.send(json.dumps({
            "text": " ",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "use_speaker_boost": True,
            },
            "generation_config": {
                # Heuristic-driven sentence chunking on ElevenLabs side.
                # We already send sentence-sized payloads, but this
                # smooths boundaries.
                "chunk_length_schedule": [50, 80, 120, 160],
            },
            "xi_api_key": api_key,
        }))
        # Text payload.
        await ws.send(json.dumps({"text": text + " ", "try_trigger_generation": True}))
        # Close marker.
        await ws.send(json.dumps({"text": ""}))

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # Some EL flows send raw bytes; pass through.
                yield msg
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            audio_b64 = payload.get("audio")
            if audio_b64:
                try:
                    yield base64.b64decode(audio_b64)
                except Exception:
                    pass
            if payload.get("isFinal"):
                break
    except Exception as e:  # noqa: BLE001
        _log.warning("elevenlabs WS stream error: %r", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Smoke (developer use)
# ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":  # pragma: no cover
    import sys
    if not is_available():
        print("EL not configured (set ELEVENLABS_API_KEY + a voice ID)")
        sys.exit(1)
    import time
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello from NAO. Quick test."
    t0 = time.perf_counter()
    audio = synthesize(text)
    dt = (time.perf_counter() - t0) * 1000.0
    if audio:
        out = "/tmp/eleven_smoke.mp3"
        with open(out, "wb") as f:
            f.write(audio)
        print(f"OK: {len(audio)} bytes in {dt:.0f} ms → {out}")
    else:
        print(f"FAILED in {dt:.0f} ms")
