# -*- coding: utf-8 -*-
"""ElevenLabs Scribe v2 Realtime STT — feature-flagged candidate.

NOT wired into the production STT path. The current ``_transcribe`` in
``server.server`` (used by the legacy Flask) and the WS handler's
``legacy.transcribe`` keep their existing routing (Deepgram + OpenAI
Whisper fallback). This module provides:

  - ``transcribe_file(path)`` — sync, batches the whole file through
    the realtime WS and returns the final transcript. Drop-in shape
    for the existing Whisper/Deepgram bench harness.

  - ``async transcribe_stream(pcm_iter)`` — yields partial /
    committed transcripts as they arrive. For future integration
    once the A/B benchmark says to swap.

  - ``is_available()`` — env check.

A/B benchmark in ``sim/stt_ab.py`` calls ``transcribe_file`` on a fixed
clip set and compares to the existing providers.

Refs: ElevenLabs Speech-to-Text (Scribe v2 Realtime) docs.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import wave
from typing import AsyncIterator

from server import config

_log = logging.getLogger(__name__)


def is_available() -> bool:
    """True iff EL STT is enabled + key set."""
    if not getattr(config, "USE_ELEVENLABS_STT", False):
        return False
    if not getattr(config, "ELEVENLABS_API_KEY", ""):
        return False
    return True


def _read_wav_pcm(path: str) -> tuple[bytes, int]:
    """Return (raw PCM16 mono, sample_rate). Caller resamples if needed."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise ValueError(f"WAV must be 16-bit, got {sampwidth*8}-bit")
    if n_channels == 2:
        # Cheap stereo->mono mixdown for benching.
        import struct
        s = struct.unpack(f"<{len(frames)//2}h", frames)
        mono = [(s[i] + s[i + 1]) // 2 for i in range(0, len(s), 2)]
        frames = struct.pack(f"<{len(mono)}h", *mono)
    return frames, rate


_REST_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Scribe reports the language it heard plus a confidence, unlike OpenAI which
# just echoes back the `language` hint we sent. Above this confidence we trust
# it and drop the turn rather than answering in a language the speaker never
# used. Measured: English speech mis-heard as German came back p=0.97.
_NON_ENGLISH_MIN_PROB = 0.5


def transcribe_rest_detailed(path: str,
                             timeout_s: float = 20.0) -> tuple[str, bool]:
    """Like `transcribe_rest`, but also reports *why* the text is empty.

    Returns ``(text, non_english)``. The flag lets the caller tell a genuine
    failure (fall through to another provider) apart from a confidently
    non-English result (do not fall through -- another provider would just
    hand the same foreign text back, and Scribe's language confidence is
    better evidence than any downstream word-list heuristic).
    """
    if not is_available():
        return "", False
    model = getattr(config, "ELEVENLABS_STT_MODEL", "scribe_v1")
    if "realtime" in model:
        model = "scribe_v1"
    try:
        import requests  # local import: keeps module import cheap
        with open(path, "rb") as fh:
            resp = requests.post(
                _REST_URL,
                headers={"xi-api-key": config.ELEVENLABS_API_KEY},
                files={"file": (os.path.basename(path), fh, "audio/wav")},
                data={"model_id": model},
                timeout=timeout_s,
            )
        if resp.status_code != 200:
            _log.warning("EL STT rest %s: %s", resp.status_code,
                         resp.text[:200])
            return "", False
        body = resp.json()
    except Exception as e:  # noqa: BLE001
        _log.warning("EL STT rest failed: %r", e)
        return "", False

    text = (body.get("text") or "").strip()
    lang = (body.get("language_code") or "").lower()
    prob = body.get("language_probability")
    if text and lang and not lang.startswith("en"):
        if prob is None or prob >= _NON_ENGLISH_MIN_PROB:
            _log.warning("EL STT non-english (%s p=%s): %r", lang, prob,
                         text[:80])
            return "", True
    return text, False


def transcribe_rest(path: str, timeout_s: float = 20.0) -> str:
    """Transcribe a whole audio file via Scribe's batch REST endpoint.

    Separate from the realtime WS path above for a concrete reason: keys
    without the realtime entitlement get 403 on the WS handshake while this
    endpoint works fine (verified 2026-07-28 -- 0.55 s round trip, correct
    transcript). The WS is lower latency when available; this is what a
    standard key can actually use.

    Returns "" on any failure so the caller falls through to Whisper.
    """
    return transcribe_rest_detailed(path, timeout_s=timeout_s)[0]


def transcribe_file(path: str, timeout_s: float = 30.0) -> str:
    """Sync wrapper for the bench harness. Reads a WAV, streams its PCM
    into the EL Scribe Realtime WS, returns the final committed text.

    Returns "" on any failure (so the bench can attribute failures
    cleanly without raising).
    """
    if not is_available():
        return ""
    try:
        pcm, rate = _read_wav_pcm(path)
    except Exception as e:  # noqa: BLE001
        _log.warning("EL STT: bad WAV %s: %r", path, e)
        return ""

    async def _go() -> str:
        chunks_text: list[str] = []
        # Frame the PCM into ~100ms chunks so we look like a live stream.
        frame_ms = 100
        frame_bytes = int(rate * 2 * (frame_ms / 1000.0))

        async def _pcm_iter() -> AsyncIterator[bytes]:
            for i in range(0, len(pcm), frame_bytes):
                yield pcm[i:i + frame_bytes]
                await asyncio.sleep(frame_ms / 1000.0 * 0.05)  # mild pacing

        try:
            async for ev in transcribe_stream(_pcm_iter(), sample_rate=rate):
                if ev.get("is_final") and ev.get("text"):
                    chunks_text.append(ev["text"])
        except Exception as e:  # noqa: BLE001
            _log.warning("EL STT stream error: %r", e)
            return ""
        return " ".join(chunks_text).strip()

    try:
        return asyncio.run(asyncio.wait_for(_go(), timeout=timeout_s))
    except asyncio.TimeoutError:
        _log.warning("EL STT: timed out after %s s", timeout_s)
        return ""
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(_go(),
                                                              timeout=timeout_s))
        except Exception:
            return ""
        finally:
            loop.close()


# ──────────────────────────────────────────────────────────────────────────
# Per-session streaming STT — long-lived WS, fed PCM frames live
# ──────────────────────────────────────────────────────────────────────────


class StreamingSttSession:
    """Holds one open ElevenLabs Scribe Realtime WS for the lifetime of
    a NAO session. The WS handler feeds PCM frames as they arrive; a
    background receiver task collects partial + final transcripts.

    Public methods (all async unless noted):
        open(sample_rate=16000, language="en")    — connect once
        feed(pcm_bytes)                            — forward one chunk
        signal_eou()                               — send end_of_audio
        await_final(timeout_s=1.5) -> str | None   — wait for final
        reset()                                    — between turns
        close()                                    — at session_close

    State exposed (read-only):
        latest_partial    — most recent partial transcript text
        first_partial_ms  — ms from first PCM in to first partial event
        final_ms          — ms from signal_eou() to final-transcript event
        opened            — True once WS connect succeeded
        error             — last error string, or None
    """

    def __init__(self):
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._final_event = asyncio.Event()
        self._final_text: str = ""
        self.latest_partial: str = ""
        self.first_partial_ms: float | None = None
        self.final_ms: float | None = None
        self._t_first_audio: float | None = None
        self._t_eou: float | None = None
        self.opened: bool = False
        self.error: str | None = None
        self._lang: str = "en"
        self._rate: int = 16000

    async def open(self, sample_rate: int = 16000,
                    language: str = "en") -> bool:
        """Open the EL Scribe WS. Returns True on success."""
        if not is_available():
            self.error = "EL_STT_unavailable"
            return False
        api_key = config.ELEVENLABS_API_KEY
        model = getattr(config, "ELEVENLABS_STT_MODEL", "scribe_v2_realtime")
        self._rate = sample_rate
        self._lang = language

        url = (
            "wss://api.elevenlabs.io/v1/speech-to-text/stream"
            f"?model_id={model}"
            f"&language_code={language}"
            f"&sample_rate={sample_rate}"
            "&encoding=pcm_s16le"
        )
        headers = [("xi-api-key", api_key)]
        try:
            import websockets  # type: ignore
        except ImportError:
            self.error = "websockets_missing"
            return False
        try:
            try:
                self._ws = await websockets.connect(
                    url, additional_headers=headers, ping_interval=20,
                )
            except TypeError:
                self._ws = await websockets.connect(
                    url, extra_headers=headers, ping_interval=20,
                )
        except Exception as e:  # noqa: BLE001
            self.error = f"connect_failed: {e!r}"
            return False

        self.opened = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        return True

    async def _recv_loop(self):
        """Background receiver. Updates partial / final state."""
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                mtype = msg.get("type") or msg.get("event") or ""
                text = (
                    msg.get("text") or msg.get("transcript")
                    or msg.get("delta") or ""
                )
                if not text:
                    continue
                is_final = bool(
                    msg.get("is_final")
                    or "final" in mtype.lower()
                    or msg.get("type") == "transcript_final"
                )
                if self.first_partial_ms is None and self._t_first_audio:
                    self.first_partial_ms = round(
                        (time.perf_counter() - self._t_first_audio) * 1000.0, 1
                    )
                if is_final:
                    self._final_text = (self._final_text + " " + text).strip()
                    if self._t_eou:
                        self.final_ms = round(
                            (time.perf_counter() - self._t_eou) * 1000.0, 1
                        )
                    self._final_event.set()
                else:
                    self.latest_partial = text
        except Exception as e:  # noqa: BLE001
            self.error = f"recv_error: {e!r}"

    async def feed(self, pcm_bytes: bytes) -> None:
        """Forward one PCM chunk to the EL WS. Silent no-op if not open."""
        if not self.opened or self._ws is None:
            return
        if self._t_first_audio is None:
            self._t_first_audio = time.perf_counter()
        try:
            await self._ws.send(pcm_bytes)
        except Exception as e:  # noqa: BLE001
            self.error = f"send_error: {e!r}"

    async def signal_eou(self) -> None:
        """Send end_of_audio so EL emits a final event on the buffered audio."""
        if not self.opened or self._ws is None:
            return
        self._t_eou = time.perf_counter()
        try:
            await self._ws.send(json.dumps({"type": "end_of_audio"}))
        except Exception as e:  # noqa: BLE001
            self.error = f"eou_error: {e!r}"

    async def await_final(self, timeout_s: float = 1.5) -> str | None:
        """Wait up to ``timeout_s`` for a final transcript. Returns None
        on timeout so the caller can fall back to legacy STT."""
        try:
            await asyncio.wait_for(self._final_event.wait(),
                                     timeout=timeout_s)
            return self._final_text
        except asyncio.TimeoutError:
            return None

    async def reset(self) -> None:
        """Clear per-turn state for the next utterance."""
        self._final_event.clear()
        self._final_text = ""
        self.latest_partial = ""
        self.first_partial_ms = None
        self.final_ms = None
        self._t_first_audio = None
        self._t_eou = None

    async def close(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        if self._recv_task is not None:
            self._recv_task.cancel()


async def transcribe_stream(pcm_iter: AsyncIterator[bytes],
                              sample_rate: int = 16000,
                              language: str = "en"
                              ) -> AsyncIterator[dict]:
    """Yield {is_final: bool, text: str, t_ms: float} events as the
    Scribe v2 Realtime endpoint emits partial / committed transcripts.

    PCM input must be 16-bit signed LE mono. ``sample_rate`` declares
    the rate — EL accepts 8/16/22.05/24/44.1/48 kHz; we default to 16
    kHz to match the robot stream.
    """
    if not is_available():
        return

    api_key = config.ELEVENLABS_API_KEY
    model = getattr(config, "ELEVENLABS_STT_MODEL", "scribe_v2_realtime")

    # ElevenLabs Speech-to-Text WebSocket (Scribe Realtime). Endpoint
    # path/params follow their public docs; if these change, update
    # here.
    url = (
        "wss://api.elevenlabs.io/v1/speech-to-text/stream"
        f"?model_id={model}"
        f"&language_code={language}"
        f"&sample_rate={sample_rate}"
        "&encoding=pcm_s16le"
    )
    headers = [("xi-api-key", api_key)]

    try:
        import websockets  # type: ignore
    except ImportError:
        _log.warning("EL STT: `websockets` package missing")
        return

    try:
        try:
            ws = await websockets.connect(url, additional_headers=headers)
        except TypeError:
            ws = await websockets.connect(url, extra_headers=headers)
    except Exception as e:  # noqa: BLE001
        _log.warning("EL STT WS connect failed: %r", e)
        return

    t_open = time.perf_counter()

    async def _sender():
        try:
            async for chunk in pcm_iter:
                if chunk:
                    await ws.send(chunk)
            # End-of-audio sentinel per EL Realtime contract.
            await ws.send(json.dumps({"type": "end_of_audio"}))
        except Exception:
            pass

    sender_task = asyncio.create_task(_sender())
    try:
        async for raw in ws:
            t_ms = (time.perf_counter() - t_open) * 1000.0
            if isinstance(raw, bytes):
                continue  # EL sends JSON text; bytes shouldn't appear
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type") or msg.get("event") or ""
            text = (msg.get("text") or msg.get("transcript")
                    or msg.get("delta") or "")
            if not text:
                continue
            is_final = bool(
                msg.get("is_final")
                or "final" in mtype.lower()
                or msg.get("type") == "transcript_final"
            )
            yield {"is_final": is_final, "text": text, "t_ms": t_ms}
            if is_final and msg.get("type") == "session_end":
                break
    except Exception as e:  # noqa: BLE001
        _log.warning("EL STT recv error: %r", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        sender_task.cancel()
