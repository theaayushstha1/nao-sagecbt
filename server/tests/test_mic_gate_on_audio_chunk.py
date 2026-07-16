"""The mic gate must close on TTS audio arriving, not on a control frame.

Regression test for the self-echo loop: NAO asked "Sorry, what name should
I call you?", heard *itself* through its own mic, decided the echo wasn't a
name, and asked again — forever.

Root cause: ``gate(True)`` lived only in ``_on_tts_started``, and the server
emits ``tts_started`` from exactly one code path. Seven other reply paths —
including the onboarding name retry — send audio and then ``tts_ended``
with no ``tts_started`` at all, so the gate never closed and the recorder
ran straight through the robot's own speech. (``tts_started`` appeared 0
times in a robot log carrying 46 ``tts_ended``.)

The fix hangs mic safety on the audio itself, which every reply path must
send, rather than on a control frame seven of them forget. ``gate()`` is
idempotent, so closing per chunk is safe.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest


def _load_ws_client():
    """Import ``nao/ws_client.py`` with naoqi stubbed out."""
    for name in ("naoqi", "qi"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ALProxy = object
            mod.ALModule = object
            sys.modules[name] = mod
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("ws_client")


class _FakeStreamer:
    """Records gate() transitions the way audio_module does."""

    def __init__(self) -> None:
        self.gate_calls: list[bool] = []

    def gate(self, closed: bool) -> None:
        self.gate_calls.append(bool(closed))


def test_audio_chunk_closes_mic_gate_without_tts_started():
    ws_client = _load_ws_client()

    client = object.__new__(ws_client.NaoWsClient)
    streamer = _FakeStreamer()
    client.audio_streamer = streamer
    # Only the collaborators _close_mic_gate_for_tts actually touches.
    client._mic_timer_lock = threading.Lock()
    client._mic_open_timer = None

    ws_client.NaoWsClient._close_mic_gate_for_tts(client)

    assert streamer.gate_calls == [True], (
        "TTS audio arriving must close the mic gate; otherwise NAO records "
        "its own speech and transcribes itself."
    )


def test_speaking_gestures_disabled_by_env(monkeypatch):
    """SPEAKING_GESTURES=0 must stop the arm-waving-while-talking loop.

    The micro-gesture loop moves NAO's arms whenever TTS plays. The only
    existing suppression is a temporary window used while explicit actions
    run — there was no way to turn the behavior off outright.
    """
    ws_client = _load_ws_client()
    monkeypatch.setenv("SPEAKING_GESTURES", "0")

    client = object.__new__(ws_client.NaoWsClient)
    assert ws_client.NaoWsClient._speaking_gestures_disabled(client) is True


def test_speaking_gestures_enabled_by_default(monkeypatch):
    """Default stays on — this is a teammate's deliberate embodiment feature."""
    ws_client = _load_ws_client()
    monkeypatch.delenv("SPEAKING_GESTURES", raising=False)

    client = object.__new__(ws_client.NaoWsClient)
    assert ws_client.NaoWsClient._speaking_gestures_disabled(client) is False
