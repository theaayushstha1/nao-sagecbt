"""Wake must report the *recognised name*, not NAOqi's throwaway face id.

ALFaceDetection's extra_info carries two different things:

    extra_info[0] -> internal face id  ("12", "18") — transient, renumbered
                     on every re-detection, meaningless across sessions
    extra_info[2] -> recognised name   ("Aayush")   — the label learnFace()
                     stored, stable forever

The server's `users` table is keyed on the *name* (rows: 'aayush', 'guest'),
and `_lookup_returning_user` does `SELECT display_name FROM users WHERE
face_id = ?`. wake_state sent extra_info[0], so the query asked for "18" in a
table of names, never matched, and every session looked like a brand-new
user — the same person was face_id=12 in one session and face_id=18 in the
next.

Send the name when NAO recognises the face; fall back to the numeric id when
it doesn't, so an unknown face still has *something* stable within a session.
"""
from __future__ import annotations

import sys
import types

import pytest


def _load_wake_state():
    for name in ("naoqi", "qi"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.ALProxy = object
            mod.ALModule = object
            sys.modules[name] = mod
    if "nao" not in sys.path:
        sys.path.insert(0, "nao")
    return pytest.importorskip("wake_state")


def test_identity_key_prefers_recognised_name():
    ws = _load_wake_state()
    face = {"face_id": "18", "name": "Aayush", "confidence": 0.9}
    assert ws.identity_key_for_face(face) == "Aayush"


def test_identity_key_falls_back_to_face_id_when_unrecognised():
    ws = _load_wake_state()
    face = {"face_id": "18", "name": "", "confidence": 0.0}
    assert ws.identity_key_for_face(face) == "18"


def test_identity_key_empty_when_nothing_known():
    ws = _load_wake_state()
    assert ws.identity_key_for_face({}) == ""
