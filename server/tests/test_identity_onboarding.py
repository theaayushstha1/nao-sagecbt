from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("server.app_ws")

from server import app_ws  # noqa: E402


class _FakeWs:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_text(self, payload: str) -> None:
        self.frames.append(json.loads(payload))


@pytest.mark.asyncio
async def test_recognized_identity_ignores_late_unknown_scan(monkeypatch):
    """A late unknown face scan must not re-open name onboarding.

    Robot-side face recognition can send a confident match and then a later
    unknown result as lighting/angle changes. Once a session is recognized,
    identity is sticky for that session.
    """
    app_ws._IDENTIFIED_USERS.clear()
    sess = app_ws._Session("guest")
    ws = _FakeWs()
    calls: list[tuple[str, str | None]] = []

    async def _fake_greet(_ws, _sess, display_name, *, reason):
        calls.append(("greet", display_name))

    async def _fake_prompt(_ws, _sess, *, reason):
        calls.append(("prompt", reason))
        _sess.asking_name = True

    monkeypatch.setattr(app_ws, "_emit_returning_identity_greeting", _fake_greet)
    monkeypatch.setattr(app_ws, "_emit_onboarding_name_prompt", _fake_prompt)

    await app_ws._ingest_control(ws, sess, {
        "subtype": "user_identified",
        "data": {
            "name": "Aayush",
            "recognized": True,
            "face_visible": True,
            "source": "face",
        },
    })
    await app_ws._ingest_control(ws, sess, {
        "subtype": "user_identified",
        "data": {
            "name": None,
            "recognized": False,
            "face_visible": True,
            "source": "late_unknown_face",
        },
    })

    assert calls == [("greet", "Aayush")]
    assert sess.username == "aayush"
    assert sess.asking_name is False
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["recognized"] is True
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["name"] == "Aayush"


@pytest.mark.asyncio
async def test_onboarding_prompt_skips_when_identity_already_recognized():
    app_ws._IDENTIFIED_USERS.clear()
    sess = app_ws._Session("aayush")
    ws = _FakeWs()
    sess.asking_name = True
    app_ws._IDENTIFIED_USERS[sess.session_id] = {
        "name": "Aayush",
        "recognized": True,
        "face_visible": True,
        "greeted": True,
        "prompted": False,
    }

    await app_ws._emit_onboarding_name_prompt(
        ws,
        sess,
        reason="late_unknown_face",
    )

    assert ws.frames == []
    assert sess.asking_name is False


def test_onboarding_prompt_echo_is_detected():
    transcript = (
        "Heads up. My camera is on for this conversation. "
        "Hi, I'm NAO. What should I call you?"
    )

    assert app_ws._is_onboarding_prompt_echo(transcript) is True


def test_onboarding_prompt_echo_does_not_match_real_name():
    assert app_ws._is_onboarding_prompt_echo("you can call me Aayush") is False


@pytest.mark.asyncio
async def test_wake_event_binds_returning_username_before_session_resume(monkeypatch):
    app_ws._IDENTIFIED_USERS.clear()
    sess = app_ws._Session("guest")
    ws = _FakeWs()
    ensure_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        app_ws,
        "_lookup_returning_user",
        lambda face_id: (True, "Aayush"),
    )
    monkeypatch.setattr(app_ws, "_last_recap_line", lambda username: None)
    monkeypatch.setattr(app_ws, "_synth_for", lambda username, text: b"mp3")

    def _fake_ensure(username, hint):
        ensure_calls.append((username, hint))
        return 1

    monkeypatch.setattr(app_ws.legacy, "ensure_active_session", _fake_ensure)

    await app_ws._handle_wake_event(ws, sess, {
        "face_id": "Aayush",
        "gate": "face",
        "confidence": 0.91,
        "distance_m": 0.8,
    })

    assert sess.username == "aayush"
    assert ensure_calls == [("aayush", None)]
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["recognized"] is True
    assert app_ws._IDENTIFIED_USERS[sess.session_id]["prompted"] is False


def test_onboarding_echo_guard_catches_the_retry_prompt():
    """NAO's *retry* prompt must be caught when it hears its own voice.

    The retry says "what NAME should I call you"; the marker list only had
    "what should i call you". That one interposed word let the robot's own
    voice through as if it were the user, and since the echo is never a
    name, it re-asked and re-heard itself forever.
    """
    assert app_ws._is_onboarding_prompt_echo(app_ws._ONBOARDING_NAME_RETRY) is True


def test_onboarding_echo_guard_covers_every_prompt_constant():
    """Guard must stay in sync with the prompts it is meant to catch.

    Hand-written markers drifted from the spoken text once already; pin
    every prompt constant so a reworded prompt can't silently slip past.
    """
    for text in (app_ws._ONBOARDING_NAME_PROMPT, app_ws._ONBOARDING_NAME_RETRY):
        assert app_ws._is_onboarding_prompt_echo(text) is True, text


def test_onboarding_name_retry_gives_up_after_max_retries(monkeypatch):
    """The name prompt must stop asking eventually and proceed as guest.

    With no counter, a user who never gives a parseable name is asked
    forever. Give up after a bounded number of tries.

    Driven via ``asyncio.run`` rather than ``@pytest.mark.asyncio`` because
    pytest-asyncio is not a dependency of this project — the marker silently
    skips instead of running.
    """
    monkeypatch.setattr(app_ws, "_synth_for", lambda username, text: b"mp3")
    sess = app_ws._Session("guest")
    sess.asking_name = True
    ws = _FakeWs()

    async def _drive() -> None:
        for _ in range(app_ws._ONBOARDING_NAME_MAX_RETRIES + 1):
            await app_ws._emit_onboarding_name_retry(ws, sess, reason="test")

    asyncio.run(_drive())

    assert sess.asking_name is False


def test_wake_event_asks_name_for_new_user(monkeypatch):
    """A new user must be asked their name on wake, not silently deferred.

    The name prompt used to fire only from a `user_identified` frame with an
    unrecognised face. Wake by touch or proximity resolves no face, so the
    robot woke, said nothing, and waited — looking broken. Ask on wake.
    """
    app_ws._IDENTIFIED_USERS.clear()
    prompts: list[str] = []

    async def _fake_prompt(_ws, _sess, *, reason):
        prompts.append(reason)
        _sess.asking_name = True

    monkeypatch.setattr(app_ws, "_emit_onboarding_name_prompt", _fake_prompt)
    monkeypatch.setattr(app_ws, "_lookup_returning_user", lambda face_id: (False, None))
    monkeypatch.setattr(app_ws, "_last_recap_line", lambda username: None)
    monkeypatch.setattr(app_ws, "_synth_for", lambda username, text: b"mp3")
    monkeypatch.setattr(app_ws.legacy, "ensure_active_session", lambda u, h: 1)

    sess = app_ws._Session("guest")
    ws = _FakeWs()

    asyncio.run(app_ws._handle_wake_event(ws, sess, {
        "face_id": "18",
        "gate": "touch",
        "confidence": 0.0,
        "distance_m": 0.5,
    }))

    assert prompts, "new user was not asked their name on wake"
    assert sess.asking_name is True


def test_wake_event_does_not_ask_returning_user(monkeypatch):
    """A recognised user must be greeted, never re-asked for their name."""
    app_ws._IDENTIFIED_USERS.clear()
    prompts: list[str] = []

    async def _fake_prompt(_ws, _sess, *, reason):
        prompts.append(reason)

    async def _fake_greet(_ws, _sess, name, *, reason):
        pass

    monkeypatch.setattr(app_ws, "_emit_onboarding_name_prompt", _fake_prompt)
    monkeypatch.setattr(app_ws, "_emit_returning_identity_greeting", _fake_greet)
    monkeypatch.setattr(app_ws, "_lookup_returning_user", lambda face_id: (True, "Aayush"))
    monkeypatch.setattr(app_ws, "_last_recap_line", lambda username: None)
    monkeypatch.setattr(app_ws, "_synth_for", lambda username, text: b"mp3")
    monkeypatch.setattr(app_ws.legacy, "ensure_active_session", lambda u, h: 1)

    sess = app_ws._Session("guest")
    ws = _FakeWs()

    asyncio.run(app_ws._handle_wake_event(ws, sess, {
        "face_id": "Aayush",
        "gate": "face",
        "confidence": 0.91,
        "distance_m": 0.8,
    }))

    assert prompts == [], "returning user must not be asked their name"


def test_learn_face_persists_user_row(monkeypatch):
    """Learning a face must also write the users row that identifies it.

    `learn_face` teaches the NAOqi face database on the robot, which is what
    makes NAO's eyes recognise you later. But the name -> profile mapping
    lives in server-side SQLite, and `memory.ensure_user` — its only writer —
    is called from the legacy Flask app (`server/server.py:33`), never from
    the live WS path. So a face could be learned on the robot ('Mia' was)
    while the users table never heard of it, and the returning-user lookup
    could never match. Teach both or neither.
    """
    saved: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app_ws.memory, "ensure_user",
        lambda face_id, display_name=None, **kw: saved.append((face_id, display_name)),
    )
    monkeypatch.setattr(app_ws, "_synth_for", lambda username, text: b"mp3")
    monkeypatch.setattr(app_ws, "_reset_reply_chunks", lambda u, t: None)

    sess = app_ws._Session("guest")
    sess.asking_name = True
    ws = _FakeWs()
    motion = app_ws.motion_trigger.MotionMatch(
        action="learn_face",
        args={"name": "Mingma"},
        ack="Nice to meet you, Mingma.",
    )

    asyncio.run(app_ws._emit_motion(ws, sess, "Mingma", motion, {}))

    assert saved == [("Mingma", "Mingma")], (
        "learn_face must upsert the users row so the face NAO learns is the "
        "same identity the server can look up next session"
    )
    assert sess.username == "mingma", "session must bind to the learned name"
