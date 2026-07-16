# wake_state.py
# -*- coding: utf-8 -*-
"""
Phase 3: Hybrid Wake — Face-First with Word Fallback (state machine).

This module owns the robot's top-level wake behaviour. It replaces the
"keyword-only, then enter conversation" loop with a five-state machine that
keeps the robot continuously *aware* of the room, but only *engages* (LEDs
brighten, chime fires, WS session opens) when one of several engagement
gates fires. Walking past the robot must NOT wake it; standing in front of
it and looking at it should.

State chart (from PRD v2 §Phase 3 + PHASE_3_TASK_MAP §Wake state machine):

    IDLE       — eyes dim gray, downward gaze, no audio activity
                 trigger: face conf >= face_min_conf AND distance in
                          [0, face_max_distance_m] AND |angle| <= face_max_angle_deg
                 -> AWARE
    AWARE      — face detected, NOT YET ENGAGED. Eyes soft blue (animacy
                 cue, no chime, no speech). Head tracks the face gently.
                 Concurrently evaluates 5 engagement gates:
                   * mutual_gaze   - sustained mutual gaze >= gaze_required_s
                   * proximity     - distance < 1.0 m stable for proximity_required_s
                   * sustained_face- conf >= sustained_conf for sustained_required_s
                                     with frontal angle <= sustained_angle_deg
                   * speech        - AdaptiveVad signals speech onset
                   * keyword       - WakeListener (ALSpeechRecognition) heard "hey nao"
                 If no gate fires within aware_timeout_s OR face lost -> IDLE silently.
                 -> ENGAGED
    ENGAGED    — engagement gate fired. Soft chime + eyes solid blue.
                 Calls on_engaged(face_id, gate, confidence, distance_m) so
                 main.py can open the WS session and send wake_event.
                 -> LISTENING (set externally once server greeting plays / user speaks)
    LISTENING  — robot is listening to the user; mic stream active.
                 Eyes cyan; gaze aversion handled outside this module.
    SPEAKING   — TTS is playing. Eyes warm yellow; mic gated by Phase 1.
                 Returns to LISTENING via set_state("LISTENING") from the
                 WS client when TTS finishes.

Threading model
---------------
``start()`` is BLOCKING. It launches three daemon worker threads and then
parks on a stop event:

    1. _face_loop   — polls ALFaceDetection at ~30 fps, drives IDLE<->AWARE
                      transitions and feeds each engagement gate evaluator.
    2. _vad_loop    — watches AdaptiveVad for speech onset; sets the speech
                      gate flag when it fires.
    3. _keyword_loop- polls the existing WakeListener fallback path; sets
                      the keyword gate flag when it fires.

A single ``threading.Lock`` (``self._state_lock``) guards every state
transition so callbacks fire exactly once per change and external
``set_state`` calls cannot race the face loop.

Public API contract (PHASE_3_TASK_MAP):
    WakeStateMachine(nao_ip, nao_port, leds, fallback_word_listener,
                     on_engaged, on_lost, on_listening, on_speaking_done,
                     face_min_conf=0.35, face_max_distance_m=1.5,
                     face_max_angle_deg=60.0,
                     aware_timeout_s=8.0, gaze_required_s=1.5,
                     proximity_required_s=1.0, sustained_conf=0.5,
                     sustained_required_s=2.0, sustained_angle_deg=30.0)
    .start() -> blocks until stop()
    .stop()  -> idempotent
    .current_state() -> str
    .set_state(state) -> external trigger (server force-transitions)

Python 2.7 compatible. ``from __future__ import print_function`` only.
No f-strings, no type hints, no asyncio.
"""
from __future__ import print_function

import threading
import time

# ── Optional dependencies — guarded so this file imports cleanly on a
#    developer laptop without naoqi, and so parallel worktrees that haven't
#    landed sibling files (face_naoqi extensions, leds.py, logger.py) yet
#    don't break our py_compile / AST parse. The runtime checks ``is None``
#    on each guarded import before relying on it.
try:
    from naoqi import ALProxy  # pragma: no cover - robot only
except ImportError:  # pragma: no cover - dev environment
    ALProxy = None

# Sibling Phase 3 worktree adds these helpers to face_naoqi. We optimistically
# import them; if the sibling hasn't merged yet, _face_loop falls back to a
# minimal inline ALMemory("FaceDetected") reader that exposes the same
# {face_id, confidence, distance_m, yaw_deg, pitch_deg} shape.
try:
    from utils.face_naoqi import (
        detect_faces_with_geometry as _detect_faces_with_geometry,
        closest_face as _closest_face,
        is_mutually_gazing as _is_mutually_gazing,
    )
except Exception:  # pragma: no cover - sibling worktree race
    _detect_faces_with_geometry = None
    _closest_face = None
    _is_mutually_gazing = None

# Logger is project-internal. The contract spec calls for
# ``nao.logger.get_logger("wake_state")`` — but the actual signature in
# nao/logger.py is keyword-only (``get_logger(**ctx)``). We adapt to whatever
# is available so this file is robust to either signature, and document the
# discrepancy in the contract questions returned to the orchestrator.
try:
    from logger import get_logger as _get_logger  # type: ignore
except Exception:  # pragma: no cover
    try:
        from nao.logger import get_logger as _get_logger  # type: ignore
    except Exception:  # pragma: no cover
        _get_logger = None


def _make_logger():
    """Return a structured logger if available; else a no-op shim.

    Spec asks for ``get_logger("wake_state")`` (positional). The current
    impl is ``get_logger(**ctx)`` (kwargs). We try the kwarg form first
    since that's what's checked into the repo, then fall back to positional
    in case a future PR changes it. Either way, never raise.
    """
    if _get_logger is None:
        class _NoopLogger(object):
            def bind(self, **kw):
                return self

            def info(self, event, **kw):
                pass

            def warn(self, event, **kw):
                pass

            def error(self, event, **kw):
                pass

            def debug(self, event, **kw):
                pass

            def exception(self, event, **kw):
                pass

        return _NoopLogger()
    # Try kwarg form (current impl).
    try:
        return _get_logger(component="wake_state")
    except TypeError:
        # Fall back to positional form per spec wording.
        try:
            return _get_logger("wake_state")
        except Exception:
            class _NoopLogger(object):
                def bind(self, **kw):
                    return self

                def info(self, event, **kw):
                    pass

                def warn(self, event, **kw):
                    pass

                def error(self, event, **kw):
                    pass

                def debug(self, event, **kw):
                    pass

                def exception(self, event, **kw):
                    pass

            return _NoopLogger()


# ── State constants (kept as plain strings so logs and JSON envelopes can
#    pass them around without any enum import dance on py2.7).
STATE_IDLE = "IDLE"
STATE_AWARE = "AWARE"
STATE_ENGAGED = "ENGAGED"
STATE_LISTENING = "LISTENING"
STATE_SPEAKING = "SPEAKING"

# Engagement gate names (used in callback payload + telemetry).
GATE_MUTUAL_GAZE = "mutual_gaze"
GATE_PROXIMITY = "proximity"
GATE_SUSTAINED_FACE = "sustained_face"
GATE_SPEECH = "speech"
GATE_KEYWORD = "keyword"
GATE_TOUCH = "touch"  # head tactile sensor — manual wake bypass

# ALMemory keys for the three NAO V6 head tactile sensors. Any of them
# touched fires the touch gate (we don't care which — front/middle/rear
# all mean "user wants my attention").
def identity_key_for_face(face):
    """Return the key the server should identify this face by.

    ALFaceDetection hands back two different things: an internal face id
    (``extra_info[0]``) that is renumbered on every re-detection, and the
    recognised name (``extra_info[2]``) that ``learnFace()`` stored. Only the
    name is stable across sessions, and the server's ``users`` table is keyed
    on it.

    Sending the numeric id meant the returning-user lookup asked for "18" in
    a table of names — never matched, so the same person was a brand-new user
    every session. Prefer the name; fall back to the id for an unrecognised
    face so it is at least consistent within one session.
    """
    face = face or {}
    name = str(face.get("name") or "").strip()
    if name:
        return name
    return str(face.get("face_id") or "").strip()


_HEAD_TOUCH_KEYS = (
    "FrontTactilTouched",
    "MiddleTactilTouched",
    "RearTactilTouched",
)

# Internal cadence knobs. 30 fps face polling matches PHASE_3_TASK_MAP and
# matches what ALFaceDetection delivers when subscribed at ``period=100`` ms
# with the underlying camera at 30 fps. We poll faster (33 ms) so the IDLE
# -> AWARE transition is bound by ALMemory write latency rather than our
# loop cadence.
_FACE_POLL_INTERVAL_S = 1.0 / 30.0
_KEYWORD_POLL_INTERVAL_S = 0.05
# How long without seeing a face before we declare "face lost" inside AWARE.
# 200 ms is one ALFaceDetection event period; we wait ~3 events to filter
# brief detection dropouts (head turn, hand passes in front of camera).
_FACE_LOSS_TIMEOUT_S = 0.6
# Distance is reported by face_naoqi.detect_faces_with_geometry. If the
# sibling worktree hasn't landed yet, we fall back to a constant estimate.
_FALLBACK_DISTANCE_M = 0.8


# ── Inline fallback face reader ──────────────────────────────────────────
# Used only if face_naoqi.detect_faces_with_geometry isn't available
# (parallel worktree race). Shape matches the documented contract so the
# rest of the state machine doesn't care which path produced the data.
def _fallback_detect_faces(memory, max_age_ms=200):
    """Minimal ALMemory("FaceDetected") parser.

    Returns a list of dicts with the keys our gates need:
        face_id, name, confidence, distance_m, yaw_deg, pitch_deg.
    Most of these are best-effort approximations — the real implementation
    in face_naoqi.py is owned by the sibling worktree and will replace
    this stub once both branches merge.
    """
    if memory is None:
        return []
    try:
        data = memory.getData("FaceDetected")
    except Exception:
        return []
    if not data or not isinstance(data, list) or len(data) < 2:
        return []
    info_array = data[1]
    if not isinstance(info_array, list) or not info_array:
        return []
    out = []
    for entry in info_array:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        # ALFaceDetection layout: entry[0] is the shape info (head angles),
        # entry[1] is the extra info (face id, score, name, ...).
        shape_info = entry[0] if isinstance(entry[0], list) else []
        extra_info = entry[1] if isinstance(entry[1], list) else []
        # Head angles (yaw/pitch) — when shape info is the "circle in image"
        # form we don't get them directly; default to zero (frontal).
        yaw_deg = 0.0
        pitch_deg = 0.0
        # Confidence — extra_info[1] is the recogntion score per ALDocs.
        confidence = 0.0
        try:
            if len(extra_info) >= 2:
                confidence = float(extra_info[1] or 0.0)
        except Exception:
            confidence = 0.0
        # Name + face_id — extra_info[2] is the recognized name string,
        # extra_info[0] is the internal face id.
        face_id = ""
        name = ""
        try:
            if len(extra_info) >= 1:
                face_id = str(extra_info[0] or "")
        except Exception:
            face_id = ""
        try:
            if len(extra_info) >= 3:
                name = str(extra_info[2] or "")
        except Exception:
            name = ""
        # Distance — without the proper geometry block we can only estimate
        # via the fraction of the frame the face occupies. Fall back to a
        # constant; the proximity gate will be permissive in this mode.
        distance_m = _FALLBACK_DISTANCE_M
        # We use the shape_info array length as a sanity check that we got
        # geometry; if so, we can compute a rough distance from the face
        # circle's width fraction. This is a soft heuristic — sibling
        # face_naoqi will replace it with the real camera-FOV calculation.
        try:
            if len(shape_info) >= 4:
                # shape_info: [alpha, beta, sizeX, sizeY] — sizeX is the
                # face width as a fraction of the image. With NAO's 60.97°
                # H-FOV camera and ~16 cm typical face width: distance_m =
                # 0.16 / (2 * tan(0.5 * fov_rad * sizeX)). For sizeX ~0.2
                # this lands around 0.7 m, matching the observed scale.
                size_x = float(shape_info[2] or 0.0)
                if size_x > 0.0:
                    # Tight approximation around NAO's camera optics.
                    # 0.20 frame fraction ~ 0.7 m, 0.10 ~ 1.4 m.
                    distance_m = max(0.10, min(3.0, 0.14 / size_x))
        except Exception:
            distance_m = _FALLBACK_DISTANCE_M
        out.append({
            "face_id": face_id,
            "name": name,
            "confidence": confidence,
            "distance_m": float(distance_m),
            "yaw_deg": float(yaw_deg),
            "pitch_deg": float(pitch_deg),
        })
    return out


def _fallback_closest(faces):
    if not faces:
        return None
    # Lowest distance wins; ties broken by highest confidence.
    faces_sorted = sorted(
        faces,
        key=lambda f: (float(f.get("distance_m", 1e9)),
                       -float(f.get("confidence", 0.0))),
    )
    return faces_sorted[0]


def _fallback_is_mutually_gazing(face, yaw_tol=15.0, pitch_tol=15.0):
    if not face:
        return False
    try:
        yaw = abs(float(face.get("yaw_deg", 0.0) or 0.0))
        pitch = abs(float(face.get("pitch_deg", 0.0) or 0.0))
    except Exception:
        return False
    return yaw <= yaw_tol and pitch <= pitch_tol


# ── WakeStateMachine ─────────────────────────────────────────────────────
class WakeStateMachine(object):
    """Continuous five-state wake machine. Blocking ``start()``.

    Owned and driven by ``nao/main.py``. The state machine is the only thing
    that decides when the robot transitions between IDLE and ENGAGED — it is
    NOT triggered by the WS client, the conversation loop, or any agent
    output. The server can force-transition (for example, on crisis lock) via
    ``set_state(STATE_LISTENING)`` and similar.
    """

    STATES = (STATE_IDLE, STATE_AWARE, STATE_ENGAGED, STATE_LISTENING, STATE_SPEAKING)

    # ------------------------------------------------------------------
    def __init__(self, nao_ip, nao_port,
                 leds, fallback_word_listener,
                 on_engaged, on_lost, on_listening, on_speaking_done,
                 face_min_conf=0.35, face_max_distance_m=1.5,
                 face_max_angle_deg=60.0,
                 aware_timeout_s=8.0, gaze_required_s=1.5,
                 proximity_required_s=1.0, sustained_conf=0.5,
                 sustained_required_s=2.0, sustained_angle_deg=30.0,
                 vad=None, memory_proxy=None, face_detection_proxy=None,
                 multi_person_callback=None,
                 multi_person_distance_m=1.5,
                 returning_user_resolver=None):
        """Construct the state machine.

        Parameters mirror ``PHASE_3_TASK_MAP §Public APIs``. The trailing
        three Phase 3 args (``vad``, ``memory_proxy``,
        ``face_detection_proxy``) are OPTIONAL injection seams for tests
        + future server-driven hooks. The Phase 8 trio
        (``multi_person_callback``, ``multi_person_distance_m``,
        ``returning_user_resolver``) is also OPTIONAL — every existing
        Phase 3 caller continues to work without supplying them.

        Production callers typically only pass the contract-required args
        and let the state machine create its own naoqi proxies.

        Parameters
        ----------
        nao_ip, nao_port : str, int
            Robot connection (passed straight to ALProxy when we own the
            ALFaceDetection / ALMemory subscriptions).
        leds : nao.leds.LedDriver instance
            Drives eye / chest LEDs and plays the engagement chime. Must
            expose at minimum: ``set_idle()``, ``set_aware()``,
            ``set_engaged()``, ``set_listening()``, ``set_speaking()``,
            ``chime()``. We never inspect LedDriver internals, so any
            duck-typed substitute works (the self-test uses one).
        fallback_word_listener : nao.wake_listener.WakeListener-like
            Optional. If non-None, ``start()`` polls it for the keyword
            engagement gate. The state machine never instantiates one
            itself — main.py decides whether to wire the keyword path.
        on_engaged : callable(face_id, gate_name, confidence, distance_m
                              [, returning_user_hint])
            Fired exactly once per IDLE -> AWARE -> ENGAGED transition.
            main.py uses this hook to open the WS session and send the
            ``wake_event`` control frame. Phase 8 adds an optional
            ``returning_user_hint`` 5th positional argument; the state
            machine probes the callback signature and degrades to the
            Phase 3 4-arg call when the caller doesn't accept the hint
            (so existing Phase 3 tests keep passing).
        on_lost : callable()
            Fired on AWARE timeout / face-lost-during-AWARE. Used to clean
            up WS state if main.py opened anything optimistically.
        on_listening : callable()
            Fired on transition into LISTENING (whether from ENGAGED via
            user speech or from SPEAKING via server set_state).
        on_speaking_done : callable()
            Fired on SPEAKING -> LISTENING transition (TTS finished).

        Phase 8 extensions
        ------------------
        multi_person_callback : callable(faces_list) -> None, optional
            Fired exactly once on the IDLE -> AWARE transition (and again
            on AWARE -> ENGAGED if the same condition still holds) when
            two or more faces are simultaneously visible within
            ``multi_person_distance_m``. ``faces_list`` is a list of dicts
            shaped like the per-face payload from
            ``utils.face_naoqi.detect_faces_with_geometry`` — keys
            include ``face_id``, ``name``, ``confidence``, ``distance_m``,
            ``yaw_deg``, ``pitch_deg``. The callback runs on the face
            loop thread; long-running work belongs on a daemon spawned
            inside the callback, not in the callback body itself.

            If ``None`` (the Phase 3 default), the multi-person path is
            inert. The default implementation in the docstring spec is
            "logs"; main.py is the production source of the actual TTS
            "Hi everyone — who'd like to chat first?" greeting per
            PHASE_8_TASK_MAP §Group scenario.
        multi_person_distance_m : float, default 1.5
            Distance threshold (in metres) used to decide whether a face
            counts as "in conversation range" for the multi-person gate.
            Defaults to the same 1.5 m used by ``face_max_distance_m``
            so the engagement window and the multi-person window stay
            aligned.
        returning_user_resolver : callable(face_id) -> str | dict | None,
                                  optional
            Looked up at ENGAGED time to populate the new
            ``returning_user_hint`` argument on ``on_engaged``. Typical
            wiring: ``main.py`` passes a closure that consults
            ``brain_cache.get_user_for_face(face_id)`` and returns the
            stored display name (or a small dict with extra metadata).

            If ``None`` we still attempt to derive a hint from the live
            face record — the ``name`` field populated by
            ``ALFaceDetection`` is itself a returning-user signal — so
            existing Phase 3 callers see a smooth upgrade path.

        Threshold knobs match PHASE_3_TASK_MAP defaults. They can be
        overridden per-deployment via env if a future iteration exposes
        them, but the defaults are tuned for the v2 demo classroom.
        """
        self._nao_ip = nao_ip
        self._nao_port = nao_port
        self._leds = leds
        self._fallback_word_listener = fallback_word_listener

        # State callbacks — copy refs so a None on the contract path doesn't
        # turn into an AttributeError later. Each call site guards with
        # ``callable(...)`` to be defensive against stub objects in tests.
        self._on_engaged = on_engaged
        self._on_lost = on_lost
        self._on_listening = on_listening
        self._on_speaking_done = on_speaking_done
        # Optional barge callback: fires when the user touches the head
        # tactile sensors while the robot is in SPEAKING state. main.py
        # wires this to tts_player.stop() so the user can cut NAO off
        # mid-reply just by tapping its head.
        self._on_barge = None

        # Phase 8: multi-person greeting + returning-user hint.
        self._multi_person_callback = multi_person_callback
        self._multi_person_distance_m = float(multi_person_distance_m)
        self._returning_user_resolver = returning_user_resolver

        # Threshold knobs — kept on the instance so external tooling can
        # log them when explaining why a wake fired/didn't fire.
        self._face_min_conf = float(face_min_conf)
        self._face_max_distance_m = float(face_max_distance_m)
        self._face_max_angle_deg = float(face_max_angle_deg)
        self._aware_timeout_s = float(aware_timeout_s)
        self._gaze_required_s = float(gaze_required_s)
        self._proximity_required_s = float(proximity_required_s)
        self._sustained_conf = float(sustained_conf)
        self._sustained_required_s = float(sustained_required_s)
        self._sustained_angle_deg = float(sustained_angle_deg)

        # Optional injections (tests + future hooks).
        self._vad = vad
        self._memory_proxy = memory_proxy
        self._face_detection = face_detection_proxy

        # State machine plumbing.
        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Speech / keyword gate flags. The face loop reads these without a
        # lock — they're set by their respective worker threads via simple
        # boolean writes (atomic on CPython under the GIL on py2.7).
        self._speech_onset_flag = False
        self._keyword_flag = False

        # Telemetry / context — last seen face becomes the wake_event payload.
        self._last_face = None

        # Phase 8: cache of all visible faces from the most recent face-loop
        # tick. Populated alongside ``_last_face`` so the multi-person
        # callback has the full set, not just the closest one. Updated under
        # ``_state_lock`` to keep snapshots consistent across threads.
        self._last_faces = []

        # Phase 8: latched flag so ``multi_person_callback`` fires at most
        # once per fresh wake cycle. Reset alongside the speech/keyword gate
        # flags on entry to IDLE / LISTENING (see ``_transition``).
        self._multi_person_fired = False

        # Worker thread handles. Created in start(), joined in stop().
        self._face_thread = None
        self._vad_thread = None
        self._keyword_thread = None
        self._touch_thread = None

        # Track whether we own the ALFaceDetection subscription so stop()
        # can unsubscribe cleanly without tripping over an injected proxy
        # that the caller wants to keep alive.
        self._owns_face_subscription = False
        self._face_subscriber_name = "WakeFaceDetection"

        self._log = _make_logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def current_state(self):
        """Return the current state string. Threadsafe snapshot."""
        with self._state_lock:
            return self._state

    def set_barge_callback(self, cb):
        """Wire a callable to fire when the user touches NAO's head while
        TTS is playing (SPEAKING state). Used for head-touch barge-in.
        Pass None to clear. Idempotent.
        """
        self._on_barge = cb if callable(cb) else None

    @property
    def current_face_id(self):
        """Phase 8: face_id of the most recently engaged / closest face.

        Returns the empty string (NOT None) when no face has been seen
        yet — keeps callers from having to special-case both None and ""
        when piping the value into ``learn_new_face_naoqi(name)`` or the
        ``brain_cache.get_user_for_face(face_id)`` lookup. Use
        ``bool(wsm.current_face_id)`` for "have we seen anybody?".

        Threadsafe snapshot — reads ``_last_face`` under ``_state_lock``
        so callers don't observe a half-updated face dict.
        """
        with self._state_lock:
            face = self._last_face or {}
        try:
            value = face.get("face_id", "") if isinstance(face, dict) else ""
        except Exception:
            return ""
        if value is None:
            return ""
        try:
            return str(value)
        except Exception:
            return ""

    @property
    def current_faces(self):
        """Phase 8: list snapshot of all faces from the latest face-loop tick.

        The list is a shallow copy so callers may mutate it without
        racing the face loop. Returned in the same shape as
        ``utils.face_naoqi.detect_faces_with_geometry``: a list of dicts
        with ``face_id``, ``name``, ``confidence``, ``distance_m``,
        ``yaw_deg``, ``pitch_deg`` keys. Empty list when no face is
        visible.
        """
        with self._state_lock:
            return [dict(f) for f in (self._last_faces or [])]

    def set_state(self, state):
        """External force-transition.

        Used by main.py / WS client when authoritative external events
        force a state change (server greeting started -> SPEAKING; server
        TTS finished -> LISTENING; crisis_lock -> SPEAKING then IDLE).

        This is the *only* path by which LISTENING / SPEAKING are entered:
        the engagement gates inside this module promote IDLE -> AWARE ->
        ENGAGED, but turning ENGAGED into LISTENING is owned by main.py
        once it has opened the WS session and either greeted or yielded
        the floor to the user.

        Idempotent — passing the current state is a no-op (callbacks
        do NOT re-fire on entry to the same state).
        """
        if state not in self.STATES:
            self._log.warn("set_state_invalid", target=str(state))
            return
        self._transition(state, source="external")

    def start(self):
        """Block until ``stop()`` is called.

        Spins three daemon worker threads (face / vad / keyword), drives
        IDLE LEDs, and parks on the stop event. The worker threads do all
        the actual gate evaluation; this method exists so the caller has a
        clean ``join`` semantic from main.py:

            wsm = WakeStateMachine(...)
            try:
                wsm.start()           # blocks
            except KeyboardInterrupt:
                wsm.stop()

        Reentrant calls (calling start twice) are rejected with a logged
        warning rather than raising — main.py shouldn't do that anyway,
        and raising would cascade into a robot that won't boot.
        """
        if self._face_thread is not None and self._face_thread.is_alive():
            self._log.warn("start_already_running")
            return

        self._stop_event.clear()
        self._setup_face_subscription()
        # Initial LEDs — IDLE eyes, dim gray.
        self._safe_led_call(self._leds, "set_idle")

        # Face polling loop is the heart of the machine.
        self._face_thread = threading.Thread(
            target=self._face_loop, name="wake-face-loop"
        )
        self._face_thread.daemon = True
        self._face_thread.start()

        # AdaptiveVad -> speech gate.
        if self._vad is not None:
            self._vad_thread = threading.Thread(
                target=self._vad_loop, name="wake-vad-loop"
            )
            self._vad_thread.daemon = True
            self._vad_thread.start()

        # WakeListener fallback -> keyword gate.
        if self._fallback_word_listener is not None:
            self._keyword_thread = threading.Thread(
                target=self._keyword_loop, name="wake-keyword-loop"
            )
            self._keyword_thread.daemon = True
            self._keyword_thread.start()

        # Head tactile sensor -> touch gate. Always-on, no setup required —
        # ALMemory exposes the keys directly. This is the manual override
        # path: a user can touch any of the three head pads to skip the
        # face-detection warm-up entirely.
        self._touch_thread = threading.Thread(
            target=self._touch_loop, name="wake-touch-loop"
        )
        self._touch_thread.daemon = True
        self._touch_thread.start()

        self._log.info("wake_state_started",
                       state=self._state,
                       face_min_conf=self._face_min_conf,
                       aware_timeout_s=self._aware_timeout_s)

        # Park here until stop() is called. We use the event with a long
        # poll so KeyboardInterrupt at the REPL still surfaces promptly.
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1.0)

        self._log.info("wake_state_stopping")
        self._teardown_face_subscription()

    def stop(self):
        """Idempotent shutdown.

        Signals the worker threads to exit, then joins them with a short
        timeout each. Always returns — never blocks indefinitely on a
        hung naoqi proxy.
        """
        if self._stop_event.is_set() and (
            self._face_thread is None or not self._face_thread.is_alive()
        ):
            # Already stopped (or never started).
            return

        self._stop_event.set()

        for th in (self._face_thread, self._vad_thread,
                   self._keyword_thread, self._touch_thread):
            if th is None:
                continue
            try:
                th.join(timeout=1.5)
            except Exception:
                pass

        # Best-effort teardown for case the user calls stop() without a
        # prior start() (and start's own teardown didn't run yet).
        self._teardown_face_subscription()

    # ------------------------------------------------------------------
    # Internal: state transition primitive
    # ------------------------------------------------------------------
    def _transition(self, new_state, source="internal", gate=None):
        """Atomically transition states and fire callbacks.

        Callbacks fire ONLY on a real change (per task map §5).
        State callbacks are invoked outside the state lock so a slow
        callback (e.g. WS handshake in on_engaged) doesn't block the
        face loop from re-evaluating the next frame.

        Phase 8 additions:
            * On AWARE / ENGAGED entry, the multi-person callback is
              fired (at most once per wake cycle) when ≥ 2 faces are
              within ``multi_person_distance_m`` of the camera.
            * ``on_engaged`` is invoked with a 5th ``returning_user_hint``
              argument when the callback signature accepts it; legacy 4-arg
              callbacks keep working unchanged.
        """
        callback_to_fire = None
        callback_args = ()

        with self._state_lock:
            prev = self._state
            if prev == new_state:
                return  # never re-fire on entry to the same state
            self._state = new_state
            face = self._last_face or {}
            face_id = identity_key_for_face(face)
            confidence = float(face.get("confidence", 0.0) or 0.0)
            distance_m = float(face.get("distance_m", 0.0) or 0.0)
            # Snapshot the visible faces so the multi-person callback
            # always sees the same list the gate decision used.
            faces_snapshot = [dict(f) for f in (self._last_faces or [])]

        self._log.info("wake_state_transition",
                       prev=prev, next=new_state, source=source, gate=gate)

        # LED update + callback dispatch happen outside the lock. We pick
        # the callback up front so a callback-issued ``set_state`` call
        # can't get blocked on our own lock.
        if new_state == STATE_IDLE:
            self._safe_led_call(self._leds, "set_idle")
            if prev == STATE_AWARE:
                # Lost / timed out without engaging.
                callback_to_fire = self._on_lost
        elif new_state == STATE_AWARE:
            self._safe_led_call(self._leds, "set_aware")
        elif new_state == STATE_ENGAGED:
            self._safe_led_call(self._leds, "chime")
            self._safe_led_call(self._leds, "set_engaged")
            callback_to_fire = self._on_engaged
            callback_args = (face_id, gate or "unknown", confidence,
                             distance_m, face)
        elif new_state == STATE_LISTENING:
            self._safe_led_call(self._leds, "set_listening")
            if prev == STATE_SPEAKING:
                callback_to_fire = self._on_speaking_done
            else:
                callback_to_fire = self._on_listening
        elif new_state == STATE_SPEAKING:
            self._safe_led_call(self._leds, "set_speaking")

        # Phase 8 multi-person check. Fired BEFORE the engaged callback
        # so main.py's "Hi everyone" greeting can land before the WS
        # session opens — matches PHASE_8_TASK_MAP §Group scenario where
        # the multi-person announcement replaces the solo greeting.
        if new_state in (STATE_AWARE, STATE_ENGAGED):
            self._maybe_fire_multi_person(faces_snapshot)

        # Fire after lock release. Wrap in try/except so a callback raising
        # never corrupts the state machine — the next face frame will
        # still drive the next transition.
        if callback_to_fire is not None and callable(callback_to_fire):
            try:
                if new_state == STATE_ENGAGED:
                    self._invoke_on_engaged(callback_to_fire, callback_args)
                else:
                    callback_to_fire(*callback_args)
            except Exception as exc:
                self._log.exception("wake_callback_error",
                                    callback=str(callback_to_fire),
                                    err=str(exc))

        # Reset gate flags on transitions that conceptually start a fresh
        # wake cycle. Failing to reset here would let a stale speech-onset
        # flag from session N immediately wake session N+1.
        if new_state in (STATE_IDLE, STATE_LISTENING):
            self._speech_onset_flag = False
            self._keyword_flag = False

        # Phase 8: latch the multi-person callback on a per-cycle basis.
        # Reset on IDLE so a fresh AWARE entry can fire it again, and on
        # LISTENING so the same group standing in front of NAO across a
        # series of utterances doesn't repeatedly trigger the greeting.
        if new_state in (STATE_IDLE, STATE_LISTENING):
            self._multi_person_fired = False

    # ------------------------------------------------------------------
    # Internal: face polling loop (the master loop)
    # ------------------------------------------------------------------
    def _face_loop(self):
        """Poll ALFaceDetection at 30 fps and drive IDLE/AWARE transitions.

        Loop logic (see PRD §Phase 3 + PHASE_3_TASK_MAP §Wake state machine):
            * In IDLE: enter AWARE the first time a face passes
              {confidence >= face_min_conf, distance <= face_max_distance_m,
              |angle| <= face_max_angle_deg}.
            * In AWARE: keep evaluating the 5 engagement gates each frame.
              Track per-gate accumulators (gaze duration, proximity duration,
              sustained-face duration). If any gate fires -> ENGAGED.
            * If face is lost for _FACE_LOSS_TIMEOUT_S OR aware_timeout_s
              elapses with no gate firing -> IDLE silently.
            * In ENGAGED / LISTENING / SPEAKING: face polling continues
              (we still want telemetry on whether the user is in frame),
              but we don't drive transitions from this loop. main.py /
              the WS client own those via set_state().
        """
        # Per-frame timers used in AWARE.
        aware_entered_at = 0.0
        gaze_started_at = None
        proximity_started_at = None
        sustained_started_at = None
        last_face_seen_at = 0.0

        while not self._stop_event.is_set():
            cycle_start = time.time()
            faces = self._read_faces()
            picked = self._pick_face(faces)

            cur = self.current_state()

            # Phase 8: keep ``_last_faces`` populated every tick so the
            # ``current_faces`` property + multi-person callback see a
            # fresh snapshot. We always assign (even when ``faces`` is
            # empty) so a transient detection dropout isn't reported as a
            # stale crowd.
            with self._state_lock:
                self._last_faces = list(faces or [])

            if picked is not None:
                self._last_face = picked
                last_face_seen_at = cycle_start

            # ── IDLE: look for trigger condition ─────────────────
            if cur == STATE_IDLE:
                if picked is not None and self._idle_trigger_met(picked):
                    self._transition(STATE_AWARE, source="face")
                    aware_entered_at = cycle_start
                    gaze_started_at = None
                    proximity_started_at = None
                    sustained_started_at = None
                    last_face_seen_at = cycle_start

            # ── AWARE: evaluate engagement gates ─────────────────
            elif cur == STATE_AWARE:
                # Face-loss / timeout watchers first — both take us back to IDLE.
                face_lost = (
                    picked is None
                    and (cycle_start - last_face_seen_at) > _FACE_LOSS_TIMEOUT_S
                )
                aware_timeout = (
                    cycle_start - aware_entered_at
                ) >= self._aware_timeout_s

                if face_lost or aware_timeout:
                    self._transition(STATE_IDLE,
                                     source="face_lost" if face_lost
                                     else "aware_timeout")
                    continue

                if picked is None:
                    # Brief detection dropout; stay AWARE, reset gate timers
                    # that depend on continuous face presence.
                    gaze_started_at = None
                    sustained_started_at = None
                    # Proximity is best-effort: a one-frame dropout shouldn't
                    # reset a running 1.0 s proximity timer (we may still be
                    # in front of the camera; the detector just blinked).
                    if self._speech_onset_flag:
                        self._fire_engagement(GATE_SPEECH)
                        continue
                    if self._keyword_flag:
                        self._fire_engagement(GATE_KEYWORD)
                        continue
                    elapsed_sleep = time.time() - cycle_start
                    if elapsed_sleep < _FACE_POLL_INTERVAL_S:
                        time.sleep(_FACE_POLL_INTERVAL_S - elapsed_sleep)
                    continue

                # 1) Mutual gaze gate.
                if self._is_gazing(picked):
                    if gaze_started_at is None:
                        gaze_started_at = cycle_start
                    if (cycle_start - gaze_started_at) >= self._gaze_required_s:
                        self._fire_engagement(GATE_MUTUAL_GAZE)
                        continue
                else:
                    gaze_started_at = None

                # 2) Proximity gate (< 1.0 m for proximity_required_s).
                #    The contract specifies hard-coded 1.0 m here, distinct
                #    from face_max_distance_m which gates IDLE entry.
                if float(picked.get("distance_m", 9.99) or 9.99) < 1.0:
                    if proximity_started_at is None:
                        proximity_started_at = cycle_start
                    if (cycle_start - proximity_started_at) >= self._proximity_required_s:
                        self._fire_engagement(GATE_PROXIMITY)
                        continue
                else:
                    proximity_started_at = None

                # 3) Sustained-face gate (high conf + frontal angle for X s).
                if (
                    float(picked.get("confidence", 0.0) or 0.0) >= self._sustained_conf
                    and abs(float(picked.get("yaw_deg", 0.0) or 0.0)) <= self._sustained_angle_deg
                    and abs(float(picked.get("pitch_deg", 0.0) or 0.0)) <= self._sustained_angle_deg
                ):
                    if sustained_started_at is None:
                        sustained_started_at = cycle_start
                    if (cycle_start - sustained_started_at) >= self._sustained_required_s:
                        self._fire_engagement(GATE_SUSTAINED_FACE)
                        continue
                else:
                    sustained_started_at = None

                # 4) Speech onset gate.
                if self._speech_onset_flag:
                    self._fire_engagement(GATE_SPEECH)
                    continue

                # 5) Keyword gate.
                if self._keyword_flag:
                    self._fire_engagement(GATE_KEYWORD)
                    continue

            # ── ENGAGED / LISTENING / SPEAKING: telemetry only ────
            # (state moves out of these via set_state from main.py)

            elapsed = time.time() - cycle_start
            if elapsed < _FACE_POLL_INTERVAL_S:
                time.sleep(_FACE_POLL_INTERVAL_S - elapsed)

    # ------------------------------------------------------------------
    # Internal: gate-firing helper
    # ------------------------------------------------------------------
    def _fire_engagement(self, gate_name):
        """Promote AWARE -> ENGAGED with telemetry.

        Wraps ``_transition`` so each gate-firing site stays a one-liner
        in the face loop. The face loop guarantees we're in AWARE when
        this is called, but we double-check under the lock to handle a
        race where set_state forced us out of AWARE between checks.
        """
        with self._state_lock:
            if self._state != STATE_AWARE:
                return
        self._transition(STATE_ENGAGED, source="gate", gate=gate_name)

    # ------------------------------------------------------------------
    # Phase 8 helpers: multi-person callback + on_engaged signature probe
    # ------------------------------------------------------------------
    def _maybe_fire_multi_person(self, faces_snapshot):
        """Fire ``multi_person_callback`` once per cycle when a crowd is here.

        "Crowd" = ≥ 2 faces with ``distance_m`` ≤ ``multi_person_distance_m``
        (faces with unknown distance — distance_m == 0 — are conservatively
        excluded so noisy detections don't trigger a false greeting).

        The callback runs on the face-loop thread; long-running work
        belongs on a daemon spawned inside the callback body, not here.
        Errors are logged + swallowed so a buggy callback never corrupts
        the wake cycle.
        """
        callback = self._multi_person_callback
        if callback is None or not callable(callback):
            return
        if not faces_snapshot:
            return
        # Latch — fire at most once per wake cycle. Reset on IDLE entry.
        if self._multi_person_fired:
            return

        nearby = []
        for face in faces_snapshot:
            try:
                distance_m = float(face.get("distance_m", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if distance_m <= 0.0:
                # Unknown distance — exclude. The fallback ALMemory parser
                # uses size-based estimation so 0.0 means "no geometry".
                continue
            if distance_m <= self._multi_person_distance_m:
                nearby.append(dict(face))

        if len(nearby) < 2:
            return

        self._multi_person_fired = True
        self._log.info("multi_person_detected",
                       count=len(nearby),
                       distance_threshold_m=self._multi_person_distance_m)
        try:
            callback(nearby)
        except Exception as exc:
            self._log.exception("multi_person_callback_error",
                                err=str(exc))

    def _resolve_returning_user_hint(self, face):
        """Build the optional ``returning_user_hint`` for ``on_engaged``.

        Resolution order:
            1. Caller-supplied ``returning_user_resolver(face_id)`` — wins
               when it returns a truthy value (string OR dict). main.py is
               expected to wire this against ``brain_cache``.
            2. The ALFaceDetection ``name`` field on the live face record
               — non-empty when the face has been previously learned.
            3. ``None`` — first-time / unknown user. The on_engaged callee
               will treat this as the "no hint, run the new-user
               onboarding" signal.

        Returns whatever the resolver gave us (string / dict) OR a string
        derived from the face name OR ``None``.
        """
        face = face or {}
        face_id = ""
        try:
            face_id = str(face.get("face_id", "") or "")
        except Exception:
            face_id = ""

        resolver = self._returning_user_resolver
        if face_id and resolver is not None and callable(resolver):
            try:
                hint = resolver(face_id)
            except Exception as exc:
                self._log.warn("returning_user_resolver_error",
                               face_id=face_id, err=str(exc))
                hint = None
            if hint:
                return hint

        # Fall back to the live ALFaceDetection name field if populated.
        try:
            live_name = face.get("name", "") if isinstance(face, dict) else ""
        except Exception:
            live_name = ""
        if live_name:
            try:
                return str(live_name)
            except Exception:
                return None
        return None

    def _invoke_on_engaged(self, callback, args):
        """Call ``on_engaged`` with backwards-compatible arg-count probing.

        ``args`` is a 5-tuple: (face_id, gate, confidence, distance_m, face).
        We resolve a returning_user_hint from ``face`` and try the new
        5-arg signature first, falling back to 4-arg on TypeError so
        existing Phase 3 tests / wirings keep passing unchanged.
        """
        try:
            face_id, gate, confidence, distance_m, face = args
        except Exception:
            # Defensive: if args don't unpack we still want to try the
            # legacy 4-arg path with whatever we got.
            try:
                callback(*args)
            except Exception as exc:
                self._log.exception("on_engaged_legacy_fallback_failed",
                                    err=str(exc))
            return

        hint = self._resolve_returning_user_hint(face)
        legacy_args = (face_id, gate, confidence, distance_m)

        if hint is None:
            # Even with no hint, prefer the 5-arg signature so callers
            # that always expect 5 args (Phase 8 main.py) work without
            # a sentinel default — they explicitly pass returning_user_hint=None.
            try:
                callback(face_id, gate, confidence, distance_m, hint)
                return
            except TypeError:
                # 4-arg legacy callback (Phase 3 tests). Fall through.
                pass
            callback(*legacy_args)
            return

        # We have a hint to deliver. Try the 5-arg signature first.
        try:
            callback(face_id, gate, confidence, distance_m, hint)
            return
        except TypeError as exc:
            # Distinguish a real signature mismatch (4-arg callback) from
            # an unrelated TypeError raised inside the callback body. We
            # can't perfectly tell the two apart on Py2.7 without the
            # inspect module, but the message contains "argument" only on
            # signature errors thrown by Python itself.
            err_text = str(exc)
            if "argument" not in err_text and "given" not in err_text:
                # Unrelated — re-raise so the outer handler logs it.
                raise
            self._log.debug("on_engaged_legacy_signature",
                            note="callback rejected returning_user_hint",
                            err=err_text)
            callback(*legacy_args)

    # ------------------------------------------------------------------
    # Internal: VAD speech-onset watcher
    # ------------------------------------------------------------------
    def _vad_loop(self):
        """Watch AdaptiveVad (Phase 2) for speech onset.

        AdaptiveVad fires ``on_speech_start`` callbacks; we set the flag
        when we're in IDLE/AWARE so the face loop can promote to ENGAGED.
        We also clear the flag when we leave AWARE so a stray onset
        during LISTENING doesn't pollute the next session's IDLE -> AWARE.

        The simplest contract here is to install our own ``on_speech_start``
        callback on the VAD instance — but since AdaptiveVad takes the
        callback as a kwarg to ``run()``, not at construction time, we
        instead poll its internal "utterance_active" state if the caller
        has hooked us into a running VAD. Polling beats reaching into a
        protected attribute on Phase 2's class.
        """
        # The AdaptiveVad in nao/audio_handler.py exposes a ``thresholds()``
        # snapshot but no clean "is currently in speech" public read. We
        # accept either a pre-installed flag setter on the vad object
        # (``vad._wake_state_speech_callback``) or fall back to introspection.
        # Production wiring (planned in main-rewire worktree) installs a
        # speech-start callback that flips ``self._speech_onset_flag``
        # directly. Both paths produce the same outward behaviour.
        if self._vad is None:
            return
        # If the VAD lets us register a hook, prefer that.
        if hasattr(self._vad, "_wake_state_install_hook"):
            try:
                self._vad._wake_state_install_hook(self._on_speech_onset)
            except Exception as exc:
                self._log.warn("vad_hook_install_failed", err=str(exc))

        # Fall-back polling so we still function if the hook isn't there.
        last_check = 0.0
        check_interval = 0.05
        while not self._stop_event.is_set():
            now = time.time()
            if (now - last_check) >= check_interval:
                last_check = now
                # Exposed knob: callers can drop a sentinel attribute on the
                # VAD when they detect speech start outside this loop.
                if getattr(self._vad, "_wake_speech_seen", False):
                    self._on_speech_onset()
                    try:
                        self._vad._wake_speech_seen = False
                    except Exception:
                        pass
            self._stop_event.wait(timeout=check_interval)

    def _on_speech_onset(self):
        """Called when AdaptiveVad detects speech onset.

        Public for testing — safe to call from any thread. Only meaningful
        when we're in IDLE or AWARE; we ignore otherwise so recording
        ourselves talking doesn't loop back into a wake.
        """
        cur = self.current_state()
        if cur in (STATE_IDLE, STATE_AWARE):
            self._speech_onset_flag = True
            # In IDLE we still need to enter AWARE first so the gate path
            # is consistent. The face loop will catch it on the next cycle.

    # ------------------------------------------------------------------
    # Internal: keyword fallback watcher
    # ------------------------------------------------------------------
    def _keyword_loop(self):
        """Poll the wake_listener fallback for the keyword gate.

        ``fallback_word_listener`` is expected to expose a ``check()``
        method that returns a truthy value when the keyword fired since
        the last call, falsy otherwise. We don't reach into the existing
        ``listen_for_command`` blocking loop (it's incompatible with
        Phase 3's continuous-perception model); instead, main.py wraps
        the keyword path in a thin polling object that exposes ``check()``
        and ``stop()``. If the caller passes the legacy ``listen_for_command``
        directly we still tolerate it (calling it is a no-op poll).
        """
        if self._fallback_word_listener is None:
            return
        check = getattr(self._fallback_word_listener, "check", None)
        if not callable(check):
            self._log.warn("keyword_listener_missing_check")
            return
        while not self._stop_event.is_set():
            try:
                hit = bool(check())
            except Exception as exc:
                self._log.warn("keyword_check_error", err=str(exc))
                hit = False
            if hit:
                cur = self.current_state()
                if cur in (STATE_IDLE, STATE_AWARE):
                    self._keyword_flag = True
                    # In IDLE the face loop transitions to AWARE first if
                    # a face is in frame; if NOT, the keyword still wakes
                    # us — see _read_faces / _idle_trigger_met which
                    # honour the keyword flag as a soft override.
                    if cur == STATE_IDLE:
                        # Keyword fallback can wake us with no face (per
                        # PRD: "for occluded users / lighting failures").
                        self._transition(STATE_AWARE, source="keyword")
                        # Falling through — face loop will pick up the
                        # keyword flag on its next iteration and engage.
            self._stop_event.wait(timeout=_KEYWORD_POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # Touch gate: poll the head tactile sensors via ALMemory at 10 Hz.
    # A touch on any of the three head pads is a deterministic engagement
    # signal that bypasses face detection entirely — useful when the user
    # is right in front of the robot but ALFaceDetection hasn't warmed up
    # yet, or in poor lighting where face detection won't fire at all.
    # Touch promotes IDLE -> AWARE -> ENGAGED in one shot.
    # ------------------------------------------------------------------
    def _touch_loop(self):
        if self._memory_proxy is None:
            self._log.warn("touch_loop_skipped",
                           reason="no_memory_proxy")
            return
        import sys as _sys
        print("[wake_state] touch loop active (head tactile sensors)")
        _sys.stderr.flush()
        # Edge-detect: only fire on a 0->1 transition so a held touch
        # doesn't spam engagements every poll.
        prev_touched = False
        while not self._stop_event.is_set():
            touched = False
            touched_key = None
            for key in _HEAD_TOUCH_KEYS:
                try:
                    val = self._memory_proxy.getData(key)
                except Exception:
                    continue
                # NAOqi returns 1.0 for touched, 0.0 otherwise. Coerce
                # tolerantly: any truthy non-zero counts.
                try:
                    if float(val) >= 0.5:
                        touched = True
                        touched_key = key
                        break
                except Exception:
                    if val:
                        touched = True
                        touched_key = key
                        break
            if touched and not prev_touched:
                cur = self.current_state()
                if touched_key == "RearTactilTouched":
                    print("[wake_state] rear head touch stop "
                          "(state={0})".format(cur))
                    _sys.stderr.flush()
                    if self._on_barge is not None:
                        try:
                            try:
                                self._on_barge(touched_key)
                            except TypeError:
                                self._on_barge()
                        except Exception as exc:
                            self._log.warn("barge_callback_failed",
                                           err=str(exc))
                    prev_touched = touched
                    self._stop_event.wait(timeout=0.1)
                    continue
                if cur in (STATE_IDLE, STATE_AWARE):
                    print("[wake_state] head touch detected → engage")
                    _sys.stderr.flush()
                    # Promote through AWARE if needed, then ENGAGED.
                    if cur == STATE_IDLE:
                        self._transition(STATE_AWARE, source="touch")
                    self._transition(STATE_ENGAGED, source="gate",
                                     gate=GATE_TOUCH)
                else:
                    # Touch in any post-engagement state (ENGAGED,
                    # LISTENING, SPEAKING) → fire barge. The WS-driven
                    # architecture doesn't transition us into
                    # STATE_SPEAKING reliably (TTS playback runs on a
                    # separate thread that the WSM doesn't observe), so
                    # we can't gate on STATE_SPEAKING alone — we'd
                    # never fire. The barge callback in main.py
                    # short-circuits to no-op if nothing is actually
                    # playing, so over-firing is safe.
                    print("[wake_state] head touch during conversation "
                          "(state={0}, key={1}) → barge".format(
                              cur, touched_key))
                    _sys.stderr.flush()
                    if self._on_barge is not None:
                        try:
                            try:
                                self._on_barge(touched_key)
                            except TypeError:
                                # Legacy callback signature.
                                self._on_barge()
                        except Exception as exc:
                            self._log.warn("barge_callback_failed",
                                           err=str(exc))
            prev_touched = touched
            self._stop_event.wait(timeout=0.1)

    # ------------------------------------------------------------------
    # Internal: face I/O helpers
    # ------------------------------------------------------------------
    def _read_faces(self):
        """Return the latest list of faces from face_naoqi (or fallback)."""
        if _detect_faces_with_geometry is not None and self._face_detection is not None and self._memory_proxy is not None:
            try:
                return _detect_faces_with_geometry(
                    self._face_detection, self._memory_proxy
                )
            except Exception as exc:
                self._log.warn("face_detect_error", err=str(exc))
                return []
        # Fallback: direct ALMemory read.
        return _fallback_detect_faces(self._memory_proxy)

    def _pick_face(self, faces):
        """Pick the closest / highest-confidence face."""
        if not faces:
            return None
        if _closest_face is not None:
            try:
                return _closest_face(faces)
            except Exception:
                pass
        return _fallback_closest(faces)

    def _is_gazing(self, face):
        """Mutual-gaze check."""
        if _is_mutually_gazing is not None:
            try:
                return _is_mutually_gazing(face)
            except Exception:
                return False
        return _fallback_is_mutually_gazing(face)

    def _idle_trigger_met(self, face):
        """Return True if this face is "good enough" to enter AWARE."""
        try:
            conf = float(face.get("confidence", 0.0) or 0.0)
            distance_m = float(face.get("distance_m", 9.99) or 9.99)
            yaw = abs(float(face.get("yaw_deg", 0.0) or 0.0))
            pitch = abs(float(face.get("pitch_deg", 0.0) or 0.0))
        except Exception:
            return False
        if conf < self._face_min_conf:
            return False
        if distance_m > self._face_max_distance_m:
            return False
        # Angle check - either yaw or pitch over the cap rejects.
        if yaw > self._face_max_angle_deg or pitch > self._face_max_angle_deg:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal: subscription lifecycle
    # ------------------------------------------------------------------
    def _setup_face_subscription(self):
        """Subscribe to ALFaceDetection at ~30 fps.

        Skipped entirely when running off-robot (no naoqi) or when the
        caller passed pre-built proxies (test path). The subscriber name
        is unique per instance so multiple WakeStateMachines (e.g.
        recovered after a stop/start) don't collide.
        """
        if ALProxy is None:
            return
        if self._face_detection is not None and self._memory_proxy is not None:
            # Caller injected proxies; assume they manage subscription.
            return
        try:
            if self._memory_proxy is None:
                self._memory_proxy = ALProxy("ALMemory", self._nao_ip, self._nao_port)
            if self._face_detection is None:
                self._face_detection = ALProxy(
                    "ALFaceDetection", self._nao_ip, self._nao_port
                )
            # Best-effort unsubscribe first — if the prior main.py died
            # without cleaning up, NAOqi will reject the next subscribe()
            # under the same name and ALFaceDetection stops writing to
            # ALMemory at all. Ignoring this exception is critical.
            try:
                self._face_detection.unsubscribe(self._face_subscriber_name)
                import sys as _sys
                print("[wake_state] cleaned up stale face subscription "
                      "'{0}'".format(self._face_subscriber_name))
                _sys.stderr.flush()
            except Exception:
                pass
            try:
                # 100 ms period -> ~10 events/s from ALFaceDetection; we
                # poll faster (33 ms) so the wake transition is not gated
                # by ALMemory write latency.
                self._face_detection.subscribe(self._face_subscriber_name, 100, 0.0)
                self._owns_face_subscription = True
                import sys as _sys
                print("[wake_state] face subscription active name='{0}' "
                      "period=100ms".format(self._face_subscriber_name))
                _sys.stderr.flush()
            except Exception as exc:
                # Subscribe still failed even after the cleanup — try a
                # process-unique name as a last resort so we definitely
                # get our own ALMemory writer attached.
                self._log.warn("face_subscribe_warn", err=str(exc))
                import os as _os
                fallback_name = "WakeFaceDetection_{0}".format(_os.getpid())
                try:
                    self._face_detection.subscribe(fallback_name, 100, 0.0)
                    self._face_subscriber_name = fallback_name
                    self._owns_face_subscription = True
                    import sys as _sys
                    print("[wake_state] face subscription active under "
                          "fallback name='{0}'".format(fallback_name))
                    _sys.stderr.flush()
                except Exception as exc2:
                    self._log.error("face_subscribe_fallback_failed",
                                    err=str(exc2))
                    import sys as _sys
                    print("[wake_state] FACE SUBSCRIBE FAILED: {0}; fallback "
                          "also failed: {1}".format(exc, exc2))
                    _sys.stderr.flush()
        except Exception as exc:
            self._log.error("face_proxy_init_failed", err=str(exc))
            self._face_detection = None
            self._memory_proxy = None

    def _teardown_face_subscription(self):
        """Unsubscribe ALFaceDetection if we own the subscription."""
        if not self._owns_face_subscription:
            return
        try:
            if self._face_detection is not None:
                self._face_detection.unsubscribe(self._face_subscriber_name)
        except Exception as exc:
            self._log.warn("face_unsubscribe_warn", err=str(exc))
        self._owns_face_subscription = False

    # ------------------------------------------------------------------
    # Internal: led helper
    # ------------------------------------------------------------------
    def _safe_led_call(self, leds, method_name):
        """Call ``leds.<method_name>()`` if leds exposes it; never raise."""
        if leds is None:
            return
        method = getattr(leds, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except Exception as exc:
            self._log.warn("leds_call_failed",
                           method=method_name, err=str(exc))


# ── Self-test: synthetic face detection driving IDLE -> AWARE -> ENGAGED -> LISTENING ─
class _FakeLeds(object):
    """Minimal LedDriver substitute that records method calls.

    Used by the __main__ self-test so we can verify each state transition
    fires its expected LED hook (set_idle / set_aware / chime / set_engaged
    / set_listening / set_speaking).
    """

    def __init__(self):
        self.calls = []

    def _record(self, name):
        self.calls.append((name, time.time()))
        print("[FakeLeds] {0}".format(name))

    def set_idle(self):
        self._record("set_idle")

    def set_aware(self):
        self._record("set_aware")

    def set_engaged(self):
        self._record("set_engaged")

    def set_listening(self):
        self._record("set_listening")

    def set_speaking(self):
        self._record("set_speaking")

    def chime(self):
        self._record("chime")


class _FakeMemoryProxy(object):
    """Stand-in for ALProxy("ALMemory") that returns scripted FaceDetected payloads.

    The state machine's fallback path reads ALMemory("FaceDetected"). We
    provide a small queue of scripted detections so we can drive the test
    deterministically without real naoqi.
    """

    def __init__(self):
        self._payload = None

    def set_face(self, confidence=0.6, distance_m=0.7, yaw=0.0, pitch=0.0,
                 face_id="self_test_face"):
        # Build the ALFaceDetection-shaped list.
        # Layout matches what _fallback_detect_faces parses:
        #   [time_block, [ [shape_info_unused, [face_id, confidence, name, ...]] ] ]
        size_x_for_distance = max(0.05, min(0.5, 0.14 / distance_m))
        shape_info = [0.0, 0.0, size_x_for_distance, size_x_for_distance]
        extra_info = [face_id, float(confidence), "Test User"]
        # Keep yaw/pitch in instance state so a wrapper test reader can
        # inject angled faces via _yaw / _pitch (the fallback parser only
        # reads from the shape block which doesn't carry yaw/pitch). The
        # self-test below patches _detect_faces_with_geometry to honour
        # these when we explicitly want non-frontal faces.
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self._face_id = face_id
        self._confidence = float(confidence)
        self._distance = float(distance_m)
        self._payload = [
            [0.0, 0.0],
            [[shape_info, extra_info]],
        ]

    def clear(self):
        self._payload = None

    def getData(self, key):
        if key == "FaceDetected":
            return self._payload
        return None


class _ScriptedFaceReader(object):
    """Test seam that returns whatever face dict was last set.

    Replaces ``WakeStateMachine._read_faces`` for the self-test so we can
    test each engagement gate independently. The fallback ALMemory parser
    is exercised separately by the regular parser tests; here we want to
    drive the state machine's own logic.

    Phase 8: also accepts a list of faces via ``set_many`` so the
    multi-person callback can be exercised against a synthetic crowd.
    """

    def __init__(self):
        self._faces = []

    def set(self, face):
        self._faces = [face] if face is not None else []

    def set_many(self, faces):
        self._faces = list(faces or [])

    def clear(self):
        self._faces = []

    def read(self):
        if not self._faces:
            return []
        # Return copies so the state machine can mutate without us caring.
        return [dict(f) for f in self._faces]


def _build_wsm(reader, leds, **overrides):
    """Construct a WakeStateMachine wired up with the scripted reader."""
    transitions = []

    def on_engaged(face_id, gate, conf, dist_m):
        transitions.append(("engaged", face_id, gate, conf, dist_m))
        print("[on_engaged] face={0} gate={1} conf={2} dist={3}".format(
            face_id, gate, conf, dist_m))

    def on_lost():
        transitions.append(("lost",))
        print("[on_lost]")

    def on_listening():
        transitions.append(("listening",))
        print("[on_listening]")

    def on_speaking_done():
        transitions.append(("speaking_done",))
        print("[on_speaking_done]")

    kwargs = dict(
        nao_ip="127.0.0.1", nao_port=9559,
        leds=leds,
        fallback_word_listener=None,
        on_engaged=on_engaged,
        on_lost=on_lost,
        on_listening=on_listening,
        on_speaking_done=on_speaking_done,
        face_min_conf=0.35,
        face_max_distance_m=1.5,
        face_max_angle_deg=60.0,
        aware_timeout_s=2.0,
        gaze_required_s=0.3,
        proximity_required_s=0.3,
        sustained_conf=0.5,
        sustained_required_s=0.4,
        sustained_angle_deg=30.0,
    )
    kwargs.update(overrides)
    wsm = WakeStateMachine(**kwargs)
    # Patch the face reader so the loop reads from our scripted reader.
    wsm._read_faces = reader.read  # type: ignore[assignment]
    return wsm, transitions


def _self_test():
    """Walk synthesized face streams through every documented transition.

    Drives, in order:
        1. IDLE -> AWARE via a frontal face entering the gate window.
        2. AWARE -> ENGAGED via the proximity gate (mutual gaze disabled
           by giving the face a yaw outside gaze tolerance).
        3. ENGAGED -> LISTENING via external set_state.
        4. LISTENING -> SPEAKING -> LISTENING via external set_state
           round-trip (validates SPEAKING -> LISTENING fires
           on_speaking_done while a SPEAKING entry from LISTENING does
           not fire on_speaking_done).
        5. Fresh AWARE entry that times out -> IDLE silently
           (validates on_lost callback).
    """
    print("=== WakeStateMachine self-test (no naoqi) ===")

    leds = _FakeLeds()
    reader = _ScriptedFaceReader()
    wsm, transitions = _build_wsm(reader, leds)

    runner = threading.Thread(target=wsm.start, name="self-test-wsm")
    runner.daemon = True
    runner.start()
    time.sleep(0.15)

    print("--- step 1: IDLE -> AWARE via face entering ---")
    # yaw 25 deg pushes us outside gaze tolerance (15 deg) so the gaze
    # gate doesn't fire instantly; we want to test proximity first.
    reader.set({
        "face_id": "self_test_face",
        "name": "Test User",
        "confidence": 0.6,
        "distance_m": 0.7,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    time.sleep(0.2)
    print("state after face:", wsm.current_state())
    assert wsm.current_state() == STATE_AWARE, (
        "expected AWARE, got {0}".format(wsm.current_state()))

    print("--- step 2: AWARE -> ENGAGED via proximity gate ---")
    time.sleep(0.5)  # > proximity_required_s so the gate fires
    state_after_gate = wsm.current_state()
    print("state after proximity wait:", state_after_gate)
    assert state_after_gate == STATE_ENGAGED, (
        "expected ENGAGED after proximity gate; got {0}".format(
            state_after_gate))
    engaged_events = [t for t in transitions if t[0] == "engaged"]
    assert engaged_events, "expected an engaged callback"
    assert engaged_events[0][2] == GATE_PROXIMITY, (
        "expected gate=proximity, got {0}".format(engaged_events[0][2]))

    print("--- step 3: ENGAGED -> LISTENING via external set_state ---")
    wsm.set_state(STATE_LISTENING)
    time.sleep(0.1)
    assert wsm.current_state() == STATE_LISTENING
    listening_events = [t for t in transitions if t[0] == "listening"]
    assert listening_events, (
        "expected on_listening callback on ENGAGED -> LISTENING")

    print("--- step 4: LISTENING -> SPEAKING -> LISTENING (TTS round-trip) ---")
    wsm.set_state(STATE_SPEAKING)
    time.sleep(0.1)
    assert wsm.current_state() == STATE_SPEAKING
    wsm.set_state(STATE_LISTENING)
    time.sleep(0.1)
    assert wsm.current_state() == STATE_LISTENING
    speaking_done_events = [t for t in transitions if t[0] == "speaking_done"]
    assert speaking_done_events, (
        "expected on_speaking_done callback after SPEAKING -> LISTENING")

    print("--- step 5: SPEAKING entry should NOT fire on_speaking_done ---")
    sd_before = len(speaking_done_events)
    wsm.set_state(STATE_LISTENING)  # no-op (already LISTENING)
    time.sleep(0.05)
    sd_after = len([t for t in transitions if t[0] == "speaking_done"])
    assert sd_after == sd_before, (
        "no-op set_state should not retrigger on_speaking_done")

    print("--- step 6: AWARE -> IDLE timeout from fresh session ---")
    reader.clear()
    wsm.set_state(STATE_IDLE)
    time.sleep(0.2)
    assert wsm.current_state() == STATE_IDLE

    # Re-enter AWARE with a face that does NOT cross any gate (>1 m so no
    # proximity, yaw 25 so no gaze, conf only just over face_min_conf so
    # not sustained_conf). That gives us a clean AWARE timeout path.
    reader.set({
        "face_id": "fresh_face",
        "name": "Fresh User",
        "confidence": 0.4,
        "distance_m": 1.2,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    time.sleep(0.15)
    if wsm.current_state() == STATE_AWARE:
        print("AWARE entered for timeout test")
    # Remove the face and wait for aware_timeout_s + face_loss_timeout.
    reader.clear()
    time.sleep(2.6)
    state_after_timeout = wsm.current_state()
    print("state after AWARE timeout:", state_after_timeout)
    assert state_after_timeout == STATE_IDLE, (
        "expected IDLE after timeout; got {0}".format(state_after_timeout))
    lost_events = [t for t in transitions if t[0] == "lost"]
    assert lost_events, "expected on_lost callback after AWARE timeout"

    wsm.stop()
    runner.join(timeout=2.0)

    # ── Phase 8 additions ──────────────────────────────────────────────
    print("=== Phase 8 onboarding extensions ===")
    _phase8_multi_person_test()
    _phase8_returning_user_hint_test()
    _phase8_legacy_on_engaged_test()
    _phase8_current_face_id_test()

    print("--- summary ---")
    print("transitions: {0}".format(transitions))
    print("led calls   : {0}".format([c[0] for c in leds.calls]))
    print("=== self-test OK ===")
    return 0


# ---------------------------------------------------------------------------
# Phase 8 self-tests. Standalone so each can be run individually if a
# regression hits one specifically.
# ---------------------------------------------------------------------------


def _phase8_multi_person_test():
    """Multi-person callback fires once on >=2 faces within 1.5 m."""
    print("--- phase 8: multi_person_callback fires on >= 2 nearby faces ---")
    leds = _FakeLeds()
    reader = _ScriptedFaceReader()
    multi_calls = []

    def on_multi(faces):
        multi_calls.append(list(faces))
        print("[multi_person] count={0}".format(len(faces)))

    wsm, _ = _build_wsm(reader, leds, multi_person_callback=on_multi,
                        multi_person_distance_m=1.5)
    runner = threading.Thread(target=wsm.start, name="phase8-multi")
    runner.daemon = True
    runner.start()
    time.sleep(0.15)

    # Two faces in conversation range — should trigger.
    reader.set_many([
        {"face_id": "u1", "name": "", "confidence": 0.6,
         "distance_m": 0.7, "yaw_deg": 4.0, "pitch_deg": 0.0},
        {"face_id": "u2", "name": "", "confidence": 0.6,
         "distance_m": 0.9, "yaw_deg": -6.0, "pitch_deg": 0.0},
    ])
    # Allow the face loop (~30 fps) one or two ticks AND the AWARE
    # transition itself before checking.
    time.sleep(0.3)
    assert len(multi_calls) >= 1, (
        "expected multi_person_callback to fire on AWARE transition; "
        "got {0}".format(multi_calls))
    assert len(multi_calls[0]) == 2, multi_calls[0]
    print("[phase8] multi_person fired with {0} faces".format(
        len(multi_calls[0])))

    # Already latched — calling _maybe_fire_multi_person again must not
    # re-trigger inside the same wake cycle.
    fires_before = len(multi_calls)
    wsm._maybe_fire_multi_person(reader.read())
    assert len(multi_calls) == fires_before, (
        "multi_person_callback latch failed; refired without IDLE reset")

    # Reset the cycle (force IDLE) and verify the latch clears.
    wsm.set_state(STATE_IDLE)
    time.sleep(0.05)
    reader.set_many([
        {"face_id": "u3", "name": "", "confidence": 0.6,
         "distance_m": 0.6, "yaw_deg": 0.0, "pitch_deg": 0.0},
        {"face_id": "u4", "name": "", "confidence": 0.6,
         "distance_m": 0.8, "yaw_deg": 0.0, "pitch_deg": 0.0},
    ])
    time.sleep(0.3)
    assert len(multi_calls) >= fires_before + 1, (
        "expected multi_person_callback to refire after IDLE reset")
    print("[phase8] latch resets after IDLE")

    wsm.stop()
    runner.join(timeout=2.0)

    # Far-away crowd -> should NOT fire. New WSM so the latch is fresh.
    leds2 = _FakeLeds()
    reader2 = _ScriptedFaceReader()
    far_calls = []
    wsm2, _ = _build_wsm(reader2, leds2,
                         multi_person_callback=lambda fs: far_calls.append(fs),
                         multi_person_distance_m=1.5,
                         face_max_distance_m=4.0)  # let face_idle gate accept
    runner2 = threading.Thread(target=wsm2.start, name="phase8-multi-far")
    runner2.daemon = True
    runner2.start()
    time.sleep(0.15)
    reader2.set_many([
        {"face_id": "u5", "name": "", "confidence": 0.6,
         "distance_m": 2.5, "yaw_deg": 0.0, "pitch_deg": 0.0},
        {"face_id": "u6", "name": "", "confidence": 0.6,
         "distance_m": 3.0, "yaw_deg": 0.0, "pitch_deg": 0.0},
    ])
    time.sleep(0.3)
    assert far_calls == [], (
        "far-away crowd should not trigger multi_person; got {0}".format(
            far_calls))
    print("[phase8] far-away crowd correctly ignored")
    wsm2.stop()
    runner2.join(timeout=2.0)


def _phase8_returning_user_hint_test():
    """on_engaged receives returning_user_hint when the resolver returns one."""
    print("--- phase 8: returning_user_hint flows through on_engaged ---")
    leds = _FakeLeds()
    reader = _ScriptedFaceReader()
    engaged_calls = []

    def on_engaged_5arg(face_id, gate, conf, dist_m, returning_user_hint=None):
        engaged_calls.append((face_id, gate, conf, dist_m, returning_user_hint))
        print("[on_engaged 5-arg] hint={0}".format(returning_user_hint))

    def on_lost():
        pass

    def on_listening():
        pass

    def on_speaking_done():
        pass

    # Resolver returns a display name only when the face has been seen.
    user_db = {"face-known": "Aayush"}

    def resolver(face_id):
        return user_db.get(face_id)

    wsm = WakeStateMachine(
        nao_ip="127.0.0.1", nao_port=9559,
        leds=leds, fallback_word_listener=None,
        on_engaged=on_engaged_5arg,
        on_lost=on_lost, on_listening=on_listening,
        on_speaking_done=on_speaking_done,
        face_min_conf=0.35, face_max_distance_m=1.5,
        face_max_angle_deg=60.0,
        aware_timeout_s=2.0,
        gaze_required_s=0.3,
        proximity_required_s=0.3,
        sustained_conf=0.5,
        sustained_required_s=0.4,
        sustained_angle_deg=30.0,
        returning_user_resolver=resolver,
    )
    wsm._read_faces = reader.read
    runner = threading.Thread(target=wsm.start, name="phase8-hint")
    runner.daemon = True
    runner.start()
    time.sleep(0.15)

    reader.set({
        "face_id": "face-known",
        "name": "",
        "confidence": 0.6,
        "distance_m": 0.7,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    # Wait for AWARE then proximity gate.
    time.sleep(0.6)
    assert wsm.current_state() == STATE_ENGAGED, wsm.current_state()
    assert engaged_calls, "expected on_engaged to fire"
    last = engaged_calls[-1]
    assert last[0] == "face-known", last
    assert last[4] == "Aayush", (
        "expected returning_user_hint='Aayush', got {0!r}".format(last[4]))
    print("[phase8] returning_user_hint delivered: {0}".format(last[4]))

    wsm.stop()
    runner.join(timeout=2.0)

    # Unknown face — resolver returns None, ALFaceDetection name field
    # is also empty, so the hint should be None (new user).
    leds2 = _FakeLeds()
    reader2 = _ScriptedFaceReader()
    engaged_calls2 = []

    wsm2 = WakeStateMachine(
        nao_ip="127.0.0.1", nao_port=9559,
        leds=leds2, fallback_word_listener=None,
        on_engaged=lambda fi, g, c, d, h=None: engaged_calls2.append(
            (fi, g, c, d, h)),
        on_lost=lambda: None,
        on_listening=lambda: None,
        on_speaking_done=lambda: None,
        face_min_conf=0.35, face_max_distance_m=1.5,
        face_max_angle_deg=60.0,
        aware_timeout_s=2.0,
        gaze_required_s=0.3,
        proximity_required_s=0.3,
        sustained_conf=0.5,
        sustained_required_s=0.4,
        sustained_angle_deg=30.0,
        returning_user_resolver=resolver,
    )
    wsm2._read_faces = reader2.read
    runner2 = threading.Thread(target=wsm2.start, name="phase8-hint-new")
    runner2.daemon = True
    runner2.start()
    time.sleep(0.15)
    reader2.set({
        "face_id": "face-unknown",
        "name": "",
        "confidence": 0.6,
        "distance_m": 0.7,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    time.sleep(0.6)
    assert wsm2.current_state() == STATE_ENGAGED, wsm2.current_state()
    assert engaged_calls2, "expected new-user on_engaged to fire"
    last2 = engaged_calls2[-1]
    assert last2[4] is None, (
        "expected hint=None for unknown face; got {0!r}".format(last2[4]))
    print("[phase8] new-user hint=None as expected")
    wsm2.stop()
    runner2.join(timeout=2.0)


def _phase8_legacy_on_engaged_test():
    """4-arg legacy on_engaged keeps working (Phase 3 backwards compat)."""
    print("--- phase 8: legacy 4-arg on_engaged still fires ---")
    leds = _FakeLeds()
    reader = _ScriptedFaceReader()
    legacy_calls = []

    def legacy_on_engaged(face_id, gate, conf, dist_m):
        # Note: NO returning_user_hint kwarg/positional — strict 4-arg.
        legacy_calls.append((face_id, gate, conf, dist_m))

    # Resolver returns a hint, but the callback can't accept it.
    wsm = WakeStateMachine(
        nao_ip="127.0.0.1", nao_port=9559,
        leds=leds, fallback_word_listener=None,
        on_engaged=legacy_on_engaged,
        on_lost=lambda: None,
        on_listening=lambda: None,
        on_speaking_done=lambda: None,
        face_min_conf=0.35, face_max_distance_m=1.5,
        face_max_angle_deg=60.0,
        aware_timeout_s=2.0,
        gaze_required_s=0.3,
        proximity_required_s=0.3,
        sustained_conf=0.5,
        sustained_required_s=0.4,
        sustained_angle_deg=30.0,
        returning_user_resolver=lambda fid: "Aayush",
    )
    wsm._read_faces = reader.read
    runner = threading.Thread(target=wsm.start, name="phase8-legacy")
    runner.daemon = True
    runner.start()
    time.sleep(0.15)
    reader.set({
        "face_id": "legacy-face",
        "name": "Aayush",
        "confidence": 0.6,
        "distance_m": 0.7,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    time.sleep(0.6)
    assert wsm.current_state() == STATE_ENGAGED
    assert legacy_calls, "legacy on_engaged should still fire"
    last = legacy_calls[-1]
    assert last == ("legacy-face", GATE_PROXIMITY, 0.6, 0.7), last
    print("[phase8] legacy 4-arg on_engaged fired: {0}".format(last))
    wsm.stop()
    runner.join(timeout=2.0)


def _phase8_current_face_id_test():
    """current_face_id property reflects the latest seen face."""
    print("--- phase 8: current_face_id property ---")
    leds = _FakeLeds()
    reader = _ScriptedFaceReader()
    wsm, _ = _build_wsm(reader, leds)
    runner = threading.Thread(target=wsm.start, name="phase8-prop")
    runner.daemon = True
    runner.start()
    time.sleep(0.15)
    # Initial: no face seen.
    assert wsm.current_face_id == "", (
        "expected '' before any face; got {0!r}".format(wsm.current_face_id))
    reader.set({
        "face_id": "abc-123",
        "name": "Test",
        "confidence": 0.6,
        "distance_m": 0.7,
        "yaw_deg": 25.0,
        "pitch_deg": 0.0,
    })
    time.sleep(0.15)
    assert wsm.current_face_id == "abc-123", wsm.current_face_id
    snapshot = wsm.current_faces
    assert isinstance(snapshot, list) and len(snapshot) == 1, snapshot
    assert snapshot[0]["face_id"] == "abc-123"
    print("[phase8] current_face_id={0!r}, current_faces[0]={1!r}".format(
        wsm.current_face_id, snapshot[0]["face_id"]))
    wsm.stop()
    runner.join(timeout=2.0)


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
