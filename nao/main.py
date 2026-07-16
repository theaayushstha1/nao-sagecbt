# -*- coding: utf-8 -*-
"""NAO entry point - Phase 3 v2 rework.

Boots the structured logger, pins speaker volume, disables ALAutonomousLife,
then drives a face-first ``WakeStateMachine`` that ONLY opens the WebSocket
session once an engagement gate fires (mutual gaze, proximity, sustained
face, speech onset, or "hey nao" keyword fallback). On crash, stops audio
recorder + player and sleeps 2 s before reconnecting - same crash recovery
shape as Phase 1, just gated by wake state above the WS client.

Phase 3 contract (see docs/PHASE_3_TASK_MAP.md):
  IDLE -> AWARE (face) -> ENGAGED (gate fired) -> WS session opens.
  AWARE timeout (8 s with no gate) silently returns to IDLE; no chime,
  no TTS, no WS handshake. This is the main false-wake protection.

Defensive imports - every Phase 3 sibling module (``leds``, ``wake_state``)
and Phase 1/2 dependency (``audio_module``, ``stream_tts``, ``ws_client``,
``audio_handler``, ``wake_listener``) is guarded with try/except so this
file ``py_compile``s in isolation while sibling agents finish their work
in parallel worktrees. Real ImportError surfaces at boot as a structured
log + 2 s retry, not a crashing import at top of file.

Python 2.7 compatible - runs under naoqi on the robot.
"""
from __future__ import print_function

import os
import sys
import threading
import time
import traceback


# --- Logger first, before any naoqi import that might fail noisily.
# The logger is stdlib-only and will configure itself on first use; doing it
# explicitly here lets us route boot-time errors through the same pipeline.
from logger import configure_logger, get_logger

# Local config + utilities - these are pure python, no naoqi binding.
import config
from utils import nao_execute, user_cache  # `user_cache` doubles as the brain
                                            # cache placeholder until Phase 7
                                            # ships nao/utils/brain.py.

# naoqi proxy is robot-only. Guarded so this module can be byte-compiled and
# unit-imported on a developer laptop without naoqi installed.
try:
    from naoqi import ALProxy, ALBroker  # type: ignore  # noqa: F401
    _HAS_NAOQI = True
except ImportError:
    ALProxy = None
    ALBroker = None
    _HAS_NAOQI = False


# ---------------------------------------------------------------------------
# ALBroker singleton - REQUIRED before any ALModule subclass can be
# instantiated. Without a broker registered as Python's "current broker",
# inaoqi.module.__init__ raises "Could not create a module, as there is no
# current broker in Python's world." Phase 1 missed this; we mint it once
# here at boot and reuse for the lifetime of the process.
# ---------------------------------------------------------------------------
_BROKER = None


def _ensure_broker(nao_ip, nao_port):
    """Create an ALBroker if one isn't already registered. Idempotent."""
    global _BROKER
    if not _HAS_NAOQI or ALBroker is None:
        return None
    if _BROKER is not None:
        return _BROKER
    # ("name", listen_ip, listen_port=0 -> auto, parent_ip, parent_port).
    # listen_ip "0.0.0.0" lets the broker bind any local interface.
    _BROKER = ALBroker("nao_assist_broker", "0.0.0.0", 0,
                       nao_ip, int(nao_port))
    return _BROKER


# ---------------------------------------------------------------------------
# Boot helpers preserved from Phase 1 main.py - volume pinning, autonomous
# life shutdown, and crash recovery teardown. Phase 3 keeps these verbatim;
# the only new thing is the wake state machine sitting above the WS client.
# ---------------------------------------------------------------------------


def _set_volume(ip, port, level=100):
    """Pin NAO's master speaker output high so OpenAI TTS MP3 is audible
    in a noisy classroom. setOutputVolume takes 0-100. Also nudge the
    input gain so the mic actually picks up the user's voice.
    Best-effort.
    """
    if not _HAS_NAOQI:
        return
    try:
        ad = ALProxy("ALAudioDevice", ip, port)
    except Exception as exc:
        print("[volume] ALAudioDevice proxy failed:", exc)
        ad = None
    if ad is not None:
        try:
            ad.setOutputVolume(int(level))
        except Exception as exc:
            print("[volume] setOutputVolume failed:", exc)
        # Mic input gain — some NAO V6 ship with input volume at 0 which
        # makes ALAudioRecorder capture pure silence. setInputVolume is the
        # NAOqi 2.5+ API; older firmware uses the same call.
        try:
            ad.setInputVolume(int(level))
            print("[volume] mic input volume set to {0}".format(level))
        except Exception as exc:
            print("[volume] setInputVolume failed (older firmware?):", exc)
    # Keep NAO's local ALTextToSpeech permanently muted. The robot's native
    # kid voice clashes with the ElevenLabs streaming TTS (dual-voice on
    # every reply, and on connection-error filler phrases). All "real"
    # speech comes from server-streamed MP3 played through the audio
    # player; ALTextToSpeech is only used for short announcer fillers,
    # which we silence here so they never leak.
    try:
        ALProxy("ALTextToSpeech", ip, port).setVolume(0.0)
    except Exception:
        pass


def _disable_autonomous(ip, port):
    """Kill NAO's built-in autonomous life so it doesn't talk over us.
    setAutonomousAbilityEnabled persists across reboots; setState is per
    session. Best-effort, swallows naoqi exceptions.
    """
    if not _HAS_NAOQI:
        return
    abilities = [
        "AutonomousBlinking",
        "BackgroundMovement",
        "BasicAwareness",
        "ListeningMovement",
        "SpeakingMovement",
    ]
    try:
        al = ALProxy("ALAutonomousLife", ip, port)
        for a in abilities:
            try:
                al.setAutonomousAbilityEnabled(a, False)
            except Exception:
                pass
        try:
            al.setState("disabled")
        except Exception:
            pass
    except Exception:
        pass
    for svc, calls in [
        ("ALBasicAwareness", [("stopAwareness", [])]),
        ("ALAutonomousMoves", [("setBackgroundStrategy", ["none"]),
                               ("setExpressiveListeningEnabled", [False])]),
        ("ALSpeakingMovement", [("setEnabled", [False])]),
    ]:
        try:
            p = ALProxy(svc, ip, port)
            for method, args in calls:
                try:
                    getattr(p, method)(*args)
                except Exception:
                    pass
        except Exception:
            pass


def _stop_audio_proxies(ip, port):
    """Crash-recovery teardown - make sure the recorder + player are quiet
    before the next reconnect attempt. Same shape as the old conversation
    loop's except-branch.
    """
    if not _HAS_NAOQI:
        return
    try:
        ALProxy("ALAudioRecorder", ip, port).stopMicrophonesRecording()
    except Exception:
        pass
    try:
        ALProxy("ALAudioPlayer", ip, port).stopAll()
    except Exception:
        pass


def _stop_motion_behaviors_safe(log, reason="manual_stop"):
    """Best-effort hard stop for long Choregraphe behaviors and motion tasks."""
    stopped_behaviors = []
    if not _HAS_NAOQI:
        return stopped_behaviors
    try:
        mgr = ALProxy("ALBehaviorManager", config.NAO_IP, config.NAO_PORT)
        try:
            running = mgr.getRunningBehaviors() or []
        except Exception:
            running = []
        for behavior in running:
            try:
                mgr.stopBehavior(behavior)
                stopped_behaviors.append(behavior)
            except Exception:
                pass
        try:
            mgr.stopAllBehaviors()
        except Exception:
            pass
    except Exception as exc:
        try:
            log.debug("motion_behavior_stop_failed", error=str(exc))
        except Exception:
            pass

    try:
        motion = ALProxy("ALMotion", config.NAO_IP, config.NAO_PORT)
        for method in ("stopMove", "killAll"):
            try:
                fn = getattr(motion, method, None)
                if callable(fn):
                    fn()
            except Exception:
                pass
    except Exception as exc:
        try:
            log.debug("motion_task_stop_failed", error=str(exc))
        except Exception:
            pass

    try:
        log.info("motion_stop_requested",
                 reason=reason, stopped_behaviors=stopped_behaviors)
    except Exception:
        pass
    return stopped_behaviors


# ---------------------------------------------------------------------------
# Component factories. Imports are deferred so this file ``py_compile``s
# even while sibling agents are still authoring ``nao/wake_state.py`` and
# ``nao/leds.py`` in parallel worktrees. On the robot, these modules will
# be present after the Phase 3 consolidator merges.
# ---------------------------------------------------------------------------


def _build_audio_streamer(log):
    """Construct the ALAudioDevice subscriber. Owned by Phase 1 sibling
    ``nao-audio-module``. Constructed but NOT started here - ``start()`` is
    deferred to ``on_engaged`` so the mic only subscribes once a wake gate
    fires (saves ~1% CPU and avoids surfacing user audio before consent).
    """
    from audio_module import NaoAudioStreamer  # Phase 1 sibling
    return NaoAudioStreamer(
        broker_ip="0.0.0.0",
        broker_port=0,
        nao_ip=config.NAO_IP,
        nao_port=config.NAO_PORT,
        name="NaoAudioStream",
    )


def _build_tts_player(log):
    """Construct the streaming TTS chunk player. Owned by Phase 1 sibling
    ``nao-stream-tts``. Cheap to construct (no naoqi handles touched until
    first ``enqueue``).
    """
    from stream_tts import StreamTtsPlayer  # Phase 1 sibling
    return StreamTtsPlayer(config.NAO_IP)


def _build_adaptive_vad(log):
    """Construct the Phase 2 adaptive VAD. The wake state machine queries
    this for the "speech onset" engagement gate; once a session is open we
    swap in the live ``ws_client`` so EoU hints flow as control frames.
    """
    from audio_handler import AdaptiveVad  # Phase 2 sibling
    return AdaptiveVad(ws_client=None)


def _build_led_driver(log):
    """Construct the Phase 3 LED driver. Owned by sibling ``led-driver``
    (``nao/leds.py``). Cheap to construct - no naoqi calls until first
    ``fade()``/``pulse()``.
    """
    from leds import LedDriver  # Phase 3 sibling
    return LedDriver(config.NAO_IP, config.NAO_PORT)


def _build_wake_listener(log):
    """Resolve the keyword fallback listener.

    The Phase 3 task map references ``wake_listener.WakeListener`` as the
    expected class. The current ``nao/wake_listener.py`` exposes a
    procedural ``listen_for_command(nao_ip, port)`` instead. We probe both
    so this file works whether the wake-state-machine sibling sticks with
    the procedural API or wraps it in a class.

    Returns whichever symbol exists, or ``None`` if neither is importable
    (in which case the WSM disables its keyword gate cleanly).
    """
    try:
        import wake_listener  # Phase 3 reused-as-is sibling
    except Exception as exc:
        log.warn("wake_listener_import_failed", error=str(exc))
        return None
    cls = getattr(wake_listener, "WakeListener", None)
    if cls is not None:
        try:
            return cls(config.NAO_IP, config.NAO_PORT)
        except Exception as exc:
            log.warn("wake_listener_construct_failed", error=str(exc))
    fn = getattr(wake_listener, "listen_for_command", None)
    if fn is not None:
        # Hand back the module itself - the WSM can call either symbol it
        # finds. Module objects expose attribute access just like an
        # instance, so the sibling's "fallback_word_listener.listen()"
        # contract is satisfied either way.
        return wake_listener
    return None


def _build_action_dispatcher(log):
    """Construct a `(name, args) -> dispatch(...)` callable with naoqi
    proxies (motion, posture, leds, behav_mgr, tts) baked in. Without
    this, ws_client called nao_execute.dispatch with all proxies = None,
    so body actions like `follow_movement` and `play_animation` would
    silently no-op while TTS still played the ack ("Following you now."
    with no actual following).
    """
    if not _HAS_NAOQI:
        log.info("action_dispatcher_skipped_no_naoqi")
        # Stub: still callable but a no-op, so callers don't need to
        # branch on dev-box vs robot.
        return lambda name, args=None: None

    ip, port = config.NAO_IP, config.NAO_PORT
    proxies = {}
    for key, svc in (
        ("motion",   "ALMotion"),
        ("posture",  "ALRobotPosture"),
        ("leds",     "ALLeds"),
        ("behav_mgr", "ALBehaviorManager"),
        ("tts",      "ALTextToSpeech"),
    ):
        try:
            proxies[key] = ALProxy(svc, ip, port)
        except Exception as exc:
            log.warn("action_proxy_failed", svc=svc, error=str(exc))
            proxies[key] = None

    def _dispatcher(name, args=None):
        try:
            return nao_execute.dispatch(
                name, args or {},
                motion=proxies.get("motion"),
                posture=proxies.get("posture"),
                leds=proxies.get("leds"),
                behav_mgr=proxies.get("behav_mgr"),
                tts=proxies.get("tts"),
            )
        except Exception as exc:
            log.error("action_dispatch_error", name=name, error=str(exc))
            return False

    log.info("action_dispatcher_ready",
             have=sorted(k for k, v in proxies.items() if v is not None))
    return _dispatcher


def _build_ws_client(log, audio, tts, brain):
    """Construct the long-lived WS client. Owned by Phase 1 sibling
    ``nao-ws-client``. We build a fresh instance per ENGAGED transition so
    a torn-down session doesn't carry state into the next one.
    """
    from ws_client import NaoWsClient  # Phase 1 sibling
    ws_url = os.environ.get(
        "WS_URL",
        "ws://{0}:{1}/ws/{2}".format(
            config.SERVER_IP,
            config.SERVER_PORT,
            os.environ.get("USER_NAME", "guest"),
        ),
    )
    # Build a proxy-bound dispatcher so body actions actually fire.
    # Previously this was a bare `nao_execute.dispatch` reference, which
    # ws_client invoked as `dispatcher(name, args)` — leaving every
    # naoqi proxy at its default `None`. That made follow_movement,
    # play_animation, dance, etc. silent no-ops even though the server
    # was sending the action frame and TTS played the ack.
    dispatcher = _build_action_dispatcher(log)
    return NaoWsClient(
        server_url=ws_url,
        username=os.environ.get("USER_NAME", "guest"),
        shared_secret=os.environ.get(
            "NAO_SHARED_SECRET", config.NAO_SHARED_SECRET
        ),
        audio_streamer=audio,
        tts_player=tts,
        action_dispatcher=dispatcher,
        brain_cache=brain,
    )


def _build_wake_state_machine(log, leds, fallback_listener, vad,
                              on_engaged, on_lost, on_listening,
                              on_speaking_done):
    """Construct the Phase 3 wake state machine. Owned by sibling
    ``wake-state-machine``. The constructor signature is pinned in
    docs/PHASE_3_TASK_MAP.md; callbacks are wired here so main.py is the
    single coordinator between wake gates and WS session lifecycle.
    """
    from wake_state import WakeStateMachine  # Phase 3 sibling
    # Pass the AdaptiveVad as a kwarg so the WSM can consult it for the
    # "speech onset" engagement gate without reinventing the energy
    # calculation. The sibling's exact kwarg name is open; we try the
    # documented one first and, if the constructor rejects it, fall back
    # to the spec-required positional signature without the VAD wired.
    try:
        return WakeStateMachine(
            config.NAO_IP, config.NAO_PORT,
            leds, fallback_listener,
            on_engaged, on_lost, on_listening, on_speaking_done,
            adaptive_vad=vad,
        )
    except TypeError:
        log.debug("wsm_no_adaptive_vad_kwarg",
                  note="constructor signature lacks adaptive_vad= - falling back")
        return WakeStateMachine(
            config.NAO_IP, config.NAO_PORT,
            leds, fallback_listener,
            on_engaged, on_lost, on_listening, on_speaking_done,
        )


# ---------------------------------------------------------------------------
# Session controller - bundles the per-ENGAGED state (WS client + thread)
# and gives main.py a clean handle to tear it down on AWARE timeout / face
# loss / shutdown. Defined as a class so the wake-state callbacks (which
# fire on the WSM thread) close over a stable object rather than a tangle
# of nonlocals.
# ---------------------------------------------------------------------------


class _SessionController(object):
    """Owns the WS client lifecycle for one engagement period.

    Lifecycle:
        engage(face_id, gate, conf, dist) -> opens audio.start(), spawns the
            WS client thread, and pushes the ``wake_event`` control frame so
            the server can resume a 24 h SQLiteSession + greet.
        disengage(reason)                 -> shuts the WS client down, joins
            its thread, and stops the audio streamer. Idempotent - the
            wake-state callbacks may fire ``on_lost`` more than once on
            edge-case face flicker.
    """

    def __init__(self, log, audio, tts, vad, brain):
        self._log = log
        self._audio = audio
        self._tts = tts
        self._vad = vad
        self._brain = brain
        self._client = None
        self._thread = None
        self._lock = threading.Lock()
        # Lifelike head behavior: SoundLocalizer turns the head toward
        # whoever just spoke (auto_track=True), and ALTracker locks the
        # head onto the closest visible face the rest of the time.
        self._sound_localizer = None
        self._face_tracker = None

    def engage(self, face_id, gate, confidence, distance_m,
               returning_user_hint=None):
        """Open the audio subscriber + spawn WS client + send wake_event."""
        with self._lock:
            if self._client is not None:
                # Already engaged - this is a duplicate fire. Just refresh
                # the wake_event so the server sees the latest gate metadata.
                try:
                    self._client.push_control(
                        "wake_event",
                        {
                            "face_id": face_id,
                            "gate": gate,
                            "confidence": float(confidence or 0.0),
                            "distance_m": float(distance_m or 0.0),
                            "is_returning_user": bool(face_id),
                        },
                    )
                except Exception as exc:
                    self._log.warn("wake_event_refresh_failed", error=str(exc))
                self._push_early_identity(
                    returning_user_hint, face_id, gate, confidence)
                if gate == "touch":
                    self.kick_stand_up(reason="touch_refresh")
                return

            # 1. Start mic subscription (deferred from boot until wake).
            print("[mic_trace] audio.start_called audio_present={0}".format(
                self._audio is not None))
            sys.stderr.flush()
            try:
                if self._audio is not None:
                    mode = self._audio.start()
                    print("[mic_trace] audio.start_ok mode={0}".format(mode))
                    sys.stderr.flush()
                    self._log.info("audio_start_ok", mode=mode)
            except Exception as exc:
                print("[mic_trace] audio.start_failed err={0}: {1}".format(
                    type(exc).__name__, exc))
                sys.stderr.flush()
                self._log.exception("audio_start_failed_on_engage",
                                    error=str(exc))
                # Bail without spawning the client so the WSM falls back to
                # IDLE on the next tick. Caller's outer crash loop will
                # re-init audio on the next attempt.
                return

            # 2. Build a fresh WS client per engagement.
            try:
                self._client = _build_ws_client(
                    self._log, self._audio, self._tts, self._brain,
                )
            except Exception as exc:
                self._log.exception("ws_client_build_failed", error=str(exc))
                self._stop_audio_safe()
                return

            # 3. Wire the live WS client into the AdaptiveVad so EoU hints
            # ride the same socket. AdaptiveVad accepts any object with
            # ``push_control`` so this is duck-typed and safe.
            try:
                if self._vad is not None:
                    self._vad.ws_client = self._client
            except Exception as exc:
                self._log.debug("vad_ws_attach_failed", error=str(exc))

            # 4. Spawn the WS run loop on a daemon thread. ``client.run()``
            # blocks until ``client.shutdown()`` is called or the socket
            # closes terminally, so we must not call it on the WSM thread
            # (which still needs to drive face detection + state).
            self._thread = threading.Thread(
                target=self._run_client_safe,
                name="nao-ws-engaged",
            )
            self._thread.daemon = True
            self._thread.start()

            # 5. Initial wake_event control frame - per Phase 3 contract.
            # ``push_control`` is thread-safe; the sender thread will pick
            # it up as soon as the connection is up.
            try:
                self._client.push_control(
                    "wake_event",
                    {
                        "face_id": face_id,
                        "gate": gate,
                        "confidence": float(confidence or 0.0),
                        "distance_m": float(distance_m or 0.0),
                        "is_returning_user": bool(face_id),
                    },
                )
            except Exception as exc:
                self._log.warn("wake_event_send_failed", error=str(exc))

            self._push_early_identity(
                returning_user_hint, face_id, gate, confidence)

            self._log.info(
                "session_engaged",
                face_id=face_id, gate=gate,
                confidence=float(confidence or 0.0),
                distance_m=float(distance_m or 0.0),
                returning_user_hint=returning_user_hint,
            )
            self.kick_stand_up(reason="engage_{0}".format(gate or "unknown"))

            # 5b. Start lifelike head behavior:
            #   • SoundLocalizer auto-tracks who just spoke (turns head
            #     toward sound source within ~300 ms of speech onset).
            #   • ALTracker face mode locks onto the closest visible
            #     face when no recent sound event — this is what makes
            #     conversation feel like NAO is *looking at you* rather
            #     than staring straight ahead.
            self._start_head_behaviors()

            # 6. Onboarding face-recognition scan. Runs ~3 s after engage
            # on a daemon thread so the camera_announce + first turn
            # aren't blocked. If a known face is in frame, server gets
            # the user's name and greets them. If unknown, server gets
            # `name=None` and the chat agent will introduce NAO + ask
            # for their name (the user can say "remember me as X" to
            # trigger the learn_face tool).
            self._spawn_face_recognition()

    def _hint_to_name(self, returning_user_hint):
        if not returning_user_hint:
            return None
        try:
            if isinstance(returning_user_hint, dict):
                for key in ("display_name", "name", "username"):
                    val = returning_user_hint.get(key)
                    if val:
                        return str(val).strip() or None
                return None
        except Exception:
            pass
        try:
            name = str(returning_user_hint).strip()
            return name or None
        except Exception:
            return None

    def _push_early_identity(self, returning_user_hint, face_id, gate,
                             confidence):
        """Send the WSM's immediate face identity before the async scan.

        The old 4-arg callback discarded `returning_user_hint`, so the
        first turn could run as guest. This early control frame preserves
        recognized names and also marks face-gated unknown users visible so
        the server can ask for their name deterministically.
        """
        if self._client is None:
            return
        name = self._hint_to_name(returning_user_hint)
        face_visible = bool(name)
        if not face_visible:
            try:
                conf = float(confidence or 0.0)
            except Exception:
                conf = 0.0
            face_visible = bool(face_id) and conf > 0.0 and gate != "touch"
        if not name and not face_visible:
            return
        payload = {
            "name": name,
            "recognized": bool(name),
            "face_visible": bool(face_visible),
            "source": "wake_hint" if name else "wake_unknown",
        }
        try:
            self._client.push_control("user_identified", payload)
            self._log.info("user_identified_early",
                           name=name, face_visible=face_visible)
        except Exception as exc:
            self._log.debug("user_identified_early_failed", error=str(exc))

    def kick_stand_up(self, reason="engage"):
        """Move NAO to the engagement posture without blocking wake/touch.

        The WS client also posts NAO on first TTS, but touch wake should
        visibly bring the robot up immediately, even before a reply starts.

        The posture itself is `ENGAGE_POSTURE` (env, default "Sit").
        Standing holds every joint stiff under load, which drains the
        battery fast and is loud next to the mic; sitting still gives a
        visible reaction to touch. Set ENGAGE_POSTURE=Stand to restore the
        original behavior, or ENGAGE_POSTURE=none to not move at all.
        """
        posture_name = os.environ.get("ENGAGE_POSTURE", "Sit").strip()
        if posture_name.lower() == "none":
            self._log.debug("engage_posture_disabled", reason=reason)
            return
        def _do_stand():
            if not _HAS_NAOQI or ALProxy is None:
                return
            try:
                posture = ALProxy("ALRobotPosture",
                                  config.NAO_IP, config.NAO_PORT)
                family = None
                try:
                    family = posture.getPostureFamily()
                except Exception:
                    family = None
                try:
                    if family and str(family).lower().startswith(
                            posture_name.lower()[:5]):
                        self._log.debug("engage_stand_skipped",
                                        reason=reason,
                                        posture_family=family)
                        return
                except Exception:
                    pass
                try:
                    ALProxy("ALMotion",
                            config.NAO_IP, config.NAO_PORT).wakeUp()
                except Exception:
                    pass
                self._log.info("engage_stand_start", reason=reason,
                               posture_family=family,
                               posture=posture_name)
                ok = posture.goToPosture(posture_name, 0.7)
                self._log.info("engage_stand_done", reason=reason, ok=ok,
                               posture=posture_name)
            except Exception as exc:
                self._log.warn("engage_stand_failed", reason=reason,
                               error=str(exc))

        try:
            t = threading.Thread(target=_do_stand, name="nao-engage-stand")
            t.daemon = True
            t.start()
        except Exception as exc:
            self._log.debug("engage_stand_thread_failed", reason=reason,
                            error=str(exc))

    def _spawn_face_recognition(self):
        """Run a quick (~3 s) face-recognition scan on a daemon thread.

        Pushes a ``user_identified`` control frame to the server with:
            { name: <str|null>, recognized: <bool>, source: "face"|"unknown" }

        Server uses ``name`` to greet returning users by name and to seed
        ``sess.username`` for the agent context. If no face was learned
        for this user yet, the chat agent prompt instructs NAO to ask
        for their name and then call the ``learn_face`` tool.
        """
        def _do_scan():
            # Tiny initial delay so camera_announce (~600 ms TTS) gets
            # to play before we burn CPU on face recognition.
            time.sleep(2.0)
            recognized_name = None
            face_visible = False
            try:
                import qi as _qi
                qi_session = _qi.Session()
                try:
                    qi_session.connect(
                        "tcp://" + str(config.NAO_IP) + ":"
                        + str(config.NAO_PORT))
                except Exception as exc:
                    self._log.debug("face_qi_connect_failed", error=str(exc))
                    qi_session = None
            except Exception as exc:
                self._log.debug("face_qi_import_failed", error=str(exc))
                qi_session = None

            if qi_session is not None:
                try:
                    from utils.face_naoqi import recognize_face_naoqi
                    res = recognize_face_naoqi(
                        qi_session, None,
                        subscriber_name="OnboardScan",
                        timeout=3.0,
                        return_seen=True,
                    )
                    if isinstance(res, tuple):
                        recognized_name, face_visible = res
                    else:
                        recognized_name = res
                except Exception as exc:
                    self._log.debug("face_recognize_failed", error=str(exc))

            try:
                if recognized_name and str(recognized_name).strip().lower() in (
                        "guest", "unknown"):
                    recognized_name = None
            except Exception:
                pass

            payload = {
                "name": recognized_name or None,
                "recognized": bool(recognized_name),
                "face_visible": bool(face_visible),
                "source": "face" if recognized_name else "unknown",
            }
            self._log.info("user_identified_scan",
                           name=recognized_name,
                           face_visible=face_visible)
            import sys as _sys
            print("[onboarding] user_identified={0}".format(payload))
            _sys.stderr.flush()

            # Push to server. Best-effort — if WS already closed, skip.
            try:
                if self._client is not None:
                    self._client.push_control("user_identified", payload)
            except Exception as exc:
                self._log.debug("user_identified_push_failed",
                               error=str(exc))

        try:
            t = threading.Thread(target=_do_scan,
                                 name="nao-onboard-face-scan")
            t.daemon = True
            t.start()
        except Exception as exc:
            self._log.debug("face_scan_thread_failed", error=str(exc))

    def _start_head_behaviors(self):
        """Spin up SoundLocalizer (auto-track) + ALTracker face mode.
        Best-effort. Each piece logs and continues if construction fails.
        """
        if not _HAS_NAOQI:
            return

        # Sound localizer — turns the head toward whoever just spoke.
        try:
            from sound_localize import SoundLocalizer
            sl = SoundLocalizer(
                nao_ip=config.NAO_IP,
                nao_port=config.NAO_PORT,
                max_yaw_deg=55.0,
                max_pitch_deg=18.0,
                turn_speed_dps=45.0,    # snappier than default 30 — feels alive
                confidence_min=0.35,
                auto_track=True,
            )
            sl.start()
            self._sound_localizer = sl
            self._log.info("sound_localizer_started")
        except Exception as exc:
            self._log.warn("sound_localizer_start_failed", error=str(exc))
            self._sound_localizer = None

        # ALTracker face mode — keeps eyes on the closest face when no
        # recent sound event drives a re-aim. NAOqi's tracker handles
        # smoothing + reacquire on its own; we just turn it on.
        try:
            tracker = ALProxy("ALTracker", config.NAO_IP, config.NAO_PORT)
            try:
                tracker.stopTracker()
            except Exception:
                pass
            tracker.setMode("Head")
            try:
                # 0.15 m face size hint helps the tracker estimate distance.
                tracker.registerTarget("Face", 0.15)
            except Exception as exc:
                self._log.warn("tracker_register_failed", error=str(exc))
            tracker.track("Face")
            self._face_tracker = tracker
            self._log.info("face_tracker_started")
        except Exception as exc:
            self._log.warn("face_tracker_start_failed", error=str(exc))
            self._face_tracker = None

    def _stop_head_behaviors(self):
        sl = self._sound_localizer
        self._sound_localizer = None
        if sl is not None:
            try:
                sl.stop()
            except Exception:
                pass

        tracker = self._face_tracker
        self._face_tracker = None
        if tracker is not None:
            try:
                tracker.stopTracker()
            except Exception:
                pass
            try:
                tracker.unregisterAllTargets()
            except Exception:
                pass

    def disengage(self, reason):
        """Shut the WS client down and join the run thread. Idempotent."""
        with self._lock:
            client = self._client
            thread = self._thread
            self._client = None
            self._thread = None

        if client is None and thread is None:
            # Nothing to tear down.
            return

        # Detach VAD first so its EoU emitter doesn't push to a dying ws.
        try:
            if self._vad is not None:
                self._vad.ws_client = None
        except Exception:
            pass

        # Stop head-tracking before tearing down the WS so the head
        # doesn't keep tracking after we're done.
        try:
            self._stop_head_behaviors()
        except Exception as exc:
            self._log.debug("head_behaviors_stop_failed", error=str(exc))

        if client is not None:
            try:
                client.shutdown()
            except Exception as exc:
                self._log.warn("ws_shutdown_failed", error=str(exc))

        if thread is not None:
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass

        self._stop_audio_safe()
        self._log.info("session_disengaged", reason=reason)

    def _run_client_safe(self):
        """Wrap ``client.run()`` so a crash in the WS loop doesn't take the
        whole process down - we just log + let the wake state machine
        decide whether to re-engage.
        """
        client = self._client
        if client is None:
            return
        try:
            client.run()
        except Exception as exc:
            self._log.exception("ws_run_crashed", error=str(exc))

    def _stop_audio_safe(self):
        try:
            if self._audio is not None:
                self._audio.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main loop - boots the wake state machine and blocks on it.
# ---------------------------------------------------------------------------


def main():
    configure_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
    log = get_logger(component="main")
    log.info("boot_start",
             nao_ip=config.NAO_IP,
             server_ip=config.SERVER_IP,
             server_port=config.SERVER_PORT,
             has_naoqi=_HAS_NAOQI,
             phase="phase_3_main_rewire")

    _disable_autonomous(config.NAO_IP, config.NAO_PORT)
    _set_volume(config.NAO_IP, config.NAO_PORT, level=100)

    # ALBroker MUST exist before any ALModule subclass is constructed
    # (NaoAudioStreamer is one). Failure here means ALModule creation will
    # raise "no current broker in Python's world" - we want that surfaced
    # at boot, not buried in the streamer's __init__.
    try:
        _ensure_broker(config.NAO_IP, config.NAO_PORT)
        log.info("broker_ready", nao_ip=config.NAO_IP, nao_port=config.NAO_PORT)
    except Exception as exc:
        log.exception("broker_create_failed", error=str(exc))
        # Without a broker, NaoAudioStreamer will crash anyway; bail loudly.
        raise

    brain = user_cache  # placeholder for the brain cache (Phase 7 replaces
                        # this with nao/utils/brain.py - capped 64 KB JSON)

    while True:
        wsm = None
        session = None
        audio = None
        tts = None
        leds = None
        vad = None
        try:
            # Build long-lived components. Each builder returns ``None``-or
            # raises on import failure; we catch and log so the outer loop
            # retries cleanly rather than crashing the whole process.
            try:
                audio = _build_audio_streamer(log)
            except Exception as exc:
                log.exception("audio_streamer_build_failed", error=str(exc))
                raise

            try:
                tts = _build_tts_player(log)
            except Exception as exc:
                log.exception("tts_player_build_failed", error=str(exc))
                raise

            try:
                vad = _build_adaptive_vad(log)
            except Exception as exc:
                # AdaptiveVad is optional for the speech-onset gate; without
                # it the WSM still works on face/proximity/keyword.
                log.warn("adaptive_vad_build_failed", error=str(exc))
                vad = None

            try:
                leds = _build_led_driver(log)
            except Exception as exc:
                log.exception("led_driver_build_failed", error=str(exc))
                raise

            fallback = _build_wake_listener(log)

            session = _SessionController(log, audio, tts, vad, brain)

            # ------------------------------------------------------------------
            # Wake-state callbacks. These run on the WSM thread; they delegate
            # heavy work (audio.start / WS client thread) to ``session`` so the
            # WSM stays responsive to face detection.
            # ------------------------------------------------------------------
            def on_engaged(face_id, gate, confidence, distance_m,
                           returning_user_hint=None):
                log.info("wake_engaged",
                         face_id=face_id, gate=gate,
                         confidence=float(confidence or 0.0),
                         distance_m=float(distance_m or 0.0),
                         returning_user_hint=returning_user_hint)
                session.engage(face_id, gate, confidence, distance_m,
                               returning_user_hint)

            def on_lost():
                log.info("wake_lost", reason="aware_timeout_or_face_lost")
                session.disengage(reason="wake_lost")
                if leds is not None:
                    try:
                        leds.set_idle()
                    except Exception:
                        pass

            def on_listening():
                log.info("wake_listening")

            def on_speaking_done():
                log.info("wake_speaking_done")

            wsm = _build_wake_state_machine(
                log, leds, fallback, vad,
                on_engaged, on_lost, on_listening, on_speaking_done,
            )

            # Head-touch barge-in: when the user touches NAO's head while
            # TTS is playing, stop the audio immediately so they can speak.
            def on_barge(touch_key=None):
                log.info("wake_barge_requested", touch_key=touch_key)
                if touch_key == "RearTactilTouched":
                    _stop_motion_behaviors_safe(
                        log, reason="rear_head_touch_wake_state")
                else:
                    try:
                        session.kick_stand_up(reason="head_touch_barge")
                    except Exception as exc:
                        log.debug("stand_on_head_touch_failed",
                                  error=str(exc))
                if touch_key == "MiddleTactilTouched":
                    try:
                        cli = session._client if session is not None else None
                        reset = getattr(cli, "force_listen_reset", None)
                        if callable(reset):
                            reset(reason="middle_head_touch",
                                  touch_key=touch_key)
                            return
                    except Exception as exc:
                        log.debug("middle_touch_listen_reset_failed",
                                  error=str(exc))
                try:
                    if tts is not None:
                        tts.stop()
                except Exception as exc:
                    log.warn("tts_stop_on_barge_failed", error=str(exc))
                # Also push a barge_in control frame to the server so it
                # cancels any in-flight TTS chunks queued for send.
                try:
                    cli = session._client if session is not None else None
                    if cli is not None:
                        if touch_key == "RearTactilTouched":
                            cancel = getattr(cli, "cancel_actions", None)
                            if callable(cancel):
                                cancel(reason="rear_head_touch")
                        cli.push_control("barge_in", {
                            "source": "touch",
                            "touch_key": touch_key,
                        })
                except Exception as exc:
                    log.debug("barge_control_push_failed", error=str(exc))

            try:
                if hasattr(wsm, "set_barge_callback"):
                    wsm.set_barge_callback(on_barge)
            except Exception as exc:
                log.warn("barge_callback_wire_failed", error=str(exc))

            log.info("wake_state_machine_start")
            # Idle LED before we hand control to the WSM, in case its own
            # init lags. Best-effort.
            if leds is not None:
                try:
                    leds.set_idle()
                except Exception:
                    pass

            wsm.start()  # blocks until stop()
            log.info("wake_state_machine_stopped")

            # Clean exit out of start(): treat as graceful shutdown.
            break

        except KeyboardInterrupt:
            log.info("shutdown_requested")
            break
        except Exception as exc:
            log.exception("crash", error=str(exc))
            # Mirror the Phase 1 stdout trace for SSH-only debug sessions
            # without log shipping.
            try:
                traceback.print_exc()
            except Exception:
                pass
            # Tear down the wake state machine first so its face-detection
            # subscriber doesn't outlive the next iteration's instance.
            if wsm is not None:
                try:
                    wsm.stop()
                except Exception:
                    pass
            # Drop any active session before we recycle audio handles.
            if session is not None:
                try:
                    session.disengage(reason="crash")
                except Exception:
                    pass
            # Final safety net: kill recorder + player at the proxy level
            # so the next boot starts with a quiet audio stack.
            _stop_audio_proxies(config.NAO_IP, config.NAO_PORT)
            time.sleep(2.0)
            continue

    # Final teardown on graceful shutdown.
    if wsm is not None:
        try:
            wsm.stop()
        except Exception:
            pass
    if session is not None:
        try:
            session.disengage(reason="shutdown")
        except Exception:
            pass
    try:
        if audio is not None:
            audio.stop()
    except Exception:
        pass
    try:
        if tts is not None:
            tts.shutdown()
    except Exception:
        pass
    try:
        if vad is not None:
            vad.stop()
    except Exception:
        pass
    if leds is not None:
        try:
            leds.set_idle()
        except Exception:
            pass
    log.info("boot_end")


if __name__ == "__main__":
    main()
