"""Pattern-based motion intent detector.

Bypasses the LLM for unambiguous body-action requests. The router was
unreliable here — it would sometimes hand off to a generic agent that
replied "I'm a virtual assistant, I can't stand up" instead of calling
the tool. This module catches those transcripts BEFORE the agent runs
and emits the action + a short ack directly.

Order matters: longer, more specific phrases come first so that
"sit down" doesn't match the "sit" inside "sit-down comedy".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (action_name, args_dict, ack_text, list_of_phrases)
# Phrases are matched as case-insensitive whole-word substrings of the
# transcript. Order top-to-bottom = priority.
_TRIGGERS: list[tuple[str, dict, str, list[str]]] = [
    # ── Posture ─────────────────────────────────────────────
    ("stand_up", {}, "Standing up.", [
        "stand up", "get up", "stand straight", "rise up", "to your feet",
        "stand please", "please stand", "could you stand", "can you stand",
    ]),
    ("sit_down", {}, "Sitting down.", [
        "sit down", "have a seat", "take a seat", "please sit", "could you sit",
    ]),
    ("kneel", {}, "Kneeling.", [
        "kneel down", "kneel", "go on one knee",
    ]),

    # ── Gestures ────────────────────────────────────────────
    ("wave_both_hands", {}, "Waving with both hands!", [
        "wave with both hands", "wave both hands", "wave both",
    ]),
    ("wave_hand", {"hand": "right"}, "Waving hi!", [
        "wave hi", "wave hello", "say hi", "say hello", "wave hand",
        "wave at me", "give me a wave", "wave please", "could you wave",
        "can you wave", "just wave",
        # NOT bare "wave"/"waving" -- test_wave_negative_idiom pins that,
        # because "a wave of nausea" must not make NAO wave. Only add
        # phrasings that unambiguously address the robot.
        "wave your hand", "wave to me", "wave for me", "wave to us",
    ]),
    ("nod_head", {"times": 2}, "*nods*", [
        "nod your head", "nod twice", "nod yes", "give me a nod", "just nod",
    ]),
    ("shake_head", {"times": 2}, "*shakes head*", [
        "shake your head", "shake head", "say no with your head",
    ]),
    ("clap_hands", {"times": 3}, "Clapping.", [
        "clap", "clap now", "can you clap", "could you clap",
        "please clap", "clap your hands", "clap for me",
        "clap for yourself", "clap for your self", "clap for ur self",
        "give me a clap", "applaud", "round of applause",
    ]),

    # ── Locomotion ──────────────────────────────────────────
    ("move_forward", {"meters": 0.3}, "Walking forward.", [
        "step forward", "walk forward", "come forward", "move forward",
        "come closer",
        # Bare "walk"/"come" phrasings. Without these the turn falls through
        # to the tool-less fast-chat lane, where NAO answers "I'm all
        # digital, no legs to move around" -- it denies having a body.
        "can you walk", "could you walk", "please walk", "walk please",
        "walk to me", "walk towards me", "walk toward me",
        "start walking", "keep walking", "let me see you walk",
        "i want to see you walk", "show me you can walk", "show me a walk",
        "come here", "come to me", "walk over here",
    ]),
    ("move_backward", {"meters": 0.3}, "Stepping back.", [
        "step back", "walk back", "move back", "step backward", "back up",
        "can you walk back", "can you walk backward",
        "can you walk backwards", "walk backward", "walk backwards",
        "move backward", "move backwards", "go back",
    ]),
    ("turn_left", {"degrees": 45.0}, "Turning left.", [
        "turn left", "rotate left", "look left",
    ]),
    ("turn_right", {"degrees": 45.0}, "Turning right.", [
        "turn right", "rotate right", "look right",
    ]),
    ("spin", {"degrees": 360.0}, "Spinning!", [
        "spin around", "do a spin", "twirl", "full turn",
    ]),

    # ── Performance ─────────────────────────────────────────
    ("dance", {"style": "robot"}, "Let's dance!", [
        "do a dance", "show me a dance", "dance for me", "dance please",
        "give me a dance", "can you dance", "could you dance", "let's dance",
        "do a robot dance", "do the robot", "dance now", "you dance now",
        "start dancing", "show your dance", "show us your dance",
        "let me see you dance", "i want to see you dance",
        "i would love to see you dance",
    ]),
    ("follow_movement", {}, "Following you now.", [
        # Canonical short form — fires the Choregraphe `follow-me` pack.
        # Bare "follow" alone is omitted on purpose — too greedy
        # ("follow up on what we said" would mis-fire).
        "follow me", "come follow me", "follow me around",
        "start following me", "follow me now",
        # Tracking semantics (these are unambiguous robot commands)
        "track me", "stay close", "stay with me",
        # Mirror-me semantics (legacy — copies user's pose)
        "follow my movement", "mirror me", "copy me", "follow what i do",
    ]),
    # Stop the follow behavior. NOTE: avoid single-word "stop" / "halt"
    # here because triggers match on word boundaries, and a bare "stop"
    # would mis-fire on phrases like "stop watching me" (camera-off
    # trigger) or "stop talking". Use multi-word phrases only.
    ("stop_follow", {}, "Stopping.", [
        "stop following me", "stop following", "don't follow me",
        "do not follow me", "stop tracking me", "stop tracking",
        "stay there", "stay here", "stay still", "freeze",
        "enough following",
    ]),

    # ── Camera consent ──────────────────────────────────────
    # Action names are server-side identifiers, not NAO motor calls. The
    # consumer (app_ws.py) flips session.set_camera_consent(...) and emits a
    # `control { subtype: "camera_state", data: {enabled: ...} }` frame so
    # the client UI updates immediately. The fast path here exists because
    # the LLM sometimes mis-routes "stop watching me" to a generic chat
    # reply instead of calling the tool — a regex match guarantees the
    # state flip and the canonical ack land on the same turn.
    ("disable_camera", {}, "Camera off.", [
        "stop watching me", "stop watching", "don't watch me", "do not watch me",
        "stop looking at me", "don't look at me", "do not look at me",
        "turn off the camera", "turn the camera off", "camera off",
        "disable the camera", "disable camera", "close your eyes",
        "stop recording me", "stop seeing me",
    ]),
    ("enable_camera", {}, "Camera on.", [
        "you can watch me again", "you can look at me again", "watch me again",
        "look at me again", "turn on the camera", "turn the camera on",
        "camera on", "enable the camera", "enable camera", "open your eyes",
        "you can see me now", "see me again",
    ]),

    # ── Voice profile picker (Phase 11.8) ───────────────────
    # Three voices: girl, man, neutral. Recognized via short phrases the
    # user can say at any time during a session. The handler in app_ws
    # reads `args.profile` and persists via session.set_voice_profile.
    ("set_voice_profile", {"profile": "girl"}, "Switching to girl voice.", [
        "use the girl voice", "use girl voice", "girl voice",
        "switch to girl voice", "switch to the girl voice",
        "use a woman's voice", "use the woman voice", "female voice",
        "use voice one", "voice one", "voice 1", "first voice",
    ]),
    ("set_voice_profile", {"profile": "man"}, "Switching to man voice.", [
        "use the man voice", "use man voice", "man voice",
        "switch to man voice", "switch to the man voice",
        "use a man's voice", "use a male voice", "male voice", "guy voice",
        "use voice two", "voice two", "voice 2", "second voice",
    ]),
    ("set_voice_profile", {"profile": "neutral"}, "Switching to neutral voice.", [
        "use the neutral voice", "use neutral voice", "neutral voice",
        "switch to neutral voice", "switch to the neutral voice",
        # Deepgram has misheard "neutral voice" as "bureau voice".
        "bureau voice", "switch to the bureau voice", "use the bureau voice",
        "use voice three", "voice three", "voice 3", "third voice",
    ]),
    ("set_voice_profile", {"profile": "my"}, "Switching to your voice.", [
        "switch to my voice", "use my voice", "my voice",
        "switch to your voice", "use your voice", "your voice",
        "switch to aayush voice", "aayush voice", "use aayush voice",
        "switch to operator voice", "operator voice",
        "use voice four", "voice four", "voice 4", "fourth voice",
    ]),

    # ── LEDs ────────────────────────────────────────────────
    ("change_eye_color", {"color": "red"}, "Eyes red.", [
        "eyes red", "red eyes", "turn your eyes red", "make your eyes red",
    ]),
    ("change_eye_color", {"color": "green"}, "Eyes green.", [
        "eyes green", "green eyes", "turn your eyes green", "make your eyes green",
    ]),
    ("change_eye_color", {"color": "blue"}, "Eyes blue.", [
        "eyes blue", "blue eyes", "turn your eyes blue", "make your eyes blue",
    ]),
    ("change_eye_color", {"color": "purple"}, "Eyes purple.", [
        "eyes purple", "purple eyes",
    ]),
    ("change_eye_color", {"color": "yellow"}, "Eyes yellow.", [
        "eyes yellow", "yellow eyes",
    ]),
    ("change_eye_color", {"color": "white"}, "Eyes white.", [
        "eyes white", "white eyes", "reset your eyes", "default eyes",
    ]),
]

# Pre-compile a list of (compiled_regex, action_name, args, ack)
_COMPILED: list[tuple[re.Pattern, str, dict, str]] = []
# Longest phrase first, so the most specific trigger wins regardless of which
# action declared it. detect() takes the first match, and declaration order
# alone would send "can you walk back" forward -- move_forward is declared
# above move_backward, and "can you walk" is a prefix of it.
_ALL_PHRASES = [
    (p, action, args, ack)
    for action, args, ack, phrases in _TRIGGERS
    for p in phrases
]
for p, action, args, ack in sorted(_ALL_PHRASES, key=lambda x: len(x[0]),
                                   reverse=True):
    # Word-boundary match around the phrase. Allows "please stand up now"
    # to match "stand up" but not "withstand uphill".
    pattern = re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
    _COMPILED.append((pattern, action, args, ack))

_FOLLOW_SOCIAL_RE = re.compile(r"\bfollow\s+me\s+on\b", re.IGNORECASE)


# Animation names NAO can map to installed Choregraphe behaviors. These are
# deliberately behind an action verb ("do", "act like", "pretend to be",
# etc.) so "I saw an elephant" remains normal conversation while "do an
# elephant" becomes an immediate robot action.
_ANIMATION_ALIASES: dict[str, tuple[str, ...]] = {
    "elephant": ("elephant",),
    "gorilla": ("gorilla", "gorrila", "ape"),
    "monkey": ("monkey",),
    "dragon": ("dragon",),
    "dinosaur": ("dinosaur", "dino"),
    "lion": ("lion",),
    "tiger": ("tiger",),
    "bear": ("bear",),
    "bird": ("bird",),
    "eagle": ("eagle",),
    "chicken": ("chicken",),
    "penguin": ("penguin",),
    "duck": ("duck",),
    "rabbit": ("rabbit", "bunny"),
    "cat": ("cat",),
    "dog": ("dog", "puppy"),
    "horse": ("horse",),
    "snake": ("snake",),
    "spider": ("spider",),
    "shark": ("shark",),
    "frog": ("frog",),
    "kungfu": ("kung fu", "kung-fu", "kungfu", "martial arts"),
    "air_guitar": ("air guitar", "air-guitar", "airguitar", "guitar"),
    "headbang": ("headbang", "head bang", "head-bang"),
    "bandmaster": ("bandmaster", "conductor", "conducting"),
    "fitness": ("fitness", "workout", "exercise"),
    "air_juggle": ("air juggle", "juggle", "juggling"),
    "binoculars": ("binoculars",),
    "drive_car": ("drive car", "driving", "car"),
    "helicopter": ("helicopter",),
    "knight": ("knight",),
    "monster": ("monster",),
    "magic": ("magic", "mystic", "wizard"),
    "spaceship": ("spaceship", "space shuttle", "rocket"),
    "take_picture": ("take picture", "camera pose"),
    "taxi": ("taxi",),
    "vacuum": ("vacuum",),
    "waddle": ("waddle",),
    "zombie": ("zombie",),
    "claw": ("claw", "claws"),
    "wings": ("wings", "flap your wings"),
    "love_you": ("love you", "heart"),
}

_ANIMATION_PATTERNS: list[tuple[re.Pattern, str, str]] = []
for _animation, _aliases in _ANIMATION_ALIASES.items():
    for _alias in sorted(_aliases, key=len, reverse=True):
        _a = re.escape(_alias).replace(r"\ ", r"\s+")
        _ANIMATION_PATTERNS.extend([
            (
                re.compile(
                    r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?"
                    r"(?:do|perform|play|run|show\s+me|show\s+us)\s+"
                    r"(?:a|an|the)?\s*" + _a + r"\b",
                    re.IGNORECASE,
                ),
                _animation,
                _alias,
            ),
            (
                re.compile(
                    r"\b(?:can\s+you\s+|could\s+you\s+|please\s+)?"
                    r"(?:act\s+like|pretend\s+to\s+be|pretend\s+you're|be)\s+"
                    r"(?:a|an|the)?\s*" + _a + r"\b",
                    re.IGNORECASE,
                ),
                _animation,
                _alias,
            ),
        ])


@dataclass
class MotionMatch:
    action: str
    args: dict
    ack: str


# ---------------------------------------------------------------------------
# Face-learn fast-path with name extraction.
# ---------------------------------------------------------------------------
# Catches high-confidence "remember me as X" / "save my face as X"
# patterns *before* the LLM routes the turn. Plain "my name is X" is
# handled only while the server is explicitly asking for a name; otherwise
# it pollutes the face DB when a recognized user casually reintroduces
# themselves.
# Without this the router
# (default agent on bare-wake) sends "remember me as Aayush" to the
# chatbot agent, which replies with the Morgan State greeting instead of
# learning the face.
_NON_NAMES_FAST = frozenset({
    "tired", "fine", "good", "okay", "ok", "great", "happy", "sad",
    "anxious", "stressed", "depressed", "lonely", "scared", "angry",
    "ready", "back", "here", "sorry", "done", "leaving", "going",
    "trying", "thinking", "yes", "yeah", "yep", "no", "nope", "hi",
    "hello", "hey", "thanks", "thank", "nao", "use", "using", "switch",
    "voice", "terminal", "obsidian", "money", "update", "process",
    "medicine", "camera", "therapy", "therapist",
    "assistant",
})

_SUSPICIOUS_BARE_NAME_ANSWERS = frozenset({
    # Common STT hallucinations seen while the robot is waiting for a name.
    # A real person can still enroll with an explicit phrase like
    # "my name is Rafael"; this only blocks a bare one-word answer.
    "rafael",
    "rafaell",
    "raphael",
})


def _looks_like_name(candidate: str) -> bool:
    """Conservative gate: don't fire learn_face on common adjectives."""
    c = (candidate or "").strip().lower()
    if not c or c in _NON_NAMES_FAST:
        return False
    if not (2 <= len(c) <= 24):
        return False
    return bool(re.match(r"^[a-z][a-z'\-]+$", c))


_LEARN_FACE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), ack_tmpl)
    for p, ack_tmpl in (
        # Highest confidence: explicit teach verbs.
        (r"\bremember\s+me\s+as\s+([a-z][a-z'\-]+)\b",
         "Remembering you as {name}."),
        (r"\bsave\s+my\s+face\s+as\s+([a-z][a-z'\-]+)\b",
         "Saving your face as {name}."),
        (r"\blearn\s+my\s+face\s+as\s+([a-z][a-z'\-]+)\b",
         "Learning your face as {name}."),
        (r"\bremember\s+(?:that\s+)?my\s+name\s+is\s+([a-z][a-z'\-]+)\b",
         "Remembering you as {name}."),
    )
)


def _detect_learn_face(transcript: str) -> MotionMatch | None:
    """Extract a name + emit a `learn_face` action without an LLM hop.
    Returns None when no high-confidence pattern matches.
    """
    for pattern, ack_tmpl in _LEARN_FACE_PATTERNS:
        m = pattern.search(transcript)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if not _looks_like_name(raw):
            continue
        name = raw[0].upper() + raw[1:].lower()
        return MotionMatch(
            action="learn_face",
            args={"name": name},
            ack=ack_tmpl.format(name=name),
        )
    return None


def detect_name_answer(transcript: str) -> MotionMatch | None:
    """Handle a bare name answer during the onboarding name prompt.

    This is intentionally separate from `detect()` so random one-word turns
    do not teach faces unless the caller already knows it is asking for a
    name.
    """
    text = (transcript or "").strip()
    if not text:
        return None
    text = re.sub(r"^[\"'\s]+|[\"'.,!?\s]+$", "", text)
    if not text:
        return None

    m = re.match(
        r"^(?:it(?:'| i)?s|i am|i'm|my name is|call me|"
        r"you can call me)\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    explicit_intro = m is not None
    raw = (m.group(1) if m else text).strip()
    raw = re.sub(r"[\"'.,!?\s]+$", "", raw)
    parts = raw.split()
    if not (1 <= len(parts) <= 2):
        return None
    if (not explicit_intro and len(parts) == 1
            and parts[0].strip().lower() in _SUSPICIOUS_BARE_NAME_ANSWERS):
        return None
    if not all(_looks_like_name(p) for p in parts):
        return None
    name = " ".join(p[0].upper() + p[1:].lower() for p in parts)
    return MotionMatch(
        action="learn_face",
        args={"name": name},
        ack="Nice to meet you, {name}.".format(name=name),
    )


def _detect_animation_request(transcript: str) -> MotionMatch | None:
    """Catch explicit requests for a named installed/aliased animation."""
    for pattern, animation, alias in _ANIMATION_PATTERNS:
        if not pattern.search(transcript):
            continue
        pretty = alias.replace("_", " ")
        return MotionMatch(
            action="play_animation",
            args={"animation": animation},
            ack="Doing {0}.".format(pretty),
        )
    return None


def detect(transcript: str) -> MotionMatch | None:
    """Return MotionMatch if `transcript` clearly requests a NAO body action.

    Returns None for ambiguous or non-motion input — those go to the LLM.

    Order:
      1. Face-learn fast-path (regex with name capture). Saves an LLM
         hop on the very common "remember me as X" turn and avoids the
         router sending it to chatbot/Morgan by mistake.
      2. Named animation fast-path for phrases like "do a gorilla" or
         "act like an elephant".
      3. Static-phrase trigger table (`_COMPILED`). Posture, gestures,
         dances, follow-me, voice picker, etc.
    """
    if not transcript:
        return None
    t = transcript.strip()
    if not t:
        return None

    # 1. learn_face fast-path.
    lf = _detect_learn_face(t)
    if lf is not None:
        return lf

    if _FOLLOW_SOCIAL_RE.search(t):
        return None

    anim = _detect_animation_request(t)
    if anim is not None:
        return anim

    # 2. Static phrases.
    for pattern, action, args, ack in _COMPILED:
        if pattern.search(t):
            return MotionMatch(action=action, args=dict(args), ack=ack)
    return None
