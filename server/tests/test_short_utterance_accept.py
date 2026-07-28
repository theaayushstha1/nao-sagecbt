"""Short single-word speech (names, yes/no) must survive the noise filter.

Regression: saying just your name -- "Mia" -- was silently dropped as
`hallucination_or_noise`, so NAO never replied. The culprit is the blanket
"single word and <= 4 chars" rule in `_looks_like_hallucination`.

That rule exists as a backstop against Whisper hallucinating on silence. But
on the live WS path (`app_ws.py`) silence is already rejected upstream by the
`no_voice` / `silero_no_speech` gates, so a transcript that reaches the filter
has *verified* speech behind it. When the caller can vouch for that, the
length rule must not fire.
"""

from server._legacy_helpers import transcript_reject_reason

# Real short names people actually introduce themselves with.
SHORT_NAMES = ["Mia", "mia", "Ana", "Bo", "Li", "Jose", "Nao", "Kim"]
SHORT_REPLIES = ["yes", "no", "yeah", "nope", "hey"]


def test_short_name_rejected_without_speech_evidence():
    """Backstop preserved: with no VAD evidence, short junk is still dropped."""
    for name in SHORT_NAMES:
        assert transcript_reject_reason("mia", name) == "hallucination_or_noise", (
            "expected default (no speech evidence) to keep rejecting %r" % name
        )


def test_short_name_accepted_when_speech_verified():
    """With VAD-confirmed speech, a bare short name must pass through."""
    for name in SHORT_NAMES:
        assert transcript_reject_reason("mia", name, had_speech=True) is None, (
            "%r was dropped despite verified speech" % name
        )


def test_short_replies_accepted_when_speech_verified():
    for word in SHORT_REPLIES:
        assert transcript_reject_reason("mia", word, had_speech=True) is None, (
            "%r was dropped despite verified speech" % word
        )


def test_empty_still_rejected_even_with_speech():
    """Speech evidence must not resurrect an empty transcript."""
    for blank in ["", "   ", None]:
        assert transcript_reject_reason("mia", blank, had_speech=True) == (
            "hallucination_or_noise"
        )


def test_whisper_artifacts_still_rejected_with_speech():
    """Whisper's YouTube-training artifacts stay rejected regardless of VAD.

    These are never things a person says to a robot, so speech evidence must
    not resurrect them.
    """
    for phrase in ["you", "Thanks for watching!", "please subscribe",
                   "subscribe to my channel", "thank you for watching"]:
        assert transcript_reject_reason("mia", phrase, had_speech=True) == (
            "hallucination_or_noise"
        ), "%r should still be filtered" % phrase


def test_robot_own_lines_still_rejected_with_speech():
    """NAO's scripted greetings must not loop back in as user input."""
    for phrase in ["how can i help you today",
                   "i'm here to listen and support you",
                   "hey there it's great to see you again"]:
        assert transcript_reject_reason("mia", phrase, had_speech=True) == (
            "hallucination_or_noise"
        ), "%r is NAO's own line and should stay filtered" % phrase


def test_human_greetings_accepted_when_speech_verified():
    """A person opening with "Good morning" must get a reply, not silence."""
    for phrase in ["Good morning.", "good morning", "Good afternoon",
                   "Hello", "Hi.", "Ok.", "okay", "Thanks.", "Thank you."]:
        assert transcript_reject_reason("mia", phrase, had_speech=True) is None, (
            "%r was dropped despite verified speech" % phrase
        )


def test_human_greetings_still_rejected_without_speech_evidence():
    """Backstop intact: with no VAD evidence these remain silence-artifacts."""
    for phrase in ["Good morning.", "Hello", "Ok.", "Thank you."]:
        assert transcript_reject_reason("mia", phrase) == "hallucination_or_noise"


def test_self_echo_still_rejected_with_speech():
    """Speech evidence must not defeat the mic-feedback guard."""
    from server import _legacy_helpers as legacy

    legacy.LAST_REPLY["mia"] = "I am doing well, how are you feeling today"
    try:
        reason = transcript_reject_reason(
            "mia", "I am doing well, how are you feeling today", had_speech=True,
        )
        assert reason == "self_echo"
    finally:
        legacy.LAST_REPLY.pop("mia", None)
