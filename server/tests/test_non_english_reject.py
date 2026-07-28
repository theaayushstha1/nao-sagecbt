"""Mis-transcribed foreign-language turns must not reach the agent.

The robot is English-only: `transcribe()` pins ``language="en"``. But that
parameter is only a *hint about the input* -- measured 2026-07-28, all three
OpenAI STT models (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
happily return German text with it set, and whisper-1's verbose_json even
reports ``language='english'`` while doing so. So the API cannot be trusted
to flag this.

When a user speaking English gets transcribed as German or Polish, the
transcript is simply wrong. Passing it through makes NAO answer in that
language ("Ja, Mingma, ich bin hier"), which strands the user. Reject it.

Non-Latin scripts (CJK, Hangul, Cyrillic) already fall out because
_clean_asr_text strips them to an empty string. This guard covers the
Latin-script languages that slip past that.
"""

from server._legacy_helpers import transcript_reject_reason

ENGLISH_MUST_PASS = [
    "Hello there",
    "Can you dance for me?",
    "I am feeling sad today",
    "What's the weather like",
    "Tell me a joke about robots",
    "My name is Mingma",
    "I felt a wave of nausea hit me",
    "Stand up and introduce yourself",
    "Here comes the sun, little darling",
    "That sounds like a really good idea to me",
    # Loanwords / diacritics that are ordinary English usage.
    "I went to a cafe",
    "That is a naive assumption",
]

NON_ENGLISH_MUST_REJECT = [
    "Hallo, bist du da? Wie geht es dir heute?",
    "Ja, ich bin hier",
    "Cześć! Miło cię słyszeć.",
    "Proszę, to ja.",
    "Hola, ¿cómo estás?",
    "Bonjour, comment ça va?",
    "Danke schön",
    "Ich möchte nicht",
]


def test_english_is_never_rejected_as_foreign():
    for t in ENGLISH_MUST_PASS:
        reason = transcript_reject_reason("mia", t, had_speech=True)
        assert reason != "non_english", "%r was wrongly flagged foreign" % t


def test_foreign_language_transcripts_rejected():
    for t in NON_ENGLISH_MUST_REJECT:
        reason = transcript_reject_reason("mia", t, had_speech=True)
        assert reason == "non_english", (
            "%r should be rejected as non-English, got %r" % (t, reason)
        )


def test_cjk_still_rejected():
    """Non-Latin scripts were already handled; keep them rejected."""
    for t in ["警察。", "えー", "안녕하세요.",
              "那邊很痛。"]:
        assert transcript_reject_reason("mia", t, had_speech=True) is not None


def test_short_english_still_passes():
    """The short-utterance fix must survive this guard."""
    for t in ["Mia", "No.", "Ok.", "Good morning."]:
        assert transcript_reject_reason("mia", t, had_speech=True) is None
