"""AffectAgent: fuses face-vision + text-sentiment into an affect vector.

Thin wrapper over server/tools/emotion.py's observe_face (GPT-4o vision).
Adds text-sentiment (same gpt-4o-mini used for other fast classifiers),
EMA-smooths the last 5 turns, and writes the resulting vector into
context["affect_vector"].

Output schema written into context:
    {
        "valence":     float in [-1, 1],   # negative = unpleasant, positive = pleasant
        "arousal":     float in [ 0, 1],   # 0 = calm, 1 = activated
        "categorical": str,                # e.g. "sad", "anxious", "angry", "happy", "neutral"
        "confidence":  float in [ 0, 1],
        "mismatch":    bool,               # face vs text disagreement
    }

State carried in context["affect_history"] as a list[dict] of up to 5 most-recent
vectors for EMA smoothing across turns within a session.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from server import config
from server.model_factory import resolve_model
from server.tools.emotion import _observe_face_impl

logger = logging.getLogger("sage.affect_agent")

_MAX_HISTORY = 5
_EMA_ALPHA = 0.45  # current-turn weight; (1 - alpha) applied to previous EMA

# Rough valence/arousal prototypes for the categorical label fallback.
_VA_PROTO = {
    "happy":     (+0.75, 0.55),
    "sad":       (-0.60, 0.30),
    "angry":     (-0.55, 0.80),
    "fearful":   (-0.55, 0.80),
    "anxious":   (-0.45, 0.75),
    "surprised": (+0.20, 0.75),
    "disgusted": (-0.50, 0.55),
    "neutral":   ( 0.00, 0.30),
    "tired":     (-0.15, 0.15),
    "stressed": (-0.55, 0.75),
}


def _openai_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _text_sentiment(user_text: str) -> dict:
    """Small classifier -> {valence, arousal, categorical}. Cheap + forgiving."""
    if not user_text or not user_text.strip():
        return {"valence": 0.0, "arousal": 0.2, "categorical": "neutral"}
    try:
        resp = _openai_client().chat.completions.create(
            model=resolve_model(config.CRISIS_MODEL),  # gpt-4o-mini by default
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return STRICT JSON with affect estimates for the user's "
                        "utterance. Schema: "
                        '{"valence": <float -1..1>, "arousal": <float 0..1>, '
                        '"categorical": <one of: happy, sad, angry, fearful, anxious, '
                        "surprised, disgusted, neutral, tired, stressed>}"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:  # noqa: BLE001
        logger.debug("text sentiment failed: %s", e)
        return {"valence": 0.0, "arousal": 0.2, "categorical": "neutral"}


def _face_to_va(face: dict) -> tuple[float, float, str, float]:
    """Map observe_face() output -> (valence, arousal, categorical, confidence)."""
    if not face or face.get("error"):
        return 0.0, 0.3, "neutral", 0.0
    label = str(face.get("dominant_emotion") or "neutral").lower().strip()
    v, a = _VA_PROTO.get(label, (0.0, 0.3))
    return v, a, label, 0.6  # vision gives no numeric confidence; assume moderate


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, x))


def _ema(history: list[dict], current: dict) -> dict:
    """EMA over valence/arousal, keep last categorical from max-confidence turn."""
    if not history:
        return current
    prev = history[-1]
    v = _EMA_ALPHA * current["valence"] + (1 - _EMA_ALPHA) * prev.get("valence", 0.0)
    a = _EMA_ALPHA * current["arousal"] + (1 - _EMA_ALPHA) * prev.get("arousal", 0.0)
    return {
        "valence": _clamp(v, -1.0, 1.0),
        "arousal": _clamp(a, 0.0, 1.0),
        "categorical": current["categorical"],
        "confidence": current["confidence"],
        "mismatch": current["mismatch"],
    }


class AffectAgent:
    """Call `observe(context, user_text)` once per turn. Writes context['affect_vector']."""

    def observe(self, context: dict, user_text: str) -> dict:
        face = {}
        if context.get("latest_image_b64"):
            try:
                face = _observe_face_impl(context) or {}
            except Exception as e:  # noqa: BLE001
                logger.debug("observe_face failed (swallowed): %s", e)
                face = {}

        face_v, face_a, face_label, face_conf = _face_to_va(face)
        text = _text_sentiment(user_text)
        text_v = _clamp(text.get("valence", 0.0), -1.0, 1.0)
        text_a = _clamp(text.get("arousal", 0.2), 0.0, 1.0)
        text_label = str(text.get("categorical", "neutral")).lower()

        # If we got a face read, fuse 60/40 vision/text; else text-only.
        if face_conf > 0:
            valence = 0.6 * face_v + 0.4 * text_v
            arousal = 0.6 * face_a + 0.4 * text_a
            categorical = face_label if face_conf >= 0.5 else text_label
            confidence = 0.5 * face_conf + 0.5 * (1.0 if user_text else 0.5)
        else:
            valence, arousal = text_v, text_a
            categorical = text_label
            confidence = 0.5 if user_text else 0.2

        # Mismatch flag: face says one sign, text says the other, with meaningful gap.
        mismatch = (
            face_conf > 0
            and abs(face_v - text_v) > 0.8
            and (face_v * text_v) < 0
        )

        current = {
            "valence": _clamp(valence, -1.0, 1.0),
            "arousal": _clamp(arousal, 0.0, 1.0),
            "categorical": categorical,
            "confidence": _clamp(confidence, 0.0, 1.0),
            "mismatch": bool(mismatch),
        }

        history: list[dict] = list(context.get("affect_history") or [])
        smoothed = _ema(history, current)
        history.append(smoothed)
        if len(history) > _MAX_HISTORY:
            history = history[-_MAX_HISTORY:]

        context["affect_history"] = history
        context["affect_vector"] = smoothed
        return smoothed


__all__ = ["AffectAgent"]
