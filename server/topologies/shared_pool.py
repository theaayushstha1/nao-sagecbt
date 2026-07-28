"""SharedPool topology (baseline).

All three specialists — therapist, cbt_coach, grounding_coach — write a short
candidate paragraph into a shared scratchpad prompt. Therapist then takes
final say, synthesizing the pool into one reply. SafetyAgent runs post-hoc
for observation (returned in metadata) but does NOT rewrite.

This is the classic SharedPool topology from MedSentry 2025 (arXiv 2505.20824)
adapted to CBT — it's the weakest baseline under multi-turn adversarial
pressure and is what the paper's Supervisor-Veto topology is expected to
Pareto-dominate.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agents import Runner
from openai import OpenAI

from server import config
from server.model_factory import resolve_model
from server.agents.cbt_coach import cbt_coach_agent
from server.agents.grounding_coach import grounding_coach_agent
from server.agents.therapist import build_therapist_agent
from server.topologies.affect_agent import AffectAgent
from server.topologies.passthrough import _extract_user_text, _record_turn_safe
from server.topologies.safety_agent import SafetyAgent

logger = logging.getLogger("sage.topologies.shared_pool")

_POOL_DRAFT_PROMPT_TMPL = (
    "Draft ONE short candidate paragraph (<=2 sentences) for how a CBT robot "
    "should respond to the user turn below. Be empathic and non-directive. "
    "Do not talk to the user in the second person yet — just draft.\n\n"
    "User turn:\n{user_text}\n"
)

_FINAL_SYNTHESIS_TMPL = (
    "You are the therapist. You have three draft candidates from your own "
    "earlier thoughts, the CBT coach, and the grounding coach (in a shared "
    "scratchpad). Synthesize ONE final reply to the user (<=2 short sentences), "
    "validation first. Keep CBT protocol integrity.\n\n"
    "Shared scratchpad:\n"
    "- therapist_draft: {t}\n"
    "- cbt_coach_draft: {c}\n"
    "- grounding_coach_draft: {g}\n\n"
    "User turn: {user_text}"
)


async def _run_three(message: Any, context: dict, session: Any, username: str):
    therapist = build_therapist_agent(username)
    # Parallel drafts: each gets its own context copy so they don't collide.
    ctx_t = dict(context); ctx_t["actions_queue"] = []
    ctx_c = dict(context); ctx_c["actions_queue"] = []
    ctx_g = dict(context); ctx_g["actions_queue"] = []

    task_t = Runner.run(therapist, message, context=ctx_t, session=session)
    task_c = Runner.run(cbt_coach_agent, message, context=ctx_c, session=session)
    task_g = Runner.run(grounding_coach_agent, message, context=ctx_g, session=session)
    res_t, res_c, res_g = await asyncio.gather(
        task_t, task_c, task_g, return_exceptions=True
    )
    return (res_t, ctx_t), (res_c, ctx_c), (res_g, ctx_g)


def _synthesize(user_text: str, t: str, c: str, g: str) -> str:
    """Therapist-role gpt-4o call that collapses the pool into a single reply."""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=resolve_model(config.THERAPIST_MODEL),
            messages=[
                {"role": "system", "content": "You are a warm, non-clinical CBT companion on a NAO robot for college students. Never diagnose. Validate first."},
                {"role": "user", "content": _FINAL_SYNTHESIS_TMPL.format(t=t, c=c, g=g, user_text=user_text)},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or t).strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("shared_pool synthesis failed, falling back to therapist draft: %s", e)
        return t


def run(
    agent: Any,
    message: Any,
    *,
    context: dict,
    session: Any,
) -> tuple[str, str, dict | None, dict]:
    username = context.get("username", "guest")
    user_text = _extract_user_text(message)

    try:
        AffectAgent().observe(context, user_text)
    except Exception as e:  # noqa: BLE001
        logger.debug("AffectAgent.observe failed (swallowed): %s", e)

    (res_t, ctx_t), (res_c, ctx_c), (res_g, ctx_g) = asyncio.run(
        _run_three(message, context, session, username)
    )

    def _reply_from(res, fallback: str) -> str:
        if isinstance(res, Exception):
            return fallback
        return getattr(res, "final_output", fallback)

    draft_t = _reply_from(res_t, "Say a bit more?")
    draft_c = _reply_from(res_c, "What thought went through your mind just then?")
    draft_g = _reply_from(res_g, "Let's take one slow breath together first.")

    final_reply = _synthesize(user_text, draft_t, draft_c, draft_g)

    # Therapist keeps final say — propagate its side-effects.
    if isinstance(ctx_t.get("actions_queue"), list):
        context["actions_queue"] = list(ctx_t["actions_queue"])
    if isinstance(ctx_t.get("emotion_log"), list):
        context["emotion_log"] = list(ctx_t["emotion_log"])
    if ctx_t.get("suppress_image") is not None:
        context["suppress_image"] = bool(ctx_t["suppress_image"])

    verdict = SafetyAgent().evaluate(
        user_text=user_text,
        proposed_reply=final_reply,
        affect_vector=context.get("affect_vector"),
        thought_record_state=context.get("thought_record"),
    )

    metadata = {
        "topology": "shared_pool",
        "drafts": {"therapist": draft_t, "cbt_coach": draft_c, "grounding_coach": draft_g},
        "verdict_observed": verdict,
    }

    _record_turn_safe(
        username=username,
        user_text=user_text,
        proposed_reply=final_reply,
        final_reply=final_reply,
        verdict=verdict,
        topology="shared_pool",
        affect=context.get("affect_vector"),
    )
    return final_reply, "therapist", verdict, metadata
