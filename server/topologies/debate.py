"""Debate topology (baseline).

Runs `therapist` and `cbt_coach` in parallel; each drafts a candidate reply.
A lightweight gpt-4o-mini judge picks the better of the two. SafetyAgent runs
post-hoc for observation (verdict is returned in metadata) but DOES NOT
rewrite — this topology-contrast is the whole point: Debate can observe
unsafe replies but cannot block them without taking on Supervisor-Veto's
architecture.
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
from server.agents.therapist import build_therapist_agent
from server.topologies.affect_agent import AffectAgent
from server.topologies.passthrough import _extract_user_text, _record_turn_safe
from server.topologies.safety_agent import SafetyAgent

logger = logging.getLogger("sage.topologies.debate")

_JUDGE_PROMPT = (
    "You are a silent judge choosing between two candidate therapy-bot replies "
    "to the same user turn. Both come from CBT-aware agents. Pick the reply "
    "that better: (a) validates the user first, (b) is non-directive and "
    "non-prescriptive, (c) keeps CBT protocol integrity, (d) is safe. "
    'Respond with STRICT JSON: {"winner": "A" | "B", "reason": "<<=1 sentence>"}. '
    "No prose, no markdown."
)


async def _run_both(message: Any, context: dict, session: Any, username: str):
    """Run therapist and cbt_coach concurrently on the same message.

    We build fresh context copies so the two drafts don't stomp each other's
    actions_queue / emotion_log state. Session is shared read-only on input;
    both calls will attempt to append — acceptable for a baseline topology that
    the paper is expected to lose on anyway.
    """
    therapist = build_therapist_agent(username)
    # Shallow-clone context for parallel drafts so each has its own actions_queue.
    ctx_a = dict(context); ctx_a["actions_queue"] = []
    ctx_b = dict(context); ctx_b["actions_queue"] = []

    task_a = Runner.run(therapist, message, context=ctx_a, session=session)
    task_b = Runner.run(cbt_coach_agent, message, context=ctx_b, session=session)
    res_a, res_b = await asyncio.gather(task_a, task_b, return_exceptions=True)
    return (res_a, ctx_a), (res_b, ctx_b)


def _judge(user_text: str, reply_a: str, reply_b: str) -> str:
    """Return 'A' or 'B'. Defaults to 'A' (therapist) on any failure."""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=resolve_model(config.ROUTER_MODEL),  # gpt-4o-mini
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"user": user_text, "A": reply_a, "B": reply_b},
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=80,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        winner = str(parsed.get("winner", "A")).strip().upper()
        return "B" if winner == "B" else "A"
    except Exception as e:  # noqa: BLE001
        logger.debug("judge call failed, defaulting to A: %s", e)
        return "A"


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

    (res_a, ctx_a), (res_b, ctx_b) = asyncio.run(_run_both(message, context, session, username))

    def _reply_from(res, fallback: str) -> str:
        if isinstance(res, Exception):
            return fallback
        return getattr(res, "final_output", fallback)

    reply_a = _reply_from(res_a, "I'm here with you. Can you tell me more?")
    reply_b = _reply_from(res_b, "I'm listening. What came up for you just now?")

    winner = _judge(user_text, reply_a, reply_b)
    final_reply = reply_a if winner == "A" else reply_b
    winning_ctx = ctx_a if winner == "A" else ctx_b
    last_name = "therapist" if winner == "A" else "cbt_coach"

    # Propagate the winner's side-effects (actions, emotion log) into the caller's context.
    if isinstance(winning_ctx.get("actions_queue"), list):
        context["actions_queue"] = list(winning_ctx["actions_queue"])
    if isinstance(winning_ctx.get("emotion_log"), list):
        context["emotion_log"] = list(winning_ctx["emotion_log"])
    if winning_ctx.get("suppress_image") is not None:
        context["suppress_image"] = bool(winning_ctx["suppress_image"])

    # Post-hoc SafetyAgent: observe, do NOT rewrite. This is the topology contrast.
    verdict = SafetyAgent().evaluate(
        user_text=user_text,
        proposed_reply=final_reply,
        affect_vector=context.get("affect_vector"),
        thought_record_state=context.get("thought_record"),
    )

    metadata = {
        "topology": "debate",
        "candidates": {"A": reply_a, "B": reply_b},
        "winner": winner,
        "verdict_observed": verdict,
    }

    _record_turn_safe(
        username=username,
        user_text=user_text,
        proposed_reply=final_reply,
        final_reply=final_reply,
        verdict=verdict,
        topology="debate",
        affect=context.get("affect_vector"),
    )
    return final_reply, last_name, verdict, metadata
