# -*- coding: utf-8 -*-
"""One chat() call that works against either OpenAI or Anthropic.

Most of the agent graph runs through the Agents SDK, where provider choice is
handled by ``server.model_factory``. But a handful of call sites talk to the
OpenAI client directly -- the crisis classifier, the CBT distortion/reframe
tools, the vision classifier, and the memory rollups. Those bypass the SDK
entirely, so they stayed on OpenAI when the rest of the brain moved to Claude.

This module gives them one entry point that dispatches on the model name, so
switching any of them is an env edit rather than a rewrite. It deliberately
takes OpenAI-shaped ``messages`` -- that is what every existing caller already
builds -- and translates to Anthropic's shape when needed:

  * ``system`` messages become Anthropic's top-level ``system`` parameter
    (Anthropic has no system role inside ``messages``).
  * ``response_format={"type": "json_object"}`` has no Anthropic equivalent, so
    ``json_mode=True`` appends an instruction and strips any code fence off the
    reply. Callers still get bare JSON either way.
  * OpenAI ``image_url`` blocks carrying a ``data:`` URI become Anthropic
    ``image`` blocks with base64 source.

Returns the assistant text as a plain string. Raises whatever the underlying
client raises -- callers already have their own error handling, and swallowing
exceptions here would break safety.py's fail-closed behavior.
"""
from __future__ import annotations

import re

from server import config

_ANTHROPIC_PREFIX = "claude-"

_openai_client = None
_anthropic_client = None

# Matches a ```json ... ``` (or bare ```) fence around a JSON body.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def is_anthropic(model: str | None) -> bool:
    return bool(model) and str(model).startswith(_ANTHROPIC_PREFIX)


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
        )
    return _anthropic_client


def _to_anthropic_content(content):
    """Translate one OpenAI message's content into Anthropic blocks."""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content or []:
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            # Callers build "data:image/jpeg;base64,<payload>". Anthropic wants
            # the media type and the raw payload as separate fields.
            if url.startswith("data:"):
                header, _, payload = url.partition(",")
                media_type = header[5:].split(";")[0] or "image/jpeg"
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": payload,
                    },
                })
            else:
                blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })
    return blocks


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text or "")
    return m.group(1) if m else (text or "")


def chat(model: str, messages: list[dict], max_tokens: int = 512,
         temperature: float = 0.0, json_mode: bool = False) -> str:
    """Send an OpenAI-shaped message list to whichever provider owns `model`."""
    if not is_anthropic(model):
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _get_openai().chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    system_parts = [
        m.get("content", "") for m in messages if m.get("role") == "system"
    ]
    convo = [
        {"role": m["role"], "content": _to_anthropic_content(m.get("content"))}
        for m in messages if m.get("role") != "system"
    ]
    system = "\n\n".join(p for p in system_parts if p)
    if json_mode:
        system = (system + "\n\nRespond with ONLY the raw JSON object. No "
                  "prose, no explanation, no markdown code fences.").strip()

    # `temperature` is deliberately dropped for Anthropic. Sampling params
    # (temperature / top_p / top_k) are removed on current Claude models --
    # Opus 5 returns `400 temperature is deprecated for this model`. Callers
    # still pass a temperature because OpenAI accepts one; we just don't
    # forward it. Steer these prompts with wording, not sampling.
    resp = _get_anthropic().messages.create(
        model=model,
        system=system or None,
        messages=convo,
        max_tokens=max_tokens,
    )
    text = "".join(
        getattr(b, "text", "") or "" for b in (resp.content or [])
    )
    return _strip_fence(text).strip() if json_mode else text.strip()
