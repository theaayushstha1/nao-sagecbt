# -*- coding: utf-8 -*-
"""Resolve a model name from config into something the Agents SDK accepts.

The Agents SDK takes either a bare string (routed to OpenAI via the default
client) or a ``Model`` instance. That means an agent whose env var says
``claude-sonnet-5`` would otherwise send an Anthropic model id to OpenAI's
API and 404.

This module lets every ``*_MODEL`` env var name a model from either provider:

    CHAT_MODEL=gpt-4.1-mini      -> passed through as a string (OpenAI)
    CHAT_MODEL=claude-sonnet-5   -> wrapped in LitellmModel (Anthropic)

Nothing else in the agent graph changes -- handoffs, tools, and sessions are
untouched, so a provider swap is one env edit per agent and is reverted the
same way.
"""
from __future__ import annotations

import logging

from server import config

_log = logging.getLogger(__name__)

# Anthropic model ids all start with "claude-". LiteLLM wants them namespaced
# by provider ("anthropic/claude-sonnet-5"), so we add the prefix here rather
# than making every env var carry it.
_ANTHROPIC_PREFIX = "claude-"
_LITELLM_PROVIDER = "anthropic"

# Built once per model id: constructing a LitellmModel per turn would rebuild
# the client on every request.
_CACHE: dict[str, object] = {}


def is_anthropic(name: str | None) -> bool:
    return bool(name) and str(name).startswith(_ANTHROPIC_PREFIX)


def resolve_model(name: str | None):
    """Return a value suitable for ``Agent(model=...)``.

    OpenAI ids pass through as plain strings. Anthropic ids are wrapped in a
    LitellmModel. On any failure -- litellm not installed, no Anthropic key --
    we log and return the string unchanged rather than raising at import time,
    which would take the whole server down at boot.
    """
    if not is_anthropic(name):
        return name

    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    api_key = getattr(config, "ANTHROPIC_API_KEY", "")
    if not api_key:
        _log.warning(
            "%s is an Anthropic model but ANTHROPIC_API_KEY is empty; "
            "falling back to the string (this will fail against OpenAI)", name,
        )
        return name

    try:
        from agents.extensions.models.litellm_model import LitellmModel
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "litellm unavailable, cannot use Anthropic model %s: %r "
            "(pip install litellm)", name, exc,
        )
        return name

    model = LitellmModel(model=f"{_LITELLM_PROVIDER}/{name}", api_key=api_key)
    _CACHE[name] = model
    _log.info("resolved %s via LiteLLM -> %s/%s", name, _LITELLM_PROVIDER, name)
    return model
