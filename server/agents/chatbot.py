"""Chatbot specialist — Morgan State CS knowledge via the CS Navigator API.

Phase 5 of the v2 rework moves Morgan-CS knowledge off the in-process Vertex
AI Search client and onto the operator's deployed CS Navigator Cloud Run
service (see ``docs/PHASE_5_TASK_MAP.md``). The new tool returns one clean,
already-RAG-enriched answer string instead of raw passages, so the agent's
job shrinks to: ask CS Navigator, then re-voice the reply for NAO.

Robustness during the staged rollout: ``server.tools.cs_navigator`` lands in
a parallel worktree (``cs-navigator-tool``). If it is not present at import
time, fall back to the legacy ``vertex_search`` tool (which itself replaced
Pinecone earlier on this branch) so the agent keeps working. ``vertex_search``
is marked deprecated and stays parked for one phase per the PRD before
deletion.
"""
from agents import Agent, ModelSettings
from server import config
from server.model_factory import resolve_model
from server.agents._memory_inject import with_memory_preamble

# Prefer the CS Navigator tool. If the cs-navigator-tool worktree hasn't been
# merged yet, fall back to the legacy vertex_search tool so the chatbot agent
# still runs end-to-end (see Phase 5 task map: "may not be merged yet in your
# worktree; guard import with try/except").
try:
    from server.tools.cs_navigator import cs_navigator_search as _SEARCH_TOOL  # type: ignore[import-not-found]
    _SEARCH_TOOL_NAME = "cs_navigator_search"
    _USING_CS_NAVIGATOR = True
except Exception:  # noqa: BLE001 — any import failure (missing module, missing dep) falls back
    from server.tools.vertex_search import vertex_search as _SEARCH_TOOL
    _SEARCH_TOOL_NAME = "vertex_search"
    _USING_CS_NAVIGATOR = False


# Agent prompt. Tool name is interpolated so the same template works for the
# CS Navigator path and the vertex_search fallback. We deliberately don't
# mention "embeddings", "Pinecone", "Vertex", or "RAG" — those are
# implementation details the user-facing voice should never leak.
SYSTEM = (
    "You are NAO, the Morgan State University Computer Science department's "
    "humanoid robot assistant. For ANY question about Morgan's CS curriculum, "
    "courses, faculty, schedule, advising, deadlines, or graduation "
    "requirements, ALWAYS call `{tool}(query)` first with the user's question, "
    "then re-voice the returned answer in your own warm, brief, conversational "
    "tone. Keep replies tight — 25 words or fewer per turn — because they are "
    "spoken aloud, not read. If the tool returns an apology or empty result, "
    "say you're not sure and offer to look again. Never paste raw passages, "
    "URLs, or filenames into your reply — synthesize one clean spoken sentence."
).format(tool=_SEARCH_TOOL_NAME)


chatbot_agent = Agent(
    name="chatbot",
    instructions=with_memory_preamble(SYSTEM),
    model=resolve_model(config.CHATBOT_MODEL),
    model_settings=ModelSettings(max_tokens=config.MINI_MAX_TOKENS),
    tools=[_SEARCH_TOOL],
)
