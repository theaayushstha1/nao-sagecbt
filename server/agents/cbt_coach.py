"""CBT coach — walks a thought record one step at a time.

Tracks progress in `ctx["cbt_step"]` so we know where we are between turns.
On completion, persists a short summary via `update_user_note` so the next
session can reference it.
"""
from agents import Agent, RunContextWrapper, function_tool
from server import config, memory
from server.model_factory import resolve_model
from server.tools.emotion import identify_distortion, suggest_reframe, log_emotion


_STEPS = (
    "1: situation",        # what happened
    "2: automatic_thought",  # what thought went through your mind
    "3: feeling",          # how that made you feel, 1-10
    "4: evidence",         # for / against
    "5: balanced_thought",  # more balanced view
)


_BASE = (
    "You are a CBT coach on a NAO robot. You walk the user through ONE "
    "thought record, ONE step at a time, asking a single question per turn "
    "and waiting for the user's reply before moving on.\n"
    "\n"
    "STEPS (advance only when the user has answered the current one):\n"
    "  Step 1 - Situation: 'What happened? Just the facts, like a camera saw it.'\n"
    "  Step 2 - Automatic thought: 'What thought went through your mind in "
    "    that moment?' Then call `identify_distortion` and gently name it.\n"
    "  Step 3 - Feeling: 'How did that make you feel, 1-10?'\n"
    "  Step 4a - Evidence FOR the thought: 'What makes that thought feel true?'\n"
    "  Step 4b - Evidence AGAINST: 'What might argue against it?'\n"
    "  Step 5 - Balanced thought: call `suggest_reframe` to offer 2 options, "
    "    let the user pick or word their own.\n"
    "\n"
    "RULES:\n"
    "1) Track progress with the `cbt_step_*` tools. Call `cbt_get_step` at "
    "   the start of every turn to see where you are. Call `cbt_set_step` "
    "   when the user has clearly answered the current step and you're "
    "   moving on.\n"
    "2) Reflect what the user said before asking the next question, but "
    "   do not announce that you heard them or narrate what they asked/said. "
    "   One reflection + one question per turn, max ~25 words.\n"
    "3) Never rush. If the user seems stuck or upset, slow down — append "
    "   'tts_pacing: slow' on its own line.\n"
    "4) When the user lands on a balanced thought, call "
    "   `cbt_finish(summary)` with a one-sentence summary of the record. "
    "   Then hand back to the therapist.\n"
    "5) Never push. If the user resists CBT, hand back to the therapist.\n"
)


def _unwrap(ctx) -> dict:
    return ctx.context if isinstance(ctx, RunContextWrapper) else ctx


@function_tool
def cbt_get_step(ctx: RunContextWrapper) -> str:
    """Return the user's current CBT step (one of: 1..5). Defaults to '1'."""
    store = _unwrap(ctx)
    return str(store.get("cbt_step", "1"))


@function_tool
def cbt_set_step(ctx: RunContextWrapper, step: str) -> str:
    """Advance the user's CBT step. Pass '1'..'5' (or '4a' / '4b')."""
    store = _unwrap(ctx)
    store["cbt_step"] = str(step)
    return f"cbt_step={step}"


@function_tool
def cbt_finish(ctx: RunContextWrapper, summary: str) -> str:
    """Mark the thought record complete and save a one-sentence summary
    to the user's profile under `last_thought_record`."""
    store = _unwrap(ctx)
    face_id = store.get("username", "guest")
    try:
        memory.update_profile(face_id, {"last_thought_record": summary})
    except Exception:
        pass
    store["cbt_step"] = "done"
    return "saved"


def build_cbt_coach_agent(username: str) -> Agent:
    def _instructions(_ctx, _agent) -> str:
        preamble = memory.build_context_preamble(username)
        if preamble:
            return _BASE + "\n" + preamble
        return _BASE

    return Agent(
        name="cbt_coach",
        instructions=_instructions,
        model=resolve_model(config.THERAPIST_MODEL),
        tools=[
            identify_distortion, suggest_reframe, log_emotion,
            cbt_get_step, cbt_set_step, cbt_finish,
        ],
    )


# Back-compat: existing imports of `cbt_coach_agent` still work.
# Built lazily-ish with a guest face_id; the therapist hands off to the
# `username`-specific instance via build_cbt_coach_agent above.
cbt_coach_agent = build_cbt_coach_agent("guest")
