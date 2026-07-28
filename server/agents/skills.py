"""Skills specialist — time, weather, timers, todos, plus NAO motions.

Motions live here too (not just in chat) so a user in skills mode can naturally
mix "set a timer for 5 minutes" with "and wave at me when it's done" without
having to switch modes. Same tool bundle as chat — the LLM only fires the
matching tool per user phrase, so adding more tools doesn't bias utility
answers.
"""
from agents import Agent, ModelSettings
from server import config
from server.model_factory import resolve_model
from server.agents._memory_inject import with_memory_preamble
from server.tools.skills_tools import (
    get_time, get_date, get_weather_baltimore,
    set_timer, add_todo, list_todos, complete_todo,
)
from server.tools.nao_actions import CHAT_ACTIONS

SYSTEM = (
    "You are NAO's utility assistant. Handle time, date, weather, timers, "
    "and todos by calling the matching tool, then reply with the result in "
    "one short sentence. You can also perform physical actions (wave, nod, "
    "dance, change eye color, follow the user, etc.) by calling the matching "
    "action tool — call multiple in one turn if the user asks for it."
)

_UTILITY_TOOLS = [
    get_time, get_date, get_weather_baltimore,
    set_timer, add_todo, list_todos, complete_todo,
]

skills_agent = Agent(
    name="skills",
    instructions=with_memory_preamble(SYSTEM),
    model=resolve_model(config.SKILLS_MODEL),
    model_settings=ModelSettings(max_tokens=config.NANO_MAX_TOKENS),
    tools=_UTILITY_TOOLS + list(CHAT_ACTIONS),
)
