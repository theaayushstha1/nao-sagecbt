"""Weekly and monthly rollups of therapist session recaps."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from server import config, session as s
from server import llm_compat

def _week_start(d: date | None = None) -> str:
    d = d or date.today()
    monday = d - timedelta(days=d.weekday())
    return monday.isoformat()


def _month_start(d: date | None = None) -> str:
    d = d or date.today()
    return d.replace(day=1).isoformat()


def _summarize_to_theme(recaps: list[str]) -> str:
    joined = "\n- ".join(recaps)
    return llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this week's therapy session recaps into 1-2 sentences. "
                    "Focus on recurring themes, growth, or stuck points. Warm, non-clinical tone."
                ),
            },
            {"role": "user", "content": f"- {joined}"},
        ],
        temperature=0.3,
        max_tokens=300,
    )


def _summarize_to_persona(themes: list[str]) -> str:
    joined = "\n- ".join(themes)
    return llm_compat.chat(
        model=config.CRISIS_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this month's weekly themes into a short user persona: "
                    "2-3 sentences capturing who this person has been lately, what matters to them. "
                    "Warm, observational, non-clinical."
                ),
            },
            {"role": "user", "content": f"- {joined}"},
        ],
        temperature=0.4,
        max_tokens=400,
    )


def _save_theme(username: str, when: datetime, body: str) -> None:
    with s._conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO weekly_themes (username, week_start, body) "
            "VALUES (?, ?, ?)",
            (username, _week_start(when.date()), body),
        )


def _save_persona(username: str, when: datetime, body: str) -> None:
    with s._conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO monthly_personas (username, month, body) "
            "VALUES (?, ?, ?)",
            (username, _month_start(when.date()), body),
        )


def maybe_rollup_week(username: str) -> None:
    """If >=3 recaps this week and no theme yet, generate and save the theme."""
    week_start = _week_start()
    with s._conn() as c:
        existing = c.execute(
            "SELECT 1 FROM weekly_themes WHERE username=? AND week_start=?",
            (username, week_start),
        ).fetchone()
        if existing:
            return
        rows = c.execute(
            "SELECT body FROM recaps WHERE username=? "
            "AND DATE(created_at) >= DATE(?) ORDER BY id",
            (username, week_start),
        ).fetchall()
    if len(rows) < 3:
        return
    theme = _summarize_to_theme([row[0] for row in rows])
    _save_theme(username, datetime.now(), theme)


def maybe_rollup_month(username: str) -> None:
    """If >=2 weekly themes this month and no persona yet, generate and save."""
    month_start = _month_start()
    with s._conn() as c:
        existing = c.execute(
            "SELECT 1 FROM monthly_personas WHERE username=? AND month=?",
            (username, month_start),
        ).fetchone()
        if existing:
            return
        rows = c.execute(
            "SELECT body FROM weekly_themes WHERE username=? "
            "AND DATE(week_start) >= DATE(?) ORDER BY id",
            (username, month_start),
        ).fetchall()
    if len(rows) < 2:
        return
    persona = _summarize_to_persona([row[0] for row in rows])
    _save_persona(username, datetime.now(), persona)


def load_week_themes(username: str, n: int = 1) -> list[str]:
    with s._conn() as c:
        rows = c.execute(
            "SELECT body FROM weekly_themes WHERE username=? "
            "ORDER BY id DESC LIMIT ?",
            (username, n),
        ).fetchall()
    return [row[0] for row in rows]


def load_month_personas(username: str, n: int = 1) -> list[str]:
    with s._conn() as c:
        rows = c.execute(
            "SELECT body FROM monthly_personas WHERE username=? "
            "ORDER BY id DESC LIMIT ?",
            (username, n),
        ).fetchall()
    return [row[0] for row in rows]
