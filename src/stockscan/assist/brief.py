"""A nightly natural-language operations brief, grounded in the ops-state dicts.

The monitor already records everything overnight — job deltas, watchlist percentile
moves, new filings, alerts, paper-forward drift. This turns that machine output into a
short morning read, using ONLY the numbers in that record (``core.grounded_answer``), so
the brief can't invent a stat. It reports; it never advises and never touches the signal.
"""

from __future__ import annotations

from .core import grounded_answer

BRIEF_SYSTEM = (
    "You write a concise overnight operations brief for a quant-equity monitoring tool. "
    "You are given a JSON CONTEXT of last night's scheduled-job results (with a 'deltas' "
    "summary each), unseen alerts, and paper-forward status. Write 3-6 short sentences "
    "summarizing what actually happened and what needs attention. RULES: use ONLY numbers "
    "that appear in the context; never invent or compute a figure. State facts, not "
    "advice — no buy/sell calls, no predictions. If little of note happened, say so "
    "briefly. Lead with anything that needs a human (failed jobs, alerts, degraded runs)."
)


def build_brief_context(state, jobs=("nightly", "prices", "fsds", "universe",
                                     "news", "monitor"), paper: dict | None = None) -> dict:
    """Gather the overnight record from an :class:`~stockscan.ops.state.OpsState`.

    ``state`` need only expose ``last_run(job)`` and ``alerts(...)`` (duck-typed, so a
    fake stands in for tests). ``paper`` is an optional paper-forward compare dict."""
    runs: dict = {}
    for j in jobs:
        r = state.last_run(j)
        if r:
            runs[j] = {"status": r.get("status"), "finished": r.get("finished"),
                       "deltas": r.get("deltas")}
    # routine filing notices are dropped from BOTH the count and the list: a brief
    # that says "1 alert" while listing thirteen would be incoherent, and the model
    # writes from whatever it is handed. Filtered here rather than via a state method
    # so the duck-typed `state` contract stays `last_run` + `alerts`.
    from ..config import ROUTINE_ALERT_KINDS

    alerts = [a for a in state.alerts(unseen_only=True, limit=50)
              if a.get("kind") not in ROUTINE_ALERT_KINDS]
    ctx: dict = {
        "jobs": runs,
        "n_unseen_alerts": len(alerts),
        "unseen_alerts": [{"kind": a.get("kind"), "message": a.get("message"),
                           "cik": a.get("cik")} for a in alerts],
    }
    if paper is not None:
        ctx["paper_forward"] = paper
    return ctx


def nightly_brief(context: dict, llm) -> dict:
    """Write the overnight brief from a context dict. Returns the grounded_answer result."""
    return grounded_answer(context, "Write the overnight operations brief.", llm,
                           BRIEF_SYSTEM)
