"""Out-of-app alert delivery — local-first, deterministic, high-severity only.

Alerts land in ops_state.sqlite and are visible in the web watch tab, but a
``paper_degraded`` or ``distress_risk`` fired at 23:30 shouldn't wait silently
until the app is next opened. The nightly's last step pushes ONE summary through
macOS Notification Center (osascript — fully local, no third party sees ticker
names or positions; the owner's standing rule is that portfolio data never
leaves the machine). Only high-severity kinds get lines of their own; the noisy
kinds (percentile moves, filing detections) stay in-app so the notification
never trains the user to ignore it.

Deliberately NO LLM anywhere in this path — delivery must be deterministic; a
notification that depends on a model being up is a notification that silently
stops firing. ``STOCKSCAN_NOTIFY=off`` disables; the default ``auto`` delivers
on macOS and no-ops elsewhere.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from ..config import NOTIFY_MODE, ROUTINE_ALERT_KINDS

# Alert kinds worth interrupting a human for. Everything else stays in-app.
HIGH_SEVERITY = frozenset({
    "paper_degraded", "paper_recovered", "paper_graded",   # the live experiment moved
    "distress_risk",                                       # a held/watched name at risk
    "drawdown_risk",                                       # entered HIGH drawdown risk
    "universe_death",                                      # a tracked name delisted
    "unregistered_artifact",                               # vintage discipline broken
    "health_critical",                                     # the machinery itself is sick
})

_MAX_LINES = 4          # Notification Center truncates anyway; keep it scannable


def _worth_counting(alerts: list[dict]) -> list[dict]:
    """Alerts a banner may claim are 'waiting'. A banner that says 13 routine alerts
    are waiting, when all 13 are filing notices, teaches the reader to ignore it."""
    return [a for a in alerts if a.get("kind") not in ROUTINE_ALERT_KINDS]


def _osascript_available() -> bool:
    return sys.platform == "darwin" and shutil.which("osascript") is not None


def notify_mac(title: str, message: str, timeout: float = 10.0) -> bool:
    """One Notification Center banner. Message text is passed through argv (never
    interpolated into the AppleScript source), so quotes/newlines in alert text
    cannot break or inject the script."""
    try:
        proc = subprocess.run(
            ["osascript",
             "-e", "on run argv",
             "-e", "display notification (item 1 of argv) with title (item 2 of argv)",
             "-e", "end run",
             message, title],
            capture_output=True, timeout=timeout,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def nightly_summary(status: str, alerts: list[dict]) -> tuple[str, str]:
    """(title, message) for the end-of-nightly banner — deterministic, number-safe
    (echoes alert messages verbatim, invents nothing)."""
    high = [a for a in alerts if a.get("kind") in HIGH_SEVERITY]
    title = f"Argus nightly: {status}" + (f" · {len(high)} alert(s)" if high else "")
    lines = [a["message"] for a in high[:_MAX_LINES]]
    if len(high) > _MAX_LINES:
        lines.append(f"… and {len(high) - _MAX_LINES} more in the app")
    if not lines:
        n = len(_worth_counting(alerts))
        lines = [f"{n} routine alert(s) waiting in the app" if n else "no new alerts"]
    return title, "\n".join(lines)


def morning_summary(state) -> tuple[str, str]:
    """(title, message) for the 8am banner. The nightly's own banner fires ~23:30
    when nobody is awake and macOS banners quietly expire — this re-delivers what
    still matters at a humane hour. Deterministic: the stored overnight brief is
    READ (the nightly already wrote it); nothing is generated here."""
    alerts = state.alerts(unseen_only=True, limit=500)
    high = [a for a in alerts if a.get("kind") in HIGH_SEVERITY]
    lines = [a["message"] for a in high[:_MAX_LINES]]
    if len(high) > _MAX_LINES:
        lines.append(f"… and {len(high) - _MAX_LINES} more in the app")

    brief = state.kv_get("digest_brief") or {}
    first = str(brief.get("answer") or "").split(". ")[0].strip()
    if first:
        lines.append(first if first.endswith(".") else first + ".")
    if not lines:
        n = len(_worth_counting(alerts))
        lines = [f"quiet overnight — {n} routine alert(s) waiting" if n
                 else "quiet overnight — nothing needs attention"]
    title = "Argus this morning" + (f" · {len(high)} alert(s)" if high else "")
    return title, "\n".join(lines)


def deliver_morning(state, mode: str = NOTIFY_MODE) -> dict:
    """Push the morning banner. Same never-raise contract as the nightly's."""
    alerts = state.alerts(unseen_only=True, limit=500)
    high = sum(1 for a in alerts if a.get("kind") in HIGH_SEVERITY)
    out = {"mode": mode, "alerts": len(alerts), "high": high, "delivered": False}
    if mode == "off":
        return out
    if not _osascript_available():
        out["mode"] = "unavailable"
        return out
    title, message = morning_summary(state)
    out["delivered"] = notify_mac(title, message)
    if not out["delivered"]:
        out["_status"] = "degraded"
    return out


def deliver_nightly(status: str, alerts: list[dict], mode: str = NOTIFY_MODE) -> dict:
    """Push the end-of-nightly summary. Never raises — the nightly that just did the
    real work must not fail because a banner couldn't be shown."""
    high = sum(1 for a in alerts if a.get("kind") in HIGH_SEVERITY)
    out = {"mode": mode, "alerts": len(alerts), "high": high, "delivered": False}
    if mode == "off":
        return out
    if not _osascript_available():
        out["mode"] = "unavailable"
        return out
    title, message = nightly_summary(status, alerts)
    out["delivered"] = notify_mac(title, message)
    if not out["delivered"]:
        out["_status"] = "degraded"     # visible in job_runs, checked by health
    return out
