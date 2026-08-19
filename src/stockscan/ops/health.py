"""Health check: is the unattended machinery actually healthy right now?

Every check returns (level, name, ok, detail). ``critical`` failures exit
non-zero (data stale, artifact drift, corrupt stores); ``warn`` failures are
reported but don't fail the command (LLM down is fine — narration degrades to
template by design). The command is cheap enough to run ad hoc or from cron.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import (
    ARTIFACTS_DIR,
    HEALTH_FSDS_GRACE_DAYS,
    HEALTH_HEAD_STALE_DAYS,
    HEALTH_PRICE_STALE_DAYS,
    HEALTH_WEB_LAG_DAYS,
    LLM_BASE_URL,
    OPS_STATE_PATH,
    PAPER_DIR,
    WEB_URL,
)
from ..prices import PRICES_DIR

# Every frozen head that carries a trained_through in its on-disk meta. Optional
# heads that aren't built simply don't appear in the staleness check.
_HEAD_METAS = (
    ("model", "model/meta.json"),
    ("distress", "distress_model/meta.json"),
    ("drawdown", "drawdown_model/meta.json"),
    ("confidence_cal", "confidence_cal/calibration.json"),
)


@dataclass
class Check:
    level: str          # 'critical' | 'warn' | 'info'
    name: str
    ok: bool
    detail: str


def _quarter_end(quarter: str) -> pd.Timestamp:
    y, q = int(quarter[:4]), int(quarter[-1])
    return pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)


def _horizon_days(meta) -> int:
    """How long a head must wait for its labels to mature before it can train on a date.

    This is the part of a head's apparent age that is structural rather than neglect: a
    12-month distress label CANNOT be trained closer than ~365 days to the present,
    however promptly the head is rebuilt. Heads that declare no horizon (or a meta that
    predates the field) get 0, which reduces both checks to their old behaviour."""
    if meta.get("label_horizon_days") is not None:
        return int(meta["label_horizon_days"])
    if meta.get("horizon_months") is not None:
        return int(round(float(meta["horizon_months"]) * 365.25 / 12.0))
    return 0


def _head_freezes(base: Path) -> dict[str, dict]:
    """Per head: ``trained_through``, its label horizon, and the derived REBUILD date.

    ``rebuilt = trained_through + horizon`` is the day that head's newest label matured
    — the earliest it could have been built. On the real artifacts it reproduces the
    censor date each head actually recorded, to the day. That derived date, not
    ``trained_through``, is what answers "how long since we retrained this"."""
    out: dict[str, dict] = {}
    for name, rel in _HEAD_METAS:
        p = base / rel
        if not p.exists():
            continue
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            out[name] = {"unreadable": True}
            continue
        tt = meta.get("trained_through")
        if not tt:
            continue
        h = _horizon_days(meta)
        tt = pd.Timestamp(tt).normalize()
        out[name] = {"trained_through": tt, "horizon_days": h,
                     "rebuilt": tt + pd.Timedelta(days=h)}
    return out


def head_staleness(today, artifacts_dir: Path = None,
                   stale_days: int = HEALTH_HEAD_STALE_DAYS) -> Check | None:
    """Warn when a frozen head has not been REBUILT in a long time.

    Frozen-by-design is not frozen-forever: each head keeps displaying its number with
    authority while its training anchor ages. But the age that means neglect is rebuild
    lag, NOT the age of ``trained_through`` — the two differ by the head's label
    horizon, and that gap is arithmetic, not staleness.

    Measuring ``trained_through`` against one flat threshold made the same number mean
    wildly different things per head: ~310d of real budget for the 63-day return model,
    but only ~35d for the 12-month distress head, whose ``trained_through`` can never be
    fresher than ~365d. Distress sat red at 506d while having been rebuilt more recently
    than the drawdown head was — and would have gone red again about five weeks after
    every rebuild, forever. A permanently-red check teaches you to ignore it.

    Heads that aren't built are simply absent; None when nothing carries a date."""
    base = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    t = pd.Timestamp(today).normalize()
    freezes = _head_freezes(base)
    ages, stale = [], []
    for name, f in sorted(freezes.items()):
        if f.get("unreadable"):
            stale.append(f"{name}: meta unreadable")
            continue
        lag = (t - f["rebuilt"]).days
        ages.append(f"{name} {lag}d")
        if lag > stale_days:
            horizon = (f" ({f['horizon_days']}d label horizon, so it trains through "
                       f"{f['trained_through'].date()})" if f["horizon_days"] else "")
            stale.append(f"{name} last rebuilt {f['rebuilt'].date()} ({lag}d ago)"
                         f"{horizon}")
    if not ages and not stale:
        return None
    return Check(
        "warn", "head_staleness", not stale,
        ("all heads rebuilt inside the window: " + ", ".join(ages)) if not stale
        else "; ".join(stale) + f" — stale past {stale_days}d since rebuild; re-freeze "
        f"deliberately (paper retrain-record) or accept the drift")


def head_co_freeze(artifacts_dir: Path = None) -> Check | None:
    """Warn when a risk head was built against an OLDER return model than the current one.

    The risk heads (distress / drawdown / confidence calibration) describe the model they
    were built beside; a return-model refreeze silently orphans them — the confidence
    calibration in particular is a read of the return model's OWN OOS record.

    Compared on REBUILD dates, not on ``trained_through``. Comparing the latter asked a
    question no long-horizon head can ever answer yes to: the return model has a 63-day
    label and distress a 12-month one, so rebuild all of them the same morning from the
    same data and distress still trails by ~10 months. That check could not go green for
    distress or drawdown at any point, on any schedule — permanent noise around the one
    head (confidence_cal, same 63-day horizon) where it was saying something real.

    A WARNING, never a gate: a lagging head degrades visibly, it doesn't block the
    nightly. Absent heads are simply not compared."""
    base = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    freezes = {n: f for n, f in _head_freezes(base).items() if not f.get("unreadable")}
    if "model" not in freezes or len(freezes) < 2:
        return None
    model_built = freezes["model"]["rebuilt"]
    lagging = [f"{n} built {f['rebuilt'].date()} vs model {model_built.date()}"
               for n, f in sorted(freezes.items())
               if n != "model" and f["rebuilt"] < model_built]
    heads = ", ".join(sorted(n for n in freezes if n != "model"))
    return Check(
        "warn", "head_co_freeze", not lagging,
        f"all risk heads built at/past the return model ({heads})" if not lagging
        else "risk heads predate the current return model: " + "; ".join(lagging)
        + " — rebuild them beside the next refreeze (they describe the OLD model)")


def web_freshness(served_as_of, store_max_date,
                  lag_days: int = HEALTH_WEB_LAG_DAYS) -> Check | None:
    """Warn when the running app serves an as-of older than the price store on disk.

    The gap this catches is structural, not hypothetical: the nightly ingests into
    files and exits, while the always-on server keeps the cross-section it built at
    startup. Nothing in either process closes that loop by itself, so the liveness
    probe stays green for as long as the stale app keeps answering. ``None`` when
    there is nothing to compare (app down, no prices, or an unparseable date) —
    absence of evidence is not a warning."""
    if served_as_of is None or store_max_date is None:
        return None
    try:
        served = pd.Timestamp(served_as_of).normalize()
        store = pd.Timestamp(store_max_date).normalize()
    except (ValueError, TypeError):
        return None
    lag = (store - served).days
    return Check(
        "warn", "web_freshness", lag <= lag_days,
        f"app serving as-of {served.date()}, store has {store.date()} (in step)"
        if lag <= lag_days else
        f"app is serving as-of {served.date()} but the store has {store.date()} "
        f"({lag}d behind, stale past {lag_days}d) — the running server never reloaded; "
        f"the nightly's reload job should close this, or click 'update data'")


def llm_models_present(missing: list[str] | None = None) -> Check:
    """Warn when a model Argus is CONFIGURED to serve is not installed.

    Same shape of blind spot as ``web_freshness``, one layer over: the endpoint being
    up says nothing about the model behind it still existing. Pull a new model, drop
    the old one to make room, and /models keeps answering 200 while every completion
    404s — panels refuse 100%, the digest refuses, the judge samples nothing, and
    narration silently falls back to template. Observed 2026-08-18, and it read as
    healthy for a full day because reachability was the only thing being asked."""
    if missing is None:
        from ..narrate.hardware import missing_configured
        missing = missing_configured()
    return Check(
        "warn", "llm_models", not missing,
        "configured models installed" if not missing else
        f"configured but NOT installed: {', '.join(missing)} — every call to "
        f"{'that tier' if len(missing) == 1 else 'those tiers'} 404s (panels/digest/"
        f"chat refuse, narration falls back to template); "
        f"`uv run python scripts/ops.py models` shows the fix")


def run_checks(today=None, prices_dir: Path = PRICES_DIR) -> list[Check]:
    from ..model import MODEL_DIR
    from ..panel import matrix_cache_fresh, matrix_cache_paths
    from .jobs import latest_elapsed_quarter, quarters_present
    from .paper import artifact_fingerprint, current_vintage

    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    checks: list[Check] = []

    # prices freshness — via the matrix-cache meta when fresh, else a direct file
    _, _, meta_p = matrix_cache_paths()
    max_date = None
    if meta_p.exists():
        try:
            max_date = pd.Timestamp(json.loads(meta_p.read_text())["max_date"])
        except Exception:
            max_date = None
    if max_date is None:
        ref = sorted(Path(prices_dir).glob("A*.parquet"))
        if ref:
            max_date = pd.read_parquet(ref[0], columns=["date"])["date"].max()
    if max_date is None:
        checks.append(Check("critical", "prices", False, "no price data found"))
    else:
        age = (t.normalize() - pd.Timestamp(max_date).normalize()).days
        checks.append(Check(
            "critical", "prices", age <= HEALTH_PRICE_STALE_DAYS,
            f"latest bar {pd.Timestamp(max_date).date()} ({age}d ago; "
            f"stale after {HEALTH_PRICE_STALE_DAYS}d)"))

    checks.append(Check(
        "warn", "matrix_cache", matrix_cache_fresh(prices_dir=prices_dir),
        "wide-matrix cache in sync with the per-column store"
        if matrix_cache_fresh(prices_dir=prices_dir)
        else "stale/missing — loaders fall back to the ~2min slow path"))

    # fundamentals recency
    quarters = quarters_present()
    latest_have = quarters[-1] if quarters else None
    expected = latest_elapsed_quarter(t)
    if latest_have == expected:
        checks.append(Check("critical", "fundamentals", True, f"{latest_have} ingested"))
    else:
        overdue = t > _quarter_end(expected) + pd.Timedelta(days=HEALTH_FSDS_GRACE_DAYS)
        checks.append(Check(
            "critical" if overdue else "info", "fundamentals", not overdue,
            f"latest ingested {latest_have}, latest elapsed {expected}"
            + ("" if overdue else " (inside the FSDS publication window)")))

    # artifact + vintage discipline
    try:
        fp = artifact_fingerprint(MODEL_DIR)
        vintage = current_vintage()
        if vintage is None:
            checks.append(Check("warn", "artifact", True,
                                f"artifact {fp} present; no paper baseline frozen yet"))
        else:
            ok = vintage["hash"] == fp
            checks.append(Check(
                "critical", "artifact", ok,
                f"artifact {fp} == registered vintage" if ok else
                f"artifact {fp} != registered vintage {vintage['hash']} — "
                f"unregistered retrain or corrupted artifact"))
    except FileNotFoundError:
        checks.append(Check("critical", "artifact", False, "no frozen artifact on disk"))

    baseline = Path(PAPER_DIR) / "baseline.json"
    checks.append(Check("warn", "paper_baseline", baseline.exists(),
                        "frozen" if baseline.exists() else
                        "not frozen — run 'ops.py paper freeze'"))

    # paper cadence: every completed month since the freeze should have a file
    if baseline.exists():
        signals = sorted((Path(PAPER_DIR) / "signals").glob("*.jsonl"))
        frozen_on = pd.Timestamp(json.loads(baseline.read_text())["frozen_on"][:10])
        prev_month_end = (t.normalize().replace(day=1) - pd.Timedelta(days=1))
        due = prev_month_end >= frozen_on.normalize()
        have_prev = any(pd.Timestamp(p.stem).to_period("M") == prev_month_end.to_period("M")
                        for p in signals)
        checks.append(Check(
            "warn", "paper_signals", (not due) or have_prev,
            f"{len(signals)} month(s) logged"
            + ("" if (not due) or have_prev else
               f"; previous month ({prev_month_end.to_period('M')}) missing — "
               f"nightly will backfill")))

    # job recency (only meaningful once the scheduler has run at least once)
    try:
        from .state import OpsState

        with OpsState(OPS_STATE_PATH) as st:
            last = st.last_run("nightly")
        if last is None:
            checks.append(Check("info", "nightly_job", True, "never run yet"))
        else:
            age_h = (pd.Timestamp.now("UTC").tz_localize(None)
                     - pd.Timestamp(last["started"]).tz_localize(None)).total_seconds() / 3600
            ok = age_h <= 48 and last["status"] in ("ok", "noop", "degraded")
            checks.append(Check("warn", "nightly_job", ok,
                                f"last {last['status']} {age_h:.0f}h ago"))
    except sqlite3.Error as exc:
        checks.append(Check("critical", "ops_state", False, f"state DB unreadable: {exc}"))

    # narration cache openable
    try:
        con = sqlite3.connect(str(ARTIFACTS_DIR / "narration_cache.sqlite"), timeout=5.0)
        con.execute("select count(*) from sqlite_master")
        con.close()
        checks.append(Check("warn", "narration_cache", True, "openable"))
    except sqlite3.Error as exc:
        checks.append(Check("warn", "narration_cache", False, str(exc)))

    staleness = head_staleness(t)
    if staleness is not None:
        checks.append(staleness)
    co_freeze = head_co_freeze()
    if co_freeze is not None:
        checks.append(co_freeze)

    # web UI (informational — a personal tool's server may simply not be running).
    # Liveness and freshness are SEPARATE questions, and only the second one can
    # actually be wrong for a month: the server builds its scored cross-section once
    # at startup and rebuilds it only on an explicit reload, so an always-on process
    # answers 200 indefinitely while serving whatever as-of it loaded with.
    served_as_of = None
    try:
        import httpx

        r = httpx.get(WEB_URL.rstrip("/") + "/api/status", timeout=2.0)
        argus = r.status_code in (200, 503)   # 503 = still loading, still argus
        if argus:
            try:
                served_as_of = (r.json() or {}).get("as_of")
            except ValueError:      # up, but not answering json — liveness still holds
                served_as_of = None
        checks.append(Check("info", "web_ui", argus,
                            f"{WEB_URL} up (status {r.status_code})" if argus else
                            f"{WEB_URL} answers but not argus (status {r.status_code}) — "
                            f"another service on the port, or set STOCKSCAN_WEB_URL"))
    except Exception:
        checks.append(Check("info", "web_ui", True,
                            f"{WEB_URL} not running (fine unless you expect it up)"))

    web_lag = web_freshness(served_as_of, max_date)
    if web_lag is not None:
        checks.append(web_lag)

    # LLM endpoint (informational — template fallback is by design). Probes the
    # OpenAI-compatible /models route, which every supported server (Ollama,
    # llama.cpp, vLLM, LM Studio) answers — the old Ollama-only /api/tags probe
    # also used rstrip("/v1"), a CHARACTER-set strip that mangles base URLs
    # ending in 'v' or '1'.
    try:
        import httpx

        r = httpx.get(LLM_BASE_URL.rstrip("/") + "/models", timeout=3.0)
        checks.append(Check("info", "llm", r.status_code == 200,
                            f"{LLM_BASE_URL} reachable" if r.status_code == 200
                            else f"status {r.status_code}"))
        if r.status_code == 200:
            checks.append(llm_models_present())
    except Exception:
        checks.append(Check("info", "llm", False,
                            f"{LLM_BASE_URL} unreachable — narration falls back to template"))

    free_gb = shutil.disk_usage(str(ARTIFACTS_DIR)).free / 1e9
    checks.append(Check("warn", "disk", free_gb > 5.0, f"{free_gb:.1f} GB free"))
    return checks


def health_record(checks: list[Check], prev_failing: set[str], add_alert) -> dict:
    """Turn one health screen into the job-deltas payload the nightly stores, alerting
    (via ``add_alert(kind, message)``) ONLY on newly-failing criticals: a persistent
    failure alerts once, and a recover-then-refail alerts again. ``prev_failing`` is
    the previous stored record's ``critical_failing`` — the caller must capture it
    BEFORE opening its own job row (the fresh 'running' row has empty deltas and
    would make every failure look new)."""
    failing = sorted(c.name for c in checks if c.level == "critical" and not c.ok)
    for name in failing:
        if name not in prev_failing:
            detail = next(c.detail for c in checks if c.name == name)
            add_alert("health_critical", f"health: {name} critical — {detail}")
    return {
        "checks": [{"level": c.level, "name": c.name, "ok": c.ok, "detail": c.detail}
                   for c in checks],
        "critical_failing": failing,
        "_status": "degraded" if failing else "ok",
    }


def report(checks: list[Check]) -> tuple[str, int]:
    """Human-readable table + exit code (1 if any critical check failed)."""
    lines = []
    worst = 0
    for c in checks:
        mark = "OK " if c.ok else ("FAIL" if c.level == "critical" else "warn")
        lines.append(f"  [{mark:>4}] {c.level:<8} {c.name:<16} {c.detail}")
        if not c.ok and c.level == "critical":
            worst = 1
    return "\n".join(lines), worst
