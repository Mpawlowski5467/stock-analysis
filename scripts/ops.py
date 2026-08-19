"""Continuous-operation CLI: every scheduled job and its manual override.

  uv run python scripts/ops.py nightly            # the launchd entry (see below)
  uv run python scripts/ops.py prices             # nightly price refetch only
  uv run python scripts/ops.py fsds               # ingest any missing FSDS quarter
  uv run python scripts/ops.py universe           # security-master refresh
  uv run python scripts/ops.py delistings         # rebuild the delist/dereg ledger
  uv run python scripts/ops.py monitor [--no-llm] [--no-edgar]
  uv run python scripts/ops.py news [--no-llm]     # watchlist headline memory (live-view)
  uv run python scripts/ops.py themes              # auto-tag AI/SaaS/EV markets (live-view)
  uv run python scripts/ops.py reload              # make the running web app re-read disk
  uv run python scripts/ops.py watch add AAPL --note "core holding"
  uv run python scripts/ops.py watch ls | rm AAPL
  uv run python scripts/ops.py alerts [--all]
  uv run python scripts/ops.py paper freeze | log | compare
  uv run python scripts/ops.py paper retrain-record --reason "quarterly retrain 2026q3"
  uv run python scripts/ops.py health
  uv run python scripts/ops.py backup              # snapshot the sqlite stores now
  uv run python scripts/ops.py digest              # overnight brief (local model, grounded)
  uv run python scripts/ops.py morning             # 8am banner: unseen alerts + stored brief
  uv run python scripts/ops.py install-launchd [--dry-run] [--uninstall]   # 3 agents:
                                                   # nightly 22:45, web always-on, morning 08:00

``nightly`` is the one entry the scheduler needs: prices -> FSDS (when a new
quarter is due) -> delistings (when a quarter has passed; the ledger sets the
distress head's censor date, so nobody rebuilding it freezes that head's training
anchor) -> universe (when a month has passed) -> paper log (when a completed
month is unlogged) -> monitor -> news (watchlist headline memory,
firewalled from the signal) -> reload (hand the always-on app the new data;
without it the nightly refreshes disk into a server that never re-reads it).
Every step is idempotent, so a run missed while the machine slept simply catches
up on the next firing; a single repo-wide lock makes wake-coalesced double-fires
and manual overlap harmless. Deltas of every run land in ops_state.job_runs.
"""

import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from stockscan.config import LOGS_DIR, REPO_ROOT
from stockscan.ops.lock import JobAlreadyRunning, job_lock
from stockscan.ops.state import OpsState

LAUNCHD_LABEL = "com.stockscan.nightly"
WEB_LABEL = "com.stockscan.web"
MORNING_LABEL = "com.stockscan.morning"
NIGHTLY_HOUR, NIGHTLY_MINUTE = 22, 45
MORNING_HOUR, MORNING_MINUTE = 8, 0
UNIVERSE_DUE_DAYS = 28
DELISTINGS_DUE_DAYS = 90    # quarterly: EDGAR's master index gains a quarter that often


def _run_logged(state: OpsState, job: str, fn, *args, **kwargs) -> dict:
    """Run one job under the shared lock discipline, logging deltas + status."""
    run_id = state.job_start(job)
    try:
        deltas = fn(*args, **kwargs) or {}
    except Exception as exc:
        state.job_finish(run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    # honor an explicit _status, a {"status": "noop"} return (paper log's idempotent
    # no-op), or a {"noop": True} flag; default ok
    status = (deltas.pop("_status", None)
              or (deltas.get("status") if deltas.get("status") in ("noop", "degraded") else None)
              or ("noop" if deltas.get("noop") else "ok"))
    state.job_finish(run_id, status, deltas)
    print(f"[{job}] {status}: {json.dumps(deltas, default=str)[:600]}")
    return deltas


# --- individual jobs ---------------------------------------------------------------

def job_prices(state: OpsState) -> dict:
    from stockscan.ops.jobs import refresh_active_prices

    return _run_logged(state, "prices", refresh_active_prices)


def job_fsds(state: OpsState) -> dict:
    from stockscan.ops.jobs import ingest_new_fsds, missing_quarters

    if not missing_quarters():
        return _run_logged(state, "fsds", lambda: {"noop": True, "note": "up to date"})
    return _run_logged(state, "fsds", ingest_new_fsds)


def _universe_due(state: OpsState) -> bool:
    from stockscan.intrinio_universe import UNIVERSE_PATH

    last = state.last_run("universe", status="ok")
    if last is not None:
        anchor = pd.Timestamp(last["started"]).tz_localize(None)
    elif Path(UNIVERSE_PATH).exists():
        anchor = pd.Timestamp(os.stat(UNIVERSE_PATH).st_mtime, unit="s")
    else:
        return True
    return (pd.Timestamp.now("UTC").tz_localize(None) - anchor).days >= UNIVERSE_DUE_DAYS


def job_universe(state: OpsState) -> dict:
    from stockscan.ops.jobs import refresh_universe

    # lambda, not kwargs: _run_logged takes `state` itself, so a bare state=state
    # binds to the wrapper and never reaches refresh_universe (TypeError).
    return _run_logged(state, "universe", lambda: refresh_universe(state=state))


def _delistings_due(state: OpsState) -> bool:
    """Same shape as ``_universe_due``, one cadence slower.

    Falls back to the ledger's mtime when there is no successful run on record, so a
    machine that has been rebuilding the file by hand (which was the ONLY way until this
    stage existed) doesn't rescan sixty quarters on the first scheduled night."""
    from stockscan.edgar.delistings import LEDGER_PATH

    last = state.last_run("delistings", status="ok")
    if last is not None:
        anchor = pd.Timestamp(last["started"]).tz_localize(None)
    elif Path(LEDGER_PATH).exists():
        anchor = pd.Timestamp(os.stat(LEDGER_PATH).st_mtime, unit="s")
    else:
        return True
    return (pd.Timestamp.now("UTC").tz_localize(None) - anchor).days >= DELISTINGS_DUE_DAYS


def job_delistings(state: OpsState) -> dict:
    """Quarterly rebuild of the delist/dereg ledger — the distress head's censor date.

    Thin wrapper; the destroy-nothing rules (full range, temp-file commit, empty and
    shrunken scans rejected, .bak kept) live in ops/jobs.refresh_delistings."""
    from stockscan.ops.jobs import refresh_delistings

    return _run_logged(state, "delistings", refresh_delistings)


def _paper_due(state: OpsState) -> bool:
    from stockscan.config import PAPER_DIR
    from stockscan.panel import load_matrices_cached
    from stockscan.ops.paper import missing_paper_months

    if not (Path(PAPER_DIR) / "baseline.json").exists():
        return False
    close, _ = load_matrices_cached()
    if close.empty:
        return False
    return bool(missing_paper_months(close.index))


def job_paper_log(state: OpsState, as_of=None) -> dict:
    from stockscan.ops.paper import log_signals

    return _run_logged(state, "paper_log", log_signals, state, as_of=as_of)


def _backfill_missing_paper_months(state: OpsState) -> None:
    """Log EVERY completed month-end since the freeze that lacks a file (oldest
    first), so a multi-month outage leaves no permanent hole in the record."""
    from stockscan.ops.paper import missing_paper_months
    from stockscan.panel import load_matrices_cached

    close, _ = load_matrices_cached()
    months = missing_paper_months(close.index)
    if not months:
        print("[paper_log] up to date")
        return
    print(f"[paper_log] backfilling {len(months)} month(s): "
          f"{[str(m.date()) for m in months]}")
    for m in months:  # oldest first: the hysteresis book must be built in order
        job_paper_log(state, as_of=m)


def job_news(state: OpsState, no_llm: bool = False, backfill: int = 0) -> dict:
    """Watchlist news ingest (LIVE-VIEW ONLY — never scoring/backtest/panel).

    Fetch + dedup + extract headline/summary for every watched name into news.sqlite.
    Idempotent and quota-capped: the nightly ingest skips the Intrinio pull for a name
    fetched inside NEWS_REFETCH_HOURS, so a re-run does no network work; extraction
    (light tier, or the deterministic heuristic under --no-llm) backfills only articles
    missing the current version. ``backfill=N`` instead paginates N pages of history per
    name to SEED the memory so recall's 'notable past' has depth on day one. Runs
    independent of price freshness — news is firewalled from the signal."""
    from stockscan.intrinio_universe import load_universe
    from stockscan.narrate.llm import make_llm
    from stockscan.newsmem import (
        NewsStore,
        backfill_watchlist,
        ingest_watchlist,
        watchlist_targets,
    )

    wl = state.watchlist()
    job = "news_backfill" if backfill else "news"
    if not wl:
        return _run_logged(state, job, lambda: {"noop": True, "note": "empty watchlist"})
    targets = watchlist_targets(wl, load_universe())
    llm = None if no_llm else make_llm("light")

    def _run() -> dict:
        with NewsStore() as store:
            if backfill:
                return backfill_watchlist(store, targets, llm=llm, pages=backfill)
            return ingest_watchlist(store, targets, llm=llm)

    return _run_logged(state, job, _run)


def job_themes(state: OpsState) -> dict:
    """Auto-tag thematic markets (AI/SaaS/EV…) from Intrinio descriptions (LIVE-VIEW ONLY).

    Fetches the (cached) business description for every liquid name in the current
    cross-section and keyword-tags it into themes.sqlite for the markets page. Firewalled
    from the signal; idempotent — descriptions are cached (PROFILE_REFETCH_DAYS), so a
    re-run re-tags from cache. The FIRST run fetches the whole cross-section (~3k names),
    so it is the slow one; later runs are cheap."""
    from stockscan.serve import build_cross_section, load_serve_data
    from stockscan.themes import refresh_theme_tags

    def _build() -> dict:
        data = load_serve_data()
        cross = build_cross_section(data, data.close.index[-1])
        return refresh_theme_tags([int(c) for c in cross["cik"].tolist()])

    return _run_logged(state, "themes", _build)


def job_paper_check(state: OpsState) -> dict:
    """Grade-progress alerts: a newly scored OOS month / a degradation flip
    reaches the user as an alert instead of waiting to be looked at."""
    from stockscan.ops import paper

    def _run() -> dict:
        rep = paper.compare()
        deltas = paper.paper_progress_alerts(state, rep)
        # one markdown scorecard per scoreable month (idempotent; refreshed when the
        # running gate numbers move) — the artifact the whole experiment reports into
        deltas["reports"] = paper.write_month_reports(rep)
        return deltas

    return _run_logged(state, "paper_check", _run)


def job_backup(state: OpsState) -> dict:
    """SQLite-store + frozen-artifact backups + log rotation (see ops/housekeeping.py)."""
    from stockscan.ops.housekeeping import backup_artifacts, backup_stores, rotate_logs

    _run_logged(state, "rotate_logs", rotate_logs)
    _run_logged(state, "backup_artifacts", backup_artifacts)
    return _run_logged(state, "backup", backup_stores)


def job_reload(state: OpsState) -> dict:
    """Tell the always-on web app to rebuild its cross-section from tonight's data.

    The nightly writes files and exits; the server (launchd KeepAlive, so it can run
    for months) holds its scored cross-section in memory and rebuilds it ONLY on an
    explicit reload — see web/state.py. Without this job the two never meet: disk goes
    fresh, the app keeps serving the as-of it started with, and the liveness probe
    stays green the whole time. The 'update data' button was the only thing closing
    the loop, which made freshness depend on the owner remembering to click it.

    Best-effort by construction: the app is optional (it may simply not be running),
    so a failed poke is a 'noop', never a failed night. The reload itself is async —
    the endpoint returns as soon as the background load starts, and the health check
    that runs after this reports whether the app actually came back fresh."""
    from stockscan.config import WEB_URL

    def _run() -> dict:
        import httpx

        try:
            r = httpx.post(WEB_URL.rstrip("/") + "/api/reload", timeout=5.0)
        except Exception as exc:
            return {"noop": True, "reloaded": False,
                    "note": f"app not reachable at {WEB_URL} ({type(exc).__name__})"}
        if r.status_code != 200:
            return {"noop": True, "reloaded": False,
                    "note": f"{WEB_URL} answered {r.status_code}"}
        return {"reloaded": True, "url": WEB_URL}

    return _run_logged(state, "reload", _run)


def job_health_check(state: OpsState) -> dict:
    """The health screen at the end of the night, stored as job deltas so the
    web UI can render it (GET /api/health). prev_failing is captured BEFORE the job
    row opens — see health_record's docstring."""
    from stockscan.ops.health import health_record, run_checks

    prev_failing = set((((state.last_run("health") or {}).get("deltas")) or {})
                       .get("critical_failing", []))
    return _run_logged(
        state, "health",
        lambda: health_record(run_checks(), prev_failing, state.add_alert))


def job_digest(state: OpsState) -> dict:
    """Generate the grounded morning brief while the model is warm from the monitor
    pass, and store it so the web digest card opens instantly. Fail-open: a down or
    refusing LLM stores nothing and the card falls back to on-demand generation."""
    from stockscan.assist.brief import build_brief_context, nightly_brief
    from stockscan.narrate.llm import make_llm

    def _run() -> dict:
        ctx = build_brief_context(state)
        res = nightly_brief(ctx, make_llm("chat"))
        if res.get("refused") or not res.get("answer"):
            return {"stored": False, "refused": bool(res.get("refused")),
                    "_status": "degraded"}
        state.kv_set("digest_brief", {"answer": res["answer"],
                                      "attempts": res.get("attempts")})
        return {"stored": True, "chars": len(res["answer"])}

    return _run_logged(state, "digest", _run)


def job_judge(state: OpsState) -> dict:
    """Advisory faithfulness pass over a sample of the last day's SHOWN LLM turns
    (assist.telemetry) — paraphrase-level drift gets measured nightly instead of
    assumed. Flags raise an in-app 'judge_flag' alert; never high-severity."""
    from stockscan.assist.telemetry import judge_sample
    from stockscan.narrate.llm import make_llm

    since = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).isoformat(timespec="seconds")
    return _run_logged(state, "judge",
                       lambda: judge_sample(state, make_llm("full"), since))


def job_panels(state: OpsState, max_names: int = 25) -> dict:
    """Pre-generate analyst panels for held + watched names while the model is warm
    from the monitor pass — morning ticker pages open with all four memos waiting.
    Idempotent via the (cik, context-hash) cache: an unchanged name costs four cache
    hits, not eight model calls; refusals/suppressions are never cached, so they
    simply retry tomorrow."""
    from stockscan.assist.analyst import ROLES
    from stockscan.view.data import ArgusData

    def _run() -> dict:
        held = {int(pos["cik"]) for pos in state.positions()}
        watched = {int(w["cik"]) for w in state.watchlist()}
        ciks = sorted(held) + sorted(watched - held)   # held names first
        dropped = max(0, len(ciks) - max_names)
        ciks = ciks[:max_names]
        if not ciks:
            return {"noop": True, "note": "no held or watched names"}
        a = ArgusData.load()
        stats = {"names": len(ciks), "generated": 0, "cached": 0,
                 "refused": 0, "suppressed": 0, "errors": 0, "dropped": dropped}
        for cik in ciks:
            for role in ROLES:
                try:
                    r = a.panel_role(cik, role)
                except Exception:
                    stats["errors"] += 1
                    continue
                if r.get("cached"):
                    stats["cached"] += 1
                elif r.get("refused"):
                    stats["refused"] += 1
                elif r.get("suppressed"):
                    stats["suppressed"] += 1
                else:
                    stats["generated"] += 1
        if stats["errors"] and stats["errors"] >= stats["names"] * len(ROLES) // 2:
            stats["_status"] = "degraded"
        return stats

    return _run_logged(state, "panels", _run)


def job_monitor(state: OpsState, no_llm: bool = False, edgar: bool = True,
                alerts_ok: bool = True) -> dict:
    from stockscan.narrate.llm import make_llm
    from stockscan.ops.monitor import run_monitor

    # A degraded price night (alerts_ok=False) also disables narration: a
    # cross-section built over a half-updated store has jittered percentiles, and
    # a full-tier narration would cache against that wrong percentile and reset the
    # materiality baseline. Template runs never cache, so no_llm here is safe.
    use_llm = not no_llm and alerts_ok
    llm_full = make_llm("full") if use_llm else None
    llm_light = make_llm("light") if use_llm else None
    return _run_logged(state, "monitor", run_monitor, state,
                       llm_full=llm_full, llm_light=llm_light,
                       narrate=True, edgar=edgar, alerts_ok=alerts_ok)


def job_nightly(state: OpsState, no_llm: bool = False) -> int:
    """The scheduler entry: each stage self-checks whether it is due."""
    started_iso = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    reaped = state.reap_stale_runs()
    if reaped:
        print(f"[reap] {len(reaped)} stranded 'running' row(s) marked aborted: "
              + ", ".join(sorted({r['job'] for r in reaped})))
    deltas = job_prices(state)
    checked = max(1, deltas.get("active_columns", 1))
    # degraded if too many columns failed OR the heartbeat column itself failed
    # (then freshness is unverifiable and the whole store may be a day behind)
    degraded = (deltas.get("failed", 0) / checked > 0.02
                or not deltas.get("reference_ok", True))

    job_fsds(state)

    # before the universe refresh, which reads the ledger to date each company's death.
    # Wrapped: this stage makes ~60 throttled EDGAR fetches, and a vendor hiccup four
    # times a year must not take the rest of the night down with it — the ledger it
    # failed to refresh is exactly as stale as it was a minute earlier, and the health
    # screen at the end of the run is what says so.
    if _delistings_due(state):
        try:
            job_delistings(state)
        except Exception as exc:
            print(f"[delistings] skipped: {type(exc).__name__}: {exc}")
    else:
        print("[delistings] not due")

    if _universe_due(state):
        job_universe(state)
    else:
        print("[universe] not due")

    if _paper_due(state):
        _backfill_missing_paper_months(state)
    else:
        print("[paper_log] not due")

    # progress alerts on the paper record (newly graded OOS month, degradation flip)
    from stockscan.config import PAPER_DIR
    if (Path(PAPER_DIR) / "baseline.json").exists():
        job_paper_check(state)

    # a degraded price night suppresses percentile alerts AND narration: ranks over
    # a half-updated store fire false alerts, then fire them in reverse on recovery
    job_monitor(state, no_llm=no_llm, alerts_ok=not degraded)
    # news ingest is firewalled from the signal, so a degraded price night doesn't
    # gate it; it just refreshes the watchlist's headline memory (quota-capped)
    job_news(state, no_llm=no_llm)
    # theme tags are firewalled + description-cached (re-tag is cheap after the first
    # build); refresh so new/changed names flow into the markets page's AI/SaaS/EV groups
    job_themes(state)
    # every facade-visible store is now current on disk, so hand the running app its
    # new cross-section. Async and best-effort: it returns once the background load
    # starts, and the health screen below is what actually confirms the app came back
    # fresh. Placed before the slow tail (backup/digest/panels) so it has finished by
    # then. Never fails the night — the app is optional.
    try:
        job_reload(state)
    except Exception as exc:
        print(f"[reload] skipped: {type(exc).__name__}: {exc}")
    # housekeeping last: snapshot the stores AFTER tonight's writes, rotate fat logs
    job_backup(state)

    # grounded morning brief while the model is warm; stored for the web digest card.
    # fail-open twice over: job_digest degrades on refusal, and a crash here must
    # never fail the night that just did the real work
    if not no_llm:
        try:
            job_digest(state)
        except Exception as exc:
            print(f"[digest] skipped: {type(exc).__name__}: {exc}")
        # advisory faithfulness pass over yesterday's shown answers (sample of 3)
        try:
            job_judge(state)
        except Exception as exc:
            print(f"[judge] skipped: {type(exc).__name__}: {exc}")

    # analyst panels for held/watched names while the model is warm — a degraded
    # price night skips this (jittered percentiles would be cached into the memos
    # until the next context change; same policy as narration)
    if not no_llm and not degraded:
        try:
            job_panels(state)
        except Exception as exc:
            print(f"[panels] skipped: {type(exc).__name__}: {exc}")

    # health screen over tonight's end state — cheap; newly-failing criticals become
    # alerts (picked up by the notification below). Same never-fail-the-night wrap.
    try:
        job_health_check(state)
    except Exception as exc:
        print(f"[health] skipped: {type(exc).__name__}: {exc}")

    status = "degraded" if degraded else "ok"
    run_id = state.job_start("nightly")
    state.job_finish(run_id, status,
                     {"prices_failed_frac": round(deltas.get("failed", 0) / checked, 4),
                      "reference_ok": deltas.get("reference_ok", True)})

    # the one human-facing step: a single local notification summarizing the night
    # (deterministic — no LLM in the delivery path; high-severity kinds only)
    from stockscan.ops.notify import deliver_nightly

    new_alerts = [a for a in state.alerts(unseen_only=True, limit=500)
                  if a["created"] >= started_iso]
    _run_logged(state, "notify", deliver_nightly, status, new_alerts)
    return 0


# --- launchd ------------------------------------------------------------------------

def _plists() -> dict[str, dict]:
    """The three agents of daily operation: the 22:45 nightly (data + alerts +
    reports), the always-on web app (KeepAlive — Argus is simply THERE at
    localhost:8000), and the 08:00 morning banner (the nightly's own banner fires
    ~23:30 and expires unseen; this one lands when the human does)."""
    uv = shutil.which("uv") or "/usr/local/bin/uv"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    common = {"WorkingDirectory": str(REPO_ROOT), "ProcessType": "Background"}
    return {
        LAUNCHD_LABEL: {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": [uv, "run", "python", "scripts/ops.py", "nightly"],
            "StartCalendarInterval": {"Hour": NIGHTLY_HOUR, "Minute": NIGHTLY_MINUTE},
            "StandardOutPath": str(LOGS_DIR / "nightly.log"),
            "StandardErrorPath": str(LOGS_DIR / "nightly.err.log"),
            **common,
        },
        WEB_LABEL: {
            "Label": WEB_LABEL,
            "ProgramArguments": [uv, "run", "python", "scripts/argus_web.py"],
            "RunAtLoad": True,
            "KeepAlive": True,   # crash or reboot -> it comes back on its own
            "StandardOutPath": str(LOGS_DIR / "web.log"),
            "StandardErrorPath": str(LOGS_DIR / "web.err.log"),
            **common,
        },
        MORNING_LABEL: {
            "Label": MORNING_LABEL,
            "ProgramArguments": [uv, "run", "python", "scripts/ops.py", "morning"],
            "StartCalendarInterval": {"Hour": MORNING_HOUR, "Minute": MORNING_MINUTE},
            "StandardOutPath": str(LOGS_DIR / "morning.log"),
            "StandardErrorPath": str(LOGS_DIR / "morning.err.log"),
            **common,
        },
    }


def install_launchd(dry_run: bool = False, uninstall: bool = False) -> int:
    domain = f"gui/{os.getuid()}"
    worst = 0
    for label, payload in _plists().items():
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        if uninstall:
            if dry_run:
                print(f"would: launchctl bootout {domain}/{label}; rm {plist_path}")
                continue
            subprocess.run(["launchctl", "bootout", f"{domain}/{label}"],
                           capture_output=True)
            plist_path.unlink(missing_ok=True)
            print(f"uninstalled {label}")
            continue
        if dry_run:
            print(f"would write {plist_path}:\n{plistlib.dumps(payload).decode()}")
            print(f"would: launchctl bootstrap {domain} {plist_path}")
            continue
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_bytes(plistlib.dumps(payload))
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"],
                       capture_output=True)  # replace an older registration quietly
        res = subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            print(f"launchctl bootstrap failed for {label} ({res.returncode}): "
                  f"{res.stderr.strip()}\nplist written to {plist_path}; load manually:\n"
                  f"  launchctl bootstrap {domain} {plist_path}")
            worst = 1
            continue
        print(f"installed + loaded {label}")
    if not uninstall and not dry_run and worst == 0:
        print(f"nightly {NIGHTLY_HOUR:02d}:{NIGHTLY_MINUTE:02d} · web always-on "
              f"(http://127.0.0.1:8000) · morning banner {MORNING_HOUR:02d}:"
              f"{MORNING_MINUTE:02d} · logs in {LOGS_DIR}")
    return worst


# --- CLI ------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prices")
    sub.add_parser("fsds")
    sub.add_parser("universe")
    sub.add_parser("delistings")
    p = sub.add_parser("monitor")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-edgar", action="store_true")
    p = sub.add_parser("news")
    p.add_argument("--no-llm", action="store_true", help="heuristic extraction only")
    sub.add_parser("themes")
    sub.add_parser("reload")
    p = sub.add_parser("nightly")
    p.add_argument("--no-llm", action="store_true")

    p = sub.add_parser("watch")
    p.add_argument("action", choices=["add", "rm", "ls"])
    p.add_argument("company", nargs="?", help="ticker, TICKER~CIK, or CIK")
    p.add_argument("--note", default="")

    p = sub.add_parser("alerts")
    p.add_argument("--all", action="store_true", help="include already-seen alerts")
    p.add_argument("--keep-unseen", action="store_true",
                   help="don't mark the listed alerts as seen")

    p = sub.add_parser("paper")
    p.add_argument("action", choices=["freeze", "log", "compare", "retrain-record"])
    p.add_argument("--as-of", default=None, help="(log) month-end override")
    p.add_argument("--reason", default="", help="(retrain-record) why the retrain")

    sub.add_parser("health")
    sub.add_parser("backup")
    sub.add_parser("digest")
    sub.add_parser("morning")
    lt = sub.add_parser("llm-turns")
    lt.add_argument("--days", type=int, default=7)
    p = sub.add_parser("install-launchd")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--uninstall", action="store_true")
    p = sub.add_parser("models", help="probe this machine, recommend serving models")
    p.add_argument("--json", action="store_true")
    p.add_argument("--offline", action="store_true",
                   help="skip the registry lookup — size only what is installed")

    args = ap.parse_args(argv)

    if args.cmd == "health":
        from stockscan.ops.health import report, run_checks

        text, code = report(run_checks())
        print("stockscan health:")
        print(text)
        return code

    if args.cmd == "models":
        from stockscan.narrate.hardware import fit_payload, fit_report

        if args.json:
            print(json.dumps(fit_payload(network=not args.offline), indent=2))
            return 0
        text, code = fit_report(network=not args.offline)
        print("argus model fit:")
        print(text)
        return code   # non-zero when a CONFIGURED model isn't installed

    if args.cmd == "morning":
        from stockscan.ops.notify import deliver_morning

        with OpsState() as state:
            _run_logged(state, "morning", deliver_morning, state)
        return 0   # a missed banner is logged 'degraded', never a failed exit

    if args.cmd == "digest":
        from stockscan.assist.brief import build_brief_context, nightly_brief
        from stockscan.narrate.llm import make_llm

        with OpsState() as state:
            ctx = build_brief_context(state)
        # chat tier: capped + reasoning-off — same client the web digest card uses
        print(nightly_brief(ctx, make_llm("chat"))["answer"])
        return 0

    if args.cmd == "llm-turns":
        since = (pd.Timestamp.now("UTC")
                 - pd.Timedelta(days=args.days)).isoformat(timespec="seconds")
        with OpsState() as state:
            s = state.llm_turn_stats(since)
            print(f"last {args.days}d: {s['turns']} turns · {s['refused']} refused · "
                  f"avg {s['avg_latency_ms']}ms · max {s['max_latency_ms']}ms · "
                  f"{s['judged_with_issues']} judged-with-issues")
            rows = state._db.execute(
                "select ts, surface, refused, latency_ms, judged, judge_issues, "
                "question from llm_turns where ts >= ? order by id desc limit 30",
                (since,)).fetchall()
            for r in rows:
                mark = "REFUSED" if r[2] else ("ISSUES" if r[4] and r[5] not in
                                               (None, "[]") else "ok")
                print(f"  {r[0][:16]}  {r[1]:<9} {mark:<8} {r[3] or 0:>6}ms  "
                      f"{(r[6] or '')[:60]}")
        return 0

    if args.cmd == "install-launchd":
        return install_launchd(dry_run=args.dry_run, uninstall=args.uninstall)

    # everything below touches shared state — one repo-wide lock
    try:
        with job_lock("ops"):
            with OpsState() as state:
                if args.cmd == "prices":
                    job_prices(state)
                elif args.cmd == "fsds":
                    job_fsds(state)
                elif args.cmd == "universe":
                    job_universe(state)
                elif args.cmd == "delistings":
                    job_delistings(state)   # manual override: always rebuilds
                elif args.cmd == "monitor":
                    job_monitor(state, no_llm=args.no_llm, edgar=not args.no_edgar)
                elif args.cmd == "news":
                    job_news(state, no_llm=args.no_llm)
                elif args.cmd == "themes":
                    job_themes(state)
                elif args.cmd == "reload":
                    job_reload(state)
                elif args.cmd == "backup":
                    job_backup(state)
                elif args.cmd == "nightly":
                    return job_nightly(state, no_llm=args.no_llm)
                elif args.cmd == "watch":
                    return cmd_watch(state, args)
                elif args.cmd == "alerts":
                    return cmd_alerts(state, args)
                elif args.cmd == "paper":
                    return cmd_paper(state, args)
        return 0
    except JobAlreadyRunning:
        print("another ops run is active; nothing to do (the lock holder is doing it)")
        return 0


def cmd_watch(state: OpsState, args) -> int:
    if args.action == "ls":
        rows = state.watchlist()
        if not rows:
            print("watchlist empty — add with: ops.py watch add TICKER")
        for w in rows:
            print(f"  cik {w['cik']:>8}  {w['column'] or '?':<14} "
                  f"added {w['added'][:10]}  {w['note']}")
        return 0
    if not args.company:
        print("watch add/rm needs a company")
        return 1
    from stockscan.intrinio_universe import universe_ticker_map
    from stockscan.serve import resolve_company

    cik, column = resolve_company(
        int(args.company) if args.company.isdigit() else args.company,
        universe_ticker_map())
    if args.action == "add":
        state.watch_add(cik, column, args.note)
        print(f"watching cik {cik} ({column or 'no price column'})")
    else:
        state.watch_remove(cik)
        print(f"removed cik {cik}")
    return 0


def cmd_alerts(state: OpsState, args) -> int:
    rows = state.alerts(unseen_only=not args.all)
    if not rows:
        print("no " + ("" if args.all else "unseen ") + "alerts")
        return 0
    for a in rows:
        mark = " " if a["seen"] else "*"
        print(f" {mark} [{a['id']:>4}] {a['created'][:16]}  {a['kind']:<22} {a['message']}")
    if not args.all and not args.keep_unseen:
        state.mark_alerts_seen([a["id"] for a in rows])
    return 0


def cmd_paper(state: OpsState, args) -> int:
    from stockscan.ops import paper

    if args.action == "freeze":
        res = paper.freeze_baseline()
        print(f"baseline {res['status']}: artifact {res['artifact']['hash']} "
              f"(trained through {res['artifact']['trained_through']}), "
              f"frozen {res['frozen_on']}")
        return 0
    if args.action == "log":
        if args.as_of:  # explicit month
            res = job_paper_log(state, as_of=args.as_of)
            return 0 if res.get("status") in ("logged", "noop") else 1
        _backfill_missing_paper_months(state)  # all completed months since the freeze
        return 0
    if args.action == "compare":
        rep = paper.compare()
        print(json.dumps(rep, indent=2, default=str))
        return 0
    if args.action == "retrain-record":
        entry = paper.record_retrain(args.reason)
        print(f"registered vintage {entry['hash']} "
              f"(trained through {entry['trained_through']}): {entry['reason']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
