"""Idempotent scheduled ingestion jobs: nightly prices, quarterly FSDS, universe refresh.

Every job here is safe to re-run at any time: work already done is detected and
skipped, partial failures leave the previous good state untouched (atomic file
replaces only), and each run reports a deltas dict of what actually changed —
recorded to ops_state.job_runs by the CLI wrapper.

Two of these jobs are FULL REBUILDS rather than appends, for the same reason in
different clothing: the source rewrites history, so an incremental update would bake a
seam into it.

Nightly prices are a FULL-HISTORY refetch per active column, not an incremental
append. Adjusted series rebase retroactively whenever a split or dividend lands
(the vendor rescales all history), so appending new rows onto an old file would
manufacture a scale break at the seam — the exact artifact class the Phase-3
repair fought. A security's full daily history fits in one API page, so the
refetch costs the same request count as an increment and heals vendor revisions
for free. Dead columns never print new bars and are skipped entirely; deaths and
renames are the universe-refresh job's business.

The quarterly delisting-ledger refresh is a full rebuild because the writer keeps the
EARLIEST event per CIK over exactly the quarters it was handed and overwrites the file
with that result — see :func:`refresh_delistings`.
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

from ..config import INTRINIO_API_KEY
from ..edgar.delistings import LEDGER_PATH, build_delisting_ledger
from ..edgar.fsds import FUNDAMENTALS_DIR, ingest_quarter, iter_quarters, parse_quarter
from ..intrinio_universe import UNIVERSE_PATH, load_universe, select_universe
from ..panel import load_matrices, save_matrix_cache
from ..prices import (
    PRICES_DIR,
    IntrinioTransientError,
    _intrinio_price_rows,
    _intrinio_to_tidy,
    download_intrinio_universe,
)

def columns_missing_unadjusted(universe: pd.DataFrame, out_dir: Path = PRICES_DIR) -> list[str]:
    """Existing price files written before the uclose/uvolume schema addition.

    The data-layer backlog fix (unadjusted liquidity floors) needs the raw close;
    files fetched pre-schema lack it. This lists them so a backfill can refetch
    ONLY those, resumably, without re-pulling the whole universe.
    """
    import pyarrow.parquet as pq

    out = []
    for col in universe["column"].unique():
        p = Path(out_dir) / f"{col}.parquet"
        if not p.exists():
            continue
        try:
            if "uclose" not in pq.ParquetFile(p).schema.names:
                out.append(col)
        except Exception:
            out.append(col)  # unreadable -> refetch
    return out


FIRST_QUARTER = "2011q1"          # the fundamentals backfill horizon
PRICE_START = "2011-01-01"
REVISION_CHECK_DAYS = 30          # trailing shared days compared old-vs-new (informational)


# --- nightly prices ---------------------------------------------------------------

def _active_groups(universe: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(column, candidate rows) for companies with at least one ACTIVE security."""
    out = []
    for col, g in universe.groupby("column", sort=True):
        if bool(g["active"].any()):
            out.append((col, g.sort_values("priority")))
    return out


def _fetch_column_frame(client, g: pd.DataFrame, column: str, start, end, api_key: str,
                        pause: float) -> pd.DataFrame | None:
    """Fetch + splice one company's candidate securities (by id). None = no data.

    Raises IntrinioTransientError if any candidate fails transiently — a partial
    splice must never replace a good file.
    """
    frames = []
    for prio, sec_id in zip(g["priority"], g["security_id"]):
        rows = _intrinio_price_rows(client, sec_id, start, end, api_key)
        if pause:
            time.sleep(pause)
        if not rows:
            continue
        tidy = _intrinio_to_tidy(rows, column)
        tidy["ticker"] = column
        tidy["_prio"] = prio
        frames.append(tidy)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "_prio"]).drop_duplicates("date", keep="first")
    return df.drop(columns="_prio").sort_values("date").reset_index(drop=True)


def _replace_verdict(old: pd.DataFrame | None, new: pd.DataFrame | None) -> str:
    """Decide whether a freshly fetched frame may replace the stored one.

    'replace' | 'suspect' (new frame lost history — keep the old file and flag it)
    | 'empty' (nothing fetched for a column with no stored file).
    A shrunken response usually means a vendor-side gap or partial outage; blindly
    replacing would destroy good history that tomorrow's run can't restore.
    """
    if new is None or new.empty:
        return "empty" if old is None else "suspect"
    if old is None or old.empty:
        return "replace"
    if new["date"].max() < old["date"].max():
        return "suspect"
    if len(new) < 0.9 * len(old):
        return "suspect"
    return "replace"


def _revised(old: pd.DataFrame, new: pd.DataFrame, days: int = REVISION_CHECK_DAYS) -> bool:
    """Did the vendor change already-stored closes (revision or adjustment rebase)?"""
    o = old.set_index("date")["close"].tail(days)
    n = new.set_index("date")["close"]
    shared = o.index.intersection(n.index)
    if shared.empty:
        return False
    diff = (o.loc[shared] - n.loc[shared]).abs()
    scale = o.loc[shared].abs().clip(lower=1e-9)
    return bool((diff / scale > 1e-6).any())


def _refresh_one_column(client, col: str, g: pd.DataFrame, start, end, api_key: str,
                        pause: float, out_dir: Path, target_date) -> tuple[str, dict]:
    """Refetch one active column. Returns (status, info)."""
    path = out_dir / f"{col}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and target_date is not None and not old.empty:
        if pd.Timestamp(old["date"].max()) >= pd.Timestamp(target_date):
            return "fresh", {}
    try:
        new = _fetch_column_frame(client, g, col, start, end, api_key, pause)
    except IntrinioTransientError:
        return "failed", {}
    verdict = _replace_verdict(old, new)
    if verdict != "replace":
        return verdict, {}
    # Identical content (and already on the current schema) -> no write, so a
    # same-night re-run doesn't churn mtimes or trigger a matrix-cache rebuild.
    if (
        old is not None
        and "uclose" in old.columns
        and len(old) == len(new)
        and old["date"].equals(new["date"])
        and old["close"].equals(new["close"])
    ):
        return "fresh", {}
    info = {
        "new_rows": int(len(new) - (len(old) if old is not None else 0)),
        "revised": bool(old is not None and _revised(old, new)),
    }
    tmp = out_dir / f".{col}.parquet.tmp"
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return "written", info


def refresh_active_prices(
    universe: pd.DataFrame | None = None,
    start=PRICE_START,
    end=None,
    api_key: str = "",
    pause: float = 0.15,
    workers: int = 4,
    out_dir: Path = PRICES_DIR,
    reference_column: str = "AAPL",
    rebuild_cache: bool = True,
    transport: httpx.BaseTransport | None = None,
    log_every: int = 500,
) -> dict:
    """Nightly job: full-history refetch of every ACTIVE universe column.

    Idempotent via the data itself: the reference column is fetched first and its
    max date becomes the session target; any column already at the target is
    skipped without a request, so a re-run after a partial failure only touches
    what's left. Returns the deltas dict for the job log.
    """
    api_key = api_key or INTRINIO_API_KEY
    if not api_key:
        raise ValueError("Intrinio key missing; set STOCKSCAN_INTRINIO_KEY")
    universe = universe if universe is not None else load_universe()
    if universe.empty:
        raise FileNotFoundError("no universe map; run scripts/build_intrinio_universe.py")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _active_groups(universe)

    counts = {"written": 0, "fresh": 0, "suspect": 0, "empty": 0, "failed": 0}
    revised = 0
    new_rows = 0
    suspects: list[str] = []

    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0,
                      transport=transport) as client:
        # Establish the target trading date from the reference column (a liquid
        # name whose bar is the market calendar's heartbeat).
        target_date = None
        ref_ok = False
        ref = [(c, g) for c, g in groups if c == reference_column]
        rest = [(c, g) for c, g in groups if c != reference_column]
        for col, g in ref:
            status, info = _refresh_one_column(
                client, col, g, start, end, api_key, pause, out_dir, None)
            counts[status] += 1
            if status == "written":
                new_rows += info["new_rows"]
                revised += int(info["revised"])
            # Only adopt the target date when the reference bar is actually CURRENT
            # (written this run, or already fresh). If it failed/suspect, its file
            # holds a stale date; adopting it would short-circuit every other column
            # to 'fresh' and strand the whole universe a day behind while reporting
            # success. Leave target_date=None so the rest is fetched normally.
            ref_path = out_dir / f"{col}.parquet"
            if status in ("written", "fresh") and ref_path.exists():
                target_date = pd.read_parquet(ref_path, columns=["date"])["date"].max()
                ref_ok = True

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {
                pool.submit(_refresh_one_column, client, col, g, start, end,
                            api_key, pause, out_dir, target_date): col
                for col, g in rest
            }
            for i, fut in enumerate(as_completed(futs)):
                try:
                    status, info = fut.result()
                except Exception as e:  # one bad column must not strand the run
                    print(f"column {futs[fut]} raised {type(e).__name__}: {e}", flush=True)
                    status, info = "failed", {}
                counts[status] += 1
                if status == "written":
                    new_rows += info["new_rows"]
                    revised += int(info["revised"])
                elif status == "suspect":
                    suspects.append(futs[fut])
                if log_every and (i + 1) % log_every == 0:
                    print(f"[{i + 1}/{len(rest)}] {counts}", flush=True)

    deltas = {
        "active_columns": len(groups),
        "target_date": str(pd.Timestamp(target_date).date()) if target_date is not None else None,
        "reference_ok": ref_ok,
        **counts,
        "net_new_rows": new_rows,
        "revised_columns": revised,
        "suspect_columns": suspects[:20],
    }
    if not ref_ok:
        # the heartbeat column failed -> we could not set a session target, so the
        # run's freshness is unverifiable; flag it degraded for the nightly gate
        deltas["_status"] = "degraded"
    # rebuild the cache whenever something changed OR the cache is stale/missing
    # (a prior universe refresh invalidates the manifest without any write here)
    from ..panel import matrix_cache_fresh

    if rebuild_cache and (counts["written"] or not matrix_cache_fresh(prices_dir=out_dir)):
        close, dv = load_matrices(prices_dir=out_dir)
        save_matrix_cache(close, dv, prices_dir=out_dir)
        deltas["matrix_cache"] = {"max_date": str(close.index.max().date()),
                                  "n_columns": int(close.shape[1])}
    return deltas


# --- quarterly FSDS ------------------------------------------------------------------

def quarters_present(fundamentals_dir: Path = FUNDAMENTALS_DIR) -> list[str]:
    """Quarters with a READABLE parquet on disk. The file's existence is the ingest
    checkpoint, so an unreadable file (pre-atomic-write crash damage) must count as
    missing — otherwise it pins a corrupt quarter forever."""
    import pyarrow.parquet as pq

    out = []
    for p in sorted(Path(fundamentals_dir).glob("*.parquet")):
        try:
            pq.ParquetFile(p)
            out.append(p.stem)
        except Exception:
            print(f"unreadable quarter file (will re-ingest): {p}", flush=True)
    return out


def latest_elapsed_quarter(today=None) -> str:
    """The most recent quarter that has fully ENDED (its FSDS may not be out yet)."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    y, q = t.year, (t.month - 1) // 3 + 1
    y, q = (y - 1, 4) if q == 1 else (y, q - 1)
    return f"{y}q{q}"


def missing_quarters(today=None, fundamentals_dir: Path = FUNDAMENTALS_DIR,
                     first: str = FIRST_QUARTER) -> list[str]:
    have = set(quarters_present(fundamentals_dir))
    return [q for q in iter_quarters(first, latest_elapsed_quarter(today)) if q not in have]


def ingest_new_fsds(today=None, fundamentals_dir: Path = FUNDAMENTALS_DIR,
                    ingest_fn=None, rebuild_wide: bool = True,
                    first: str = FIRST_QUARTER) -> dict:
    """Quarterly job: ingest any FSDS quarter that has elapsed but isn't on disk.

    A quarter that isn't published yet 404s — recorded as 'waiting', not a
    failure; the weekly schedule simply retries until DERA posts it. After any
    successful ingest the wide fundamentals table is rebuilt (that is the moment
    new filings become visible to the serve path and the monitor).
    """
    ingest_fn = ingest_fn or ingest_quarter
    todo = missing_quarters(today, fundamentals_dir, first)
    newest = todo[-1] if todo else None
    ingested, waiting, failed = [], [], []
    rows = 0
    for q in todo:
        parse_quarter(q)
        try:
            summ = ingest_fn(q)
            ingested.append(q)
            rows += int(summ.get("rows", 0))
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                waiting.append(q)   # not published yet
            else:
                failed.append(q)
        except (httpx.HTTPError, RuntimeError, OSError):
            # EdgarClient.download retries 403/5xx and raises a generic RuntimeError
            # with no status attached. For the NEWEST elapsed quarter that pattern is
            # overwhelmingly "not published yet" (SEC serves some missing static
            # paths as 403); an OLDER missing quarter is a real gap worth alarming.
            (waiting if q == newest else failed).append(q)
    deltas: dict = {"ingested": ingested, "fact_rows": rows,
                    "waiting": waiting, "failed": failed}
    if ingested and rebuild_wide:
        from ..concepts import build_fundamentals_wide

        deltas["wide_rows"] = int(build_fundamentals_wide())
    return deltas


# --- delisting ledger (quarterly) --------------------------------------------------

FIRST_DELIST_QUARTER = "2011q1"   # the ledger's fixed horizon — NEVER move this forward


def current_quarter(today=None) -> str:
    """The quarter currently IN PROGRESS (contrast :func:`latest_elapsed_quarter`).

    EDGAR's full-index gains its ``QTR`` directory on the first day of a quarter and
    fills it as filings land, so the in-progress quarter is scannable today — unlike
    FSDS, which only publishes weeks after a quarter has ended."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return f"{t.year}q{(t.month - 1) // 3 + 1}"


def _ledger_stats(path: Path) -> dict | None:
    """``{"rows", "max_event"}`` for an existing ledger; None if absent or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["cik", "delist_date"])
    except Exception:
        return None
    if df.empty:
        return {"rows": 0, "max_event": None}
    return {"rows": int(len(df)),
            "max_event": str(pd.Timestamp(df["delist_date"].max()).date())}


def refresh_delistings(today=None, ledger_path: Path = LEDGER_PATH, build_fn=None,
                       first: str = FIRST_DELIST_QUARTER) -> dict:
    """Quarterly job: rebuild the delisting/deregistration ledger over its FULL range.

    Nothing scheduled this before 2026-08: ``scripts/build_delistings.py`` was a manual
    command, so the ledger aged silently — and it is not an inert reference file.
    :func:`stockscan.distress.build_distress_panel` defaults its ``censor_date`` to the
    ledger's newest event and derives ``trained_through = censor - horizon``, so a stale
    ledger pins the distress head's training anchor wherever it was last built. Observed
    2026-08-19: the file had gone unbuilt since 2026-07-01 and its newest event was
    2026-03-31 — the exact censor date already frozen into the distress artifact, so a
    retrain that day would have reproduced its dates and changed nothing.

    QUARTERLY because that is the cadence of the source: a new ``master.idx`` directory
    appears once per quarter, and the in-progress one accrues filings continuously.

    THE FULL RANGE, EVERY TIME. ``build_delisting_ledger`` keeps the earliest event per
    CIK across exactly the quarters it is handed and then ``to_parquet``s the result over
    the destination — it does not merge with what is already there. Handing it only the
    new quarters would not extend the ledger, it would REPLACE fifteen years of history
    with one quarter of it. So the range is always ``first``..:func:`current_quarter`,
    computed here rather than typed by a human — the other half of the 2026-08 staleness
    was the end of the range, not just the date of the run: a rebuild done in 2026q3 left
    a newest event of 2026-03-31, i.e. it was handed a range ending two quarters back.

    Three guards keep a bad scan from destroying a good file:

    * the rebuild is written to a temp path and ``os.replace``d in — a scan that raises
      part-way through never touches the live ledger;
    * an EMPTY scan (0 events) is a failure, not an empty ledger. Every real scan of
      2011..now returns five figures; zero means EDGAR served us nothing useful;
    * a SHRUNKEN scan is a failure too. The range only ever grows and ``master.idx``
      files are immutable, so a rebuild covering the same start date cannot legitimately
      know about fewer CIKs than the file it would overwrite.

    A ``<name>.bak`` copy of the previous ledger is kept beside it regardless, so even a
    rebuild that passes every guard and is still wrong stays recoverable until the next
    quarterly run.

    Returns a deltas dict; ``wrote`` says whether the live file changed. Cost: one
    throttled EDGAR fetch per quarter (~60 and growing), so a few minutes, four times
    a year.
    """
    build_fn = build_fn or build_delisting_ledger
    ledger_path = Path(ledger_path)
    quarters = iter_quarters(first, current_quarter(today))
    before = _ledger_stats(ledger_path)
    deltas: dict = {"quarters": len(quarters), "range": f"{quarters[0]}:{quarters[-1]}",
                    "previous_rows": (before or {}).get("rows"),
                    "previous_max_event": (before or {}).get("max_event")}

    if ledger_path.exists():
        try:
            shutil.copy2(ledger_path, ledger_path.with_suffix(ledger_path.suffix + ".bak"))
            deltas["backed_up"] = True
        except OSError as exc:
            deltas["backed_up"] = False
            deltas["backup_error"] = str(exc)

    tmp = ledger_path.with_name("." + ledger_path.name + ".tmp")
    try:
        rows = int(build_fn(quarters, out_path=tmp))
        if rows == 0:
            deltas.update(wrote=False, rows=0, _status="degraded",
                          note="scan returned 0 events — ledger left untouched")
            return deltas
        floor = (before or {}).get("rows") or 0
        if rows < floor:
            deltas.update(wrote=False, rows=rows, _status="degraded",
                          note=f"scan returned {rows:,} ciks vs {floor:,} on disk — a "
                               f"full-range rebuild cannot shrink; ledger left untouched")
            return deltas
        os.replace(tmp, ledger_path)
    finally:
        Path(tmp).unlink(missing_ok=True)   # a raised scan leaves no half-built file

    after = _ledger_stats(ledger_path) or {}
    deltas.update(wrote=True, rows=rows, max_event=after.get("max_event"),
                  added=rows - ((before or {}).get("rows") or 0))
    return deltas


# --- universe refresh ------------------------------------------------------------------

def universe_diff(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Per-company changes between two universe maps (top-priority row each).

    Returns dict with ``added`` (new ciks), ``died`` (active -> dead: column gains
    the ~CIK suffix), ``revived``, ``renamed`` (same aliveness, new column), and
    ``renames`` [(old_column, new_column)] covering all three rename classes.
    """
    o = old.sort_values("priority").drop_duplicates("cik").set_index("cik")["column"] \
        if len(old) else pd.Series(dtype=object)
    n = new.sort_values("priority").drop_duplicates("cik").set_index("cik")["column"] \
        if len(new) else pd.Series(dtype=object)
    added = sorted(set(n.index) - set(o.index))
    common = n.index.intersection(o.index)
    changed = [c for c in common if o[c] != n[c]]
    died = [c for c in changed if "~" not in o[c] and "~" in n[c]]
    revived = [c for c in changed if "~" in o[c] and "~" not in n[c]]
    renamed = [c for c in changed if c not in died and c not in revived]
    return {
        "added": added,
        "died": died,
        "revived": revived,
        "renamed": renamed,
        "renames": [(o[c], n[c]) for c in changed],
    }


def apply_renames(renames: list[tuple[str, str]], out_dir: Path = PRICES_DIR) -> dict:
    """Carry price files across column renames (death, relist, ticker change).

    History must survive the rename — the file IS the checkpoint. Two subtleties
    both found by review:

    - The parquet's INTERNAL ``ticker`` column is what the matrix pivot keys on,
      so a bare file rename would leave the matrix column under the OLD name (and
      a later company reusing the ticker would silently average into it). The
      rename therefore rewrites the ticker column, atomically.
    - Re-runs after a crash: source already gone + target present = done (skip).
      Source AND target present (e.g. a death refetch already wrote the new
      column, or a previous partial run): keep whichever file carries the longer
      history, never silently clobber the fuller one.
    """
    moved, skipped, kept_target = [], [], []
    for old_col, new_col in renames:
        src = Path(out_dir) / f"{old_col}.parquet"
        dst = Path(out_dir) / f"{new_col}.parquet"
        if not src.exists():
            skipped.append(old_col)  # never fetched, or already moved by a prior run
            continue
        if dst.exists():
            try:
                src_max = pd.read_parquet(src, columns=["date"])["date"].max()
                dst_max = pd.read_parquet(dst, columns=["date"])["date"].max()
            except Exception:
                src_max, dst_max = pd.Timestamp.min, pd.Timestamp.max  # keep readable dst
            if dst_max >= src_max:
                src.unlink()         # target already carries the fuller history
                kept_target.append((old_col, new_col))
                continue
        df = pd.read_parquet(src)
        df["ticker"] = new_col
        tmp = dst.with_name("." + dst.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, dst)
        src.unlink()
        moved.append((old_col, new_col))
    return {"moved": moved, "missing": skipped, "kept_target": kept_target}


DEATH_GRACE_DAYS = 120   # recently-dead columns keep getting refetched this long
                         # (late OTC prints / vendor backfills inside open windows)


def refetch_columns(universe: pd.DataFrame, columns: list[str], api_key: str,
                    out_dir: Path, start=PRICE_START, pause: float = 0.15,
                    workers: int = 2,
                    transport: httpx.BaseTransport | None = None) -> dict:
    """Full re-splice fetch of specific columns (ALL candidate securities, by id).

    Used when a company dies — its OTC-afterlife securities enter the candidate
    list only at that point, so a file rename alone would truncate the death
    decline — and for the post-death grace window. Sanity-guarded like the
    nightly job: a shrunken response never replaces a fuller file.
    """
    sub = universe[universe["column"].isin(set(columns))]
    written, suspect, failed = [], [], []
    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0,
                      transport=transport) as client:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {
                pool.submit(_refresh_one_column, client, col, g.sort_values("priority"),
                            start, None, api_key, pause, out_dir, None): col
                for col, g in sub.groupby("column", sort=True)
            }
            for fut in as_completed(futs):
                try:
                    status, _ = fut.result()
                except Exception:
                    status = "failed"
                if status == "written":
                    written.append(futs[fut])
                elif status == "suspect":
                    suspect.append(futs[fut])
                elif status == "failed":
                    failed.append(futs[fut])
    return {"written": written, "suspect": suspect, "failed": failed}


def recent_dead_columns(universe: pd.DataFrame, out_dir: Path = PRICES_DIR,
                        today=None, grace_days: int = DEATH_GRACE_DAYS) -> list[str]:
    """Dead columns whose last stored bar is inside the grace window."""
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    cols = universe.loc[universe["column"].str.contains("~"), "column"].unique()
    out = []
    for col in cols:
        p = Path(out_dir) / f"{col}.parquet"
        if not p.exists():
            continue
        try:
            last = pd.read_parquet(p, columns=["date"])["date"].max()
        except Exception:
            continue
        if pd.notna(last) and (t - pd.Timestamp(last)).days <= grace_days:
            out.append(col)
    return out


def refresh_universe(
    api_key: str = "",
    fetch_new: bool = True,
    out_dir: Path = PRICES_DIR,
    universe_path: Path = UNIVERSE_PATH,
    state=None,
    pause: float = 0.15,
    workers: int = 4,
    transport: httpx.BaseTransport | None = None,
    rebuild_cache: bool = True,
    today=None,
) -> dict:
    """Monthly job: re-enumerate the Intrinio security master, diff, apply.

    Ordering is the crash-safety story (review-hardened):
      1. enumerate + diff (no side effects)
      2. deaths: full re-splice refetch under the NEW column name (the OTC
         afterlife securities only become candidates now)
      3. renames: internal-ticker rewrite + move (idempotent; keeps fuller file)
      4. new companies: full fetch
      5. universe parquet replaced LAST — the commit point; a crash before this
         leaves the old universe consistent with the old file names
      6. matrix cache invalidated (renames don't touch mtimes; the manifest
         check catches it, this makes it explicit)
    """
    import duckdb

    from ..concepts import WIDE_PATH
    from ..edgar.delistings import load_delistings
    from ..intrinio_universe import enumerate_companies, enumerate_securities
    from ..panel import matrix_cache_paths

    api_key = api_key or INTRINIO_API_KEY
    if not api_key:
        raise ValueError("Intrinio key missing; set STOCKSCAN_INTRINIO_KEY")
    our_ciks = {
        r[0] for r in duckdb.query(
            f"select distinct cik from read_parquet('{WIDE_PATH}')"
        ).fetchall()
    }
    with httpx.Client(base_url="https://api-v2.intrinio.com", timeout=60.0,
                      transport=transport) as client:
        companies = enumerate_companies(client, api_key)
        securities = pd.concat(
            [enumerate_securities(client, api_key, active=True),
             enumerate_securities(client, api_key, active=False)],
            ignore_index=True,
        )
    new_uni = select_universe(securities, companies, our_ciks, load_delistings())
    old_uni = load_universe(universe_path)
    diff = universe_diff(old_uni, new_uni)

    # deaths first: fetch the full multi-security splice under the NEW column name
    # (taken from the NEW universe — the dead record's ticker often differs, e.g.
    # bankruptcy Q-suffixes) so the death decline on afterlife securities is kept.
    new_cols = new_uni.sort_values("priority").drop_duplicates("cik").set_index("cik")["column"]
    died_cols = [new_cols[c] for c in diff["died"] if c in new_cols.index]
    grace_cols = [c for c in recent_dead_columns(new_uni, out_dir, today) if c not in died_cols]
    death_fetch = refetch_columns(new_uni, died_cols + grace_cols, api_key, out_dir,
                                  pause=pause, workers=workers, transport=transport) \
        if (died_cols or grace_cols) else {"written": [], "suspect": [], "failed": []}

    rename_result = apply_renames(diff["renames"], out_dir)

    new_files = 0
    if fetch_new and diff["added"]:
        subset = new_uni[new_uni["cik"].isin(set(diff["added"]))]
        new_files = len(download_intrinio_universe(
            subset, start=PRICE_START, api_key=api_key, pause=pause,
            workers=workers, out_dir=out_dir, transport=transport,
        ))

    # commit point
    tmp = Path(universe_path).with_name("." + Path(universe_path).name + ".tmp")
    new_uni.to_parquet(tmp, index=False)
    os.replace(tmp, universe_path)

    if rebuild_cache:
        # renames/refetches changed the column set; drop the cache meta so every
        # loader falls back to the slow path until the nightly job rebuilds it
        _, _, meta_p = matrix_cache_paths()
        meta_p.unlink(missing_ok=True)

    deltas = {
        "companies": int(new_uni["column"].nunique()),
        "added": diff["added"][:50], "n_added": len(diff["added"]),
        "died": diff["died"][:50], "n_died": len(diff["died"]),
        "revived": diff["revived"], "renamed_ciks": len(diff["renames"]),
        "files_moved": len(rename_result["moved"]),
        "rename_kept_target": rename_result["kept_target"],
        "death_refetch": {k: v[:20] if isinstance(v, list) else v
                          for k, v in death_fetch.items()},
        "grace_refetch_columns": len(grace_cols),
        "new_price_files": new_files,
    }

    if state is not None and diff["died"]:
        watched = {w["cik"] for w in state.watchlist()}
        for cik in diff["died"]:
            if cik in watched:
                state.add_alert("universe_death",
                                f"watchlist company cik {cik} is no longer active "
                                f"in the security master", cik=cik)
    return deltas
