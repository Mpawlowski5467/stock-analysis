"""Pure row shapers: a SCORED cross-section in, plain display rows out.

Split from data.py so the facade file stays about loading/caching/delegation and
these stay what they always were — pure functions over tiny DataFrames that
unit-test without a UI framework or a real data store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- pure row shapers (testable with tiny DataFrames) ---------------------------

def _decile(pct: float) -> int:
    return int(np.clip(np.ceil(pct * 10), 1, 10))


def _risk_chips(r, cols) -> list[dict]:
    """FIREWALLED display flags (distress / drawdown) as compact chips — elevated and
    high only, so a browsing table shows the risk reads without a column of 'none'.
    Display-only like everywhere else: never a trade input."""
    chips = []
    for flag_col, prob_col, kind in (("dflag", "dprob", "distress"),
                                     ("wflag", "wprob", "drawdown")):
        if flag_col not in cols:
            continue
        level = r.get(flag_col)
        if level in ("elevated", "high"):
            p = float(r[prob_col]) if pd.notna(r.get(prob_col)) else None
            chips.append({"kind": kind, "level": level,
                          "prob_pct": int(round(p * 100)) if p is not None else None})
    return chips


def industry_of(cross: pd.DataFrame) -> pd.Series:
    """Fine industry label per row, aligned to ``cross.index`` (DISPLAY ONLY).

    The ten SIC divisions the MODEL normalizes within are too coarse to browse:
    'Finance' holds both banks and bitcoin miners, 'Manufacturing' holds Boeing,
    Wolfspeed and three biotechs. ``sic_industry`` splits the same rows into ~45
    recognizable buckets and falls back to the division for unmapped codes, so
    nothing is ever dropped. A cross-section without ``sic`` (older caches, test
    frames) simply keeps the coarse label."""
    from ..sector import sic_industry

    if "sic" not in cross.columns:
        return cross["sector"].astype("object")
    return cross["sic"].map(sic_industry)


def scan_rows(cross: pd.DataFrame, sector: str | None = None) -> list[dict]:
    """Ranked rows for the scan table from a SCORED cross-section (has score/pct).

    ``sector`` matches EITHER the fine industry the table now filters on or the
    coarse division, so a saved 'Manufacturing' filter keeps working."""
    industry = industry_of(cross)
    view = cross
    if sector and sector.lower() != "all":
        view = cross[(industry == sector) | (cross["sector"] == sector)]
    view = view.sort_values("score", ascending=False)
    rows = []
    for i, (idx, r) in enumerate(view.iterrows(), 1):
        rows.append({
            "rank": i,
            "cik": int(r["cik"]),
            "ticker": str(r.get("ticker") or "—"),
            "name": str(r.get("name") or "")[:38],
            # the coarse division stays in the payload: it is the bucket the model
            # ranks within, so the signal percentiles mean nothing without it
            "sector": str(r.get("sector") or "—"),
            "industry": str(industry.get(idx) or r.get("sector") or "—"),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
            "fy": int(r["fy"]) if pd.notna(r.get("fy")) else None,
            "risk": _risk_chips(r, view.columns),
        })
    return rows


def search_rows(cross: pd.DataFrame, query: str, limit: int = 40) -> list[dict]:
    """Scan rows filtered to names whose ticker OR company-name contains ``query``.

    Case-insensitive substring, ranked by score like the scan (best first). Empty
    query returns the full ranked scan (capped)."""
    q = str(query or "").strip()
    if not q:
        return scan_rows(cross)[:limit]
    qu = q.upper()
    tick = cross["ticker"].astype(str).str.upper().str.contains(qu, regex=False, na=False)
    name = cross["name"].astype(str).str.upper().str.contains(qu, regex=False, na=False)
    return scan_rows(cross[tick | name])[:limit]


def sectors_in(cross: pd.DataFrame) -> list[str]:
    """Filter options for the scan table: the fine industries actually present.

    Only labels with rows behind them are offered — an empty bucket in the dropdown
    is a dead end, and which industries exist changes with the liquidity floor."""
    return ["all", *sorted(industry_of(cross).dropna().unique())]


def watch_rows(watchlist: list[dict], cross: pd.DataFrame,
               prev_signals: dict[int, dict], feats: pd.DataFrame,
               as_of=None) -> list[dict]:
    """Join watched CIKs to the current scored cross-section + last-seen signal.

    A watched name absent from the cross-section (dead, stale filer, illiquid)
    still appears — flagged — so the watchlist never silently drops the failures.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else None
    by_cik = cross.set_index(cross["cik"].astype(int))
    rows = []
    for w in watchlist:
        cik = int(w["cik"])
        prev = prev_signals.get(cik) or {}
        last_filing = None
        fsub = feats[feats["cik"] == cik] if "cik" in feats.columns else feats.iloc[0:0]
        if as_of is not None and "available_date" in fsub.columns:
            fsub = fsub[fsub["available_date"] <= as_of]
        if len(fsub):
            last_filing = str(pd.Timestamp(fsub["available_date"].max()).date())
        if cik in by_cik.index:
            r = by_cik.loc[cik]
            pct = int(round(float(r["pct"]) * 100))
            prev_pct = prev.get("percentile")
            # FIREWALLED distress flag (display only): surface elevated/high risk here so a
            # watched name drifting toward failure is visible; it drives no trade action.
            dflag = r.get("dflag") if "dflag" in cross.columns else None
            flag = None
            if dflag in ("elevated", "high"):
                dp = float(r["dprob"]) if pd.notna(r.get("dprob")) else None
                flag = f"⚠ distress {dflag}" + (f" (P≈{dp * 100:.0f}%)" if dp is not None else "")
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or r.get("ticker") or "—"),
                "pct": pct, "decile": _decile(float(r["pct"])),
                "delta": (pct - prev_pct) if prev_pct is not None else None,
                "last_filing": last_filing, "flag": flag,
                "risk": _risk_chips(r, cross.columns),
            })
        else:
            rows.append({
                "cik": cik, "ticker": str(w.get("column") or "—"),
                "pct": None, "decile": None, "delta": None,
                "last_filing": last_filing,
                "flag": "not in liquid universe / lapsed filer",
                "risk": [],
            })
    return rows


def _pick_row(r) -> dict:
    """One markets-page pick row from a scored cross-section row."""
    return {
        "cik": int(r["cik"]),
        "ticker": str(r.get("ticker") or "—"),
        "name": str(r.get("name") or "")[:34],
        "pct": int(round(float(r["pct"]) * 100)),
        "decile": _decile(float(r["pct"])),
    }


def market_rows(cross: pd.DataFrame, top_k: int = 6, min_names: int = 10) -> list[dict]:
    """Per-industry ML top picks for the markets overview (sized later by market cap).

    Groups by the fine ``sic_industry`` label (Semiconductors, Oil & Gas E&P,
    Software, Banks, …) — NOT the coarse model sector — ordered by how many names
    each holds. Industries thinner than ``min_names`` and the catch-all 'Unknown'
    are dropped. Each market's ``picks`` are its highest-scoring names, best first;
    the page annotates each with a live market cap fetched separately.
    """
    from ..sector import sic_industry

    df = cross.copy()
    df["_industry"] = df["sic"].map(sic_industry)
    counts = df["_industry"].value_counts()
    out = []
    for industry in counts.index:
        if not industry or str(industry) == "Unknown" or counts[industry] < min_names:
            continue
        sub = (df[df["_industry"] == industry]
               .sort_values("score", ascending=False).head(top_k))
        picks = [_pick_row(r) for _, r in sub.iterrows()]
        if picks:
            out.append({"market": str(industry), "count": int(counts[industry]),
                        "picks": picks})
    return out


def theme_market_rows(cross: pd.DataFrame, tags: dict, top_k: int = 6,
                      min_names: int = 3) -> list[dict]:
    """Thematic 'markets' (AI/SaaS/EV…) from precomputed {cik: [themes]} tags.

    Same shape as ``market_rows`` but membership comes from the auto-tagged theme
    store rather than SIC. Only names present in the current cross-section count;
    themes with fewer than ``min_names`` are dropped; ordered by tagged count.
    """
    if not tags:
        return []
    present = set(cross["cik"].astype(int))
    by_theme: dict[str, set] = {}
    for cik, themes in tags.items():
        if int(cik) not in present:
            continue
        for t in themes:
            by_theme.setdefault(t, set()).add(int(cik))

    ciks_col = cross["cik"].astype(int)
    out = []
    for theme, ciks in by_theme.items():
        if len(ciks) < min_names:
            continue
        sub = (cross[ciks_col.isin(ciks)]
               .sort_values("score", ascending=False).head(top_k))
        picks = [_pick_row(r) for _, r in sub.iterrows()]
        if picks:
            out.append({"market": str(theme), "count": len(ciks), "picks": picks})
    out.sort(key=lambda m: m["count"], reverse=True)
    return out
