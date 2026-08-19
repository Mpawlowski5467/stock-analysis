"""Delisting / deregistration ledger from EDGAR's quarterly master index.

Free survivorship infrastructure. Forms 25 / 25-NSE mark exchange delistings;
15-12B / 15-12G mark deregistration (going dark / private / bankrupt). We scan each
quarter's ``master.idx``, keep those forms, and record the earliest such event per CIK.

This does NOT recover delisted PRICE history (the unfixable free-data gap), but it lets
us (a) build a point-in-time universe that knows when a company died and (b) measure
the survivorship gap so Phase-1 IC can be treated as the upper bound it is. See DESIGN.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PARQUET_DIR
from .client import EdgarClient
from .fsds import parse_quarter

LEDGER_PATH = PARQUET_DIR / "delistings.parquet"
DELIST_FORMS = {"25", "25-NSE"}
DEREG_FORMS = {"15-12B", "15-12G"}
_ALL_FORMS = DELIST_FORMS | DEREG_FORMS


def master_idx_url(quarter: str) -> str:
    year, q = parse_quarter(quarter)
    return f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"


def _parse_master(text: str) -> list[tuple]:
    """Pull (cik, name, form, date) rows for delisting/deregistration forms."""
    out = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, name, form, date, _fname = parts
        if form in _ALL_FORMS and cik.strip().isdigit():
            out.append((int(cik), name, form, date))
    return out


def fetch_quarter_events(quarter: str, client: EdgarClient) -> list[tuple]:
    text = client.get_bytes(master_idx_url(quarter)).decode("latin-1", errors="replace")
    return _parse_master(text)


def build_delisting_ledger(quarters, client: EdgarClient | None = None, out_path=LEDGER_PATH) -> int:
    """Scan quarters, keep the earliest delist/dereg event per CIK, write Parquet.

    OVERWRITES ``out_path`` with the result of THIS scan — there is no merge with what
    is already on disk, so ``quarters`` must always be the full history, never just the
    ones that look new. ``stockscan.ops.jobs.refresh_delistings`` is the guarded caller;
    prefer it over calling this directly against the live ledger path."""
    own = client is None
    client = client or EdgarClient()
    events: list[tuple] = []
    try:
        for q in quarters:
            events.extend(fetch_quarter_events(q, client))
    finally:
        if own:
            client.close()
    if not events:
        return 0
    df = pd.DataFrame(events, columns=["cik", "company_name", "form", "delist_date"])
    df["delist_date"] = pd.to_datetime(df["delist_date"])
    df["reason"] = df["form"].map(lambda f: "delist" if f in DELIST_FORMS else "dereg")
    df = df.sort_values("delist_date").drop_duplicates("cik", keep="first")
    df.to_parquet(out_path, index=False)
    return len(df)


def load_delistings(path=LEDGER_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["cik", "company_name", "form", "delist_date", "reason"])
    return pd.read_parquet(path)
