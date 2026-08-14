"""Thematic markets (AI / SaaS / EV …) auto-tagged from Intrinio business descriptions.

Themes are NOT a standard classification — no data field says "this is an AI company"
— so membership is derived by keyword-matching the company's business description.
This is the same firewalled LIVE-VIEW layer as ``profile`` / ``marketcap``:
display-only grouping for the markets page, never a feature, never scored.

The rules are deliberately CONSERVATIVE (specific phrases, word boundaries) — recall
is traded for precision, because a false "AI" tag is worse than a miss on a page a
human reads. ``THEME_RULES`` is a plain dict: edit it (and bump ``RULES_VERSION``) to
tune membership, then rerun ``ops.py themes``. Tags are precomputed into a small store
because tagging only names you've opened would never populate a whole theme.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import THEMES_DB_PATH

RULES_VERSION = "v2"

# theme -> keyword patterns (matched case-insensitively against the description).
# Phrases are chosen to be specific; bare ambiguous tokens ("ai", "ev") are avoided
# in favor of word-boundaried or multi-word forms to keep false positives down.
THEME_RULES: dict[str, list[str]] = {
    "AI": [r"artificial intelligence", r"machine learning", r"generative ai",
           r"deep learning", r"large language model", r"neural network"],
    "SaaS": [r"software[- ]as[- ]a[- ]service", r"\bsaas\b", r"subscription[- ]based software"],
    "Cloud": [r"cloud computing", r"cloud[- ]based", r"cloud platform", r"cloud infrastructure"],
    "Cybersecurity": [r"cyber[- ]?security", r"endpoint security", r"threat detection",
                      r"network security", r"information security"],
    "Fintech": [r"fintech", r"financial technology", r"digital payment", r"payment processing",
                r"payments platform"],
    "Electric Vehicles": [r"electric vehicle", r"ev charging", r"battery electric",
                          r"electric truck", r"charging station"],
    "Clean Energy": [r"\bsolar\b", r"photovoltaic", r"renewable energy", r"clean energy",
                     r"wind power", r"wind energy"],
    "Crypto & Blockchain": [r"blockchain", r"cryptocurrenc", r"\bbitcoin\b", r"digital asset",
                            r"\bcrypto\b"],
    "Space": [r"\bsatellite", r"spacecraft", r"launch vehicle", r"space exploration"],
    "Gaming": [r"video game", r"\bgaming\b", r"\besports\b", r"interactive entertainment"],
    "Cannabis": [r"\bcannabis\b", r"marijuana", r"cannabinoid"],
    # --- v2: themes that CUT ACROSS the SIC industries ---------------------------
    # Deliberately not added: Banks, Insurance, REITs, Biotech, Defense and the like.
    # Those are already a filter on the scan table (sector.sic_industry), and a theme
    # that merely restates the industry code adds a second name for the same set.
    # What earns a rule here is a group SIC cannot express — bitcoin miners filed
    # under 'Finance', quantum computers under 'Services', nuclear split across
    # Mining/Manufacturing/Utilities.
    "Bitcoin Mining": [r"bitcoin mining", r"mining of bitcoin", r"cryptocurrency mining",
                       r"digital asset mining", r"hash ?rate", r"mining rigs?"],
    "Quantum Computing": [r"quantum comput", r"quantum annealing", r"\bqubits?\b",
                          r"quantum information"],
    "Robotics & Automation": [r"\brobotics?\b", r"robotic process automation",
                              r"industrial automation", r"warehouse automation",
                              r"autonomous mobile robot"],
    "Data Centers": [r"data cent(?:er|re)s?", r"colocation", r"hyperscale"],
    "Uranium & Nuclear": [r"\buranium\b", r"nuclear (?:power|energy|reactor|fuel)",
                          r"small modular reactor"],
    "Hydrogen & Fuel Cells": [r"green hydrogen", r"fuel cells?", r"electroly[sz]er",
                              r"hydrogen (?:fuel|energy|power|production|economy)"],
    # bare "energy storage" also matches Taylor Devices' seismic dampers ("energy
    # storage devices"), so the noun that follows has to carry the energy sense
    "Energy Storage & Grid": [r"energy storage (?:system|solution|project|facilit|segment)",
                              r"battery (?:energy )?storage", r"grid[- ]scale",
                              r"smart grid", r"microgrid"],
    "Genomics & Gene Therapy": [r"\bgenomics?\b", r"gene therapy", r"gene editing",
                                r"\bcrispr\b", r"\bmrna\b", r"cell therapy"],
    "Digital Health": [r"telehealth", r"telemedicine", r"digital health",
                       r"remote patient monitoring"],
    "Drones & Autonomy": [r"\bdrones?\b", r"unmanned aerial", r"\buavs?\b",
                          r"autonomous driving", r"self[- ]driving", r"autonomous vehicles?"],
    "Rare Earths & Critical Minerals": [r"rare earth", r"critical minerals?"],
    "Semiconductor Equipment": [r"semiconductor (?:capital )?equipment", r"wafer fabrication",
                                r"photolithograph", r"semiconductor manufacturing equipment"],
    "3D Printing": [r"3d printing", r"additive manufacturing"],
}

_COMPILED = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in THEME_RULES.items()}


def tag_themes(text) -> list[str]:
    """Themes whose keyword rules match ``text`` (pure; description order preserved)."""
    if not text:
        return []
    s = str(text)
    return [theme for theme, pats in _COMPILED.items() if any(p.search(s) for p in pats)]


# -- store (small WAL SQLite; one row per tagged cik) ------------------------------

_SCHEMA = """
create table if not exists theme_tags (
    cik integer primary key,
    themes text not null,       -- JSON list of theme names
    rules_version text not null,
    built_at text not null
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ThemeStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or THEMES_DB_PATH)   # read at call time so tests can redirect
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "ThemeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def put(self, cik: int, themes: list[str]) -> None:
        self._db.execute(
            "insert or replace into theme_tags (cik, themes, rules_version, built_at) "
            "values (?, ?, ?, ?)",
            (int(cik), json.dumps(themes), RULES_VERSION, _utcnow()))

    def clear(self, cik: int) -> None:
        self._db.execute("delete from theme_tags where cik = ?", (int(cik),))

    def commit(self) -> None:
        self._db.commit()

    def load(self) -> dict[int, list[str]]:
        """{cik: [themes]} for every tagged name (empty dict if the store is unbuilt)."""
        out: dict[int, list[str]] = {}
        for cik, tj in self._db.execute("select cik, themes from theme_tags").fetchall():
            try:
                out[int(cik)] = json.loads(tj)
            except (ValueError, TypeError):
                continue
        return out


def description_for(cik: int) -> str | None:
    """The cached Intrinio business text for a CIK (short + long), for tagging."""
    from .profile import get_profile

    p = get_profile(int(cik)) or {}
    parts = [p.get("description"), p.get("description_long")]
    text = " ".join(x for x in parts if x)
    return text or None


def refresh_theme_tags(ciks, get_desc=description_for, store: ThemeStore | None = None,
                       max_workers: int = 6) -> dict:
    """Fetch descriptions for ``ciks``, tag them, and (re)write the store. Idempotent.

    ``get_desc`` does the (cached) network fetch, so it is parallelized; the tagging
    itself is pure. A name that matches nothing is cleared, so a company that drops a
    keyword between builds loses its stale tag. Returns per-theme counts."""
    own = store is None
    store = store or ThemeStore()
    by_theme: Counter = Counter()
    tagged = 0
    try:
        def one(cik):
            try:
                return cik, get_desc(cik)
            except Exception:
                return cik, None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(one, list(ciks)))
        for cik, desc in results:
            themes = tag_themes(desc)
            if themes:
                store.put(cik, themes)
                tagged += 1
                for t in themes:
                    by_theme[t] += 1
            else:
                store.clear(cik)
        store.commit()
    finally:
        if own:
            store.close()
    return {"scanned": len(results), "tagged": tagged, "by_theme": dict(by_theme)}


def load_theme_tags() -> dict[int, list[str]]:
    """{cik: [themes]} from the store — empty if `ops.py themes` hasn't run."""
    with ThemeStore() as store:
        return store.load()
