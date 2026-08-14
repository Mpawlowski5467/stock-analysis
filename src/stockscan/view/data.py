"""The view-data facade: turn the serve/ops layer into plain rows for the web UI.

Heavy inputs (the ~11k-column price matrices, the frozen artifact, the ops DB)
load ONCE into an :class:`ArgusData`; the scored cross-section is built once and
reused across the scan and watch views. The row-shaping helpers are pure
functions so they unit-test without a UI framework or a real data store.

Nothing here trains or re-baselines. The only writes are watchlist/alert
curation, delegated to :class:`~stockscan.ops.state.OpsState`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd


from .rows import (
    _decile,
    market_rows,
    scan_rows,
    search_rows,
    sectors_in,
    theme_market_rows,
    watch_rows,
)

# --- the loaded facade ----------------------------------------------------------

@dataclass
class ArgusData:
    """Loaded-once heavy inputs + the reusable scored cross-section.

    Holds NO long-lived SQLite connections: the load runs in a background thread
    but the views query from the main thread, and a sqlite3 connection is bound
    to its creating thread. Each DB-touching method opens its own short-lived
    connection instead (cheap; WAL makes it safe alongside the nightly job).
    """

    data: object          # stockscan.serve.ServeData
    artifact: object      # stockscan.model.Artifact
    as_of: pd.Timestamp = None
    _cross: pd.DataFrame = None

    @classmethod
    def load(cls, as_of=None) -> "ArgusData":
        from ..model import load_artifact
        from ..serve import load_serve_data

        self = cls(data=load_serve_data(), artifact=load_artifact())
        self.refresh(as_of)
        return self

    def refresh(self, as_of=None) -> None:
        """(Re)build and score the cross-section — call after data updates."""
        from ..serve import build_cross_section

        self.as_of = (pd.Timestamp(as_of) if as_of is not None
                      else self.data.close.index[-1])
        cross = build_cross_section(self.data, self.as_of).reset_index(drop=True)
        cross["score"] = self.artifact.score(cross)
        cross["pct"] = cross["score"].rank(pct=True)
        # FIREWALLED distress risk-flag: a second read on the SAME ranks, for display only
        # (never touches score/pct/decile or any trade path). Absent artifact -> no column.
        dart = getattr(self.data, "distress_artifact", None)
        if dart is not None:
            from ..distress import distress_flag

            cross["dprob"] = dart.score(cross)
            cross["dflag"] = [distress_flag(p) for p in cross["dprob"]]
        # FIREWALLED large-drawdown risk-flag: display/book risk only; absent artifact ->
        # no columns. Kept separate from the return score/pct/decile and from distress.
        wart = getattr(self.data, "drawdown_artifact", None)
        if wart is not None:
            from ..drawdown import drawdown_flag

            cross["wprob"] = wart.score(cross)
            cross["wflag"] = [drawdown_flag(p) for p in cross["wprob"]]
        self._cross = cross
        # the per-cik analyze cache mirrors _cross: a rebuilt cross-section must drop
        # it, or /ask and /narrate would keep answering from the pre-update ranks
        self.__dict__.pop("_analyze_cache", None)

    # -- view data ---------------------------------------------------------------
    def sectors(self) -> list[str]:
        return sectors_in(self._cross)

    def scan(self, sector: str | None = None) -> list[dict]:
        return scan_rows(self._cross, sector)

    def search(self, query: str, limit: int = 40) -> list[dict]:
        return search_rows(self._cross, query, limit)

    def resolve(self, query) -> dict | None:
        """Resolve any ticker / TICKER~CIK / CIK to a cik (survivorship-safe). None if unknown."""
        from ..serve import resolve_company

        try:
            cik, column = resolve_company(query, self.data.ticker_map)
        except Exception:
            return None
        return {"cik": int(cik), "column": column}

    def price(self, cik: int) -> dict | None:
        """Close series + summary (last/changes/52wk/ADV) for a name, from the loaded matrices."""
        from .chart import price_summary

        column = self.data.ticker_map.get(int(cik))
        if not column or column not in self.data.close.columns:
            return None
        series = self.data.close[column].dropna()
        adv = None
        if column in self.data.dv_med.columns:
            dvs = self.data.dv_med[column].dropna()
            adv = float(dvs.iloc[-1]) if len(dvs) else None
        return {"column": column, "series": series, "summary": price_summary(series, adv)}

    def events(self, cik: int, limit: int = 8) -> list[dict]:
        """Recent newsworthy EDGAR filings (network; call from a worker).

        Session-cached per (cik, limit) like news()/profile()/quote(): the ticker
        page fetches these on open, and 'explain this move' re-reads them on every
        chip click — one submissions download per name per session, not one per
        click (each fetch is a fresh TLS handshake + a throttled multi-KB pull)."""
        cache = self.__dict__.setdefault("_events_cache", {})
        key = (int(cik), int(limit))
        if key in cache:
            return cache[key]
        from ..news import recent_filings

        rows = recent_filings(int(cik), limit=limit)
        cache[key] = rows
        return rows

    def ohlc(self, cik: int, tail: int = 252):
        """Adjusted OHLCV rows for a name from the per-column price store (local, no quota)."""
        import pandas as pd

        from ..prices import PRICES_DIR

        column = self.data.ticker_map.get(int(cik))
        if not column:
            return None
        p = PRICES_DIR / f"{column}.parquet"
        if not p.exists():
            return None
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = pd.read_parquet(p, columns=cols).dropna(subset=["close"]).sort_values("date")
        return df.tail(tail).reset_index(drop=True)

    def _universe(self):
        uni = self.__dict__.get("_uni")
        if uni is None:
            from ..intrinio_universe import load_universe
            uni = self.__dict__["_uni"] = load_universe()
        return uni

    def _pick(self, cik: int, field: str):
        u = self._universe()
        r = u[u["cik"] == int(cik)]
        if r.empty:
            return None
        return str(r.sort_values("priority").iloc[0][field])

    def news(self, cik: int, limit: int = 6) -> list[dict]:
        """Watchlist headline MEMORY for a name (LIVE-VIEW ONLY — never the signal).

        Lazily ingest into news.sqlite (heuristic extraction — instant; the nightly job
        upgrades watchlist names to the LLM tier) then recall recent + notable-past
        events. Quota-capped by the store's fetch throttle and session-cached per cik."""
        cache = self.__dict__.setdefault("_news_cache", {})
        if cik in cache:
            return cache[cik]
        from ..newsmem import NewsStore
        from ..newsmem.ingest import ingest_company_news

        tk = self._pick(cik, "ticker")
        rows: list[dict] = []
        try:
            with NewsStore() as store:
                ingest_company_news(int(cik), tk, store, llm=None, limit=limit)
                rows = store.context_for(int(cik)) or store.recall(int(cik), limit=limit)
        except Exception:
            rows = []
        cache[cik] = rows
        return rows

    def _news_context(self, cik: int) -> list[dict]:
        """Recalled news memory (recent + notable past) for the narration packet."""
        from ..newsmem import NewsStore

        try:
            with NewsStore() as store:
                return store.context_for(int(cik))
        except Exception:
            return []

    def _news_window(self, cik: int, since: str | None, limit: int = 12) -> list[dict]:
        """News memory dated on/after ``since`` — the coincidence set for 'explain
        this move'. Unlike ``_news_context`` (a curated 6: 3 recent + 3 notable),
        this returns EVERY in-window article (materiality-ordered, curated for
        dedupe/credibility, then capped) so a short horizon can't be starved by the
        notable-PAST picks that the 6-row context spends its budget on."""
        from ..newsmem import NewsStore
        from ..newsmem.curate import curate

        try:
            with NewsStore() as store:
                rows = store.recall(int(cik), since=since, limit=200)
        except Exception:
            return []
        rows = [r for r in rows if r.get("takeaway")]
        try:
            rows = curate(rows)
        except Exception:
            pass
        return rows[:limit]

    def live_quote(self, cik: int, refresh: bool = False) -> dict | None:
        """Latest live/intraday quote (Intrinio). Session-cached unless refresh=True."""
        cache = self.__dict__.setdefault("_quote_cache", {})
        if not refresh and cik in cache:
            return cache[cik]
        from ..quote import realtime_price

        sid = self._pick(cik, "security_id")
        res = realtime_price(sid) if sid else None
        cache[cik] = res
        return res

    def profile(self, cik: int) -> dict | None:
        """Company profile — what it does + HQ (Intrinio, LIVE-VIEW ONLY, never the signal).

        Looked up by CIK (recycle-proof) and served from the persistent profiles.sqlite
        cache; the network fetch only fires on a cache miss or a >30-day-stale row.
        Session-cached on top of that so re-opening a name is instant."""
        cache = self.__dict__.setdefault("_profile_cache", {})
        if cik in cache:
            return cache[cik]
        from ..profile import get_profile

        res = get_profile(int(cik))
        cache[cik] = res
        return res

    def markets(self, top_k: int = 6) -> list[dict]:
        """Per-industry ML top picks for the markets overview (caps fetched separately)."""
        return market_rows(self._cross, top_k)

    def theme_markets(self, top_k: int = 6) -> list[dict]:
        """Thematic markets (AI/SaaS/EV…) from the precomputed tag store — [] if unbuilt.

        Tags are auto-generated by `ops.py themes` (keyword-matching Intrinio company
        descriptions, LIVE-VIEW ONLY). Loaded once per session."""
        tags = self.__dict__.get("_theme_tags")
        if tags is None:
            from ..themes import load_theme_tags
            try:
                tags = load_theme_tags()
            except Exception:
                tags = {}
            self.__dict__["_theme_tags"] = tags
        return theme_market_rows(self._cross, tags, top_k)

    def _theme_tag_map(self) -> dict:
        tags = self.__dict__.get("_theme_tags")
        if tags is None:
            from ..themes import load_theme_tags
            try:
                tags = load_theme_tags()
            except Exception:
                tags = {}
            self.__dict__["_theme_tags"] = tags
        return tags

    def market_constituents(self, kind: str, name: str, cand: int = 60) -> list[dict]:
        """Names in one market (``kind`` = 'ind' | 'theme'), for the treemap.

        Capped to the top ``cand`` by recent dollar volume — a cheap in-memory size
        proxy so the caller live-fetches market cap for only the biggest-liquidity
        candidates (the true megacaps are always among them), not the whole market.
        """
        df = self._cross
        if kind == "theme":
            ciks = {int(c) for c, ts in self._theme_tag_map().items() if name in ts}
            sub = df[df["cik"].astype(int).isin(ciks)]
        else:
            from ..sector import sic_industry
            sub = df[df["sic"].map(sic_industry) == name]
        if sub.empty:
            return []
        dv = {}
        for cik in sub["cik"].astype(int):
            col = self.data.ticker_map.get(int(cik))
            s = (self.data.dv_med[col].dropna()
                 if col and col in self.data.dv_med.columns else None)
            dv[int(cik)] = float(s.iloc[-1]) if s is not None and len(s) else 0.0
        sub = (sub.assign(_dv=sub["cik"].astype(int).map(dv))
               .sort_values("_dv", ascending=False).head(cand))
        return [{
            "cik": int(r["cik"]),
            "ticker": str(r.get("ticker") or "—"),
            "name": str(r.get("name") or ""),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
        } for _, r in sub.iterrows()]

    def fundamentals(self, cik: int) -> dict | None:
        """Recent financials for one name, straight from the in-memory cross row.

        Key ratios (revenue growth, margins, returns, leverage) formatted like the
        ticker view's signals, each with its within-sector percentile rank. No I/O —
        this is what makes the treemap hover instant."""
        from ..narrate.packet import LABELS, _PCT

        sub = self._cross[self._cross["cik"].astype(int) == int(cik)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        metrics = []
        for f in ("revenue_growth", "op_margin", "roa", "roe", "leverage"):
            v = r.get(f)
            if pd.isna(v):
                continue
            rank = r.get(f"{f}_rank")
            metrics.append({
                "label": LABELS[f].split(" (")[0],
                "value": f"{v * 100:+.1f}%" if f in _PCT else f"{v:.2f}x",
                "rank": int(round(float(rank) * 100)) if pd.notna(rank) else None,
            })
        return {
            "name": str(r.get("name") or ""),
            "ticker": str(r.get("ticker") or "—"),
            "fy": int(r["fy"]) if pd.notna(r.get("fy")) else None,
            "filed": (str(pd.Timestamp(r["available_date"]).date())
                      if pd.notna(r.get("available_date")) else None),
            "pct": int(round(float(r["pct"]) * 100)),
            "decile": _decile(float(r["pct"])),
            "metrics": metrics,
        }

    def market_cap(self, cik: int) -> float | None:
        """Current market cap (Intrinio, LIVE-VIEW ONLY — sizing, never the signal).

        Persistent-cached (hours TTL) then session-cached, so the markets page fetches
        each name's cap at most once a day and reopening is instant."""
        cache = self.__dict__.setdefault("_mktcap_cache", {})
        if cik in cache:
            return cache[cik]
        from ..marketcap import get_market_cap

        res = get_market_cap(int(cik))
        cache[cik] = res
        return res

    # -- watchlist (the only writes — sanctioned per the read-mostly design) ---------
    def is_watched(self, cik: int) -> bool:
        from ..ops.state import OpsState

        with OpsState() as st:
            return any(int(w["cik"]) == int(cik) for w in st.watchlist())

    def toggle_watch(self, cik: int) -> bool:
        """Add/remove the name from the watchlist. Returns the new watched state."""
        from ..ops.state import OpsState

        with OpsState() as st:
            watched = any(int(w["cik"]) == int(cik) for w in st.watchlist())
            if watched:
                st.watch_remove(int(cik))
                return False
            st.watch_add(int(cik), self.data.ticker_map.get(int(cik)))
            return True

    # -- positions (PERSONAL holdings — DISPLAY-ONLY; NEVER an input to the signal) -----
    def positions(self) -> list[dict]:
        from ..ops.state import OpsState

        with OpsState() as st:
            return st.positions()

    def set_position(self, cik: int, shares: float, cost: float) -> dict | None:
        """Save the user's holding (personal live-view; firewalled from the score).
        Returns the stored row so the browser can echo the persisted values."""
        from ..ops.state import OpsState

        with OpsState() as st:
            st.position_set(int(cik), float(shares), float(cost))
            return next((p for p in st.positions() if int(p["cik"]) == int(cik)), None)

    def remove_position(self, cik: int) -> None:
        from ..ops.state import OpsState

        with OpsState() as st:
            st.position_remove(int(cik))

    def scorecard(self) -> dict:
        """Book-level scorecard over the names the user tracks (DISPLAY-ONLY; firewalled).

        Covers HELD positions (shares + cost) AND WATCHLIST-only names (followed, no
        shares) — the latter get model standing / distress but no value / P&L. Joins
        them onto the already-scored cross-section; value / P&L use the last close from
        the loaded matrix (no live quota). Never touches the score, the paper book, or
        any trade path."""
        from ..ops.state import OpsState
        from ..portfolio import scorecard as build_scorecard

        with OpsState() as st:
            positions = st.positions()
            watchlist = st.watchlist()
        held = {int(p["cik"]) for p in positions}
        entries = list(positions)
        for w in watchlist:                       # watched-but-not-held → no shares
            if int(w["cik"]) not in held:
                entries.append({"cik": int(w["cik"]), "shares": None,
                                "cost_basis": None, "added_at": w.get("added")})
        prices: dict[int, float] = {}
        for e in entries:
            cik = int(e["cik"])
            col = self.data.ticker_map.get(cik)
            if col and col in self.data.close.columns:
                s = self.data.close[col].dropna()
                if len(s):
                    prices[cik] = float(s.iloc[-1])
        return build_scorecard(entries, self._cross, prices, as_of=self.as_of)

    def ticker(self, query, as_of=None) -> dict:
        """Full per-name analysis (deterministic; template narration included).

        Session-cached per cik at the loaded as-of: analyze() re-scores the whole
        cross-section (~seconds), and the ticker page, /narrate and /ask each need
        the SAME result — one analyze per name per session, not one per call. The
        cache stores and hands out deep copies so no caller (narrate attaches news
        to the packet in place) can leak into another's response; refresh() drops
        it and a reload rebuilds the facade, so it never outlives the ranks it
        mirrors. Failures (unknown cik, stale filer) propagate and are never
        cached. Explicit ``as_of`` bypasses — a historical read is not this view."""
        from ..serve import analyze

        is_cik = isinstance(query, (int, np.integer)) or str(query).isdigit()
        if as_of is not None or not is_cik:
            return analyze(query, as_of=as_of or self.as_of, data=self.data,
                           artifact=self.artifact, llm=None)
        cik = int(query)
        cache = self.__dict__.setdefault("_analyze_cache", {})
        if cik in cache:
            return copy.deepcopy(cache[cik])
        res = analyze(cik, as_of=self.as_of, data=self.data,
                      artifact=self.artifact, llm=None)
        cache[cik] = copy.deepcopy(res)
        return res

    def narrate(self, packet, llm_full=None, llm_light=None, cache=None) -> dict:
        """On-demand narration with the local model, tiered through the SAME
        NarrationCache/narrate_smart machinery the nightly monitor uses.

        Unchanged fundamentals serve the cached narration instantly (tier "cache" —
        no 30-90s LLM wait); a minor wiggle re-narrates at the light tier; a
        material change (new filing / big percentile move / new top drivers) gets
        the full tier. CURRENT news context (LIVE-VIEW) is attached first so any
        FRESH narration weaves it in — news is excluded from the durable packet
        hash by design, so headline churn never busts the cache. A dead LLM
        endpoint degrades to the grounded template inside narrate_packet (never
        crashes) and narrate_smart keeps that out of the cache."""
        from ..narrate.cache import NarrationCache, narrate_smart
        from ..narrate.llm import make_llm
        from ..narrate.packet import news_context

        ctx = news_context(self._news_context(int(packet["meta"]["cik"])))
        if ctx:
            packet.setdefault("context", {})["news"] = ctx
        own = cache is None
        if own:   # short-lived connection per call, like every DB touch on this facade
            cache = NarrationCache()
        try:
            res = narrate_smart(packet, llm_full=llm_full or make_llm("full"),
                                llm_light=llm_light or make_llm("light"),
                                cache=cache)
        finally:
            if own:
                cache.close()
        if ctx and res.get("source") == "llm" and res.get("tier") in ("full", "light"):
            res["tier"] += "+news"
        return res

    @staticmethod
    def _chat_llm():
        """The interactive-chat client (make_llm 'chat' tier): independently swappable
        model, hard token cap, short timeout. Grounding checks every numeral no
        matter the model, so a smaller/faster one loses polish, not honesty."""
        from ..narrate.llm import make_llm

        return make_llm("chat")

    def chat_context(self, cik: int) -> dict:
        """The EXACT grounding context the ticker 'ask' hands the chat model: the
        deterministic narration packet plus current recalled news, WIDENED with the
        firewalled display reads (verdict / confidence / distress / drawdown /
        price / flags) so the chat can explain every number the ticker page shows.

        Factored out of :meth:`ask` so the chat-model benchmark
        (scripts/bench_chat.py) measures the REAL surface by construction rather
        than a hand-mirrored copy that silently drifts when the context is widened."""
        from ..assist.qa import build_chat_context
        from ..narrate.packet import news_context
        from .chart import verdict as call_verdict

        res = dict(self.ticker(int(cik)))
        packet = dict(res["packet"])
        news = news_context(self._news_context(int(packet["meta"]["cik"])))
        if news:
            packet["context"] = {**(packet.get("context") or {}), "news": news}
        res["packet"] = packet
        pr = self.price(int(cik)) or {}
        return build_chat_context(
            res, price_summary=pr.get("summary"),
            verdict=call_verdict((res.get("percentile") or 0) / 100.0))

    def ask(self, cik: int, question: str, history: list | None = None, llm=None) -> dict:
        """Grounded chat about ONE name — the narration made interactive (web 'ask' box).

        Every numeral in the answer must trace to :meth:`chat_context` or the
        assistant refuses (assist.core.grounded_answer) — never a guess, never
        advice. ``history`` rides in from the browser (the server stays stateless)
        and never expands the grounding domain."""
        import time

        from ..assist.core import grounded_answer
        from ..assist.qa import CHAT_SYSTEM
        from ..assist.telemetry import record_turn

        ctx = self.chat_context(int(cik))
        client = llm or self._chat_llm()
        t0 = time.monotonic()
        r = grounded_answer(ctx, question, client, CHAT_SYSTEM, history=history)
        record_turn("ask", ctx, question, r, time.monotonic() - t0,
                    usage=getattr(client, "last_usage", None))
        meta = ctx.get("meta") or {}
        news = (ctx.get("context") or {}).get("news") or []
        return {**r, "cik": int(cik), "ticker": meta.get("ticker"),
                "name": meta.get("name"), "n_news": len(news)}

    def ask_book(self, question: str, history: list | None = None, llm=None) -> dict:
        """Grounded chat about the BOOK — the scorecard made interactive (web ask box).

        Same contract as :meth:`ask`, one level up: the grounding context is exactly
        the scorecard the book tab shows, widened only with display-rounded citable
        twins (assist.book.build_book_context). Every numeral in the answer must
        trace to it or the assistant refuses — never a guess, never a portfolio
        forecast, never advice. ``history`` rides in from the browser (the server
        stays stateless) and never expands the grounding domain."""
        import time

        from ..assist.book import answer_about_book
        from ..assist.telemetry import record_turn

        sc = self.scorecard()
        client = llm or self._chat_llm()
        t0 = time.monotonic()
        r = answer_about_book(sc, question, client, history=history)
        record_turn("ask_book", sc, question, r, time.monotonic() - t0,
                    usage=getattr(client, "last_usage", None))
        return {**r, "n_names": sc.get("n_total", 0), "as_of": sc.get("as_of")}

    def move_context(self, cik: int, horizon: str) -> tuple[dict, dict | None]:
        """Assemble the grounded "explain this move" context and try the code-only
        answer — ALL of it (price read, windowed news, EDGAR filings) with NO LLM
        gate held. Returns ``(bundle, deterministic)``: ``bundle`` carries the ctx
        + display fields the answer step needs; ``deterministic`` is a FINAL result
        when no model is needed (unmeasurable move / empty window), else None.

        The split lets the web route gate only the model call, so a code-only
        answer never queues behind a narration and a slow EDGAR fetch never holds
        the single-flight gate. Firewalled/display-only like ask/narrate; works off
        the price matrices + universe directly, so an unscored name with price
        history still gets an honest read."""
        from ..assist.move import (HORIZONS, build_move_context,
                                    deterministic_answer, window_cutoff)
        from ..narrate.packet import news_context

        if horizon not in HORIZONS:
            raise ValueError(f"unknown horizon {horizon!r}")
        pr = self.price(int(cik)) or {}
        summary = pr.get("summary") or {}
        series = pr.get("series")
        as_of = str(series.index[-1].date()) if series is not None and len(series) else None
        row = self._cross[self._cross["cik"] == int(cik)]
        meta = ({"ticker": str(row.iloc[0].get("ticker") or "—"),
                 "name": str(row.iloc[0].get("name") or "")}
                if not row.empty else
                {"ticker": self._pick(cik, "ticker") or "—",
                 "name": self._pick(cik, "name") or ""})
        ctx = build_move_context(
            meta, horizon, summary, as_of,
            news=news_context(self._news_window(int(cik), window_cutoff(horizon, as_of))),
            filings=self.events(int(cik)))
        bundle = {"ctx": ctx, "ticker": meta["ticker"]}
        det = deterministic_answer(ctx)
        if det is not None:
            det = {**det, "cik": int(cik), "horizon": horizon, "ticker": meta["ticker"]}
        return bundle, det

    def move_answer(self, cik: int, horizon: str, bundle: dict, llm=None) -> dict:
        """The LLM half of "explain this move" — call ONLY under the single-flight
        gate, and only when :meth:`move_context` returned no deterministic answer."""
        import time

        from ..assist.move import answer_from_context
        from ..assist.telemetry import record_turn

        client = llm or self._chat_llm()
        t0 = time.monotonic()
        r = answer_from_context(bundle["ctx"], client)
        record_turn("move", bundle["ctx"], f"explain move {horizon}", r,
                    time.monotonic() - t0, usage=getattr(client, "last_usage", None))
        return {**r, "cik": int(cik), "horizon": horizon, "ticker": bundle["ticker"]}

    def explain_move(self, cik: int, horizon: str, llm=None) -> dict:
        """The whole "explain this move" turn (context + answer) in one call —
        convenience for direct callers/tests. The WEB route splits it across the
        gate via :meth:`move_context` / :meth:`move_answer` so context work and
        code-only answers never hold the LLM gate."""
        bundle, det = self.move_context(cik, horizon)
        return det if det is not None else self.move_answer(cik, horizon, bundle, llm=llm)

    def digest(self) -> dict:
        """Overnight ops record (jobs, unseen alerts, paper status) — deterministic.

        The instant, number-true payload for the web digest card; the LLM prose
        over it is :meth:`digest_brief` (separate, slow, optional)."""
        from ..assist.brief import build_brief_context
        from ..ops.state import OpsState

        try:
            paper = self.paper()
        except Exception:
            paper = None
        if paper is not None:   # the card needs the verdict, not 12 months of detail
            paper = {k: paper.get(k) for k in
                     ("months_scored_oos", "live_mean_ic", "degraded", "note")
                     if k in paper}
        with OpsState() as st:
            ctx = build_brief_context(st, paper=paper)
            stored = st.kv_get("digest_brief")
        # the nightly pre-generates the brief (ops.py job_digest) — ship it with the
        # card when fresh so the page opens with prose instead of a button; stale
        # briefs (>24h — a skipped night) are withheld rather than shown as current
        if stored and stored.get("answer") and stored.get("_updated"):
            age_h = (pd.Timestamp.now("UTC")
                     - pd.Timestamp(stored["_updated"])).total_seconds() / 3600
            if age_h <= 24:
                ctx["stored_brief"] = {"answer": stored["answer"],
                                       "updated": stored["_updated"]}
        return ctx

    def panel_cached(self, cik: int) -> dict:
        """The analyst panel as currently cached for this name's LIVE context —
        no LLM, no gate. The frontend paints whatever exists and offers to run
        the missing roles; a context change (new filing, rank move) empties this
        by construction (the hash moved)."""
        from ..assist.analyst import ROLES, PanelCache
        from ..assist.telemetry import context_hash

        ctx = self.chat_context(int(cik))
        h = context_hash(ctx)
        with PanelCache() as pc:
            roles = pc.get(int(cik), h)
        return {"cik": int(cik), "context_hash": h,
                "roles": {r: roles.get(r) for r in ROLES},
                "complete": all(r in roles for r in ROLES)}

    def panel_role(self, cik: int, role: str, llm=None) -> dict:
        """Generate ONE panel memo under the caller's gate: grounded, judge-checked
        (suppressed on drift), cached by (cik, context-hash, role), telemetry-logged.
        Prior memos come from the cache so the frontend can chain role requests."""
        import time

        from ..assist.analyst import PanelCache, panel_role
        from ..assist.telemetry import context_hash, record_turn

        ctx = self.chat_context(int(cik))
        h = context_hash(ctx)
        client = llm or self._chat_llm()
        with PanelCache() as pc:
            cached = pc.get(int(cik), h)
            if role in cached:
                return {**cached[role], "cik": int(cik), "cached": True}
            prior = {r: p["answer"] for r, p in cached.items()
                     if not p.get("refused") and not p.get("suppressed")}
            t0 = time.monotonic()
            # the judge gets its OWN client so last_usage on `client` stays the
            # memo generation's token count (telemetry attribution)
            r = panel_role(role, ctx, client, prior=prior, judge_llm=self._chat_llm())
            record_turn(f"panel:{role}", ctx, f"panel {role} memo", r,
                        time.monotonic() - t0,
                        usage=getattr(client, "last_usage", None))
            # only CLEAN memos are cached: a refusal (fabrication retry exhausted) or
            # a judge suppression is retryable model noise, not a fact about the data
            # — caching it would pin the failure until the next filing
            if not r.get("refused") and not r.get("suppressed"):
                pc.put(int(cik), h, role, r)
        return {**r, "cik": int(cik), "cached": False}

    def digest_brief(self, llm=None) -> dict:
        """The grounded LLM morning brief over :meth:`digest` (refuses over inventing)."""
        import time

        from ..assist.brief import nightly_brief
        from ..assist.telemetry import record_turn

        ctx = self.digest()
        client = llm or self._chat_llm()
        t0 = time.monotonic()
        r = nightly_brief(ctx, client)
        record_turn("brief", ctx, "overnight operations brief", r,
                    time.monotonic() - t0, usage=getattr(client, "last_usage", None))
        return r

    def mark_alerts_seen(self, ids: list[int] | None = None) -> dict:
        """Acknowledge alerts from the UI (all unseen, or specific ids). Returns the
        new unseen count so the statusbar badge updates without a second round-trip."""
        from ..ops.state import OpsState

        with OpsState() as st:
            marked = st.mark_alerts_seen(ids)
            return {"marked": marked, "unseen_alerts": st.unseen_count()}

    def regime(self) -> dict | None:
        """Market context (breadth / median vol / EW drawdown) over the LIQUID
        cross-section — display-only facts, cached per as_of (the matrix scan is a
        few hundred ms; the numbers only move when the data does)."""
        from .regime import compute_regime, regime_line

        key = str(self.as_of)
        cached = getattr(self, "_regime_cache", None)
        if cached and cached.get("key") == key:
            return cached["value"]
        tickers = None
        if self._cross is not None and "ticker" in self._cross.columns:
            tickers = list(self._cross["ticker"].dropna().unique())
        r = compute_regime(self.data.close, tickers=tickers, as_of=self.as_of)
        if r is not None:
            r = {**r, "line": regime_line(r)}
        self._regime_cache = {"key": key, "value": r}
        return r

    def health(self) -> dict:
        """The last stored health screen + a recent job strip — read-only over the ops
        record. The checks are the NIGHTLY's (run_checks probes the web UI and LLM;
        re-running them inside a web request would have the server probe itself), so
        ``as_of`` matters: an old record means the nightly hasn't run, which is itself
        a health signal the UI should show."""
        from ..ops.state import OpsState

        with OpsState() as st:
            last = st.last_run("health")
            runs = st.recent_runs(limit=24)
        deltas = (last or {}).get("deltas") or {}
        return {
            "as_of": (last or {}).get("finished"),
            "status": (last or {}).get("status"),
            "checks": deltas.get("checks") or [],
            "critical_failing": deltas.get("critical_failing") or [],
            "runs": runs,
        }

    def watched_ciks(self) -> list[int]:
        """Just the watched CIKs — the scan page's star column wants a cheap set,
        not the full watch_rows join."""
        from ..ops.state import OpsState

        with OpsState() as st:
            return [int(w["cik"]) for w in st.watchlist()]

    def watch(self) -> dict:
        from ..ops.state import OpsState

        with OpsState() as st:
            wl = st.watchlist()
            prev = {w["cik"]: (st.get_signal(w["cik"]) or {}) for w in wl}
            alerts = st.alerts(unseen_only=False, limit=40)
        rows = watch_rows(wl, self._cross, prev, self.data.feats, self.as_of)
        return {"rows": rows, "alerts": alerts}

    def status(self) -> dict:
        from ..ops.state import OpsState

        with OpsState() as st:
            return status_dict(st, self.artifact, self.as_of)

    def paper(self) -> dict | None:
        from ..config import PAPER_DIR
        from ..ops.paper import compare

        if not (PAPER_DIR / "baseline.json").exists():
            return None
        return compare(close=self.data.close, ticker_map=self.data.ticker_map)

    def close(self) -> None:
        pass  # no long-lived connections to release


def status_dict(state, artifact, as_of) -> dict:
    """Cheap status line — no LLM ping (that stays off the hot path)."""
    from ..ops.jobs import quarters_present
    from ..ops.paper import artifact_fingerprint, current_vintage

    quarters = quarters_present()
    try:
        fp = artifact_fingerprint()
    except FileNotFoundError:
        fp = None
    vintage = current_vintage()
    last = state.last_run("nightly")
    return {
        "as_of": str(pd.Timestamp(as_of).date()) if as_of is not None else None,
        "fund_quarter": quarters[-1] if quarters else None,
        "vintage": (vintage or {}).get("hash"),
        "artifact_registered": bool(vintage and fp and vintage["hash"] == fp),
        "nightly_status": (last or {}).get("status"),
        # the badge counts what wants attention, not the full unseen record —
        # routine filing notices stay in the alert list but out of this number
        "unseen_alerts": state.unseen_count(),
    }
