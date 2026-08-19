"""Central configuration and the locked project decisions.

Everything a reader needs to know about "what did we decide" lives here or in
DESIGN.md §10. Values can be overridden via environment variables so the code
stays deterministic while remaining tweakable for experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so API tokens stay out of chat and git."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv(REPO_ROOT / ".env")
DATA_DIR = Path(os.environ.get("STOCKSCAN_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # untouched downloads (FSDS zips, Stooq dumps)
PARQUET_DIR = DATA_DIR / "parquet"  # the immutable point-in-time panel + prices
ARTIFACTS_DIR = Path(os.environ.get("STOCKSCAN_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DUCKDB_PATH = DATA_DIR / "stockscan.duckdb"

for _d in (RAW_DIR, PARQUET_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- EDGAR --------------------------------------------------------------------
# SEC requires a descriptive User-Agent that includes a contact address, and
# enforces a hard 10 req/s per IP. We stay under it.
EDGAR_USER_AGENT = os.environ.get(
    "STOCKSCAN_EDGAR_UA", "stock-analysis research mpawlowski5467@gmail.com"
)
EDGAR_MAX_RPS = float(os.environ.get("STOCKSCAN_EDGAR_MAX_RPS", "8"))

# --- price provider -----------------------------------------------------------
# "yfinance" (free, survivorship-biased), "tiingo", or "intrinio" (paid, delisted-inclusive).
PRICE_PROVIDER = os.environ.get("STOCKSCAN_PRICE_PROVIDER", "yfinance")
TIINGO_TOKEN = os.environ.get("STOCKSCAN_TIINGO_TOKEN", "")
INTRINIO_API_KEY = os.environ.get("STOCKSCAN_INTRINIO_KEY", "")

# --- local LLM (NARRATE stage) ------------------------------------------------
# OpenAI-compatible endpoint: Ollama (http://localhost:11434/v1) or llama.cpp server.
# Tiers benchmarked on the M5 Pro 2026-07-02 (scripts/bench_llm.py): gemma4:26b = full
# tier (3/3 first-pass valid, ~150s/name), phi4 = light tier for routine minor-change
# narration (3/3, ~74s). qwen3.6:27b-mlx timed out repeatedly -> the GGUF/llama.cpp
# runtime is the pick; the Ollama-MLX path was non-functional here.
# `ops.py models` reads THIS machine and says whether these defaults still fit and are
# still installed. The verdicts above are transcribed into narrate/hardware.py's
# CATALOG, which is what that command ranks by — re-run the benches to change it.
LLM_BASE_URL = os.environ.get("STOCKSCAN_LLM_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("STOCKSCAN_LLM_MODEL", "gemma4:26b")
LLM_LIGHT_MODEL = os.environ.get("STOCKSCAN_LLM_LIGHT", "phi4")
# Interactive chat (ask / ask-book / explain-move / digest brief) wants SHORT
# answers FAST, so it gets its own knobs: an independent model choice (grounding
# makes a smaller model safe — every numeral is checked either way) and a hard
# completion cap. Measured on the M5 Pro: uncapped gemma4:26b wrote 2,839 tokens
# (~76s) to a one-line chat question; capped at 500 the same turn bounds at ~4s
# warm. Narration stays on LLM_MODEL, uncapped — its long read is the point.
# DEFAULT = LLM_MODEL (gemma4:26b), and it STAYS there: scripts/bench_chat.py on
# the real ask path (2026-07-05) found phi4 both slower (30s mean / 87s worst vs
# gemma 4s / 6.9s) AND less faithful (3/5 judge-faithful, 4/9 refusals vs gemma
# 7/8, 1/9) — the token cap already solved chat latency, so the small model is a
# strict loss here. Set the env var only if a machine can't hold the full model.
LLM_CHAT_MODEL = os.environ.get("STOCKSCAN_LLM_CHAT_MODEL", LLM_MODEL)
LLM_CHAT_MAX_TOKENS = int(os.environ.get("STOCKSCAN_LLM_CHAT_MAX_TOKENS", "500"))
LLM_CHAT_TIMEOUT = float(os.environ.get("STOCKSCAN_LLM_CHAT_TIMEOUT", "120"))
# Thinking models burn hundreds of hidden reasoning tokens per turn — under the
# chat cap that can consume the WHOLE budget and return empty content. Chat turns
# reasoning off (Ollama honors reasoning_effort; other servers ignore it). Set
# STOCKSCAN_LLM_CHAT_REASONING=high to trade latency for deliberation, or "" to
# omit the field entirely.
LLM_CHAT_REASONING = os.environ.get("STOCKSCAN_LLM_CHAT_REASONING", "none") or None

# --- locked modeling decisions (DESIGN.md §10) --------------------------------
LABEL_HORIZON_DAYS = 63           # forward return horizon (~3 months)
AVAILABILITY_LAG_BDAYS = 1        # a filing's numbers are usable at filed + 1 business day
MIN_SECTOR_BUCKET = 20            # min names per (date x sector) before broad fin/non-fin fallback
FEATURE_COVERAGE_FLOOR = 0.70     # drop / bucket-fallback any feature below this per-date coverage

# tradable universe floors
MIN_MARKET_CAP = 100_000_000      # $100M
MIN_DOLLAR_VOLUME = 1_000_000     # $1M 20-day median dollar volume
MIN_PRICE = 1.0                   # $1 price floor (currently on ADJUSTED close — known bias)
MAX_STALE_DAYS = 550              # a 10-K older than this can't represent the company

# delisting-return convention — labeled ESTIMATES; the sweep is a Phase-1 gate
DELISTING_RETURN = {
    "distress": -0.70,
    "going_dark": -1.00,
    "mna": None,                  # carry last traded / deal price
}
DELISTING_HAIRCUT_SWEEP = (-0.30, -0.55, -0.70, -1.00)

# --- go/no-go gate (Phase 1) --------------------------------------------------
GATE_MIN_IC = 0.03                # out-of-sample mean rank IC
GATE_MIN_IC_TSTAT = 2.0           # overlap-corrected t-stat

# --- backtest / signal mechanics (Phase 3, DESIGN.md §6) -----------------------
# Per-SIDE trading cost in bps keyed to 20d-median dollar volume at trade time.
# Calibration anchor (DESIGN.md §9): small-cap round-trips are 50-150+ bps, not 15;
# mega-cap round-trips ~10-20 bps.
COST_TIERS_BPS = (
    (50_000_000, 10.0),
    (10_000_000, 20.0),
    (5_000_000, 35.0),
    (1_000_000, 60.0),
    (0, 100.0),
)
# Annualized borrow fee (bps) for the short book, by ADV; below SHORT_MIN_ADV a name
# is treated as hard-to-borrow and excluded from the short book entirely (the
# borrow-realism mandate: short alpha commonly dies after borrow costs).
BORROW_TIERS_BPS = (
    (10_000_000, 30.0),
    (5_000_000, 100.0),
    (0, 300.0),
)
SHORT_MIN_ADV = 5_000_000
# Hysteresis (DESIGN.md §6): enter a book in the top (bottom) 20% by model score,
# stay until falling out of the top (bottom) 40% — cuts turnover vs a hard decile.
HYSTERESIS_ENTER = 0.20
HYSTERESIS_EXIT = 0.40

# --- news memory (LIVE-VIEW + NARRATION ONLY — never scoring/backtest/panel) -----
# A local, timestamped store of Intrinio headline+summary articles + versioned LLM
# extractions, so narration can "bring up the past". FIREWALLED: news is never a
# feature, never scored, never point-in-time-joined into the panel. Everything is
# timestamped so it COULD be made point-in-time later, but it stays out of the signal.
NEWS_DB_PATH = ARTIFACTS_DIR / "news.sqlite"
NEWS_FETCH_LIMIT = 12                 # articles pulled per company per Intrinio call
NEWS_REFETCH_HOURS = 12               # hard quota cache: skip the network re-fetch for
                                      # a name fetched more recently than this
NEWS_MATERIALITY_FLOOR = 0.35         # curation: drop articles below this materiality
NEWS_CREDIBILITY_FLOOR = 0.45         # curation: drop below this source credibility
                                      # (press-wire sits at 0.4 < floor, so a wire item
                                      # must be decisively material to surface; unknown
                                      # hosts default to 0.5 and pass)

# --- company profile (LIVE-VIEW ONLY — never scoring/backtest/panel) -------------
# Qualitative company metadata from Intrinio /companies: what the company does
# (business description), where it is headquartered, sector/industry. FIREWALLED
# exactly like news memory — display-only, never a feature, never scored, never
# point-in-time-joined into the panel. Fetched lazily BY CIK (the unambiguous,
# recycle-proof identifier) and cached so it doesn't burn quota per screen.
PROFILE_DB_PATH = ARTIFACTS_DIR / "profiles.sqlite"
PROFILE_REFETCH_DAYS = 30             # profiles change rarely; skip the network re-fetch
                                      # for a name cached more recently than this

# Live market cap (Intrinio) for the markets overview — same live-view firewall as
# profiles: display-only sizing, never a feature, never scored. Moves daily, so the
# cache TTL is hours, not days.
MARKETCAP_DB_PATH = ARTIFACTS_DIR / "marketcap.sqlite"
MARKETCAP_REFETCH_HOURS = 20          # skip the network re-fetch within a trading day

# Thematic markets (AI / SaaS / EV …) — auto-tagged by keyword-matching the Intrinio
# business descriptions (same firewalled live-view layer as profiles; display-only,
# never a feature). Themes aren't a standard classification, so membership is derived
# text-matching, precomputed by `ops.py themes` into this store.
THEMES_DB_PATH = ARTIFACTS_DIR / "themes.sqlite"
THEME_MIN_NAMES = 3                   # don't surface a theme with fewer tagged names

# --- continuous operation (Phase 5, DESIGN.md §8) -------------------------------
OPS_STATE_PATH = Path(os.environ.get(                 # mutable ops state (SQLite);
    "STOCKSCAN_OPS_STATE_PATH", ARTIFACTS_DIR / "ops_state.sqlite"))  # own env knob so a
# throwaway instance (e.g. docs screenshots) can point at a demo DB without touching real data.
PAPER_DIR = ARTIFACTS_DIR / "paper_forward"           # append-only paper-forward store
LOGS_DIR = DATA_DIR / "logs"                          # launchd job stdout/stderr
MATRIX_CACHE_DIR = PARQUET_DIR / "matrix_cache"       # wide close/dv matrices (fast load)
# Alert when a watchlist name's model percentile moves at least this much between
# monitor runs (same threshold the narration cache treats as material).
MONITOR_PCTILE_ALERT = 10
# Health: prices are stale after this many calendar days without a new bar
# (covers weekends + market holidays), fundamentals after a quarter end goes
# unanswered this long (FSDS publishes in the weeks after quarter end).
HEALTH_PRICE_STALE_DAYS = 6
HEALTH_FSDS_GRACE_DAYS = 100
# Every frozen head (return model, distress, drawdown, confidence calibration)
# carries a trained_through; past this age the health check warns that the freeze
# is aging — frozen-by-design is not frozen-forever, and the number would keep
# displaying authoritatively while its OOS anchor drifts.
HEALTH_HEAD_STALE_DAYS = 400
# Alert kinds RECORDED and listed in full, but never counted as wanting attention.
# A count is a claim that something needs looking at, and routine bookkeeping drowns
# that claim: of 25 unseen alerts on 2026-08-14, 13 were 10-Q filing notices whose
# numbers simply arrive with the next FSDS batch — and three genuinely unusual ones
# (four held names flagged high-drawdown in a single pass) sat unread behind them for
# a month. Suppressed from the badge, the morning banner and the brief; still in the
# alert list, still in `ops.py alerts`.
ROUTINE_ALERT_KINDS = frozenset({"filing_detected"})
# The always-on web app holds its scored cross-section in MEMORY and only rebuilds it
# on an explicit reload, so a long-lived process keeps answering 200 while serving a
# months-old as-of. Warn once the served as-of falls this far behind the price store;
# the allowance covers a weekend plus the gap between the nightly and the next open.
HEALTH_WEB_LAG_DAYS = 4
# The local web UI, probed by the health check (info-level: not running is fine).
# Host stays loopback by default (write endpoints, no auth — never 0.0.0.0);
# the port knob exists because localhost real estate is shared with other projects.
# WEB_URL (the health probe + links) follows the port unless overridden explicitly.
WEB_HOST = os.environ.get("STOCKSCAN_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("STOCKSCAN_WEB_PORT", "8000"))
WEB_URL = os.environ.get("STOCKSCAN_WEB_URL", f"http://{WEB_HOST}:{WEB_PORT}")

# --- housekeeping: backups + log rotation (nightly) ------------------------------
# The SQLite stores under artifacts/ hold the only irreplaceable personal state
# (positions, watchlist, alerts, job history) plus quota-expensive caches (news,
# profiles). A nightly online .backup into a dated folder makes losing them a
# one-day event instead of a total loss. Point STOCKSCAN_BACKUPS_DIR at a synced
# folder (iCloud/Dropbox) for an off-machine copy.
BACKUPS_DIR = Path(os.environ.get("STOCKSCAN_BACKUPS_DIR", ARTIFACTS_DIR / "backups"))
BACKUP_KEEP_DAYS = 14                 # dated backup folders retained
LOG_ROTATE_MB = 10                    # copy-truncate a log past this size (one .1 kept)

# --- out-of-app notifications (nightly, local-first) ------------------------------
# 'auto' delivers via macOS Notification Center (osascript) and no-ops on other
# platforms; 'off' disables. Local by decision, not limitation: alert text carries
# ticker names and position hints, and the standing privacy rule is that portfolio
# data never leaves the machine — so no ntfy/email/webhook sinks here.
NOTIFY_MODE = os.environ.get("STOCKSCAN_NOTIFY", "auto")
