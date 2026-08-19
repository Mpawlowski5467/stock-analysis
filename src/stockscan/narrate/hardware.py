"""Probe this machine, then say which local models Argus should serve on it.

Two questions that are easy to conflate, and must not be:

  *Does it FIT?*  — answered here, from measured numbers only: this machine's memory,
                    and each model's real weight size (the local Ollama library first,
                    the registry manifest second). No size in this module is a guess.
  *Is it GOOD?*   — answered ONLY by scripts/bench_llm.py and scripts/bench_chat.py,
                    run on real Argus packets and scored by the grounding validator.

Fitting is necessary and nowhere near sufficient: the failure mode that matters here
isn't a model that's slow, it's one that quietly writes an unsupported number into a
narration. So the ranking puts a recorded PASS from this project's own benchmarks
ahead of every ungraded model, and any ungraded pick is labelled as the unverified
guess it is. `GRADES` is the transcript of what those benchmarks actually found —
extend it by running them, not by editing it.

Sizing note specific to Argus: the recommended budget is deliberately below the usual
"~75% of unified memory" rule of thumb, because on this box the model is NOT the only
tenant — the always-on web app plus the nightly's wide price matrix want several GB of
the same unified pool at the same time the panel job is calling the model.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..config import LLM_BASE_URL

REGISTRY = "https://registry.ollama.ai/v2/library"

# Share of memory a resident model may occupy. Unified memory is shared with macOS and
# with Argus itself (see the module docstring), so 0.70 rather than the 0.75+ a
# single-tenant inference box could afford. CUDA VRAM is dedicated -> a higher share;
# CPU-only inference pages against the same RAM the OS needs -> a lower one.
UNIFIED_BUDGET_FRACTION = 0.70
VRAM_BUDGET_FRACTION = 0.90
CPU_BUDGET_FRACTION = 0.50

TIERS = ("full", "light", "chat")


@dataclass(frozen=True)
class Hardware:
    """What this machine can actually hold. ``budget_gb`` is the headroom for resident
    model weights — the number every fit decision below is made against."""

    platform: str
    chip: str
    accelerator: str          # "metal-unified" | "cuda" | "cpu"
    ram_gb: float
    cpu_cores: int | None = None
    gpu_cores: int | None = None
    vram_gb: float | None = None
    budget_gb: float = 0.0
    budget_note: str = ""

    def describe(self) -> str:
        bits = [self.chip, self.platform, f"{self.ram_gb:.1f} GB"]
        bits[-1] += " unified" if self.accelerator == "metal-unified" else " RAM"
        if self.vram_gb:
            bits.append(f"{self.vram_gb:.1f} GB VRAM")
        cores = "/".join(str(c) for c in (self.cpu_cores, self.gpu_cores) if c)
        if cores:
            bits.append(f"{cores} CPU/GPU cores" if self.gpu_cores else f"{cores} CPU cores")
        return " · ".join(bits)


@dataclass(frozen=True)
class Candidate:
    """A model Argus could serve. ``roles`` is what it's eligible for at all; ``note``
    must cite where its claim comes from (a bench date, or 'ungraded')."""

    tag: str
    roles: tuple[str, ...]
    note: str
    grades: Mapping[str, str] = field(default_factory=dict)   # tier -> "pass" | "fail"


# The recorded verdicts of this project's own benchmarks. Sources: config.py's serving
# comments, written when the benches were run on this M5 Pro.
#   bench_llm.py  2026-07-02 — narration: first-pass grounding + citation validity
#   bench_chat.py 2026-07-05 — the real /ask path: judge-faithfulness, refusals, latency
CATALOG: tuple[Candidate, ...] = (
    Candidate(
        "gemma4:26b", ("full", "chat"),
        "bench_llm 2026-07-02: 3/3 first-pass valid, ~150s/name · "
        "bench_chat 2026-07-05: 7/8 judge-faithful, 1/9 refusals, ~4s mean warm",
        {"full": "pass", "chat": "pass"},
    ),
    Candidate(
        "phi4", ("light", "chat"),
        "bench_llm 2026-07-02: 3/3, ~74s -> light tier · "
        "bench_chat 2026-07-05: 3/5 faithful, 4/9 refusals, 30s mean — strictly worse "
        "than gemma4:26b for chat once the token cap had solved latency",
        {"light": "pass", "chat": "fail"},
    ),
    Candidate(
        "qwen3.6:27b-mlx", ("full",),
        "bench_llm 2026-07-02: repeated timeouts — the Ollama-MLX path was "
        "non-functional on this machine; the GGUF runtime is the pick",
        {"full": "fail"},
    ),
    # Ungraded candidates. Present so a machine that cannot hold the graded picks still
    # gets a usable answer — never so one can outrank a graded model. The first four
    # were bench_llm.py's original comparison set.
    Candidate("mistral-small3.2", ("full", "chat", "light"), "ungraded here"),
    Candidate("gpt-oss:20b", ("full", "chat"), "ungraded here"),
    Candidate("gemma3:27b", ("full", "chat"), "ungraded here"),
    Candidate("qwen3:32b", ("full", "chat"), "ungraded here"),
    Candidate("gemma3:12b", ("light", "chat"), "ungraded here"),
    Candidate("qwen2.5:14b", ("light", "chat"), "ungraded here"),
    Candidate("granite3.3:8b", ("light",), "ungraded here"),
)


def normalize_tag(tag: str) -> str:
    """Ollama reports 'phi4:latest' where config says 'phi4' — one spelling, or the
    'is the configured model installed?' check answers no to a model that is."""
    tag = tag.strip()
    return tag if ":" in tag else f"{tag}:latest"


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _mac_gpu_cores() -> int | None:
    """Cosmetic — best effort, and system_profiler is slow enough to need a bound."""
    try:
        out = subprocess.run(["system_profiler", "-json", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=15)
        blob = json.loads(out.stdout)
        for gpu in blob.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores") or gpu.get("spdisplays_cores")
            if cores:
                return int(re.sub(r"\D", "", str(cores)) or 0) or None
    except Exception:
        return None
    return None


def _nvidia_vram_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return max(float(x) for x in out.stdout.split()) / 1024.0   # MiB -> GiB
    except Exception:
        return None


def probe_hardware() -> Hardware:
    """Measure this machine. Never raises — an unknown field degrades to None and the
    budget falls back to the conservative CPU share, because recommending nothing is
    more useful than recommending a model that will swap."""
    system = platform.system()
    osver = f"{system} {platform.release()}"
    if system == "Darwin":
        osver = f"macOS {platform.mac_ver()[0] or platform.release()}"

    ram_bytes = _sysctl("hw.memsize")
    ram_gb = float(ram_bytes) / 2**30 if ram_bytes else 0.0
    if not ram_gb:                                    # non-macOS
        try:
            import os as _os
            ram_gb = (_os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")) / 2**30
        except Exception:
            ram_gb = 0.0

    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU"
    cpu_cores = int(_sysctl("hw.ncpu") or 0) or None

    if system == "Darwin" and platform.machine() == "arm64":
        gpu_cores = _mac_gpu_cores()
        budget = ram_gb * UNIFIED_BUDGET_FRACTION
        note = (f"{UNIFIED_BUDGET_FRACTION:.0%} of unified memory — the rest is macOS, "
                f"the always-on web app, and the nightly's price matrix")
        # A deliberately raised Metal wired limit is the operator overriding that
        # default; honour it, but never let 0 (= 'system default') read as 'no room'.
        wired_mb = float(_sysctl("iogpu.wired_limit_mb") or 0)
        if wired_mb > 0:
            budget = min(budget, wired_mb / 1024.0)
            note = f"capped by iogpu.wired_limit_mb={wired_mb:.0f}"
        return Hardware(osver, chip, "metal-unified", ram_gb, cpu_cores, gpu_cores,
                        None, round(budget, 1), note)

    vram = _nvidia_vram_gb()
    if vram:
        return Hardware(osver, chip, "cuda", ram_gb, cpu_cores, None, vram,
                        round(vram * VRAM_BUDGET_FRACTION, 1),
                        f"{VRAM_BUDGET_FRACTION:.0%} of dedicated VRAM")
    return Hardware(osver, chip, "cpu", ram_gb, cpu_cores, None, None,
                    round(ram_gb * CPU_BUDGET_FRACTION, 1),
                    f"{CPU_BUDGET_FRACTION:.0%} of system RAM — CPU inference, expect "
                    f"minutes per narration")


def installed_models(base_url: str | None = None, timeout: float = 5.0) -> dict[str, float]:
    """``{normalized tag: GB on disk}`` from the local Ollama library. Empty dict when
    the server is unreachable — indistinguishable from 'nothing installed' to callers,
    which is why the CLI reports endpoint reachability separately."""
    import httpx

    url = (base_url or LLM_BASE_URL).rstrip("/")
    url = url[:-3].rstrip("/") if url.endswith("/v1") else url    # native /api lives off /v1
    try:
        r = httpx.get(f"{url}/api/tags", timeout=timeout)
        r.raise_for_status()
        return {normalize_tag(m["name"]): float(m.get("size") or 0) / 1e9
                for m in r.json().get("models", [])}
    except Exception:
        return {}


def registry_size_gb(tag: str, timeout: float = 15.0) -> float | None:
    """Real weight size of an uninstalled model, from its registry manifest. ``None``
    when offline or the tag doesn't exist — a tag we cannot size is never recommended."""
    import httpx

    name, _, ver = normalize_tag(tag).partition(":")
    try:
        r = httpx.get(f"{REGISTRY}/{name}/manifests/{ver}", timeout=timeout,
                      headers={"Accept": "application/vnd.docker.distribution."
                                         "manifest.v2+json"})
        r.raise_for_status()
        layers = [ly["size"] for ly in r.json().get("layers", [])
                  if str(ly.get("mediaType", "")).endswith(".model")]
        return round(max(layers) / 1e9, 1) if layers else None
    except Exception:
        return None


def resolve_sizes(candidates=CATALOG, *, installed: Mapping[str, float] | None = None,
                  network: bool = True) -> dict[str, float]:
    """Size every candidate, preferring what is actually on this disk over the registry.
    Tags that resolve nowhere are simply absent — see ``registry_size_gb``."""
    installed = installed if installed is not None else installed_models()
    sizes: dict[str, float] = {}
    for cand in candidates:
        tag = normalize_tag(cand.tag)
        if tag in installed:
            sizes[tag] = round(installed[tag], 1)
        elif network:
            got = registry_size_gb(tag)
            if got is not None:
                sizes[tag] = got
    return sizes


@dataclass(frozen=True)
class Pick:
    tier: str
    tag: str | None
    size_gb: float | None
    installed: bool
    graded: bool
    reason: str


def _rank(cand: Candidate, tier: str, size: float) -> tuple:
    """Graded PASS outranks every ungraded model, whatever its size. Only *within* the
    ungraded pool does bigger win, and that tiebreak is a heuristic the caller labels
    as such — parameter count is a proxy for capability, not a measurement of it."""
    return (cand.grades.get(tier) == "pass", size)


def recommend(hw: Hardware, *, candidates=CATALOG, sizes: Mapping[str, float] | None = None,
              installed: Mapping[str, float] | None = None) -> dict[str, Pick]:
    """Pick a model per tier under one shared memory budget.

    The budget is spent on the *concurrent working set*, not per tier: the nightly
    narrates and answers in the same window, so full and light can both be resident.
    Tiers are chosen in order of how much the quality matters — full narration first,
    then chat, then light — and each subsequent tier only sees what's left, except that
    reusing an already-chosen tag is free (same weights, loaded once)."""
    installed = installed if installed is not None else installed_models()
    sizes = sizes if sizes is not None else resolve_sizes(candidates, installed=installed)

    picks: dict[str, Pick] = {}
    chosen: dict[str, float] = {}        # tag -> size, the working set so far
    for tier in ("full", "chat", "light"):
        spent = sum(chosen.values())
        pool = []
        for cand in candidates:
            tag = normalize_tag(cand.tag)
            if tier not in cand.roles or cand.grades.get(tier) == "fail":
                continue
            size = sizes.get(tag)
            if size is None:
                continue
            # A tag already in the working set costs nothing more to serve.
            if size <= (hw.budget_gb - spent) or tag in chosen:
                pool.append((cand, tag, size))
        if not pool:
            picks[tier] = Pick(tier, None, None, False, False,
                               f"nothing eligible fits the remaining "
                               f"{hw.budget_gb - spent:.1f} GB")
            continue

        cand, tag, size = max(pool, key=lambda t: _rank(t[0], tier, t[2]))
        graded = cand.grades.get(tier) == "pass"
        if graded:
            reason = f"graded PASS on this project — {cand.note}"
        else:
            reason = ("no graded model fits, so this is the largest that does — "
                      "UNVERIFIED here; run scripts/bench_llm.py before trusting it")
        if tag in chosen:
            reason += " · already resident for another tier, so it costs no extra memory"
        picks[tier] = Pick(tier, tag, size, tag in installed, graded, reason)
        chosen[tag] = size
    return picks


def working_set_gb(picks: Mapping[str, Pick]) -> float:
    """Memory the recommendation actually costs — distinct tags only."""
    return round(sum({p.tag: p.size_gb for p in picks.values()
                      if p.tag and p.size_gb}.values()), 1)


def _fmt_size(gb: float | None) -> str:
    return f"{gb:.1f}G" if gb else "?"


def fit_report(hw: Hardware | None = None, *, network: bool = True) -> tuple[str, int]:
    """The whole answer as text plus an exit code: non-zero when a model Argus is
    CONFIGURED to use isn't installed, because that is a silent outage — the endpoint
    still answers /models with a 200 and every completion 404s."""
    from ..config import LLM_CHAT_MODEL, LLM_LIGHT_MODEL, LLM_MODEL

    hw = hw or probe_hardware()
    have = installed_models()
    sizes = resolve_sizes(installed=have, network=network)
    picks = recommend(hw, sizes=sizes, installed=have)
    ws = working_set_gb(picks)

    lines = [f"  hardware   {hw.describe()}",
             f"  budget     {hw.budget_gb:.1f} GB for resident weights ({hw.budget_note})",
             ""]
    lines.append(f"  {'tier':<6} {'model':<22} {'size':>6}  {'status':<12} basis")
    for tier in ("full", "chat", "light"):
        p = picks[tier]
        if not p.tag:
            lines.append(f"  {tier:<6} {'(none)':<22} {'':>6}  {'':<12} {p.reason}")
            continue
        status = "installed" if p.installed else "NOT PULLED"
        basis = "graded" if p.graded else "ungraded"
        lines.append(f"  {tier:<6} {p.tag:<22} {_fmt_size(p.size_gb):>6}  "
                     f"{status:<12} {basis}")
    lines += ["", f"  working set {ws:.1f} GB of {hw.budget_gb:.1f} GB budget — "
                  f"{'fits, all tiers can stay resident' if ws <= hw.budget_gb else 'OVER BUDGET'}",
              ""]
    for tier in ("full", "chat", "light"):
        if picks[tier].tag:
            lines.append(f"  why {tier:<6} {picks[tier].reason}")

    # Configured vs recommended — the part that catches a machine drifting out of spec.
    configured = {"full": LLM_MODEL, "light": LLM_LIGHT_MODEL, "chat": LLM_CHAT_MODEL}
    missing, drift = [], []
    lines.append("")
    for tier in ("full", "chat", "light"):
        tag = normalize_tag(configured[tier])
        rec = picks[tier].tag
        if tag not in have:
            missing.append(tag)
            mark = "MISSING — every call to this tier 404s"
        elif rec and tag != rec:
            drift.append((tier, tag, rec))
            mark = f"installed, but {rec} is recommended"
        else:
            mark = "installed · matches recommendation" if tag == rec else "installed"
        lines.append(f"  configured {tier:<6} {tag:<22} {mark}")

    if missing:
        lines += ["", "  fix:"] + [f"    ollama pull {t.removesuffix(':latest')}"
                                   for t in dict.fromkeys(missing)]
    elif drift:
        lines += ["", "  to adopt the recommendation, set in .env:"]
        env = {"full": "STOCKSCAN_LLM_MODEL", "light": "STOCKSCAN_LLM_LIGHT",
               "chat": "STOCKSCAN_LLM_CHAT_MODEL"}
        lines += [f"    {env[t]}={rec}" for t, _, rec in drift]
    return "\n".join(lines), (1 if missing else 0)


def fit_payload(hw: Hardware | None = None, *, network: bool = True) -> dict:
    """Same answer as JSON, for the health check and any future web surface."""
    from dataclasses import asdict

    hw = hw or probe_hardware()
    have = installed_models()
    picks = recommend(hw, sizes=resolve_sizes(installed=have, network=network),
                      installed=have)
    return {"hardware": asdict(hw),
            "installed": sorted(have),
            "picks": {t: asdict(p) for t, p in picks.items()},
            "working_set_gb": working_set_gb(picks)}


def missing_configured(installed: Mapping[str, float] | None = None) -> list[str]:
    """Configured model tags that are NOT installed. The health check's whole question:
    a reachable endpoint says nothing about whether the model behind it still exists."""
    from ..config import LLM_CHAT_MODEL, LLM_LIGHT_MODEL, LLM_MODEL

    have = installed if installed is not None else installed_models()
    want = (LLM_MODEL, LLM_CHAT_MODEL, LLM_LIGHT_MODEL)
    return sorted({normalize_tag(t) for t in want} - set(have))
