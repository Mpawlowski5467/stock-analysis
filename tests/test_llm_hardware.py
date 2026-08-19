"""Hardware probe + model recommendation: fit is measured, quality is only ever cited.

The rule these tests defend is the one the module exists for — a model that merely
fits must never outrank one this project's own benchmarks graded PASS.
"""

import pytest

from stockscan.narrate.hardware import (
    CATALOG,
    Candidate,
    Hardware,
    normalize_tag,
    probe_hardware,
    recommend,
    working_set_gb,
)
from stockscan.ops.health import llm_models_present


def _hw(budget: float) -> Hardware:
    return Hardware("test-os", "test-chip", "metal-unified", 48.0, budget_gb=budget)


# a tiny closed world so the assertions are about the RANKING, not about real models
GRADED = Candidate("graded:20b", ("full", "chat"), "bench 2026-07-02", {"full": "pass"})
BIGGER = Candidate("ungraded:32b", ("full", "chat"), "ungraded here")
SMALL = Candidate("small:8b", ("light",), "ungraded here")
FAILED = Candidate("broken:27b", ("full",), "timed out", {"full": "fail"})
SIZES = {"graded:20b": 14.0, "ungraded:32b": 20.0, "small:8b": 5.0, "broken:27b": 17.0}


def test_graded_pass_outranks_a_bigger_ungraded_model():
    """The whole point: capability-by-parameter-count is a guess, a bench result isn't."""
    picks = recommend(_hw(40.0), candidates=(BIGGER, GRADED), sizes=SIZES, installed={})
    assert picks["full"].tag == "graded:20b"
    assert picks["full"].graded is True
    assert "graded PASS" in picks["full"].reason


def test_largest_ungraded_wins_only_when_nothing_graded_fits():
    picks = recommend(_hw(22.0), candidates=(BIGGER, SMALL), sizes=SIZES, installed={})
    assert picks["full"].tag == "ungraded:32b"
    assert picks["full"].graded is False
    assert "UNVERIFIED" in picks["full"].reason   # never presented as a settled choice


def test_a_failed_grade_is_never_recommended():
    """qwen3.6:27b-mlx timed out repeatedly — it fits fine, and must still not be picked."""
    picks = recommend(_hw(40.0), candidates=(FAILED,), sizes=SIZES, installed={})
    assert picks["full"].tag is None
    assert "nothing eligible" in picks["full"].reason


def test_budget_is_the_concurrent_working_set_not_per_tier():
    """full + light are resident at the same time in the nightly; a 16 GB budget cannot
    hold a 14 GB narrator AND a 5 GB light model."""
    picks = recommend(_hw(16.0), candidates=(GRADED, SMALL), sizes=SIZES, installed={})
    assert picks["full"].tag == "graded:20b"
    assert picks["light"].tag is None          # 5.0 does not fit in the 2.0 left over
    assert working_set_gb(picks) == 14.0


def test_reusing_one_tag_across_tiers_costs_no_extra_memory():
    picks = recommend(_hw(15.0), candidates=(GRADED,), sizes=SIZES, installed={})
    assert picks["full"].tag == picks["chat"].tag == "graded:20b"
    assert working_set_gb(picks) == 14.0       # counted once, not twice
    assert "no extra memory" in picks["chat"].reason


def test_a_model_that_cannot_be_sized_is_never_recommended():
    """Offline, or a tag that doesn't exist — silence beats a fabricated fit."""
    picks = recommend(_hw(40.0), candidates=(GRADED,), sizes={}, installed={})
    assert picks["full"].tag is None


@pytest.mark.parametrize("raw,want", [("phi4", "phi4:latest"), ("gemma4:26b", "gemma4:26b"),
                                      ("  phi4  ", "phi4:latest")])
def test_tag_normalization(raw, want):
    """Ollama says 'phi4:latest' where config says 'phi4'; one spelling, or the
    installed-check answers 'no' for a model that is in fact installed."""
    assert normalize_tag(raw) == want


def test_catalog_notes_cite_their_basis():
    for cand in CATALOG:
        graded = any(v == "pass" for v in cand.grades.values())
        assert ("bench_" in cand.note) if graded else True, cand.tag
        assert cand.note, cand.tag


def test_probe_never_raises_and_leaves_headroom():
    hw = probe_hardware()
    assert hw.ram_gb >= 0 and hw.budget_gb >= 0
    if hw.ram_gb:
        assert hw.budget_gb < hw.ram_gb        # never hand the whole machine to a model


def test_health_check_flags_a_configured_but_missing_model():
    """The 2026-08-18 outage: reachable endpoint, model gone, every completion 404s."""
    ok = llm_models_present(missing=[])
    assert ok.ok and ok.name == "llm_models"
    bad = llm_models_present(missing=["gemma4:26b"])
    assert not bad.ok
    assert "gemma4:26b" in bad.detail and "404" in bad.detail
