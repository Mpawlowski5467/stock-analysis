"""Row shapers: the risk-head chips on browsing tables (firewalled display flags)."""

import pandas as pd

from stockscan.view.rows import scan_rows, sectors_in, watch_rows


def _cross(with_heads=True):
    d = {
        "cik": [1, 2, 3],
        "ticker": ["AAA", "BBB", "CCC"],
        "name": ["Aaa Inc", "Bbb Corp", "Ccc Ltd"],
        "sector": ["tech", "tech", "energy"],
        "score": [3.0, 2.0, 1.0],
        "pct": [0.95, 0.50, 0.05],
        "fy": [2025, 2025, 2024],
    }
    if with_heads:
        d["dprob"] = [0.02, 0.31, 0.72]
        d["dflag"] = ["none", "elevated", "high"]
        d["wprob"] = [0.20, 0.61, 0.44]
        d["wflag"] = ["none", "elevated", "none"]
    return pd.DataFrame(d)


def _sic_cross():
    """Three names the coarse divisions lump together and the fine labels separate:
    a bank and a bitcoin miner both land in 'Finance', Boeing in 'Manufacturing'."""
    return pd.DataFrame({
        "cik": [1, 2, 3],
        "ticker": ["BANK", "MINR", "BA"],
        "name": ["A Bank", "A Miner", "Boeing"],
        "sector": ["Finance", "Finance", "Manufacturing"],
        "sic": [6022, 6199, 3721],
        "score": [3.0, 2.0, 1.0],
        "pct": [0.95, 0.50, 0.05],
        "fy": [2025, 2025, 2025],
    })


def test_scan_rows_split_the_coarse_divisions_into_fine_industries():
    by_tk = {r["ticker"]: r for r in scan_rows(_sic_cross())}
    assert by_tk["BANK"]["industry"] == "Banks"
    assert by_tk["MINR"]["industry"] == "Consumer & Specialty Finance"
    assert by_tk["BA"]["industry"] == "Aerospace & Defense"
    # the coarse division survives in the payload — it is the model's peer set
    assert by_tk["BANK"]["sector"] == by_tk["MINR"]["sector"] == "Finance"


def test_scan_filter_accepts_a_fine_industry_and_still_honors_the_old_division():
    assert [r["ticker"] for r in scan_rows(_sic_cross(), "Banks")] == ["BANK"]
    assert [r["ticker"] for r in scan_rows(_sic_cross(), "Aerospace & Defense")] == ["BA"]
    # a saved coarse filter keeps working rather than silently returning nothing
    assert sorted(r["ticker"] for r in scan_rows(_sic_cross(), "Finance")) == ["BANK", "MINR"]
    assert len(scan_rows(_sic_cross(), "all")) == 3


def test_sector_options_offer_fine_industries_that_have_rows():
    opts = sectors_in(_sic_cross())
    assert opts[0] == "all"
    assert set(opts[1:]) == {"Banks", "Consumer & Specialty Finance", "Aerospace & Defense"}


def test_rows_without_sic_keep_the_coarse_label_instead_of_crashing():
    """Older cached cross-sections have no sic column; degrade, never break."""
    rows = scan_rows(_cross())
    assert {r["industry"] for r in rows} == {"tech", "energy"}
    assert sectors_in(_cross()) == ["all", "energy", "tech"]


def test_scan_rows_carry_risk_chips_only_when_flagged():
    rows = scan_rows(_cross())
    by_tk = {r["ticker"]: r for r in rows}
    assert by_tk["AAA"]["risk"] == []                       # clean name: no chips
    assert by_tk["BBB"]["risk"] == [
        {"kind": "distress", "level": "elevated", "prob_pct": 31},
        {"kind": "drawdown", "level": "elevated", "prob_pct": 61},
    ]
    assert by_tk["CCC"]["risk"] == [
        {"kind": "distress", "level": "high", "prob_pct": 72},
    ]


def test_scan_rows_without_head_columns_stay_chipless():
    rows = scan_rows(_cross(with_heads=False))
    assert all(r["risk"] == [] for r in rows)               # heads unfrozen: no crash


def test_watch_rows_carry_risk_chips_and_absent_names_stay_flagged():
    feats = pd.DataFrame({"cik": [3], "available_date": [pd.Timestamp("2025-04-01")]})
    rows = watch_rows(
        [{"cik": 3, "column": "CCC"}, {"cik": 99, "column": "GONE"}],
        _cross(), {}, feats, as_of="2026-07-01",
    )
    assert rows[0]["risk"][0]["kind"] == "distress"
    assert "distress high" in rows[0]["flag"]               # the text flag still rides
    assert rows[1]["risk"] == [] and "not in liquid universe" in rows[1]["flag"]
