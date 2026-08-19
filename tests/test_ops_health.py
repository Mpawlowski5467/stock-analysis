"""Health check: critical failures exit non-zero, benign ones only warn."""


import json

import pandas as pd

from stockscan.ops.health import Check, _quarter_end, report


def test_quarter_end():
    assert _quarter_end("2026q1") == pd.Timestamp("2026-03-31")
    assert _quarter_end("2026q2") == pd.Timestamp("2026-06-30")
    assert _quarter_end("2026q4") == pd.Timestamp("2026-12-31")


def test_report_exit_code_on_critical():
    ok = [Check("critical", "prices", True, ""), Check("warn", "llm", False, "")]
    text, code = report(ok)
    assert code == 0  # a failing WARN check does not fail the command
    bad = ok + [Check("critical", "artifact", False, "vintage drift")]
    text, code = report(bad)
    assert code == 1
    assert "FAIL" in text and "artifact" in text


def test_report_formats_all_levels():
    checks = [
        Check("critical", "prices", True, "fresh"),
        Check("warn", "matrix_cache", False, "stale"),
        Check("info", "llm", False, "down"),
    ]
    text, code = report(checks)
    assert code == 0
    assert "prices" in text and "matrix_cache" in text and "llm" in text


def test_health_record_alerts_only_on_newly_failing_criticals():
    from stockscan.ops.health import health_record

    checks = [Check("critical", "prices", False, "latest bar 9d ago"),
              Check("critical", "artifact", False, "vintage drift"),
              Check("warn", "llm", False, "down"),
              Check("critical", "ops_state", True, "fine")]
    alerts = []

    def add(kind, msg):
        alerts.append((kind, msg))

    # first screen: both criticals are new -> two alerts; warn never alerts
    rec = health_record(checks, prev_failing=set(), add_alert=add)
    assert rec["critical_failing"] == ["artifact", "prices"]
    assert rec["_status"] == "degraded"
    assert [k for k, _ in alerts] == ["health_critical", "health_critical"]
    assert any("vintage drift" in m for _, m in alerts)

    # same failures next night: already known -> silence
    alerts.clear()
    health_record(checks, prev_failing={"artifact", "prices"}, add_alert=add)
    assert alerts == []

    # recovery then re-failure alerts again
    alerts.clear()
    health_record(checks, prev_failing={"artifact"}, add_alert=add)
    assert [m for _, m in alerts] == ["health: prices critical — latest bar 9d ago"]


def test_health_record_all_ok_is_clean():
    from stockscan.ops.health import health_record

    rec = health_record([Check("critical", "prices", True, "fresh")],
                        prev_failing={"prices"}, add_alert=lambda *a: 1 / 0)
    assert rec["critical_failing"] == [] and "_status" not in rec or rec.get("_status") == "ok"
    assert rec["checks"][0]["ok"] is True


def _write_meta(base, rel, trained_through):
    import json as _json

    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps({"trained_through": trained_through}))


def test_head_co_freeze_warns_when_a_head_lags_the_model(tmp_path):
    from stockscan.ops.health import head_co_freeze

    _write_meta(tmp_path, "model/meta.json", "2026-03-31")
    _write_meta(tmp_path, "drawdown_model/meta.json", "2025-12-31")   # lags
    _write_meta(tmp_path, "confidence_cal/calibration.json", "2026-03-31")  # co-frozen

    c = head_co_freeze(artifacts_dir=tmp_path)
    assert c is not None and c.ok is False and c.level == "warn"
    assert "drawdown" in c.detail and "2025-12-31" in c.detail
    assert "confidence" not in c.detail            # the co-frozen head is not named


def test_head_co_freeze_clean_and_absent_cases(tmp_path):
    from stockscan.ops.health import head_co_freeze

    assert head_co_freeze(artifacts_dir=tmp_path) is None        # nothing frozen
    _write_meta(tmp_path, "model/meta.json", "2026-03-31")
    assert head_co_freeze(artifacts_dir=tmp_path) is None        # no heads to compare
    _write_meta(tmp_path, "distress_model/meta.json", "2026-03-31")
    c = head_co_freeze(artifacts_dir=tmp_path)
    assert c.ok is True and "distress" in c.detail


def test_web_freshness_flags_an_app_serving_stale_data():
    """The real incident: server up for 30 days, answering 200, serving a July as-of
    against an August store. Liveness said OK the whole time."""
    from stockscan.ops.health import web_freshness

    c = web_freshness("2026-07-16", "2026-08-12")
    assert c is not None and c.ok is False and c.level == "warn"
    assert "2026-07-16" in c.detail and "2026-08-12" in c.detail
    assert "27d behind" in c.detail


def test_web_freshness_tolerates_a_weekend_and_reports_in_step():
    from stockscan.ops.health import web_freshness

    assert web_freshness("2026-08-10", "2026-08-12").ok is True   # 2d
    assert web_freshness("2026-08-08", "2026-08-12").ok is True   # 4d, the boundary
    assert web_freshness("2026-08-07", "2026-08-12").ok is False  # 5d


def test_web_freshness_is_silent_without_both_sides():
    """No app, no prices, or an unparseable date is not evidence of staleness."""
    from stockscan.ops.health import web_freshness

    assert web_freshness(None, "2026-08-12") is None       # app down
    assert web_freshness("2026-08-12", None) is None       # no price store
    assert web_freshness("not-a-date", "2026-08-12") is None


def _meta(tmp_path, rel, trained_through, **extra):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"trained_through": trained_through, **extra}))


def test_rebuild_date_is_reconstructed_from_the_label_horizon(tmp_path):
    """`trained_through + horizon` must reproduce the censor date the head recorded —
    that identity is what lets both checks talk about rebuilds without a new field.
    Verified against the real artifacts: distress 2025-03-31 + 12mo = 2026-03-31, and
    drawdown 2025-12-31 + 6mo = 2026-07-02, each matching its stored censor_date."""
    from stockscan.ops.health import _head_freezes

    _meta(tmp_path, "distress_model/meta.json", "2025-03-31", horizon_months=12)
    _meta(tmp_path, "drawdown_model/meta.json", "2025-12-31", horizon_months=6)
    _meta(tmp_path, "model/meta.json", "2026-03-31", label_horizon_days=63)
    f = _head_freezes(tmp_path)
    assert str(f["distress"]["rebuilt"].date()) == "2026-03-31"
    assert str(f["drawdown"]["rebuilt"].date()) == "2026-07-02"
    assert str(f["model"]["rebuilt"].date()) == "2026-06-02"


def test_staleness_does_not_charge_a_head_for_its_own_label_horizon(tmp_path):
    """The regression this replaced: distress read 506d stale on 2026-08-19 purely
    because a 12-month label cannot train closer than ~365d to today. It had in fact
    been rebuilt more recently (141d) than the drawdown head it was ranked behind."""
    from stockscan.ops.health import head_staleness

    _meta(tmp_path, "distress_model/meta.json", "2025-03-31", horizon_months=12)
    c = head_staleness("2026-08-19", artifacts_dir=tmp_path, stale_days=400)
    assert c.ok is True                      # 141d since rebuild, not 506d
    assert "distress 141d" in c.detail

    # and it still catches a head genuinely left alone past the budget
    _meta(tmp_path, "drawdown_model/meta.json", "2023-01-31", horizon_months=6)
    stale = head_staleness("2026-08-19", artifacts_dir=tmp_path, stale_days=400)
    assert stale.ok is False
    assert "drawdown last rebuilt 2023-08-02" in stale.detail   # 2023-01-31 + 183d
    assert "183d label horizon" in stale.detail      # the arithmetic is shown, not hidden


def test_co_freeze_compares_rebuilds_so_a_long_horizon_head_can_pass(tmp_path):
    """Compared on trained_through, a 12-month head could NEVER match a 63-day model —
    the check was unsatisfiable by construction. On rebuild dates it is answerable."""
    from stockscan.ops.health import head_co_freeze

    _meta(tmp_path, "model/meta.json", "2026-03-31", label_horizon_days=63)
    _meta(tmp_path, "distress_model/meta.json", "2025-03-31", horizon_months=12)
    lagging = head_co_freeze(artifacts_dir=tmp_path)
    assert lagging.ok is False               # built 2026-03-31, before the model's 2026-06-02
    assert "distress built 2026-03-31 vs model 2026-06-02" in lagging.detail

    # rebuilt today against the same return model -> passes, despite trained_through
    # still trailing the model by ~10 months, which is arithmetic and not a defect
    _meta(tmp_path, "distress_model/meta.json", "2025-08-18", horizon_months=12)
    c = head_co_freeze(artifacts_dir=tmp_path)
    assert c.ok is True
    assert c.detail.startswith("all risk heads built at/past the return model")
