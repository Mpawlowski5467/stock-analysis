"""Out-of-app notification: local-first, deterministic, high-severity only.

The delivery path must never raise (it runs after the nightly's real work), never
invoke an LLM, and never surface the noisy alert kinds — a banner the user learns
to ignore is worse than none.
"""

from stockscan.ops import notify
from stockscan.ops.notify import HIGH_SEVERITY, deliver_nightly, nightly_summary


def _alert(kind, message="something happened"):
    return {"kind": kind, "message": message}


def test_summary_leads_with_high_severity_lines():
    alerts = [_alert("percentile_move", "AAPL moved"),
              _alert("distress_risk", "XYZ distress risk escalated"),
              _alert("paper_degraded", "live IC below the floor")]
    title, msg = nightly_summary("ok", alerts)
    assert "2 alert(s)" in title
    assert "XYZ distress risk escalated" in msg and "live IC below the floor" in msg
    assert "AAPL moved" not in msg          # routine kinds never make the banner


def test_summary_caps_lines_and_counts_the_rest():
    alerts = [_alert("distress_risk", f"name {i}") for i in range(7)]
    _, msg = nightly_summary("ok", alerts)
    assert msg.count("name ") == 4 and "and 3 more in the app" in msg


def test_summary_quiet_night_is_honest():
    title, msg = nightly_summary("ok", [])
    assert title == "Argus nightly: ok" and msg == "no new alerts"
    _, msg2 = nightly_summary("degraded", [_alert("percentile_move")])
    assert "1 routine alert(s) waiting" in msg2


def test_routine_filing_notices_never_become_a_waiting_count():
    """A banner claiming 13 alerts are waiting, when all 13 are filing notices whose
    numbers arrive with the next batch, trains the reader to ignore the banner."""
    filings = [_alert("filing_detected", f"cik {i} filed a 10-Q") for i in range(13)]
    _, msg = nightly_summary("ok", filings)
    assert msg == "no new alerts"

    # mixed: only the one that wants attention is counted
    _, msg2 = nightly_summary("ok", filings + [_alert("percentile_move")])
    assert "1 routine alert(s) waiting" in msg2


def test_deliver_off_mode_never_calls_out(monkeypatch):
    called = []
    monkeypatch.setattr(notify, "notify_mac", lambda *a: called.append(a) or True)
    out = deliver_nightly("ok", [_alert("distress_risk")], mode="off")
    assert out == {"mode": "off", "alerts": 1, "high": 1, "delivered": False}
    assert called == []


def test_deliver_reports_unavailable_without_osascript(monkeypatch):
    monkeypatch.setattr(notify, "_osascript_available", lambda: False)
    out = deliver_nightly("ok", [], mode="auto")
    assert out["mode"] == "unavailable" and out["delivered"] is False


def test_deliver_pushes_and_degrades_on_failure(monkeypatch):
    monkeypatch.setattr(notify, "_osascript_available", lambda: True)
    sent = []
    monkeypatch.setattr(notify, "notify_mac", lambda t, m: sent.append((t, m)) or True)
    out = deliver_nightly("degraded", [_alert("universe_death", "ABC delisted")], mode="auto")
    assert out["delivered"] is True and "_status" not in out
    assert sent and "ABC delisted" in sent[0][1]

    monkeypatch.setattr(notify, "notify_mac", lambda t, m: False)
    out2 = deliver_nightly("ok", [], mode="auto")
    assert out2["delivered"] is False and out2["_status"] == "degraded"


def test_high_severity_set_stays_curated():
    # the banner contract: paper/distress/death/vintage/health interrupt; the noisy
    # kinds (percentile_move, filing_detected, fundamentals_updated) never do
    assert "percentile_move" not in HIGH_SEVERITY
    assert "filing_detected" not in HIGH_SEVERITY
    assert {"paper_degraded", "distress_risk", "universe_death",
            "health_critical"} <= HIGH_SEVERITY


class _FakeState:
    def __init__(self, alerts=(), brief=None):
        self._alerts = list(alerts)
        self._brief = brief

    def alerts(self, unseen_only=True, limit=100):
        return self._alerts[:limit]

    def kv_get(self, key):
        return self._brief if key == "digest_brief" else None


def test_morning_summary_leads_with_high_alerts_then_brief():
    from stockscan.ops.notify import morning_summary

    st = _FakeState(
        alerts=[_alert("distress_risk", "XYZ escalated overnight"),
                _alert("percentile_move", "AAPL moved")],
        brief={"answer": "Quiet night: all jobs finished ok. One name moved.",
               "_updated": "2026-07-13T06:30:00+00:00"})
    title, msg = morning_summary(st)
    assert title == "Argus this morning · 1 alert(s)"
    assert msg.splitlines()[0] == "XYZ escalated overnight"
    assert "Quiet night: all jobs finished ok." in msg      # first sentence only
    assert "One name moved" not in msg
    assert "AAPL moved" not in msg                          # routine stays in-app


def test_morning_summary_quiet_cases():
    from stockscan.ops.notify import morning_summary

    t, m = morning_summary(_FakeState())
    assert t == "Argus this morning" and m == "quiet overnight — nothing needs attention"
    _, m2 = morning_summary(_FakeState(alerts=[_alert("percentile_move")]))
    assert "1 routine alert(s) waiting" in m2
    # a night of nothing but filing notices reads as quiet, because it was
    _, m3 = morning_summary(_FakeState(
        alerts=[_alert("filing_detected", f"cik {i} filed") for i in range(13)]))
    assert m3 == "quiet overnight — nothing needs attention"


def test_deliver_morning_modes(monkeypatch):
    from stockscan.ops import notify as n

    st = _FakeState(alerts=[_alert("health_critical", "prices stale")])
    assert n.deliver_morning(st, mode="off")["delivered"] is False
    monkeypatch.setattr(n, "_osascript_available", lambda: True)
    sent = []
    monkeypatch.setattr(n, "notify_mac", lambda t, m: sent.append((t, m)) or True)
    out = n.deliver_morning(st, mode="auto")
    assert out["delivered"] is True and out["high"] == 1
    assert "prices stale" in sent[0][1]
