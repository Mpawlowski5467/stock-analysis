"""Ops state store: idempotent writes, the honesty trail, the paper book."""

import pytest

from stockscan.ops.state import OpsState


@pytest.fixture
def state(tmp_path):
    with OpsState(tmp_path / "state.sqlite") as st:
        yield st


def test_job_run_logging(state):
    run_id = state.job_start("prices")
    assert state.last_run("prices")["status"] == "running"
    state.job_finish(run_id, "ok", {"written": 3})
    last = state.last_run("prices")
    assert last["status"] == "ok"
    assert last["deltas"] == {"written": 3}
    assert state.last_run("prices", status="failed") is None
    assert state.last_run("universe") is None


def test_kv_roundtrip_and_overwrite(state):
    assert state.kv_get("digest_brief") is None
    state.kv_set("digest_brief", {"answer": "quiet night"})
    got = state.kv_get("digest_brief")
    assert got["answer"] == "quiet night" and got["_updated"]
    state.kv_set("digest_brief", {"answer": "busy night"})
    assert state.kv_get("digest_brief")["answer"] == "busy night"   # last write wins


def test_recent_runs_newest_first(state):
    a = state.job_start("prices")
    state.job_finish(a, "ok")
    b = state.job_start("monitor")
    state.job_finish(b, "degraded")
    runs = state.recent_runs(limit=10)
    assert [r["job"] for r in runs] == ["monitor", "prices"]
    assert runs[0]["status"] == "degraded" and runs[1]["finished"]


def test_reap_stale_runs_aborts_only_old_running_rows(state):
    # a stranded row from a killed process, backdated past the age guard
    old_id = state.job_start("prices")
    state._db.execute("update job_runs set started = '2026-01-01T00:00:00+00:00' "
                      "where id = ?", (old_id,))
    state._db.commit()
    state.job_start("monitor")             # genuinely running right now
    done_id = state.job_start("news")
    state.job_finish(done_id, "ok")

    reaped = state.reap_stale_runs(max_age_hours=24)
    assert [r["id"] for r in reaped] == [old_id]
    assert state.last_run("prices")["status"] == "aborted"
    assert state.last_run("prices")["deltas"]["reaped"] is True
    assert state.last_run("monitor")["status"] == "running"   # age guard held
    assert state.last_run("news")["status"] == "ok"
    assert state.reap_stale_runs(max_age_hours=24) == []      # idempotent


def test_watchlist_add_remove_idempotent(state):
    state.watch_add(320193, "AAPL", "core")
    state.watch_add(320193, "AAPL", "core")  # replay is safe
    assert [w["cik"] for w in state.watchlist()] == [320193]
    state.watch_remove(320193)
    assert state.watchlist() == []
    state.watch_add(320193, "AAPL")  # re-adding reactivates
    assert len(state.watchlist()) == 1


def test_signal_state_roundtrip(state):
    assert state.get_signal(1) is None
    state.record_signal(1, 85, 9, "2026-06-30")
    state.record_signal(1, 70, 7, "2026-07-01")  # upsert, not append
    sig = state.get_signal(1)
    assert (sig["percentile"], sig["decile"], sig["as_of"]) == (70, 7, "2026-07-01")


def test_signal_state_carries_firewalled_distress(state):
    state.record_signal(1, 85, 9, "2026-06-30")            # no distress passed -> None
    assert state.get_signal(1)["distress"] is None
    state.record_signal(1, 85, 9, "2026-07-01", distress=0.041)
    assert state.get_signal(1)["distress"] == pytest.approx(0.041)


def test_signal_state_migrates_old_db_without_distress_column(tmp_path):
    """An ops DB created before the distress column must gain it on next open."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "create table signal_state (cik integer primary key, percentile integer, "
        "decile integer, as_of text, updated text);"
    )
    con.commit()
    con.close()

    with OpsState(path) as st:                              # _migrate adds the column
        st.record_signal(7, 50, 5, "2026-07-01", distress=0.09)
        assert st.get_signal(7)["distress"] == pytest.approx(0.09)


def test_add_filings_returns_only_news(state):
    rows = [
        {"cik": 1, "form": "10-K", "filed_date": "2026-02-01",
         "period_end": "2025-12-31", "source": "fsds"},
        {"cik": 1, "form": "10-Q", "filed_date": "2026-05-01",
         "period_end": "2026-03-31", "source": "edgar"},
    ]
    assert len(state.add_filings(rows)) == 2
    assert state.add_filings(rows) == []  # replay: nothing new
    assert state.has_filings(1)
    assert state.has_filings(1, source="fsds")
    assert not state.has_filings(2)
    assert state.latest_filing_date(1) == "2026-05-01"


def test_same_day_multi_period_filings_all_recorded(state):
    """A delinquent filer catching up files several same-form docs on one day —
    period_end is part of the key, so none of them is swallowed."""
    rows = [
        {"cik": 9, "form": "10-Q", "filed_date": "2026-06-01",
         "period_end": pe, "source": "edgar"}
        for pe in ("2025-09-30", "2025-12-31", "2026-03-31")
    ]
    assert len(state.add_filings(rows)) == 3


def test_alerts_flow(state):
    state.add_alert("percentile_move", "cik 1 moved", cik=1, payload={"from": 80, "to": 60})
    state.add_alert("filing_detected", "cik 2 filed", cik=2)
    unseen = state.alerts()
    assert len(unseen) == 2
    assert state.mark_alerts_seen([unseen[0]["id"]]) == 1
    assert len(state.alerts()) == 1
    assert len(state.alerts(unseen_only=False)) == 2


def test_unseen_count_excludes_routine_kinds_without_hiding_them(state):
    """The count is a claim that something needs looking at; the list is the record.
    13 of 25 unseen alerts were filing notices, which is how the three that mattered
    stayed unread for a month."""
    state.add_alert("drawdown_risk", "held name entered high drawdown risk", cik=1)
    for cik in (2, 3, 4):
        state.add_alert("filing_detected", f"cik {cik} filed a 10-Q", cik=cik)

    assert state.unseen_count() == 1              # only the one that wants attention
    assert len(state.alerts()) == 4               # but nothing is hidden from the list

    # an explicit empty exclusion still counts everything (the pre-change behaviour)
    assert state.unseen_count(exclude=frozenset()) == 4

    state.mark_alerts_seen()
    assert state.unseen_count() == 0


def test_unseen_count_is_zero_when_only_routine_alerts_are_waiting(state):
    """The badge must go quiet, not sit on a permanent non-zero from bookkeeping."""
    state.add_alert("filing_detected", "cik 9 filed a 10-K", cik=9)
    assert state.unseen_count() == 0
    assert len(state.alerts()) == 1


def test_book_transitions_idempotent(state):
    state.book_apply({1: "A", 2: "B"}, set(), "2026-06-30")
    assert set(state.book()) == {1, 2}
    state.book_apply({3: "C"}, {1}, "2026-07-31")
    assert set(state.book()) == {2, 3}
    # replaying the same rebalance (crash-heal path) changes nothing
    state.book_apply({3: "C"}, {1}, "2026-07-31")
    assert set(state.book()) == {2, 3}
    # re-entry after exit reactivates
    state.book_apply({1: "A"}, set(), "2026-08-31")
    assert set(state.book()) == {1, 2, 3}


def test_positions_set_update_remove(state):
    """Positions store: upsert (add-or-update in one call), list, hard remove.
    Cost basis is personal live-view data — this store only round-trips it back."""
    assert state.positions() == []
    state.position_set(320193, 10, 150.0)          # add
    state.position_set(320193, 25, 172.5)          # update: upsert, not a second row
    rows = state.positions()
    assert len(rows) == 1
    assert (rows[0]["cik"], rows[0]["shares"], rows[0]["cost_basis"]) == (320193, 25, 172.5)
    state.position_set(789019, 5, 400.0)           # a second name
    assert {r["cik"] for r in state.positions()} == {320193, 789019}
    state.position_remove(320193)                  # hard delete
    assert [r["cik"] for r in state.positions()] == [789019]


def test_positions_update_preserves_added_at(state):
    """Re-saving a holding keeps the original added_at (mirrors watch_add)."""
    state.position_set(1, 3, 10.0)
    first = state.positions()[0]["added_at"]
    state.position_set(1, 7, 12.0)
    assert state.positions()[0]["added_at"] == first
