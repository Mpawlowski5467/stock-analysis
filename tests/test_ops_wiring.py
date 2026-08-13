"""The nightly's job wrappers — argument wiring, which nothing else covers.

scripts/ops.py is the unattended entry point, and its `job_*` wrappers are thin
enough to look obviously correct. One wasn't: `job_universe` forwarded
`state=state` through `_run_logged(state, job, fn, ...)`, where `state` is the
wrapper's OWN parameter, so the call raised TypeError before the job ever
started. It stayed invisible for four weeks because `_universe_due` only fires
every 28 days — and when it did fire it killed the rest of the nightly with it.

These tests exercise the wiring (a real call, a stub job) and pin the class of
mistake statically, since a `job_*` wrapper that never runs logs nothing at all.
"""

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

_OPS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops.py"
_SPEC = importlib.util.spec_from_file_location("ops_script", _OPS_PATH)
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


class _State:
    """Duck-typed OpsState: records the job_start/job_finish pair (test_assist convention)."""

    def __init__(self):
        self.runs = []

    def job_start(self, job):
        self.runs.append({"job": job, "status": "running", "deltas": None})
        return len(self.runs) - 1

    def job_finish(self, run_id, status, deltas):
        self.runs[run_id].update(status=status, deltas=deltas)


def test_job_universe_forwards_state_and_logs_ok(monkeypatch):
    """The regression: refresh_universe must actually receive the OpsState."""
    seen = {}

    def fake_refresh(**kwargs):
        seen.update(kwargs)
        return {"added": 3, "dead": 1}

    import stockscan.ops.jobs as jobs
    monkeypatch.setattr(jobs, "refresh_universe", fake_refresh)

    state = _State()
    deltas = ops.job_universe(state)

    assert seen["state"] is state          # the wrapper's state reached the job
    assert deltas == {"added": 3, "dead": 1}
    assert state.runs == [{"job": "universe", "status": "ok", "deltas": {"added": 3, "dead": 1}}]


def test_job_universe_records_failure_before_reraising(monkeypatch):
    """A job that raises still leaves a 'failed' row — the health screen reads these."""
    def boom(**_):
        raise RuntimeError("vendor 500")

    import stockscan.ops.jobs as jobs
    monkeypatch.setattr(jobs, "refresh_universe", boom)

    state = _State()
    with pytest.raises(RuntimeError):
        ops.job_universe(state)

    assert state.runs[0]["status"] == "failed"
    assert "RuntimeError: vendor 500" in state.runs[0]["deltas"]["error"]


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def test_job_reload_pokes_the_app_and_logs_ok(monkeypatch):
    """The nightly must actually hand the running app its new cross-section."""
    posted = {}

    def fake_post(url, **kwargs):
        posted["url"] = url
        return _Resp(200)

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    state = _State()
    deltas = ops.job_reload(state)

    assert posted["url"].endswith("/api/reload")
    assert deltas["reloaded"] is True
    assert state.runs[0]["job"] == "reload" and state.runs[0]["status"] == "ok"


def test_job_reload_noops_when_the_app_is_down(monkeypatch):
    """The app is optional: an unreachable server is a noop, never a failed night."""
    def refuse(url, **kwargs):
        raise OSError("connection refused")

    import httpx
    monkeypatch.setattr(httpx, "post", refuse)

    state = _State()
    deltas = ops.job_reload(state)          # must not raise

    assert deltas["reloaded"] is False
    assert state.runs[0]["status"] == "noop"


def test_job_reload_noops_on_a_non_200(monkeypatch):
    """Something else on the port answers — report it, don't claim a reload happened."""
    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Resp(404))

    state = _State()
    deltas = ops.job_reload(state)

    assert deltas["reloaded"] is False
    assert state.runs[0]["status"] == "noop"
    assert "404" in deltas["note"]


def test_nightly_reloads_the_app_after_the_data_jobs():
    """Static: the nightly must call job_reload, and only after the facade-visible
    stores are written. A reload that runs before ingestion would hand the app the
    OLD data and still look like it worked."""
    tree = ast.parse(_OPS_PATH.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "job_nightly")
    called = [n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]

    assert "job_reload" in called, "the nightly never tells the web app to re-read disk"
    order = {name: called.index(name) for name in set(called)}
    for earlier in ("job_prices", "job_monitor", "job_themes"):
        assert order[earlier] < order["job_reload"], (
            f"{earlier} must run before job_reload — reloading first serves stale data")


def test_no_run_logged_callsite_shadows_the_wrappers_own_parameters():
    """Static guard on the whole class of bug, for every callsite at once.

    `_run_logged(state, job, fn, *args, **kwargs)` forwards **kwargs to fn, so any
    keyword sharing a name with _run_logged's own parameters binds to the wrapper
    instead and raises. Pass those through a lambda.
    """
    reserved = set(inspect.signature(ops._run_logged).parameters) - {"args", "kwargs"}
    tree = ast.parse(_OPS_PATH.read_text())

    offenders = [
        (node.lineno, kw.arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_logged"
        for kw in node.keywords
        if kw.arg in reserved
    ]
    assert not offenders, (
        "these _run_logged callsites pass a keyword that collides with the wrapper's "
        f"own signature {sorted(reserved)} — wrap the job in a lambda instead: {offenders}"
    )
