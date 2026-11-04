"""Test-suite wall-time budget (AGENTS.md, Run hygiene).

The default run (``-m 'not slow'``, set in the root pyproject.toml) must finish within ``BUDGET_SECONDS``
of wall time in total.  Over budget, the session fails with the measured time; the fix is to speed up or
``@pytest.mark.slow`` the offending test, never to raise the budget.  An explicit ``-m`` selection other
than the default (e.g. ``-m slow``) is not budgeted.
"""
from __future__ import annotations

import time

import pytest

BUDGET_SECONDS = 120.0
_DEFAULT_MARKEXPR = "not slow"
_state = {"start": None, "elapsed": None, "budgeted": False}


def pytest_sessionstart(session):
    _state["start"] = time.perf_counter()


def pytest_sessionfinish(session, exitstatus):
    if _state["start"] is None:
        return
    _state["elapsed"] = time.perf_counter() - _state["start"]
    _state["budgeted"] = session.config.getoption("markexpr", "").strip() == _DEFAULT_MARKEXPR
    if _state["budgeted"] and _state["elapsed"] > BUDGET_SECONDS and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    elapsed = _state["elapsed"]
    if elapsed is None:
        return
    if not _state["budgeted"]:
        terminalreporter.write_line(f"test-suite budget: not applied to -m {config.getoption('markexpr')!r} "
                                    f"({elapsed:.1f}s)")
    elif elapsed > BUDGET_SECONDS:
        terminalreporter.write_line(f"TEST-SUITE BUDGET EXCEEDED: {elapsed:.1f}s > {BUDGET_SECONDS:.0f}s -- speed up "
                                    "or @pytest.mark.slow the slowest tests (pytest --durations=20)", red=True)
    else:
        terminalreporter.write_line(f"test-suite budget: {elapsed:.1f}s of {BUDGET_SECONDS:.0f}s")
