# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress hooks: stages, and the progress log (its timing on a fake clock)."""

import logging

import pytest

from hxex.hooks import ProgressLog, format_duration, format_percent, report, start_stage


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "hxex.progress"]


def test_durations_read_well():
    assert [format_duration(s) for s in (0, 44.6, 60, 245, 3599, 3600, 7830)] == [
        "0s", "45s", "1m 00s", "4m 05s", "59m 59s", "1h 00m", "2h 10m"]


def test_the_log_follows_stages_and_estimates_each_from_its_own_rate(caplog):
    clock = Clock()
    log = ProgressLog(interval=30.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        start_stage(log, "fit", 0, 300)
        clock.now = 10.0
        report(log, 60, 1000)  # within the interval: not logged
        clock.now = 60.0
        report(log, 150, 1000)  # half the stage in 60 s, the second update
        start_stage(log, "bootstrap", 300, 1000)
        clock.now = 660.0
        report(log, 370, 1000)  # a tenth of the bootstrap in 600 s: one update, no estimate yet
        clock.now = 670.0
        report(log, 380, 1000)  # too soon after the last line
        clock.now = 700.0
        report(log, 1000, 1000)
        report(log, 1000, 1000)  # the end is logged once
    assert lines(caplog) == [
        "fit: started",
        "fit: 50% after 1m 00s, about 1m 00s left",
        "fit: done in 1m 00s",
        "bootstrap: started",
        "bootstrap: 10% after 10m 00s",
        "bootstrap: done in 10m 40s",
        "done in 11m 40s",
    ]


def test_without_stages_the_whole_operation_is_one(caplog):
    clock = Clock()
    log = ProgressLog(interval=0.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        clock.now = 5.0
        log(0, 4)  # nothing done yet: no estimate
        clock.now = 20.0
        log(1, 4)  # a quarter, but the first update that moved: no estimate yet
        clock.now = 40.0
        log(2, 4)
        log(4, 4)
    assert lines(caplog) == ["progress: first results not in yet, 5s so far",
                             "progress: 25% after 20s", "progress: 50% after 40s, about 40s left",
                             "done in 40s"]


def test_no_time_left_until_it_is_credible_and_absurd_values_are_capped(caplog):
    """The analysis's 2 % after 5 s gave "about 4m 17s left" for a 13 s job, the bootstrap's
    first resample "about 25h 42m left" (audit 2026-11-01, Part 2 M1)."""
    clock = Clock()
    log = ProgressLog(interval=0.0, clock=clock, name="analysis")
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        for now, done in ((5.0, 2), (6.0, 5), (8.0, 9), (10.0, 20)):
            clock.now = now
            log(done, 100)
    assert lines(caplog) == ["analysis: 2% after 5s", "analysis: 5% after 6s",
                             "analysis: 9% after 8s", "analysis: 20% after 10s, about 40s left"]
    caplog.clear()
    log = ProgressLog(interval=0.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        start_stage(log, "bootstrap", 0, 100)
        clock.now += 3 * 3600.0
        log(5, 100)
        clock.now += 3 * 3600.0
        log(10, 100)  # a tenth in six hours: 54 hours left
    assert lines(caplog)[-1] == "bootstrap: 10% after 6h 00m, more than 10h left"


def test_a_stage_without_news_still_says_it_is_working(caplog):
    """While no update arrives, a line comes at the interval (the heartbeat), the latest state
    again once there is one; nothing after the operation ended or was closed."""
    clock = Clock()
    log = ProgressLog(interval=30.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        start_stage(log, "bootstrap", 0, 100)
        clock.now = 20.0
        log.beat()  # within the interval: nothing
        clock.now = 180.0
        log.beat()
        clock.now = 200.0
        log(20, 100)  # too soon after the heartbeat's line: kept
        clock.now = 215.0
        log(40, 100)  # logged by the call itself
        clock.now = 260.0
        log.beat()  # the latest state again
        start_stage(log, "write", 100, 100)  # a stage without units
        clock.now = 300.0
        log.beat()
        log(100, 100)
        clock.now = 400.0
        log.beat()  # done: nothing more
    assert lines(caplog) == [
        "bootstrap: started",
        "bootstrap: first results not in yet, 3m 00s so far",
        "bootstrap: 40% after 3m 35s, about 5m 22s left",
        "bootstrap: 40% after 4m 20s, about 6m 30s left",
        "bootstrap: done in 4m 20s",
        "write: started",
        "write: still working, 40s so far",
        "write: done in 40s",
        "done in 5m 00s",
    ]
    caplog.clear()
    with ProgressLog(interval=30.0, clock=clock) as closed, \
            caplog.at_level(logging.INFO, logger="hxex.progress"):
        start_stage(closed, "fit", 0, 10)
    clock.now += 100.0
    closed.beat()  # closed (an error, Ctrl-C): no line after it
    assert lines(caplog) == ["fit: started"]


def test_the_heartbeat_thread_runs_only_while_the_operation_does(caplog):
    """With the real clock a daemon thread gives the heartbeat; it ends when the operation is
    done or the log closed, and a fake clock (tests) starts none."""
    import time

    assert ProgressLog(clock=Clock()).heartbeat is False
    assert ProgressLog(interval=0.0).heartbeat is False  # every update is logged anyway
    log = ProgressLog(interval=0.05)
    with caplog.at_level(logging.INFO, logger="hxex.progress"):
        start_stage(log, "fit", 0, 10)
        thread = log._thread
        assert thread is not None and thread.daemon
        time.sleep(0.6)
        log(10, 10)
        thread.join(2.0)
        assert not thread.is_alive()
        n = len(lines(caplog))
        time.sleep(0.4)
        assert len(lines(caplog)) == n  # nothing after the end
    assert any("first results not in yet" in line for line in lines(caplog))
    with ProgressLog(interval=0.05) as log:
        start_stage(log, "fit", 0, 10)
        thread = log._thread
    thread.join(2.0)
    assert not thread.is_alive()


def test_plain_hooks_are_not_told_of_stages_and_settings_are_checked():
    seen = []
    start_stage(lambda d, t: seen.append(d), "fit", 0, 10)  # no start_stage method: ignored
    start_stage(None, "fit", 0, 10)
    assert seen == []
    with pytest.raises(ValueError):
        ProgressLog(interval=-1)


def test_the_fit_progress_weighs_each_restart_by_its_cost():
    """A K = 3 restart costs about three times a K = 1 one: counting restarts alike made the
    fit's time left far too short (audit 2026-11-01, Part 2 M1). Calls that do not fit the
    expected schedule count alike."""
    from types import SimpleNamespace

    from hxex.fit import FitConfig
    from hxex.pipeline import UNITS, _FitSpan

    seen = []
    config = FitConfig(k_min=1, k_max=3, n_restarts=2, nested_init=False)
    span = _FitSpan(lambda d, t: seen.append(d), 0, UNITS, config, SimpleNamespace(role=None))
    for done in range(1, 7):
        span(done, 6)
    assert seen == [round(UNITS * w / 12) for w in (1, 2, 4, 6, 9, 12)]
    seen.clear()
    span(1, 4)  # not the expected schedule: by count
    assert seen == [UNITS // 4]
    # the controls' extra K = 1 when k_min is 2, and the warm start's extra restart
    config = FitConfig(k_min=2, k_max=3, n_restarts=1, nested_init=True, kmeans_after_warm=True)
    span = _FitSpan(None, 0, UNITS, config, SimpleNamespace(role=["undeut", ""]))
    assert len(span.cumulative) == 1 + 1 + 2  # K = 1 once, K = 2 once, K = 3 once plus warm


def test_a_programs_report_gets_records_of_the_stages_and_the_end():
    clock = Clock()
    records = []
    log = ProgressLog(interval=30.0, clock=clock, heartbeat=False, report=records.append)
    start_stage(log, "fit", 0, 100)
    clock.now = 1.0
    report(log, 20, 200)
    clock.now = 1.1
    report(log, 30, 200)  # within REPORT_EVERY of the last record: none
    clock.now = 2.0
    report(log, 50, 200)
    clock.now = 3.0
    report(log, 200, 200)
    assert [r["event"] for r in records] == ["stage", "progress", "progress", "done"]
    first = records[1]
    assert first["stage"] == "fit" and first["fraction"] == 0.2 and first["overall"] == 0.1
    assert first["left"] is None  # one update: no estimate yet
    assert records[2]["left"] == 2.0 and records[2]["fraction"] == 0.5  # 1 s per 25 %
    assert records[-1]["overall"] == 1.0 and records[-1]["elapsed"] == 3.0

    def broken(record):
        raise RuntimeError("the program's hook failed")

    quiet = ProgressLog(interval=30.0, clock=clock, heartbeat=False, report=broken)
    report(quiet, 1, 10)  # a failing hook is dropped, never stops the operation
    assert quiet.report is None


def test_percents_below_ten_keep_a_decimal():
    # a slow stage still shows it moving (the fits read 0 % for minutes; owner, 2026-11-08)
    assert [format_percent(f) for f in (0, 0.004, 0.02, 0.0996, 0.12, 1)] == [
        "0%", "0.4%", "2%", "10%", "12%", "100%"]
