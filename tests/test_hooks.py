# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress hooks: stages, and the progress log (its timing on a fake clock)."""

import logging

import pytest

from hxex.hooks import ProgressLog, format_duration, report, start_stage


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
        report(log, 150, 1000)  # within the interval: not logged
        clock.now = 60.0
        report(log, 150, 1000)  # half the stage in 60 s
        start_stage(log, "bootstrap", 300, 1000)
        clock.now = 660.0
        report(log, 370, 1000)  # a tenth of the bootstrap in 600 s: its own rate, not the fit's
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
        "bootstrap: 10% after 10m 00s, about 1h 30m left",
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
        log(1, 4)
        log(4, 4)
    assert lines(caplog) == ["progress: 0% after 5s", "progress: 25% after 20s, about 1m 00s left",
                             "done in 20s"]


def test_plain_hooks_are_not_told_of_stages_and_settings_are_checked():
    seen = []
    start_stage(lambda d, t: seen.append(d), "fit", 0, 10)  # no start_stage method: ignored
    start_stage(None, "fit", 0, 10)
    assert seen == []
    with pytest.raises(ValueError):
        ProgressLog(interval=-1)
