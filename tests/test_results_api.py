# SPDX-License-Identifier: GPL-3.0-or-later
"""The Results object at a Python prompt: a one-line summary, the samples, peptides, and times
it holds, and show_series naming what exists when nothing matches (docs/guide/python.md)."""

import numpy as np
import pytest

from hxex.io import ReviewTable
from hxex.pipeline import run
from hxex.report import Results, series_table, show_series
from test_run import config, raw_spectra


@pytest.fixture(scope="module")
def results():
    raw = raw_spectra(n_series=2)
    return Results.from_run(run(raw, config()), spectra=raw)


def test_the_summary_is_one_short_line(results):
    text = repr(results)
    assert "\n" not in text and len(text) < 300
    assert text.startswith("Results(") and f"{len(results)} spectra in 2 series" in text
    assert "bootstrap at selected K" in text and "no review" in text
    sid = str(results.fit.spectrum_id[0])
    reviewed = results.with_review(ReviewTable.empty().update(sid, status="rejected"))
    assert "review: 1 rejected" in repr(reviewed)


def test_samples_peptides_and_times(results):
    t = series_table(results)
    assert results.samples() == sorted(set(t["sample"]))
    sample = results.samples()[0]
    pairs = results.peptides(sample)
    assert pairs == sorted(set(zip(t["peptide_range"][t["sample"] == sample],
                                   t["peptide"][t["sample"] == sample], strict=True)))
    with pytest.raises(ValueError, match="samples: "):
        results.peptides("no such sample")
    times = results.times()
    labelled = np.asarray(results.fit.role, dtype=str) == ""
    assert times == sorted({float(x) for x in results.fit.time[labelled]})
    assert all(np.isfinite(times))


def test_show_series_names_what_exists(results):
    t = series_table(results)
    sample, rng = str(t["sample"][0]), str(t["peptide_range"][0])
    with pytest.raises(ValueError, match=r"no sample 'nope' \(samples: "):
        show_series(results, "nope", rng)
    with pytest.raises(ValueError, match="its charges in"):
        show_series(results, sample, rng, charge=9)
    with pytest.raises(ValueError, match="lists the peptides"):
        show_series(results, sample, "900-910")
