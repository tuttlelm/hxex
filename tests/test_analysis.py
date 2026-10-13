# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-fitting analysis: the species uptake table from a run, residue-level values (checked by
hand; the same as pyHDX 0.4.3 on real data, docs/HISTORY.md), coverage tables and the coverage
and redundancy/resolution plots."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from hxex.analysis import (
    SequenceMismatch,
    SpeciesFilters,
    SpeciesUptake,
    coverage_table,
    exchanging_residues,
    residue_map,
    species_uptake,
)
from hxex.analysis.species import _combine, _is_trend
from hxex.io.bundle import ReviewTable
from hxex.report import Results, plot_residue_profile
from hxex.report.coverage import (
    FD_CMAP,
    autowrap,
    coverage_figure,
    plot_coverage,
    plot_redundancy_resolution,
)
from hxex.run import run
from test_run import config, raw_spectra  # noqa: I001

SEQUENCE = "MAKLPGSTVEDRWQAFKLT"  # residue 1 is M; a proline at 5


def table(rows, *, nterm=1, undeut_time=0.0):
    """A species table from (sample, peptide, start, time, fD, fD_se[, n_spectra])."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend")}
    for row in rows:
        sample, peptide, start, time, fd, se = row[:6]
        n = row[6] if len(row) > 6 else 1
        values = {"sample": sample, "peptide": peptide, "modification": "", "kind": "peptide",
                  "start": start, "end": start + len(peptide) - 1, "time": time, "K": 1,
                  "rank": 1, "mean": fd * 10, "mean_se": se * 10, "fD": fd, "fD_se": se,
                  "population": 1.0, "population_se": 0.0, "n_amides": 10.0, "n_spectra": n,
                  "n_charges": 1, "corrected": True, "charge_trend": False}
        for name, value in values.items():
            cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()}, nterm_subtract=nterm,
                         undeut_time=undeut_time)


# --- residue level ------------------------------------------------------------------------------

def test_exchanging_residues_drop_the_n_terminus_and_prolines():
    assert list(exchanging_residues("AKLPGS", 2, 1)) == [3, 4, 6, 7]  # P at 5 left out
    assert list(exchanging_residues("AKLPGS", 2, 2)) == [4, 6, 7]
    assert list(exchanging_residues("ApSK", 10, 1)) == [11, 12]  # pS is one residue


def test_residue_values_are_the_pyhdx_weighted_mean():
    # peptide a: residues 2-9 (AKLPGSTV), exchanging 3, 4, 6, 7, 8, 9 -> weight 1/6 each
    # peptide b: residues 6-11 (GSTVED), exchanging 7-11 -> weight 1/5 each
    up = table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.02), ("S", "GSTVED", 6, 4.0, 0.6, 0.04)])
    rm = residue_map(up, sequences=SEQUENCE)
    got = {int(r): (v, s, n, res) for r, v, s, n, res in zip(
        rm.residue, rm.fD, rm.fD_se, rm.redundancy, rm.resolution, strict=True)}
    wa, wb = (1 / 6) / (1 / 6 + 1 / 5), (1 / 5) / (1 / 6 + 1 / 5)
    v, s, n, _ = got[8]  # covered by both
    assert v == pytest.approx(wa * 0.3 + wb * 0.6) and n == 2
    assert s == pytest.approx(np.hypot(wa * 0.02, wb * 0.04))
    assert got[3][0] == pytest.approx(0.3) and got[3][2] == 1
    assert got[10][0] == pytest.approx(0.6) and got[10][3] == 2  # block 10-11 is b's alone
    # resolution: blocks between the boundaries 3, 7, 10, 12 of the exchanging stretches
    assert [got[r][3] for r in (3, 4, 6, 7, 8, 9, 10, 11)] == [4, 4, 4, 3, 3, 3, 2, 2]
    for r in (1, 2, 5, 12, 19):  # not an exchanging amide: residue 2 is a's N-terminus, 5 is P
        assert np.isnan(got[r][0]) and got[r][2] == 0 and np.isnan(got[r][3])
    assert len(rm) == len(SEQUENCE) and rm.proline[4] and rm.aa[4] == "P" and rm.aa[18] == "T"
    equal = residue_map(up, sequences=SEQUENCE, weight_exponent=0.0)  # plain mean of a and b
    assert equal.fD[equal.residue == 8][0] == pytest.approx(0.45)


def test_without_a_sequence_residues_come_from_the_peptides():
    up = table([("S", "KLPGS", 3, 0.0, 0.0, 0.0)])
    rm = residue_map(up)
    assert len(rm) == 7 and list(rm.aa) == ["", "", "K", "L", "P", "G", "S"]
    assert rm.proline[4] and not rm.covered[2]  # residue 3 is the N-terminus here


def test_peptides_are_checked_against_the_sequence():
    up = table([("S", "KLPGS", 3, 0.0, 0.0, 0.0)])
    residue_map(up, sequences={"S": SEQUENCE})
    with pytest.raises(SequenceMismatch, match="KLPGS at 3"):
        residue_map(up, sequences={"S": "MAKLAGSTV"})
    with pytest.raises(ValueError, match="weight_exponent"):
        residue_map(up, weight_exponent=-1)


def test_coverage_is_judged_at_the_chosen_times():
    up = table([("S", "AKLPGSTV", 2, 0.0, 0.0, 0.0, 3), ("S", "GSTVED", 6, 0.0, 0.0, 0.0, 2),
                ("S", "TVEDRWQ", 8, 60.0, 0.5, 0.1, 1)], undeut_time=0.0)
    undeut = coverage_table(up, sequences=SEQUENCE)  # the undeuterated control by default
    assert undeut.times == (0.0,) and int(undeut.redundancy[undeut.residue == 13][0]) == 0
    assert int(undeut.redundancy[undeut.residue == 8][0]) == 2
    spectra = coverage_table(up, sequences=SEQUENCE, count="spectra")
    assert int(spectra.redundancy[spectra.residue == 8][0]) == 5
    every = coverage_table(up, sequences=SEQUENCE, time="all")
    assert every.times == () and int(every.redundancy[every.residue == 13][0]) == 1
    late = coverage_table(up, sequences=SEQUENCE, time=[60.0])
    assert int(late.redundancy[late.residue == 3][0]) == 0
    with pytest.raises(ValueError, match="no peptide was measured"):
        coverage_table(up, time=5.0)
    with pytest.raises(ValueError, match="count"):
        coverage_table(up, count="charges")


# --- the species table from a run ---------------------------------------------------------------

@pytest.fixture(scope="module")
def results():
    raw = raw_spectra(n_series=2)
    return Results.from_run(run(raw, config()))


def test_species_uptake_combines_replicates_at_each_fixed_k(results):
    up = species_uptake(results, ks=(1, 2))
    assert set(up.K.tolist()) == {1, 2} and set(up.rank[up.K == 2].tolist()) == {1, 2}
    assert up.corrected.all() and up.undeut_time == 0.0 and up.nterm_subtract == 1
    k1 = up.select(up.where(K=1))
    # both series are the same peptide of the same sample (as two charge states would be), so they
    # are combined: 2 x 3 replicates, 2 x 2 at the last time
    assert sorted(set(k1.n_spectra.tolist())) == [4, 6] and len(set(k1.time.tolist())) == len(k1)
    np.testing.assert_allclose(k1.fD, k1.mean / k1.n_amides)
    np.testing.assert_allclose(k1.population, 1.0)
    k2 = up.select(up.where(K=2))
    for s, t in set(zip(k2.sample, k2.time, strict=True)):
        rows = k2.where(sample=s, time=t)
        assert k2.fD[rows & (k2.rank == 1)] <= k2.fD[rows & (k2.rank == 2)]
    # the fully deuterated control is near 1 after the back-exchange correction
    fd_time = k1.fD[k1.time == k1.time.max()]
    assert np.all(np.abs(fd_time - 1.0) < 0.2)
    with pytest.raises(ValueError, match="were not fitted"):
        species_uptake(results, ks=(4,))


def test_rejected_spectra_are_left_out(results):
    first = str(results.fit.spectrum_id[0])
    review = ReviewTable.empty().update(first, status="rejected")
    up = species_uptake(results.with_review(review), ks=(1,))
    full = species_uptake(results, ks=(1,))
    assert int(up.n_spectra.sum()) == int(full.n_spectra.sum()) - 1


def test_filters_keep_only_credible_k2_descriptions(results, caplog):
    import logging

    none = species_uptake(results, ks=(1, 2), filters=None)
    default = species_uptake(results, ks=(1, 2))
    assert len(default.select(default.K == 1)) == len(none.select(none.K == 1))  # K=1 untouched
    assert len(default) <= len(none)
    # every kept K=2 population within the limits, and every component's n is fine
    k2 = default.select(default.K == 2)
    assert np.all((k2.population >= 0.05) & (k2.population <= 0.95))
    # the K=1 value lies between the ranks of every kept K=2 description
    for s, p, t in set(zip(k2.sample, k2.peptide, k2.time, strict=True)):
        row = default.where(sample=s, time=t) & (default.peptide == p)
        low, high = (default.fD[row & (default.K == 2) & (default.rank == r)][0] for r in (1, 2))
        avg = default.fD[row & (default.K == 1)][0]
        assert low < avg < high
    unordered = species_uptake(results, ks=(1, 2), filters=SpeciesFilters(k1_between_ranks=False))
    assert (unordered.K == 2).sum() >= (default.K == 2).sum()
    with caplog.at_level(logging.INFO, logger="hxex.analysis.species"):
        strict = species_uptake(results, ks=(1, 2), filters=SpeciesFilters(max_fD=0.5))
    # no kept K=2 spectrum had a component above fD 0.5, so its combined values are not either
    assert (strict.K == 2).sum() < (default.K == 2).sum()
    assert np.all(strict.fD[strict.K == 2] <= 0.5 + 1e-12)
    assert "above fD 0.5" in caplog.text
    tight = species_uptake(results, ks=(2,), filters=SpeciesFilters(max_fD_se=1e-9))
    assert np.all(~np.isfinite(tight.fD_se) | (tight.fD_se < 1e-9))
    for bad in ({"min_population": 0.5}, {"max_fD_se": 0}, {"max_fD": -1}):
        with pytest.raises(ValueError):
            SpeciesFilters(**bad)


def test_combining_counts_neither_variance_twice():
    mean, se = _combine(np.array([1.0, 1.2, 0.8]), np.array([0.01, 0.01, 0.01]))
    assert mean == pytest.approx(1.0) and se == pytest.approx(0.2 / np.sqrt(3))  # scatter wins
    mean, se = _combine(np.array([1.0, 1.0]), np.array([0.3, 0.4]))
    assert se == pytest.approx(np.sqrt((0.09 + 0.16) / 2 / 2))  # own errors win
    assert _combine(np.array([0.5]), np.array([0.07])) == (0.5, 0.07)


def test_a_charge_trend_needs_a_consistent_difference():
    steady = [((0.30, 0.01), (0.40, 0.01))] * 3  # z_high above z_low every time
    assert _is_trend(steady)
    assert not _is_trend(steady[:1])  # one time is not a trend
    noisy = [((0.30, 0.01), (0.40, 0.01)), ((0.40, 0.01), (0.30, 0.01)),
             ((0.30, 0.05), (0.31, 0.05))]
    assert not _is_trend(noisy)


# --- plots --------------------------------------------------------------------------------------

def test_autowrap_keeps_peptides_apart():
    assert autowrap([1, 1, 5, 12], [10, 8, 20, 30]) == 5
    starts = np.arange(0, 60, 2)
    assert autowrap(starts, starts + 20) >= 10  # long overlapping peptides need more rows


def test_coverage_plots():
    up = table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.02), ("S", "GSTVED", 6, 4.0, np.nan, 0.04),
                ("T", "GSTVED", 6, 4.0, 0.9, 0.04)])
    fig = plot_coverage(up, "S", 4.0, length=19)
    assert isinstance(fig, Figure)
    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert sorted(b.get_gid() for b in bars) == ["peptide:2-9", "peptide:6-11"]
    nan_bar = next(b for b in bars if b.get_gid() == "peptide:6-11")
    assert np.allclose(nan_bar.get_facecolor(), FD_CMAP.get_bad())  # unknown fD is grey
    assert fig.axes[0].get_xlim() == (0.0, 19.0) and len(fig.axes) == 2  # and the colour bar
    grid = coverage_figure(up, samples=["S", "T"], times=[4.0])
    assert len([a for a in grid.axes if a.patches]) == 2
    with pytest.raises(ValueError, match="no peptides"):
        plot_coverage(up, "S", 60.0)


def test_redundancy_and_resolution_plot():
    up = table([("WT", "AKLPGSTV", 2, 0.0, 0.0, 0.0), ("WT", "GSTVED", 6, 0.0, 0.0, 0.0),
                ("V", "AKLPGSTV", 2, 0.0, 0.0, 0.0)])
    cov = coverage_table(up, sequences=SEQUENCE)
    fig = plot_redundancy_resolution(cov, samples=["WT", "V"], marks=(7,))
    top, bottom = fig.axes[:2]
    assert [line.get_gid() for line in top.lines if line.get_gid()] == [
        "redundancy:WT", "redundancy:V"]
    assert any(t.get_text() == "P" for t in top.texts)
    wt = next(line for line in top.lines if line.get_gid() == "redundancy:WT")
    y = np.asarray(wt.get_ydata(), dtype=float)
    assert np.isnan(y[4]) and y[7] == 2  # the proline is a gap; residue 8 has both peptides
    with pytest.raises(ValueError, match="no sample"):
        plot_redundancy_resolution(cov, samples=["X"])


def test_residue_profile():
    rows = [("S", "AKLPGSTV", 2, 4.0, 0.3, 0.02), ("S", "GSTVED", 6, 4.0, 0.6, 0.04)]
    up = table(rows)
    k2 = table(rows)  # the same peptides as K=2 rank 1 and 2
    import dataclasses

    both = SpeciesUptake(**{f.name: np.concatenate([getattr(up, f.name)] + [
        getattr(k2, f.name) if f.name not in ("K", "rank", "fD") else
        {"K": np.full(2, 2), "rank": np.full(2, r), "fD": k2.fD + 0.1 * r}[f.name]
        for r in (1, 2)]) for f in dataclasses.fields(SpeciesUptake)
        if f.name not in ("nterm_subtract", "undeut_time")}, nterm_subtract=1)
    rm = residue_map(both, sequences=SEQUENCE)
    fig = plot_residue_profile(rm, "S", 4.0, marks=(7,), undeut_time=0.0)
    ax = fig.axes[0]
    gids = [line.get_gid() for line in ax.lines if line.get_gid()]
    assert gids == ["fD:K1:rank1", "fD:K2:rank1", "fD:K2:rank2"]
    assert len(ax.collections) == 3  # one band per curve
    assert any(t.get_text() == "S 4 s" for t in ax.texts)
    k1 = next(line for line in ax.lines if line.get_gid() == "fD:K1:rank1")
    y = np.asarray(k1.get_ydata(), dtype=float)
    assert np.isnan(y[4]) and y[7] == pytest.approx(rm.fD[(rm.residue == 8) & (rm.K == 1)][0])
    no_band = plot_residue_profile(rm, "S", 4.0, band=0, show=((1, 1),))
    assert len(no_band.axes[0].collections) == 0
    with pytest.raises(ValueError, match="no residue values"):
        plot_residue_profile(rm, "S", 60.0)
    with pytest.raises(ValueError, match="band"):
        plot_residue_profile(rm, "S", 4.0, band=1.0)
