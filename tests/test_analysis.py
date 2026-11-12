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
    left_out_spectra,
    residue_map,
    species_uptake,
)
from hxex.analysis.species import _combine, _is_trend
from hxex.io.bundle import ReviewTable
from hxex.picking.screen import SCREEN_SCORES
from hxex.pipeline import run
from hxex.report import Results, plot_residue_profile
from hxex.report.coverage import (
    FD_CMAP,
    autowrap,
    coverage_figure,
    plot_coverage,
    plot_redundancy_resolution,
)
from test_run import config, raw_spectra  # noqa: I001

SEQUENCE = "MAKLPGSTVEDRWQAFKLT"  # residue 1 is M; a proline at 5


def table(rows, *, nterm=1, undeut_time=0.0):
    """A species table from (sample, peptide, start, time, fD, fD_se[, n_spectra]); the rows
    at ``undeut_time`` are the undeuterated controls (role ``undeut``, time NaN)."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "role", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend")}
    for row in rows:
        sample, peptide, start, time, fd, se = row[:6]
        n = row[6] if len(row) > 6 else 1
        control = undeut_time is not None and time == undeut_time
        values = {"sample": sample, "peptide": peptide, "modification": "", "kind": "peptide",
                  "start": start, "end": start + len(peptide) - 1,
                  "time": np.nan if control else time, "role": "undeut" if control else "",
                  "K": 1, "rank": 1, "mean": fd * 10, "mean_se": se * 10, "fD": fd, "fD_se": se,
                  "population": 1.0, "population_se": 0.0, "n_amides": 10.0, "n_spectra": n,
                  "n_charges": 1, "corrected": True, "charge_trend": False}
        for name, value in values.items():
            cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()}, nterm_subtract=nterm)


# --- residue level ------------------------------------------------------------------------------

def test_exchanging_residues_drop_the_n_terminus_and_prolines():
    assert list(exchanging_residues("AKLPGS", 2, 1)) == [3, 4, 6, 7]  # P at 5 left out
    assert list(exchanging_residues("AKLPGS", 2, 2)) == [4, 6, 7]
    assert list(exchanging_residues("ApSK", 10, 1)) == [11, 12]  # pS is one residue
    assert list(exchanging_residues("Ac-AKLPGS", 1, 1)) == [1, 2, 3, 5, 6]  # Ac-: residue 1
    # the residue map: residue 1 has an amide where an Ac- peptide starts the protein
    for peptide, no_amide in (("MAKL", True), ("Ac-MAKL", False)):
        rm = residue_map(table([("S", peptide, 1, 4.0, 0.3, 0.02)]), sequences=SEQUENCE)
        first = np.flatnonzero(rm.residue == 1)[0]
        assert bool(rm.no_amide[first]) is no_amide
        assert bool(np.isfinite(rm.fD[first])) is not no_amide


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
    assert undeut.times == () and int(undeut.redundancy[undeut.residue == 13][0]) == 0
    assert int(undeut.redundancy[undeut.residue == 8][0]) == 2
    spectra = coverage_table(up, sequences=SEQUENCE, count="spectra")
    assert int(spectra.redundancy[spectra.residue == 8][0]) == 5
    every = coverage_table(up, sequences=SEQUENCE, time="all")
    assert every.times == () and int(every.redundancy[every.residue == 13][0]) == 1
    late = coverage_table(up, sequences=SEQUENCE, time=[60.0])
    assert int(late.redundancy[late.residue == 3][0]) == 0
    # residue 1 (the protein's N-terminus) and the proline at 5 can never carry a value
    assert undeut.no_amide[undeut.residue == 1][0] and undeut.no_amide[undeut.residue == 5][0]
    assert not undeut.no_amide[undeut.residue == 3][0] and not undeut.covered[0]
    from hxex.analysis.residues import NoPeptideAtTime

    with pytest.raises(NoPeptideAtTime, match="no peptide was measured"):
        coverage_table(up, time=5.0)
    assert issubclass(NoPeptideAtTime, ValueError)
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
    assert up.corrected.all() and up.nterm_subtract == 1
    assert set(up.role.tolist()) == {"undeut", "", "fulldeut"}
    assert np.isnan(up.time[up.role != ""]).all()
    k1 = up.select(up.where(K=1))
    # both series are the same peptide of the same sample (as two charge states would be), so they
    # are combined: 2 x 3 replicates, 2 x 2 at the last time
    assert sorted(set(k1.n_spectra.tolist())) == [4, 6] and len(set(k1.time.tolist())) == len(k1)
    np.testing.assert_allclose(k1.fD, k1.mean / k1.n_amides)
    np.testing.assert_allclose(k1.population, 1.0)
    k2 = up.select(up.where(K=2))
    for s, t in set(zip(k2.sample, k2.time, strict=True)):
        if not np.isfinite(t):
            continue  # the controls (no time) are checked by role above
        rows = k2.where(sample=s, time=t)
        assert k2.fD[rows & (k2.rank == 1)] <= k2.fD[rows & (k2.rank == 2)]
    # the fully deuterated control is near 1 after the back-exchange correction
    fd_level = k1.fD[k1.role == "fulldeut"]
    assert len(fd_level) and np.all(np.abs(fd_level - 1.0) < 0.2)
    with pytest.raises(ValueError, match="were not fitted"):
        species_uptake(results, ks=(4,))


def test_best_set_k_pools_the_evidence_of_a_set(results):
    """best_set_K: one choice per species and time from all its spectra (their criterion summed
    per K), passing over a K the filters left out or with a population below min_population;
    best_set marks the rows of the chosen K."""
    from hxex.analysis.species import Point, best_set

    up = species_uptake(results, ks=(1, 2))
    for key in set(zip(up.sample, up.peptide, up.time.tolist(), up.role, strict=True)):
        rows = ((up.sample == key[0]) & (up.peptide == key[1]) & (up.role == key[3])
                & ((up.time == key[2]) | (np.isnan(up.time) & np.isnan(key[2]))))
        k = set(up.best_set_K[rows].tolist())
        assert len(k) == 1 and set(up.K[rows & up.best_set].tolist()) == k  # one description
        assert (up.best_set[rows] == (up.K[rows] == k.pop())).all()
    timed = np.isfinite(up.set_delta_criterion)
    assert timed.any() and set(up.best_set_reason[timed]) <= {
        "set evidence for one", "set evidence for two"} | {
        r for r in up.best_set_reason if "population below" in r or "not kept" in r}
    assert set(up.best_set_K[(up.set_delta_criterion > 0)].tolist()) <= {1}

    def table(pops):  # one species at one time: K = 1, and K = 2 with these populations
        k = [1] + [2] * len(pops)
        return {"run": [""] * len(k), "sample": ["S"] * len(k), "peptide": ["P"] * len(k),
                "modification": [""] * len(k), "start": [1] * len(k), "end": [9] * len(k),
                "time": [60.0] * len(k), "role": [""] * len(k), "K": np.array(k),
                "population": [1.0, *pops]}

    key = ("", "S", "P", "", 1, 9, Point((60.0, "")))
    two = {key: {"n": 3, 1: 100.0, 2: 80.0}}  # summed: two populations explain the set better
    out = best_set(table([0.4, 0.6]), two, (1, 2))
    assert out["best_set_K"].tolist() == [2, 2, 2] and out["best_set"].tolist() == [0, 1, 1]
    assert out["best_set_reason"][0] == "set evidence for two"
    assert out["set_delta_criterion"][0] == -20.0
    small = best_set(table([0.02, 0.98]), two, (1, 2))
    assert small["best_set_K"][0] == 1 and "population below 0.05" in small["best_set_reason"][0]
    filtered = best_set(table([]), two, (1, 2))  # the filters left K = 2 out
    assert filtered["best_set_K"][0] == 1 and "not kept" in filtered["best_set_reason"][0]
    one = best_set(table([0.4, 0.6]), {key: {"n": 3, 1: 80.0, 2: 100.0}}, (1, 2))
    assert one["best_set_K"].tolist() == [1, 1, 1] and one["best_set"].tolist() == [1, 0, 0]
    none = best_set(table([0.4, 0.6]), {}, (1, 2))
    assert none["best_set_K"][0] == 1 and none["best_set_reason"][0].startswith("no spectrum")


def test_best_set_evidence_counts_only_the_spectra_the_table_keeps(results):
    """Audit 2026-10-28 High 1 and Medium 11: a spectrum the filters leave out whole adds
    nothing to its set's evidence (as if rejected), and one the review set to one population
    (review_K = 1) keeps its K = 1 description only and cannot vote for two."""
    import dataclasses

    from hxex.analysis.species import analysis_controls

    fit = results.fit
    controls = analysis_controls(results, None, quiet=True)
    n_amides = np.asarray(controls.n_amides, dtype=float)[controls.rows(fit.data_id)]
    timed = np.flatnonzero((np.asarray(fit.role) == "") & np.isfinite(fit.time))
    i = int(timed[-1])
    k1 = fit.fits[1]
    n = k1.n.copy()
    n[i, 0] = (n_amides[i] + 0.5) / k1.p[i, 0]  # above the amide count: left out whole
    fake = dataclasses.replace(results, fit=dataclasses.replace(
        fit, fits={**fit.fits, 1: dataclasses.replace(k1, n=n)}))
    sid = str(fit.spectrum_id[i])
    assert sid in set(left_out_spectra(fake)["spectrum_id"])

    def delta(r):
        up = species_uptake(r, ks=(1, 2))
        rows = up.where(time=float(fit.time[i]), K=1)
        return up.set_delta_criterion[rows], up.n_spectra[rows]

    left, n_left = delta(fake)
    rejected, n_rejected = delta(fake.with_review(
        ReviewTable.empty().update(sid, status="rejected")))
    np.testing.assert_allclose(left, rejected)
    assert (n_left == n_rejected).all()

    criterion = fit.config.criterion
    gain = float(getattr(fit.fits[2], criterion)[i] - getattr(fit.fits[1], criterion)[i])
    assert gain != 0
    base, _ = delta(results)
    one = results.with_review(ReviewTable.empty().update(sid, k_override=1))
    reviewed, _ = delta(one)
    np.testing.assert_allclose(reviewed, base - gain)
    full = species_uptake(results, ks=(2,))
    kept = species_uptake(one, ks=(2,))
    assert int(kept.n_spectra.sum()) == int(full.n_spectra.sum()) - 2  # both ranks
    assert int(species_uptake(one, ks=(1,)).n_spectra.sum()) == int(
        species_uptake(results, ks=(1,)).n_spectra.sum())


def test_series_left_out_of_the_analysis(results):
    """without_series: a series leaves the species table; the fragment series of a run that read
    its peptides from MS1 too are found by series, not by sequence."""
    from types import SimpleNamespace

    from hxex.fragments import fragment_targets
    from hxex.io import MzmlTarget
    from hxex.report.analysis import _fragment_series
    from hxex.spectra import PeptideEntry

    both = species_uptake(results, ks=(1,))
    one = species_uptake(results.without_series([1]), ks=(1,))
    assert results.without_series([1]).excluded.sum() == (results.fit.data_id == 1).sum() > 0
    assert one.n_spectra.sum() == both.n_spectra.sum() - (results.fit.data_id == 1).sum()
    parent = PeptideEntry("f", "S", "GSHPLVFK", "11-18", 11, 18, 3)
    targets = [MzmlTarget(parent)] + fragment_targets(parent, "M c:2", activation="ETD")
    source = {"reader": "mzml", "targets": [t.to_dict() for t in targets],
              "series": [{"data_id": i, "sample": "S", "target": i}
                         for i in range(len(targets))]}
    run = SimpleNamespace(settings=SimpleNamespace(spectra_source=source))
    series, only = _fragment_series(run)
    assert series == set(range(1, len(targets))) and not only  # the MS1 peptide is kept


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
        if not np.isfinite(t):
            continue  # a control: no time to select by
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
    assert sorted(b.get_gid() for b in bars) == ["peptide:2-9:AKLPGSTV", "peptide:6-11:GSTVED"]
    nan_bar = next(b for b in bars if b.get_gid() == "peptide:6-11:GSTVED")
    assert np.allclose(nan_bar.get_facecolor(), FD_CMAP.get_bad())  # unknown fD is grey
    from hxex.report.hover import hover_text

    first = next(b for b in bars if b.get_gid() == "peptide:2-9:AKLPGSTV")
    assert hover_text(first).splitlines()[:3] == ["AKLPGSTV", "residues 2–9", "fD 0.30 ± 0.02"]
    assert "fD not known" in hover_text(nan_bar)
    # its data table (the window's data tab): one row per bar, what the figure draws
    from hxex.report.hover import row_of

    row = row_of(first)
    assert {k: row[k] for k in ("sample", "time", "time_label", "start", "end", "peptide")} == {
        "sample": "S", "time": 4.0, "time_label": "4 s", "start": 2, "end": 9,
        "peptide": "AKLPGSTV"}
    assert row["fD"] == pytest.approx(0.3) and np.isnan(row_of(nan_bar)["fD"])
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
    # a legend of several rows (long names, a pool's forms) stays inside the figure as drawn,
    # not only in a file cropped to its contents (owner, 2026-11-11)
    import dataclasses

    five = coverage_table(table([(f"S{i}", "AKLPGSTV", 2, 0.0, 0.0, 0.0) for i in range(5)]),
                          sequences=SEQUENCE)
    names = {f"S{i}": f"a_long_sample_name_{i} FORMNAME{i}" for i in range(5)}
    long = dataclasses.replace(five, sample=np.array([names[s] for s in five.sample]))
    fig = plot_redundancy_resolution(long, samples=list(names.values()))
    fig.draw_without_rendering()  # the layout, as saving and the window run it
    box = fig.axes[0].get_legend().get_window_extent()
    assert box.y1 <= fig.bbox.y1 + 0.5 and box.x0 >= -0.5 and box.x1 <= fig.bbox.x1 + 0.5


def test_residue_profile():
    rows = [("S", "AKLPGSTV", 2, 4.0, 0.3, 0.02), ("S", "GSTVED", 6, 4.0, 0.6, 0.04)]
    up = table(rows)
    k2 = table(rows)  # the same peptides as K=2 rank 1 and 2
    import dataclasses

    both = SpeciesUptake(**{f.name: np.concatenate([getattr(up, f.name)] + [
        getattr(k2, f.name) if f.name not in ("K", "rank", "fD") else
        {"K": np.full(2, 2), "rank": np.full(2, r), "fD": k2.fD + 0.1 * r}[f.name]
        for r in (1, 2)]) for f in dataclasses.fields(SpeciesUptake)
        if f.name not in ("nterm_subtract", "populations")}, nterm_subtract=1)
    rm = residue_map(both, sequences=SEQUENCE)
    fig = plot_residue_profile(rm, "S", 4.0, marks=(7,))
    ax = fig.axes[0]
    gids = [line.get_gid() for line in ax.lines if line.get_gid()]
    assert gids == ["fD:K1:rank1", "fD:K2:rank1", "fD:K2:rank2"]
    assert len(ax.collections) == 3  # one band per curve
    assert any(t.get_text() == "S 4 s" for t in ax.texts)
    k1 = next(line for line in ax.lines if line.get_gid() == "fD:K1:rank1")
    y = np.asarray(k1.get_ydata(), dtype=float)
    assert np.isnan(y[4]) and y[7] == pytest.approx(rm.fD[(rm.residue == 8) & (rm.K == 1)][0])
    from hxex.gui.figures import figure_rows

    columns, lines = figure_rows(fig)  # a row per line and residue, named by its description
    assert columns[:7] == ["sample", "time", "time_label", "description", "K", "rank", "residue"]
    assert {r["description"] for r in lines} == {"K=1", "K=2 low-D", "K=2 high-D"}
    no_band = plot_residue_profile(rm, "S", 4.0, band=0, show=((1, 1),))
    assert len(no_band.axes[0].collections) == 0
    with pytest.raises(ValueError, match="no residue values"):
        plot_residue_profile(rm, "S", 60.0)
    with pytest.raises(ValueError, match="band"):
        plot_residue_profile(rm, "S", 4.0, band=1.0)


# --- the choice between one and two populations -------------------------------------------------

def test_lever_population_and_its_error():
    from hxex.analysis import lever_population

    p, se = lever_population(0.4, 0.2, 0.6, 0.01, 0.02, 0.03)
    assert p == pytest.approx(0.5)  # (0.6 - 0.4) / (0.6 - 0.2)
    eps = 1e-7  # the error from numerical derivatives
    grads = []
    for i in range(3):
        x = [0.4, 0.2, 0.6]
        x[i] += eps
        grads.append((lever_population(*x, 0, 0, 0)[0] - p) / eps)
    assert se == pytest.approx(np.sqrt(sum((g * e) ** 2 for g, e in zip(
        grads, (0.01, 0.02, 0.03), strict=True))), rel=1e-5)
    assert np.isnan(lever_population(0.4, 0.6, 0.6, 0, 0, 0)[0])  # no split


def choice_map(rows):
    """A residue map from (residue, K, rank, fD, fD_se) for one sample and time."""
    from hxex.analysis import ResidueMap

    n = len(rows)
    cols = {"sample": np.array(["S"] * n), "time": np.full(n, 4.0),
            "K": np.array([r[1] for r in rows]), "rank": np.array([r[2] for r in rows]),
            "residue": np.array([r[0] for r in rows]), "aa": np.array(["A"] * n),
            "fD": np.array([r[3] for r in rows], dtype=float),
            "fD_se": np.array([r[4] for r in rows], dtype=float),
            "redundancy": np.ones(n, dtype=np.int64), "resolution": np.ones(n),
            "proline": np.zeros(n, dtype=bool), "covered": np.ones(n, dtype=bool)}
    return ResidueMap(**cols)


def test_controls_take_no_part_in_the_residue_choice_and_comparisons():
    """Audit 2026-10-28 Medium 10: a control's NaN time split the choice's and the regions'
    rows; the controls' rows now take no part (they stay in the residue table)."""
    import dataclasses

    from hxex.analysis import (
        Region,
        ResidueMap,
        choose_populations,
        compare_residues,
        region_summary,
    )

    rows = [(r, k, rank, fd, 0.01) for r in (1, 2)
            for k, rank, fd in ((1, 1, 0.5), (2, 1, 0.2), (2, 2, 0.8))]
    point = choice_map(rows)
    control = dataclasses.replace(point, time=np.full(len(rows), np.nan),
                                  role=np.full(len(rows), "undeut"))
    both = ResidueMap(**{f.name: np.concatenate([getattr(point, f.name),
                                                 getattr(control, f.name)])
                         for f in dataclasses.fields(ResidueMap)})
    choice = choose_populations(both)
    assert len(choice) == 2 and (choice.time == 4.0).all() and choice.two_populations.all()
    regions = region_summary(both, [Region("all", 1, 2)])
    assert len(regions) == 1 and regions.time[0] == 4.0 and regions.n_residues[0] == 2
    other = dataclasses.replace(both, sample=np.full(len(both), "V"),
                                fD=both.fD + 0.1)
    two = ResidueMap(**{f.name: np.concatenate([getattr(both, f.name), getattr(other, f.name)])
                        for f in dataclasses.fields(ResidueMap)})
    rc = compare_residues(two, "S")
    assert len(rc) == 2 and (rc.time == 4.0).all()
    assert len(both.time_points()) == len(rows)


def test_choose_populations_applies_every_test():
    from hxex.analysis import ChoiceConfig, choose_populations

    cases = {  # residue: (avg, low, high, se) -> expected reason ('' = two populations)
        1: ((0.40, 0.20, 0.60, 0.02), ""),
        2: ((0.40, 0.38, 0.42, 0.05), "not separated"),
        3: ((0.40, 0.10, 0.70, 0.30), "error"),
        4: ((0.65, 0.20, 0.60, 0.02), "K=1 outside"),
        5: ((0.59, 0.20, 0.60, 0.01), "population"),  # 2.5 % low-D
    }
    rows = []
    for r, ((avg, low, high, se), _) in cases.items():
        rows += [(r, 1, 1, avg, se), (r, 2, 1, low, se), (r, 2, 2, high, se)]
    rows.append((6, 1, 1, 0.3, 0.02))  # no K=2 at all
    ch = choose_populations(choice_map(rows))
    got = dict(zip(ch.residue.tolist(), ch.reason.tolist(), strict=True))
    for r, (_, why) in cases.items():
        assert got[r] == why, r
    assert got[6] == "no K=2"
    assert list(ch.two_populations) == [True, False, False, False, False, False]
    assert ch.population_low[0] == pytest.approx(0.5) and np.isnan(ch.population_low[3])
    loose = choose_populations(choice_map(rows), ChoiceConfig(confidence=0, max_fD_se=1,
                                                              min_population=0.01))
    assert dict(zip(loose.residue.tolist(), loose.two_populations.tolist(), strict=True)) == {
        1: True, 2: True, 3: True, 4: False, 5: True, 6: False}
    with pytest.raises(ValueError):
        ChoiceConfig(confidence=1.0)


def test_population_choice_figure():
    from hxex.analysis import choose_populations
    from hxex.report import plot_population_choice

    rows = []
    for r in range(1, 11):  # two populations at 1-5, one at 6-10
        two = r <= 5
        rows += [(r, 1, 1, 0.4, 0.02), (r, 2, 1, 0.2 if two else 0.39, 0.02),
                 (r, 2, 2, 0.6 if two else 0.41, 0.02)]
    ch = choose_populations(choice_map(rows))
    fig = plot_population_choice(ch, "S", 4.0, marks=(5.5,))
    top, bottom = fig.axes[:2]
    gids = {c.get_gid(): c for c in top.collections if c.get_gid()}
    assert len(gids["chosen:low"].get_offsets()) == 5 and len(gids["chosen:avg"].get_offsets()) == 5
    pops = [c for c in bottom.collections if c.get_gid() == "population:low"][0].get_offsets()
    np.testing.assert_allclose(np.asarray(pops)[:, 1], 0.5)
    solid = next(line for line in top.lines if line.get_gid() == "fD:low")
    assert np.isnan(np.asarray(solid.get_ydata(), dtype=float)[7])  # dashed where K=1 chosen
    with pytest.raises(ValueError, match="no choice"):
        plot_population_choice(ch, "S", 60.0)


# --- comparisons within a run -------------------------------------------------------------------

def test_pooled_sequences_are_compared_head_to_head(results):
    """compare_sequences: each panel's sequences against its first, per sample and time, by
    the peptide comparisons' rule (Welch's p below alpha and the difference beyond
    delta_crit)."""
    import dataclasses

    from hxex.analysis.compare import compare_sequences
    from hxex.analysis.pooled import HeadToHead

    binned = results.binned
    # the run's two series as two sequences over residues 1-10, one with I at 10
    other = dataclasses.replace(binned, peptide=np.where(binned.data_id == 1, "RDKVQKEYAI",
                                                         "RDKVQKEYAL"))
    panel = HeadToHead(1, 10, "A", ("RDKVQKEYAL", "RDKVQKEYAI"), ("A", "A2"), ("A", "A2: L10I"))
    sc = compare_sequences(dataclasses.replace(results, binned=other), [panel])
    cols = sc.columns()
    times = {float(t) for t in binned.time if np.isfinite(t)}
    assert len(sc) == len(times) and set(cols["time"]) == times
    assert set(cols["peptide_reference"]) == {"RDKVQKEYAL"} and set(cols["difference"]) == {
        "L10I"}
    np.testing.assert_allclose(cols["delta"], cols["value"] - cols["value_reference"])
    assert np.isfinite(sc.delta_crit) and sc.delta_crit > 0
    assert np.all(~cols["significant"] | ((cols["p"] < sc.alpha)
                                          & (np.abs(cols["delta"]) > sc.delta_crit)))
    assert compare_sequences(results, []).columns()["sample"].size == 0


def test_annotations_round_trip_and_marks():
    from hxex.analysis import Annotations, Region, Site

    ann = Annotations(regions=(Region("A", 1, 10), Region("B", 11, 20)), sites=(Site(5, "S5"),))
    assert Annotations.from_dict(ann.to_dict()) == ann
    assert ann.marks() == [0.5, 5.0, 10.5, 20.5]
    with pytest.raises(ValueError):
        Region("bad", 5, 4)
    with pytest.raises(ValueError, match="unknown"):
        Annotations.from_dict({"domains": []})


def test_compare_peptides_with_the_hybrid_rule(results):
    # the synthetic run has one sample; make a second by renaming one series
    import dataclasses

    from hxex.analysis import compare_peptides

    binned = results.binned
    shifted = dataclasses.replace(binned, sample=np.where(binned.data_id == 1, "V", "S"))
    pc = compare_peptides(dataclasses.replace(results, binned=shifted), "S")
    assert set(pc.sample.tolist()) == {"V"} and len(pc) == len(set(pc.time.tolist()))
    np.testing.assert_allclose(pc.delta, pc.value - pc.value_reference)
    assert np.isfinite(pc.delta_crit) and pc.delta_crit > 0
    # significant only where both the test and the threshold agree
    assert np.all(~pc.significant | ((pc.p < pc.alpha) & (np.abs(pc.delta) > pc.delta_crit)))
    with pytest.raises(ValueError, match="no reference"):
        compare_peptides(results, "X")
    # the population of a K=2 rank, compared the same way
    pop = compare_peptides(dataclasses.replace(results, binned=shifted), "S", K=2, rank=1,
                           quantity="population")
    assert pop.quantity == "population" and pop.K == 2
    compared = pop.reason == ""  # the others say why (here the K=1 value outside the K=2 pair)
    assert np.all((pop.value[compared] >= 0.05) & (pop.value[compared] <= 0.95))
    assert np.isnan(pop.value[~compared]).all() and not pop.significant[~compared].any()
    assert all(r.endswith(("(reference)", "(sample)", "(both)")) for r in pop.reason[~compared])
    with pytest.raises(ValueError, match="K = 2"):
        compare_peptides(results, "S", quantity="population")
    with pytest.raises(ValueError, match="quantity"):
        compare_peptides(results, "S", quantity="mean")
    # a mutated peptide over the same residues: compared by range (the default), not by sequence
    seq = str(binned.peptide[0])
    mutant = seq[:3] + ("D" if seq[3] != "D" else "E") + seq[4:]
    mutated = dataclasses.replace(shifted, peptide=np.where(binned.data_id == 1, mutant, seq))
    by_range = compare_peptides(dataclasses.replace(results, binned=mutated), "S")
    assert len(by_range) == len(pc) and not by_range.same_sequence.any()
    assert set(by_range.peptide.tolist()) == {mutant}
    assert set(by_range.peptide_reference.tolist()) == {seq}
    np.testing.assert_allclose(by_range.delta, pc.delta)
    assert len(compare_peptides(dataclasses.replace(results, binned=mutated), "S",
                                match="sequence")) == 0
    assert pc.same_sequence.all()
    with pytest.raises(ValueError, match="match"):
        compare_peptides(results, "S", match="position")


def test_residue_comparison_and_regions():
    from hxex.analysis import Region, compare_residues, region_summary

    up = table([("R", "AKLPGSTV", 2, 4.0, 0.30, 0.01), ("V", "AKLPGSTV", 2, 4.0, 0.50, 0.01),
                ("W", "AKLPGSTV", 2, 4.0, 0.31, 0.01)])
    rm = residue_map(up, sequences=SEQUENCE)
    rc = compare_residues(rm, "R", delta_crit=0.1)
    v = rc.where(sample="V") & (rc.residue == 6)
    assert rc.delta[v][0] == pytest.approx(0.2) and rc.significant[v][0]
    assert not rc.significant[rc.where(sample="W")].any()  # 0.01 apart: within delta_crit
    regions = region_summary(rm, [Region("N", 1, 5), Region("M", 6, 9), Region("C", 15, 19)])
    m = (regions.sample == "V") & (regions.region == "M")
    assert regions.fD[m][0] == pytest.approx(0.5) and regions.n_residues[m][0] == 4
    assert regions.coverage[m][0] == pytest.approx(1.0)
    n_row = (regions.sample == "V") & (regions.region == "N")  # residue 5 is a proline
    assert regions.n_residues[n_row][0] == 2 and regions.coverage[n_row][0] == pytest.approx(0.4)
    assert np.isnan(regions.fD[(regions.sample == "V") & (regions.region == "C")][0])


def test_a_residue_difference_is_judged_from_the_peptides_both_samples_measured():
    """A peptide only one sample has (left out in the other, or not measured there) changes
    that sample's own residue values, which the comparison keeps, but cannot make a difference
    significant: the test uses the peptides both measured (owner, HISTORY 202). A residue
    only such peptides cover keeps its values and is not tested."""
    from hxex.analysis import (
        compare_residues,
        concat_tables,
        judged_on_shared,
        paired_residue_maps,
    )

    up = table([("R", "AKLPGSTV", 2, 4.0, 0.30, 0.01), ("V", "AKLPGSTV", 2, 4.0, 0.30, 0.01),
                ("V", "GSTV", 6, 4.0, 0.90, 0.01), ("R", "TVEDRW", 8, 4.0, 0.50, 0.01),
                ("V", "EDRWQ", 10, 4.0, 0.50, 0.01)])
    own = compare_residues(residue_map(up, sequences=SEQUENCE), "R", delta_crit=0.1)
    assert own.significant[own.residue == 8].all()  # V's own map averages GSTV in
    pairs = paired_residue_maps(up, "R", sequences=SEQUENCE)
    assert set(pairs) == {("", "V")}
    both = pairs[("", "V")]
    assert set(both.sample) == {"R", "V"}
    shared = concat_tables([compare_residues(m, "R", delta_crit=0.1) for m in pairs.values()])
    assert not shared.significant.any() and np.allclose(shared.delta, 0.0)
    rc = judged_on_shared(own, shared)
    assert np.array_equal(rc.value, own.value) and np.array_equal(rc.delta, own.delta)
    eight = rc.residue == 8
    assert rc.delta[eight][0] > 0.1 and not rc.significant[eight][0]
    assert rc.delta_shared[eight][0] == pytest.approx(0.0)
    # residue 11: each sample has a value, from peptides the other did not measure
    eleven = rc.residue == 11
    assert eleven.any() and not rc.significant[eleven].any()
    assert np.isnan(rc.delta_shared[eleven]).all() and np.isfinite(rc.delta[eleven]).all()
    assert {"delta_shared", "delta_shared_se"} <= set(rc.columns())
    assert "delta_shared" not in own.columns()
    with pytest.raises(ValueError, match="match must be"):
        paired_residue_maps(up, "R", match="nearest")


def test_a_modified_form_is_paired_with_the_reference_as_a_state_of_its_own():
    import dataclasses

    from hxex.analysis import paired_residue_maps

    up = table([("R", "AKLPGSTV", 2, 4.0, 0.30, 0.01), ("R", "AKLPGpSTV", 2, 4.0, 0.40, 0.01),
                ("V", "AKLPGSTV", 2, 4.0, 0.35, 0.01)])
    up = dataclasses.replace(up, end=np.array([9, 9, 9]))  # the helper counts characters
    pairs = paired_residue_maps(up, "R", sequences=SEQUENCE)
    assert set(pairs) == {("", "V"), ("", "R pS7")}
    assert set(pairs[("", "R pS7")].sample) == {"R", "R pS7"}
    assert "R pS7" not in set(pairs[("", "V")].sample)
    # matched by sequence, the modified form shares no peptide with the reference: it is left
    # out of the pairs (its residue differences are not judged) instead of an empty map
    by_sequence = paired_residue_maps(up, "R", sequences=SEQUENCE, match="sequence")
    assert set(by_sequence) == {("", "V")}


def test_peptide_match_auto_compares_pooled_data_by_sequence():
    """[analysis] peptide_match = "auto" (the default): by sequence for pooled data, where two
    sequences at one range are different proteins, else by range; a setting given is kept."""
    from types import SimpleNamespace

    from hxex.analysis import AnalysisConfig
    from hxex.analysis.settings import resolve_match

    def run(reader):
        return SimpleNamespace(settings=SimpleNamespace(spectra_source={"reader": reader}))

    assert AnalysisConfig().peptide_match == "auto"
    assert resolve_match("auto", {"": run("pooled")}) == "sequence"
    assert resolve_match("auto", {"": run("specexport")}) == "range"
    assert resolve_match("auto", {"a": run("specexport"), "b": run("pooled")}) == "sequence"
    assert resolve_match("range", {"": run("pooled")}) == "range"
    with pytest.raises(ValueError, match="peptide_match must be one of auto, range, sequence"):
        AnalysisConfig(peptide_match="nearest")


def test_pooled_analysis_tables_carry_each_peptide_identity(tmp_path):
    from types import SimpleNamespace

    from hxex.chem import DEFAULT_MODIFICATIONS
    from hxex.io.pooled import candidates_record
    from hxex.report.analysis import _with_pooled_identities

    (tmp_path / "pool.fasta").write_text(">A1\nMKVLGGG\n>A2\nMKVLSS\n")
    record = candidates_record(tmp_path / "pool.fasta", ["A"])
    runs = {"": SimpleNamespace(settings=SimpleNamespace(
        spectra_source={"reader": "pooled", "pooled_candidates": record},
        registry=DEFAULT_MODIFICATIONS))}
    tables = {"species_uptake": {"run": np.array(["", ""]), "peptide": np.array(["MKVL", "GGG"]),
                                 "start": np.array([1, 5]), "end": np.array([4, 7]),
                                 "fD": np.array([0.3, 0.4])},
              "coverage": {"residue": np.array([1, 2])}}
    out = _with_pooled_identities(tables, runs)
    assert list(out["species_uptake"]) == ["run", "peptide", "start", "end", "identity",
                                           "identity_level", "candidates", "classes",
                                           "position_offset", "fD"]
    assert list(out["species_uptake"]["identity"]) == ["A", "A1"]
    assert list(out["species_uptake"]["identity_level"]) == ["class", "isoform"]
    assert out["coverage"] is tables["coverage"]
    plain = {"": SimpleNamespace(settings=SimpleNamespace(spectra_source={"reader": "pooled"}))}
    assert _with_pooled_identities(tables, plain) is tables


def test_comparison_plots():
    from hxex.analysis import (
        Annotations,
        PeptideComparison,
        Region,
        Site,
        compare_residues,
        region_summary,
    )
    from hxex.report import (
        plot_peptide_differences,
        plot_region_bars,
        residue_comparison_figure,
    )

    up = table([("R", "AKLPGSTV", 2, 4.0, 0.30, 0.01), ("V", "AKLPGSTV", 2, 4.0, 0.50, 0.01)])
    rm = residue_map(up, sequences=SEQUENCE)
    rc = compare_residues(rm, "R", delta_crit=0.1)
    ann = Annotations(regions=(Region("long region name", 1, 9),), sites=(Site(5, "P5"),))
    fig = residue_comparison_figure(rc, ["V"], 4.0, annotations=ann)
    with pytest.raises(ValueError, match="no sample to compare"):
        residue_comparison_figure(rc, [rc.reference], 4.0)
    assert rc.significant[rc.residue == 6].all()  # significance is in the table, not drawn
    assert not any((p.get_gid() or "").startswith("significant:") for p in fig.axes[0].patches)
    pc = PeptideComparison(
        sample=np.array(["V", "V"]), peptide=np.array(["AKLPGSTV", "GSTVED"]),
        modification=np.array(["", ""]), start=np.array([2, 6]), end=np.array([9, 11]),
        time=np.array([4.0, 4.0]), value_reference=np.array([0.3, 0.6]),
        value_reference_se=np.array([0.01, 0.01]), value=np.array([0.5, 0.62]),
        value_se=np.array([0.01, 0.01]),
        delta=np.array([0.2, 0.02]), delta_se=np.array([0.01, 0.01]), t=np.array([5.0, 0.5]),
        p=np.array([0.001, 0.6]), n_reference=np.array([3, 3]), n=np.array([3, 3]),
        significant=np.array([True, False]), reference="R", delta_crit=0.1)
    woods = plot_peptide_differences(pc, "V", 4.0, annotations=ann)
    lines = {line.get_gid(): line for line in woods.axes[0].lines if line.get_gid()}
    assert set(lines) == {"peptide:2-9", "peptide:6-11"}
    from hxex.report.hover import hover_text

    assert hover_text(lines["peptide:2-9"]).splitlines() == [
        "AKLPGSTV", "residues 2–9", "fD, V 0.50 ± 0.01", "fD, R 0.30 ± 0.01",
        "difference +0.20 ± 0.01", "significant"]
    assert hover_text(lines["peptide:6-11"]).endswith("not significant")
    from hxex.report.hover import row_of

    row = row_of(lines["peptide:2-9"])
    assert (row["sample"], row["reference"], row["measure"], row["significant"]) == (
        "V", "R", "fD", True)
    assert row["delta"] == pytest.approx(0.2) and row["value_reference"] == pytest.approx(0.3)
    assert row_of(lines["peptide:6-11"])["significant"] is False
    bars = plot_region_bars(region_summary(rm, ann), 4.0, samples=["R", "V"])
    patches = {p.get_gid(): p for p in bars.axes[0].patches}
    assert {"region:R:long region name", "region:V:long region name"} <= set(patches)
    assert hover_text(patches["region:V:long region name"]).startswith(
        "V, region long region name\nfD ")
    with pytest.raises(ValueError, match="no comparison"):
        plot_peptide_differences(pc, "V", 60.0)
    from hxex.report import woods_figure

    stacked = woods_figure([("K=1", pc), ("again", pc)], "V", 4.0)
    assert [ax.get_title(loc="left") for ax in stacked.axes] == ["K=1", "again"]
    # the colours and the dashed lines are named, once per kind of panel
    assert [t.get_text() for t in woods.axes[0].get_legend().texts] == [
        "significantly more D than R", "significantly less D than R", "not significant",
        "critical difference (±)"]
    assert stacked.axes[0].get_legend() is not None and stacked.axes[1].get_legend() is None
    import dataclasses

    pop = dataclasses.replace(pc, quantity="population")
    mixed = woods_figure([("K=1", pc), ("pop", pop)], "V", 4.0)
    assert [t.get_text() for t in mixed.axes[1].get_legend().texts][:2] == [
        "significantly larger low-D share than R", "significantly smaller low-D share than R"]


def test_population_comparison_per_residue():
    from hxex.analysis import choose_populations, compare_populations

    rows = []
    for high in (0.6, 0.45):  # R, then V (whose average is nearer its high population)
        for r in range(1, 6):
            rows += [(r, 1, 1, 0.4, 0.01), (r, 2, 1, 0.2, 0.01), (r, 2, 2, high, 0.01)]
    m = choice_map(rows[:15])
    v = choice_map(rows[15:])
    import dataclasses

    from hxex.analysis import ResidueMap

    both = ResidueMap(**{f.name: np.concatenate([getattr(m, f.name), getattr(v, f.name)])
                         for f in dataclasses.fields(ResidueMap)})
    both = dataclasses.replace(both, sample=np.array(["R"] * 15 + ["V"] * 15))
    ch = choose_populations(both)
    cmp = compare_populations(ch, "R")
    assert cmp.quantity == "population" and len(cmp) == 5
    # R: (0.6 - 0.4) / 0.4 = 0.5 low-D; V: (0.45 - 0.4) / 0.25 = 0.2
    np.testing.assert_allclose(cmp.value_reference, 0.5)
    np.testing.assert_allclose(cmp.value, 0.2)
    assert cmp.significant.all()
    from hxex.report import residue_comparison_figure

    fig = residue_comparison_figure(cmp, ["V"], 4.0)
    assert fig.axes[0].get_ylabel() == "population (low D)"
    np.testing.assert_array_equal(cmp.two_reference, True)
    np.testing.assert_array_equal(cmp.two, True)


def test_population_comparison_per_residue_on_shared_positions():
    """Where no K=2 position differs, a residue's population difference is judged on the
    positions averaged over both samples: their errors cancel, so a clear K=1 difference with
    the same states is a population difference (HISTORY 281); where one differs, each side keeps
    its own positions."""
    import dataclasses

    from hxex.analysis import ResidueMap, choose_populations, compare_populations, compare_residues

    rows = []
    for avg in (0.4, 0.5):  # R, then V: the same states (0.2, 0.6), V more deuterated
        for r in (1, 2):
            rows += [(r, 1, 1, avg, 0.01), (r, 2, 1, 0.2, 0.05), (r, 2, 2, 0.6, 0.05)]
    m, v = choice_map(rows[:6]), choice_map(rows[6:])
    both = ResidueMap(**{f.name: np.concatenate([getattr(m, f.name), getattr(v, f.name)])
                         for f in dataclasses.fields(ResidueMap)})
    both = dataclasses.replace(both, sample=np.array(["R"] * 6 + ["V"] * 6))
    ch = choose_populations(both)
    own = compare_populations(ch, "R")
    np.testing.assert_array_equal(own.positions, ["own", "own"])
    assert not own.significant.any()  # the positions' errors hide the difference
    low = compare_residues(both, "R", K=2, rank=1, choice=ch)
    high = compare_residues(both, "R", K=2, rank=2, choice=ch)
    assert not low.significant.any() and not high.significant.any()
    shared = compare_populations(ch, "R", positions=[low, high])
    np.testing.assert_array_equal(shared.positions, ["shared", "shared"])
    np.testing.assert_allclose(shared.delta, (0.4 - 0.5) / 0.4)
    se_sep = 0.5 * np.sqrt(4 * 0.05**2)
    np.testing.assert_allclose(shared.delta_se, np.hypot(np.hypot(0.01, 0.01) / 0.4,
                                                         0.25 * se_sep / 0.4))
    assert shared.significant.all()
    np.testing.assert_allclose(shared.value, own.value)  # each sample's own values
    # a position called different at residue 2: own positions there
    moved = dataclasses.replace(low, significant=np.array([False, True]))
    mixed = compare_populations(ch, "R", positions=[moved, high])
    np.testing.assert_array_equal(mixed.positions, ["shared", "own"])
    assert mixed.delta[1] == own.delta[1] and not mixed.significant[1]
    assert "positions" in mixed.columns()
    assert "positions" not in low.columns()


def test_residue_comparison_records_each_sides_choice():
    import dataclasses

    from hxex.analysis import ResidueMap, choose_populations, compare_residues

    rows = []
    for avgs in ((0.4, 0.4, 0.4), (0.4, 0.65, 0.4)):  # R, then V (residue 2: K=1 outside)
        for r, avg in enumerate(avgs, start=1):
            rows += [(r, 1, 1, avg, 0.01), (r, 2, 1, 0.2, 0.01), (r, 2, 2, 0.6, 0.01)]
    m, v = choice_map(rows[:9]), choice_map(rows[9:])
    both = ResidueMap(**{f.name: np.concatenate([getattr(m, f.name), getattr(v, f.name)])
                         for f in dataclasses.fields(ResidueMap)})
    both = dataclasses.replace(both, sample=np.array(["R"] * 9 + ["V"] * 9))
    ch = choose_populations(both)
    rc = compare_residues(both, "R", K=2, rank=1, choice=ch)
    assert len(rc) == 3  # the residue where V keeps one population is still compared
    np.testing.assert_array_equal(rc.two_reference, [True, True, True])
    np.testing.assert_array_equal(rc.two, [True, False, True])
    np.testing.assert_array_equal(rc.choices_differ, [False, True, False])
    plain = compare_residues(both, "R", K=2, rank=1)
    assert plain.two is None and not plain.choices_differ.any()
    np.testing.assert_allclose(plain.delta, rc.delta)


# --- the whole analysis: settings, files and the command line -----------------------------------

@pytest.fixture(scope="module")
def two_sample_bundle(tmp_path_factory):
    """A bundle of two samples (the synthetic run's series 1 renamed V)."""
    import dataclasses

    raw = raw_spectra(n_series=2)
    raw = dataclasses.replace(raw, sample=np.where(raw.data_id == 1, "V", "S"))
    out = tmp_path_factory.mktemp("analysis") / "bundle"
    run(raw, config(), out=out)
    return out


def test_analysis_config_checks_and_round_trips():
    from hxex.analysis import AnalysisConfig, Annotations, Region

    c = AnalysisConfig(reference="S", times=[4], fasta={"S": "s.fasta"},
                       annotations=Annotations(regions=(Region("A", 1, 5),)))
    assert AnalysisConfig.from_dict(c.to_dict()) == c
    assert AnalysisConfig.from_dict({"filters": False}).filters is None
    for bad in ({"ks": (2,)}, {"coverage_times": "late"}, {"format": "jpg"}, {"alpha": 2},
                {"redundancy": "charges"}, {"reference": "X", "samples": ["S"]}):
        with pytest.raises(ValueError):
            AnalysisConfig(**bad)
    with pytest.raises(ValueError, match="unknown"):
        AnalysisConfig.from_dict({"colour": "red"})
    # audit 2026-11-01 Low 66 and 67: a run:sample reference with the bare name in samples;
    # K = 1 alone with the default overlay (an explicit K = 2 overlay is still refused)
    AnalysisConfig(reference="A:S", samples=["S", "V"])
    AnalysisConfig(reference="S", samples=["A:S", "V"])
    with pytest.raises(ValueError, match="not among the samples"):
        AnalysisConfig(reference="A:S", samples=["B:S"])
    assert AnalysisConfig(ks=(1,)).overlay == ("avg",)
    assert AnalysisConfig().overlay == ("low", "high", "avg")
    with pytest.raises(ValueError, match="overlay"):
        AnalysisConfig(ks=(1,), overlay=("low",))
    # per-sample settings take a run's own entries under its name (a run of runs)
    nested = AnalysisConfig(runs={"A": "a.zip", "B": "b.zip"},
                            fasta={"S": "s.fasta", "A": {"S": "s_a.fasta"}})
    assert AnalysisConfig.from_dict(nested.to_dict()) == nested
    with pytest.raises(ValueError, match="'C' is not a run"):
        AnalysisConfig(runs={"A": "a.zip"}, fasta={"S": "s.fasta", "C": {"S": "x.fasta"}})
    with pytest.raises(ValueError, match="'A' is not a run"):
        AnalysisConfig(fasta={"A": {"S": "x.fasta"}})


def test_sample_settings_resolve_by_run_then_sample():
    from hxex.analysis import sample_setting

    setting = {"S": "all", "A": {"S": "a's", "V": "a's V"}}
    assert sample_setting(setting, "A", "S") == "a's"
    assert sample_setting(setting, "B", "S") == "all"
    assert sample_setting(setting, "", "S") == "all"  # a merged sample, or one run
    assert sample_setting(setting, "B", "V") is None  # only run A has V's
    assert sample_setting(setting, "", "A") is None  # a run's table is not a sample's value
    assert sample_setting("one", "A", "S") == "one" and sample_setting(None, "A", "S") is None


def test_write_analysis_writes_tables_and_figures(two_sample_bundle, tmp_path):
    import dataclasses

    from hxex.analysis import AnalysisConfig, Annotations, Region, Site
    from hxex.io import open_bundle
    from hxex.report import write_analysis

    seen = []
    cfg = AnalysisConfig(residue_level=True, reference="S", format="png",
                         annotations=Annotations(regions=(Region("all", 1, 10),),
                                                 sites=(Site(5, "x"),)))
    with open_bundle(two_sample_bundle) as b:
        written = write_analysis(Results.from_bundle(b), cfg, tmp_path / "a",
                                 progress=lambda d, t: seen.append((d, t)))
    names = {p.name for p in written}
    for table in ("species_uptake", "residues", "coverage", "populations", "compare_peptides",
                  "compare_residues", "compare_populations", "compare_population_counts",
                  "regions"):
        assert f"{table}.csv" in names
    assert "coverage_S.png" in names and "redundancy_resolution.png" in names
    # the Woods plot in two figures: K=1 with the number of populations, and the K=2 page
    assert any(n.startswith("woods_V_") for n in names)
    assert any(n.startswith("woods_K2_V_") for n in names)
    header = (tmp_path / "a" / "compare_residues.csv").read_text().splitlines()[0].split(",")
    assert {"two_reference", "two"} <= set(header)
    assert any(n.startswith("choice_V_") for n in names)
    assert any(n.startswith("overlay_residues_") for n in names)
    assert not any(n.startswith("compare_residues_") and n.endswith(".png") for n in names)
    assert seen[-1][0] == seen[-1][1] == len(written)
    assert all(p.stat().st_size > 0 for p in written)
    with open_bundle(two_sample_bundle) as b, pytest.raises(ValueError, match="no reference"):
        write_analysis(Results.from_bundle(b), AnalysisConfig(reference="X"), tmp_path / "b")
    # the bootstrap rule: no best_set_K gate, so no number-of-populations table
    boot = dataclasses.replace(cfg, populations="bootstrap")
    with open_bundle(two_sample_bundle) as b:
        names = {p.name for p in write_analysis(Results.from_bundle(b), boot, tmp_path / "c")}
    assert "compare_population_counts.csv" not in names
    assert any(n.startswith("woods_K2_V_") for n in names)
    with pytest.raises(ValueError, match="populations"):
        AnalysisConfig(residue_level=True, populations="loose")



def _review_k1(results, sample):
    """``results`` with every spectrum of ``sample`` reviewed to one population."""
    rows = np.flatnonzero(np.asarray(results.binned.sample) == sample)
    return _review_k1_ids(results, [str(results.binned.spectrum_id[j]) for j in rows])


def _review_k1_ids(results, ids):
    """``results`` with the spectra ``ids`` reviewed to one population."""
    n = len(ids)
    return results.with_review(ReviewTable(
        spectrum_id=np.array(ids, dtype=object),
        status=np.full(n, "unreviewed", dtype=object), k_override=np.ones(n, dtype=np.int64),
        note=np.full(n, "", dtype=object), updated=np.full(n, "", dtype=object)))


def test_a_sample_without_k2_or_time_points_does_not_stop_the_analysis(two_sample_bundle,
                                                                        tmp_path, caplog):
    """Audit 2026-11-01 High 5 and Part 2 B3: a sample reviewed to one population (either
    side) keeps rows that say so; a sample with controls only is left out with a warning; a
    reference without time points, or one that names no sample, is refused before the
    filters run."""
    import dataclasses
    import logging

    from hxex.analysis import AnalysisConfig, compare_peptides
    from hxex.io import open_bundle
    from hxex.report import write_analysis

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    for sample, side in (("V", "sample"), ("S", "reference")):
        reviewed = _review_k1(results, sample)
        c = compare_peptides(reviewed, "S", K=2, rank=1)
        assert len(c) and set(c.reason) == {f"no K=2 ({side})"}
        assert not np.isfinite(c.delta).any()
        assert np.isfinite(c.value_reference if side == "sample" else c.value).all()
        pop = compare_peptides(reviewed, "S", K=2, quantity="population")
        assert set(pop.reason) == {f"no K=2 ({side})"}
        names = {p.name for p in write_analysis(reviewed, AnalysisConfig(reference="S"),
                                                tmp_path / f"k1_{sample}")}
        assert "compare_peptides.csv" in names
    # V with controls only: its labelled spectra left out of the run's input
    raw = raw_spectra(n_series=2)
    raw = dataclasses.replace(raw, sample=np.where(raw.data_id == 1, "V", "S"))
    raw = raw.select(~((raw.data_id == 1) & (np.asarray(raw.role) == "")))
    only = Results.from_run(run(raw, config()))
    with caplog.at_level(logging.WARNING, logger="hxex"):
        names = {p.name for p in write_analysis(only, AnalysisConfig(reference="S"),
                                                tmp_path / "controls_only")}
    assert "V has no labelled time points" in caplog.text
    assert "species_uptake.csv" in names
    with pytest.raises(ValueError, match="reference sample 'V' has no labelled time point"):
        write_analysis(only, AnalysisConfig(reference="V"), tmp_path / "ref_controls_only")
    # a reference naming no sample: refused before the filters say what they left out
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hxex"), pytest.raises(
            ValueError, match="no reference sample 'Mut'"):
        write_analysis(results, AnalysisConfig(reference="Mut"), tmp_path / "typo")
    assert "left out" not in caplog.text


def test_a_set_of_three_populations_is_compared_at_k2(two_sample_bundle):
    """Audit 2026-11-01 Analysis 33 (owner, 2026-11-02): a set chosen at K = 3 counts as two
    or more: compared at its K = 2 description with a reason saying so, and its peptides vote
    for two populations at the residues."""
    import dataclasses

    from hxex.analysis import best_sets, compare_peptides
    from hxex.io import open_bundle

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    two = {key: (2, v[1]) for key, v in best_sets(species_uptake(results)).items()}
    three = {key: (3 if key[1] == "V" else 2, v[1]) for key, v in two.items()}
    kw = dict(K=2, rank=1, filters=SpeciesFilters(min_population=0.0))
    plain = compare_peptides(results, "S", best_sets=two, **kw)
    more = compare_peptides(results, "S", best_sets=three, **kw)
    assert set(plain.reason) == {""}
    assert set(more.reason) == {"three populations (sample): K = 2 compared"}
    np.testing.assert_allclose(more.delta, plain.delta)
    assert np.isfinite(more.delta).all()
    both = compare_peptides(results, "S", best_sets={k: (3, v[1]) for k, v in two.items()},
                            **kw)
    assert set(both.reason) == {"three populations (both): K = 2 compared"}
    up = species_uptake(results, filters=SpeciesFilters(min_population=0.0))
    rm = residue_map(dataclasses.replace(up, best_set_K=np.full(len(up), 3)),
                     two_population_peptides=True)
    weight = np.asarray(rm.two_population_weight, float)
    assert np.nanmin(weight) == 1.0


def test_coverage_falls_back_to_every_time_when_none_was_measured(two_sample_bundle, tmp_path,
                                                                  caplog):
    """coverage_times naming times without spectra (a project without time-0 spectra, say) does
    not stop the analysis: every time is used, with a warning."""
    import logging

    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import write_analysis

    cfg = AnalysisConfig(residue_level=True, coverage_times=(999.0,), format="png")
    with open_bundle(two_sample_bundle) as b, caplog.at_level(logging.WARNING):
        written = write_analysis(Results.from_bundle(b), cfg, tmp_path / "c")
    assert "coverage.csv" in {p.name for p in written}
    assert any('coverage_times = "all" is used instead' in r.message for r in caplog.records)


def test_times_left_out_of_the_analysis(two_sample_bundle, tmp_path):
    """exclude_times: the measurements at those times leave every table; controls refused."""
    import dataclasses

    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import write_analysis

    with open_bundle(two_sample_bundle) as b:
        res = Results.from_bundle(b)
        up = species_uptake(res, ks=(1,))
        times = sorted({float(t) for t in up.time if np.isfinite(t)})
        late = times[-1]
        dropped = res.without_times([late])
        assert dropped.excluded.sum() == np.isclose(res.fit.time, late).sum() > 0
        assert late not in species_uptake(dropped, ks=(1,)).time.tolist()
        cfg = AnalysisConfig(reference="S", ks=(1,), overlay=("avg",), exclude_times=[late])
        # the synthetic run has one time point: without it the controls alone (not time
        # points) leave nothing to compare
        with pytest.raises(ValueError, match="reference sample 'S' has no labelled time point"):
            write_analysis(res, cfg, tmp_path / "a")
        assert all(float(t) != late for t in species_uptake(dropped, ks=(1,)).time)
        # the controls have no time, so no time names them; a time point that also serves as
        # a control (a role and a time) is refused
        fit = res.fit
        role = np.array(fit.role, dtype="<U8")
        role[np.isclose(fit.time, late)] = "fulldeut"
        promoted = dataclasses.replace(res, fit=dataclasses.replace(fit, role=role))
        with pytest.raises(ValueError, match="a control's time"):
            write_analysis(promoted, AnalysisConfig(exclude_times=[late]), tmp_path / "b")
    assert AnalysisConfig.from_dict(cfg.to_dict()).exclude_times == (late,)


def test_a_project_file_for_analysis_only(two_sample_bundle, tmp_path, capsys):
    """Comparing bundles needs no [input]: hxex analyse takes a file with only [analysis],
    and the commands that read input refuse it with a pointer to analyse."""
    from hxex import cli

    project = tmp_path / "compare.toml"
    project.write_text(f"""project = "compare"
[analysis]
format = "png"
ks = [1]
overlay = ["avg"]
out = "out"
[analysis.runs]
A = "{two_sample_bundle.as_posix()}"
B = "{two_sample_bundle.as_posix()}"
[analysis.merge]
"A:S" = "S"
"B:S" = "S"
""")
    assert cli.main(["-q", "analyse", str(project)]) == 0
    assert (tmp_path / "out" / "species_uptake.csv").is_file()
    assert cli.main(["-q", "run", str(project)]) == cli.EXIT_ERROR
    assert cli.main(["check", str(project)]) == 0  # the summary says there is no input
    assert "(none: analysis only)" in capsys.readouterr().out
    from hxex.config import ProjectError, load_project

    with pytest.raises(ProjectError, match="only an \\[analysis\\] table is for hxex analyse"):
        load_project(project)
    assert load_project(project, analysis_only=True).input is None

def test_analyse_command(two_sample_bundle, tmp_path, capsys):
    from hxex import cli

    out = tmp_path / "plain"
    assert cli.main(["-q", "analyse", str(two_sample_bundle), "--out", str(out),
                     "--format", "svg"]) == 0
    assert (out / "species_uptake.csv").is_file() and not (out / "compare_peptides.csv").exists()
    # one peptide per sample and no reference: nothing on residues, nothing compared; only the
    # coverage maps, which place the peptides whatever residue_level says (owner, 2026-11-09)
    assert {p.name for p in out.iterdir() if p.suffix == ".svg"} == {"coverage_S.svg",
                                                                     "coverage_V.svg"}
    assert "table(s)" in capsys.readouterr().out
    project = tmp_path / "hxex.toml"
    project.write_text(f"""project = "p"
[input]
format = "specexport"
data_dir = "d"
[output]
bundle = "{two_sample_bundle.as_posix()}"
[analysis]
reference = "S"
residue_level = true
format = "png"
out = "from_project"
[[analysis.regions]]
name = "all"
start = 1
end = 10
""")
    assert cli.main(["-q", "analyse", str(project)]) == 0
    assert (tmp_path / "from_project" / "regions.csv").is_file()
    assert (tmp_path / "from_project" / "compare_peptides.csv").is_file()
    project.write_text(project.read_text() + "[analysis.extra]\n")
    assert cli.main(["-q", "analyse", str(project)]) == 1  # an unknown key is an error


# --- separate runs ------------------------------------------------------------------------------

def _with_settings(results, **changes):
    """``results`` whose recorded run settings have ``changes`` (dotted names)."""
    import copy
    import dataclasses

    configs = copy.deepcopy(results.settings.configs)
    for name, value in changes.items():
        *path, last = name.split(".")
        node = configs["run"]
        for part in path:
            node = node[part]
        node[last] = value
    return dataclasses.replace(results, settings=dataclasses.replace(results.settings,
                                                                     configs=configs))


def test_runs_are_analysed_like_samples(results):
    from hxex.analysis import (
        RunSet,
        check_identity,
        compare_peptides,
        compare_settings,
        display_labels,
        resolve_sample,
    )

    rs = RunSet({"A": results, "B": results})
    up = species_uptake(rs)
    # the run is a column: the sample keeps its name, and run:sample is a display label
    assert set(up.sample.tolist()) == {"S"} and set(up.run.tolist()) == {"A", "B"}
    assert list(up.columns())[:2] == ["run", "sample"]
    assert display_labels(up.run, up.sample) == {("A", "S"): "A:S", ("B", "S"): "B:S"}
    assert display_labels(["", ""], ["S", "V"]) == {("", "S"): "S", ("", "V"): "V"}
    assert display_labels(["A", "B"], ["S", "V"]) == {("A", "S"): "S", ("B", "V"): "V"}
    assert resolve_sample(up.run, up.sample, "B:S") == ("B", "S")
    assert resolve_sample(["A", "B"], ["S", "V"], "V") == ("B", "V")
    a, b = up.select(up.where(run="A")), up.select(up.where(run="B"))
    np.testing.assert_array_equal(a.fD, b.fD)
    np.testing.assert_array_equal(a.fD_se, b.fD_se)
    pc = compare_peptides(rs, "A:S")
    assert set(pc.sample.tolist()) == {"S"} and set(pc.run.tolist()) == {"B"}
    assert (pc.reference, pc.reference_run) == ("S", "A")
    assert len(pc) == len({float(t) for t in a.time if np.isfinite(t)})  # the time points
    np.testing.assert_allclose(pc.delta, 0.0)
    assert not pc.significant.any()
    assert len(check_identity(up)) == 0 and len(compare_settings(rs)) == 0
    with pytest.raises(ValueError, match="several runs: name one of A:S, B:S"):
        compare_peptides(rs, "S")
    with pytest.raises(ValueError, match="samples: A:S, B:S"):
        compare_peptides(rs, "X")
    for bad in ({}, {"a:b": results}, {"": results}):
        with pytest.raises(ValueError):
            RunSet(bad)


def test_settings_that_matter_are_reported(results, caplog):
    from hxex.analysis import RunSet, compare_settings

    other = _with_settings(results, **{"fit.noise.percent": 5.0, "fit.seed": 7,
                                       "max_chunk": 10})
    with caplog.at_level("WARNING", logger="hxex.analysis.runs"):
        diff = compare_settings({"A": results, "B": other})
    got = {str(s): (bool(m), list(v)) for s, m, v in zip(diff.setting, diff.matters, diff.values,
                                                          strict=True)}
    assert got == {"fit.noise.percent": (True, ["3.0", "5.0"]),
                   "fit.seed": (False, ["0", "7"])}  # max_chunk is ignored
    assert "fit.noise.percent" in caplog.text and "fit.seed" not in caplog.text
    assert list(diff.columns()) == ["setting", "matters", "A", "B"]
    shifted = _with_settings(results, **{"picker.nterm_subtract": 2})
    with pytest.raises(ValueError, match="N-terminal"):
        species_uptake(RunSet({"A": results, "B": shifted}))


def test_identity_is_checked_between_runs():
    import dataclasses

    from hxex.analysis import check_identity

    up = table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.01), ("S", "AKLPGSTV", 3, 4.0, 0.3, 0.01),
                ("S", "GSTVED", 6, 4.0, 0.6, 0.01), ("V", "GSTVED", 6, 4.0, 0.6, 0.01),
                ("S", "TVEDRWQ", 8, 4.0, 0.6, 0.01), ("S", "TVEDRWQ", 8, 4.0, 0.6, 0.01)])
    up = dataclasses.replace(up, n_amides=np.array([10.0, 10.0, 10.0, 10.0, 6.0, 5.0]),
                             run=np.array(["A", "B", "A", "B", "A", "B"], dtype=object))
    check = check_identity(up)
    issues = {(str(p), str(r)): (str(s), str(i)) for p, r, s, i in zip(
        check.peptide, check.run, check.start, check.issue, strict=True)}
    assert issues == {("AKLPGSTV", "A"): ("2", "position"), ("AKLPGSTV", "B"): ("3", "position"),
                      ("TVEDRWQ", "A"): ("8", "amides"), ("TVEDRWQ", "B"): ("8", "amides")}


def test_differing_population_choices_are_reported():
    import dataclasses

    from hxex.analysis import ResidueMap, choose_populations, compare_choices

    rows = [(r, *v) for r in range(1, 4) for v in ((1, 1, 0.4, 0.01), (2, 1, 0.2, 0.01),
                                                     (2, 2, 0.6, 0.01))]
    outside = [(r, *v) for r in range(1, 4) for v in ((1, 1, 0.7, 0.01), (2, 1, 0.2, 0.01),
                                                        (2, 2, 0.6, 0.01))]
    m, v = choice_map(rows), choice_map(outside)
    both = ResidueMap(**{f.name: np.concatenate([getattr(m, f.name), getattr(v, f.name)])
                         for f in dataclasses.fields(ResidueMap)})
    both = dataclasses.replace(both, sample=np.array(["A:S"] * 9 + ["B:S"] * 9))
    cc = compare_choices(choose_populations(both), "A:S")
    assert len(cc) == 3 and cc.two_reference.all() and not cc.two.any() and not cc.agree.any()
    assert set(cc.reason.tolist()) == {"K=1 outside"}
    with pytest.raises(ValueError, match="no reference"):
        compare_choices(choose_populations(both), "S")


def test_butterfly_plot():
    import dataclasses

    from hxex.analysis import PeptideComparison
    from hxex.report import plot_butterfly

    n = 4
    pc = PeptideComparison(
        sample=np.array(["B:S"] * n), peptide=np.array(["GSTVED", "AKLPGSTV"] * 2),
        modification=np.array([""] * n), start=np.array([6, 2, 6, 2]),
        end=np.array([11, 9, 11, 9]), time=np.array([4.0, 4.0, 60.0, 60.0]),
        value_reference=np.array([0.3, 0.2, 0.6, 0.5]), value_reference_se=np.full(n, 0.01),
        value=np.array([0.4, 0.2, 0.7, 0.5]), value_se=np.full(n, 0.01),
        delta=np.array([0.1, 0.0, 0.1, 0.0]), delta_se=np.full(n, 0.01), t=np.zeros(n),
        p=np.array([0.01, 0.9, 0.01, 0.9]), n_reference=np.full(n, 3), n=np.full(n, 3),
        significant=np.array([True, False, True, False]), reference="A:S", delta_crit=0.05)
    fig = plot_butterfly(pc, "B:S")
    top, bottom = fig.axes
    lines = {line.get_gid(): line for line in top.lines if line.get_gid()}
    assert set(lines) == {"butterfly:4:reference", "butterfly:4:sample",
                          "butterfly:60:reference", "butterfly:60:sample"}
    # peptides in order of position: 2-9 first
    np.testing.assert_allclose(lines["butterfly:60:reference"].get_ydata(), [0.5, 0.6])
    np.testing.assert_allclose(lines["butterfly:60:sample"].get_ydata(), [-0.5, -0.7])
    assert [t.get_text() for t in bottom.get_xticklabels()] == ["2-9", "6-11"]
    assert len(plot_butterfly(pc, "B:S", times=[60.0]).axes[0].get_legend().texts) == 1
    # the differences' filled circles and crosses and the dashed line are named
    assert [t.get_text() for t in bottom.get_legend().texts] == [
        "significant difference", "not significant", "critical difference (±)"]
    assert bottom.get_legend().legend_handles[1].get_marker() == "x"
    markers = {c.get_paths()[0].vertices.shape for c in bottom.collections if c.get_offsets().size}
    assert len(markers) == 2  # the circles and the crosses
    # the frame follows the values: a value at max_fD (1.1) does not sit on it
    top_wide = plot_butterfly(dataclasses.replace(pc, value=np.array([1.1, 0.2, 0.7, 0.5])),
                              "B:S").axes[0]
    assert top_wide.get_ylim()[1] > 1.1 and top.get_ylim()[1] == pytest.approx(1.05 * 1.05)
    with pytest.raises(ValueError, match="no comparison"):
        plot_butterfly(pc, "C:S")


def test_analyse_separate_runs(two_sample_bundle, tmp_path, capsys):
    from hxex import cli

    out = tmp_path / "runs"
    assert cli.main(["-q", "analyse", f"A={two_sample_bundle}", f"B={two_sample_bundle}",
                     "--reference", "A:S", "--out", str(out), "--format", "png"]) == 0
    names = {p.name for p in out.iterdir()}
    for table in ("run_settings", "run_identity", "compare_peptides"):
        assert f"{table}.csv" in names
    assert {"butterfly_B_S.png", "butterfly_A_V.png"} <= names
    # one peptide per sample: no residues to resolve, so no residue-level outputs by default;
    # the coverage maps are drawn all the same (owner, 2026-11-09)
    assert "residues.csv" not in names and "coverage.csv" not in names
    assert any(n.startswith("coverage_") for n in names)
    rows = (out / "compare_peptides.csv").read_text().splitlines()
    assert rows[0].startswith("comparison,run,sample,")
    assert {tuple(r.split(",")[1:3]) for r in rows[1:]} == {
        ("A", "V"), ("B", "S"), ("B", "V")}  # the compared samples
    # one bundle given a name keeps it (the single-bundle shortcut is for a bare path)
    one = tmp_path / "one"
    assert cli.main(["-q", "analyse", f"A={two_sample_bundle}", "--reference", "A:S",
                     "--out", str(one), "--format", "png"]) == 0
    rows = (one / "compare_peptides.csv").read_text().splitlines()
    assert {r.split(",")[1] for r in rows[1:]} == {"A"}
    project = tmp_path / "hxex.toml"
    project.write_text(f"""project = "p"
[input]
format = "specexport"
data_dir = "d"
[analysis]
reference = "one:S"
residue_level = true
samples = ["one:S", "two:S"]
format = "png"
out = "from_runs"
[analysis.runs]
one = "{two_sample_bundle.as_posix()}"
two = "{two_sample_bundle.as_posix()}"
""")
    assert cli.main(["-q", "analyse", str(project)]) == 0
    assert (tmp_path / "from_runs" / "butterfly_two_S.png").is_file()
    assert not (tmp_path / "from_runs" / "butterfly_one_V.png").exists()
    project.write_text(project.read_text().replace('one = "', '"o:ne" = "'))
    assert cli.main(["-q", "analyse", str(project)]) == 1
    assert "run name 'o:ne'" in capsys.readouterr().err


def test_residue_overlay_shows_the_chosen_values():
    from hxex.analysis import AnalysisConfig, ResidueMap
    from hxex.report import plot_residue_overlay, residue_overlay_figure

    rows = []  # (sample, residue, K, rank, fD)
    for sample, shift in (("WT", 0.0), ("V", 0.1)):
        for r in (3, 4, 6):
            rows += [(sample, r, 1, 1, 0.4 + shift), (sample, r, 2, 1, 0.2 + shift),
                     (sample, r, 2, 2, 0.6 + shift)]
    n = len(rows)
    rm = ResidueMap(sample=np.array([r[0] for r in rows]), time=np.full(n, 4.0),
                    K=np.array([r[2] for r in rows]), rank=np.array([r[3] for r in rows]),
                    residue=np.array([r[1] for r in rows]), aa=np.array(["A"] * n),
                    fD=np.array([r[4] for r in rows]), fD_se=np.full(n, 0.02),
                    redundancy=np.ones(n, dtype=np.int64), resolution=np.ones(n),
                    proline=np.zeros(n, dtype=bool), covered=np.ones(n, dtype=bool))
    fig = plot_residue_overlay(rm, "WT", "V", 4.0)
    lines = {line.get_gid(): line for line in fig.axes[0].lines if line.get_gid()}
    assert set(lines) == {"overlay:WT:low", "overlay:WT:high", "overlay:V:low",
                          "overlay:V:high"}
    np.testing.assert_allclose(lines["overlay:V:high"].get_ydata(), 0.7)
    assert lines["overlay:WT:low"].get_linestyle() == "None"  # points, no lines
    every = plot_residue_overlay(rm, "WT", "V", 4.0, show=("avg", "low", "high"))
    avg = next(line for line in every.axes[0].lines if line.get_gid() == "overlay:V:avg")
    assert avg.get_marker() == "o" and avg.get_markerfacecolor()[3] == pytest.approx(0.65)
    assert lines["overlay:V:low"].get_marker() == "^"
    assert lines["overlay:V:high"].get_marker() == "v"
    # the legend names the samples (their colours) and, in grey, the shapes
    assert [t.get_text() for t in every.axes[0].get_legend().texts] == [
        "WT", "V", r"fD$_{\rm avg}$", r"fD$_{\rm low}$", r"fD$_{\rm high}$"]
    assert [t.get_text() for t in fig.axes[0].get_legend().texts] == [
        "WT", "V", r"fD$_{\rm low}$", r"fD$_{\rm high}$"]
    assert every.axes[0].get_ylabel() == "fD"  # the legend names which
    one = plot_residue_overlay(rm, "WT", "V", 4.0, show=("avg",))  # circles alone: no names
    assert [t.get_text() for t in one.axes[0].get_legend().texts] == ["WT", "V"]
    assert len(residue_overlay_figure(rm, "WT", ["WT", "V"], 4.0).axes) == 1
    with pytest.raises(ValueError, match="show"):
        plot_residue_overlay(rm, "WT", "V", 4.0, show=("mean",))
    with pytest.raises(ValueError, match="no residue values"):
        plot_residue_overlay(rm, "WT", "V", 60.0)
    assert AnalysisConfig(overlay=["avg"], ks=[1]).overlay == ("avg",)
    assert AnalysisConfig().overlay_two_only  # K=2 values only where two populations are chosen
    for bad in ({"overlay": []}, {"overlay": ["mean"]}, {"ks": [1], "overlay": ["low"]}):
        with pytest.raises(ValueError, match="overlay"):
            AnalysisConfig(**bad)
    # only where two populations are chosen: V's residue 6 has its K=1 value outside its ranks
    import dataclasses

    from hxex.analysis import choose_populations

    odd = (rm.sample == "V") & (rm.residue == 6) & (rm.K == 1)
    choice = choose_populations(dataclasses.replace(rm, fD=np.where(odd, 0.75, rm.fD)))
    kept = plot_residue_overlay(rm, "WT", "V", 4.0, show=("low", "high", "avg"), choice=choice)
    lines = {line.get_gid(): line for line in kept.axes[0].lines if line.get_gid()}
    assert list(lines["overlay:V:high"].get_xdata()) == [3, 4]
    assert list(lines["overlay:WT:high"].get_xdata()) == [3, 4, 6]
    assert list(lines["overlay:V:avg"].get_xdata()) == [3, 4, 6]  # the average is never filtered


def test_residue_figures_keep_values_above_full_deuteration_in_view():
    """Audit 2026-10-28 Medium 19: fD is not clipped at 1 (max_fD allows 1.1), so the
    comparison, overlay and region figures raise their top above the highest point drawn."""
    import dataclasses

    from hxex.analysis import Region, ResidueMap, compare_residues, region_summary
    from hxex.report import plot_region_bars, plot_residue_comparison, plot_residue_overlay

    rows = [(sample, r, fd) for sample, fd in (("WT", 0.9), ("V", 1.09)) for r in (3, 4, 6)]
    n = len(rows)
    rm = ResidueMap(sample=np.array([r[0] for r in rows]), time=np.full(n, 4.0),
                    K=np.ones(n, dtype=np.int64), rank=np.ones(n, dtype=np.int64),
                    residue=np.array([r[1] for r in rows]), aa=np.array(["A"] * n),
                    fD=np.array([r[2] for r in rows]), fD_se=np.full(n, 0.02),
                    redundancy=np.ones(n, dtype=np.int64), resolution=np.ones(n),
                    proline=np.zeros(n, dtype=bool), covered=np.ones(n, dtype=bool))
    for fig in (plot_residue_overlay(rm, "WT", "V", 4.0, show=("avg",)),
                plot_residue_comparison(compare_residues(rm, "WT"), "V", 4.0),
                plot_region_bars(region_summary(rm, [Region("all", 3, 6)]), 4.0)):
        assert fig.axes[0].get_ylim()[1] >= 1.11 * 1.05 - 1e-9
    plain = dataclasses.replace(rm, fD=np.full(n, 0.5))
    assert plot_residue_overlay(plain, "WT", "V", 4.0, show=("avg",)).axes[0].get_ylim()[1] \
        == pytest.approx(1.05)


def test_range_matching_never_crosses_modified_forms():
    from hxex.analysis.compare import _pairs

    ref = {("AKLPGS", "", 2, 7, 4.0): 0, ("AKLPpS", "", 2, 7, 4.0): 0,
           ("GSTVED", "", 6, 11, 4.0): 0}
    other = {("AKLPGS", "", 2, 7, 4.0): 0, ("GSTVDD", "", 6, 11, 4.0): 0,
             ("TVEDRW", "", 8, 13, 4.0): 0}
    pairs = _pairs(ref, other, "range")
    # 2-7: two forms in the reference, so only the identical one; 6-11: one each, a mutation
    assert pairs == [(("AKLPGS", "", 2, 7, 4.0),) * 2,
                     (("GSTVED", "", 6, 11, 4.0), ("GSTVDD", "", 6, 11, 4.0))]
    assert _pairs(ref, other, "sequence") == [(("AKLPGS", "", 2, 7, 4.0),) * 2]


def test_values_corrected_differently_are_marked(results):
    import dataclasses

    from hxex.analysis import PeptideComparison, ResidueMap, compare_peptides
    from hxex.report import plot_peptide_differences, plot_residue_overlay

    # the flag follows a fallback control from the series to the species and the comparison
    binned = results.binned
    shifted = dataclasses.replace(binned, sample=np.where(binned.data_id == 1, "V", "S"))
    c = results.controls
    one = dataclasses.replace(c, fallback=np.asarray(c.data_id) == 1)
    res = dataclasses.replace(results, binned=shifted, controls=one)
    up = species_uptake(res)
    assert up.control_fallback[up.sample == "V"].all() and not up.control_fallback[
        up.sample == "S"].any()
    assert not up.fd_other_charge.any()
    # a fully deuterated control from the other charge states: still a fallback, but measured
    borrowed = dataclasses.replace(one, flags=np.where(
        np.asarray(c.data_id) == 1, "fd_missing;fd_other_charge", np.asarray(c.flags)))
    up2 = species_uptake(dataclasses.replace(res, controls=borrowed))
    assert up2.control_fallback[up2.sample == "V"].all()
    assert up2.fd_other_charge[up2.sample == "V"].all()
    assert not up2.fd_other_charge[up2.sample == "S"].any()
    pc = compare_peptides(res, "S")
    assert pc.control_fallback.all() and not pc.control_fallback_reference.any()
    assert pc.corrected_differently.all()
    # residues: a star where only one side rests on a fallback control
    n = 6
    rm = ResidueMap(sample=np.array(["WT"] * 3 + ["V"] * 3), time=np.full(n, 4.0),
                    K=np.ones(n, dtype=int), rank=np.ones(n, dtype=int),
                    residue=np.array([3, 4, 6] * 2), aa=np.array(["A"] * n),
                    fD=np.full(n, 0.4), fD_se=np.full(n, 0.02),
                    redundancy=np.ones(n, dtype=np.int64), resolution=np.ones(n),
                    proline=np.zeros(n, dtype=bool), covered=np.ones(n, dtype=bool),
                    control_fallback=np.array([False, False, True, False, True, True]))
    fig = plot_residue_overlay(rm, "WT", "V", 4.0, show=("avg",))
    stars = {t.get_gid() for t in fig.axes[0].texts if (t.get_gid() or "").startswith("corr")}
    assert stars == {"corrected_differently:4"}  # 6 falls back on both sides: the same
    assert "corrected differently" in [t.get_text() for t in fig.axes[0].get_legend().texts]
    # peptides: a star over the middle of the peptide
    woods = PeptideComparison(
        sample=np.array(["V", "V"]), peptide=np.array(["AKLPGSTV", "GSTVED"]),
        modification=np.array(["", ""]), start=np.array([2, 6]), end=np.array([9, 11]),
        time=np.array([4.0, 4.0]), value_reference=np.array([0.3, 0.6]),
        value_reference_se=np.full(2, 0.01), value=np.array([0.5, 0.62]),
        value_se=np.full(2, 0.01), delta=np.array([0.2, 0.02]), delta_se=np.full(2, 0.01),
        t=np.array([5.0, 0.5]), p=np.array([0.001, 0.6]), n_reference=np.full(2, 3),
        n=np.full(2, 3), significant=np.array([True, False]), reference="R", delta_crit=0.1,
        control_fallback=np.array([True, False]))
    ax = plot_peptide_differences(woods, "V", 4.0).axes[0]
    assert {t.get_gid() for t in ax.texts if t.get_gid()} == {"corrected_differently:5.5"}
    # the butterfly plot names it in the legend, not over the panel
    from hxex.report import plot_butterfly

    top = plot_butterfly(woods, "V").axes[0]
    assert "corrected differently" in [t.get_text() for t in top.get_legend().texts]
    assert not any("corrected differently" in t.get_text() for t in top.texts)


def test_modified_peptides_are_separate_states_or_combined():
    from hxex.analysis import AnalysisConfig
    from hxex.analysis.residues import modified_state

    up = table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.01), ("S", "AKLPGpSTV", 2, 4.0, 0.5, 0.01)])
    apart = residue_map(up, sequences=SEQUENCE)  # 'separate', the default
    assert set(apart.sample.tolist()) == {"S", "S pS7"}
    at8 = apart.residue == 8
    assert apart.fD[at8 & (apart.sample == "S")][0] == pytest.approx(0.3)
    assert apart.fD[at8 & (apart.sample == "S pS7")][0] == pytest.approx(0.5)
    together = residue_map(up, sequences={"S": SEQUENCE}, modified="combined")
    assert set(together.sample.tolist()) == {"S"}
    assert together.fD[together.residue == 8][0] == pytest.approx(0.4)
    # a state takes its sample's sequence
    assert residue_map(up, sequences={"S": SEQUENCE}).aa[apart.residue == 1][0] == "M"
    assert modified_state("acKLpS", 10) == "acK10,pS12" and modified_state("AKL", 1) == ""
    assert modified_state("AKL", 1, "Hex:1 ion_type:M") == "[Hex:1]"
    with pytest.raises(ValueError, match="modified"):
        residue_map(up, modified="apart")
    assert AnalysisConfig().modified_peptides == "separate"
    with pytest.raises(ValueError, match="modified_peptides"):
        AnalysisConfig(modified_peptides="apart")


def test_controls_follow_the_review_and_can_be_recomputed(results, caplog):
    import dataclasses

    from hxex.analysis import analysis_controls

    assert analysis_controls(results) is results.controls  # nothing rejected, no changes
    fd = np.flatnonzero(results.fit.role == "fulldeut")
    series = int(results.fit.data_id[fd[0]])
    before = results.controls.n_fd[results.controls.rows([series])[0]]
    review = ReviewTable.empty().update(str(results.fit.spectrum_id[fd[0]]), status="rejected")
    with caplog.at_level("INFO", logger="hxex.analysis.species"):
        c = analysis_controls(results.with_review(review))
    assert c is not results.controls and "1 rejected control spectra left out" in caplog.text
    assert c.n_fd[c.rows([series])[0]] == before - 1
    # changed settings: recomputed with them, the rest of the run's settings kept
    changed = analysis_controls(results, {"d2o_fraction": 0.8, "retention_min": 0.99})
    assert changed.config.d2o_fraction == 0.8 and changed.config.retention_min == 0.99
    assert changed.fallback.all() and changed.retention == pytest.approx(0.8)
    up = species_uptake(results, controls={"d2o_fraction": 0.8, "retention_min": 0.99})
    base = species_uptake(results)
    assert up.control_fallback.all() and not np.allclose(up.fD, base.fD)
    # a time point promoted by [analysis.controls] leaves the labelled points of the tables
    last = float(np.nanmax(results.fit.time))
    assert ((base.role == "") & (base.time == last)).any()
    promoted = species_uptake(results, controls={"fulldeut_at": last})
    assert not ((promoted.role == "") & (promoted.time == last)).any()
    assert (promoted.role == "fulldeut").any()
    with pytest.raises(ValueError, match="unknown control"):
        from hxex.analysis import AnalysisConfig

        AnalysisConfig(controls={"d2o": 0.8})
    assert dataclasses.replace(results, controls=None).controls is None


def test_merged_runs_pool_their_spectra(results):
    import dataclasses

    from hxex.analysis import RunSet, compare_peptides

    binned = results.binned
    two = dataclasses.replace(results, binned=dataclasses.replace(
        binned, sample=np.where(binned.data_id == 1, "V", "S")))
    single = species_uptake(two)
    merged = RunSet({"A": two, "B": two},
                    merge={"A:S": "S", "B:S": "S", "A:V": "V", "B:V": "V"})
    up = species_uptake(merged)
    assert set(up.sample.tolist()) == {"S", "V"}
    np.testing.assert_allclose(up.fD, single.fD, atol=1e-12)  # the same spectra twice
    np.testing.assert_array_equal(up.n_spectra, 2 * single.n_spectra)
    pc = compare_peptides(merged, "S")
    np.testing.assert_allclose(pc.delta, compare_peptides(two, "S").delta, atol=1e-12)
    assert (pc.n == 2 * compare_peptides(two, "S").n).all()
    partly = species_uptake(RunSet({"A": two, "B": two}, merge={"A:S": "S"}))
    assert set(zip(partly.run, partly.sample, strict=True)) == {
        ("", "S"), ("A", "V"), ("B", "S"), ("B", "V")}  # a merged sample has no run of its own
    assert set(up.run.tolist()) == {""}
    # run identity: the runs and their spectra are listed; identical runs do not differ
    assert set(up.runs.tolist()) == {"A,B"} and not up.run_offset.any()
    assert all(c == f"{n // 2},{n // 2}" for c, n in zip(up.n_spectra_by_run, up.n_spectra,
                                                         strict=True))
    assert set(partly.runs[partly.where(run="A", sample="V")].tolist()) == {"A"}
    assert set(single.runs.tolist()) == {""}
    # run B corrected with a smaller fully deuterated control: higher values throughout
    c = two.controls
    shifted_b = dataclasses.replace(two, controls=dataclasses.replace(
        c, mfd=c.m0 + 0.8 * (c.mfd - c.m0)))
    odd = species_uptake(RunSet({"A": two, "B": shifted_b},
                                merge={"A:S": "S", "B:S": "S", "A:V": "V", "B:V": "V"}))
    k1 = odd.where(K=1)
    assert odd.run_offset[k1].any()
    for bad in ({"C:S": "S"}, {"A": "S"}, {"A:S": ""}):
        with pytest.raises(ValueError, match="merge"):
            RunSet({"A": two}, merge=bad)


def test_analyse_merged_runs(two_sample_bundle, tmp_path):
    from hxex import cli

    project = tmp_path / "hxex.toml"
    project.write_text(f"""project = "p"
[input]
format = "specexport"
data_dir = "d"
[analysis]
reference = "S"
format = "png"
out = "merged"
[analysis.runs]
A = "{two_sample_bundle.as_posix()}"
B = "{two_sample_bundle.as_posix()}"
[analysis.merge]
"A:S" = "S"
"B:S" = "S"
"A:V" = "V"
"B:V" = "V"
""")
    assert cli.main(["-q", "analyse", str(project)]) == 0
    rows = (tmp_path / "merged" / "compare_peptides.csv").read_text().splitlines()[1:]
    assert {tuple(r.split(",")[1:3]) for r in rows} == {("", "V")}
    assert (tmp_path / "merged" / "run_identity.csv").is_file()


def test_a_peptide_without_a_value_or_error_leaves_the_other_residues_alone():
    from hxex.analysis import ChoiceConfig, choose_populations

    # a: residues 3, 4, 6-9 without a value; b: 7-11 with an unknown error; c: 9-14 complete
    up = table([("S", "AKLPGSTV", 2, 4.0, np.nan, 0.02), ("S", "GSTVED", 6, 4.0, 0.6, np.nan),
                ("S", "TVEDRWQ", 8, 4.0, 0.5, 0.03)])
    rm = residue_map(up, sequences=SEQUENCE)

    def at(residue, name):
        return getattr(rm, name)[rm.residue == residue][0]

    assert np.isnan(at(3, "fD")) and at(3, "redundancy") == 1 and at(3, "covered")
    assert at(7, "fD") == pytest.approx(0.6) and np.isnan(at(7, "fD_se"))
    wb, wc = (1 / 5) / (1 / 5 + 1 / 6), (1 / 6) / (1 / 5 + 1 / 6)
    assert at(10, "fD") == pytest.approx(wb * 0.6 + wc * 0.5) and np.isnan(at(10, "fD_se"))
    assert at(12, "fD") == pytest.approx(0.5) and at(12, "fD_se") == pytest.approx(0.03)
    assert np.isfinite(rm.fD[rm.covered & (rm.residue >= 12)]).all()
    # the choice between one and two populations: an unknown error passes no test
    rows = [(1, 1, 1, 0.4, 0.02), (1, 2, 1, 0.2, np.nan), (1, 2, 2, 0.6, 0.02),
            (2, 1, 1, 0.4, 0.02), (2, 2, 1, 0.2, np.nan), (2, 2, 2, 0.6, np.nan),
            (3, 1, 1, 0.4, 0.02), (3, 2, 1, 0.2, 0.02), (3, 2, 2, 0.6, 0.02)]
    ch = choose_populations(choice_map(rows), ChoiceConfig(confidence=0, max_fD_se=1))
    assert ch.two_populations.tolist() == [False, False, True]
    assert ch.reason.tolist() == ["not separated", "not separated", ""]


def test_species_filters_judge_merged_samples_once_pooled(results):
    from hxex.analysis import RunSet

    merged = RunSet({"A": results, "B": results}, merge={"A:S": "S", "B:S": "S"})
    loose = SpeciesFilters(k1_between_ranks=False, max_fD_se=1e9)

    def worst(up):  # the largest fD error of the ranks of each K=2 species and time
        out = {}
        for p, t, k, se in zip(up.peptide, up.time, up.K, up.fD_se, strict=True):
            if k == 2 and np.isfinite(se):
                out[(str(p), float(t))] = max(out.get((str(p), float(t)), 0.0), float(se))
        return out

    single, pooled = worst(species_uptake(results, filters=loose)), worst(
        species_uptake(merged, filters=loose))
    key = max(single, key=lambda k: single[k] - pooled.get(k, np.inf))
    assert pooled[key] < single[key]  # pooling the same spectra twice shrinks the error
    tight = SpeciesFilters(k1_between_ranks=False, max_fD_se=(single[key] + pooled[key]) / 2)
    s2, p2 = species_uptake(results, filters=tight), species_uptake(merged, filters=tight)
    assert not (s2.where(K=2, time=key[1]) & (s2.peptide == key[0])).any()
    assert (p2.where(K=2, time=key[1]) & (p2.peptide == key[0])).any()  # judged once, pooled


def test_identity_check_needs_run_labels_or_the_runs_column():
    import dataclasses

    from hxex.analysis import check_identity

    with pytest.raises(ValueError, match="name no run: S"):
        check_identity(table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.01)]))
    # a merged sample carries its runs in the runs column
    up = table([("S", "AKLPGSTV", 2, 4.0, 0.3, 0.01), ("S", "AKLPGSTV", 3, 4.0, 0.3, 0.01)])
    check = check_identity(dataclasses.replace(up, runs=np.array(["A", "B"])), quiet=True)
    assert check.run.tolist() == ["A", "B"] and set(check.issue) == {"position"}


def test_woods_plot_without_a_finite_difference_keeps_a_default_range():
    from hxex.analysis import PeptideComparison
    from hxex.report import plot_peptide_differences

    nan = np.array([np.nan])
    pc = PeptideComparison(
        sample=np.array(["V"]), peptide=np.array(["AKLPGSTV"]), modification=np.array([""]),
        start=np.array([2]), end=np.array([9]), time=np.array([4.0]), value_reference=nan,
        value_reference_se=nan, value=nan, value_se=nan, delta=nan, delta_se=nan, t=nan, p=nan,
        n_reference=np.array([1]), n=np.array([1]), significant=np.array([False]),
        reference="R", delta_crit=np.nan)
    fig = plot_peptide_differences(pc, "V", 4.0)
    np.testing.assert_allclose(fig.axes[0].get_ylim(), (-0.22, 0.22))


def test_times_that_only_order_the_points(two_sample_bundle, tmp_path):
    """With [input] time_scale = "order" the per-time outputs are named by the number given,
    the figures say "point", and the protection analysis refuses."""
    import dataclasses
    import re

    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import write_analysis
    from hxex.report.residues import _when

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    settings = dataclasses.replace(results.settings, configs={
        **results.settings.configs, "spectra_source": {"time_scale": "order"}})
    results = dataclasses.replace(results, settings=settings)
    assert results.time_scale == "order"
    assert _when(60.0, "order") == "point 60" and _when(60.0) == "1 min"
    written = write_analysis(results, AnalysisConfig(residue_level=True, reference="S",
                                                     format="png"), tmp_path / "order")
    names = {p.name for p in written}
    assert any(re.fullmatch(r"(choice|profile)_V_\d+\.png", n) for n in names)
    assert not any(re.search(r"_\d+(s|min|h)\.png$", n) for n in names)
    labelling = {"ph_read": 7.0, "temperature": 20.0, "d2o_fraction": 0.9}
    cfg = AnalysisConfig.from_dict({"protection": True, "labelling": labelling})
    with pytest.raises(ValueError, match="only order the points"):
        write_analysis(results, cfg, tmp_path / "protection")


def test_peptide_populations_by_the_lever_rule(monkeypatch):
    """Each replicate's K=1 value between the sample's mean K=2 positions; the threshold is the
    K=1 one divided by the separation; pairs that fail the choice's tests say why. Where no
    position differs the difference is judged on shared positions with the K=1 test's verdict,
    elsewhere on each side's own positions with their errors (HISTORY 281)."""
    from hxex.analysis import compare, compare_peptides
    from hxex.analysis.species import Point

    def records(values):  # (fD, fD_se, population, population_se, charge, n_amides, rep, ...)
        return [(v, 0.01, np.nan, np.nan, 2, 10, rep + 1, False, "") for rep, v in
                enumerate(values)]

    groups = {}

    def species(sample, peptide, avg, low, high):
        key = ("", sample, peptide, "", 1, 10, Point((60.0, "")))
        groups[(*key, 1, 1)] = records(avg)
        groups[(*key, 2, 1)] = records(low)
        groups[(*key, 2, 2)] = records(high)

    # the same states (0.2, 0.8); V's K=1 values are higher: less of the low-D population
    species("S", "PEPTIDEA", [0.50, 0.52, 0.48], [0.20, 0.21, 0.19], [0.80, 0.79, 0.81])
    species("V", "PEPTIDEA", [0.62, 0.64, 0.60], [0.20, 0.19, 0.21], [0.80, 0.81, 0.79])
    # states too close to tell apart (on the reference)
    species("S", "PEPTIDEB", [0.50, 0.50, 0.50], [0.45, 0.30, 0.60], [0.55, 0.70, 0.40])
    species("V", "PEPTIDEB", [0.50, 0.51, 0.49], [0.20, 0.21, 0.19], [0.80, 0.79, 0.81])
    # the low-D state itself moves in V: each side keeps its own positions
    species("S", "PEPTIDEC", [0.50, 0.52, 0.48], [0.20, 0.21, 0.19], [0.80, 0.79, 0.81])
    species("V", "PEPTIDEC", [0.60, 0.62, 0.58], [0.35, 0.36, 0.34], [0.80, 0.81, 0.79])
    monkeypatch.setattr(compare, "_records", lambda *a, **k: (groups, {}, 1))
    every_two = {("", s, p, "", 1, 10, 60.0): (2, -50.0) for s in ("S", "V")
                 for p in ("PEPTIDEA", "PEPTIDEB", "PEPTIDEC")}  # every set two populations
    pop = compare_peptides(None, "S", K=2, rank=1, quantity="population", best_sets=every_two)
    a = np.flatnonzero(pop.peptide == "PEPTIDEA")[0]
    b = np.flatnonzero(pop.peptide == "PEPTIDEB")[0]
    assert pop.value_reference[a] == pytest.approx(0.5, abs=1e-9)
    assert pop.value[a] == pytest.approx(0.3, abs=1e-9)
    # the K=1 threshold: t(0.975, 4) x typical sd x sqrt(2/3), over both species' K=1 values;
    # the typical sd is robust: the median of the variances scaled by the chi-square median
    from scipy import stats

    k1 = [[0.50, 0.52, 0.48], [0.62, 0.64, 0.60], [0.50, 0.50, 0.50], [0.50, 0.51, 0.49],
          [0.50, 0.52, 0.48], [0.60, 0.62, 0.58]]
    sd = np.sqrt(np.median([np.var(v, ddof=1) * 2 / stats.chi2.median(2) for v in k1]))
    assert pop.delta_crit == pytest.approx(stats.t.ppf(0.975, 4) * sd * np.sqrt(2 / 3))
    assert pop.threshold[a] == pytest.approx(pop.delta_crit / 0.6)
    assert pop.significant[a] and pop.reason[a] == ""
    # the same states: the K=1 difference over the shared separation, with the K=1 verdict
    k1_cmp = compare_peptides(None, "S", best_sets=every_two)
    one = np.flatnonzero(k1_cmp.peptide == "PEPTIDEA")[0]
    assert pop.positions[a] == "shared"
    assert pop.delta[a] == pytest.approx((0.50 - 0.62) / 0.6)
    assert pop.t[a] == pytest.approx(-abs(k1_cmp.t[one])) and pop.p[a] == k1_cmp.p[one]
    assert pop.significant[a] == k1_cmp.significant[one]
    # where the positions differ, each population's whole error, the positions' included
    # (HISTORY 144)
    c = np.flatnonzero(pop.peptide == "PEPTIDEC")[0]
    assert pop.positions[c] == "own"
    assert pop.delta[c] == pytest.approx(pop.value[c] - pop.value_reference[c])
    z = pop.delta[c] / pop.delta_se[c]
    assert pop.t[c] == pytest.approx(z) and pop.p[c] == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert pop.positions[b] == ""
    # the comparisons handed in, as the report does, give the same rows
    given = compare_peptides(None, "S", K=2, rank=1, quantity="population", best_sets=every_two,
                             comparisons=[k1_cmp, *(compare_peptides(
                                 None, "S", K=2, rank=r, best_sets=every_two) for r in (1, 2))])
    np.testing.assert_array_equal(given.positions, pop.positions)
    np.testing.assert_array_equal(given.significant, pop.significant)
    assert pop.reason[b] == "not separated (reference)" and np.isnan(pop.value[b])
    assert not pop.significant[b] and np.isnan(pop.threshold[b])
    assert pop.columns()["delta_crit"][a] == pop.threshold[a]
    # rank 2: the high-D component's share
    high = compare_peptides(None, "S", K=2, rank=2, quantity="population", best_sets=every_two)
    assert high.value[np.flatnonzero(high.peptide == "PEPTIDEA")[0]] == pytest.approx(0.7)
    assert high.delta[np.flatnonzero(high.peptide == "PEPTIDEA")[0]] == pytest.approx(0.2)


def test_charge_states_of_a_replicate_are_averaged_before_replicates_are_combined(monkeypatch):
    """Charge states of one replicate share its sample and injection: they are averaged first,
    so a comparison sees the replicates (n = 3 here), not the spectra (6)."""
    from hxex.analysis import compare, compare_peptides
    from hxex.analysis.species import Point, _by_replicate, _combine_replicates

    def recs(values):  # values[rep][charge]
        return [(v, 0.02, np.nan, np.nan, z, 10, rep + 1, False, "")
                for rep, by_z in enumerate(values) for z, v in zip((2, 3), by_z, strict=True)]

    s = recs([[0.50, 0.52], [0.46, 0.48], [0.54, 0.56]])
    values, ses = _by_replicate(s)
    np.testing.assert_allclose(values, [0.51, 0.47, 0.55])
    np.testing.assert_allclose(ses, np.sqrt(2 * 0.02**2) / 2)
    mean, se = _combine_replicates(s)
    assert mean == pytest.approx(0.51)
    assert se == pytest.approx(np.sqrt(max(np.var(values, ddof=1), ses[0] ** 2) / 3))
    groups = {("", "S", "PEPTIDE", "", 1, 10, Point((60.0, "")), 1, 1): s,
              ("", "V", "PEPTIDE", "", 1, 10, Point((60.0, "")), 1, 1):
                  recs([[0.60, 0.62], [0.58, 0.60], [0.64, 0.66]])}
    monkeypatch.setattr(compare, "_records", lambda *a, **k: (groups, {}, 1))
    pc = compare_peptides(None, "S")
    assert (pc.n_reference[0], pc.n[0]) == (3, 3)
    assert pc.value_reference[0] == pytest.approx(0.51)
    assert pc.value[0] == pytest.approx(np.mean([0.61, 0.59, 0.65]))


def test_lever_rule_takes_k1_from_every_charge_state():
    """A replicate's K=1 value is averaged over every charge state that has one (as in
    compare_peptides.csv), the K=2 positions over those with both; a charge state fitted at
    K=1 only no longer leaves the K=1 side."""
    from hxex.analysis import ChoiceConfig
    from hxex.analysis.compare import _lever_sides
    from hxex.analysis.species import Point

    def rec(v, z, rep):
        return (v, 0.01, np.nan, np.nan, z, 10, rep, False, "")

    key = ("", "S", "PEPTIDE", "", 1, 10, Point((60.0, "")))
    groups = {(*key, 1, 1): [rec(v, z, r) for r in (1, 2, 3) for z, v in ((2, 0.5), (3, 0.6))],
              (*key, 2, 1): [rec(0.3, 2, r) for r in (1, 2, 3)],
              (*key, 2, 2): [rec(0.8, 2, r) for r in (1, 2, 3)]}
    info = _lever_sides(groups, ChoiceConfig(max_fD_se=1))[("", "S")][
        ("PEPTIDE", "", 1, 10, 60.0)]
    assert info["reason"] == "" and info["value"] == pytest.approx((0.8 - 0.55) / 0.5)


def test_fragment_series_are_not_corrected(results):
    """A series flagged uncorrected (a fragment ion) keeps its values in the species table and
    has no corrected values in the components table (owner, 2026-10-21)."""
    import dataclasses

    c = results.controls
    raw = dataclasses.replace(c, flags=np.full(len(c), "uncorrected"))
    up = species_uptake(dataclasses.replace(results, controls=raw))
    assert not up.corrected.any() and not up.control_fallback.any()
    # as measured: the fully deuterated row is the controls' level, not n_amides
    fd = (up.role == "fulldeut") & (up.K == 1)
    assert up.mean[fd] == pytest.approx([np.mean(c.mfd_measured)], abs=0.05)
    assert (up.mean[fd] < up.n_amides[fd] - 1).all()
    from hxex.io.bundle import _corrected_columns

    data = {"measurement": np.arange(2), "data_id": np.asarray(c.data_id[:2]),
            "mean": np.array([1.0, 2.0])}
    cols = _corrected_columns(data, None, raw)
    assert np.isnan(cols["mean_corrected"]).all()


def test_a_population_comparison_scaled_by_its_separation():
    """The Woods plot's population panel: the difference and its error times each peptide's
    separation, judged by the one K=1 delta_crit; the same peptides are significant, and the
    panel shares the fD panels' range (owner, 2026-10-21)."""
    from hxex.analysis import PeptideComparison
    from hxex.report import woods_figure

    n = 3
    base = dict(sample=np.array(["M"] * n), peptide=np.array(["AAAA", "CCCC", "DDDD"]),
                modification=np.array([""] * n), start=np.array([1, 10, 20]),
                end=np.array([8, 18, 30]), time=np.full(n, 60.0),
                value_reference=np.full(n, 0.5), value_reference_se=np.full(n, 0.01),
                value_se=np.full(n, 0.01), t=np.full(n, 5.0), p=np.full(n, 0.001),
                n_reference=np.full(n, 3), n=np.full(n, 3), reference="WT")
    sep = np.array([0.1, 0.5, 0.9])
    delta = np.array([0.2, 0.2, -0.05])
    pop = PeptideComparison(**base, value=0.5 + delta, delta=delta, delta_se=np.full(n, 0.02),
                            significant=np.abs(delta) > 0.04 / sep, K=2, delta_crit=0.04,
                            quantity="population", threshold=0.04 / sep, separation=sep)
    scaled = pop.scaled_by_separation()
    assert scaled.scaled and scaled.scaled_by_separation() is scaled
    np.testing.assert_allclose(scaled.delta, delta * sep)
    np.testing.assert_allclose(scaled.delta_se, 0.02 * sep)
    assert (scaled.threshold == 0.04).all()
    assert (np.abs(scaled.delta) > scaled.threshold).tolist() == pop.significant.tolist()
    assert np.isnan(PeptideComparison(**base, value=0.5 + delta, delta=delta,
                                      delta_se=np.full(n, 0.02),
                                      significant=np.zeros(n, bool)).separation).all()
    fd = PeptideComparison(**base, value=0.5 + delta, delta=delta, delta_se=np.full(n, 0.02),
                           significant=np.zeros(n, bool), delta_crit=0.04)
    assert fd.scaled_by_separation() is fd
    fig = woods_figure([("K=1 fD", fd), ("population × separation", scaled)], "M", 60.0)
    top, bottom = fig.axes
    assert top.get_ylim() == bottom.get_ylim()
    assert "Δpop × sep" in bottom.get_ylabel()
    dashed = [ln for ln in bottom.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2  # one line each side, across the panel
    # drawn as the high-D share: above zero (red) is more D in the sample, as on the fD panels
    from hxex.report.compare import INCREASE

    high = scaled.as_high_share()
    assert high.share == "high" and high.as_high_share() is high and fd.as_high_share() is fd
    np.testing.assert_allclose(high.delta, -scaled.delta)
    np.testing.assert_allclose(high.value, 1 - scaled.value)
    assert high.significant.tolist() == scaled.significant.tolist()
    assert "share" not in high.columns()
    fig = woods_figure([("K=1 fD", fd), ("K=2 high-D population", high)], "M", 60.0)
    bottom = fig.axes[1]
    assert "Δhigh-D share × sep" in bottom.get_ylabel()
    texts = [x.get_text() for x in bottom.get_legend().get_texts()]
    assert "significantly larger high-D share than WT (more D)" in texts
    red = [ln for ln in bottom.get_lines() if str(ln.get_gid()).startswith("peptide:")
           and ln.get_color() == INCREASE]
    assert red and all(ln.get_ydata()[0] > 0 for ln in red)


def test_delta_crit_is_robust_to_a_few_broken_replicates():
    """The hybrid rule's threshold comes from the typical replicate scatter (the median of the
    chi-square-scaled variances): one series whose replicates read a different envelope barely
    moves it, where it dominates the plain pool (owner, 2026-10-22, HISTORY 140)."""
    from scipy import stats

    from hxex.analysis.compare import _delta_crit

    rng = np.random.default_rng(3)
    good = [0.5 + 0.02 * rng.standard_normal(3) for _ in range(200)]
    clean = _delta_crit(good, 0.05)
    assert clean == pytest.approx(stats.t.ppf(0.975, 4) * 0.02 * np.sqrt(2 / 3), rel=0.15)
    broken = good + [np.array([0.4, 0.4, 2.0])] * 3
    assert _delta_crit(broken, 0.05) == pytest.approx(clean, rel=0.05)
    assert np.isnan(_delta_crit([np.array([0.5])], 0.05))


def test_spectra_left_out_whole_and_why():
    """[analysis.filters] leaves out whole spectra (owner, 2026-10-22, HISTORY 140): a replicate
    far from its other replicates (same series; the run's typical replicate sd), a K=1 mean
    above the amide count, and series with a stand-in fully deuterated control (unless the
    sample has no measured one); a spectrum the review accepted is kept."""
    import dataclasses
    from types import SimpleNamespace

    from hxex.analysis import left_out_spectra
    from hxex.analysis.species import SpeciesFilters, _left_out, analysis_controls

    results = Results.from_run(run(raw_spectra(n_series=5), config()))  # enough series for
    # a typical replicate scatter that one outlier cannot set

    assert len(left_out_spectra(results)["spectrum_id"]) == 0  # clean synthetic data
    fit = results.fit
    controls = analysis_controls(results, None, quiet=True)
    at = controls.rows(fit.data_id)
    n_am = np.asarray(controls.n_amides, dtype=float)[at]
    means = fit.fits[1].means[:, 0].copy()
    timed = np.flatnonzero((np.asarray(fit.role) == "") & np.isfinite(fit.time))
    far, above = timed[0], timed[-1]
    means[far] = means[far] + 0.3 * n_am[far]  # 0.3 fD off its replicates, still possible
    means[above] = n_am[above] + 0.5
    k1 = SimpleNamespace(means=means[:, None])
    fake = SimpleNamespace(fit=SimpleNamespace(**{**{f.name: getattr(fit, f.name)
                                                     for f in dataclasses.fields(fit)},
                                                  "fits": {1: k1}}),
                           binned=results.binned,
                           review_status=np.asarray(results.review_status).copy())
    keep = np.ones(len(fit), dtype=bool)
    out, why = _left_out(fake, SpeciesFilters(), controls, n_am, keep)
    assert why[far] == "replicate outlier" and why[above] == "above the amide count"
    assert out.sum() == 2
    # four replicated groups are too few for a typical scatter: not assessed (audit
    # 2026-11-01 Low 71)
    four = np.asarray(fit.data_id) != 4
    assert _left_out(fake, SpeciesFilters(), controls, n_am, four)[1][far] == ""
    off = SpeciesFilters(replicate_outlier=None, above_amides=False)
    assert not _left_out(fake, off, controls, n_am, keep)[0].any()
    fake.review_status[far] = "accepted"  # the review's decision wins
    assert _left_out(fake, SpeciesFilters(), controls, n_am, keep)[0].sum() == 1
    # a stand-in fully deuterated control: its series is left out while the sample has a
    # measured one, kept when every series uses a stand-in
    flags = np.asarray(controls.flags).astype(object)
    flags[0] = "fd_missing;fd_median"
    one = dataclasses.replace(controls, flags=flags.astype(str))
    out, why = _left_out(results, SpeciesFilters(), one, n_am, keep)
    first = np.asarray(fit.data_id) == controls.data_id[0]
    assert (why[first] == "stand-in control").all() and not out[~first].any()
    assert not _left_out(results, SpeciesFilters(stand_in_controls="include"), one, n_am,
                         keep)[0].any()
    # "other_sample" keeps a series that took the same peptide's control in another sample,
    # and still leaves out the retention map and the run-level stand-ins
    other = SpeciesFilters(stand_in_controls="other_sample")
    for stand_in, kept in (("fd_missing;fd_other_sample", True), ("fd_missing;fd_map", False),
                           ("fd_missing;fd_median", False)):
        flags[0] = stand_in
        out = _left_out(results, other, dataclasses.replace(controls, flags=flags.astype(str)),
                        n_am, keep)[0]
        assert out[first].any() != kept and not out[~first].any()
    every = dataclasses.replace(controls, flags=np.full(len(controls), "fd_missing;fd_median"))
    assert not _left_out(results, SpeciesFilters(), every, n_am, keep)[0].any()
    with pytest.raises(ValueError, match="stand_in_controls"):
        SpeciesFilters(stand_in_controls="drop")
    with pytest.raises(ValueError, match="replicate_outlier"):
        SpeciesFilters(replicate_outlier=0)


def test_each_undeuterated_spectrum_above_m0_max_or_set_aside_leaves_the_analysis():
    """An undeuterated spectrum the controls left out of their mean leaves the analysis's
    undeuterated rows (owner, 2026-11-05; HISTORY 286), whatever stand_in_controls says, with
    its reason: above m0_max, or far from its replicates. One above m0_max that the controls kept
    (every undeuterated spectrum of its series is: m0_high) follows stand_in_controls instead; the
    review's acceptance keeps them; labelled spectra are not judged so."""
    import dataclasses
    from types import SimpleNamespace

    from hxex.analysis.species import (
        UNDEUT_HIGH_REASON,
        UNDEUT_LEFT_OUT_REASON,
        SpeciesFilters,
        _left_out,
        analysis_controls,
    )

    results = Results.from_run(run(raw_spectra(n_series=2), config()))
    fit = results.fit
    controls = analysis_controls(results, None, quiet=True)
    assert controls.left_out == ()  # the run's controls list it (clean data: nothing)
    at = controls.rows(fit.data_id)
    n_am = np.asarray(controls.n_amides, dtype=float)[at]
    role = np.asarray(fit.role, dtype=str)
    undeut = np.flatnonzero(role == "undeut")
    high, aside, labelled = int(undeut[0]), int(undeut[-1]), int(np.flatnonzero(role == "")[0])
    means = fit.fits[1].means[:, 0].copy()
    means[high] = controls.config.m0_max + 1.0
    means[labelled] = controls.config.m0_max + 1.0  # an early time point may read so
    fake = SimpleNamespace(fit=SimpleNamespace(**{**{f.name: getattr(fit, f.name)
                                                     for f in dataclasses.fields(fit)},
                                                  "fits": {1: SimpleNamespace(
                                                      means=means[:, None])}}),
                           binned=results.binned,
                           review_status=np.asarray(results.review_status).copy())
    set_aside = dataclasses.replace(controls, left_out=(str(fit.spectrum_id[high]),
                                                        str(fit.spectrum_id[aside])))
    keep = np.ones(len(fit), dtype=bool)
    for filters in (SpeciesFilters(), SpeciesFilters(stand_in_controls="include",
                                                     replicate_outlier=None)):
        out, why = _left_out(fake, filters, set_aside, n_am, keep)
        assert why[high] == UNDEUT_HIGH_REASON and why[aside] == UNDEUT_LEFT_OUT_REASON
        assert np.flatnonzero(out).tolist() == sorted([high, aside])  # not the labelled one
    # above m0_max but kept by the controls (the series' every one is): not by these reasons
    kept = dataclasses.replace(controls, left_out=())
    assert not _left_out(fake, SpeciesFilters(stand_in_controls="include"), kept, n_am,
                         keep)[0].any()
    fake.review_status[high] = "accepted"
    assert not _left_out(fake, SpeciesFilters(), set_aside, n_am, keep)[0][high]


def test_an_undeuterated_control_above_m0_max_leaves_its_series_out(caplog):
    """stand_in_controls = "exclude" also leaves out a series whose undeuterated control read
    above m0_max (m0_high: another species in its windows; owner, 2026-10-25, HISTORY 167), with
    no exception for a sample; a missing one (m0_missing) is not a reason; the review's
    acceptance and "include" keep it."""
    import dataclasses
    import logging
    from types import SimpleNamespace

    from hxex.analysis.species import SpeciesFilters, _left_out, analysis_controls

    results = Results.from_run(run(raw_spectra(n_series=3), config()))
    fit = results.fit
    controls = analysis_controls(results, None, quiet=True)
    n_am = np.asarray(controls.n_amides, dtype=float)[controls.rows(fit.data_id)]
    keep = np.ones(len(fit), dtype=bool)
    first = np.asarray(fit.data_id) == controls.data_id[0]

    def with_flags(values):
        return dataclasses.replace(controls, flags=np.asarray(values, dtype=str))

    flags = np.asarray(controls.flags).astype(object)
    flags[0] = "m0_high;m0_multimodal"
    heavy = with_flags(flags)
    out, why = _left_out(results, SpeciesFilters(), heavy, n_am, keep)
    assert (why[first] == "stand-in control (undeuterated)").all() and not out[~first].any()
    assert not _left_out(results, SpeciesFilters(stand_in_controls="include"), heavy, n_am,
                         keep)[0].any()
    accepted = SimpleNamespace(fit=fit, binned=results.binned, review_status=np.where(
        first, "accepted", np.asarray(results.review_status)))
    assert not _left_out(accepted, SpeciesFilters(), heavy, n_am, keep)[0].any()
    flags[0] = "m0_missing"
    assert not _left_out(results, SpeciesFilters(), with_flags(flags), n_am, keep)[0].any()
    # every series heavy: all left out (no sample exception), with one warning
    with caplog.at_level(logging.WARNING, logger="hxex.analysis.species"):
        out, _ = _left_out(results, SpeciesFilters(), with_flags(
            np.full(len(controls), "m0_high")), n_am, keep)
    assert out.all()
    assert sum("above m0_max" in r.getMessage() for r in caplog.records) == 1


def test_the_left_out_summary_counts_every_step(two_sample_bundle, tmp_path):
    """left_out_summary.csv (owner, 2026-10-24): one row per step from the spectra read to the
    analysis, a count of 0 included, its whole-spectra rows equal to left_out_spectra.csv's
    reasons; the spectra [[input.exclude]] left out before fitting come from the manifest."""
    import collections
    import csv
    import dataclasses

    from hxex.analysis import AnalysisConfig, left_out_spectra, left_out_summary
    from hxex.io import open_bundle
    from hxex.report import Settings, write_analysis

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    write_analysis(results, AnalysisConfig(reference="S", format="png"), tmp_path / "a")
    with open(tmp_path / "a" / "left_out_summary.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["run", "stage", "what", "count", "unit", "setting"]
    stages = {r["stage"] for r in rows}
    assert {"before fitting", "review", "controls", "analysis: whole spectra"} <= stages
    assert any(s.startswith("analysis: two-population descriptions") for s in stages)
    count = {r["what"]: int(r["count"]) for r in rows}
    assert count["fitted"] == len(results.fit)
    assert count["spectra read"] == len(results.binned)  # nothing excluded before fitting
    assert count["left out by [[input.exclude]] (no rules set)"] == 0
    reasons = collections.Counter(left_out_spectra(results)["reason"])
    for r in rows:
        if r["what"].startswith("left out: "):
            assert int(r["count"]) == reasons.get(r["what"][len("left out: "):], 0)
    # the rules of [[input.exclude]], as the manifest records them
    record = {"rules": [{"time": 0.1, "note": "quench-flow", "matched": 3}],
              "spectrum_ids": ["a", "b", "c"]}
    assert Settings.from_manifest({"excluded": record}).excluded == record
    excluded = dataclasses.replace(results,
                                   settings=dataclasses.replace(results.settings, excluded=record))
    t = left_out_summary(excluded)
    by = dict(zip(t["what"], t["count"], strict=True))
    assert by["matched by an [[input.exclude]] rule: time 100 ms (quench-flow); stored, not "
              "fitted"] == 3
    assert by["spectra read"] == len(results.binned) + 3
    # without [analysis.filters]: the steps before the analysis only
    plain = left_out_summary(results, None)
    assert not any(s.startswith("analysis") for s in plain["stage"])


def test_choice_defaults_follow_the_populations_rule(tmp_path):
    """[analysis.choice]'s defaults depend on the rule (owner, 2026-10-22, HISTORY 142): the
    bootstrap rule's tests are the rule itself (0.95, 10 %); with best_set_K they only check
    the fits' comparison (0.5, 5 %). A key set in the project applies in either mode, and the
    keys not set take the mode's defaults."""
    import dataclasses

    from hxex.analysis import AnalysisConfig
    from hxex.config import load_project

    best = AnalysisConfig()
    assert (best.choice_limits.confidence, best.choice_limits.min_population) == (0.5, 0.05)
    boot = dataclasses.replace(best, populations="bootstrap")
    assert (boot.choice_limits.confidence, boot.choice_limits.min_population) == (0.95, 0.10)
    assert AnalysisConfig.from_dict(boot.to_dict()) == boot
    (tmp_path / "hxex.toml").write_text(
        'project = "p"\n[output]\nbundle = "results"\n[analysis]\npopulations = "bootstrap"\n'
        "[analysis.choice]\nconfidence = 0.9\n")
    limits = load_project(tmp_path / "hxex.toml", analysis_only=True).analysis.choice_limits
    assert (limits.confidence, limits.min_population) == (0.9, 0.10)



def test_k2_threshold_comes_from_the_two_population_sets(results):
    """With the best_set_K gate, the K=2 comparisons' delta_crit pools only the sets best
    described by two populations: the others' K=2 fits split one population (HISTORY 144)."""
    from hxex.analysis import best_sets, compare_peptides, species_uptake

    best = best_sets(species_uptake(results, ks=(1, 2)))
    sample = sorted({k[1] for k in best})[0]
    ungated = compare_peptides(results, sample, K=2, rank=1, two_population_sets=False)
    every = {k: (2, v[1]) for k, v in best.items()}
    none = {k: (1, v[1]) for k, v in best.items()}
    assert compare_peptides(results, sample, K=2, rank=1, best_sets=every).delta_crit == (
        pytest.approx(ungated.delta_crit))
    assert np.isnan(compare_peptides(results, sample, K=2, rank=1, best_sets=none).delta_crit)


def test_spectrum_quality_filtering(caplog):
    """[analysis.filters] spectrum_quality_filtering (owner, 2026-10-23, HISTORY 152): the
    screens' scores remove a spectrum (a control too, which then leaves the back-exchange
    correction; a series whose measured fully deuterated controls all go becomes a stand-in and
    says so), keep one at one population, or warn; strict uses lower thresholds; accepted
    spectra, fragment series and unassessed spectra are never touched."""
    import dataclasses
    import logging
    from types import SimpleNamespace

    from hxex.analysis import left_out_spectra
    from hxex.analysis.screening import screening
    from hxex.analysis.species import SpeciesFilters, _spectrum_records, analysis_controls

    results = Results.from_run(run(raw_spectra(n_series=3), config()))
    fit, b = results.fit, results.binned
    rows = {str(s): j for j, s in enumerate(b.spectrum_id)}
    role, data_id = np.asarray(fit.role).astype(str), np.asarray(fit.data_id)
    timed = np.flatnonzero((role == "") & (data_id == 0))
    point, two, beyond = timed
    mild = np.flatnonzero((role == "") & (data_id == 1))[0]
    undeut = np.flatnonzero((role == "undeut") & (data_id == 1))[0]
    fulldeut = np.flatnonzero((role == "fulldeut") & (data_id == 2))
    scores = {name: np.zeros(len(b)) for name in SCREEN_SCORES}

    def put(name, i, value):
        scores[name][rows[str(fit.spectrum_id[i])]] = value

    put("screen_below_mono", point, 0.3)
    put("screen_below_mono", undeut, 0.3)
    for i in fulldeut:
        put("screen_off_position", i, 0.5)
    put("screen_between_separate", two, 0.3)
    put("screen_beyond_reach", beyond, 0.3)
    put("screen_between", mild, 0.3)  # removed at strict (0.20), kept at loose (0.40)
    # series 2 another peptide (L -> I: the same mass and amides), so that its fully
    # deuterated control cannot come from the others
    peptide = np.where(np.asarray(b.data_id) == 2, "RDKVQKEYAI", np.asarray(b.peptide))
    results = dataclasses.replace(results, binned=dataclasses.replace(b, peptide=peptide))
    b = results.binned
    screened = dataclasses.replace(results, binned=dataclasses.replace(b, **scores))

    loose = screening(screened, "loose")
    assert set(np.flatnonzero(loose.remove)) == {point, undeut, *fulldeut}
    assert loose.reason[point] == "spectrum quality (screen_below_mono)"
    assert list(np.flatnonzero(loose.one_population)) == [two]
    assert list(np.flatnonzero(loose.warn)) == [beyond]
    assert screening(screened, "strict").remove[mild] and not loose.remove[mild]
    # beyond reach: a warning at loose below 0.5, removed at strict from 0.3 and at loose from
    # 0.5 (HISTORY 199)
    strict = screening(screened, "strict")
    assert strict.remove[beyond] and strict.reason[beyond] == (
        "spectrum quality (screen_beyond_reach)")
    high = {**scores, "screen_beyond_reach": scores["screen_beyond_reach"] * 2}
    far = screening(dataclasses.replace(results, binned=dataclasses.replace(b, **high)), "loose")
    assert far.remove[beyond] and not far.warn[beyond]
    assert not screening(screened, "none").remove.any()
    assert not screening(results, "loose").remove.any()  # clean synthetic data
    with pytest.raises(ValueError, match="spectrum_quality_filtering"):
        SpeciesFilters(spectrum_quality_filtering="medium")

    # the removed controls leave the correction: series 1 loses one undeuterated replicate,
    # series 2 its measured fully deuterated control
    before = analysis_controls(screened, None, quiet=True)
    after = analysis_controls(screened, None, quiet=True, screened=loose.remove)
    one, two_ = before.rows(np.array([1, 2]))
    assert after.m0[one] != before.m0[one] and after.n_undeut[one] == before.n_undeut[one] - 1
    assert before.fd_measured[two_] and not after.fd_measured[two_]

    left = left_out_spectra(screened)
    why = dict(zip(left["spectrum_id"], left["reason"], strict=True))
    assert why[str(fit.spectrum_id[point])] == "spectrum quality (screen_below_mono)"
    assert why[str(fit.spectrum_id[undeut])] == "spectrum quality (screen_below_mono)"
    series2 = [str(s) for s, d in zip(fit.spectrum_id, data_id, strict=True) if d == 2]
    lost = {why.get(s) for s in series2} - {"spectrum quality (screen_off_position)"}
    assert lost == {"stand-in control (screened)"}
    assert not len(left_out_spectra(screened, SpeciesFilters(spectrum_quality_filtering="none"))[
        "spectrum_id"])

    with caplog.at_level(logging.INFO):
        _, _, _, dropped = _spectrum_records(screened, (1, 2), False, SpeciesFilters())
    assert dropped["one_population"] == 1
    assert "screen_beyond_reach" in caplog.text

    # accepted spectra and fragment series are not screened
    status = np.asarray(screened.review_status).copy()
    status[point] = "accepted"
    fake = SimpleNamespace(fit=fit, binned=screened.binned, review_status=status,
                           spectra_source=None)
    assert not screening(fake, "loose").remove[point]
    fragment = dataclasses.replace(screened.binned, modification=np.where(
        np.asarray(b.data_id) == 0, "ion_type:c", np.asarray(b.modification)))
    fake = SimpleNamespace(fit=fit, binned=fragment, review_status=screened.review_status,
                           spectra_source=None)
    assert not screening(fake, "loose").remove[point] and screening(fake, "loose").remove[undeut]


def test_series_off_their_mz_are_left_out():
    """[analysis.filters] off_mz (HISTORY 200): every spectrum of a series the m/z check flags
    off_mz is left out, controls included, so its fully deuterated control leaves the
    back-exchange correction; off_mz = false keeps it; the run's controls leave its controls
    out as the default analysis does; accepted spectra stay."""
    import dataclasses
    from types import SimpleNamespace

    from hxex.analysis import left_out_spectra, left_out_summary
    from hxex.analysis.screening import control_removal, screening
    from hxex.analysis.species import SpeciesFilters, analysis_controls

    results = Results.from_run(run(raw_spectra(n_series=3), config()))
    fit, b = results.fit, results.binned
    off_series = np.asarray(b.data_id) == 2
    flagged = dataclasses.replace(results, binned=dataclasses.replace(
        b, envelope_offset_ppm=np.where(off_series, 20.0, 0.5),
        envelope_offset_limit_ppm=np.full(len(b), 3.0),
        # another peptide (L -> I), so that its controls cannot stand in from the others
        peptide=np.where(off_series, "RDKVQKEYAI", np.asarray(b.peptide))))
    mine = np.asarray(fit.data_id) == 2
    gone = screening(flagged, "none")
    assert np.array_equal(gone.remove, mine) and set(gone.reason[mine]) == {"off its m/z"}
    assert not screening(flagged, "none", off_mz=False).remove.any()
    assert not screening(results, "loose").remove.any()  # not assessed: never removed

    left = left_out_spectra(flagged)
    assert set(left["spectrum_id"]) == set(np.asarray(fit.spectrum_id)[mine].astype(str))
    assert set(left["reason"]) == {"off its m/z"}
    assert not len(left_out_spectra(flagged, SpeciesFilters(off_mz=False))["spectrum_id"])
    summary = left_out_summary(flagged, ks=(1,))
    what = dict(zip(summary["what"], summary["count"], strict=True))
    assert what["left out: off its m/z"] == mine.sum()
    role = np.asarray(fit.role).astype(str)
    assert what["control spectra left out: their series is off its m/z"] == (
        mine & (role != "")).sum()

    # its controls leave the correction (the run's too)
    before = analysis_controls(flagged, None, quiet=True)
    after = analysis_controls(flagged, None, quiet=True, screened=gone.remove)
    i = before.rows(np.array([2]))[0]
    assert before.fd_measured[i] and not after.fd_measured[i]
    removal = control_removal(flagged.binned, fragments=())
    assert set(np.flatnonzero(removal)) == set(np.flatnonzero(
        off_series & (np.asarray(b.role).astype(str) != "")))

    # accepted spectra stay
    status = np.asarray(results.review_status).copy()
    status[np.flatnonzero(mine)[0]] = "accepted"
    fake = SimpleNamespace(fit=fit, binned=flagged.binned, review_status=status,
                           spectra_source=None)
    assert screening(fake, "none").remove.sum() == mine.sum() - 1


def test_the_runs_controls_leave_out_the_screened_controls(tmp_path):
    """Audit 2026-10-28 question 9: the run's controls leave out the control spectra the loose
    level removes (recorded by id), so series.csv shows the state the default analysis uses;
    the analysis reuses them at that level and recomputes them at another."""
    import dataclasses

    from hxex.analysis.screening import control_removal, screening
    from hxex.analysis.species import analysis_controls
    from hxex.controls import fit_controls
    from hxex.io import open_bundle

    results = Results.from_run(run(raw_spectra(n_series=2), config()))
    assert results.controls.screened == ()  # screened, nothing removed
    fit, b = results.fit, results.binned
    role = np.asarray(fit.role).astype(str)
    undeut = np.flatnonzero((role == "undeut") & (np.asarray(fit.data_id) == 1))[0]
    labelled = np.flatnonzero(role == "")[0]
    scores = np.zeros(len(b))
    rows = {str(s): j for j, s in enumerate(b.spectrum_id)}
    scores[rows[str(fit.spectrum_id[undeut])]] = 0.3
    scores[rows[str(fit.spectrum_id[labelled])]] = 0.3  # not a control: stays in the controls
    picked = dataclasses.replace(b, screen_below_mono=scores)
    removal = control_removal(picked, fragments=())
    assert list(np.flatnonzero(removal)) == [rows[str(fit.spectrum_id[undeut])]]
    assert not control_removal(picked, fragments=(1,)).any()  # fragment series: not screened

    by_run = fit_controls(fit, np.asarray(results.measurements.n_max) / 1.2,
                          results.controls.config,
                          screened=np.isin(fit.spectrum_id, b.spectrum_id[removal]))
    assert by_run.screened == (str(fit.spectrum_id[undeut]),)
    screened = dataclasses.replace(results, binned=picked, controls=by_run)
    level = screening(screened, "loose").remove
    assert analysis_controls(screened, None, quiet=True) is by_run
    assert analysis_controls(screened, None, quiet=True, screened=level) is by_run
    none = analysis_controls(screened, None, quiet=True, screened=np.zeros(len(fit), bool))
    i = by_run.rows([1])[0]
    assert none.n_undeut[i] == by_run.n_undeut[i] + 1 and none.screened is None

    # the bundle keeps the ids
    out = run(raw_spectra(n_series=2), config())
    out = dataclasses.replace(out, controls=dataclasses.replace(out.controls,
                                                                screened=("a", "b")))
    out.write(tmp_path / "r")
    with open_bundle(tmp_path / "r") as bundle:
        assert bundle.controls().screened == ("a", "b")


def test_residue_significance_map():
    """Each residue's class: significant on the shared peptides, by direction and the shared
    difference over the threshold; not tested where no shared peptide covers it (hatched)."""
    from hxex.analysis import ResidueComparison
    from hxex.report import SIGNIFICANCE_COLOURS, plot_residue_significance
    from hxex.report.significance import significance_classes

    n = 7
    delta_shared = np.array([0.0, -0.15, -0.3, 0.15, 0.3, 0.5, np.nan])
    rc = ResidueComparison(
        sample=np.array(["V"] * n), time=np.full(n, 4.0), residue=np.arange(1, n + 1),
        value_reference=np.full(n, 0.3), value_reference_se=np.full(n, 0.01),
        value=np.full(n, 0.5), value_se=np.full(n, 0.01), delta=np.full(n, 0.2),
        delta_se=np.full(n, 0.01), z=np.full(n, 20.0),
        significant=np.array([False, True, True, True, True, True, False]),
        reference="R", delta_crit=0.1, delta_shared=delta_shared,
        delta_shared_se=np.full(n, 0.01))
    assert list(significance_classes(rc)) == [0, 1, 2, 4, 5, 6, -1]
    fig = plot_residue_significance(rc)
    ax = fig.axes[0]
    assert [p.get_gid() for p in ax.patches] == ["xxxx:V  4 s:7-7"]
    assert len(SIGNIFICANCE_COLOURS) == 7
    # a page of a long time course: its times only, its rows only
    import dataclasses

    from hxex.gui.figures import figure_rows

    two = ResidueComparison(**{
        f.name: (np.concatenate([getattr(rc, f.name)] * 2) if f.name != "time" else
                 np.concatenate([rc.time, np.full(n, 60.0)]))
        if isinstance(getattr(rc, f.name), np.ndarray) else getattr(rc, f.name)
        for f in dataclasses.fields(ResidueComparison)})
    page = plot_residue_significance(two, times=[60.0])
    assert [t.get_text() for t in page.axes[0].get_yticklabels()] == ["V  1 min"]
    assert {r["time"] for r in figure_rows(page)[1]} == {60.0}
    with pytest.raises(ValueError, match="no comparison"):
        plot_residue_significance(two, times=[999.0])
    with pytest.raises(ValueError, match="empty"):
        plot_residue_significance(ResidueComparison(**{
            name: np.array([]) for name in ("sample", "time", "residue", "value_reference",
                                            "value_reference_se", "value", "value_se",
                                            "delta", "delta_se", "z", "significant")}))


def test_residue_significance_map_at_k2():
    """Where only one side has two populations: textures (no bootstrap needed); where both have
    two, the population share's difference (with a bootstrap at K = 2: blue for more of the
    low-D population), else grey; position sections only where both keep two."""
    from hxex.analysis import ChoiceComparison, ResidueComparison
    from hxex.report import plot_residue_significance_k2

    n = 5
    choices = ChoiceComparison(
        run=np.array([""] * n, dtype=object), sample=np.array(["V"] * n),
        time=np.full(n, 4.0), residue=np.arange(1, n + 1),
        two_reference=np.array([True, False, True, True, False]),
        two=np.array([False, True, True, True, False]),
        reason_reference=np.array([""] * n), reason=np.array([""] * n),
        agree=np.array([False, False, True, True, True]), reference="R")
    untested = plot_residue_significance_k2(choices)
    gids = sorted(p.get_gid().split(":")[0] for p in untested.axes[0].patches)
    assert gids == ["////", "\\\\\\\\"]  # sample only (residue 2), reference only (residue 1)
    assert "need a bootstrap" in untested.texts[0].get_text()
    keep = np.array([2, 3])  # residues 3 and 4: two in both
    pop = ResidueComparison(
        sample=np.array(["V", "V"]), time=np.full(2, 4.0), residue=keep + 1,
        value_reference=np.full(2, 0.5), value_reference_se=np.full(2, 0.02),
        value=np.array([0.8, 0.52]), value_se=np.full(2, 0.02), delta=np.array([0.3, 0.02]),
        delta_se=np.full(2, 0.03), z=np.array([10.0, 0.7]), significant=np.array([True, False]),
        reference="R", delta_crit=0.1, quantity="population",
        delta_shared=np.array([0.3, 0.02]), delta_shared_se=np.full(2, 0.03))
    tested = plot_residue_significance_k2(choices, pop, pop, pop)
    image = tested.axes[0].collections[0].get_array().reshape(-1, 5)
    # row 1 is the populations section's first sample row (row 0 its heading)
    assert image[1, 2] == 2  # more of the low-D share: less D, 2-4x (blue)
    assert image[1, 3] == 0  # not significant
    assert np.ma.is_masked(image[1, 4])  # one population in both: white
    with pytest.raises(ValueError, match="empty"):
        plot_residue_significance_k2(ChoiceComparison(**{
            f: np.array([]) for f in ("run", "sample", "time", "residue", "two_reference", "two",
                                      "reason_reference", "reason", "agree")}))


def test_a_residue_difference_needs_a_covering_peptide_to_carry_it():
    """The gate (HISTORY 206): significant only where a peptide covering the residue (its
    exchanging part) is significant with the same sign; the residue test's other calls are
    residue_test_only; no peptide comparison for the state: left as is."""
    from hxex.analysis import PeptideComparison, ResidueComparison, gated_by_peptides
    from hxex.report.significance import significance_classes

    n = 5  # residues 1-5 of sample V at 4 s; residue 5 of a modified state without peptides
    rc = ResidueComparison(
        sample=np.array(["V"] * 4 + ["V pS7"]), time=np.full(n, 4.0),
        residue=np.array([2, 3, 6, 9, 5]), value_reference=np.full(n, 0.3),
        value_reference_se=np.full(n, 0.01), value=np.full(n, 0.5), value_se=np.full(n, 0.01),
        delta=np.array([0.2, 0.2, -0.2, 0.2, 0.2]), delta_se=np.full(n, 0.01),
        z=np.full(n, 20.0), significant=np.ones(n, dtype=bool), reference="R", delta_crit=0.1)
    pc = PeptideComparison(
        sample=np.array(["V", "V"]), peptide=np.array(["AKLPGSTV", "TVEDRW"]),
        modification=np.array(["", ""]), start=np.array([2, 8]), end=np.array([9, 13]),
        time=np.array([4.0, 4.0]), value_reference=np.array([0.3, 0.3]),
        value_reference_se=np.array([0.01, 0.01]), value=np.array([0.5, 0.5]),
        value_se=np.array([0.01, 0.01]), delta=np.array([0.2, 0.2]),
        delta_se=np.array([0.01, 0.01]), t=np.array([5.0, 5.0]), p=np.array([0.001, 0.2]),
        n_reference=np.array([3, 3]), n=np.array([3, 3]),
        significant=np.array([True, False]), reference="R", delta_crit=0.1)
    gated = gated_by_peptides(rc, pc, nterm_subtract=1)
    # residue 2: AKLPGSTV's first residue (no amide counted); 3: carried; 6: opposite sign;
    # 9: AKLPGSTV's last residue (TVEDRW covers it too but is not significant)
    assert list(gated.peptide_support[:4]) == [False, True, False, True]
    assert list(gated.significant) == [False, True, False, True, True]
    assert list(gated.residue_test_only) == [True, False, True, False, False]
    assert np.isnan(gated.peptide_support[4])  # V pS7: no peptide comparison, left as it was
    assert list(significance_classes(gated)[:3]) == [-2, 5, -3]  # 0.2 = 2x: 2-4x
    assert {"peptide_support", "residue_test_only"} <= set(gated.columns())


def test_the_gate_and_the_region_test_take_the_exchanging_residues():
    """Audit 2026-11-01 Analysis 64: an Ac- peptide's first residue has an amide and carries
    its residue; a proline has none."""
    import dataclasses

    from hxex.analysis import (
        Annotations,
        PeptideComparison,
        Region,
        ResidueComparison,
        compare_regions,
        gated_by_peptides,
    )

    n = 3
    rc = ResidueComparison(
        sample=np.array(["V"] * n), time=np.full(n, 4.0), residue=np.array([1, 4, 5]),
        value_reference=np.full(n, 0.3), value_reference_se=np.full(n, 0.01),
        value=np.full(n, 0.5), value_se=np.full(n, 0.01), delta=np.full(n, 0.2),
        delta_se=np.full(n, 0.01), z=np.full(n, 20.0), significant=np.ones(n, dtype=bool),
        reference="R", delta_crit=0.1)
    pc = PeptideComparison(
        sample=np.array(["V"]), peptide=np.array(["Ac-AKLPGS"]), modification=np.array([""]),
        start=np.array([1]), end=np.array([6]), time=np.array([4.0]),
        value_reference=np.array([0.3]), value_reference_se=np.array([0.01]),
        value=np.array([0.5]), value_se=np.array([0.01]), delta=np.array([0.2]),
        delta_se=np.array([0.01]), t=np.array([5.0]), p=np.array([0.001]),
        n_reference=np.array([3]), n=np.array([3]), significant=np.array([True]),
        reference="R", delta_crit=0.1)
    gated = gated_by_peptides(rc, pc, nterm_subtract=1)
    assert list(gated.peptide_support) == [True, False, True]  # residue 4 is the proline
    # residues 1-3 hold three of the five exchanging residues (1, 2, 3, 5, 6): supported
    regions = compare_regions(gated, Annotations(regions=(Region("N", 1, 3),)), pc,
                              nterm_subtract=1)
    assert list(regions.peptide_support) == [True]
    # Low 69: a modified peptide supports its sample's residues when the map combined it
    phospho = dataclasses.replace(pc, peptide=np.array(["Ac-AKLPGpS"]))
    apart = gated_by_peptides(rc, phospho, nterm_subtract=1)  # its own state, 'V pS6'
    assert np.isnan(apart.peptide_support.astype(float)).all()
    combined = gated_by_peptides(rc, phospho, nterm_subtract=1, modified="combined")
    assert list(combined.peptide_support) == [True, False, True]
    regions = compare_regions(combined, Annotations(regions=(Region("N", 1, 3),)), phospho,
                              nterm_subtract=1, modified="combined")
    assert list(regions.peptide_support) == [True]


def test_residue_blocks_are_kept_and_outlined():
    """The block each residue lies in (the stretch between peptide boundaries: the resolution)
    is in the residue map, carried into the residue comparison from the shared peptides, and
    outlined on the significance map (HISTORY 207)."""
    from hxex.analysis import (
        compare_residues,
        concat_tables,
        judged_on_shared,
        paired_residue_maps,
    )
    from hxex.report import plot_residue_significance

    up = table([("R", "AKLPGSTV", 2, 4.0, 0.30, 0.01), ("R", "GSTV", 6, 4.0, 0.50, 0.01),
                ("V", "AKLPGSTV", 2, 4.0, 0.60, 0.01), ("V", "GSTV", 6, 4.0, 0.80, 0.01)])
    rm = residue_map(up, sequences=SEQUENCE)
    v = rm.where(sample="V", K=1) & np.isin(rm.residue, [3, 4, 6, 7, 9])
    assert [(int(a), int(b)) for a, b in zip(rm.block_start[v], rm.block_end[v], strict=True)] == [
        (3, 6), (3, 6), (3, 6), (7, 9), (7, 9)]
    assert np.isnan(rm.block_start[rm.where(sample="V", K=1) & (rm.residue == 12)]).all()
    pairs = paired_residue_maps(up, "R", sequences=SEQUENCE)
    rc = judged_on_shared(compare_residues(rm, "R", delta_crit=0.1),
                          concat_tables([compare_residues(m, "R", delta_crit=0.1)
                                         for m in pairs.values()]))
    shared_blocks = zip(rc.block_start, rc.block_end, strict=True)
    assert {(int(a), int(b)) for a, b in shared_blocks} == {(3, 6), (7, 9)}
    assert {"block_start", "block_end"} <= set(rc.columns())
    fig = plot_residue_significance(rc)
    blocks = sorted(p.get_gid() for p in fig.axes[0].patches
                    if (p.get_gid() or "").startswith("block:"))
    assert blocks == ["block:V  4 s:3-6", "block:V  4 s:7-9"]
    plain = plot_residue_significance(rc, block_outline=None)
    assert not any((p.get_gid() or "").startswith("block:") for p in plain.axes[0].patches)


def test_k2_map_marks_where_the_own_peptides_decide_differently():
    """The K = 2 map decides one or two on the shared peptides; a yellow triangle marks the
    residues each sample's own peptides decide differently (HISTORY 207)."""
    from hxex.analysis import ChoiceComparison
    from hxex.report import plot_residue_significance_k2

    def choices(two_ref, two):
        n = len(two)
        return ChoiceComparison(
            run=np.array([""] * n, dtype=object), sample=np.array(["V"] * n),
            time=np.full(n, 4.0), residue=np.arange(1, n + 1),
            two_reference=np.array(two_ref), two=np.array(two),
            reason_reference=np.array([""] * n), reason=np.array([""] * n),
            agree=np.array(two_ref) == np.array(two), reference="R")

    shared = choices([False, False, True], [False, False, True])
    own = choices([False, False, True], [False, True, True])  # residue 2: two in V's own
    fig = plot_residue_significance_k2(shared, own_choices=own)
    gids = [p.get_gid() for p in fig.axes[0].patches if p.get_gid()]
    assert [g for g in gids if g.startswith("differs:")] == ["differs:V  4 s:2-2"]
    assert not any(g.startswith("////") for g in gids)  # the texture follows the shared choice
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert any("decided differently" in t for t in texts)


def test_regions_are_compared_from_their_residues_with_the_same_gate():
    """compare_regions (HISTORY 209): the mean of the region's shared residue differences and of
    their errors, z and delta_crit, and a significant peptide of the same sign mostly inside the
    region; the map draws one column per region."""
    import dataclasses

    from hxex.analysis import PeptideComparison, Region, ResidueComparison, compare_regions
    from hxex.report import plot_region_significance

    res = np.arange(2, 12)  # residues 2-11 of V at 4 s
    n = len(res)
    delta = np.where(res <= 6, 0.3, 0.02)
    rc = ResidueComparison(
        sample=np.array(["V"] * n), time=np.full(n, 4.0), residue=res,
        value_reference=np.full(n, 0.3), value_reference_se=np.full(n, 0.01),
        value=0.3 + delta, value_se=np.full(n, 0.01), delta=delta, delta_se=np.full(n, 0.02),
        z=delta / 0.02, significant=delta > 0.1, reference="R", delta_crit=0.1,
        delta_shared=delta, delta_shared_se=np.full(n, 0.02))
    pc = PeptideComparison(
        sample=np.array(["V"]), peptide=np.array(["AKLPG"]), modification=np.array([""]),
        start=np.array([2]), end=np.array([6]), time=np.array([4.0]),
        value_reference=np.array([0.3]), value_reference_se=np.array([0.01]),
        value=np.array([0.6]), value_se=np.array([0.01]), delta=np.array([0.3]),
        delta_se=np.array([0.01]), t=np.array([20.0]), p=np.array([0.001]),
        n_reference=np.array([3]), n=np.array([3]), significant=np.array([True]),
        reference="R", delta_crit=0.1)
    regions = [Region("N", 2, 6), Region("C", 7, 11), Region("wide", 2, 11),
               Region("far", 15, 19)]
    cmp = compare_regions(rc, regions, pc, nterm_subtract=1)
    by = {r: i for i, r in enumerate(cmp.region)}
    assert cmp.significant[by["N"]] and cmp.delta_shared[by["N"]] == pytest.approx(0.3)
    assert cmp.n_residues[by["N"]] == 5 and cmp.coverage[by["N"]] == pytest.approx(1.0)
    assert not cmp.significant[by["C"]] and not cmp.region_test_only[by["C"]]  # too small
    # "wide": mean 0.16 > delta_crit, z 8; the peptide 3-6 (exchanging) has 4 of 4 inside
    assert cmp.significant[by["wide"]]
    assert cmp.n_residues[by["far"]] == 0 and np.isnan(cmp.delta_shared[by["far"]])
    gated = compare_regions(rc, [Region("N", 2, 6)], dataclasses.replace(
        pc, significant=np.array([False])), nterm_subtract=1)
    assert gated.region_test_only[0] and not gated.significant[0]
    assert "delta_crit" in cmp.columns() and "threshold" not in cmp.columns()
    fig = plot_region_significance(cmp)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
        "N (2-6)", "wide (2-11)", "C (7-11)", "far (15-19)"]
    assert any((p.get_gid() or "").startswith("xxxx:") for p in fig.axes[0].patches)


def test_two_populations_far_apart_take_their_average_from_k2():
    """Where a set has two populations at least K2_AVERAGE_SEPARATION deuterons apart, the K = 1
    fit settles on one of them: the set's average is the K = 2 population-weighted mean, each
    spectrum's own (HISTORY 210)."""
    from hxex.analysis.species import (
        K2_AVERAGE_SEPARATION,
        Point,
        _k2_average_records,
        _set_average_ks,
    )

    assert K2_AVERAGE_SEPARATION == 2.0
    point = Point((60.0, ""))
    base = ("", "V", "AKLPGSTVED", "", 2, 11, point)

    def rec(value, se, population, rep):  # the record layout of species._records
        return (value, se, population, 0.01, 2, 9.0, rep, False, "", False, False)

    groups = {base + (1, 1): [rec(0.05, 0.01, 1.0, 1), rec(0.90, 0.01, 1.0, 2)],  # one mode each
              base + (2, 1): [rec(0.0, 0.02, 0.5, 1), rec(0.1, 0.02, 0.4, 2)],
              base + (2, 2): [rec(1.0, 0.02, 0.5, 1), rec(0.9, 0.02, 0.6, 2)]}
    key = ("", "V", "AKLPGSTVED", "", 2, 11, 60.0)
    out = _k2_average_records(groups, {key})
    values = [r[0] for r in out[base + (1, 1)]]
    assert values == pytest.approx([0.5, 0.4 * 0.1 + 0.6 * 0.9])
    se = out[base + (1, 1)][0][1]
    assert se == pytest.approx(np.sqrt((0.5 * 0.02) ** 2 + (0.5 * 0.02) ** 2))
    assert out[base + (2, 1)] is groups[base + (2, 1)]  # the K = 2 records stay as they are
    assert _k2_average_records(groups, set()) is groups
    cols = {"run": np.array(["", "", ""]), "sample": np.array(["V"] * 3),
            "peptide": np.array(["AKLPGSTVED"] * 3), "modification": np.array([""] * 3),
            "start": np.array([2] * 3), "end": np.array([11] * 3), "time": np.array([60.0] * 3),
            "role": np.array([""] * 3), "K": np.array([1, 2, 2]), "rank": np.array([1, 1, 2]),
            "fD": np.array([0.45, 0.05, 0.95]), "n_amides": np.array([9.0] * 3),
            "best_set_K": np.array([2, 2, 2])}
    assert _set_average_ks(cols) == {key: 2}  # 0.9 fD x 9 amides = 8.1 D apart
    near = dict(cols, fD=np.array([0.45, 0.40, 0.55]))  # 1.35 D apart
    assert _set_average_ks(near) == {}
    one = dict(cols, best_set_K=np.array([1, 1, 1]))
    assert _set_average_ks(one) == {}
    # three populations or more: the set's own number, the population test's where it ran
    # (best_set_K follows it only up to the largest K analysed), whatever the separation
    assert _set_average_ks(dict(near, set_test_K=np.array([3, 3, 3]))) == {key: 3}
    assert _set_average_ks(dict(near, best_set_K=np.array([3, 3, 3]))) == {key: 3}
    assert _set_average_ks(dict(cols, set_test_K=np.array([4, 4, 4]))) == {key: 4}


def test_a_fasta_numbered_by_its_header_keeps_the_datas_residue_numbers(tmp_path):
    """[analysis] fasta with a header >NAME/START-END numbers its first residue START (a tag
    before residue 1, or a fragment of the protein), as the SpecExport read does: the residue
    letters by number are padded or trimmed so that a peptide at residues 2-4 is checked
    against residues 2-4 of the protein."""
    from hxex.analysis import AnalysisConfig
    from hxex.analysis.residues import check_sequence, protein_letters
    from hxex.chem import DEFAULT_MODIFICATIONS as reg

    (tmp_path / "tag.fasta").write_text(">P/-2-5\nGGGMKVLA\n")  # GGG before residue 1
    (tmp_path / "part.fasta").write_text(">P/3-6\nVLAE\n")  # residues 3 to 6 only
    (tmp_path / "plain.fasta").write_text(">P\nMKVLA\n")
    tag = AnalysisConfig(fasta="tag.fasta").sequences(tmp_path)
    assert tag.first == -2 and protein_letters(tag, reg) == list("MKVLA")
    part = AnalysisConfig(fasta="part.fasta").sequences(tmp_path)
    assert protein_letters(part, reg) == ["", ""] + list("VLAE")
    check_sequence("KVL", 2, protein_letters(tag, reg), reg)
    check_sequence("VLA", 3, protein_letters(part, reg), reg)
    plain = AnalysisConfig(fasta="plain.fasta").sequences(tmp_path)
    assert type(plain) is str and protein_letters(plain, reg) == list("MKVLA")


def test_a_k2_average_counts_the_spectra_it_averages(results, monkeypatch):
    """Audit 2026-11-01 Analysis 36: a K = 1 row whose value comes from the K = 2 descriptions
    (average_K 2) counts the spectra that have one, not the K = 1 group's."""
    from hxex.analysis import species as sp

    original = sp.best_set

    def two(cols, evidence, ks, min_population=0.05):  # every set reads two populations
        out = original(cols, evidence, ks, min_population)
        out["best_set_K"][:] = 2
        return out

    monkeypatch.setattr(sp, "best_set", two)
    monkeypatch.setattr(sp, "K2_AVERAGE_SEPARATION", 0.0)
    labelled = np.flatnonzero((np.asarray(results.fit.role) == "")
                              & (np.asarray(results.fit.data_id) == 0))
    reviewed = _review_k1_ids(results, [str(results.fit.spectrum_id[labelled[0]])])
    up = species_uptake(reviewed, filters=SpeciesFilters(min_population=0.0))
    rows = (up.role == "") & (up.time == 60.0)
    assert (up.average_K[rows] == 2).all()
    k1, k2 = rows & (up.K == 1), rows & (up.K == 2) & (up.rank == 1)
    np.testing.assert_array_equal(up.n_spectra[k1], up.n_spectra[k2])


def test_sample_styles_keep_a_sample_s_colour_in_every_figure():
    from hxex.report.coverage import SAMPLE_STYLES, sample_styles

    every = sample_styles(["WT", "A", "B", "C"], "WT")
    assert every["WT"].colour == "#000000" and every["A"].colour == SAMPLE_STYLES[1][0]
    assert [every[s].colour for s in ("A", "B", "C")] == [c for c, _, _ in SAMPLE_STYLES[1:4]]
    # chosen colours, and the defaults skipping the ones chosen
    chosen = sample_styles(["WT", "A", "B"], "WT", {"B": SAMPLE_STYLES[1][0], "WT": "grey"})
    assert chosen["WT"].colour == "grey" and chosen["B"].colour == SAMPLE_STYLES[1][0]
    assert chosen["A"].colour == SAMPLE_STYLES[2][0]
    assert sample_styles(["A", "WT"], "WT")["WT"].colour == "#000000"  # the reference first


def test_analysis_colours_reach_every_figure_that_colours_samples(two_sample_bundle, caplog):
    from matplotlib.colors import to_hex

    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import analysis_outputs

    with pytest.raises(ValueError, match="not a colour"):
        AnalysisConfig(colours={"V": "blurple"})
    cfg = AnalysisConfig(residue_level=True, reference="S",
                         colours={"V": "orange", "nobody": "red"})
    with open_bundle(two_sample_bundle) as b:
        outputs = analysis_outputs(Results.from_bundle(b), cfg)
    assert "[analysis.colours] nobody: no such sample" in caplog.text
    figures = dict(outputs.figures)
    overlay = next(draw for stem, draw in figures.items() if stem.startswith("overlay_residues"))
    fig = overlay()
    colours = {to_hex(line.get_color()) for ax in fig.axes for line in ax.get_lines()
               if (line.get_gid() or "").startswith("overlay:V:")}
    assert colours == {to_hex("orange")}
    redundancy = figures["redundancy_resolution"]()
    assert to_hex("orange") in {to_hex(line.get_color()) for line in redundancy.axes[0].get_lines()}


def test_a_reference_list_or_all_makes_one_analysis_each(two_sample_bundle, tmp_path, capsys):
    import shutil

    from hxex import cli
    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import analysis_outputs, analysis_references

    assert AnalysisConfig(reference=["S", "V", "S"]).reference == ("S", "V")  # each once
    assert AnalysisConfig(reference=["S"]).reference == "S"  # one stays one
    assert AnalysisConfig(reference="all").references == ("all",)
    assert not AnalysisConfig(reference="all").single_reference
    with pytest.raises(ValueError, match="not among the samples"):
        AnalysisConfig(reference=["S", "X"], samples=["S", "V"])
    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    assert analysis_references(results, AnalysisConfig(reference="all")) == ["S", "V"]
    assert analysis_references(results, AnalysisConfig(reference="all",
                                                       samples=["V", "S"])) == ["V", "S"]
    assert analysis_references(results, AnalysisConfig()) == [""]
    with pytest.raises(ValueError, match="no reference sample 'X'"):
        analysis_references(results, AnalysisConfig(reference=["S", "X"]))
    with pytest.raises(ValueError, match="one reference at a time"):
        analysis_outputs(results, AnalysisConfig(reference=["S", "V"]))
    # hxex analyse: a folder per reference
    bundle = tmp_path / "results"
    shutil.copytree(two_sample_bundle, bundle)
    assert cli.main(["-q", "analyse", str(bundle), "--reference", "all", "--format", "png"]) == 0
    assert (tmp_path / "results_analysis_S").is_dir() and (tmp_path / "results_analysis_V").is_dir()
    assert any(p.name.startswith("butterfly_V") for p in (tmp_path / "results_analysis_S")
               .iterdir())
    assert any(p.name.startswith("butterfly_S") for p in (tmp_path / "results_analysis_V")
               .iterdir())
    out = capsys.readouterr().out
    assert out.count("figure(s) in") == 2
    assert cli.main(["-q", "analyse", str(bundle), "--reference", "S,V", "--out",
                     str(tmp_path / "one")]) == 1
    assert "make a folder each" in capsys.readouterr().err


def test_every_samples_residue_values_on_one_axis():
    """Every sample's K=1 fD by residue on one axis, none a reference (owner: the all-samples
    overlay, drawn as redundancy and resolution draw the samples): a colour and marker each, the
    samples named in order, a data-table row per sample and residue."""
    from hxex.analysis import ResidueMap
    from hxex.gui.figures import figure_rows
    from hxex.report.compare import plot_residue_samples
    from hxex.report.coverage import sample_styles

    rows = [(s, r, 0.3 + 0.1 * n) for n, s in enumerate(("WT", "V", "M")) for r in (3, 4, 6)]
    n = len(rows)
    rm = ResidueMap(sample=np.array([r[0] for r in rows]), time=np.full(n, 4.0),
                    K=np.ones(n, dtype=np.int64), rank=np.ones(n, dtype=np.int64),
                    residue=np.array([r[1] for r in rows]), aa=np.array(["A"] * n),
                    fD=np.array([r[2] for r in rows]), fD_se=np.full(n, 0.02),
                    redundancy=np.ones(n, dtype=np.int64), resolution=np.ones(n),
                    proline=np.zeros(n, dtype=bool), covered=np.ones(n, dtype=bool))
    styles = sample_styles(["M", "V", "WT"])
    fig = plot_residue_samples(rm, ["M", "V", "WT"], 4.0, styles=styles)
    ax = fig.axes[0]
    points = {line.get_gid(): line for line in ax.lines if line.get_gid()}
    assert set(points) == {"samples:M", "samples:V", "samples:WT"}
    for name in ("M", "V", "WT"):  # each sample in its own colour and marker
        assert points[f"samples:{name}"].get_marker() == styles[name].marker
    np.testing.assert_allclose(points["samples:WT"].get_ydata(), 0.3)
    assert [t.get_text() for t in ax.get_legend().texts] == ["M", "V", "WT"]  # the order given
    assert ax.get_ylabel() == r"fD$_{\rm avg}$"  # the analysis's average, as on the overlay
    columns, table = figure_rows(fig)
    assert columns == ["sample", "time", "time_label", "residue", "fD", "fD_se"]
    assert len(table) == 9 and {r["sample"] for r in table} == {"M", "V", "WT"}
    assert ax.get_title(loc="left") == "±1 standard error"  # no threshold given: the errors
    # with the analysis's K=1 threshold: bands of +-delta_crit / 2, its value said (owner)
    crit = plot_residue_samples(rm, ["M", "V", "WT"], 4.0, delta_crit=0.04)
    assert crit.axes[0].get_title(loc="left") == "±½ δcrit (δcrit = 0.040 fD)"
    band = crit.axes[0].collections[0].get_paths()[0].vertices[:, 1]
    assert np.nanmax(band) - np.nanmin(band) == pytest.approx(0.04, abs=1e-9)  # 0.3 +- 0.02
    # one y axis for every time of an analysis (owner): the highest value given holds it
    own, held = fig.axes[0].get_ylim()[1], plot_residue_samples(
        rm, ["M", "V", "WT"], 4.0, top=1.3).axes[0].get_ylim()[1]
    assert held > own and held >= 1.3
    with pytest.raises(ValueError, match="no residue values"):
        plot_residue_samples(rm, ["WT"], 60.0)


def test_every_residue_figure_carries_its_data_table(two_sample_bundle):
    """The data tab beside a figure (owner, 2026-11-04: the residue figures first): each
    residue figure of an analysis carries the rows it draws, in the analysis tables' column
    names; a row drawn twice (the reference in every panel of an overlay) is kept once."""
    from hxex.analysis import AnalysisConfig, Annotations, Region
    from hxex.gui.figures import figure_rows
    from hxex.io import open_bundle
    from hxex.report import Results
    from hxex.report.analysis import analysis_outputs

    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    outputs = analysis_outputs(results, AnalysisConfig(residue_level=True, 
        reference="S", annotations=Annotations(regions=(Region("A", 1, 5),))))
    drawn = {stem.split("_1min")[0]: figure_rows(make()) for stem, make in outputs.figures}
    expected = {  # the figure -> columns its rows must have
        "redundancy_resolution": {"sample", "residue", "redundancy", "resolution"},
        "choice_S": {"residue", "two_populations", "fD_avg", "fD_low_se", "population_low"},
        "residues_all": {"sample", "residue", "fD", "fD_se"},
        "overlay_residues": {"sample", "reference", "measure", "residue", "value", "value_se"},
        "residue_significance": {"sample", "residue", "delta", "significant"},
        "residue_significance_K2_V": {"section", "sample", "residue", "two", "two_reference"},
        "region_significance": {"sample", "region", "significant"},
        "regions": {"sample", "region", "fD", "fD_se", "time_label"},
        "butterfly_V": {"sample", "start", "end", "peptide", "delta", "time_label"},
    }
    for name, need in expected.items():
        columns, rows = drawn[name]
        assert rows, f"{name} carries no rows"
        assert need <= set(columns), f"{name}: {need - set(columns)}"
    # the K=2 Woods page in the order of the K=2 map: the population first (owner)
    woods = dict(outputs.figures)["woods_K2_V_1min"]()
    titles = [ax.get_title(loc="left") for ax in woods.axes if ax.get_title(loc="left")]
    assert titles[0] == "K=2 high-D population" and len(titles) == 3
    _, overlay = drawn["overlay_residues"]
    keys = [tuple(sorted((k, repr(v)) for k, v in r.items())) for r in overlay]
    assert len(keys) == len(set(keys))


def test_figures_that_grow_go_on_over_pages_of_the_page_size(two_sample_bundle):
    """[analysis] page_size (owner, 2026-11-04: a page size, as a setting; plots differ in
    shape, so no fixed grid): a figure that grows with the data goes on over pages NAME,
    NAME_2, each holding as many of its own panels as fit; the maps keep only a page's times."""
    from hxex.analysis import AnalysisConfig
    from hxex.io import open_bundle
    from hxex.report import Results
    from hxex.report.analysis import _map_pages, _pages, analysis_outputs
    from hxex.report.catalogue import kind_of

    assert _pages([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]] and _pages([], 3) == [[]]
    # a map's rows: (12 - 2.4) / 0.2 = 48 rows, 4 samples: 12 times a page
    assert [len(p) for p in _map_pages((16.0, 12.0), list(range(30)), 4, 2.4)] == [12, 12, 6]
    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    small = analysis_outputs(results, AnalysisConfig(residue_level=True, reference="S",
                                                     page_size=(6, 4)))
    stems = [stem for stem, _ in small.figures]
    # a panel a page: the undeuterated control's coverage, then the time point's
    assert {"coverage_S", "coverage_S_2"} <= set(stems)
    assert kind_of("coverage_S_2").name == "coverage"
    assert kind_of("residue_significance_2").name == "residue_significance"
    assert kind_of("population_tiers_3").name == "population_tiers"
    first = dict(small.figures)["coverage_S"]()
    assert [ax.get_title() for ax in first.axes if ax.get_title()][0].endswith("undeuterated")
    default = analysis_outputs(results, AnalysisConfig(residue_level=True, reference="S"))
    assert not any(stem.endswith("_2") for stem, _ in default.figures)  # one page each
    with pytest.raises(ValueError):
        AnalysisConfig(residue_level=True, page_size=(3, 2))
    with pytest.raises(ValueError, match="width and height"):
        AnalysisConfig(residue_level=True, page_size="big")
    assert AnalysisConfig.from_dict({"page_size": [20, 15]}).page_size == (20.0, 15.0)


def test_residue_level_outputs_need_peptides_that_make_one_protein(two_sample_bundle,  # noqa: F811
                                                                   tmp_path):
    """Pooled unrelated peptides have no residues to resolve (owner, 2026-11-08): a sample with
    one peptide range, with peptides laid end to end that overlap nowhere, or whose peptides
    name different residues at one number has no residue-level outputs by default."""
    from hxex.analysis import AnalysisConfig
    from hxex.analysis.residues import residue_level_states
    from hxex.io import open_bundle
    from hxex.report import analysis_outputs

    up = table([
        ("P", "PEPTIDEK", 1, 60.0, 0.4, 0.02), ("P", "TIDEKAAL", 4, 60.0, 0.5, 0.02),
        ("one", "PEPTIDEK", 1, 60.0, 0.4, 0.02),
        ("end", "PEPTIDEK", 1, 60.0, 0.4, 0.02), ("end", "AALLKR", 9, 60.0, 0.5, 0.02),
        ("clash", "PEPTIDEK", 1, 60.0, 0.4, 0.02), ("clash", "WWWWWWW", 3, 60.0, 0.5, 0.02),
        ("mod", "PEpSTIDEK", 1, 60.0, 0.4, 0.02), ("mod", "STIDEKA", 3, 60.0, 0.5, 0.02),
    ])
    level = {s: why for (_, s), why in residue_level_states(up).items()}
    assert level["P"] == "" and level["mod"] == ""  # overlapping; pS matches S
    assert level["one"].startswith("one peptide range")
    assert level["end"].startswith("no two of its 2 peptide ranges overlap")
    assert "different residues at residue 3 (P and W)" in level["clash"]
    # a sequence for the sample: the residue map checks its peptides against it instead
    assert residue_level_states(up, sequences={"clash": "PEPTIDEKAAL"})[("", "clash")] == ""
    assert set(residue_level_states(up, True).values()) == {""}
    assert "" not in residue_level_states(up, False).values()
    with pytest.raises(ValueError, match="residue_level"):
        residue_level_states(up, "yes")
    with pytest.raises(ValueError, match="residue_level"):
        AnalysisConfig(residue_level="sometimes")
    # the synthetic run has one peptide per sample: the peptide outputs only, by default
    with open_bundle(two_sample_bundle) as b:
        results = Results.from_bundle(b)
    auto = analysis_outputs(results, AnalysisConfig(reference="S"))
    assert "residues" not in auto.tables and "coverage" not in auto.tables
    assert "compare_residues" not in auto.tables and "compare_peptides" in auto.tables
    stems = [stem for stem, _ in auto.figures]
    # the coverage maps place the peptides, averaging nothing: drawn whatever residue_level
    # says (owner, 2026-11-09); every other figure here is peptide-level too
    assert stems and all(stem.startswith(("butterfly_", "coverage_")) for stem in stems)
    assert {"coverage_S", "coverage_V"} <= set(stems)
    forced = analysis_outputs(results, AnalysisConfig(reference="S", residue_level=True))
    assert {"residues", "coverage", "compare_residues"} <= set(forced.tables)


@pytest.fixture(scope="module")
def results_k3():
    """The synthetic run fitted up to K = 3."""
    import dataclasses

    base = config()
    return Results.from_run(run(raw_spectra(n_series=2),
                                config(fit=dataclasses.replace(base.fit, k_max=3))))


def test_sets_at_three_take_their_average_from_their_own_k(results_k3, monkeypatch, caplog):
    """A set with three populations or more (owner, 2026-11-09) takes its average from every
    spectrum's population-weighted mean at the set's K, not from the K = 1 fit: charge states
    averaged within a replicate, then the replicates' mean. Every spectrum the analysis keeps
    counts (the filters on describing populations do not judge an average); a spectrum the
    review keeps at one population gives its K = 1 mean. The comparisons use the same values."""
    import logging

    from hxex.analysis import species as sp
    from hxex.controls import correct_means

    original = sp.best_set

    def three(cols, evidence, ks, min_population=0.05):  # every set reads three populations
        out = original(cols, evidence, ks, min_population)
        out["best_set_K"][:] = 3
        return out

    monkeypatch.setattr(sp, "best_set", three)
    r = results_k3
    fit, ctl, b = r.fit, r.controls, r.binned
    one = str(fit.spectrum_id[np.flatnonzero(np.asarray(fit.role) == "")[0]])
    reviewed = _review_k1_ids(r, [one])
    with caplog.at_level(logging.WARNING, logger="hxex"):
        up = species_uptake(reviewed, ks=(1, 2, 3), filters=None, populations="best_set")
    # the run bootstrapped each spectrum at its own K only: the error is the replicates'
    assert "without a bootstrap at that number" in caplog.text and "hxex resume" in caplog.text
    assert (up.average_K[up.role == ""] == 3).all() and (up.average_K[up.role != ""] == 1).all()

    at = ctl.rows(fit.data_id)
    amides = np.asarray(ctl.n_amides, dtype=float)[at]
    kf = fit.fits[3]
    k3 = np.nansum(kf.means[:, :3] * kf.populations[:, :3], axis=1)
    value = np.where(np.asarray(fit.spectrum_id).astype(str) == one, fit.fits[1].means[:, 0], k3)
    fd = correct_means(ctl, fit.data_id, value[:, None]).mean[:, 0] / amides
    index = {str(s): j for j, s in enumerate(b.spectrum_id)}
    rows = np.array([index[str(s)] for s in fit.spectrum_id])
    rep, time = np.asarray(b.rep)[rows], np.asarray(fit.time, dtype=float)
    labelled = np.asarray(fit.role) == ""
    for t in np.unique(time[labelled]):
        here = labelled & (time == t)
        per_rep = [fd[here & (rep == x)].mean() for x in np.unique(rep[here])]
        row = np.flatnonzero((up.role == "") & (up.K == 1) & (up.time == t))
        assert len(row) == 1
        assert up.fD[row[0]] == pytest.approx(np.mean(per_rep), rel=1e-9)
        assert up.n_spectra[row[0]] == int(here.sum())
    # the K = 3 rows themselves stay the K = 3 description's
    k3_rows = up.where(K=3)
    assert k3_rows.any()

    # with the default filters every kept spectrum still counts (none judged on its K = 3
    # populations): the counts of the K = 1 fit's rows, other values
    up_f = species_uptake(r, ks=(1, 2, 3), populations="best_set")
    k1_f = (up_f.role == "") & (up_f.K == 1)
    averages = sp.set_average_ks(up_f)
    assert averages and set(averages.values()) == {3}
    monkeypatch.undo()
    plain = species_uptake(r, ks=(1, 2, 3), populations="best_set")
    k1_p = (plain.role == "") & (plain.K == 1)
    assert (plain.average_K[k1_p] == 1).all()
    np.testing.assert_array_equal(up_f.n_spectra[k1_f], plain.n_spectra[k1_p])
    assert not np.allclose(up_f.fD[k1_f], plain.fD[k1_p])


def test_head_to_head_leaves_stand_in_controls_untested(results):
    """A head-to-head row whose sequence is corrected by a stand-in (the retention map here)
    keeps its values but is not tested, with the reason; the sequence is marked as such for
    the figure (owner, 2026-11-10)."""
    import dataclasses

    from hxex.analysis import SpeciesFilters
    from hxex.analysis.compare import compare_sequences
    from hxex.analysis.pooled import HeadToHead

    binned = results.binned
    other = dataclasses.replace(binned, peptide=np.where(binned.data_id == 1, "RDKVQKEYAI",
                                                         "RDKVQKEYAL"))
    ct = results.controls
    assert ct is not None and ct.sequence_measured.all()
    at = ct.rows([1])[0]
    flags = np.array(ct.flags, dtype=object).copy()
    flags[at] = "fd_missing;fd_map"
    fallback = np.array(ct.fallback, dtype=bool).copy()
    fallback[at] = True
    n_fd = np.array(ct.n_fd).copy()
    n_fd[at] = 0
    stand_in = dataclasses.replace(ct, flags=flags, fallback=fallback, n_fd=n_fd)
    assert not stand_in.sequence_measured[at] and stand_in.sequence_measured.sum() == len(ct) - 1
    panel = HeadToHead(1, 10, "A", ("RDKVQKEYAL", "RDKVQKEYAI"), ("A", "A2"), ("A", "A2: L10I"))
    patched = dataclasses.replace(results, binned=other, controls=stand_in)
    kept = compare_sequences(patched, [panel],
                             filters=SpeciesFilters(stand_in_controls="include"))
    cols = kept.columns()
    assert len(kept) and set(cols["reason"]) == {
        "stand-in control (this sequence): not tested between different sequences"}
    assert np.isnan(cols["p"]).all() and not cols["significant"].any()
    assert np.isfinite(cols["delta"]).all()
    assert {k[2] for k, own in kept.own_controls.items() if not own} == {"RDKVQKEYAI"}
    # the default leaves the stand-in series out of the analysis: no row at all
    assert len(compare_sequences(patched, [panel])) == 0
    # both corrected by their own controls: tested, no reason
    tested = compare_sequences(dataclasses.replace(results, binned=other), [panel]).columns()
    assert set(tested["reason"]) == {""} and np.isfinite(tested["p"]).any()



def test_a_quiet_recomputation_keeps_the_controls_lines_for_the_first_caller_not_quiet(results,
                                                                                    caplog):
    from hxex.analysis import analysis_controls

    changes = {"d2o_fraction": 0.81, "retention_min": 0.98}  # every control rejected: lines

    def lines():
        return [r.getMessage() for r in caplog.records if r.name == "hxex.controls"]

    with caplog.at_level("INFO", logger="hxex.controls"):
        quiet = analysis_controls(results, changes, quiet=True)
        assert lines() == []
        assert analysis_controls(results, changes) is quiet  # the same controls, once
        shown = lines()
        assert shown
        caplog.clear()
        analysis_controls(results, changes)
        assert lines() == []  # shown once
