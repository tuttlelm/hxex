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
    assert np.all((pop.value >= 0.05) & (pop.value <= 0.95))  # the filters still apply
    with pytest.raises(ValueError, match="K >= 2"):
        compare_peptides(results, "S", quantity="population")
    with pytest.raises(ValueError, match="quantity"):
        compare_peptides(results, "S", quantity="mean")


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
    assert any(p.get_gid() == "significant:6" for p in fig.axes[0].patches)
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
    bars = plot_region_bars(region_summary(rm, ann), 4.0, samples=["R", "V"])
    gids = {p.get_gid() for p in bars.axes[0].patches}
    assert {"region:R:long region name", "region:V:long region name"} <= gids
    with pytest.raises(ValueError, match="no comparison"):
        plot_peptide_differences(pc, "V", 60.0)
    from hxex.report import woods_figure

    stacked = woods_figure([("K=1", pc), ("again", pc)], "V", 4.0)
    assert [ax.get_title(loc="left") for ax in stacked.axes] == ["K=1", "again"]


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


def test_write_analysis_writes_tables_and_figures(two_sample_bundle, tmp_path):
    from hxex.analysis import AnalysisConfig, Annotations, Region, Site
    from hxex.io import open_bundle
    from hxex.report import write_analysis

    seen = []
    cfg = AnalysisConfig(reference="S", format="png",
                         annotations=Annotations(regions=(Region("all", 1, 10),),
                                                 sites=(Site(5, "x"),)))
    with open_bundle(two_sample_bundle) as b:
        written = write_analysis(Results.from_bundle(b), cfg, tmp_path / "a",
                                 progress=lambda d, t: seen.append((d, t)))
    names = {p.name for p in written}
    for table in ("species_uptake", "residues", "coverage", "populations", "compare_peptides",
                  "compare_residues", "compare_populations", "regions"):
        assert f"{table}.csv" in names
    assert "coverage_S.png" in names and "redundancy_resolution.png" in names
    assert any(n.startswith("choice_V_") for n in names)
    assert any(n.startswith("compare_residues_") for n in names)
    assert seen[-1][0] == seen[-1][1] == len(written)
    assert all(p.stat().st_size > 0 for p in written)
    with open_bundle(two_sample_bundle) as b, pytest.raises(ValueError, match="no reference"):
        write_analysis(Results.from_bundle(b), AnalysisConfig(reference="X"), tmp_path / "b")


def test_analyse_command(two_sample_bundle, tmp_path, capsys):
    from hxex import cli

    out = tmp_path / "plain"
    assert cli.main(["-q", "analyse", str(two_sample_bundle), "--out", str(out),
                     "--format", "svg"]) == 0
    assert (out / "species_uptake.csv").is_file() and not (out / "compare_peptides.csv").exists()
    assert any(p.suffix == ".svg" for p in out.iterdir())
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
    from hxex.analysis import RunSet, check_identity, compare_peptides, compare_settings

    rs = RunSet({"A": results, "B": results})
    up = species_uptake(rs)
    assert set(up.sample.tolist()) == {"A:S", "B:S"}
    a, b = up.select(up.sample == "A:S"), up.select(up.sample == "B:S")
    np.testing.assert_array_equal(a.fD, b.fD)
    np.testing.assert_array_equal(a.fD_se, b.fD_se)
    pc = compare_peptides(rs, "A:S")
    assert set(pc.sample.tolist()) == {"B:S"} and len(pc) == len(set(a.time.tolist()))
    np.testing.assert_allclose(pc.delta, 0.0)
    assert not pc.significant.any()
    assert len(check_identity(up)) == 0 and len(compare_settings(rs)) == 0
    with pytest.raises(ValueError, match="samples: A:S, B:S"):
        compare_peptides(rs, "S")
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

    up = table([("A:S", "AKLPGSTV", 2, 4.0, 0.3, 0.01), ("B:S", "AKLPGSTV", 3, 4.0, 0.3, 0.01),
                ("A:S", "GSTVED", 6, 4.0, 0.6, 0.01), ("B:V", "GSTVED", 6, 4.0, 0.6, 0.01),
                ("A:S", "TVEDRWQ", 8, 4.0, 0.6, 0.01), ("B:S", "TVEDRWQ", 8, 4.0, 0.6, 0.01)])
    up = dataclasses.replace(up, n_amides=np.array([10.0, 10.0, 10.0, 10.0, 6.0, 5.0]))
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
    with pytest.raises(ValueError, match="no comparison"):
        plot_butterfly(pc, "C:S")


def test_analyse_separate_runs(two_sample_bundle, tmp_path, capsys):
    from hxex import cli

    out = tmp_path / "runs"
    assert cli.main(["-q", "analyse", f"A={two_sample_bundle}", f"B={two_sample_bundle}",
                     "--reference", "A:S", "--out", str(out), "--format", "png"]) == 0
    names = {p.name for p in out.iterdir()}
    for table in ("run_settings", "run_identity", "compare_peptides", "compare_choices"):
        assert f"{table}.csv" in names
    assert {"coverage_A_S.png", "butterfly_B_S.png", "butterfly_A_V.png"} <= names
    rows = (out / "compare_peptides.csv").read_text().splitlines()[1:]
    assert {r.split(",")[1] for r in rows} == {"A:V", "B:S", "B:V"}  # the compared samples
    project = tmp_path / "hxex.toml"
    project.write_text(f"""project = "p"
[input]
format = "specexport"
data_dir = "d"
[analysis]
reference = "one:S"
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
    for sample, shift in (("WT", 0.0), ("D3", 0.1)):
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
    fig = plot_residue_overlay(rm, "WT", "D3", 4.0)
    lines = {line.get_gid(): line for line in fig.axes[0].lines if line.get_gid()}
    assert set(lines) == {"overlay:WT:low", "overlay:WT:high", "overlay:D3:low",
                          "overlay:D3:high"}
    np.testing.assert_allclose(lines["overlay:D3:high"].get_ydata(), 0.7)
    assert lines["overlay:WT:low"].get_linestyle() == "None"  # points, no lines
    every = plot_residue_overlay(rm, "WT", "D3", 4.0, show=("avg", "low", "high"))
    avg = next(line for line in every.axes[0].lines if line.get_gid() == "overlay:D3:avg")
    assert avg.get_markerfacecolor() == "white"
    assert "avg (K=1)" in [t.get_text() for t in every.axes[0].get_legend().texts]
    assert len(residue_overlay_figure(rm, "WT", ["WT", "D3"], 4.0).axes) == 1
    with pytest.raises(ValueError, match="show"):
        plot_residue_overlay(rm, "WT", "D3", 4.0, show=("mean",))
    with pytest.raises(ValueError, match="no residue values"):
        plot_residue_overlay(rm, "WT", "D3", 60.0)
    assert AnalysisConfig(overlay=["avg"], ks=[1]).overlay == ("avg",)
    for bad in ({"overlay": []}, {"overlay": ["mean"]}, {"ks": [1]},
                {"overlay": ["avg"], "ks": [1], "overlay_two_only": True}):
        with pytest.raises(ValueError, match="overlay"):
            AnalysisConfig(**bad)
    # only where two populations are chosen: D3's residue 6 has its K=1 value outside its ranks
    import dataclasses

    from hxex.analysis import choose_populations

    odd = (rm.sample == "D3") & (rm.residue == 6) & (rm.K == 1)
    choice = choose_populations(dataclasses.replace(rm, fD=np.where(odd, 0.75, rm.fD)))
    kept = plot_residue_overlay(rm, "WT", "D3", 4.0, show=("low", "high", "avg"), choice=choice)
    lines = {line.get_gid(): line for line in kept.axes[0].lines if line.get_gid()}
    assert list(lines["overlay:D3:high"].get_xdata()) == [3, 4]
    assert list(lines["overlay:WT:high"].get_xdata()) == [3, 4, 6]
    assert list(lines["overlay:D3:avg"].get_xdata()) == [3, 4, 6]  # the average is never filtered
