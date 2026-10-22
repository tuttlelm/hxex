# SPDX-License-Identifier: GPL-3.0-or-later
"""The K=2 comparisons made only where both sides are best described by two populations, the
two-population peptides' vote at the residues, and the comparison of the number of populations
(owner, 2026-10-21; HISTORY 137 and 141)."""

import numpy as np
import pytest

from hxex.analysis import (
    SpeciesUptake,
    choose_populations,
    compare_peptides,
    compare_population_counts,
    compare_residues,
    residue_map,
)
from hxex.analysis import compare as compare_module
from hxex.analysis.species import Point
from hxex.report import plot_population_counts, woods_populations_figure

SEQUENCE = "MAKLPGSTVEDRWQAFKLT"  # a proline at 5
# (peptide, start, best_set_K, fD K=1, K=2 low, K=2 high)
PEPTIDES = (("AKLPGSTVE", 2, 2, 0.5, 0.2, 0.8),   # amides 3..10 (no 5): weight 1/7 each
            ("GSTVEDRWQ", 6, 2, 0.5, 0.2, 0.8),   # amides 7..14: 1/8
            ("DRWQAF", 11, 1, 0.5, 0.45, 0.9))    # amides 12..16: 1/5; its K=2 is a tail


def uptake(samples=("WT",)):
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "role", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend", "best_set_K",
                                  "set_delta_criterion")}
    for sample in samples:
        for peptide, start, best, avg, low, high in PEPTIDES:
            best = best if sample == "WT" else 1  # the other sample: one population everywhere
            for k, rank, fd in ((1, 1, avg), (2, 1, low), (2, 2, high)):
                values = {"sample": sample, "peptide": peptide, "modification": "",
                          "kind": "peptide", "start": start, "end": start + len(peptide) - 1,
                          "time": 60.0, "role": "", "K": k, "rank": rank, "mean": fd * 5,
                          "mean_se": 0.05, "fD": fd, "fD_se": 0.01, "population": 0.5,
                          "population_se": 0.05, "n_amides": 5.0, "n_spectra": 3,
                          "n_charges": 1, "corrected": True, "charge_trend": False,
                          "best_set_K": best, "set_delta_criterion": -30.0 if best == 2 else 5.0}
                for name, value in values.items():
                    cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()})


def value(rm, residue, K, rank, sample="WT"):
    w = rm.where(sample=sample, K=K, rank=rank) & (rm.residue == residue)
    return float(rm.fD[w][0]) if w.any() else np.nan


def test_residue_values_from_every_peptide_and_the_two_population_vote():
    """The K=2 residue values use every covering peptide, like K=1, so the three describe the
    same peptides (owner, 2026-10-22, HISTORY 141); the two-population peptides' vote decides
    which residues have two populations."""
    up = uptake()
    plain = residue_map(up, sequences=SEQUENCE)
    gated = residue_map(up, sequences=SEQUENCE, two_population_peptides=True)
    for residue in (8, 13, 15):
        for k, rank in ((1, 1), (2, 1), (2, 2)):
            assert value(gated, residue, k, rank) == value(plain, residue, k, rank)
    assert value(gated, 13, 2, 2) > 0.8  # the one-population peptide's split counts too
    # the vote: the share of the weight the two-population peptides carry
    k1 = gated.where(sample="WT", K=1, rank=1)
    share = dict(zip(gated.residue[k1].tolist(), gated.two_population_weight[k1].tolist(),
                     strict=True))
    assert share[8] == pytest.approx(1.0)
    assert share[13] == pytest.approx((1 / 8) / (1 / 8 + 1 / 5))  # below half
    assert share[15] == 0.0 and np.isnan(plain.two_population_weight).all()
    # a line joins residues one peptide measures together; 16 is the last such residue
    linked = dict(zip(gated.residue[k1].tolist(), gated.linked_next[k1].tolist(), strict=True))
    assert linked[4] and linked[15] and not linked[16]
    choice = choose_populations(gated)
    reason = dict(zip(choice.residue.tolist(), choice.reason.tolist(), strict=True))
    two = dict(zip(choice.residue.tolist(), choice.two_populations.tolist(), strict=True))
    assert two[8] and reason[13] == "one-population peptides"
    assert reason[15] == "one-population peptides"  # K=2 values, but no two-population peptide


def test_k2_residue_differences_count_only_where_both_keep_two():
    gated = residue_map(uptake(("WT", "M")), sequences=SEQUENCE, two_population_peptides=True)
    choice = choose_populations(gated)
    assert not choice.two_populations[choice.sample == "M"].any()
    for rank in (1, 2):
        cmp = compare_residues(gated, "WT", K=2, rank=rank, delta_crit=0.0, choice=choice)
        assert not cmp.significant.any()


def test_peptide_k2_comparisons_need_two_populations_on_both_sides(monkeypatch):
    def records(values):
        return [(v, 0.01, np.nan, np.nan, 2, 10, rep + 1, False, "") for rep, v in
                enumerate(values)]

    groups = {}
    for sample, low in (("S", [0.20, 0.21, 0.19]), ("V", [0.40, 0.41, 0.39])):
        for peptide in ("PEPTIDEA", "PEPTIDEB"):
            key = ("", sample, peptide, "", 1, 10, Point((60.0, "")))
            groups[(*key, 1, 1)] = records([0.5, 0.5, 0.5])
            groups[(*key, 2, 1)] = records(low)
            groups[(*key, 2, 2)] = records([0.8, 0.8, 0.8])
    monkeypatch.setattr(compare_module, "_records", lambda *a, **k: (groups, {}, 1))
    best = {("", "S", "PEPTIDEA", "", 1, 10, 60.0): (2, -30.0),
            ("", "V", "PEPTIDEA", "", 1, 10, 60.0): (2, -30.0),
            ("", "S", "PEPTIDEB", "", 1, 10, 60.0): (2, -30.0),
            ("", "V", "PEPTIDEB", "", 1, 10, 60.0): (1, 4.0)}
    low = compare_peptides(None, "S", K=2, rank=1, best_sets=best)
    a = np.flatnonzero(low.peptide == "PEPTIDEA")[0]
    b = np.flatnonzero(low.peptide == "PEPTIDEB")[0]
    assert low.delta[a] == pytest.approx(0.2) and low.reason[a] == ""
    assert np.isnan(low.delta[b]) and not low.significant[b]
    assert low.reason[b] == "one population (sample)"
    assert compare_module._one_population({}, ("", "V"), ("P", "", 1, 2, 60.0), ("", "S"),
                                          ("P", "", 1, 2, 60.0)) == "one population (both)"


def test_the_number_of_populations_compared():
    best = {("", "WT", "PA", "", 1, 10, 60.0): (2, -30.0),
            ("", "M", "PA", "", 1, 10, 60.0): (1, 5.0),
            ("", "WT", "PB", "", 5, 15, 60.0): (1, 2.0),
            ("", "M", "PB", "", 5, 15, 60.0): (2, -4.0),
            ("", "WT", "PC", "", 12, 20, 60.0): (2, -40.0),
            ("", "M", "PC", "", 12, 20, 60.0): (2, -25.0)}
    counts = compare_population_counts(best, "WT")
    row = {p: i for i, p in enumerate(counts.peptide)}
    assert counts.differs[row["PA"]] and not counts.borderline[row["PA"]]  # clear
    assert counts.differs[row["PB"]] and counts.borderline[row["PB"]]  # -4: weak evidence
    assert not counts.differs[row["PC"]]
    assert counts.K_reference[row["PA"]] == 2 and counts.K[row["PA"]] == 1
    assert set(counts.columns()) >= {"run", "K_reference", "K", "delta_criterion",
                                     "differs", "borderline"}
    fig = plot_population_counts(counts, "M", 60.0)
    bars = {c.get_gid(): c for c in fig.axes[0].collections if c.get_gid()}
    assert set(bars) == {"count:reference:1-10", "count:sample:5-15", "count:both:12-20"}
    face = bars["count:sample:5-15"].get_facecolor()
    assert len(face) == 0 or face[0][3] == 0  # borderline: an outline only
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["M only", "WT only", "both"]
    assert fig.axes[0].get_title(loc="left").startswith("Two Populations Selected")


def test_the_woods_first_page_has_the_population_counts():
    from hxex.analysis import PeptideComparison

    n = 2
    k1 = PeptideComparison(
        sample=np.array(["M"] * n), peptide=np.array(["PA", "PB"]), modification=np.array([""] * n),
        start=np.array([1, 5]), end=np.array([10, 15]), time=np.full(n, 60.0),
        value_reference=np.full(n, 0.5), value_reference_se=np.full(n, 0.01),
        value=np.full(n, 0.6), value_se=np.full(n, 0.01), delta=np.full(n, 0.1),
        delta_se=np.full(n, 0.01), t=np.full(n, 5.0), p=np.full(n, 0.001),
        n_reference=np.full(n, 3), n=np.full(n, 3), significant=np.ones(n, bool),
        reference="WT", delta_crit=0.04)
    best = {("", "WT", "PA", "", 1, 10, 60.0): (2, -30.0),
            ("", "M", "PA", "", 1, 10, 60.0): (1, 5.0),
            ("", "WT", "PB", "", 5, 15, 60.0): (2, -30.0),
            ("", "M", "PB", "", 5, 15, 60.0): (2, -30.0)}
    fig = woods_populations_figure(k1, compare_population_counts(best, "WT"), "M", 60.0)
    top, bottom = fig.axes
    assert top.get_title(loc="left") == "K=1 fD"
    assert {c.get_gid() for c in bottom.collections if c.get_gid()} == {
        "count:reference:1-10", "count:both:5-15"}


def test_populations_are_drawn_as_the_k2_values_are():
    """The population figure is the residue figure with the populations: each sample's, where
    it keeps two (here WT only), marked by rank, the high-D one the complement."""
    from hxex.report import plot_residue_overlay

    gated = residue_map(uptake(("WT", "M")), sequences=SEQUENCE, two_population_peptides=True)
    choice = choose_populations(gated)
    fig = plot_residue_overlay(gated, "WT", "M", 60.0, show=("pop_low", "pop_high"),
                               choice=choice)
    ax = fig.axes[0]
    lines = {line.get_gid(): line for line in ax.lines if line.get_gid()}
    assert set(lines) == {"overlay:WT:pop_low", "overlay:WT:pop_high"}  # M keeps one
    low, high = lines["overlay:WT:pop_low"], lines["overlay:WT:pop_high"]
    assert low.get_marker() == "^" and high.get_marker() == "v"
    np.testing.assert_allclose(low.get_ydata() + high.get_ydata(), 1.0)
    assert "pop" in ax.get_ylabel()
    assert [t.get_text() for t in ax.get_legend().texts] == ["WT", "M"]
    with pytest.raises(ValueError, match="choice"):
        plot_residue_overlay(gated, "WT", "M", 60.0, show=("pop_low",))
    # the corrected-differently mark can be switched off
    plot_residue_overlay(gated, "WT", "M", 60.0, show=("pop_low",), choice=choice,
                         mark_corrected_differently=False)
