# SPDX-License-Identifier: GPL-3.0-or-later
"""MS1 protection factors: intrinsic rates in the protein, the per-time and fitted factors,
bounds, and the differences between samples, on uptake made from known factors."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex import chem
from hxex.analysis import SpeciesUptake
from hxex.analysis.protection import compare_protection, peptide_protection, peptide_rates
from hxex.kint import ExchangeConditions, intrinsic_rates

PROTEIN = "MAKLVGSTVEDRWQAFKLTH"
LABELLING = ExchangeConditions(7.0, 20.0, 0.9)
TIMES = (10.0, 100.0, 1000.0, 10000.0)


def rates_in_protein(start, end, nterm=1):
    """The rates of the counted amides of PROTEIN[start..end], from the whole protein."""
    full = intrinsic_rates(PROTEIN, LABELLING)
    return full.rate[full.at(range(start + nterm, end + 1))]


def uptake(entries, *, nterm=1, times=TIMES, se=0.01, corrected=True):
    """A species table from (sample, start, end, fD at each time); one row per time."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend")}
    for sample, start, end, fds in entries:
        peptide = PROTEIN[start - 1:end]
        n = len(peptide) - nterm
        for t, fd in zip(times, fds, strict=True):
            values = {"sample": sample, "peptide": peptide, "modification": "",
                      "kind": "peptide", "start": start, "end": end, "time": t, "K": 1,
                      "rank": 1, "mean": fd * n, "mean_se": se * n, "fD": fd, "fD_se": se,
                      "population": 1.0, "population_se": 0.0, "n_amides": n, "n_spectra": 3,
                      "n_charges": 1, "corrected": corrected, "charge_trend": False}
            for name, value in values.items():
                cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()}, nterm_subtract=nterm)


def model(k, protection, times=TIMES):
    k = np.asarray(k)[None, :] / np.asarray(protection)
    return np.mean(1 - np.exp(-np.asarray(times)[:, None] * k), axis=1)


def test_rates_are_those_of_the_protein():
    """A peptide's rates: the residue before it and no termini inside the protein."""
    for start, end in ((4, 12), (12, 20)):
        k, approx = peptide_rates(PROTEIN[start - 1:end], start, LABELLING,
                                  before=PROTEIN[start - 2], protein_end=end == len(PROTEIN))
        assert np.allclose(k, rates_in_protein(start, end)) and not approx.any()
    # without the residue before, the first amide has no rate (counted with nterm_subtract 0)
    k, _ = peptide_rates(PROTEIN[3:12], 4, LABELLING, nterm_subtract=0)
    assert np.isnan(k[0]) and np.allclose(k[1:], rates_in_protein(4, 12))
    # the protein's N-terminus: a counted free amine exchanges at once
    k, _ = peptide_rates(PROTEIN[:8], 1, LABELLING, nterm_subtract=0)
    assert np.isinf(k[0]) and np.allclose(k[1:], rates_in_protein(1, 8))
    # the peptide's own termini would be wrong: its last residue is not a carboxyl here
    alone = intrinsic_rates(PROTEIN[3:12], LABELLING, start=4)
    assert not np.isclose(alone.rate[-1], rates_in_protein(4, 12)[-1])


def test_protection_factors_are_recovered():
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("A", 2, 9, model(k1, 10.0)), ("A", 10, 20, model(k2, 1000.0))])
    table, by_time = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.log10_protection.tolist() == pytest.approx([1.0, 3.0], abs=1e-4)
    assert np.allclose(table.protection, [10.0, 1000.0], rtol=1e-3)
    assert (table.bound == "").all() and table.one_factor_fits.all()
    assert np.all(table.log10_protection_se < 0.05)
    # drift needs two times between the limits: protection 10 has one at these times
    assert np.isnan(table.protection_drift[0]) and abs(table.protection_drift[1]) < 1e-3
    # each time on its own gives the same factor where fD is between the limits
    inside = by_time.bound == ""
    first = by_time.where(start=2)
    assert np.allclose(by_time.log10_protection[first & inside], 1.0, atol=1e-4)
    assert (by_time.bound[first] != "").any()  # 10000 s: fully exchanged at protection 10
    # without the protein sequence: the residue before from the other peptides, same values
    # (the second peptide ends at the protein's C-terminus, which only the sequence tells)
    other, _ = peptide_protection(up, {"A": LABELLING})
    assert other.log10_protection[0] == pytest.approx(1.0, abs=1e-4)
    assert abs(other.log10_protection[1] - 3.0) > 0.05


def test_unequal_amides_make_the_factor_drift():
    """Half the amides exposed, half buried: the apparent factor rises with time and one factor
    does not fit."""
    k = rates_in_protein(2, 13)
    half = len(k) // 2
    fd = (model(k[:half], 1.0) * half + model(k[half:], 1e4) * (len(k) - half)) / len(k)
    table, _ = peptide_protection(uptake([("A", 2, 13, fd)]), LABELLING, sequences=PROTEIN)
    assert table.protection_drift[0] > 1.0 and not table.one_factor_fits[0]
    assert table.misfit[0] > 10


def test_bounds_instead_of_values():
    up = uptake([("A", 2, 9, [0.98, 0.99, 1.0, 1.0]), ("A", 10, 20, [0.0, 0.01, 0.02, 0.03])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.bound.tolist() == ["upper", "lower"]
    assert np.isnan(table.log10_protection).all() and np.isfinite(table.limit).all()
    # the limits are the factors that give the thresholds at the most telling time
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    assert model(k1, table.limit[0], times=(10.0,))[0] == pytest.approx(0.95, abs=1e-4)
    assert model(k2, table.limit[1], times=(10000.0,))[0] == pytest.approx(0.05, abs=1e-4)


def test_samples_left_out_say_why(caplog):
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0))])
    raw = uptake([("B", 2, 9, model(k, 10.0))], corrected=False)
    both = SpeciesUptake(**{name: np.concatenate([a, b]) for (name, a), b in zip(
        up.columns().items(), raw.columns().values(), strict=True)}, nterm_subtract=1)
    with caplog.at_level(logging.WARNING):
        table, _ = peptide_protection(both, {"A": LABELLING, "B": LABELLING}, sequences=PROTEIN)
    assert set(table.sample) == {"A"} and "not corrected" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        table, _ = peptide_protection(up, {"other": LABELLING}, sequences=PROTEIN)
    assert len(table) == 0 and "no labelling conditions" in caplog.text


def test_differences_between_samples():
    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("WT", 10, 20, model(k2, 100.0)),
                 ("M", 2, 9, model(k1, 1000.0)), ("M", 10, 20, [1.0, 1.0, 1.0, 1.0])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    cmp = compare_protection(table, "WT")
    assert cmp.sample.tolist() == ["M", "M"]
    assert cmp.delta[0] == pytest.approx(2.0, abs=1e-3) and cmp.significant[0]
    assert np.isnan(cmp.delta[1]) and cmp.note[1] == "sample upper bound"
    with pytest.raises(ValueError, match="match"):
        compare_protection(table, "WT", match="other")


def test_unknown_errors_take_the_median_of_the_known_ones():
    """A NaN standard error (no bootstrap) must not get the largest weight: it takes the
    median of the peptide's known errors; with none known the fit is unweighted."""
    from hxex.analysis.protection import _known_errors

    assert _known_errors(np.array([np.nan, np.nan])) is None
    assert _known_errors(np.array([0.02, np.nan, 0.04, 0.0])).tolist() == [0.02, 0.03, 0.04, 0.03]
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0) + np.array([0.0, 0.02, -0.02, 0.0]))])
    known, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    half = dataclasses.replace(up, fD_se=np.where(up.time == 100.0, np.nan, up.fD_se))
    table, _ = peptide_protection(half, LABELLING, sequences=PROTEIN)
    assert np.isfinite(table.misfit[0]) and table.misfit[0] == pytest.approx(known.misfit[0])
    assert table.log10_protection[0] == pytest.approx(known.log10_protection[0], abs=1e-6)
    none = dataclasses.replace(up, fD_se=np.full(len(up), np.nan))
    table, _ = peptide_protection(none, LABELLING, sequences=PROTEIN)
    assert np.isnan(table.misfit[0]) and np.isfinite(table.log10_protection_se[0])


def test_empty_peptides_leave_the_residue_fit(caplog):
    """A peptide with no finite fD is not a member of the design (its residues were reported
    resolved with a regulariser-only value); the peptide table keeps it with n_times 0."""
    from hxex.analysis import residue_protection

    up = residue_uptake("A", TRUE, [(2, 9), (3, 10), (2, 20)], times=TIMES_WIDE)
    gone = np.isin(up.start, [2]) & np.isin(up.end, [20])
    up = dataclasses.replace(up, fD=np.where(gone, np.nan, up.fD))
    with caplog.at_level(logging.INFO):
        out = residue_protection(up, LABELLING, sequences=PROTEIN)
    assert "1 peptide(s) without a finite fD" in caplog.text
    assert out.residue.max() == 10 and not np.isin(out.residue, range(11, 21)).any()
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    assert table.n_times[table.where(start=2, end=20)].tolist() == [0]


def test_modified_forms_are_states_of_their_own_in_the_residue_fit():
    """modified_peptides = 'separate': a phosphorylated form is fitted from its own peptides,
    with its sample's sequence and conditions; 'combined' pools it with the sample."""
    from hxex.analysis import residue_protection

    plain = residue_uptake("WT", TRUE, PEPTIDES, times=TIMES_WIDE)
    more = {r: v + 2.0 for r, v in TRUE.items()}
    mod = residue_uptake("WT", more, [(2, 9), (3, 11), (5, 13)], times=TIMES_WIDE)
    mod = dataclasses.replace(mod, peptide=np.array([p.replace("S", "pS", 1) for p in
                                                     mod.peptide]))
    both = SpeciesUptake(**{name: np.concatenate([x, y]) for (name, x), y in zip(
        plain.columns().items(), mod.columns().values(), strict=True)}, nterm_subtract=1)
    out = residue_protection(both, {"WT": LABELLING}, sequences={"WT": PROTEIN})
    states = list(dict.fromkeys(out.sample.tolist()))
    assert states == ["WT", "WT pS7"]
    own = out.where(sample="WT pS7") & np.isfinite(out.log10_protection)
    base = out.where(sample="WT") & out.resolved & np.isfinite(out.log10_protection)
    assert own.any() and np.median(out.log10_protection[own]) > np.median(
        out.log10_protection[base]) + 1.0  # the form's own factors, not the pooled ones
    pooled = residue_protection(both, {"WT": LABELLING}, sequences={"WT": PROTEIN},
                                modified="combined")
    assert set(pooled.sample) == {"WT"}
    with pytest.raises(ValueError, match="modified"):
        residue_protection(both, LABELLING, modified="together")


def test_two_reference_forms_over_one_range_do_not_collapse():
    k1 = rates_in_protein(2, 9)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("M", 2, 9, model(k1, 100.0))])
    mod = uptake([("WT", 2, 9, model(k1, 1000.0)), ("M", 2, 9, model(k1, 1000.0))])
    mod = dataclasses.replace(mod, peptide=np.array([p.replace("S", "pS", 1) for p in
                                                     mod.peptide]))
    both = SpeciesUptake(**{name: np.concatenate([x, y]) for (name, x), y in zip(
        up.columns().items(), mod.columns().values(), strict=True)}, nterm_subtract=1)
    table, _ = peptide_protection(both, LABELLING, sequences=PROTEIN)
    cmp = compare_protection(table, "WT")  # range matching: identical sequences pair
    assert len(cmp) == 2 and sorted(cmp.peptide.tolist()) == sorted(cmp.peptide_reference)
    deltas = dict(zip(cmp.peptide.tolist(), cmp.delta.tolist(), strict=True))
    assert deltas[PROTEIN[1:9]] == pytest.approx(1.0, abs=1e-3)
    assert deltas[PROTEIN[1:9].replace("S", "pS", 1)] == pytest.approx(0.0, abs=1e-3)
    assert len(compare_protection(table, "WT", match="sequence")) == 2


def test_a_falling_series_is_refused():
    """Back-exchange kinetics (fD falls with time) do not fit the forward model."""
    k = rates_in_protein(2, 9)
    up = uptake([("A", 2, 9, model(k, 10.0)[::-1]), ("A", 10, 20, [0.9, 0.6, 0.3, 0.1])])
    with pytest.raises(ValueError, match="falls with time.*two-direction"):
        peptide_protection(up, LABELLING, sequences=PROTEIN)
    mixed = uptake([("A", 2, 9, model(k, 10.0)), ("A", 10, 20, [0.9, 0.6, 0.3, 0.1])])
    peptide_protection(mixed, LABELLING, sequences=PROTEIN)  # not most of them: accepted


def test_modified_residue_counts_as_approximate():
    peptide = "AKLVGpSTVE"  # residues 2-11 with S7 phosphorylated
    k, approx = peptide_rates(peptide, 2, LABELLING, before="M")
    assert approx.sum() == 2 and np.isfinite(k).all()
    assert len(k) == len(chem.parse_sequence(peptide)) - 1


def test_protection_figures_and_settings(tmp_path):
    from hxex.analysis import AnalysisConfig
    from hxex.config import ProjectError, load_project
    from hxex.report import protection_difference_figure, protection_figure

    k1, k2 = rates_in_protein(2, 9), rates_in_protein(10, 20)
    up = uptake([("WT", 2, 9, model(k1, 10.0)), ("WT", 10, 20, model(k2, 100.0)),
                 ("M", 2, 9, model(k1, 1000.0)), ("M", 10, 20, [1.0, 1.0, 1.0, 1.0])])
    table, _ = peptide_protection(up, LABELLING, sequences=PROTEIN)
    fig = protection_figure(table, "M", length=20)
    (ax,) = [a for a in fig.axes if a.get_title()]
    assert ax.get_title() == "M, average (K=1)"
    hatched = [p.get_hatch() for p in ax.patches]
    assert sorted(hatched, key=str) == ["////", None]
    fig = protection_difference_figure(compare_protection(table, "WT"), "M")
    assert [a.get_title() for a in fig.axes if a.get_title()] == ["M - WT, average (K=1)"]
    with pytest.raises(ValueError, match="no protection"):
        protection_figure(table, "other")

    # settings: opt-in, refused without the labelling conditions
    assert not AnalysisConfig().protection
    with pytest.raises(ValueError, match="labelling"):
        AnalysisConfig(protection=True)
    cfg = AnalysisConfig(protection=True, labelling={"ph_read": 7.0, "temperature": 20.0,
                                                     "d2o_fraction": 0.9})
    assert cfg.labelling == LABELLING
    again = AnalysisConfig.from_dict(cfg.to_dict())
    assert again.labelling == LABELLING and again.protection
    # the residue level is opt-in on its own, and needs the peptide level
    assert not cfg.protection_residues and cfg.protection_max_se == 5.0
    with pytest.raises(ValueError, match="needs protection"):
        AnalysisConfig(protection_residues=True)
    project = tmp_path / "hxex.toml"
    project.write_text('project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'
                       '[analysis]\nprotection = true\n'
                       '[analysis.labelling.WT]\nph_read = 7.0\ntemperature = 20.0\n'
                       'd2o_fraction = 0.9\n')
    loaded = load_project(project).analysis
    assert loaded.protection and loaded.labelling == {"WT": LABELLING}
    # per run of [analysis.runs]: [analysis.labelling.<run>.<sample>]
    project.write_text(project.read_text() + '[analysis.runs]\nA = "a.zip"\n'
                       '[analysis.labelling.A.WT]\nph_read = 7.4\ntemperature = 20.0\n'
                       'd2o_fraction = 0.9\n')
    loaded = load_project(project).analysis
    assert loaded.labelling["A"]["WT"].ph_read == 7.4
    _rewrite(project, '[analysis.runs]\nA = "a.zip"', '[analysis.runs]\nB = "a.zip"')
    with pytest.raises(ProjectError, match="'A' is not a run"):
        load_project(project)


def _rewrite(path, old, new):
    path.write_text(path.read_text().replace(old, new))
    return path


def residue_uptake(sample, true, peptides, *, noise=0.0, seed=0, times=TIMES):
    """A species table of ``peptides`` ((start, end) of PROTEIN) made from per-residue log10
    protection factors ``true`` (residue -> value; others 0)."""
    rng = np.random.default_rng(seed)
    full = intrinsic_rates(PROTEIN, LABELLING)
    entries = []
    for start, end in peptides:
        am = np.arange(start + 1, end + 1)
        x = np.array([true.get(int(r), 0.0) for r in am])
        fd = np.mean(1 - np.exp(-np.outer(times, full.rate[full.at(am)] / 10**x)), axis=1)
        entries.append((sample, start, end, fd + rng.normal(0, noise, len(times))))
    return uptake(entries, times=times)


# overlapping peptides that resolve residues 6-15 one by one
PEPTIDES = [(2, 9), (2, 10), (3, 11), (4, 12), (5, 13), (5, 14), (6, 15), (6, 16), (7, 16),
            (9, 17), (10, 18), (2, 5), (2, 6), (2, 7), (2, 8), (11, 20), (12, 20)]
TRUE = {r: (3.0 if 8 <= r <= 13 else 1.5) for r in range(1, 21)}
TIMES_WIDE = (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)


def test_residue_protection_recovers_each_residue():
    from hxex.analysis import residue_protection

    up = residue_uptake("A", TRUE, PEPTIDES, noise=0.003, times=TIMES_WIDE)
    out = residue_protection(up, LABELLING, sequences=PROTEIN, r1=0.1)
    assert out.r1 == 0.1 and set(out.K) == {1}
    good = out.resolved & np.isfinite(out.log10_protection)
    assert good.sum() >= 8
    err = out.log10_protection[good] - np.array([TRUE[r] for r in out.residue[good]])
    assert np.sqrt(np.mean(err**2)) < 0.15
    assert np.all(np.abs(err) <= 4 * out.log10_protection_se[good])
    assert out.misfit[0] < 3
    assert np.all(out.n_peptides >= 1) and set(out.aa) <= set(PROTEIN)
    # the same data, run again: the same answer (the search is seeded)
    again = residue_protection(up, LABELLING, sequences=PROTEIN, r1=0.1)
    assert np.array_equal(again.log10_protection, out.log10_protection, equal_nan=True)


def test_residue_protection_compared_and_drawn():
    from hxex.analysis import compare_residue_protection, residue_protection
    from hxex.hooks import Cancelled
    from hxex.report import residue_protection_comparison_figure, residue_protection_figure

    more = {r: v + (1.0 if 8 <= r <= 13 else 0.0) for r, v in TRUE.items()}
    a = residue_uptake("WT", TRUE, PEPTIDES, times=TIMES_WIDE)
    b = residue_uptake("M", more, PEPTIDES, times=TIMES_WIDE)
    both = SpeciesUptake(**{name: np.concatenate([x, y]) for (name, x), y in zip(
        a.columns().items(), b.columns().values(), strict=True)}, nterm_subtract=1)
    seen = []
    out = residue_protection(both, LABELLING, sequences=PROTEIN,
                             progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (2, 2)
    cmp = compare_residue_protection(out, "WT")
    core = cmp.resolved & np.isin(cmp.residue, range(9, 13))
    assert core.any() and np.allclose(cmp.delta[core], 1.0, atol=0.2)
    assert cmp.significant[core].all() and not cmp.significant[~cmp.resolved].any()
    fig = residue_protection_figure(out, length=20)
    legend = fig.axes[0].get_legend()
    names = [t.get_text() for t in legend.texts]
    assert names[:2] == ["WT", "M"] and "upper bound" not in names  # only what is drawn
    assert [h.get_marker() for h in legend.legend_handles[:2]] == ["s", "o"]
    assert "misfit" not in fig.axes[0].get_title()
    # residues with a larger error than max_se are left out, and the panel says how many
    tight = residue_protection_figure(out, length=20, max_se=1e-6)
    notes = [t.get_text() for t in tight.axes[0].texts]
    assert any("not shown" in n for n in notes)
    assert not any(c.get_offsets().size for c in tight.axes[0].collections
                   if (c.get_gid() or "").startswith("protection:"))
    fig = residue_protection_comparison_figure(out, cmp, "M")
    assert any((p.get_gid() or "").startswith("significant:") for p in fig.axes[0].patches)
    with pytest.raises(Cancelled):
        residue_protection(a, LABELLING, sequences=PROTEIN, cancel=lambda: True)
    with pytest.raises(ValueError, match="r1"):
        residue_protection(a, LABELLING, r1=-1)
