# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment kinetics: the uptake table, ladder steps, scan-matched differences, decay fits,
per-residue least squares and the ladder figure, on synthetic ladders with known rates."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from hxex import chem
from hxex.analysis import (
    FragmentUptake,
    closure_table,
    closure_trace,
    difference_trace,
    fit_decay,
    fit_residues,
    fragment_trace,
    fragment_uptake,
    ladder_steps,
)
from hxex.fragments import fragment_amides, fragment_targets
from hxex.picking import PickerConfig
from hxex.report import ladder_kinetics_figure
from hxex.spectra import PeptideEntry

PARENT = PeptideEntry("f", "S", "GSHPLVFK", "11-18", 11, 18, 3)  # proline at residue 14
RATES = {12: 0.05, 13: 0.02, 15: 0.01, 16: 0.004, 17: 0.002, 18: 0.008}  # per residue, 1/s
TIMES = np.geomspace(5.0, 900.0, 120)


def residue_d(t, residue):
    return 0.9 * np.exp(-RATES[residue] * t) if residue in RATES else 0.0 * t


def synthetic(labels=("M:8", "c:1", "c:2", "c:4", "c:5", "c:6", "c:7", "z:1", "z:2", "z:3",
                      "z:4", "z:6", "z:7"), conditions=("gentle",), noise=0.0, seed=0):
    """A FragmentUptake whose fragments are the sums of their residues' decays."""
    rng = np.random.default_rng(seed)
    cols = {name: [] for name in ("sample", "condition", "parent", "parent_start",
                                  "parent_charge", "fragment", "ion_type", "length", "charge",
                                  "activation", "data_id", "rep", "time", "mean", "mean_se",
                                  "n_amides", "flagged", "spectrum_id")}
    for c_i, cond in enumerate(conditions):
        for d_i, label in enumerate(labels):
            held = fragment_amides(PARENT, label)
            ion, _, length = label.rpartition(":")
            for t in TIMES:
                cols["sample"].append("S")
                cols["condition"].append(cond)
                cols["parent"].append(PARENT.peptide)
                cols["parent_start"].append(PARENT.start_seq)
                cols["parent_charge"].append(3)
                cols["fragment"].append(label)
                cols["ion_type"].append(ion)
                cols["length"].append(int(length))
                cols["charge"].append(1)
                cols["activation"].append("ETD")
                cols["data_id"].append(c_i * 100 + d_i)
                cols["rep"].append(1)
                cols["time"].append(t)
                cols["mean"].append(sum(residue_d(t, r) for r in held) + noise * rng.normal())
                cols["mean_se"].append(0.01)
                cols["n_amides"].append(len(held))
                cols["flagged"].append(False)
                cols["spectrum_id"].append(f"p:S/f/{cond}/{label}/{t:.3f}")
    kinds = {"time": float, "mean": float, "mean_se": float, "flagged": bool}
    return FragmentUptake(**{k: np.array(v, dtype=kinds.get(k)) for k, v in cols.items()},
                          nterm_subtract=1)


def test_ladder_steps_name_each_residue_or_say_why_not():
    c = ladder_steps(PARENT, "c", [1, 2, 4, 5, 6, 7])
    got = [(s.larger, s.smaller, s.span, s.residues, s.note) for s in c]
    assert got == [
        ("c:1", None, (11, 12), (12,), ""),          # c1 holds residue 12's amide (c: next N-H)
        ("c:2", "c:1", (13, 13), (13,), ""),
        ("c:4", "c:2", (14, 15), (15,), ""),         # no c3 (before a proline): c4 - c2 adds
                                                     # P14 (no amide H) and L15
        ("c:5", "c:4", (16, 16), (16,), ""),
        ("c:6", "c:5", (17, 17), (17,), ""),
        ("c:7", "c:6", (18, 18), (18,), ""),
    ]
    z = ladder_steps(PARENT, "z", [1, 2, 3, 4, 6, 7])
    got = [(s.larger, s.smaller, s.span, s.residues, s.note) for s in z]
    assert got == [
        (None, None, (11, 11), (), "N-terminal"),
        (None, None, (12, 12), (), "no ion"),        # residue 12's amide is in no z ion
        ("z:7", "z:6", (13, 13), (13,), ""),
        ("z:6", "z:4", (14, 15), (15,), ""),         # z6 - z4: P14 and L15, one amide
        ("z:4", "z:3", (16, 16), (16,), ""),
        ("z:3", "z:2", (17, 17), (17,), ""),
        ("z:2", "z:1", (18, 18), (18,), ""),
    ]
    block = ladder_steps(PARENT, "c", [2, 5])
    assert (block[1].larger, block[1].residues, block[1].note) == ("c:5", (15, 16), "block")
    assert not block[1].single and block[0].single is False  # c2 alone holds 12 and 13
    with pytest.raises(ValueError):
        ladder_steps(PARENT, "M", [8])


def test_differences_follow_one_residue_and_its_rate_is_recovered():
    u = synthetic(noise=0.01)
    for step in ladder_steps(PARENT, "c", [1, 2, 4, 5, 6, 7]) + ladder_steps(
            PARENT, "z", [1, 2, 3, 4, 6, 7]):
        if not step.single or step.smaller is None:
            continue
        tr = difference_trace(u, step.larger, step.smaller, condition="gentle")
        assert len(tr) == len(TIMES)
        (residue,) = step.residues
        exact = difference_trace(synthetic(), step.larger, step.smaller)
        assert np.allclose(exact.mean, residue_d(exact.time, residue), atol=1e-12)
        f = fit_decay(tr, min_time=10.0)
        assert f.ok and f.resolved and f.n == int(np.sum(TIMES >= 10.0))
        assert f.rate == pytest.approx(RATES[residue], rel=0.15), (step, f)
        assert f.rate_se > 0 and np.all(np.isfinite(f.curve([10.0, 100.0])))
    # the standard errors add in quadrature; the difference with nothing is the fragment
    tr = difference_trace(u, "c:5", "c:4")
    assert np.allclose(tr.mean_se, np.hypot(0.01, 0.01))
    single = difference_trace(u, "c:5", None)
    assert np.allclose(single.mean, fragment_trace(u, "c:5").mean)


def test_differences_match_scans_within_their_condition():
    u = synthetic(conditions=("gentle", "harsh"))
    shifted = u.select(np.ones(len(u), dtype=bool))
    harsh = shifted.condition == "harsh"
    shifted.mean[harsh] += 1.0  # harsh fragments all 1 D higher: their differences unchanged
    for cond in ("gentle", "harsh"):
        tr = difference_trace(shifted, "c:5", "c:4", condition=cond)
        assert np.allclose(tr.mean, residue_d(tr.time, 16))
    # without a condition, scans pair only within their own condition (both averaged)
    both = difference_trace(shifted, "c:5", "c:4")
    assert len(both) == len(TIMES) and np.allclose(both.mean, residue_d(both.time, 16))


def test_decay_fit_edge_cases():
    t = np.geomspace(1, 1000, 50)
    rise = SimpleNamespace(time=t, mean=-0.8 * np.exp(-0.01 * t) + 0.8, mean_se=t * 0)
    from hxex.analysis import Trace

    f = fit_decay(Trace(rise.time, rise.mean, rise.mean_se))
    assert f.ok and f.rate == pytest.approx(0.01, rel=1e-4) and f.amplitude < 0
    few = fit_decay(Trace(t[:3], t[:3], t[:3]))
    assert not few.ok and np.isnan(few.rate)
    slow = fit_decay(Trace(t, 2.0 - 1e-4 * t, t * 0))  # a straight line: no decay in range
    assert slow.ok and not slow.resolved
    fast = fit_decay(Trace(t[10:], 0.5 * np.exp(-1.0 * t[10:]), t[10:] * 0))
    assert not fast.resolved  # over before the first point
    assert not fit_decay(Trace(t, t * 0, t * 0), min_time=2000.0).ok


def test_residues_by_least_squares():
    values = {label: sum(0.7 if r in (12, 15) else 0.2 for r in fragment_amides(PARENT, label))
              for label in ("c:1", "c:2", "c:4", "c:5", "M:8")}
    fit = fit_residues(PARENT, values)
    by = dict(zip(fit.residue.tolist(), fit.value.tolist(), strict=True))
    assert by[12] == pytest.approx(0.7) and by[13] == pytest.approx(0.2)
    assert by[15] == pytest.approx(0.7) and by[16] == pytest.approx(0.2)
    ident = dict(zip(fit.residue.tolist(), fit.identifiable.tolist(), strict=True))
    assert ident == {12: True, 13: True, 15: True, 16: True, 17: False, 18: False}
    assert by[17] + by[18] == pytest.approx(0.4)  # only their sum is determined
    assert np.allclose(fit.residual, 0.0, atol=1e-6)
    over = fit_residues(PARENT, {"c:1": 1.5})  # bounded: at most 1 per amide
    assert over.value[0] == pytest.approx(1.0) and over.residual[0] == pytest.approx(0.5)


class _Fit:
    def __init__(self, means, data_id, time, ids, role=None):
        self.fits = {1: SimpleNamespace(means=np.asarray(means, float)[:, None])}
        self.data_id, self.time, self.spectrum_id = data_id, time, ids
        self.role = np.full(len(ids), "", dtype="<U8") if role is None else np.asarray(role)

    def __len__(self):
        return len(self.time)


def test_fragment_uptake_from_a_run():
    """From a run's records: the targets' fragment labels, conditions and parents."""
    targets = (fragment_targets(PARENT, "M c:2,4 z:3", activation="ETD", label="gentle")
               + fragment_targets(PARENT, "c:4", activation="ETD", filter_contains="sid"))
    source = {"reader": "mzml", "targets": [t.to_dict() for t in targets],
              "series": [{"data_id": i, "sample": "S", "target": i}
                         for i in range(len(targets))]}
    n = len(targets)
    ids = np.array([f"p:S/x/{i}" for i in range(n)])
    results = SimpleNamespace(
        settings=SimpleNamespace(spectra_source=source, picker=PickerConfig(nterm_subtract=1)),
        fit=_Fit(np.arange(n) * 0.5, np.arange(n), np.full(n, 30.0), ids),
        binned=SimpleNamespace(spectrum_id=ids, sample=np.full(n, "S"), rep=np.ones(n, int),
                               charge=np.array([t.entry.charge for t in targets])),
        review=None, quality=None, bootstrap_for=lambda i, k: None)
    u = fragment_uptake(results)
    assert len(u) == n and set(u.condition) == {"gentle", "ETD +sid"}
    assert list(u.fragment[:4]) == ["M:8", "M:8", "M:8", "c:2"]
    c4 = u.where(fragment="c:4", condition="gentle", charge=1)
    assert u.n_amides[c4][0] == len(fragment_amides(PARENT, "c:4")) == 3
    assert np.isnan(u.mean_se).all() and u.parent_entry().start_seq == 11
    assert u.select(c4).time.tolist() == [30.0]
    # the fully deuterated control is not a point of the kinetics
    results.fit.role = np.full(n, "fulldeut")
    assert len(fragment_uptake(results)) == 0
    with pytest.raises(ValueError, match="mzML"):
        fragment_uptake(SimpleNamespace(settings=SimpleNamespace(spectra_source=None)))


def test_ladder_figure_has_a_row_per_step():
    u = synthetic(conditions=("gentle", "harsh"))
    fig = ladder_kinetics_figure(u, "c", labels={"gentle": "G"}, min_time=10.0)
    axes = fig.axes
    steps = ladder_steps(PARENT, "c", [1, 2, 4, 5, 6, 7])
    assert len(axes) == 2 * (len(steps) + 1)
    titles = [ax.get_title() for ax in axes[::2]]
    assert titles[0] == "S (Z3)  ETD M_z1"
    assert titles[1].endswith("G(S)-12") and titles[3].endswith("GSHP(L)-15")
    rate_labels = [t.get_text() for t in axes[2 * 4 + 1].get_legend().get_texts()]
    assert rate_labels[0].startswith("D G, rate: 4.00e-03")  # residue 16 (V), exact data
    assert all(ax.get_xscale() == "log" for ax in axes)
    assert len({ax.get_xlim() for ax in axes}) == 1
    # traces of one's own in the right column, alone
    seen = []

    def extra(step):
        seen.append(None if step is None else step.larger)
        if step is None:
            return [("ratio", fragment_trace(u, "c:5", condition="gentle"), False)]
        if step.single and step.smaller is not None:
            tr = difference_trace(u, step.larger, step.smaller, condition="gentle")
            return [("mine", tr, True)]
        return []

    fig = ladder_kinetics_figure(u, "c", min_time=10.0, right_traces=extra,
                                 right_defaults=False)
    assert seen[0] is None and "c:5" in seen
    right = fig.axes[1::2]
    assert [t.get_text() for t in right[0].get_legend().get_texts()] == ["ratio"]
    texts = [t.get_text() for t in right[4].get_legend().get_texts()]
    assert texts == ["mine, rate: 4.00e-03±0.00e+00 s⁻¹"] or (
        len(texts) == 1 and texts[0].startswith("mine, rate: 4.00e-03"))
    with pytest.raises(ValueError, match="choose one"):
        both = synthetic()
        both = both.select(np.ones(len(both), dtype=bool))
        both.sample[: len(both) // 2] = "T"
        ladder_kinetics_figure(both, "c")
    assert chem.ion_terminus("c") == "N"


def test_closure_of_complementary_ions():
    """c_k + z_(n-k) and a_k + x_(n-k) hold the precursor's amides between them: the synthetic
    ladder closes exactly, and a fragment fitted 0.4 D too high is flagged."""
    u = synthetic(conditions=("gentle", "harsh"))
    tr = closure_trace(u, "c:4", "z:4", precursor_charge=1, condition="gentle")
    assert len(tr) == len(TIMES) and np.allclose(tr.mean, 0.0, atol=1e-12)
    assert np.allclose(tr.mean_se, np.sqrt(3) * 0.01)
    # the default precursor is the unreacted one (charge 3), which this table does not have
    assert len(closure_trace(u, "c:4", "z:4", condition="gentle")) == 0
    off = u.select(np.ones(len(u), dtype=bool))
    off.mean[(off.fragment == "z:3") & (off.condition == "harsh")] += 0.4
    table = closure_table(off, "c", "z", precursor_charge=1, min_time=10.0)
    pairs = {(c, k) for c, k in zip(table.condition, table.k, strict=True)}
    assert pairs == {(c, k) for c in ("gentle", "harsh") for k in (1, 2, 4, 5, 6, 7)}
    flagged = {(c, int(k)) for c, k, f in zip(table.condition, table.k, table.flagged,
                                              strict=True) if f}
    assert flagged == {("harsh", 5)}  # c5 + z3
    assert np.all(table.n == np.sum(TIMES >= 10.0))
    worst = table.median[table.flagged][0]
    assert worst == pytest.approx(0.4) and table.spread[table.flagged][0] == pytest.approx(0.0)
    ax = synthetic(labels=("M:8", "a:3", "x:5", "b:5", "y:3"))
    assert np.allclose(closure_trace(ax, "a:3", "x:5", precursor_charge=1).mean, 0.0)
    assert np.allclose(closure_trace(ax, "b:5", "y:3", precursor_charge=1).mean, 0.0)
    for bad in [("c:4", "y:4"), ("z:4", "c:4"), ("c:3", "z:4")]:
        with pytest.raises(ValueError):
            closure_trace(u, *bad)


def stub_run(conditions=("gentle", "harsh"), parent=None):
    """A fitted fragment run (the objects fragment_uptake reads) whose fragments are the sums of
    their residues' decays, every fragment of one condition from the same scans."""
    from hxex.analysis import fragment_uptake as _  # noqa: F401 (the reader under test)

    targets = []
    for cond in conditions:
        targets += fragment_targets(parent or PARENT, "M c:1,2,4,5,6,7:1 z:1,2,3,4,6,7:1",
                                    activation="ETD", label=cond)
    source = {"reader": "mzml", "targets": [t.to_dict() for t in targets],
              "series": [{"data_id": i, "sample": "S", "target": i}
                         for i in range(len(targets))]}
    data_id, time, means, ids, charge = [], [], [], [], []
    for i, t in enumerate(targets):
        label = t.entry.extra["fragment"]
        held = fragment_amides(PARENT, label)
        for x in TIMES:
            data_id.append(i)
            time.append(x)
            means.append(sum(residue_d(x, r) for r in held))
            ids.append(f"p:S/f/{i}/{x:.4f}")
            charge.append(t.entry.charge)
    n = len(time)
    ids = np.array(ids)
    return SimpleNamespace(
        settings=SimpleNamespace(spectra_source=source, picker=PickerConfig(nterm_subtract=1)),
        fit=_Fit(np.array(means), np.array(data_id), np.array(time), ids),
        binned=SimpleNamespace(spectrum_id=ids, sample=np.full(n, "S"), rep=np.ones(n, int),
                               charge=np.array(charge)),
        review=None, quality=None, bootstrap_for=lambda i, k: None)


def test_ladder_rates_recover_each_residue():
    from hxex.analysis import ladder_rates

    u = fragment_uptake(stub_run())
    rates = ladder_rates(u, min_time=10.0)
    assert set(rates.ion_type) == {"c", "z"} and set(rates.condition) == {"gentle", "harsh"}
    c = rates.where(ion_type="c", condition="gentle")
    assert rates.residue[c].tolist() == [12, 13, 15, 16, 17, 18]  # c1 alone is residue 12
    assert rates.smaller[c][0] == "" and rates.aa[c].tolist() == list("SHLVFK")
    for r, k, ok in zip(rates.residue, rates.rate, rates.resolved, strict=True):
        assert ok and k == pytest.approx(RATES[r], rel=1e-3)
    assert len(ladder_rates(u, ion_types=["z"], charges=(2,))) == 0
    assert rates.kint is None and "kint" not in rates.columns()
    # with the kinetics' solvent: the residue's intrinsic rate (from the parent) and kint / rate
    from hxex.kint import ExchangeConditions, intrinsic_rates

    cond = ExchangeConditions(7.0, 20.0, 0.05)
    with_k = ladder_rates(u, min_time=10.0, conditions=cond)
    parent = str(with_k.parent[0])
    ref = intrinsic_rates(parent, cond, start=int(with_k.parent_start[0]))
    assert np.allclose(with_k.kint, ref.rate[ref.at(with_k.residue)])
    assert np.allclose(with_k.protection, with_k.kint / with_k.rate)
    assert np.allclose(with_k.rate, rates.rate)
    assert "protection" in with_k.columns()
    # per sample: a sample without conditions gets NaN
    other = ladder_rates(u, min_time=10.0, conditions={"other": cond})
    assert np.all(np.isnan(other.kint))
    assert np.allclose(ladder_rates(u, min_time=10.0, conditions={"S": cond}).kint,
                       with_k.kint)


def test_fragment_analysis_settings():
    from hxex.analysis import AnalysisConfig, FragmentAnalysis

    cfg = AnalysisConfig.from_dict({"fragments": {"min_time": 20.0, "labels": {"gentle": "G"},
                                                  "ion_types": ["c-1"]}})
    assert cfg.fragments == FragmentAnalysis(ion_types=("c-1",), min_time=20.0,
                                             labels={"gentle": "G"})
    assert AnalysisConfig.from_dict(cfg.to_dict()) == cfg
    assert AnalysisConfig.from_dict({"fragments": False}).fragments is None
    assert AnalysisConfig().fragments == FragmentAnalysis()
    import tomllib
    from pathlib import Path

    from hxex.config import _analysis_from

    toml = tomllib.loads('[analysis.fragments]\ncharges = [1, 2]\nmin_time = 20.0\n'
                        '[analysis.fragments.labels]\ngentle = "G"\n')
    parsed = _analysis_from(toml["analysis"], Path("."))
    assert parsed.fragments.charges == (1, 2) and parsed.fragments.labels == {"gentle": "G"}
    assert parsed.fragments.exchange is None
    from hxex.kint import ExchangeConditions

    toml = tomllib.loads('[analysis.fragments.exchange]\nph_read = 2.5\ntemperature = 0.0\n'
                        'd2o_fraction = 0.05\n[analysis.fragments.exchange.factors]\n'
                        'pS = [-0.9, -0.12, 0.69, 0.6]\n')
    parsed = _analysis_from(toml["analysis"], Path("."))
    assert parsed.fragments.exchange == ExchangeConditions(
        2.5, 0.0, 0.05, factors={"pS": (-0.9, -0.12, 0.69, 0.6)})
    assert not parsed.fragments.kint  # conditions alone compute nothing: kint is opt-in
    assert AnalysisConfig.from_dict(parsed.to_dict()) == parsed
    toml = tomllib.loads('[analysis.fragments]\nkint = true\n'
                        '[analysis.fragments.exchange.S]\nph_read = 2.8\ntemperature = 20\n'
                        'd2o_fraction = 0.05\ninitial_fraction = 0.9\n')
    parsed = _analysis_from(toml["analysis"], Path("."))
    assert parsed.fragments.kint and parsed.fragments.exchange == {
        "S": ExchangeConditions(2.8, 20.0, 0.05, initial_fraction=0.9)}
    assert AnalysisConfig.from_dict(parsed.to_dict()) == parsed
    with pytest.raises(ValueError, match="kint needs"):
        FragmentAnalysis(kint=True)
    for bad in ({"ion_types": ["M"]}, {"charges": [0]}, {"tolerance": 0}, {"bogus": 1}):
        with pytest.raises(ValueError):
            FragmentAnalysis.from_dict(bad)


def test_write_analysis_of_a_fragment_run(tmp_path):
    from hxex.analysis import AnalysisConfig, FragmentAnalysis
    from hxex.report import closure_figure, residue_rates_figure, write_analysis

    run = stub_run()
    cfg = AnalysisConfig(format="png", fragments=FragmentAnalysis(min_time=10.0,
                                                                  labels={"gentle": "G"}))
    written = write_analysis(run, cfg, tmp_path)
    names = sorted(p.name for p in written)
    assert names == sorted([
        "fragment_uptake.csv", "ladder_rates.csv", "ladder_residues.csv", "closure.csv",
        "ladder_S_11-18_z3_c_z1.png", "ladder_S_11-18_z3_z_z1.png",
        "closure_S_11-18_z3_c+z.png", "residue_rates_S_11-18_z3.png"])
    assert all(p.stat().st_size > 0 for p in written)
    # without the fragment outputs a fragment-only run has nothing else to write
    assert write_analysis(run, AnalysisConfig(fragments=None), tmp_path / "none") == []

    u = fragment_uptake(run)
    fig = closure_figure(u, "c", "z", precursor_charge=None, tolerance=0.1, min_time=10.0)
    assert [ax.get_title() for ax in fig.axes] == ["S (Z3)  gentle", "S (Z3)  harsh"]
    # the stub's precursor ion is read at charge 1, so give it as the precursor's charge
    fig = closure_figure(u, "c", "z", precursor_charge=1)
    pts = fig.axes[0].collections
    assert pts and np.allclose(np.concatenate([c.get_offsets()[:, 1] for c in pts]), 0.0)
    from hxex.analysis import ladder_rates

    rates = ladder_rates(u, min_time=10.0)
    fig = residue_rates_figure(rates, labels={"gentle": "G"})
    top, bottom = fig.axes
    assert [t.get_text() for t in top.get_legend().get_texts()] == [
        "G c", "G z", "harsh c", "harsh z"]
    assert bottom.get_xticklabels()[1].get_text() == "S12"
    with pytest.raises(ValueError, match="no c and y"):
        closure_figure(u, "c", "y")


def test_intrinsic_rates_in_the_fragment_outputs(tmp_path):
    from hxex.analysis import AnalysisConfig, FragmentAnalysis, ladder_rates
    from hxex.kint import ExchangeConditions, intrinsic_rates
    from hxex.report import ladder_kinetics_figure, residue_rates_figure, write_analysis

    run = stub_run()
    cond = ExchangeConditions(2.8, 20.0, 0.05, initial_fraction=0.9)
    cfg = AnalysisConfig(format="png", fragments=FragmentAnalysis(
        min_time=10.0, kint=True, exchange={"S": cond}))
    written = write_analysis(run, cfg, tmp_path)
    header = (tmp_path / "ladder_rates.csv").read_text().splitlines()[0]
    assert header.endswith("kint,protection,log10_protection,log10_protection_se,merged")
    assert len(written) == 8
    # without the opt-in the same conditions change nothing
    off = AnalysisConfig(format="png", fragments=FragmentAnalysis(min_time=10.0,
                                                                  exchange={"S": cond}))
    write_analysis(run, off, tmp_path / "off")
    assert "kint" not in (tmp_path / "off" / "ladder_rates.csv").read_text().splitlines()[0]

    u = fragment_uptake(run)
    rates = ladder_rates(u, min_time=10.0, conditions=cond)
    fig = residue_rates_figure(rates)
    assert len(fig.axes) == 3 and fig.axes[2].get_ylabel() == "protection factor"
    kint = intrinsic_rates(str(u.parent[0]), cond, start=int(u.parent_start[0]))
    fig = ladder_kinetics_figure(u, "c", kint=kint)
    dashed = [line for ax in fig.axes for line in ax.get_lines() if line.get_linestyle() == "--"
              and line.get_label().startswith("intrinsic")]
    labels = [line.get_label() for line in dashed]
    assert "intrinsic, all amides" in labels and sum("k = " in x for x in labels) >= 5
    # without a starting fraction there is no curve
    bare = intrinsic_rates(str(u.parent[0]), ExchangeConditions(2.8, 20.0, 0.05),
                           start=int(u.parent_start[0]))
    fig = ladder_kinetics_figure(u, "c", kint=bare)
    assert not [line for ax in fig.axes for line in ax.get_lines()
                if line.get_label().startswith("intrinsic")]


def test_decay_fit_in_a_late_window_and_protection_only_when_resolved():
    from hxex.analysis import Trace, ladder_rates
    from hxex.kint import ExchangeConditions

    t = np.linspace(2000.0, 3000.0, 60)
    f = fit_decay(Trace(t, 0.7 * np.exp(-0.0012 * t) + 0.1, t * 0))
    assert f.ok and f.resolved and f.rate == pytest.approx(0.0012, rel=1e-4)
    assert f.amplitude == pytest.approx(0.7, rel=1e-3) and f.offset == pytest.approx(0.1, abs=1e-4)
    np.testing.assert_allclose(f.curve(t), 0.7 * np.exp(-0.0012 * t) + 0.1, atol=1e-6)
    # fitted late, a fast residue's decay is over before the first point: no protection factor
    u = fragment_uptake(stub_run(conditions=("gentle",)))
    late = ladder_rates(u, min_time=100.0, conditions=ExchangeConditions(7.0, 20.0, 0.05))
    assert (~late.resolved).any() and late.resolved.any()
    assert late.residue[~late.resolved].min() == 12
    assert np.isnan(late.protection[~late.resolved]).all()
    ok = late.resolved
    np.testing.assert_allclose(late.protection[ok], late.kint[ok] / late.rate[ok])


def test_fragment_outputs_say_why_they_are_missing(caplog):
    import logging

    from hxex.analysis import FragmentAnalysis
    from hxex.report.fragments import fragment_outputs

    run = stub_run(conditions=("gentle",))
    cfg = FragmentAnalysis(min_time=10.0)
    assert fragment_outputs(run, cfg)[0].keys() == {"fragment_uptake", "ladder_rates",
                                                     "ladder_residues", "closure"}
    no_source = SimpleNamespace(**{**vars(run), "settings": SimpleNamespace(
        spectra_source=None, picker=run.settings.picker)})
    assert fragment_outputs(no_source, cfg) == ({}, [])
    no_k1 = SimpleNamespace(**{**vars(run), "fit": SimpleNamespace(**{**vars(run.fit),
                                                                       "fits": {}})})
    with caplog.at_level(logging.WARNING, logger="hxex.report.fragments"):
        assert fragment_outputs(no_k1, cfg) == ({}, [])
    assert "needs the K = 1 fits" in caplog.text
    # a folder-format run in a RunSet has no fragments and is skipped, not an error
    from hxex.analysis import RunSet

    folder = SimpleNamespace(**{**vars(run), "settings": SimpleNamespace(
        spectra_source={"reader": "specexport"}, picker=run.settings.picker)})
    both = RunSet({"ms1": folder, "frag": run})
    assert set(fragment_uptake(both).run) == {"frag"}
    assert fragment_outputs(both, cfg)[0].keys() == {"fragment_uptake", "ladder_rates",
                                                      "ladder_residues", "closure"}
    assert fragment_outputs(RunSet({"ms1": folder}), cfg) == ({}, [])
    # once there are fragment series, any other error is raised, not swallowed
    other = stub_run(conditions=("gentle",))
    other.settings.picker = PickerConfig(nterm_subtract=2)
    with pytest.raises(ValueError, match="count fragment amides differently"):
        fragment_outputs(RunSet({"a": run, "b": other}), cfg)
    # the closure warning fires for a single run as well as for several
    from hxex.io.mzml import mzml_series_targets

    off = stub_run(conditions=("gentle",))
    targets = mzml_series_targets(off.settings.spectra_source)
    z3 = [i for i, d in enumerate(off.fit.data_id)
          if targets[int(d)].entry.extra["fragment"] == "z:3"]
    off.fit.fits[1].means[z3, 0] += 0.4
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hxex.report.fragments"):
        fragment_outputs(off, cfg)
    assert "miss the precursor by more than" in caplog.text


def test_fragment_outputs_of_several_runs(tmp_path):
    """Runs of one sample (two activations, say) side by side, never pooled; unmerged runs
    each on their own."""
    from hxex.analysis import AnalysisConfig, FragmentAnalysis, RunSet, ladder_rates
    from hxex.report import residue_rates_figure, write_analysis

    runs = {"etd": stub_run(conditions=("gentle",)), "ethcd": stub_run(conditions=("gentle",))}
    # EThcD of the 2+ precursor: its own figures, but one residue-rate figure for both
    runs_z = {**runs, "ethcd": stub_run(conditions=("gentle",),
                                        parent=dataclasses.replace(PARENT, charge=2))}
    merged = RunSet(runs, merge={"etd:S": "S", "ethcd:S": "S"})
    u = fragment_uptake(merged)
    # the rows keep their run; the merged sample is one state (state_run ''), the condition
    # stays as recorded and the figures name it run:condition where both runs have it
    assert set(u.sample) == {"S"} and set(u.condition) == {"gentle"} and u.merged.all()
    assert set(u.run) == {"etd", "ethcd"} and len(u) == 2 * len(fragment_uptake(runs["etd"]))
    assert u.states() == [("", "S")]
    assert u.select(u.where(run="etd")).run.tolist() == ["etd"] * (len(u) // 2)
    rates = ladder_rates(u, min_time=10.0)
    single = ladder_rates(fragment_uptake(runs["etd"]), min_time=10.0)
    assert len(rates) == 2 * len(single)
    assert np.array_equal(rates.n[rates.where(run="etd")], single.n)  # not pooled
    fig = residue_rates_figure(rates)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "etd:gentle c", "etd:gentle z", "ethcd:gentle c", "ethcd:gentle z"]

    # a run's own labels under its name; a plain label is prefixed by the run where needed
    named = FragmentAnalysis(labels={"gentle": "G", "etd": {"gentle": "ETD G"}})
    fig = residue_rates_figure(rates, labels=named.labels)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "ETD G c", "ETD G z", "ethcd:G c", "ethcd:G z"]
    with pytest.raises(ValueError, match="'uvpd' is not a run"):
        AnalysisConfig(runs={"etd": "a"}, fragments=FragmentAnalysis(
            labels={"uvpd": {"gentle": "U"}}))

    cfg = AnalysisConfig(format="png", fragments=FragmentAnalysis(min_time=10.0))
    written = write_analysis(merged, cfg, tmp_path / "merged")
    assert sorted(p.name for p in written) == sorted([
        "fragment_uptake.csv", "ladder_rates.csv", "ladder_residues.csv", "closure.csv",
        "ladder_S_11-18_z3_c_z1.png", "ladder_S_11-18_z3_z_z1.png",
        "closure_S_11-18_z3_c+z.png", "residue_rates_S_11-18_z3.png"])
    for name in ("fragment_uptake", "ladder_rates", "ladder_residues", "closure"):
        head = (tmp_path / "merged" / f"{name}.csv").read_text().splitlines()[0]
        assert head.startswith("run,sample,"), head

    # unmerged: a state per run (shown as run:sample), the conditions as they are, figures
    # per run
    plain = fragment_uptake(RunSet(runs))
    assert set(plain.sample) == {"S"} and set(plain.condition) == {"gentle"}
    assert plain.states() == [("etd", "S"), ("ethcd", "S")] and not plain.merged.any()
    names = {p.name for p in write_analysis(RunSet(runs), cfg, tmp_path / "plain")}
    assert {"residue_rates_etd_S_11-18_z3.png", "residue_rates_ethcd_S_11-18_z3.png"} <= names

    both = RunSet(runs_z, merge={"etd:S": "S", "ethcd:S": "S"})
    names = {p.name for p in write_analysis(both, cfg, tmp_path / "charges")}
    assert {"ladder_S_11-18_z2_c_z1.png", "ladder_S_11-18_z3_c_z1.png",
            "residue_rates_S_11-18.png"} <= names
    assert not any(n.startswith("residue_rates_S_11-18_z") for n in names)
    fig = residue_rates_figure(ladder_rates(fragment_uptake(both), min_time=10.0))
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        "etd:gentle Z3 c", "etd:gentle Z3 z", "ethcd:gentle Z2 c", "ethcd:gentle Z2 z"]
    assert "(Z2/3)" in fig.axes[0].get_title()

    # runs that count amides differently cannot be compared
    other = stub_run(conditions=("gentle",))
    other.settings.picker = PickerConfig(nterm_subtract=2)
    with pytest.raises(ValueError, match="count fragment amides differently"):
        fragment_uptake(RunSet({"a": runs["etd"], "b": other}))


def test_the_fragment_analysis_refuses_times_that_only_order_the_points(tmp_path):
    from hxex.analysis import AnalysisConfig, FragmentAnalysis
    from hxex.report import write_analysis

    run = stub_run()
    run.time_scale = "order"  # as hxex.report.Results.time_scale reads it from the bundle
    with pytest.raises(ValueError, match="only order the points"):
        write_analysis(run, AnalysisConfig(fragments=FragmentAnalysis()), tmp_path)


def test_ladder_protection_errors_and_precise_steps():
    from hxex.analysis import ladder_rates
    from hxex.kint import ExchangeConditions

    u = synthetic(noise=0.01, seed=1)
    cond = ExchangeConditions(7.0, 20.0, 0.05)
    rates = ladder_rates(u, min_time=10.0, conditions=cond)
    ok = np.isfinite(rates.protection)
    assert np.allclose(rates.log10_protection[ok], np.log10(rates.protection[ok]))
    assert np.allclose(rates.log10_protection_se[ok],
                       rates.rate_se[ok] / (rates.rate[ok] * np.log(10)))
    assert np.array_equal(rates.precise, rates.resolved & (rates.rate_se / rates.rate <= 0.5))
    assert rates.max_rate_error == 0.5
    strict = ladder_rates(u, min_time=10.0, max_rate_error=1e-9)
    assert not strict.precise.any() and "log10_protection" not in strict.columns()
    with pytest.raises(ValueError, match="max_rate_error"):
        ladder_rates(u, max_rate_error=0)
    # the fitted curve's error: about the scatter over the square root of the points, and
    # in a band around the curve in the ladder figure
    from hxex.analysis import difference_trace, fit_decay

    tr = difference_trace(u, "c:5", "c:4", condition="gentle", charge=1)
    f = fit_decay(tr, min_time=10.0)
    se = f.curve_se(tr.time)
    assert np.all(np.isfinite(se)) and 0.001 < float(np.median(se)) < 0.01
    assert np.all(np.isnan(dataclasses.replace(f, covariance=None).curve_se(tr.time)))
    from matplotlib.collections import PolyCollection

    from hxex.report import ladder_kinetics_figure

    fig = ladder_kinetics_figure(u, "c", min_time=10.0)
    assert any(isinstance(c, PolyCollection) for ax in fig.axes for c in ax.collections)
    # the amplitude's error, and error bars in every panel of the residue-rates figure
    from matplotlib.container import ErrorbarContainer

    from hxex.report import residue_rates_figure

    assert np.all(np.isfinite(rates.amplitude_se[rates.resolved]))
    assert np.all(rates.amplitude_se[rates.resolved] < 0.1)
    fig = residue_rates_figure(rates)
    assert all(any(isinstance(c, ErrorbarContainer) for c in ax.containers) for ax in fig.axes)


def test_ladder_residues_combine_the_two_ladders_within_a_condition():
    from hxex.analysis import ladder_rates, ladder_residues
    from hxex.kint import ExchangeConditions

    u = synthetic(conditions=("gentle", "harsh"), noise=0.005, seed=2)
    cond = ExchangeConditions(7.0, 20.0, 0.05)
    rates = ladder_rates(u, min_time=10.0, conditions=cond)
    res = ladder_residues(rates)
    # gentle and harsh are never pooled: one row per condition and residue
    for c in ("gentle", "harsh"):
        rows = res.where(condition=c)
        assert sorted(res.residue[rows].tolist()) == sorted(
            set(rates.residue[rates.where(condition=c) & rates.precise].tolist()))
    both = res.n == 2
    assert both.any() and (res.n_ion[both] == "c").all() and (res.c_ion[both] == "z").all()
    for r, k in zip(res.residue, res.rate, strict=True):
        assert k == pytest.approx(RATES[r], rel=0.1)
    assert set(res.agreement[both]) <= {"agree", "disagree"}
    assert (res.agreement[res.n == 1] == "single").all()
    one = res.where(residue=12)  # c1 alone: no z step names residue 12
    assert res.c_ion[one].tolist() == ["", ""]
    assert np.allclose(res.log10_protection, np.log10(res.kint) - res.log10_rate)
    # a disagreement between the ladders
    i, j = np.flatnonzero(rates.where(condition="gentle", residue=16) & rates.precise)[:2]
    fast = rates.rate.copy()
    fast[j] = rates.rate[i] * 10.0
    bad = ladder_residues(dataclasses.replace(rates, rate=fast, rate_se=fast * 0.05))
    row = bad.where(condition="gentle", residue=16)
    assert bad.agreement[row].tolist() == ["disagree"] and bad.misfit[row][0] > 10
    # fragment charges are not combined; two ion types of one terminus are alternatives
    z2 = dataclasses.replace(rates, charge=np.where(rates.ion_type == "z", 2, rates.charge))
    split = ladder_residues(z2)
    assert (split.n == 1).all() and set(split.charge) == {1, 2}
    variant = dataclasses.replace(rates, ion_type=np.where(
        (rates.ion_type == "c") & (rates.condition == "harsh"), "c-1", rates.ion_type),
        condition=np.full(len(rates), "gentle"))
    rows = ladder_residues(variant).where(residue=16)
    pairs = sorted(zip(ladder_residues(variant).n_ion[rows], ladder_residues(variant).c_ion[rows],
                       strict=True))
    assert pairs == [("c", "z"), ("c-1", "z")]
    # without conditions: no kint columns
    plain = ladder_residues(ladder_rates(u, min_time=10.0))
    assert "kint" not in plain.columns() and len(plain) == len(res)
