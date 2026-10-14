# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment kinetics: the uptake table, ladder steps, scan-matched differences, decay fits,
per-residue least squares and the ladder figure, on synthetic ladders with known rates."""

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
    def __init__(self, means, data_id, time, ids):
        self.fits = {1: SimpleNamespace(means=np.asarray(means, float)[:, None])}
        self.data_id, self.time, self.spectrum_id = data_id, time, ids

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
