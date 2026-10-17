# SPDX-License-Identifier: GPL-3.0-or-later
"""Ion-type mixtures: the mixed clusters, the weights estimated from a reference, the choice of
reference or fallback, and a run whose spectra are a mixture of two ion types."""

import numpy as np
import pytest
from scipy.stats import binom

from hxex import chem
from hxex.fit import FitConfig
from hxex.fragments import fragment_targets
from hxex.io import MzmlTarget, envelope_groups, open_bundle
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import (
    EnvelopeConfig,
    PickerConfig,
    estimate_envelope,
    plan_bins,
    reference_residual,
    series_envelopes,
)
from hxex.picking.pick import composition_mz
from hxex.quality import QualityConfig
from hxex.run import RunConfig, run
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

PEPTIDE, CHARGE, ION = "RDKVQKEYAL", 2, "c-1"
PICK = PickerConfig(nterm_subtract=0, zero_filling=3, noise=50.0)
TRUE_WEIGHTS = np.array([0.25, 0.75])


def mixture_spectrum(weights, level, height=1e5, sites=None):
    """A spectrum of the mixture of ``c-1`` and ``c`` (weights), labelled to the mean deuteration
    ``level``: three points around each cluster's position, the clusters' heights the mixture's
    abundance convolved with a binomial deuterium distribution."""
    shape, offset = chem.isotope_clusters(PEPTIDE, ION, nterm_subtract=0)
    mixed = chem.mixed_clusters(shape, offset, weights)
    plan = plan_bins(PEPTIDE, CHARGE, modification=f"ion_type:{ION}", config=PICK,
                     clusters=mixed)
    n_sites = plan.n_sites if sites is None else sites
    d = binom.pmf(np.arange(n_sites + 1), n_sites, min(max(level / n_sites, 1e-9), 1 - 1e-9))
    pos = composition_mz(plan, level)
    heights = np.zeros(len(pos))
    conv = np.convolve(mixed[0], d)[: len(pos)]
    heights[: len(conv)] = conv * height
    mz = np.concatenate([p + np.array([-0.0008, 0.0, 0.0008]) for p in pos])
    y = np.concatenate([h * np.array([0.4, 1.0, 0.4]) for h in heights])
    keep = y > 1.0
    return mz[keep], y[keep]


def test_mixed_clusters():
    shape, offset = chem.isotope_clusters(PEPTIDE, ION, nterm_subtract=0)
    s1, o1 = chem.mixed_clusters(shape, offset, [1.0])
    np.testing.assert_allclose(s1, shape) and np.testing.assert_allclose(o1, offset)
    s, o = chem.mixed_clusters(shape, offset, [1, 3])  # normalised to 0.25, 0.75
    assert len(s) == len(shape) + 1 and s.sum() == pytest.approx(1.0)
    assert s[0] == pytest.approx(0.25 * shape[0])
    # the second cluster: c-1's 13C cluster and c's monoisotopic peak (one hydrogen heavier)
    w = np.array([0.25 * shape[1], 0.75 * shape[0]])
    assert o[1] == pytest.approx((w[0] * offset[1] + w[1] * chem.H_ATOM) / w.sum())
    shifted = chem.mixed_clusters(shape, offset, [0, 1])
    np.testing.assert_allclose(shifted[0][1:], shape)  # all c: the c-1 ladder moved by one
    for bad in ([], [-1, 2], [0, 0]):
        with pytest.raises(ValueError):
            chem.mixed_clusters(shape, offset, bad)


def test_weights_are_recovered_from_the_reference():
    refs = [mixture_spectrum(TRUE_WEIGHTS, 0.0, height=h) for h in (1e5, 7e4, 1.2e5)]
    e = estimate_envelope(refs, PEPTIDE, CHARGE, modification=f"ion_type:{ION}",
                          config=EnvelopeConfig(mode="mixed", components=3), picker=PICK)
    np.testing.assert_allclose(e.weights, [0.25, 0.75, 0.0], atol=0.01)
    assert e.good and e.residual < 0.02 and e.n_reference == 3
    np.testing.assert_allclose(e.fit_shape, e.clusters[0])
    # the residual is that of the references picked with the final weights' bins (HISTORY 88),
    # also after a single iteration, whose weights were fitted with the equal weights' bins
    mod = f"ion_type:{ION}"
    for iterations in (1, 3):
        f = estimate_envelope(refs, PEPTIDE, CHARGE, modification=mod, picker=PICK,
                              config=EnvelopeConfig(mode="mixed", components=3,
                                                    iterations=iterations))
        assert f.residual == pytest.approx(
            reference_residual(refs, f, PEPTIDE, CHARGE, modification=mod, picker=PICK))
    one = estimate_envelope(refs, PEPTIDE, CHARGE, modification=f"ion_type:{ION}",
                            config=EnvelopeConfig(mode="mixed", components=1), picker=PICK)
    assert not one.good and one.residual > 0.3  # one ion type cannot describe the mixture
    inferred = estimate_envelope(refs, PEPTIDE, CHARGE, modification=f"ion_type:{ION}",
                                 config=EnvelopeConfig(mode="infer", components=2), picker=PICK)
    assert inferred.fit_shape.sum() == pytest.approx(1.0)
    assert np.abs(inferred.fit_shape[: len(e.fit_shape)] - e.fit_shape[: len(inferred.fit_shape)]
                  ).max() < 0.02  # the picked reference envelope is the mixture
    theory = estimate_envelope(refs, PEPTIDE, CHARGE, modification=f"ion_type:{ION}",
                               picker=PICK)
    assert theory.mode == "theory" and list(theory.weights) == [1.0]
    empty = estimate_envelope([], PEPTIDE, CHARGE, modification=f"ion_type:{ION}",
                              config=EnvelopeConfig(mode="mixed"), picker=PICK)
    assert not empty.good and empty.n_reference == 0


def series(spectra_by_time, data_id=0, sample="S"):
    entry = PeptideEntry("", sample, PEPTIDE, "0001-0010", 1, 10, CHARGE, f"ion_type:{ION}")
    out = []
    for k, (t, (mz, y)) in enumerate(spectra_by_time):
        out.append(Spectrum(mz, y, entry, t, 1 + k % 2, data_id=data_id, label=f"{t}",
                            spectrum_id=f"p:{data_id}/{k}"))
    return out


def test_reference_fallback_and_theory():
    good = mixture_spectrum(TRUE_WEIGHTS, 0.0)
    noise = (good[0], np.random.default_rng(1).uniform(0, 1e5, len(good[0])))
    cfg = EnvelopeConfig(mode="mixed", components=2)
    s = SpectrumSet.from_spectra(
        series([(0.0, good), (0.0, good), (60.0, good)], 0)  # a good reference
        + series([(0.0, noise), (1e7, good), (60.0, good)], 1)  # a poor one, a good fallback
        + series([(60.0, good)], 2))  # nothing to learn from
    table, by = series_envelopes(s, config=cfg, reference_time=0.0, fallback_time=1e7,
                                 picker=PICK)
    assert list(table.source) == ["reference", "fallback", "theory"]
    assert list(table.good) == [True, True, False]
    np.testing.assert_allclose(table.weights[0], [0.25, 0.75], atol=0.01)
    assert np.isnan(table.residual[2]) and table.n_reference.tolist() == [2, 1, 0]
    assert len(table.shape(0)) == len(by[0].fit_shape)
    no_fallback, _ = series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK)
    assert no_fallback.source[1] == "reference" and not no_fallback.good[1]


def test_conditions_share_an_envelope():
    """Gentle and harsh scans of one ion share its envelope: a condition without references
    borrows it, one whose references it does not describe is left out of it and marked poor."""
    good = mixture_spectrum(TRUE_WEIGHTS, 0.0)
    other = mixture_spectrum(np.array([0.9, 0.1]), 0.0)  # mostly c-1: not the same mixture
    later = mixture_spectrum(TRUE_WEIGHTS, 2.0)
    cfg = EnvelopeConfig(mode="mixed", components=2, share=True)
    s = SpectrumSet.from_spectra(
        series([(0.0, good), (60.0, later)], 0)  # gentle
        + series([(0.0, mixture_spectrum(TRUE_WEIGHTS, 0.0, height=6e4)), (60.0, later)], 1)
        + series([(60.0, later)], 2)  # no reference of its own
        + series([(0.0, other), (60.0, later)], 3)  # a reference the others do not agree with
        + series([(60.0, later)], 4))  # another ion: nothing to learn from
    groups = {0: "ion", 1: "ion", 2: "ion", 3: "ion"}
    table, by = series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK, groups=groups)
    assert list(table.source) == ["shared"] * 4 + ["theory"]
    assert table.n_reference.tolist() == [1, 1, 0, 1, 0]
    for d in range(4):  # one envelope, fitted without series 3
        np.testing.assert_allclose(table.weights[d], [0.25, 0.75], atol=0.01)
        assert by[d] is by[0]
    own = table.own_residual
    assert own[0] < 0.02 and own[1] < 0.02 and own[3] > cfg.max_residual
    assert table.agrees.tolist() == [True, True, True, False, True]
    assert np.isnan(own[2]) and np.isnan(own[4])
    assert table.poor.tolist() == [False, False, False, True, True]
    mod = f"ion_type:{ION}"
    assert reference_residual([other], by[0], PEPTIDE, CHARGE, modification=mod,
                              picker=PICK) == pytest.approx(own[3])
    assert np.isnan(reference_residual([], by[0], PEPTIDE, CHARGE, modification=mod,
                                       picker=PICK))
    # two conditions that disagree: no telling which is right, so both are marked
    pair, _ = series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK,
                               groups={0: "ion", 3: "ion"})
    assert pair.agrees.tolist() == [False, True, True, False, True]
    assert pair.poor.tolist() == [True, False, True, True, True]
    # without share (or without groups) every series has its own
    alone, _ = series_envelopes(s, config=EnvelopeConfig(mode="mixed", components=2),
                                reference_time=0.0, picker=PICK, groups=groups)
    assert list(alone.source) == ["reference", "reference", "theory", "reference", "theory"]
    np.testing.assert_allclose(alone.weights[3], [0.9, 0.1], atol=0.01)
    assert np.isnan(alone.own_residual).all() and alone.agrees.all()
    assert alone.poor.tolist() == [False, False, True, False, True]


def test_three_conditions_agreeing_with_the_core_share_its_envelope():
    """Two conditions off the core in opposite directions agree with it but not with each
    other: the core (the condition agreeing with most) describes both within max_residual,
    so they are admitted and the envelope is refitted with every condition."""
    cfg = EnvelopeConfig(mode="mixed", components=2, share=True)
    a = mixture_spectrum(TRUE_WEIGHTS, 0.0)
    b = mixture_spectrum(np.array([0.35, 0.65]), 0.0)
    c = mixture_spectrum(np.array([0.15, 0.85]), 0.0)
    s = SpectrumSet.from_spectra(series([(0.0, a)], 0) + series([(0.0, b)], 1)
                                 + series([(0.0, c)], 2) + series([(0.0, c)], 3))
    groups = {0: "ion", 1: "ion", 2: "ion"}
    table, by = series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK, groups=groups)
    assert table.agrees.tolist() == [True, True, True, True]
    assert table.poor.tolist() == [False, False, False, False]
    assert by[0] is by[1] is by[2] and by[0].n_reference == 3  # refitted with all three
    assert table.n_reference.tolist() == [1, 1, 1, 1]  # each series' own references
    # the two off-core conditions alone disagree, and stay on record when the group then
    # falls back to the theory (no fallback time, the pair's envelope not good)
    pair, _ = series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK,
                               groups={1: "ion", 2: "ion"})
    assert pair.agrees[[1, 2]].tolist() == [False, False]
    assert np.isfinite(pair.own_residual[[1, 2]]).all()


def test_envelopes_report_progress_and_stop_on_cancel(tmp_path):
    from hxex.hooks import Cancelled

    good = mixture_spectrum(TRUE_WEIGHTS, 0.0)
    s = SpectrumSet.from_spectra(series([(0.0, good)], 0) + series([(0.0, good)], 1))
    cfg = EnvelopeConfig(mode="mixed", components=2)
    seen = []
    series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK,
                     progress=lambda d, t: seen.append((d, t)))
    assert seen[0] == (0, 2) and seen[-1] == (2, 2)
    with pytest.raises(Cancelled):
        series_envelopes(s, config=cfg, reference_time=0.0, picker=PICK, cancel=lambda: True)
    # the run names the stage before picking and stops there when cancelled
    stages = []

    class Recorder:
        def start_stage(self, name, start, end):
            stages.append((name, start, end))

        def __call__(self, done, total):
            pass

    fit = FitConfig(k_min=1, k_max=1, adam_steps=20, n_restarts=1,
                    backend=Backend("cpu", "float64"), noise=NoiseConfig("gaussian", percent=3.0))
    config = RunConfig(picker=PICK, fit=fit, quality=None, envelope=cfg)
    run(s, config, progress=Recorder())
    assert [x[0] for x in stages[:2]] == ["envelope", "pick"]
    assert stages[0][1] == 0 and stages[0][2] == stages[1][1] < stages[1][2]
    with pytest.raises(Cancelled):
        run(s, config, cancel=lambda: True)


def test_envelope_groups():
    """Series that differ only in their scan filters or list label share a key."""
    def target(label, **kw):
        entry = PeptideEntry("", "", PEPTIDE, "0001-0010", 1, 10, CHARGE, f"ion_type:{ION}",
                             extra={"condition": label, "parent_peptide": PEPTIDE})
        return MzmlTarget(entry, ms_level=2, activation="ETD", precursor_charge=3, **kw)

    targets = [target("gentle", filter_excludes="sid"), target("harsh", filter_contains="sid"),
               MzmlTarget(target("x").entry, ms_level=2, activation="EThcD"),
               target("gentle", filter_excludes="sid", preactivation="HCD")]
    source = {"reader": "mzml", "targets": [t.to_dict() for t in targets],
              "series": [{"data_id": i, "sample": "S", "target": i} for i in range(4)]
              + [{"data_id": 4, "sample": "T", "target": 0}]}
    g = envelope_groups(source)
    assert g[0] == g[1] and len({g[0], g[2], g[3], g[4]}) == 4
    assert envelope_groups({"reader": "specexport"}) == {} and envelope_groups(None) == {}
    # targets as the fragment lists make them: a gentle and a harsh c4 share, c2 does not
    parent = PeptideEntry("f", "S", "GSHPLVFK", "11-18", 11, 18, 3)
    made = (fragment_targets(parent, "c:2,4", activation="ETD", label="gentle",
                             filter_excludes="sid")
            + fragment_targets(parent, "c:4", activation="ETD", label="harsh",
                               filter_contains="sid"))
    source = {"reader": "mzml", "targets": [t.to_dict() for t in made],
              "series": [{"data_id": i, "sample": "S", "target": i} for i in range(len(made))]}
    g = envelope_groups(source)
    ion = [(t.entry.peptide, t.entry.modification, t.entry.charge) for t in made]
    assert len(set(ion)) == 4 < len(made)
    for i in range(len(made)):
        for j in range(len(made)):
            assert (g[i] == g[j]) == (ion[i] == ion[j])


def test_a_poor_envelope_is_flagged(tmp_path):
    """A series fitted with a mixed envelope but without references is flagged poor_envelope,
    in the run and in the bundle."""
    times, levels = [0.0, 30.0, 60.0, 120.0, 240.0], [0.0, 1.5, 3.0, 4.5, 6.0]
    with_ref = series([(t, mixture_spectrum(TRUE_WEIGHTS, lv))
                       for t, lv in zip(times, levels, strict=True)], 0)
    without = series([(t, mixture_spectrum(TRUE_WEIGHTS, lv))
                      for t, lv in zip(times[1:], levels[1:], strict=True)], 1, sample="T")
    fit = FitConfig(k_min=1, k_max=1, adam_steps=100, n_restarts=1,
                    backend=Backend("cpu", "float64"),
                    noise=NoiseConfig("gaussian", percent=3.0))
    cfg = RunConfig(picker=PICK, fit=fit, quality=QualityConfig(),
                    envelope=EnvelopeConfig(mode="mixed", components=2))
    res = run(SpectrumSet.from_spectra(with_ref + without), cfg, out=tmp_path / "b")
    assert list(res.envelopes.source) == ["reference", "theory"]
    flag = np.asarray(res.quality.poor_envelope)
    np.testing.assert_array_equal(flag, res.fit.data_id == 1)
    assert res.quality.flagged[flag].all()
    with open_bundle(tmp_path / "b") as b:
        np.testing.assert_array_equal(b.quality().poor_envelope, flag)
        np.testing.assert_array_equal(b.envelopes().own_residual, res.envelopes.own_residual)
        np.testing.assert_array_equal(b.envelopes().agrees, res.envelopes.agrees)


def test_a_run_with_a_mixture_recovers_the_deuteration(tmp_path):
    """Spectra of a c-1 / c mixture: with the mixture the fitted deuteration is the true one;
    with the theory of c-1 alone it is about one deuterium (the hydrogen) too high."""
    levels = [0.0, 1.5, 3.0, 4.5, 6.0]
    times = [0.0, 30.0, 60.0, 120.0, 240.0]
    spectra = SpectrumSet.from_spectra(series(
        [(t, mixture_spectrum(TRUE_WEIGHTS, lv)) for t, lv in zip(times, levels, strict=True)]
        + [(0.0, mixture_spectrum(TRUE_WEIGHTS, 0.0, height=8e4))]))
    fit = FitConfig(k_min=1, k_max=1, adam_steps=200, n_restarts=2,
                    backend=Backend("cpu", "float64"),
                    noise=NoiseConfig("gaussian", percent=3.0))
    mixed = RunConfig(picker=PICK, fit=fit, quality=None,
                      envelope=EnvelopeConfig(mode="mixed", components=2))
    res = run(spectra, mixed, out=tmp_path / "b")
    means = res.fit.fits[1].means[:, 0]
    np.testing.assert_allclose(means[:5], levels, atol=0.25)
    assert res.envelopes.source[0] == "reference" and res.envelopes.good[0]
    plain = run(spectra, RunConfig(picker=PICK, fit=fit, quality=None))
    off = plain.fit.fits[1].means[:5, 0] - np.array(levels)
    assert np.median(off) > 0.4 and plain.envelopes is None
    with open_bundle(tmp_path / "b") as b:
        got = b.envelopes()
        assert got.config == mixed.envelope and list(got.source) == list(res.envelopes.source)
        np.testing.assert_array_equal(got.weights, res.envelopes.weights)
        np.testing.assert_array_equal(got.fit_shape, res.envelopes.fit_shape)
        np.testing.assert_array_equal(b.measurements().shapes[1], res.measurements.shapes[1])
    with pytest.raises(ValueError, match="needs the raw spectra"):
        run(res.binned, mixed)


def test_envelope_config():
    c = EnvelopeConfig(mode="infer", components=3)
    assert EnvelopeConfig.from_dict(c.to_dict()) == c
    assert EnvelopeConfig.from_dict({"share": True}).share
    for bad in ({"mode": "other"}, {"components": 0}, {"max_residual": 0}, {"iterations": 0}):
        with pytest.raises(ValueError):
            EnvelopeConfig(**bad)
    with pytest.raises(ValueError, match="unknown"):
        EnvelopeConfig.from_dict({"bogus": 1})
    r = RunConfig(envelope=c, envelope_fallback_time=1e7)
    assert RunConfig.from_dict(r.to_dict()) == r
