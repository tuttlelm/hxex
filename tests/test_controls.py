# SPDX-License-Identifier: GPL-3.0-or-later
"""Controls: replicate averaging, per-control fallbacks, the correction and its uncertainty."""

import numpy as np
import pytest

from hxex.controls import ControlConfig, correct_means, fit_controls
from hxex.fit import FitConfig
from hxex.fit.result import FitResult, KFit
from hxex.noise import NoiseEstimate

CFG = ControlConfig(undeut_time=0.0, fd_time=1e6)


def fake_fit(rows, outlier=None):
    """rows: (data_id, time, K=1 mean). One component, n=mean, p=1."""
    d = np.array([r[0] for r in rows])
    t = np.array([r[1] for r in rows], dtype=float)
    mean = np.array([r[2] for r in rows], dtype=float)
    m = len(rows)
    one = np.ones((m, 1))
    kf = KFit(k=1, A=one, n=mean[:, None], p=one, log_A=one, n_raw=one, logit_p=one,
              nll=np.zeros(m), bic=np.zeros(m), aic=np.zeros(m),
              passes_min_pop=np.ones(m, dtype=bool))
    est = None
    if outlier is not None:
        est = NoiseEstimate(gain=np.full(m, np.nan), estimated=np.zeros(m, dtype=bool),
                            reason=np.array([""] * m), score=np.full(m, np.nan),
                            outlier=np.asarray(outlier, dtype=bool),
                            n_replicated=np.zeros(m, dtype=np.int64))
    return FitResult(
        config=FitConfig(), spectrum_id=np.array([f"p:{i}" for i in range(m)]), data_id=d,
        time=t, k_values=(1,), fits={1: kf}, selected_K=np.ones(m, dtype=int),
        score=np.zeros((m, 1)), obs_scale=np.ones(m), sample_size=np.ones(m),
        n_max=np.ones(m), noise_kind=np.array(["gaussian"] * m), noise_estimate=est)


def table(rows, **kw):
    fit = fake_fit(rows, **kw)
    return fit_controls(fit, np.full(len(rows), 10.0), CFG)


def test_good_controls_are_averaged_and_used():
    c = table([(0, 0, 0.02), (0, 0, 0.04), (0, 5, 4.0), (0, 1e6, 8.0), (0, 1e6, 8.4)])
    assert c.m0[0] == pytest.approx(0.03) and c.mfd[0] == pytest.approx(8.2)
    assert c.m0_se[0] == pytest.approx(0.01) and c.mfd_se[0] == pytest.approx(0.2)
    assert (c.n_undeut[0], c.n_fd[0]) == (2, 2)
    assert not c.fallback[0] and c.flags[0] == ""
    assert c.retention[0] == pytest.approx((8.2 - 0.03) / 10)


def test_each_control_falls_back_on_its_own():
    # high m0 only: m0 -> 0, the measured mFD (retention checked against m0 = 0) is kept
    c = table([(0, 0, 2.0), (0, 0, 2.2), (0, 1e6, 8.0), (0, 1e6, 8.2)])
    assert c.m0[0] == 0.0 and c.m0_se[0] == 0.0 and c.mfd[0] == pytest.approx(8.1)
    assert c.flags[0] == "m0_high" and c.fallback[0]
    # low retention only: mFD -> n_amides, m0 kept
    c = table([(0, 0, 0.1), (0, 0, 0.1), (0, 1e6, 3.0), (0, 1e6, 3.0)])
    assert c.m0[0] == pytest.approx(0.1) and c.mfd[0] == 10.0 and c.mfd_se[0] == 0.0
    assert c.flags[0] == "retention_low"
    c = table([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 12.0), (0, 1e6, 12.0)])
    assert c.flags[0] == "retention_high" and c.mfd[0] == 10.0


def test_missing_controls_use_theory_and_are_flagged():
    c = table([(0, 5, 3.0)])
    assert (c.m0[0], c.mfd[0]) == (0.0, 10.0) and c.fallback[0]
    assert c.flags[0] == "m0_missing;fd_missing"
    assert np.isnan(c.m0_measured[0]) and (c.n_undeut[0], c.n_fd[0]) == (0, 0)
    assert c.retention[0] == 1.0


def test_unreplicated_controls_have_no_uncertainty_and_are_flagged():
    c = table([(0, 0, 0.1), (0, 1e6, 8.0)])
    assert np.isnan(c.m0_se[0]) and np.isnan(c.mfd_se[0]) and not c.fallback[0]
    assert c.flags[0] == "m0_unreplicated;fd_unreplicated"


def test_series_are_separate_and_outlier_controls_are_left_out():
    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 0, 3.0), (0, 1e6, 8.0), (1, 0, 0.0), (1, 1e6, 6.0)]
    c = table(rows, outlier=[0, 0, 1, 0, 0, 0])
    assert list(c.data_id) == [0, 1] and c.n_undeut.tolist() == [2, 1]
    assert c.m0[0] == 0.0 and c.mfd.tolist() == [8.0, 6.0]
    np.testing.assert_array_equal(c.rows([1, 0, 1]), [1, 0, 1])


def test_a_fit_without_k1_is_refused_and_amides_must_be_constant():
    fit = fake_fit([(0, 0, 0.0)])
    object.__setattr__(fit, "fits", {})
    with pytest.raises(ValueError, match="K=1"):
        fit_controls(fit, [10.0], CFG)
    with pytest.raises(ValueError, match="amide count"):
        fit_controls(fake_fit([(0, 0, 0.0), (0, 1e6, 8.0)]), [10.0, 11.0], CFG)


def test_correction_values_and_clipping():
    c = table([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 8.0), (0, 1e6, 8.0)])
    out = correct_means(c, [0, 0, 0], np.array([4.0, 8.0, -0.3]))
    np.testing.assert_allclose(out.mean, [5.0, 10.0, 0.0])
    assert out.se is None and out.ci is None


def test_correction_uncertainty_matches_a_simulation():
    c = table([(0, 0, 0.1), (0, 0, 0.3), (0, 1e6, 7.0), (0, 1e6, 8.0)])
    x, sx = np.array([[3.0]]), np.array([[0.2]])
    ci = np.array([[[2.6, 3.4]]])
    out = correct_means(c, [0], x, se=sx, ci=ci)
    rng = np.random.default_rng(0)
    m0 = rng.normal(c.m0[0], c.m0_se[0], 200000)
    mf = rng.normal(c.mfd[0], c.mfd_se[0], 200000)
    sim = (3.0 - m0) / (mf - m0) * 10
    assert out.control_se[0, 0] == pytest.approx(sim.std(), rel=0.02)
    total = (rng.normal(3.0, 0.2, 200000) - m0) / (mf - m0) * 10
    assert out.se[0, 0] == pytest.approx(total.std(), rel=0.02)
    lo, hi = out.ci[0, 0]
    point = out.mean[0, 0]
    assert lo < (2.6 - c.m0[0]) / (c.mfd[0] - c.m0[0]) * 10 and hi > point
    assert lo < point < hi


def test_theoretical_controls_add_no_uncertainty_and_nan_stays_nan():
    c = table([(0, 5, 3.0)])
    out = correct_means(c, [0, 0], np.array([4.0, np.nan]), se=np.array([0.5, np.nan]))
    assert out.control_se[0] == 0.0 and out.se[0] == pytest.approx(0.5)
    assert np.isnan(out.mean[1])


def test_config_round_trip_and_checks():
    assert ControlConfig.from_dict(CFG.to_dict()) == CFG
    with pytest.raises(ValueError, match="unknown"):
        ControlConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        ControlConfig(retention_min=1.2)
