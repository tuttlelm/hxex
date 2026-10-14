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
    # low retention only: mFD -> m0 + d2o_fraction * n_amides (default 0.9), m0 kept
    c = table([(0, 0, 0.1), (0, 0, 0.1), (0, 1e6, 3.0), (0, 1e6, 3.0)])
    assert c.m0[0] == pytest.approx(0.1) and c.mfd[0] == pytest.approx(9.1)
    assert c.mfd_se[0] == 0.0 and c.retention[0] == pytest.approx(0.9)
    assert c.flags[0] == "retention_low"
    c = table([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 12.0), (0, 1e6, 12.0)])
    assert c.flags[0] == "retention_high" and c.mfd[0] == pytest.approx(9.0)


def test_missing_controls_use_theory_and_are_flagged():
    c = table([(0, 5, 3.0)])
    assert c.m0[0] == 0.0 and c.mfd[0] == pytest.approx(9.0) and c.fallback[0]
    assert c.flags[0] == "m0_missing;fd_missing"
    assert np.isnan(c.m0_measured[0]) and (c.n_undeut[0], c.n_fd[0]) == (0, 0)
    assert c.retention[0] == pytest.approx(0.9)  # the buffer's D2O fraction
    # the buffer's fraction is a user setting
    full = fit_controls(fake_fit([(0, 5, 3.0)]), np.full(1, 10.0),
                        ControlConfig(undeut_time=0.0, fd_time=1e6, d2o_fraction=0.8))
    assert full.mfd[0] == pytest.approx(8.0) and full.retention[0] == pytest.approx(0.8)
    for bad in (0.0, 1.2):
        with pytest.raises(ValueError, match="d2o_fraction"):
            ControlConfig(d2o_fraction=bad)


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
    assert out.control_se[0] == 0.0 and out.se[0] == pytest.approx(0.5 / 0.9)
    assert np.isnan(out.mean[1])


def test_the_lower_limit_follows_the_buffer_and_the_fallback_can_be_the_median():
    # retention_min defaults to half of d2o_fraction: 0.45 at 0.9, so a retention of 0.47 is kept
    rows = [(0, 0, 0.0), (0, 1e6, 4.7), (1, 0, 0.0), (1, 1e6, 6.0), (2, 0, 0.0), (2, 1e6, 7.0),
            (3, 0, 0.0)]
    c = table(rows)
    assert CFG.min_retention == pytest.approx(0.45) and not c.fallback[0]
    assert c.retention[:3].tolist() == pytest.approx([0.47, 0.6, 0.7])
    assert c.retention[3] == pytest.approx(0.9) and "fd_missing" in c.flags[3].split(";")
    # with fd_fallback='median' the missing control takes the median accepted retention (0.6)
    med = fit_controls(fake_fit(rows), np.full(len(rows), 10.0),
                       ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="median"))
    assert med.retention[3] == pytest.approx(0.6) and med.mfd[3] == pytest.approx(6.0)
    assert med.flags[3].split(";")[-2:] == ["fd_missing", "fd_median"] and med.fallback[3]
    assert ControlConfig(d2o_fraction=0.8).min_retention == pytest.approx(0.4)
    assert ControlConfig(retention_min=0.3).min_retention == 0.3
    with pytest.raises(ValueError, match="fd_fallback"):
        ControlConfig(fd_fallback="mean")


def test_config_round_trip_and_checks():
    assert ControlConfig.from_dict(CFG.to_dict()) == CFG
    with pytest.raises(ValueError, match="unknown"):
        ControlConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        ControlConfig(retention_min=1.2)


# --- theoretical back-exchange (fd_fallback 'kint') ----------------------------------------

PEPTIDES = ["LGEHNIDVLEGNEQF", "MDIAIHHPW", "FDQFFGEHLLE", "SPFYLRPPSF", "RLEKDRFSVNL",
            "LDVKHFSPEEL", "EELKVKVLGD", "VIEVHGKHEERQDEHG", "SSLSSDGVL", "TVNGPRKQVSGPERTIP",
            "ITREEKPAVTAAPKK", "FISREFHRKYRIPADV", "AIHHPWIRRPFF", "GLSEMRLEKDRF", "DVKHFSPEE"]
QUENCH = {"ph_read": 2.5, "temperature": 0.0, "d2o_fraction": 0.0}


def kint_case(a=0.85, t0=2000.0, off=(), missing=(), extra=()):
    """Series 0..14 with FD controls following a * sum exp(-k t0) (plus a little scatter);
    ``off`` series hold half that, ``missing`` have no FD control; ``extra`` ids are series
    without rates (a fragment ion), FD control at 0.3 per amide."""
    from hxex.controls import _kint_predict, series_rates
    from hxex.kint import ExchangeConditions

    rates = series_rates(PEPTIDES, ["" for _ in PEPTIDES], range(len(PEPTIDES)),
                         ExchangeConditions(**QUENCH))
    rng = np.random.default_rng(1)
    rows, n = [], []
    for d in range(len(PEPTIDES)):
        k = rates[d]
        fd = float(_kint_predict(a, t0, [k])[0]) * (0.5 if d in off else 1.0)
        rows += [(d, 0.0, 0.05), (d, 0.0, 0.05)]
        if d not in missing:
            rows += [(d, 1e6, 0.05 + fd + rng.normal(0, 0.05)), (d, 1e6, 0.05 + fd)]
        n += [len(k)] * (2 if d in missing else 4)
    for d in extra:
        rows += [(d, 0.0, 0.0), (d, 0.0, 0.0), (d, 1e6, 3.0), (d, 1e6, 3.0)]
        n += [10] * 4
    return fake_fit(rows), np.array(n, float), rates


def test_kint_fallback_fits_the_run_and_replaces_missing_and_off_model_controls():
    from hxex.controls import _kint_predict

    fit, n, rates = kint_case(off=(3,), missing=(5,), extra=(99,))
    cfg = ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="kint", quench=QUENCH)
    c = fit_controls(fit, n, cfg, rates=rates)
    m = c.fd_model
    assert m["a"] == pytest.approx(0.85, rel=0.05) and m["t0"] == pytest.approx(2000, rel=0.2)
    assert m["n_series"] == 13 and m["n_rejected"] == 1 and m["sd_per_amide"] < 0.03
    flags = dict(zip(c.data_id.tolist(), c.flags.tolist(), strict=True))
    assert flags[3] == "fd_off_model;fd_kint" and flags[5] == "fd_missing;fd_kint"
    for d in (3, 5):
        i = c.rows([d])[0]
        want = 0.05 + float(_kint_predict(m["a"], m["t0"], [rates[d]])[0])
        assert c.fallback[i] and c.mfd[i] == pytest.approx(want)
        assert c.mfd_se[i] == pytest.approx(m["sd_per_amide"] * c.n_amides[i])
    # good controls are measured, and a low retention alone is no reason to reject one
    good = c.rows([0, 1, 2])
    assert not c.fallback[good].any() and (c.retention[good] < 0.9).all()
    assert cfg.min_retention == 0.0
    # a series without rates (a fragment ion) keeps its measured control, or d2o_fraction
    i = c.rows([99])[0]
    assert c.mfd[i] == pytest.approx(3.0) and "fd_kint" not in c.flags[i]


def test_kint_fallback_needs_conditions_rates_and_enough_controls():
    fit, n, rates = kint_case(missing=tuple(range(8)))
    cfg = ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="kint", quench=QUENCH)
    c = fit_controls(fit, n, cfg, rates=rates)  # 7 controls < kint_min_series 10
    assert c.fd_model is None and "fd_kint" not in ";".join(c.flags)
    i = c.rows([0])[0]
    assert c.mfd[i] == pytest.approx(c.m0[i] + 0.9 * c.n_amides[i])
    c = fit_controls(fit, n, ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="kint",
                                           quench=QUENCH, kint_min_series=5), rates=rates)
    assert c.fd_model["n_series"] == 7
    with pytest.raises(ValueError, match="intrinsic rates"):
        fit_controls(fit, n, cfg)
    with pytest.raises(ValueError, match="quench"):
        ControlConfig(fd_fallback="kint")
    for bad in ({"kint_tolerance": 0}, {"kint_min_series": 2}):
        with pytest.raises(ValueError):
            ControlConfig(fd_fallback="kint", quench=QUENCH, **bad)
    assert ControlConfig.from_dict(cfg.to_dict()) == cfg
    # the fitted model is kept in the bundle's control table
    from hxex.io.bundle import _controls_table, _read_controls

    c = fit_controls(fit, n, ControlConfig(undeut_time=0.0, fd_time=1e6, fd_fallback="kint",
                                           quench=QUENCH, kint_min_series=5), rates=rates)
    back = _read_controls(_controls_table(c), cfg.to_dict())
    assert back.fd_model == c.fd_model
    assert _read_controls(_controls_table(table([(0, 0, 0.0), (0, 1e6, 8.0)])),
                          CFG.to_dict()).fd_model is None


def test_series_rates_count_the_amides_of_parent_ions_only():
    from hxex import chem
    from hxex.controls import series_rates
    from hxex.kint import ExchangeConditions

    cond = ExchangeConditions(**QUENCH)
    r = series_rates(["PEPTIDE", "PEPTIDE", "PEPTIDEK", "PEPTIDEK"],
                     ["", "", "ion_type:c", "Hex:1"], [0, 0, 1, 2], cond)
    assert set(r) == {0} and len(r[0]) == chem.count_amides("PEPTIDE")
    free = series_rates(["AAAAA"], [""], [0], cond, nterm_subtract=0)[0]
    assert np.isinf(free[0]) and np.isfinite(free[1:]).all()


def test_kint_is_the_default_fallback_when_quench_conditions_are_given():
    assert ControlConfig().fd_mode == "d2o" and ControlConfig().fd_fallback is None
    auto = ControlConfig(quench=QUENCH)
    assert auto.fd_mode == "kint" and auto.min_retention == 0.0
    assert ControlConfig.from_dict(auto.to_dict()) == auto  # stays automatic when stored
    assert ControlConfig(quench=QUENCH, fd_fallback="d2o").fd_mode == "d2o"  # explicit wins
    assert ControlConfig(quench=QUENCH, fd_fallback="median").fd_mode == "median"
    fit, n, rates = kint_case()
    c = fit_controls(fit, n, ControlConfig(undeut_time=0.0, fd_time=1e6, quench=QUENCH),
                     rates=rates)
    assert c.fd_model is not None
