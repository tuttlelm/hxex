# SPDX-License-Identifier: GPL-3.0-or-later
"""Controls: replicate averaging, per-control fallbacks, the correction and its uncertainty."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex.controls import ControlConfig, correct_means, fit_controls
from hxex.fit import FitConfig
from hxex.fit.result import FitResult, KFit
from hxex.noise import NoiseEstimate

CFG = ControlConfig()
UN, FD = 0.0, 1e6  # the test rows name the controls by these times: fake_fit gives them roles


def fake_fit(rows, outlier=None, selected_K=None):
    """rows: (data_id, time, K=1 mean). One component, n=mean, p=1. A row at time ``UN`` is an
    undeuterated control and one at ``FD`` a fully deuterated one (their role; a control's
    time is NaN, as the readers leave it)."""
    d = np.array([r[0] for r in rows])
    t = np.array([r[1] for r in rows], dtype=float)
    role = np.where(t == UN, "undeut", np.where(t == FD, "fulldeut", ""))
    t = np.where(role == "", t, np.nan)
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
        time=t, role=role, k_values=(1,), fits={1: kf},
        selected_K=np.ones(m, dtype=int) if selected_K is None else np.asarray(selected_K),
        score=np.zeros((m, 1)), obs_scale=np.ones(m), sample_size=np.ones(m),
        n_max=np.ones(m), noise_kind=np.array(["gaussian"] * m), noise_estimate=est)


def table(rows, config=CFG, groups=None, **kw):
    fit = fake_fit(rows, **kw)
    return fit_controls(fit, np.full(len(rows), 10.0), config, groups=groups)


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
    # low retention only (below a set floor): mFD -> m0 + d2o_fraction * n_amides (default
    # 0.9), m0 kept
    c = table([(0, 0, 0.1), (0, 0, 0.1), (0, 1e6, 3.0), (0, 1e6, 3.0)],
              ControlConfig(retention_min=0.45))
    assert c.m0[0] == pytest.approx(0.1) and c.mfd[0] == pytest.approx(9.1)
    assert c.mfd_se[0] == 0.0 and c.retention[0] == pytest.approx(0.9)
    assert c.flags[0] == "retention_low"
    # replicates above the amide count are left out before the mean (HISTORY 143), so with
    # m0 near 0 a retention above retention_max (1.1) leaves no replicate: the control is gone
    c = table([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 12.0), (0, 1e6, 12.0)])
    assert c.flags[0] == "fd_replicate_left_out;fd_missing" and c.mfd[0] == pytest.approx(9.0)
    # retention_high still applies below the amide count (an m0 below 0)
    c = table([(0, 0, -1.5), (0, 0, -1.5), (0, 1e6, 9.8), (0, 1e6, 9.8)])
    assert c.flags[0] == "retention_high"


def test_missing_controls_use_theory_and_are_flagged():
    c = table([(0, 5, 3.0)])
    assert c.m0[0] == 0.0 and c.mfd[0] == pytest.approx(9.0) and c.fallback[0]
    assert c.flags[0] == "m0_missing;fd_missing"
    assert np.isnan(c.m0_measured[0]) and (c.n_undeut[0], c.n_fd[0]) == (0, 0)
    assert c.retention[0] == pytest.approx(0.9)  # the buffer's D2O fraction
    # the buffer's fraction is a user setting
    full = fit_controls(fake_fit([(0, 5, 3.0)]), np.full(1, 10.0),
                        ControlConfig(d2o_fraction=0.8))
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


def test_there_is_no_fixed_floor_unless_set_and_the_fallback_is_the_median():
    # no fixed floor by default (too few controls for the run's limit), so a retention of 0.47
    # is kept; a set retention_min is a fixed floor
    rows = [(0, 0, 0.0), (0, 1e6, 4.7), (1, 0, 0.0), (1, 1e6, 6.0), (2, 0, 0.0), (2, 1e6, 7.0),
            (3, 0, 0.0)]
    c = table(rows)
    assert CFG.min_retention == 0.0 and not c.fallback[0]
    assert c.limits == {"retention_floor": 0.0, "median": None, "robust_sd": None,
                        "n_series": 3}
    assert "retention_low" in table(rows, ControlConfig(retention_min=0.5)).flags[0]
    assert c.retention[:3].tolist() == pytest.approx([0.47, 0.6, 0.7])
    # by default (no quench conditions) the missing control takes the median accepted
    # retention (0.6)
    assert c.retention[3] == pytest.approx(0.6) and c.mfd[3] == pytest.approx(6.0)
    assert c.flags[3].split(";")[-2:] == ["fd_missing", "fd_median"] and c.fallback[3]
    # with fd_fallback='d2o' it takes d2o_fraction
    d2o = fit_controls(fake_fit(rows), np.full(len(rows), 10.0), ControlConfig(fd_fallback="d2o"))
    assert d2o.retention[3] == pytest.approx(0.9) and d2o.flags[3].split(";")[-1] == "fd_missing"
    assert ControlConfig(d2o_fraction=0.8).min_retention == 0.0
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


def kint_case(a=0.85, t0=2000.0, off=(), missing=(), extra=(), extra_fd=3.0):
    """Series 0..14 with FD controls following a * sum exp(-k t0) (plus a little scatter);
    ``off`` series hold half that, ``missing`` have no FD control; ``extra`` ids are series
    without rates (a fragment ion), FD control at ``extra_fd`` (0.3 per amide)."""
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
        rows += [(d, 0.0, 0.0), (d, 0.0, 0.0), (d, 1e6, extra_fd), (d, 1e6, extra_fd)]
        n += [10] * 4
    return fake_fit(rows), np.array(n, float), rates


def test_kint_fallback_fits_the_run_and_replaces_missing_and_off_model_controls():
    from hxex.controls import _kint_predict

    fit, n, rates = kint_case(off=(3,), missing=(5,), extra=(99,), extra_fd=0.5)
    cfg = ControlConfig(fd_fallback="kint", quench=QUENCH)
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
    # a series without rates (extra Hex sites) is not judged by the prediction: the run's limit
    # judges it, and below it it takes the median of the accepted controls, never kint
    i = c.rows([99])[0]
    assert c.mfd_measured[i] == pytest.approx(0.5)
    assert c.flags[i] == "retention_outlier;fd_median" and c.fallback[i]
    assert c.mfd[i] == pytest.approx(0.0 + np.median(c.retention[~c.fallback]) * 10)
    assert c.limits["n_series"] == 15 and c.limits["retention_floor"] > 0.05
    rows = [(d, t, m) for d, t, m in [(98, 0.0, 0.0), (98, 0.0, 0.0), (98, 1e6, 6.0),
                                       (98, 1e6, 6.0)]]
    c2 = fit_controls(fake_fit(rows), np.array([10.0] * 4), cfg, rates=rates)
    assert c2.mfd[0] == pytest.approx(6.0) and c2.flags[0] == "" and not c2.fallback[0]


def test_kint_fallback_needs_conditions_rates_and_enough_controls():
    fit, n, rates = kint_case(missing=tuple(range(8)))
    cfg = ControlConfig(fd_fallback="kint", quench=QUENCH)
    c = fit_controls(fit, n, cfg, rates=rates)  # 7 controls < kint_min_series 10
    assert c.fd_model is None and "fd_kint" not in ";".join(c.flags)
    # without the model, the missing ones take the median of the accepted controls (HISTORY 131)
    i = c.rows([0])[0]
    assert c.flags[i].endswith("fd_median")
    assert c.mfd[i] == pytest.approx(c.m0[i] + np.median(c.retention[~c.fallback])
                                     * c.n_amides[i])
    c = fit_controls(fit, n, ControlConfig(fd_fallback="kint",
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

    c = fit_controls(fit, n, ControlConfig(fd_fallback="kint",
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
    assert ControlConfig().fd_mode == "median" and ControlConfig().fd_fallback is None
    auto = ControlConfig(quench=QUENCH)
    assert auto.fd_mode == "kint" and auto.min_retention == 0.0
    assert ControlConfig.from_dict(auto.to_dict()) == auto  # stays automatic when stored
    assert ControlConfig(quench=QUENCH, fd_fallback="d2o").fd_mode == "d2o"  # explicit wins
    assert ControlConfig(quench=QUENCH, fd_fallback="median").fd_mode == "median"
    fit, n, rates = kint_case()
    c = fit_controls(fit, n, ControlConfig(quench=QUENCH),
                     rates=rates)
    assert c.fd_model is not None


def test_a_fully_deuterated_level_not_above_m0_is_rejected_whatever_the_floor(caplog):
    # no retention floor (as with fd_fallback='kint'): a span of 0 or less still scales nothing
    cfg = ControlConfig(retention_min=0.0)
    for fd in (0.1, 0.05):
        c = fit_controls(fake_fit([(0, 0, 0.1), (0, 0, 0.1), (0, 1e6, fd), (0, 1e6, fd)]),
                         np.full(4, 10.0), cfg)
        assert c.flags[0] == "retention_low" and c.fallback[0]
        assert c.mfd[0] == pytest.approx(9.1) and c.retention[0] == pytest.approx(0.9)
        out = correct_means(c, [0, 0], np.array([4.6, 9.1]))
        np.testing.assert_allclose(out.mean, [5.0, 10.0])
    # a series without any exchangeable site: flagged, its retention unknown
    c = fit_controls(fake_fit([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 0.0)]), np.zeros(3), cfg)
    assert c.flags[0] == "no_amides" and np.isnan(c.retention[0]) and c.fallback[0]
    # a table holding such a row (made by hand) corrects to NaN, with a warning naming the series
    c = table([(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 8.0), (0, 1e6, 8.0)])
    bad = dataclasses.replace(c, mfd=np.array([0.0]))
    with caplog.at_level(logging.WARNING, logger="hxex.controls"):
        out = correct_means(bad, [0, 0], np.array([4.0, 8.0]))
    assert np.isnan(out.mean).all()
    assert "data_id 0" in caplog.text and "at or below" in caplog.text


def test_too_few_controls_for_the_kint_model_are_still_judged_by_the_time_course(caplog):
    """When the theoretical model is not fitted (and too few controls make a run limit) the
    time course still judges the rated series' controls (four peptides with a fully deuterated
    level of 0.1 per amide, below their own data, gave a tenfold inflated correction)."""
    _, _, rates = kint_case()
    n = {d: float(len(rates[d])) for d in (0, 1)}
    rows = [(0, 0.0, 0.0), (0, 0.0, 0.0), (0, 1e6, 0.1 * n[0]), (0, 600.0, 0.4 * n[0]),
            (1, 0.0, 0.0), (1, 0.0, 0.0), (1, 1e6, 0.6 * n[1]), (1, 600.0, 0.4 * n[1])]
    amides = np.array([n[0]] * 4 + [n[1]] * 4)
    cfg = ControlConfig(fd_fallback="kint", quench=QUENCH)
    with caplog.at_level(logging.WARNING):
        c = fit_controls(fake_fit(rows), amides, cfg, rates=rates)
    assert c.fd_model is None and "not fitted" in caplog.text
    low, ok = c.rows([0])[0], c.rows([1])[0]
    assert c.flags[low] == "fd_below_data;fd_median" and c.fallback[low]
    assert c.mfd[low] == pytest.approx(0.6 * n[0]) and c.mfd_se[low] == 0.0
    assert c.retention[low] == pytest.approx(0.6)  # the accepted control's
    assert c.flags[ok] == "fd_unreplicated" and not c.fallback[ok]
    assert c.retention[ok] == pytest.approx(0.6)
    # a generous tolerance lets it through
    c = fit_controls(fake_fit(rows), amides,
                     dataclasses.replace(cfg, time_course_tolerance=0.5), rates=rates)
    assert not c.fallback.any()


def test_a_time_point_can_serve_as_a_control_as_well():
    """undeut_at / fulldeut_at: the spectra at that time gain the role and keep their time."""
    from hxex.controls import apply_control_roles

    role = np.array(["undeut", "", "", "", ""])
    time = np.array([np.nan, 4.0, 60.0, 72000.0, 72000.0])
    cfg = ControlConfig(fulldeut_at=72000.0)
    out = apply_control_roles(role, time, cfg)
    assert list(out) == ["undeut", "", "", "fulldeut", "fulldeut"] and list(role[3:]) == ["", ""]
    both = apply_control_roles(out, time, ControlConfig(undeut_at="4"))
    assert list(both) == ["undeut", "undeut", "", "fulldeut", "fulldeut"]
    with pytest.raises(ValueError, match="fulldeut_at = 61 s: no spectrum was measured at that "
                                          "time.*4 s, 1 min, 20 h"):
        apply_control_roles(role, time, ControlConfig(fulldeut_at=61.0), where="here")
    with pytest.raises(ValueError, match="already undeut"):
        apply_control_roles(both, time, ControlConfig(fulldeut_at=4.0))
    with pytest.raises(ValueError, match="cannot be the same time"):
        ControlConfig(undeut_at=4.0, fulldeut_at=4.0)
    with pytest.raises(ValueError, match="fulldeut_at must be a time"):
        ControlConfig(fulldeut_at=float("inf"))
    # the controls are then fitted from the promoted spectra
    fit = fake_fit([(0, UN, 0.0), (0, UN, 0.1), (0, 60.0, 4.0), (0, 72000.0, 8.0),
                    (0, 72000.0, 8.2)])
    fit = dataclasses.replace(fit, role=apply_control_roles(fit.role, fit.time, cfg))
    c = fit_controls(fit, np.full(5, 10.0), cfg)
    assert c.n_fd[0] == 2 and c.mfd[0] == pytest.approx(8.1) and not c.fallback[0]


def test_the_marked_controls_can_be_replaced_by_a_time_point():
    """fulldeut_marked = false: the marked fully deuterated files do not serve; only the
    spectra at fulldeut_at do (a measured file that reads low, replaced by the plateau)."""
    from hxex.controls import apply_control_roles

    rows = [(0, UN, 0.0), (0, UN, 0.0), (0, 60.0, 4.0), (0, 72000.0, 8.0), (0, 72000.0, 8.2),
            (0, FD, 6.0)]
    fit = fake_fit(rows)
    both = ControlConfig(fulldeut_at=72000.0)
    fit = dataclasses.replace(fit, role=apply_control_roles(fit.role, fit.time, both))
    assert fit_controls(fit, np.full(6, 10.0), both).mfd[0] == pytest.approx((8.0 + 8.2 + 6) / 3)
    only = ControlConfig(fulldeut_at=72000.0, fulldeut_marked=False)
    c = fit_controls(fit, np.full(6, 10.0), only)
    assert c.n_fd[0] == 2 and c.mfd[0] == pytest.approx(8.1)
    assert list(fit.role) == ["undeut", "undeut", "", "fulldeut", "fulldeut", "fulldeut"]
    with pytest.raises(ValueError, match="fulldeut_marked = false needs fulldeut_at"):
        ControlConfig(fulldeut_marked=False)
    with pytest.raises(ValueError, match="undeut_marked = false needs undeut_at"):
        ControlConfig(undeut_marked=False)


def test_the_median_fallback_without_an_accepted_control_is_the_buffer_fraction(caplog):
    # a control below its own data (rejected) and none
    rows = [(0, 0, 0.0), (0, 1e6, 1.0), (0, 5, 5.0), (1, 0, 0.0)]
    with caplog.at_level("WARNING"):
        c = table(rows)
    assert c.retention.tolist() == pytest.approx([0.9, 0.9]) and c.fallback.all()
    assert not any("fd_median" in f for f in c.flags)
    assert "no fully deuterated control was accepted" in caplog.text


def test_a_control_outside_its_time_course_is_rejected():
    # fully deuterated below the highest time point (tolerance 0.1 per amide: 1 D here)
    c = table([(0, 0, 0.0), (0, 1e6, 5.0), (0, 60, 3.0), (0, 600, 6.5), (0, 600, 6.3)])
    assert c.flags[0] == "m0_unreplicated;fd_below_data" and c.fallback[0]
    c = table([(0, 0, 0.0), (0, 1e6, 5.6), (0, 600, 6.5)])  # within the tolerance
    assert not c.fallback[0]
    # one bad replicate does not decide: the time point counts by its replicates' median
    c = table([(0, 0, 0.0), (0, 1e6, 5.0), (0, 600, 4.5), (0, 600, 4.6), (0, 600, 9.0)])
    assert not c.fallback[0]
    # undeuterated above the lowest time point: the time point is suspect, the control is kept
    c = table([(0, 0, 0.45), (0, 0, 0.45), (0, 5, -0.8), (0, 1e6, 8.0), (0, 1e6, 8.0)])
    assert c.flags[0] == "data_below_m0" and not c.fallback[0]
    assert c.m0[0] == pytest.approx(0.45)
    # spectra serving as a control (fulldeut_at) are not part of the time course
    from hxex.controls import apply_control_roles

    cfg = ControlConfig(fulldeut_at=600.0)
    fit = fake_fit([(0, 0, 0.0), (0, 0, 0.0), (0, 60, 3.0), (0, 600, 7.0), (0, 600, 7.0)])
    fit = dataclasses.replace(fit, role=apply_control_roles(fit.role, fit.time, cfg))
    c = fit_controls(fit, np.full(5, 10.0), cfg)
    assert not c.fallback[0] and c.mfd[0] == pytest.approx(7.0)


def test_the_run_limit_rejects_an_outlier_and_records_itself():
    ret = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.20]
    rows = [r for d, x in enumerate(ret) for r in ((d, 0, 0.0), (d, 0, 0.0), (d, 1e6, 10 * x),
                                                   (d, 1e6, 10 * x))]
    c = table(rows)
    assert c.flags[7] == "retention_outlier;fd_median" and c.fallback[7]
    assert c.retention[7] == pytest.approx(0.50)
    assert not c.fallback[:7].any()
    assert c.limits["median"] == pytest.approx(0.50) and c.limits["n_series"] == 8
    assert c.limits["retention_floor"] == pytest.approx(0.50 - 4 * c.limits["robust_sd"])
    # a set retention_min replaces the run's limit; a wider outlier_sd lets it through
    fixed = table(rows, ControlConfig(retention_min=0.1))
    assert fixed.limits is None and not fixed.fallback.any()
    assert not table(rows, ControlConfig(outlier_sd=20.0)).fallback.any()
    # the limit is kept in the bundle (old bundles have none)
    from hxex.io.bundle import _controls_table, _read_controls

    assert _read_controls(_controls_table(c), CFG.to_dict()).limits == c.limits
    for bad in ({"outlier_sd": 0}, {"time_course_tolerance": -1}, {"sample_tolerance": 0}):
        with pytest.raises(ValueError):
            ControlConfig(**bad)


def test_flags_that_only_warn():
    # two populations in a fully deuterated spectrum: flagged, the control still used
    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 6.0), (0, 1e6, 6.0)]
    c = table(rows, selected_K=[1, 1, 2, 2])  # most (here both) replicates
    assert c.flags[0] == "fd_multimodal" and not c.fallback[0]
    assert table(rows, selected_K=[2, 2, 1, 1]).flags[0] == "m0_multimodal"
    assert table(rows, selected_K=[1, 1, 2, 1]).flags[0] == ""  # one replicate of two
    # the same peptide in three samples: the one far from the other two is flagged
    rows = [r for d, x in enumerate([0.50, 0.52, 0.70]) for r in ((d, 0, 0.0), (d, 0, 0.0),
                                                                   (d, 1e6, 10 * x),
                                                                   (d, 1e6, 10 * x))]
    groups = {0: ("PEPTIDE", "", 2), 1: ("PEPTIDE", "", 2), 2: ("PEPTIDE", "", 2)}
    c = table(rows, groups=groups)
    assert [f for f in c.flags] == ["", "", "fd_disagrees"] and not c.fallback.any()
    assert table(rows).flags.tolist() == ["", "", ""]  # without groups nothing is compared
    pair = table(rows[:8], groups={0: 1, 1: 1})
    assert pair.flags.tolist() == ["", ""]  # 0.50 and 0.52 agree
    # a split fully deuterated envelope whose retention agrees with the other samples' is noted
    # as such (the correction uses the retention); one whose retention differs is fd_disagrees
    # as well
    k = [1, 1, 2, 2] + [1] * 8  # the first sample's fully deuterated replicates choose two
    c = table(rows, groups=groups, selected_K=k)
    assert c.flags[0] == "fd_multimodal;fd_retention_agrees" and not c.fallback[0]
    k = [1] * 8 + [1, 1, 2, 2]  # the third sample's, which is 0.70 against 0.50 and 0.52
    assert table(rows, groups=groups, selected_K=k).flags[2] == "fd_multimodal;fd_disagrees"
    from hxex.report.tables import _control_notes

    assert _control_notes("fd_multimodal;fd_retention_agrees") == "fd_multimodal (retention agrees)"
    assert _control_notes("fd_multimodal;fd_disagrees") == (
        "fd_multimodal (retention differs);fd_disagrees")
    assert _control_notes("fd_multimodal") == "fd_multimodal (retention unchecked)"
    assert _control_notes("fd_missing;fd_median") == ""
    rows2 = rows[:4] + [(1, 0, 0.0), (1, 0, 0.0), (1, 1e6, 7.0), (1, 1e6, 7.0)]
    assert table(rows2, groups={0: 1, 1: 1}).flags.tolist() == ["fd_disagrees"] * 2


def test_a_missing_or_rejected_control_takes_the_other_charge_states():
    """combine_charges = false: a series without a good fully deuterated control takes the
    mean retention of the same peptide's accepted controls at its other charges in the same
    sample (fd_other_charge, still a fallback), before the run's median; another sample's or
    another peptide's controls are not used, and the others keep their own."""
    from hxex.controls import charge_groups

    cfg = ControlConfig(combine_charges=False)

    rows = [(0, 0, 0.0), (0, 0, 0.0),  # WT PEP 3+: no fully deuterated control
            (1, 0, 0.0), (1, 1e6, 5.1), (1, 1e6, 5.3),  # WT PEP 4+: retention 0.52
            (2, 0, 0.0), (2, 1e6, 5.4),  # WT PEP 5+: 0.54, one replicate
            (3, 0, 0.0), (3, 1e6, 4.0), (3, 1e6, 4.0),  # D109H PEP 3+: 0.40
            (4, 0, 0.0), (4, 1e6, 7.0), (4, 1e6, 7.0)]  # WT QQQ 2+: 0.70
    who = {0: ("WT", "PEP", 3), 1: ("WT", "PEP", 4), 2: ("WT", "PEP", 5),
           3: ("D109H", "PEP", 3), 4: ("WT", "QQQ", 2)}
    ids = [r[0] for r in rows]
    charges = charge_groups([who[d][0] for d in ids], [who[d][1] for d in ids],
                            ["" for _ in ids], [who[d][2] for d in ids], ids)
    c = fit_controls(fake_fit(rows), np.full(len(rows), 10.0), cfg, charges=charges)
    assert c.flags[0] == "fd_missing;fd_other_charge" and c.fallback[0]
    assert c.retention[0] == pytest.approx(0.53) and c.mfd[0] == pytest.approx(5.3)
    assert c.mfd_se[0] == pytest.approx(10 * 0.01 / 2)  # 4+: se 0.1 D; 5+: unreplicated
    assert not c.fallback[1:].any()
    assert c.measured.all()  # the other charges' control counts as measured (HISTORY 130)
    # without charges, or switched off: the run's median
    for off in (fit_controls(fake_fit(rows), np.full(len(rows), 10.0), cfg),
                fit_controls(fake_fit(rows), np.full(len(rows), 10.0),
                             ControlConfig(combine_charges=False, fd_other_charge=False),
                             charges=charges)):
        assert off.flags[0] == "fd_missing;fd_median" and not off.measured[0]
        assert off.retention[0] == pytest.approx(np.median([0.52, 0.54, 0.40, 0.70]))
    # a rejected control too, and its fallback flag is replaced
    low = [r if r[:2] != (1, 1e6) else (1, 1e6, 3.0) for r in rows]
    c = fit_controls(fake_fit(low), np.full(len(low), 10.0),
                     ControlConfig(retention_min=0.45, combine_charges=False), charges=charges)
    assert c.flags[1] == "m0_unreplicated;retention_low;fd_other_charge"
    assert c.retention[1] == pytest.approx(0.54)  # 5+ only: 3+ has none and 4+ is rejected
    assert c.retention[0] == pytest.approx(0.54) and c.measured[1]
    # without a measured undeuterated control the series is not measured
    no_m0 = [(0, 100.0, 2.0)] + rows[2:]  # 3+: a time point, no controls
    c = fit_controls(fake_fit(no_m0), np.full(len(no_m0), 10.0), cfg, charges=charges)
    assert "fd_other_charge" in c.flags[0] and "m0_missing" in c.flags[0]
    assert not c.measured[0]
    with pytest.raises(ValueError, match="fd_other_charge"):
        ControlConfig(fd_other_charge="yes")


def test_fully_deuterated_controls_are_combined_over_charge_states():
    """combine_charges (the default; owner, 2026-10-22): every charge state of a peptide in
    one sample takes the plain mean retention of its accepted measured controls; one without
    its own takes it as measured (not a fallback, so never 'corrected differently'); a
    peptide with no accepted control at any charge falls back as before; the undeuterated
    control stays per charge; charge states that disagree are combined and flagged."""
    from hxex.controls import charge_groups

    rows = [(0, 0, 0.0), (0, 0, 0.0),  # WT PEP 3+: no fully deuterated control
            (1, 0, 0.0), (1, 1e6, 5.1), (1, 1e6, 5.3),  # WT PEP 4+: retention 0.52
            (2, 0, 0.1), (2, 1e6, 5.5),  # WT PEP 5+: m0 0.1, retention 0.54, one replicate
            (3, 0, 0.0), (3, 1e6, 4.0), (3, 1e6, 4.0),  # D109H PEP 3+: 0.40, alone
            (4, 0, 0.0), (4, 0, 0.0)]  # WT QQQ 2+: no control at any charge
    who = {0: ("WT", "PEP", 3), 1: ("WT", "PEP", 4), 2: ("WT", "PEP", 5),
           3: ("D109H", "PEP", 3), 4: ("WT", "QQQ", 2)}
    ids = [r[0] for r in rows]
    charges = charge_groups([who[d][0] for d in ids], [who[d][1] for d in ids],
                            ["" for _ in ids], [who[d][2] for d in ids], ids)
    c = fit_controls(fake_fit(rows), np.full(len(rows), 10.0), CFG, charges=charges)
    assert c.retention[:3] == pytest.approx([0.53] * 3)  # the 4+ and 5+ mean, for all three
    assert c.mfd[2] == pytest.approx(0.1 + 5.3)  # its own m0 kept
    assert c.flags[0] == "fd_missing;fd_other_charge" and not c.fallback[0]
    assert c.flags[1] == "m0_unreplicated;fd_charges_combined"  # one m0 replicate
    assert c.flags[2] == "m0_unreplicated;fd_unreplicated;fd_charges_combined"
    assert c.measured[:4].all() and not c.fallback[:4].any()
    assert c.retention[3] == pytest.approx(0.40)  # alone: its own
    assert c.flags[3] == "m0_unreplicated"
    assert c.flags[4] == "fd_missing;fd_median" and c.fallback[4] and not c.measured[4]
    # charge states that disagree are combined and flagged
    far = [r if r[:2] != (2, 1e6) else (2, 1e6, 7.1) for r in rows]  # 5+: 0.70 against 0.52
    c = fit_controls(fake_fit(far), np.full(len(far), 10.0), CFG, charges=charges)
    assert c.retention[1] == pytest.approx(0.61)
    assert all("fd_charges_disagree" in f for f in c.flags[:3])
    from hxex.report.tables import _control_notes

    assert _control_notes(c.flags[1]) == "fd_charges_disagree"
    with pytest.raises(ValueError, match="combine_charges"):
        ControlConfig(combine_charges=1)


def test_fragment_ions_are_not_corrected():
    """Fragment series keep their values (owner, 2026-10-21): flagged uncorrected, no fallback,
    left out of the run's median, and passed through the correction unchanged."""
    from hxex.controls import correct_means, fragment_ids

    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 5.0), (0, 1e6, 5.0),  # retention 0.5
            (1, 0, 0.0), (1, 0, 0.0), (1, 1e6, 6.0), (1, 1e6, 6.0),  # 0.6
            (2, 0, 0.5), (2, 0, 0.5), (2, 1e6, 1.0), (2, 600.0, 3.0),  # fragment, retention 0.05
            (3, 0, 0.0), (3, 0, 0.0)]  # no fully deuterated control
    c = fit_controls(fake_fit(rows), np.full(len(rows), 10.0), CFG, uncorrected={2})
    f = c.rows([2])[0]
    assert c.flags[f] == "uncorrected" and not c.fallback[f] and c.uncorrected[f]
    assert not c.measured[f] and np.isnan(c.retention[f])
    assert c.m0_measured[f] == pytest.approx(0.5) and c.mfd_measured[f] == pytest.approx(1.0)
    assert c.retention[c.rows([3])[0]] == pytest.approx(0.55)  # the fragment is not counted
    out = correct_means(c, [2, 0], np.array([3.0, 2.5]), se=np.array([0.2, 0.2]))
    assert out.mean.tolist() == pytest.approx([3.0, 5.0])  # as measured; 2.5 / 5 * 10
    assert out.se[0] == pytest.approx(0.2) and out.control_se[0] == 0.0
    # fragment series: a fragment scan target, or an ion type other than M
    assert fragment_ids(["", "ion_type:c", ""], [1, 2, 3], fragments=[3]) == {2, 3}


def test_charge_groups_leave_out_fragment_ions_and_the_explicit_mz():
    from hxex.controls import charge_groups

    got = charge_groups(["WT"] * 5, ["PEP"] * 5,
                        ["undeut_mz:500.1", "Hex:1 undeut_mz:400.2", "ion_type:c", "",
                         "undeut_mz:300.0"], [2, 3, 1, 4, 5], [0, 1, 2, 3, 4], fragments=[3])
    assert got == {0: (("WT", "PEP", ""), 2), 1: (("WT", "PEP", "Hex:1"), 3),
                   4: (("WT", "PEP", ""), 5)}


def test_control_replicates_are_checked_before_their_mean():
    """A control replicate far from its replicates' median (replicate_outlier typical sd of the
    run's time-point replicates; three or more) or above the amide count is left out of the
    control's mean, flagged as a warning (owner, 2026-10-22, HISTORY 143)."""
    rows = []
    for d in range(6):  # six series with tight time-point replicates (sd about 0.01 per amide)
        rows += [(d, 100.0, 5.0), (d, 100.0, 5.1), (d, 100.0, 4.9), (d, 0, 0.0), (d, 0, 0.0)]
    rows += [(0, 1e6, 8.0), (0, 1e6, 8.1), (0, 1e6, 3.0)]  # one replicate read something else
    rows += [(1, 1e6, 8.0), (1, 1e6, 8.1), (1, 1e6, 7.9)]
    c = table(rows)
    assert c.flags[0] == "fd_replicate_left_out" and c.mfd[0] == pytest.approx(8.05)
    assert c.flags[1] == "" and c.mfd[1] == pytest.approx(8.0)
    off = table(rows, ControlConfig(replicate_outlier=None))
    assert off.mfd[0] == pytest.approx((8.0 + 8.1 + 3.0) / 3)
    from hxex.report.tables import _control_notes

    assert _control_notes(c.flags[0]) == "fd_replicate_left_out"
    with pytest.raises(ValueError, match="replicate_outlier"):
        ControlConfig(replicate_outlier=-1)


def test_a_multimodal_control_is_checked_against_its_other_charge_states():
    """With no other sample to compare, a multimodal fully deuterated control's retention is
    compared with the same peptide's at its other charge states (HISTORY 145)."""
    from hxex.controls import charge_groups
    from hxex.report.tables import _control_notes

    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 6.0), (0, 1e6, 6.1),  # PEP 2+: 0.605, split
            (1, 0, 0.0), (1, 0, 0.0), (1, 1e6, 6.0), (1, 1e6, 6.2)]  # PEP 3+: 0.61
    ids = [r[0] for r in rows]
    charges = charge_groups(["S"] * len(rows), ["PEP"] * len(rows), [""] * len(rows),
                            [2 if d == 0 else 3 for d in ids], ids)
    k = [1, 1, 2, 2, 1, 1, 1, 1]  # 2+'s fully deuterated replicates choose two populations
    c = fit_controls(fake_fit(rows, selected_K=k), np.full(len(rows), 10.0), CFG,
                     charges=charges)
    assert "fd_charges_agree" in c.flags[0].split(";")
    assert _control_notes(c.flags[0]).startswith(
        "fd_multimodal (retention agrees with its other charge states)")
    far = rows[:2] + [(0, 1e6, 8.0), (0, 1e6, 8.1)] + rows[4:]  # 0.805 against 0.61
    c = fit_controls(fake_fit(far, selected_K=k), np.full(len(far), 10.0), CFG, charges=charges)
    assert "fd_charges_agree" not in c.flags[0]
    assert "fd_multimodal (retention unchecked)" in _control_notes(c.flags[0])
