# SPDX-License-Identifier: GPL-3.0-or-later
"""The Levenberg-Marquardt polish."""

import numpy as np
import pytest
import torch

from hxex.fit import FitConfig, Measurements, fit_models
from hxex.fit.optimise import prepare, run_adam
from hxex.fit.polish import lm_polish
from hxex.model import Backend, MixtureBatch, PaddedBatch, shape_weights
from hxex.noise import NoiseConfig
from test_fit import BINS, SHAPE, synthetic  # noqa: I001

CPU = Backend("cpu", "float64")


def trapped_model(n_true, p_true, n_start, p_start, amplitude=1e5, seed=0, noise=0.004):
    """A one-component model started at (n_start, p_start) and data from (n_true, p_true)."""
    n_max = 1.2 * 14.0
    batch = PaddedBatch.from_arrays([BINS], [np.zeros(len(BINS))], [n_max], CPU)
    w = shape_weights([SHAPE], 1, CPU)
    truth = MixtureBatch(
        batch, w, torch.log(torch.tensor([[amplitude]])).double(),
        torch.tensor([[np.log(n_true / (n_max - n_true))]]).double(),
        torch.tensor([[np.log(p_true / (1 - p_true))]]).double(), "gaussian",
    )
    with torch.no_grad():
        pred = truth.predict()[:, 0].numpy()
    rng = np.random.default_rng(seed)
    obs = pred + rng.normal(0, noise * pred.max(), len(pred))
    scale = 0.01 * obs.max()
    batch = PaddedBatch.from_arrays([BINS], [obs / scale], [n_max], CPU)
    model = MixtureBatch(
        batch, w, torch.log(torch.tensor([[amplitude / scale]])).double(),
        torch.tensor([[np.log(n_start / (n_max - n_start))]]).double(),
        torch.tensor([[np.log(p_start / (1 - p_start))]]).double(), "gaussian",
    )
    return model, batch.obs


def test_polish_never_raises_the_loss_and_improves_an_unconverged_fit():
    model, obs = trapped_model(9.7, 0.5, 8.0, 0.4)
    run_adam(model, obs, 30, 0.05, 0.002, "per_measurement")  # deliberately short
    with torch.no_grad():
        before = model.nll(obs).numpy().copy()
    after = lm_polish(model, obs, 40)
    assert after[0] < before[0] and after[0] == pytest.approx(model.nll(obs).item(), rel=1e-9)
    np.testing.assert_allclose(model.ns.item(), 9.7, atol=0.3)  # reaches the truth
    np.testing.assert_allclose(model.ps.item(), 0.5, atol=0.03)
    again = lm_polish(model, obs, 20)
    assert again[0] <= after[0] + 1e-12  # monotone


def test_polish_matches_a_reference_solver():
    from scipy.optimize import least_squares

    model, obs = trapped_model(8.3, 0.6, 7.5, 0.5, seed=2)
    y = obs[:, 0].numpy()
    start = np.array([model.log_A.item(), model.n_raw.item(), model.logit_p.item()])
    polished = lm_polish(model, obs, 60)[0]

    def residual(theta):
        m = MixtureBatch(
            PaddedBatch(model.b, obs, model.mask, model.n_max), model.shape_w,
            torch.tensor([[theta[0]]]), torch.tensor([[theta[1]]]), torch.tensor([[theta[2]]]),
            "gaussian",
        )
        with torch.no_grad():
            return m.predict()[:, 0].numpy() - y

    ref = least_squares(residual, start, method="lm", xtol=1e-13, ftol=1e-13, gtol=1e-13)
    assert polished == pytest.approx(0.5 * float((ref.fun**2).sum()), rel=1e-3)


def test_polish_is_per_measurement():
    specs = [[(3.0, 0.5), (11.0, 0.5)], [(6.0, 1.0)], [(2.0, 0.3), (7.0, 0.4), (12.5, 0.3)]]
    meas, _ = synthetic(specs)
    cfg = FitConfig(k_min=2, k_max=2, noise=NoiseConfig("gaussian", sd=1500.0, percent=None),
                    backend=CPU, adam_steps=60, n_restarts=1, polish_steps=0)
    prep = prepare(meas.bins, meas.observed, meas.shapes, meas.n_max,
                   np.full(3, 1500.0), "gaussian", CPU)

    def build(rows):
        sub = PaddedBatch.from_arrays(
            [prep.bins[i] for i in rows], [prep.observed[i] for i in rows],
            [prep.n_max[i] for i in rows], CPU)
        w = shape_weights([prep.shapes[i] for i in rows], 2, CPU)
        init = torch.zeros(len(rows), 2, dtype=torch.float64)
        m = MixtureBatch(sub, w, init + torch.tensor([10.0, 10.0]).double(),
                         init + torch.tensor([0.0, 1.0]).double(),
                         init + torch.tensor([-1.0, 1.0]).double(), "gaussian")
        return m, sub.obs

    full, obs_full = build([0, 1, 2])
    alone, obs_alone = build([1])
    lf = lm_polish(full, obs_full, 40)
    la = lm_polish(alone, obs_alone, 40)
    assert lf[1] == pytest.approx(la[0], rel=1e-8)
    np.testing.assert_allclose(full.parameters_numpy()["n"][1], alone.parameters_numpy()["n"][0],
                               rtol=1e-7)
    assert cfg.polish_steps == 0


def test_polish_poisson_and_non_finite():
    model, obs = trapped_model(9.7, 0.5, 8.0, 0.4)
    model.set_noise_kind("poisson")
    obs = obs.abs() + 1.0
    with torch.no_grad():
        before = model.nll(obs).item()
    assert lm_polish(model, obs, 30)[0] < before
    # a measurement whose loss is not finite is left alone
    meas, _ = synthetic([[(6.0, 1.0)], [(6.0, 1.0)]])
    o = [x / 1500.0 for x in meas.observed]
    o[0][3] = np.nan
    batch = PaddedBatch.from_arrays(meas.bins, o, meas.n_max, CPU)
    zero = torch.zeros(2, 1, dtype=torch.float64)
    two = MixtureBatch(
        batch, shape_weights(meas.shapes, 1, CPU), zero + 10.0, zero, zero, "gaussian"
    )
    before_params = two.parameters_numpy()["n"][0].copy()
    losses = lm_polish(two, batch.obs, 20)
    assert not np.isfinite(losses[0]) and np.isfinite(losses[1])
    np.testing.assert_array_equal(two.parameters_numpy()["n"][0], before_params)


def test_the_polish_leaves_the_start_that_was_a_trap_for_the_hard_cut_off():
    """Data from n = 14, p = 0.93, started at n = 13.93, p = 0.975: with a mask at b <= n bin 14 had
    no probability there, and the loss was far from its minimum with no gradient to show the way."""
    model, obs = trapped_model(14.0, 0.93, 13.93, 0.975)
    loss = lm_polish(model, obs, 60)[0]
    assert loss < 60 and model.ps.item() == pytest.approx(0.93, abs=0.02)  # near the noise floor


def test_n_is_not_restricted_to_integers():
    """Data from a non-integer n: the fitted n stays a free real number."""
    meas, _ = synthetic([[(11.0, 1.0)]])  # n = 14 truth in the helper; build our own below
    n_true, p_true = 13.4, 0.82
    model, obs = trapped_model(n_true, p_true, 12.0, 0.6, seed=4)
    run_adam(model, obs, 300, 0.05, 0.002, "per_measurement")
    lm_polish(model, obs, 40)
    n_fit = model.ns.item()
    assert abs(n_fit - n_true) < 0.5 and abs(n_fit - round(n_fit)) > 0.05  # not snapped to 13 or 14
    assert meas is not None


def test_fit_models_polish_lowers_the_loss_and_can_be_switched_off():
    meas, _ = synthetic([[(11.0, 1.0)], [(3.0, 0.5), (11.0, 0.5)], [(6.0, 1.0)]])
    base = dict(noise=NoiseConfig("gaussian", percent=1.0), backend=CPU, adam_steps=120,
                n_restarts=2, k_max=2)
    off = fit_models(meas, FitConfig(polish_steps=0, **base))
    on = fit_models(meas, FitConfig(**base))
    assert FitConfig().polish_steps == 40
    for k in (1, 2):
        assert np.all(on.fits[k].nll <= off.fits[k].nll + 1e-9)
    assert np.any(on.fits[2].nll < off.fits[2].nll)


def test_config_polish_validation():
    with pytest.raises(ValueError, match="polish_steps"):
        FitConfig(polish_steps=-1)
    assert Measurements is not None
