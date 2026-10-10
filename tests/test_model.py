# SPDX-License-Identifier: GPL-3.0-or-later
"""Model tests: independent checks (scipy, numpy) and the prototype's values as the oracle."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import binom

from hxex.model import (
    Backend,
    MixtureBatch,
    PaddedBatch,
    aic,
    bic,
    convolve,
    log_binom_pmf,
    logit,
    n_params,
    normalise_shape,
    shape_weights,
)

GOLD = np.load(Path(__file__).parent / "golden" / "model_reference.npz")
CPU64 = Backend("cpu", "float64")


def gold_inputs(backend=CPU64):
    sizes = GOLD["sizes"]
    obs = [GOLD["obs_padded"][: s, i] for i, s in enumerate(sizes)]
    bins = [np.arange(s, dtype=float) for s in sizes]
    shapes = [GOLD[f"shape_{i}"] for i in range(len(sizes))]
    batch = PaddedBatch.from_arrays(bins, obs, GOLD["n_max"], backend)
    shape_w = shape_weights(shapes, GOLD["log_A"].shape[1], backend)
    model = MixtureBatch(
        batch, shape_w, backend.tensor(GOLD["log_A"]), backend.tensor(GOLD["n_raw"]),
        backend.tensor(GOLD["logit_p"]),
    )
    return batch, model, shapes


# --- backend --------------------------------------------------------------------------------

def test_backend_resolution():
    cpu = Backend("cpu")
    assert cpu.torch_device().type == "cpu" and cpu.torch_dtype() == torch.float64
    assert Backend("cpu", "float32").torch_dtype() == torch.float32
    auto = Backend()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert auto.torch_device().type == expected
    assert auto.torch_dtype() == (torch.float32 if expected == "cuda" else torch.float64)


def test_backend_tensor_and_serialisation():
    b = Backend("cpu", "float32")
    t = b.tensor([1, 2, 3])
    assert t.dtype == torch.float32 and t.device.type == "cpu"
    assert b.tensor(torch.ones(2, dtype=torch.float64)).dtype == torch.float32
    assert Backend.from_dict(b.to_dict()) == b


@pytest.mark.parametrize("kwargs", [{"dtype": "float16"}, {"device": "tpu"}, {"device": "gpu"}])
def test_backend_validation(kwargs):
    with pytest.raises(ValueError):
        Backend(**kwargs)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_backend_cuda_unavailable():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        Backend("cuda").torch_device()


# --- binomial pmf and convolution -----------------------------------------------------------

def test_log_pmf_matches_scipy_for_integer_n():
    b = torch.arange(0.0, 13.0, dtype=torch.float64).unsqueeze(1).repeat(1, 2)  # (13, 2)
    n = torch.tensor([[5.0, 8.0], [10.0, 3.0]], dtype=torch.float64)
    p = torch.tensor([[0.3, 0.6], [0.5, 0.9]], dtype=torch.float64)
    out = log_binom_pmf(b, n, p).numpy()  # (13, 2, 2)
    for m in range(2):
        for k in range(2):
            valid = b[:, m].numpy() <= n[m, k].item()
            expected = binom.logpmf(b[:, m].numpy(), n[m, k].item(), p[m, k].item())
            np.testing.assert_allclose(out[valid, m, k], expected[valid], rtol=1e-12)
            assert np.all(out[~valid, m, k] == -1e10)


def test_pmf_sums_to_one_for_integer_n():
    b = torch.arange(0.0, 21.0, dtype=torch.float64).unsqueeze(1)
    pmf = log_binom_pmf(b, torch.tensor([[12.0]]).double(), torch.tensor([[0.35]]).double()).exp()
    assert pmf.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_convolve_matches_numpy_convolution_for_integer_n():
    shape = np.array([0.6, 0.3, 0.1])
    bins = np.arange(0.0, 16.0)
    n, p = 9.0, 0.4
    pmf = binom.pmf(np.arange(0, 10), n, p)
    expected = np.convolve(pmf, shape)[: len(bins)]
    expected = np.concatenate([expected, np.zeros(len(bins) - len(expected))])
    w = shape_weights([shape], 1, CPU64)
    f64 = {"dtype": torch.float64}
    out = convolve(
        torch.tensor(bins, **f64).unsqueeze(1), torch.tensor([[n]], **f64),
        torch.tensor([[p]], **f64), w,
    )[:, 0, 0].numpy()
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_normalise_shape_and_weights():
    np.testing.assert_allclose(normalise_shape(np.array([2.0, 2.0])), [0.5, 0.5])
    np.testing.assert_array_equal(normalise_shape(None), [1.0])
    np.testing.assert_array_equal(normalise_shape(np.zeros(3)), np.zeros(3))
    w = shape_weights([np.array([1.0, 1.0]), None, np.ones(4)], 3, CPU64).numpy()
    assert w.shape == (4, 3, 3)
    np.testing.assert_allclose(w[:, 0, 0], [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(w[:, 1, 2], [1.0, 0, 0, 0])
    np.testing.assert_allclose(w[:, 2, 1], [0.25] * 4)
    assert np.all(w[:, :, 0] == w[:, :, 2])  # the same shape for every component


def test_padded_batch():
    batch = PaddedBatch.from_arrays(
        [np.arange(3.0), np.arange(5.0)], [np.ones(3), 2 * np.ones(5)], [4.0, 6.0], CPU64
    )
    assert batch.b.shape == (5, 2) and batch.n_measurements == 2
    assert batch.mask[:, 0].tolist() == [True] * 3 + [False] * 2
    assert batch.obs[3:, 0].tolist() == [0.0, 0.0] and batch.b[3:, 0].tolist() == [0.0, 0.0]
    assert batch.n_max.tolist() == [4.0, 6.0]
    with pytest.raises(ValueError, match="3 bins but 2"):
        PaddedBatch.from_arrays([np.arange(3.0)], [np.ones(2)], [4.0], CPU64)
    with pytest.raises(ValueError, match="one entry per measurement"):
        PaddedBatch.from_arrays([np.arange(3.0)], [np.ones(3)], [4.0, 5.0], CPU64)


# --- the prototype as oracle ----------------------------------------------------------------

def test_pmf_matches_the_prototype_for_every_bin_it_gives_probability():
    """The prototype masks b > n; hxex also gives the bin just above n (n < b < n + 1) its
    gamma-function probability. Everywhere else the two are the same."""
    batch, model, _ = gold_inputs()
    ours = log_binom_pmf(batch.b, model.ns, model.ps).detach().numpy()
    gold, n = GOLD["log_pmf"], GOLD["ns"][None]  # (B, M, K) and (1, M, K)
    b = GOLD["b_padded"][:, :, None]
    in_proto = gold > -1e9
    np.testing.assert_allclose(ours[in_proto], gold[in_proto], rtol=1e-12)
    extra = ~in_proto & (b < n + 1) & (b >= 0)  # the one bin above n
    assert extra.any() and np.all(ours[extra] > -1e9)
    assert np.all(ours[~in_proto & ~extra] == -1e10)  # bins from n + 1 up have no probability


def test_values_match_the_prototype_at_near_integer_n():
    """With n just above an integer the extra bin carries about 1e-9, so predictions and losses
    can be compared with the prototype directly."""
    sizes = GOLD["sizes"]
    obs = [GOLD["obs_padded"][: s, i] for i, s in enumerate(sizes)]
    bins = [np.arange(s, dtype=float) for s in sizes]
    shapes = [GOLD[f"shape_{i}"] for i in range(len(sizes))]
    batch = PaddedBatch.from_arrays(bins, obs, GOLD["n_max"], CPU64)
    model = MixtureBatch(
        batch, shape_weights(shapes, GOLD["log_A"].shape[1], CPU64), CPU64.tensor(GOLD["log_A"]),
        CPU64.tensor(GOLD["n_raw_int"]), CPU64.tensor(GOLD["logit_p"]),
    )
    np.testing.assert_allclose(model.ns.detach().numpy(), GOLD["ns_int"], rtol=1e-12)
    with torch.no_grad():
        np.testing.assert_allclose(
            convolve(batch.b, model.ns, model.ps, model.shape_w).numpy(), GOLD["convolved_int"],
            rtol=1e-6, atol=1e-12,
        )
        np.testing.assert_allclose(model.predict().numpy(), GOLD["predict_int"], rtol=1e-6)
        for kind in ("gaussian", "poisson"):
            model.noise_kind = kind
            np.testing.assert_allclose(
                model.nll(batch.obs).numpy(), GOLD[f"nll_{kind}_int"], rtol=1e-6
            )


# --- the mixture ----------------------------------------------------------------------------

def test_parameters_stay_inside_their_bounds():
    batch, model, _ = gold_inputs()
    with torch.no_grad():
        model.n_raw.fill_(-50.0)
        model.logit_p.fill_(50.0)
        model.log_A.fill_(-3.0)
    assert torch.all(model.ns >= 0) and torch.all(model.ns <= batch.n_max.unsqueeze(1))
    assert torch.all(model.ps <= 1.0) and torch.all(model.amplitudes > 0)
    with torch.no_grad():
        model.n_raw.fill_(50.0)
    np.testing.assert_allclose(model.ns.detach().numpy()[:, 0], batch.n_max.numpy())


def test_gaussian_nll_matches_the_formula_and_ignores_padding():
    batch, model, _ = gold_inputs()
    with torch.no_grad():
        pred = model.predict().numpy()
        obs = batch.obs.numpy().copy()
        mask = batch.mask.numpy()
        expected = (0.5 * (pred - obs) ** 2 * mask).sum(axis=0)
        np.testing.assert_allclose(model.nll(batch.obs).numpy(), expected, rtol=1e-12)
        obs[~mask] = 1e9  # garbage in the padded positions changes nothing
        np.testing.assert_allclose(
            model.nll(torch.tensor(obs)).numpy(), expected, rtol=1e-12
        )


def test_poisson_nll_matches_the_formula():
    batch, model, _ = gold_inputs()
    model.noise_kind = "poisson"
    with torch.no_grad():
        pred = np.maximum(model.predict().numpy(), 1e-10)
        obs, mask = batch.obs.numpy(), batch.mask.numpy()
        expected = ((pred - obs * np.log(pred)) * mask).sum(axis=0)
        np.testing.assert_allclose(model.nll(batch.obs).numpy(), expected, rtol=1e-12)


def test_gradients_are_finite_and_noise_kind_is_checked():
    batch, model, _ = gold_inputs()
    model.nll(batch.obs).sum().backward()
    for param in model.parameters():
        assert torch.isfinite(param.grad).all()
    with pytest.raises(ValueError, match="noise_kind"):
        MixtureBatch(batch, model.shape_w, model.log_A, model.n_raw, model.logit_p, "laplace")


def test_components_sum_to_the_prediction():
    _, model, _ = gold_inputs()
    with torch.no_grad():
        np.testing.assert_allclose(
            model.components().sum(dim=2).numpy(), model.predict().numpy(), rtol=1e-12
        )
    assert model.n_measurements == 6 and model.n_components == 3


def test_parameters_numpy_and_float32_agreement():
    _, model, _ = gold_inputs()
    par = model.parameters_numpy()
    assert set(par) == {"A", "n", "p"} and all(v.dtype == np.float64 for v in par.values())
    b32, m32, _ = gold_inputs(Backend("cpu", "float32"))
    p32 = m32.parameters_numpy()
    np.testing.assert_allclose(p32["n"], par["n"], rtol=2e-5)
    with torch.no_grad():
        np.testing.assert_allclose(
            m32.predict().numpy(), model.predict().numpy(), rtol=5e-4, atol=1e-3
        )


def test_criteria():
    assert n_params(3) == 9
    assert bic(2, 100.0, 5.0) == pytest.approx(6 * np.log(100.0) + 10.0)
    assert bic(1, 1.0, 5.0) == pytest.approx(3 * np.log(2.0) + 10.0)  # N is at least 2
    np.testing.assert_allclose(bic(2, np.array([10.0, 20.0]), np.array([1.0, 2.0])),
                               [6 * np.log(10.0) + 2.0, 6 * np.log(20.0) + 4.0])
    assert aic(2, 5.0) == pytest.approx(12.0 + 10.0)


def test_logit_clips():
    np.testing.assert_allclose(logit(np.array([0.5])), [0.0])
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


# --- the cut-off at b < n + 1 --------------------------------------------------------------

def _pmf(n, p, bmax=30):
    b = torch.arange(0.0, float(bmax), dtype=torch.float64).unsqueeze(1)
    n_t, p_t = torch.tensor([[n]], dtype=torch.float64), torch.tensor([[p]], dtype=torch.float64)
    return log_binom_pmf(b, n_t, p_t).exp()[:, 0, 0].numpy()


def test_pmf_is_continuous_in_n():
    near = [_pmf(n, 0.9)[14] for n in (13.999, 14.0)]
    assert abs(near[1] - near[0]) < 1e-3  # no jump where the bin above n enters
    assert _pmf(13.0001, 0.9)[14] < 1e-3  # it starts from zero ...
    values = [_pmf(n, 0.9)[14] for n in np.linspace(13.0, 14.0, 11)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))  # ... and rises smoothly
    assert _pmf(14.0, 0.9)[14] == pytest.approx(0.9**14, rel=1e-12)  # an integer n: the binomial


def test_pmf_is_nearly_normalised_and_centred_between_integers():
    bins = np.arange(30)
    for n in (13.25, 13.5, 13.75):
        pmf = _pmf(n, 0.9)
        assert abs(pmf.sum() - 1.0) < 0.03
        assert abs(np.sum(bins * pmf) / pmf.sum() - n * 0.9) < 0.06  # centre = n p


def test_mixture_loss_is_continuous_in_n_with_finite_gradients():
    w = shape_weights([GOLD["shape_0"]], 1, CPU64)
    bins = np.arange(0.0, 20.0)
    batch = PaddedBatch.from_arrays([bins], [np.ones(len(bins))], [20.0], CPU64)
    out = []
    for n in (13.999, 14.0001):
        n_raw = torch.tensor([[np.log(n / (20.0 - n))]], dtype=torch.float64)
        m = MixtureBatch(batch, w, torch.zeros(1, 1).double(), n_raw,
                         torch.tensor([[2.2]]).double(), "gaussian")
        loss = m.nll(batch.obs)
        loss.sum().backward()
        assert torch.isfinite(m.n_raw.grad).all()
        out.append(loss.item())
    assert abs(out[1] - out[0]) < 1e-3
