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
    integer_grid,
    log_binom_norm,
    log_binom_pmf,
    logit,
    n_params,
    normalise_shape,
    predict_mixture,
    shape_weights,
)

GOLD = np.load(Path(__file__).parent / "golden" / "model_reference.npz")
CPU64 = Backend("cpu", "float64")
CPU32 = Backend("cpu", "float32")


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
    assert cpu.torch_device().type == "cpu" and cpu.torch_dtype() == torch.float32  # the default
    assert Backend("cpu", "float64").torch_dtype() == torch.float64  # the option
    auto = Backend()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert auto.torch_device().type == expected and auto.torch_dtype() == torch.float32


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
    norm = log_binom_norm(model.ns, model.ps).detach().numpy()  # the prototype does not normalise
    ours = log_binom_pmf(batch.b, model.ns, model.ps).detach().numpy()
    ours = np.where(ours > -1e9, ours + norm[None], ours)
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
            model.set_noise_kind(kind)
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
    model.set_noise_kind("poisson")
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


def test_pmf_is_normalised_at_any_n_and_p_and_centred_between_integers():
    """Without the normalisation the bin above n carries (1 - p)^(n - b) with a negative
    exponent, and the mass reached 1.7 at n = 13.5, p = 0.99 and 5 at n = 0.5, p = 0.99."""
    bins = np.arange(30)
    for n in (0.5, 13.25, 13.5, 13.75):
        for p in (0.5, 0.9, 0.97, 0.99, 0.999):
            pmf = _pmf(n, p)
            assert pmf.sum() == pytest.approx(1.0, abs=1e-12), (n, p)
    for n in (13.25, 13.5, 13.75):
        pmf = _pmf(n, 0.9)
        assert abs(np.sum(bins * pmf) - n * 0.9) < 0.06  # centre = n p (within 0.05 D at p <= 0.9)


def test_components_are_normalised_in_float32_where_sigmoid_saturates():
    """With logit_p = 30 float32's sigmoid is exactly 1; the log terms from the logit keep the
    pmf finite and normalised, with finite gradients."""
    b = torch.arange(0.0, 20.0).unsqueeze(1)
    n = torch.tensor([[13.5]], requires_grad=True)
    logit_p = torch.tensor([[30.0]], requires_grad=True)
    pmf = log_binom_pmf(b, n, torch.sigmoid(logit_p), logit_p).exp()[:, 0, 0]
    assert torch.isfinite(pmf).all() and pmf.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert pmf[14].item() == pytest.approx(1.0, abs=1e-5)  # all the mass in the bin above n
    (pmf * b[:, 0]).sum().backward()
    assert torch.isfinite(n.grad).all() and torch.isfinite(logit_p.grad).all()
    w = shape_weights([np.array([0.7, 0.3])], 1, CPU32)
    batch = PaddedBatch.from_arrays([np.arange(20.0)], [np.ones(20)], [20.0], CPU32)
    conv = convolve(batch.b, n.detach(), torch.sigmoid(logit_p.detach()), w, None, logit_p.detach())
    assert conv[:, 0, 0].sum().item() == pytest.approx(1.0, abs=1e-5)


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


@pytest.mark.parametrize("layout", ["consecutive", "integer", "fractional"])
def test_the_fast_convolutions_equal_the_direct_one(layout):
    """With whole-number bins the pmf is evaluated once per integer (gathered, or as windows when
    every measurement's bins are 0, 1, 2, ...): the same values and gradients as evaluating it at
    every shifted bin, padding included."""
    rng = np.random.default_rng(3)
    sizes = [12, 7, 20, 1, 15]
    if layout == "consecutive":
        bins = [np.arange(s, dtype=float) for s in sizes]
    elif layout == "integer":  # whole numbers, but not 0, 1, 2, ... (an offset and a gap)
        bins = [np.arange(s, dtype=float) + 2 for s in sizes]
        bins[2] = np.delete(np.arange(21, dtype=float), 5)
    else:
        bins = [np.arange(s, dtype=float) + 0.5 for s in sizes]
    obs = [rng.uniform(0, 5, len(b)) for b in bins]
    shapes = [rng.uniform(0.1, 1, int(rng.integers(1, 9))) for _ in bins]
    k = 3
    batch = PaddedBatch.from_arrays(bins, obs, np.full(len(bins), 25.0), CPU64)
    shape_w = shape_weights(shapes, k, CPU64)
    grid = integer_grid(batch.b, shape_w.shape[0])
    assert (grid is None) == (layout == "fractional")
    if grid is not None:
        assert grid.consecutive == (layout == "consecutive")
    n = torch.tensor(rng.uniform(0.3, 24, (len(bins), k)), requires_grad=True)
    p = torch.tensor(rng.uniform(0.05, 0.9, (len(bins), k)), requires_grad=True)
    direct = convolve(batch.b, n, p, shape_w)
    fast = convolve(batch.b, n, p, shape_w, grid)
    np.testing.assert_allclose(fast.detach().numpy(), direct.detach().numpy(), rtol=1e-13,
                               atol=1e-300)
    weights = torch.tensor(rng.normal(size=direct.shape))
    g_direct = torch.autograd.grad((direct * weights).sum(), (n, p))
    g_fast = torch.autograd.grad((convolve(batch.b, n, p, shape_w, grid) * weights).sum(), (n, p))
    for a, b in zip(g_fast, g_direct, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-9, atol=1e-12)
    zeros = CPU64.tensor(np.zeros((len(bins), k)))
    model = MixtureBatch(batch, shape_w, zeros, zeros, zeros)
    assert (model.grid is None) == (layout == "fractional")


@pytest.mark.parametrize("layout", ["consecutive", "integer", "fractional"])
def test_the_mixture_prediction_is_the_sum_of_the_components(layout):
    """predict_mixture convolves the amplitude-weighted mixture once instead of each component."""
    rng = np.random.default_rng(5)
    sizes = [12, 7, 20, 1, 15]
    if layout == "consecutive":
        bins = [np.arange(s, dtype=float) for s in sizes]
    elif layout == "integer":
        bins = [np.arange(s, dtype=float) + 2 for s in sizes]
    else:
        bins = [np.arange(s, dtype=float) + 0.5 for s in sizes]
    shapes = [rng.uniform(0.1, 1, int(rng.integers(1, 9))) for _ in bins]
    k = 3
    batch = PaddedBatch.from_arrays(bins, [np.ones(len(b)) for b in bins], np.full(5, 25.0), CPU64)
    shape_w = shape_weights(shapes, k, CPU64)
    grid = integer_grid(batch.b, shape_w.shape[0])
    log_a = torch.tensor(rng.normal(0, 1, (5, k)), requires_grad=True)
    n = torch.tensor(rng.uniform(0.3, 24, (5, k)), requires_grad=True)
    logit_p = torch.tensor(rng.normal(0, 2, (5, k)), requires_grad=True)
    p = torch.sigmoid(logit_p)
    parts = (log_a.exp().unsqueeze(0) * convolve(batch.b, n, p, shape_w, grid, logit_p)).sum(2)
    mixed = predict_mixture(batch.b, log_a, n, p, shape_w[:, :, 0], grid, logit_p)
    np.testing.assert_allclose(mixed.detach().numpy(), parts.detach().numpy(), rtol=1e-12,
                               atol=1e-300)
    w = torch.tensor(rng.normal(size=parts.shape))
    g1 = torch.autograd.grad((parts * w).sum(), (log_a, n, logit_p))
    g2 = torch.autograd.grad((predict_mixture(batch.b, log_a, n, torch.sigmoid(logit_p),
                                              shape_w[:, :, 0], grid, logit_p) * w).sum(),
                             (log_a, n, logit_p))
    for a, b in zip(g2, g1, strict=True):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-9, atol=1e-12)


def test_gradients_are_finite_in_float32_where_p_is_near_1():
    """Above the support the unnormalised terms grow as (1 - p)^(n - b); evaluated before
    masking they overflowed float32 and made the gradient NaN (an optimiser then stopped
    moving n and p for that measurement). Seen at n just above an integer with p near 1."""
    bins = np.arange(30.0)
    batch = PaddedBatch.from_arrays([bins], [np.ones(30)], [9.6], CPU32)
    w = shape_weights([GOLD["shape_0"]], 2, CPU32)
    n = np.array([[7.0017, 4.0]])
    n_raw = np.log(n / (9.6 - n))
    for model_fn in ("nll", "components"):
        m = MixtureBatch(batch, w, CPU32.tensor(np.zeros((1, 2))), CPU32.tensor(n_raw),
                         CPU32.tensor(np.array([[7.44, 16.7]])))
        out = m.nll(batch.obs).sum() if model_fn == "nll" else m.components().sum()
        out.backward()
        for prm in m.parameters():
            assert torch.isfinite(prm.grad).all(), model_fn
