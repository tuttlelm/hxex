# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate model_reference.npz: the prototype's model and noise functions on random inputs.

The prototype masks bins above ``n`` (``b <= n``); hxex gives the bin just above ``n`` its gamma-
function probability (``b < n + 1``), so the two agree for every bin up to ``n`` and differ only
in that one bin. The file therefore holds two sets: random real ``n`` (compared bin by bin) and
``n`` just above an integer (``*_int`` keys: the extra bin is negligible, so predictions and
losses can be compared directly).

Usage (in the prototype's environment; CUDA_VISIBLE_DEVICES= forces the CPU and float64):

    CUDA_VISIBLE_DEVICES= python tests/golden/make_model_golden.py PROTOTYPE_DIR OUT.npz

PROTOTYPE_DIR holds ``fit_multimodal_binomial.py``. The inputs are fixed random numbers; the
outputs of the binomial pmf, the shape convolution, the batch model's prediction and its Gaussian
and Poisson losses, and of the noise scale for several settings, are stored for comparison.
"""

import sys

import numpy as np


def main(prototype_dir, out):
    sys.path.insert(0, prototype_dir)
    import fit_multimodal_binomial as fm
    import torch

    rng = np.random.default_rng(1)
    k, sizes = 3, [9, 12, 8, 14, 10, 11]
    m = len(sizes)
    bins = [np.arange(s, dtype=float) for s in sizes]
    observed = [rng.gamma(2.0, 500.0, s) for s in sizes]
    n_max = rng.uniform(5.0, 20.0, m)
    shapes = [rng.uniform(0.1, 1.0, rng.integers(4, 10)) for _ in range(m)]  # not normalised
    log_a = rng.normal(8.0, 1.0, (m, k))
    n_raw = rng.normal(0.0, 1.0, (m, k))
    logit_p = rng.normal(0.0, 1.0, (m, k))

    b, obs, mask, n_max_t = fm._pad_to_batch(bins, observed, list(n_max))
    shapes_batch = fm._resolve_shapes_batch(shapes, m, k)
    t = lambda x: torch.tensor(x, dtype=fm.DTYPE, device=fm.DEVICE)  # noqa: E731
    model = fm.BinomialMixtureBatch(
        m, k, n_max_t, mask, shapes_batch, t(log_a), t(n_raw), t(logit_p)
    )
    model.set_b_padded(b)
    res = {"sizes": np.array(sizes), "n_max": n_max, "log_A": log_a, "n_raw": n_raw,
           "logit_p": logit_p, "obs_padded": obs.cpu().numpy(), "b_padded": b.cpu().numpy(),
           "mask": mask.cpu().numpy()}
    for i in range(m):
        res[f"shape_{i}"] = shapes[i]
    with torch.no_grad():
        ns, ps = model.ns, model.ps
        res["ns"], res["ps"] = ns.cpu().numpy(), ps.cpu().numpy()
        res["log_pmf"] = fm.log_real_binom_pmf_batch(b, ns, ps).cpu().numpy()
        conv = fm._convolve_with_shape_batch(b, ns, ps, model.shape_w, mask)
        res["convolved"] = conv.cpu().numpy()
        res["predict"] = model.predict(b).cpu().numpy()
        for kind in ("gaussian", "poisson"):
            model.noise_kind = kind
            res[f"nll_{kind}"] = model.poisson_nll(obs).cpu().numpy()

    # a second set with n = integer + 1e-9, where the bin just above n carries about 1e-9
    n_int = np.maximum(np.floor(n_max[:, None] * rng.uniform(0.2, 0.9, (m, k))), 1.0) + 1e-9
    n_raw_int = np.log(n_int / (n_max[:, None] - n_int))
    model_i = fm.BinomialMixtureBatch(
        m, k, n_max_t, mask, shapes_batch, t(log_a), t(n_raw_int), t(logit_p)
    )
    model_i.set_b_padded(b)
    res["n_raw_int"] = n_raw_int
    with torch.no_grad():
        ns_i, ps_i = model_i.ns, model_i.ps
        res["ns_int"] = ns_i.cpu().numpy()
        res["log_pmf_int"] = fm.log_real_binom_pmf_batch(b, ns_i, ps_i).cpu().numpy()
        conv_i = fm._convolve_with_shape_batch(b, ns_i, ps_i, model_i.shape_w, mask)
        res["convolved_int"] = conv_i.cpu().numpy()
        res["predict_int"] = model_i.predict(b).cpu().numpy()
        for kind in ("gaussian", "poisson"):
            model_i.noise_kind = kind
            res[f"nll_{kind}_int"] = model_i.poisson_nll(obs).cpu().numpy()

    refs = [[0], [0], [0], [3, 4], [3, 4], None]
    refs_idx = [r if r is not None else i for i, r in enumerate(refs)]
    ns_ = fm._noise_scale
    res["noise_gauss_pct"] = ns_(observed, None, 3.0, refs_idx, "gaussian", None)
    res["noise_gauss_self"] = ns_(observed, None, 5.0, "self", "gaussian", None)
    res["noise_gauss_sd_scalar"] = ns_(observed, None, None, None, "gaussian", 123.0)
    res["noise_gauss_sd_array"] = ns_(observed, None, None, None, "gaussian", np.arange(1.0, m + 1))
    res["noise_poisson_neff"] = ns_(observed, 500.0, None, refs_idx, "poisson", None)
    res["noise_poisson_pct"] = ns_(observed, None, 5.0, refs_idx, "poisson", None)
    res["noise_poisson_none"] = ns_(observed, None, None, None, "poisson", None)
    np.savez_compressed(out, **res)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
