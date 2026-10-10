# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate engine_baseline_cpu.npz: the prototype engine's results on the engine inputs.

Usage (in the prototype's environment, which has torch; CUDA_VISIBLE_DEVICES= forces the CPU):

    CUDA_VISIBLE_DEVICES= python tests/golden/make_engine_baseline.py \\
        PROTOTYPE_DIR engine_inputs.npz OUT.npz N_BOOT [deterministic]

PROTOTYPE_DIR holds ``fit_multimodal_binomial.py`` (the single-file prototype this package
replaces). Settings are those of the prototype notebook: Gaussian noise at 3 % of the tallest
peak of the undeuterated control, ``n_max = 1.5 x amides``, K = 1 to 3, bootstrap with 3
restarts. The stored results are the frozen baseline the refactored engine is compared with:
selected K, the NLL/BIC/AIC grids, the parameters of every K, and the bootstrap summaries.

The prototype never seeds torch's random generator, and its warm starts (every K >= 2, and the
bootstrap) add ``torch.randn`` noise, so two runs differ for about a third of the K >= 2 fits
(the selected K agrees for 541 of 542 spectra). With ``deterministic`` that noise is set to zero,
which makes the whole run reproducible; the refactored engine with warm-start noise 0 must
reproduce that run.
"""

import sys
import time

import numpy as np


def unragged(flat, off):
    return [flat[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def main(prototype_dir, inputs, out, n_boot, deterministic=False):
    sys.path.insert(0, prototype_dir)
    import fit_multimodal_binomial as fm
    import torch

    if deterministic:  # remove the unseeded warm-start noise
        torch.randn = lambda *size, **kw: torch.zeros(*size, **kw)

    data = np.load(inputs)
    bins = unragged(data["bins"], data["bins_off"])
    obs = unragged(data["obs"], data["bins_off"])
    shapes = unragged(data["shape"], data["shape_off"])
    refs = [
        [int(i) for i in u] if has else None
        for u, has in zip(unragged(data["und"], data["und_off"]), data["has_und"], strict=True)
    ]
    n_max = [1.5 * float(n) for n in data["namides"]]
    print("device", fm.DEVICE, "dtype", fm.DTYPE, "measurements", len(obs), flush=True)

    t0 = time.perf_counter()
    models, info, results = fm.fit_mixture_auto_batch(
        bins, obs, K_max=3, shapes=shapes, noise_model="gaussian", noise_pct=3.0,
        noise_ref_idx=refs, n_max_list=n_max,
    )
    res = {"t_fit": time.perf_counter() - t0, "device": str(fm.DEVICE), "dtype": str(fm.DTYPE),
           "deterministic": deterministic}
    res.update(best_K=info["best_K"], nll_grid=info["nll_grid"], bic_grid=info["bic_grid"],
               aic_grid=info["aic_grid"], obs_scale=info["obs_scale"], n_eff=info["n_eff_per_m"])
    for k in (1, 2, 3):
        batch = results[k]["batch_model"]
        with torch.no_grad():
            res[f"A_{k}"] = batch.amplitudes.cpu().numpy()
            res[f"n_{k}"] = batch.ns.cpu().numpy()
            res[f"p_{k}"] = batch.ps.cpu().numpy()
        res[f"passes_{k}"] = results[k]["passes_min_pop"]
        res[f"fracs_{k}"] = results[k]["fracs_per_m"]
    if n_boot > 0:
        t1 = time.perf_counter()
        boot = fm.bootstrap_errors(
            models, bins, obs, shapes=shapes, N=n_boot, fix_K=True, n_restarts=3, n_max_list=n_max
        )
        res["t_boot"] = time.perf_counter() - t1
        for key in ("A_se", "n_se", "p_se", "mean_se", "frac_se"):
            res["b_" + key] = np.concatenate(boot[key])
        for key in ("A_ci", "n_ci", "p_ci", "mean_ci", "frac_ci"):
            res["b_" + key] = np.concatenate(boot[key])
        res["b_total_lo"] = np.concatenate(boot["total_ci_lo"])
        res["b_total_hi"] = np.concatenate(boot["total_ci_hi"])
        res["b_nsucc"] = np.array(boot["n_boot_success"])
        res["b_K"] = np.array([m.K for m in models])
    np.savez_compressed(out, **res)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), "deterministic" in sys.argv[5:])
