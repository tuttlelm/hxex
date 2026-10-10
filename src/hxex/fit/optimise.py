# SPDX-License-Identifier: GPL-3.0-or-later
"""The batched optimiser: Adam with a cosine learning rate, then L-BFGS, with restarts.

Every measurement is fitted independently but in parallel (the loss is a sum over measurements,
so they do not interact). A measurement whose loss becomes non-finite is restored to its last
finite parameters and masked out of the summed loss, so it cannot stop or corrupt the others; if
it is non-finite in every restart its loss is ``inf`` and its parameters are placeholders.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..hooks import CancelHook, check_cancel
from ..model import Backend, MixtureBatch, PaddedBatch, shape_weights
from .init import Start, kmeans_start
from .polish import lm_polish

log = logging.getLogger(__name__)

_CANCEL_EVERY = 25  # Adam steps between cancel checks

# Total pseudo-counts a spectrum is rescaled to for the k-means start of a Gaussian fit (see
# ``kmeans_start``, which clusters ``round(intensity)`` copies of each bin, at most 2000 per bin).
# The clustering therefore does not depend on the intensity of the data; the amplitude is
# restored afterwards. 1000 is the prototype's value.
KMEANS_PSEUDO_COUNTS = 1000.0


class FitError(RuntimeError):
    """The fit failed for every measurement (non-finite in every restart)."""


@dataclass
class Prepared:
    """The data of a fit, scaled by the noise scale and padded: what ``fit_batch`` works on."""

    batch: PaddedBatch
    bins: Sequence[np.ndarray]
    observed: Sequence[np.ndarray]  # scaled
    shapes: Sequence[np.ndarray]
    n_max: np.ndarray
    noise_kind: str
    keys: np.ndarray  # stable integer per measurement; seeds its random starting points

    @property
    def n_measurements(self) -> int:
        return len(self.bins)


def stable_keys(ids: Sequence[str]) -> np.ndarray:
    """A deterministic non-negative integer for each id (CRC-32 of its UTF-8 bytes)."""
    return np.array([zlib.crc32(str(i).encode()) for i in ids], dtype=np.int64)


def prepare(
    bins: Sequence[np.ndarray],
    observed: Sequence[np.ndarray],
    shapes: Sequence[np.ndarray],
    n_max: np.ndarray,
    scale: np.ndarray,
    noise_kind: str,
    backend: Backend,
    keys: Sequence[int] | None = None,
) -> Prepared:
    """Divide each spectrum by its noise scale and pad the batch.

    ``keys`` (one integer per measurement, default its position) seed the random starting points
    of that measurement, so its fit does not depend on the other measurements in the batch when
    the keys come from stable ids (:func:`stable_keys`).
    """
    scaled = [np.asarray(o, dtype=np.float64) / scale[m] for m, o in enumerate(observed)]
    return Prepared(
        PaddedBatch.from_arrays(bins, scaled, n_max, backend), bins, scaled, shapes,
        np.asarray(n_max, dtype=np.float64), noise_kind,
        np.arange(len(bins)) if keys is None else np.asarray(keys, dtype=np.int64),
    )


def cosine_lr(step: int, total: int, lr0: float, lr1: float) -> float:
    """Cosine-decayed learning rate: ``lr0`` at step 0 down to ``lr1`` at the end."""
    return lr1 + (lr0 - lr1) * 0.5 * (1.0 + math.cos(math.pi * min(step / max(total, 1), 1.0)))


def clip_per_measurement(params, max_norm: float) -> None:
    """Scale the gradient of each measurement so that its own norm is at most ``max_norm``.

    Every parameter is ``(M, K)``, so a measurement's gradient is its row in each parameter.
    Unlike clipping the norm of the whole batch (``torch.nn.utils.clip_grad_norm_``), this does
    not let one measurement's gradient change another's step.
    """
    sq = sum((p.grad**2).sum(dim=1) for p in params if p.grad is not None)
    coef = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(coef.unsqueeze(1))


def _clean_grads(params) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


def run_adam(
    model: MixtureBatch,
    obs: torch.Tensor,
    steps: int,
    lr0: float,
    lr1: float,
    grad_clip: str,
    cancel: CancelHook | None = None,
) -> None:
    """Adam on the summed loss with a cosine-decayed learning rate, updating ``model`` in place.

    A measurement whose loss turns non-finite is restored to its last finite parameters and left
    out of the summed loss for that step.
    """
    params = list(model.parameters())
    snap = [p.detach().clone() for p in params]  # last parameters with a finite loss
    opt = torch.optim.Adam(params, lr=lr0)
    for step in range(steps):
        if step % _CANCEL_EVERY == 0:
            check_cancel(cancel)
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, steps, lr0, lr1)
        opt.zero_grad()
        nll = model.nll(obs)
        bad = ~torch.isfinite(nll)
        if bool(bad.any()):
            with torch.no_grad():  # undo the bad measurements only
                for p, s_ in zip(params, snap, strict=True):
                    p.data[bad] = s_[bad]
            nll = model.nll(obs)
            bad = ~torch.isfinite(nll)
        with torch.no_grad():
            for p, s_ in zip(params, snap, strict=True):
                s_.copy_(torch.where(bad.unsqueeze(1), s_, p.data))
        nll.masked_fill(bad, 0.0).sum().backward()
        _clean_grads(params)
        if grad_clip == "per_measurement":
            clip_per_measurement(params, 5.0)
        else:
            torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()


def fit_batch(
    prep: Prepared,
    k: int,
    backend: Backend,
    *,
    n_restarts: int = 3,
    adam_steps: int = 400,
    lbfgs_steps: int = 100,
    lr_adam: float = 0.05,
    lr_adam_end: float = 0.002,
    seed: int = 0,
    warm: Start | None = None,
    warm_noise: float = 0.10,
    kmeans_after_warm: bool = False,
    grad_clip: str = "global",
    polish_steps: int = 0,
    cancel: CancelHook | None = None,
    on_restart: Callable[[], None] | None = None,
) -> tuple[MixtureBatch, np.ndarray]:
    """Fit ``k`` components to every measurement of ``prep``.

    Restart 0 is ``warm`` (with ``warm_noise`` jitter) if given, else the k-means start; restart 1
    is also the k-means start when ``kmeans_after_warm``; the other restarts are random. Every
    random number of a measurement comes from a generator seeded by ``seed`` and its key. The best
    restart per measurement, by loss, is kept. Returns the model (in scaled units) and the loss of
    each measurement (``inf`` where every restart was non-finite). Raises :class:`FitError` if that
    is the case for all of them.

    ``polish_steps`` > 0 adds a final Levenberg-Marquardt polish of that many iterations
    (:func:`~hxex.fit.polish.lm_polish`).

    ``grad_clip='global'`` clips the gradient norm of the whole batch (the prototype's choice, so
    measurements influence each other); ``'per_measurement'`` clips each measurement's own.
    """
    if grad_clip not in ("global", "per_measurement"):
        raise ValueError("grad_clip must be 'global' or 'per_measurement'")
    m = prep.n_measurements
    dev = backend.torch_device()
    shape_w = shape_weights(prep.shapes, k, backend)
    obs_dev = prep.batch.obs

    best_model: MixtureBatch | None = None
    best_nll = np.full(m, np.inf)

    for restart in range(n_restarts):
        check_cancel(cancel)
        if restart == 0 and warm is not None:
            jitter = np.stack(
                [np.random.default_rng([seed, 1, int(key)]).standard_normal((3, k))
                 for key in prep.keys]
            ) * warm_noise  # (M, 3, K), one stream per measurement
            la = warm.log_A + jitter[:, 0]
            nr = warm.n_raw + jitter[:, 1]
            lp = warm.logit_p + jitter[:, 2]
        elif restart == 0 or (restart == 1 and warm is not None and kmeans_after_warm):
            la, nr, lp = np.zeros((m, k)), np.zeros((m, k)), np.zeros((m, k))
            for i in range(m):
                ob_init, la_shift = prep.observed[i], 0.0
                if prep.noise_kind == "gaussian":
                    # k-means pseudo-samples need count-like weights: normalise for the
                    # clustering, then restore the amplitude scale
                    ob_init = np.clip(ob_init, 0, None)
                    total = max(float(ob_init.sum()), 1e-12)
                    ob_init = ob_init * (KMEANS_PSEUDO_COUNTS / total)
                    la_shift = float(np.log(max(total, 1.0) / KMEANS_PSEUDO_COUNTS))
                a, n_, p_ = kmeans_start(
                    prep.bins[i], ob_init, k, prep.shapes[i], float(prep.n_max[i]),
                    seed=(seed + int(prep.keys[i])) % 2**31,
                )
                la[i], nr[i], lp[i] = a + la_shift, n_, p_
        else:
            la, nr, lp = np.zeros((m, k)), np.zeros((m, k)), np.zeros((m, k))
            for i in range(m):
                rng = np.random.default_rng([seed, 2 + restart, int(prep.keys[i])])
                n_tot = float(np.asarray(prep.observed[i]).sum())
                la[i] = np.log(np.clip(n_tot / k * rng.uniform(0.5, 2.0, k), 1, None))
                nr[i] = rng.normal(0, 2.0, k)
                lp[i] = rng.normal(0, 1.5, k)

        model = MixtureBatch(
            prep.batch, shape_w, backend.tensor(la), backend.tensor(nr), backend.tensor(lp),
            noise_kind=prep.noise_kind,
        )
        # Phase 1: Adam with a cosine-decayed learning rate
        run_adam(model, obs_dev, adam_steps, lr_adam, lr_adam_end, grad_clip, cancel)
        params = list(model.parameters())

        # Phase 2: L-BFGS
        with torch.no_grad():
            nll_pre = model.nll(obs_dev)
            pre_ok = torch.isfinite(nll_pre)
            nll_fallback = torch.where(pre_ok, nll_pre, torch.zeros_like(nll_pre))
            snap2 = [p.detach().clone() for p in params]
        opt2 = torch.optim.LBFGS(
            params, lr=1.0, max_iter=lbfgs_steps, line_search_fn="strong_wolfe"
        )

        def closure(opt=opt2, mdl=model, fallback=nll_fallback, prm=params):
            opt.zero_grad()
            nll = mdl.nll(obs_dev)
            bad = ~torch.isfinite(nll)
            loss = torch.where(bad, fallback, nll).sum()  # a bad measurement is neutral
            loss.backward()
            _clean_grads(prm)
            return loss

        try:
            if lbfgs_steps > 0:
                opt2.step(closure)
        except Exception as exc:  # noqa: BLE001 - L-BFGS may fail; the pre-L-BFGS fit is kept
            log.debug("L-BFGS stopped: %s", exc)

        with torch.no_grad():
            nll_post = model.nll(obs_dev)
            worse = (~torch.isfinite(nll_post)) | ((nll_post > nll_pre) & pre_ok)
            for p, s_ in zip(params, snap2, strict=True):
                p.data[worse] = s_[worse]
            nll_m = torch.where(worse, nll_pre, nll_post).detach().cpu().numpy().astype(float)
        nll_m[~np.isfinite(nll_m)] = np.inf

        if best_model is None:
            best_model, best_nll = model, nll_m.copy()
        else:
            improved = nll_m < best_nll
            if improved.any():
                idx = torch.as_tensor(np.nonzero(improved)[0], device=dev)
                with torch.no_grad():
                    best_model.log_A.data[idx] = model.log_A.data[idx]
                    best_model.n_raw.data[idx] = model.n_raw.data[idx]
                    best_model.logit_p.data[idx] = model.logit_p.data[idx]
                best_nll = np.minimum(best_nll, nll_m)
        if on_restart is not None:
            on_restart()

    assert best_model is not None
    if polish_steps > 0:
        finite = np.isfinite(best_nll)
        polished = lm_polish(best_model, obs_dev, polish_steps, cancel=cancel)
        best_nll = np.where(finite, polished, best_nll)
    bad_final = ~np.isfinite(best_nll)
    if bad_final.all():
        raise FitError(f"all {n_restarts} restarts were non-finite for K={k}")
    if bad_final.any():
        with torch.no_grad():  # finite placeholder parameters
            idx = torch.as_tensor(np.nonzero(bad_final)[0], device=dev)
            best_model.log_A.data[idx] = 0.0
            best_model.n_raw.data[idx] = 0.0
            best_model.logit_p.data[idx] = 0.0
        log.warning(
            "K=%d: %d of %d measurements had a non-finite loss in every restart; their loss "
            "is inf and their parameters are placeholders", k, int(bad_final.sum()), m,
        )
    return best_model, best_nll
