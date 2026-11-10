# SPDX-License-Identifier: GPL-3.0-or-later
"""Levenberg-Marquardt polish, per measurement.

Adam converges slowly in narrow valleys and stops short of the exact minimum. Levenberg-Marquardt
takes only steps that lower the loss (a failed step raises the
damping and is rejected), uses second-order information from the Jacobian of the prediction, and
treats every measurement on its own, so measurements still do not influence each other.

The Gauss-Newton matrix is ``J' W J`` and the gradient ``J' W r`` with ``r = pred - obs``; ``W``
is 1 for Gaussian noise and ``1 / pred`` (the Fisher information) for Poisson noise. With
per-bin noise that follows the prediction (:class:`hxex.noise.BinNoise`), ``W`` is the inverse
variance at each iteration's prediction (iteratively reweighted) and a step is taken only if it
lowers the Gaussian negative log-likelihood ``0.5 sum (r^2 / v + log v)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.func import jvp

from ..hooks import CancelHook, check_cancel
from ..model import MixtureBatch, integer_grid, noise_loss, predict_mixture
from ..model.mixture import capped_p, cut_reading, predicted_variance


def _predict(theta, b, shape_w, n_max, k, grid, cut=None, d_cap=None):
    """Predictions ``(M, B)`` of every measurement from its row of ``theta = [log_A, n_raw,
    logit_p]`` (``(M, 3K)``); ``b`` is ``(B, M)``. ``cut``: ``(scan_w, scan_c)`` of the
    per-scan cut (:func:`hxex.model.mixture.cut_reading`), or None. ``d_cap``: the position cap
    ``(M,)`` (:func:`hxex.model.mixture.capped_p`), or None."""
    n = n_max.unsqueeze(1) * torch.sigmoid(theta[:, k : 2 * k])
    p, logit_p = capped_p(n, theta[:, 2 * k :], d_cap)
    pred = predict_mixture(b, theta[:, :k], n, p, shape_w[:, :, 0], grid, logit_p)
    if cut is not None:
        pred = cut_reading(pred, *cut)
    return pred.T


def _loss(pred, obs, mask, is_gaussian, weight=None):
    """Loss per measurement from ``(M, B)`` arrays; ``is_gaussian`` is ``(M,)``; ``weight``
    ``(M, B)``, the per-bin inverse noise variance of the Gaussian loss (None: 1), or a
    :class:`_Noise` (variances at the prediction: the Gaussian negative log-likelihood)."""
    if isinstance(weight, _Noise):
        var = weight.variance(pred)
        per_bin = 0.5 * ((pred - obs) ** 2 / var + torch.log(var))
    else:
        per_bin = noise_loss(pred, obs, is_gaussian.unsqueeze(1), weight)
    return per_bin.masked_fill(~mask, 0.0).sum(dim=1)


class _Noise:
    """Per-bin noise that follows the prediction, ``(M, B)`` arrays (polish's layout)."""

    def __init__(self, base, floor, model_error):
        self.base, self.floor, self.model_error = base, floor, model_error

    def variance(self, pred):
        return predicted_variance(pred, self.base, self.floor, self.model_error)

    def __getitem__(self, rows):
        return _Noise(self.base[rows], self.floor[rows], self.model_error)


def lm_polish(
    model: MixtureBatch,
    obs: torch.Tensor,
    iters: int,
    lam0: float = 1e-3,
    max_step: float = 2.0,
    tol: float = 1e-9,
    cancel: CancelHook | None = None,
) -> np.ndarray:
    """Improve every measurement of ``model`` in place by up to ``iters`` damped Gauss-Newton
    steps, keeping a step only if it lowers that measurement's loss. Returns the losses ``(M,)``
    (``inf`` stays ``inf``: a measurement whose loss is not finite is left alone).

    Each step is limited to ``max_step`` per parameter (a trust region in the logit coordinates):
    without a limit the first step from a poor start can jump to a distant, worse optimum (for
    example ``n`` pinned at its upper bound) from which no gradient leads back.

    Every 5 iterations the measurements that gained less than ``tol`` (relative loss) and have
    either converged (damping very small) or stalled (damping very large) are taken out, so the
    work goes to those still converging.
    """
    k = model.n_components
    with torch.no_grad():
        theta_all = torch.cat([model.log_A, model.n_raw, model.logit_p], dim=1).detach().clone()
        loss_all = None
    obs_all = obs.T
    n_meas = theta_all.shape[0]
    p_dim = theta_all.shape[1]
    eye = torch.eye(p_dim, dtype=theta_all.dtype, device=theta_all.device)

    def make_fns(b, shape_w, n_max, mask, obs_m, is_gauss, wgt, cut, d_cap):
        bt = b.T  # (B, M), as the model holds it
        grid = integer_grid(bt, shape_w.shape[0])

        def predict(th):
            return _predict(th, bt, shape_w, n_max, k, grid, cut, d_cap)

        def pred_and_jac(th):
            # each measurement's prediction depends on its own row only, so one forward-mode
            # pass per parameter column gives that column of every measurement's Jacobian
            cols = []
            with warnings.catch_warnings():
                # torch's own forward-mode rules are built with torch.jit.script, which recent
                # torch deprecates: the warning (once per process) is about torch, not hxex
                warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated",
                                        category=FutureWarning)
                for q in range(p_dim):
                    tangent = torch.zeros_like(th)
                    tangent[:, q] = 1.0
                    pred, col = jvp(predict, (th,), (tangent,))
                    cols.append(col)
            return pred, torch.stack(cols, dim=2)  # (M, B), (M, B, 3K)

        def loss_of(th):
            return _loss(predict(th), obs_m, mask, is_gauss, wgt)

        return pred_and_jac, loss_of

    active = torch.arange(n_meas, device=theta_all.device)
    b, mask, obs_m = model.b.T, model.mask.T, obs_all
    shape_w, n_max = model.shape_w, model.n_max
    is_gauss = model.is_gaussian
    wgt = (_Noise(model.base_var.T, model.floor_var.T, model.model_error)
           if model.follows_prediction else model.weight.T)
    cut = None if model.scan_w is None else (model.scan_w, model.scan_c)
    d_cap = model.d_cap
    pred_and_jac, loss_of = make_fns(b, shape_w, n_max, mask, obs_m, is_gauss, wgt, cut, d_cap)
    theta = theta_all
    with torch.no_grad():
        loss = loss_of(theta)
        loss_all = loss.clone()
    lam = torch.full_like(loss, lam0)
    loss_ref = loss.clone()
    for it in range(iters):
        if it % 5 == 0:
            check_cancel(cancel)
        with torch.no_grad():
            pred, jac = pred_and_jac(theta)
            jac = jac * mask.unsqueeze(-1)
            r = (pred - obs_m) * mask
            gauss_w = 1.0 / wgt.variance(pred) if isinstance(wgt, _Noise) else wgt
            w = torch.where(is_gauss.unsqueeze(1), gauss_w, 1.0 / pred.clamp(min=1e-10))
            g = torch.einsum("mbp,mb->mp", jac, w * r)
            h = torch.einsum("mbp,mb,mbq->mpq", jac, w, jac)
            damped = h + lam[:, None, None] * (
                torch.diagonal(h, dim1=1, dim2=2)[:, :, None] * eye + 1e-12 * eye
            )
            delta = torch.linalg.solve_ex(damped, -g.unsqueeze(-1))[0].squeeze(-1)
            cand = theta + delta.clamp(-max_step, max_step)
            cand_loss = loss_of(cand)
            ok = torch.isfinite(cand_loss) & torch.isfinite(loss) & (cand_loss < loss)
            theta = torch.where(ok.unsqueeze(1), cand, theta)
            loss = torch.where(ok, cand_loss, loss)
            lam = torch.where(ok, (lam * 0.3).clamp(min=1e-12), (lam * 5.0).clamp(max=1e12))
            if (it + 1) % 5 == 0 and it + 1 < iters:
                gain = (loss_ref - loss) / loss.abs().clamp(min=1.0)
                # take a measurement out only if it gained nothing AND is either converged (steps
                # accepted, damping shrunk) or hopelessly stuck (damping huge); a run of rejected
                # steps raises the damping gradually and must be given time to recover
                done = (gain <= tol) & ((lam < 1e-5) | (lam > 1e3))
                keep = ~done | ~torch.isfinite(gain)
                loss_ref = loss.clone()
                if not bool(keep.all()):
                    theta_all[active] = theta  # store the converged rows
                    loss_all[active] = loss
                    if not bool(keep.any()):
                        active = active[:0]
                        break
                    active = active[keep]
                    theta, loss, lam, loss_ref = theta[keep], loss[keep], lam[keep], loss_ref[keep]
                    b, mask, obs_m, wgt = b[keep], mask[keep], obs_m[keep], wgt[keep]
                    shape_w, n_max, is_gauss = shape_w[:, keep], n_max[keep], is_gauss[keep]
                    if cut is not None:
                        cut = (cut[0][keep], cut[1][:, keep])
                    if d_cap is not None:
                        d_cap = d_cap[keep]
                    pred_and_jac, loss_of = make_fns(b, shape_w, n_max, mask, obs_m, is_gauss,
                                                     wgt, cut, d_cap)
    with torch.no_grad():
        if len(active):
            theta_all[active] = theta
            loss_all[active] = loss
        model.log_A.data.copy_(theta_all[:, :k])
        model.n_raw.data.copy_(theta_all[:, k : 2 * k])
        model.logit_p.data.copy_(theta_all[:, 2 * k :])
    return loss_all.detach().cpu().numpy().astype(float)
