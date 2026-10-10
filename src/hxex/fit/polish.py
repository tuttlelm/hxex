# SPDX-License-Identifier: GPL-3.0-or-later
"""Levenberg-Marquardt polish, per measurement.

Adam converges slowly in narrow valleys and stops short of the exact minimum. Levenberg-Marquardt
takes only steps that lower the loss (a failed step raises the
damping and is rejected), uses second-order information from the Jacobian of the prediction, and
treats every measurement on its own, so measurements still do not influence each other.

The Gauss-Newton matrix is ``J' W J`` and the gradient ``J' W r`` with ``r = pred - obs``; ``W``
is 1 for Gaussian noise and ``1 / pred`` (the Fisher information) for Poisson noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jacfwd, vmap

from ..hooks import CancelHook, check_cancel
from ..model import MixtureBatch, convolve


def _predict_one(theta, b, shape_w, n_max, k):
    """Prediction ``(B,)`` of one measurement from ``theta = [log_A, n_raw, logit_p]``."""
    amp = torch.exp(theta[:k])
    n = n_max * torch.sigmoid(theta[k : 2 * k])
    p = torch.sigmoid(theta[2 * k :])
    conv = convolve(b.unsqueeze(1), n.unsqueeze(0), p.unsqueeze(0), shape_w.unsqueeze(1))
    return (amp * conv[:, 0, :]).sum(dim=-1)


def _loss(pred, obs, mask, kind):
    if kind == "gaussian":
        per_bin = 0.5 * (pred - obs) ** 2
    else:
        pred = pred.clamp(min=1e-10)
        per_bin = pred - obs * torch.log(pred)
    return per_bin.masked_fill(~mask, 0.0).sum(dim=1)


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
    kind = model.noise_kind
    with torch.no_grad():
        theta_all = torch.cat([model.log_A, model.n_raw, model.logit_p], dim=1).detach().clone()
        loss_all = None
    obs_all = obs.T
    n_meas = theta_all.shape[0]
    p_dim = theta_all.shape[1]
    eye = torch.eye(p_dim, dtype=theta_all.dtype, device=theta_all.device)

    def make_fns(b, shape_w, n_max, mask, obs_m):
        def pred_and_jac(th):
            def f(t, bb, sw, nm):
                pr = _predict_one(t, bb, sw, nm, k)
                return pr, pr

            jac, pred = vmap(jacfwd(f, has_aux=True), in_dims=(0, 0, 1, 0))(th, b, shape_w, n_max)
            return pred, jac

        def loss_of(th):
            pred = vmap(_predict_one, in_dims=(0, 0, 1, 0, None))(th, b, shape_w, n_max, k)
            return _loss(pred, obs_m, mask, kind)

        return pred_and_jac, loss_of

    active = torch.arange(n_meas, device=theta_all.device)
    b, mask, obs_m = model.b.T, model.mask.T, obs_all
    shape_w, n_max = model.shape_w, model.n_max
    pred_and_jac, loss_of = make_fns(b, shape_w, n_max, mask, obs_m)
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
            w = torch.ones_like(pred) if kind == "gaussian" else 1.0 / pred.clamp(min=1e-10)
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
                    b, mask, obs_m = b[keep], mask[keep], obs_m[keep]
                    shape_w, n_max = shape_w[:, keep], n_max[keep]
                    pred_and_jac, loss_of = make_fns(b, shape_w, n_max, mask, obs_m)
    with torch.no_grad():
        if len(active):
            theta_all[active] = theta
            loss_all[active] = loss
        model.log_A.data.copy_(theta_all[:, :k])
        model.n_raw.data.copy_(theta_all[:, k : 2 * k])
        model.logit_p.data.copy_(theta_all[:, 2 * k :])
    return loss_all.detach().cpu().numpy().astype(float)
