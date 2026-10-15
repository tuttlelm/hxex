# SPDX-License-Identifier: GPL-3.0-or-later
"""The binomial mixture: real-valued binomial components convolved with a known shape.

For one measurement, with bins ``b`` and a normalised shape ``D`` (the natural-abundance
envelope), the predicted intensity is ``I(b) = sum_k A_k * (B_k * D)[b]`` where ``B_k`` is a
binomial with real-valued ``n_k`` and ``p_k`` (defined through the gamma function).

Parameterisation (always inside their bounds): ``A_k = exp(log_A_k)``,
``n_k = n_max * sigmoid(n_raw_k)`` and ``p_k = sigmoid(logit_p_k)``. ``n`` has no lower bound
other than 0, so small ``n`` with large ``p`` and large ``n`` with small ``p`` are both reachable.

Many measurements are fitted together: ragged bins are padded to ``(B_max, M)`` with a mask, and
padded positions are excluded from the loss. Losses (per measurement, on spectra already divided
by the noise scale, see :mod:`hxex.noise`): ``gaussian`` is ``0.5 * (pred - obs)^2`` (unit-sd
noise, i.e. least squares) and ``poisson`` is ``pred - obs * log(pred)`` (log-factorial dropped).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backend import Backend

NOISE_KINDS = ("gaussian", "poisson")


def _log_binom(b, n, p, log_p=None, log_q=None):
    """Log of the generalised binomial, before normalisation, and the mask of the bins that
    have probability.

    ``b``, ``n`` and ``p`` broadcast together. A bin has probability for ``0 <= b < n + 1``. For
    ``b <= n`` this is the ordinary binomial written with gamma functions. For ``n < b < n + 1``
    the gamma function in ``Gamma(n - b + 1)`` is finite and positive, and ``1 / Gamma`` falls to
    0 as ``n`` decreases to ``b - 1``, so the pmf is continuous in ``n`` (a mask at ``b <= n``,
    as in the prototype and pyHXExpress, makes the loss jump each time ``n`` crosses an integer,
    loses up to 15 % of the probability between integers at high ``p`` and biases the centre of
    the component low). For integer ``n`` the result is the ordinary binomial. For ``b >= n + 1``
    the gamma function of a negative argument would give negative values, so the bin is masked.

    At non-integer ``n`` the bins do not sum to 1 (the bin above ``n`` carries
    ``(1 - p)^(n - b)`` with a negative exponent, which grows without bound as ``p`` nears 1:
    the mass is 1.02 at ``n = 13.5, p = 0.9`` but 1.7 at ``p = 0.99`` and 5 at ``n = 0.5,
    p = 0.99``), so every user divides by :func:`log_binom_norm`. ``log_p`` and ``log_q`` are
    ``log p`` and ``log (1 - p)``; given (from the logit, see :func:`log_terms`) they keep
    ``log (1 - p)`` exact where ``sigmoid`` saturates in float32.
    """
    b_safe = b.clamp(min=0.0)
    rest = (n - b + 1.0).clamp(min=1e-12)  # argument of the third gamma, kept positive
    valid = (b >= 0) & (n - b + 1.0 > 0)
    if log_p is None:
        log_p = torch.log(p.clamp(min=1e-10))
    if log_q is None:
        log_q = torch.log((1.0 - p).clamp(min=1e-10))
    log_pmf = (
        torch.lgamma(n + 1)
        - torch.lgamma(b_safe + 1)
        - torch.lgamma(rest)
        + b_safe * log_p
        + (n - b_safe) * log_q
    )
    return log_pmf, valid


def log_terms(p: torch.Tensor, logit_p: torch.Tensor | None):
    """``(log p, log (1 - p))`` from the logit when it is given (``logsigmoid``, exact in
    float32 where ``sigmoid(logit_p)`` rounds to 1), else from ``p`` (clamped at 1e-10)."""
    if logit_p is None:
        return None, None
    return F.logsigmoid(logit_p), F.logsigmoid(-logit_p)


def log_binom_norm(n: torch.Tensor, p: torch.Tensor, log_p=None, log_q=None) -> torch.Tensor:
    """``log`` of the total mass of the unnormalised pmf of :func:`_log_binom` over its valid
    bins ``0 .. ceil(n)``, with the shape of ``n`` (``n`` and ``p`` broadcast). Subtracting it
    makes every component sum to exactly 1 for any real ``n``, so the amplitudes are areas and
    the populations ``A_k / sum A`` are shares of the molecules. Differentiable in ``n`` and
    ``p``."""
    n_b, p_b = torch.broadcast_tensors(n, p)
    flat_n, flat_p = n_b.reshape(-1), p_b.reshape(-1)
    if flat_n.numel() == 0:
        return torch.zeros_like(n_b)
    top = int(torch.nan_to_num(flat_n.detach(), nan=0.0, posinf=0.0).max().ceil().item()) + 1
    u = torch.arange(top + 1, dtype=n.dtype, device=n.device).unsqueeze(1)  # (U, MK)
    lp = lq = None
    if log_p is not None:
        lp, lq = torch.broadcast_tensors(log_p, log_q)
        lp, lq = lp.reshape(1, -1), lq.reshape(1, -1)
    log_pmf, valid = _log_binom(u, flat_n.unsqueeze(0), flat_p.unsqueeze(0), lp, lq)
    log_pmf = log_pmf.masked_fill(~valid, float("-inf"))  # bin 0 is always valid for n >= 0
    return torch.logsumexp(log_pmf, dim=0).reshape(n_b.shape)


def log_binom_pmf(
    b: torch.Tensor, n: torch.Tensor, p: torch.Tensor, logit_p: torch.Tensor | None = None
) -> torch.Tensor:
    """Log pmf of the real-valued binomial, normalised to sum to 1 over its bins. ``b`` is
    ``(B, M)``; ``n``, ``p`` and ``logit_p`` (optional, see :func:`log_terms`) are ``(M, K)``.
    Returns ``(B, M, K)``; bins without probability are set to -1e10 (see
    :func:`_log_binom`)."""
    log_p, log_q = log_terms(p, logit_p)
    lp = None if log_p is None else log_p.unsqueeze(0)
    lq = None if log_q is None else log_q.unsqueeze(0)
    log_pmf, valid = _log_binom(b.unsqueeze(2), n.unsqueeze(0), p.unsqueeze(0), lp, lq)
    log_pmf = log_pmf - log_binom_norm(n, p, log_p, log_q).unsqueeze(0)
    return torch.where(valid, log_pmf, torch.full_like(log_pmf, -1e10))


@dataclass(frozen=True)
class IntegerGrid:
    """Bins that are whole numbers (0 or more), prepared for :func:`convolve`: the integers
    ``0 .. U-1`` the pmf is needed at (``u``, ``(U, 1, 1)``), and for each shape point ``j`` and
    bin ``b`` the integer ``b - j`` (``index``, ``(L*B, M, 1)``, clamped at 0) and whether it
    is at least 0 (``inside``, ``(L, B, M, 1)``). ``consecutive`` is True when every
    measurement's bins are ``0, 1, 2, ...`` (then the shifts are slices, see :func:`convolve`),
    and ``real`` (``(B, M, 1)``) marks the bins that are not padding."""

    u: torch.Tensor
    index: torch.Tensor
    inside: torch.Tensor
    consecutive: bool = False
    real: torch.Tensor | None = None


def integer_grid(b: torch.Tensor, length: int) -> IntegerGrid | None:
    """The :class:`IntegerGrid` of the bins ``b`` (``(B, M)``) for shapes of ``length`` points,
    or None if a bin is not a whole number of 0 or more (then :func:`convolve` evaluates the
    pmf at every shifted bin)."""
    if b.numel() == 0 or not bool(torch.equal(b, b.round())) or bool((b < 0).any()):
        return None
    n_bins, m = b.shape
    j = torch.arange(length, device=b.device).view(length, 1, 1)
    shifted = b.long().unsqueeze(0) - j  # (L, B, M)
    rows = torch.arange(n_bins, dtype=b.dtype, device=b.device).unsqueeze(1)
    # consecutive: each column is 0, 1, ..., n-1 and then zeros (the padding)
    real = (b == rows).long().cumprod(dim=0).bool()
    consecutive = bool(((b == rows) | (~real & (b == 0))).all())
    if consecutive:
        u = rows.view(-1, 1, 1)
    else:
        u = torch.arange(int(b.max()) + 1, dtype=b.dtype, device=b.device).view(-1, 1, 1)
    return IntegerGrid(u=u, index=shifted.clamp(min=0).reshape(length * n_bins, m, 1),
                       inside=(shifted >= 0).unsqueeze(3), consecutive=consecutive,
                       real=real.unsqueeze(2))


def convolve(
    b: torch.Tensor,
    n: torch.Tensor,
    p: torch.Tensor,
    shape_w: torch.Tensor,
    grid: IntegerGrid | None = None,
    logit_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each binomial component convolved with its shape: ``sum_j D[j] * B(b - j; n, p)``.

    ``b`` is ``(B, M)``, ``n`` and ``p`` are ``(M, K)``, ``shape_w`` is ``(L, M, K)`` (see
    :func:`shape_weights`). Returns ``(B, M, K)``. The shape index is vectorised. Each
    component's pmf is normalised to sum to 1 (:func:`log_binom_norm`); ``logit_p`` (optional)
    gives exact log terms in float32 (:func:`log_terms`).

    With ``grid`` (:func:`integer_grid` of ``b``; the bins are whole numbers), the pmf is
    evaluated once at each integer and gathered for every shifted bin, instead of once per
    shift and bin: the same values, about ``L`` times fewer special-function evaluations. When
    every measurement's bins are ``0, 1, 2, ...`` the shifted pmfs are windows of one padded
    array (no gather); padded positions (bin 0, masked out of every loss) get the value of bin
    0 as before.
    """
    length, m, k = shape_w.shape
    log_p, log_q = log_terms(p, logit_p)
    norm = log_binom_norm(n, p, log_p, log_q)  # (M, K)
    if log_p is not None:
        log_p, log_q = log_p.view(1, m, k), log_q.view(1, m, k)
    if grid is not None and grid.consecutive:
        log_pmf, valid = _log_binom(grid.u, n.view(1, m, k), p.view(1, m, k), log_p, log_q)
        pmf = torch.where(valid, (log_pmf - norm).exp(), torch.zeros_like(log_pmf))  # (B, M, K)
        padded = torch.cat([pmf.new_zeros(length - 1, m, k), pmf], dim=0)
        windows = padded.unfold(0, length, 1)  # (B, M, K, L): pmf at b - (L-1) + i
        out = (windows * shape_w.flip(0).permute(1, 2, 0)).sum(dim=-1)
        return torch.where(grid.real, out, out[:1])  # padding is bin 0, as row 0
    if grid is not None:
        log_pmf, valid = _log_binom(grid.u, n.view(1, m, k), p.view(1, m, k), log_p, log_q)
        pmf = torch.where(valid, (log_pmf - norm).exp(), torch.zeros_like(log_pmf))  # (U, M, K)
        shifted = pmf.gather(0, grid.index.expand(-1, -1, k)).view(length, b.shape[0], m, k)
        shifted = torch.where(grid.inside, shifted, torch.zeros_like(shifted))
        return (shape_w.unsqueeze(1) * shifted).sum(dim=0)
    j = torch.arange(length, dtype=b.dtype, device=b.device).view(length, 1, 1, 1)
    bj = b.unsqueeze(0).unsqueeze(3) - j  # (L, B, M, 1)
    if log_p is not None:
        log_p, log_q = log_p.unsqueeze(0), log_q.unsqueeze(0)
    log_pmf, valid = _log_binom(bj, n.view(1, 1, m, k), p.view(1, 1, m, k), log_p, log_q)
    pmf = torch.where(valid, (log_pmf - norm).exp(), torch.zeros_like(log_pmf))
    return (shape_w.unsqueeze(1) * pmf).sum(dim=0)


def normalise_shape(shape: np.ndarray | None) -> np.ndarray:
    """A shape scaled to sum to 1 (``None`` is the identity ``[1.0]``)."""
    if shape is None:
        return np.array([1.0])
    s = np.asarray(shape, dtype=np.float64)
    total = s.sum()
    return s / total if total > 0 else s


def shape_weights(shapes: Sequence[np.ndarray | None], n_components: int, backend: Backend):
    """Stack one normalised shape per measurement into an ``(L, M, K)`` tensor (zero-padded);
    every component of a measurement uses that measurement's shape."""
    normed = [normalise_shape(s) for s in shapes]
    length = max(len(s) for s in normed)
    w = np.zeros((length, len(normed), n_components))
    for m, s in enumerate(normed):
        w[: len(s), m, :] = s[:, None]
    return backend.tensor(w)


@dataclass
class PaddedBatch:
    """M measurements with different numbers of bins, padded to ``(B_max, M)`` tensors.

    Padded positions have ``b = 0`` (safe for the gamma function), ``obs = 0`` and
    ``mask = False``. ``n_max`` is the upper bound of ``n`` for each measurement.
    """

    b: torch.Tensor
    obs: torch.Tensor
    mask: torch.Tensor
    n_max: torch.Tensor

    @classmethod
    def from_arrays(
        cls,
        bins: Sequence[np.ndarray],
        observed: Sequence[np.ndarray],
        n_max: Sequence[float],
        backend: Backend,
    ) -> PaddedBatch:
        m = len(bins)
        if not (len(observed) == len(n_max) == m):
            raise ValueError("bins, observed and n_max must have one entry per measurement")
        b_max = max(len(bc) for bc in bins)
        b = np.zeros((b_max, m))
        obs = np.zeros((b_max, m))
        mask = np.zeros((b_max, m), dtype=bool)
        for i, (bc, ob) in enumerate(zip(bins, observed, strict=True)):
            if len(bc) != len(ob):
                raise ValueError(f"measurement {i}: {len(bc)} bins but {len(ob)} intensities")
            b[: len(bc), i] = bc
            obs[: len(ob), i] = ob
            mask[: len(bc), i] = True
        return cls(
            b=backend.tensor(b),
            obs=backend.tensor(obs),
            mask=torch.tensor(mask, device=backend.torch_device()),
            n_max=backend.tensor(np.asarray(n_max, dtype=np.float64)),
        )

    @property
    def n_measurements(self) -> int:
        return self.b.shape[1]


def logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """``log(x / (1 - x))`` with ``x`` clipped to ``[eps, 1 - eps]``."""
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def noise_loss(pred: torch.Tensor, obs: torch.Tensor, is_gaussian: torch.Tensor) -> torch.Tensor:
    """Loss of every bin, ``(B, M)``: ``0.5 (pred - obs)^2`` (Gaussian, unit-sd noise) where
    ``is_gaussian`` is True (broadcast against ``(B, M)``, one flag per measurement) and
    ``pred - obs log pred`` (Poisson, log-factorial dropped) elsewhere."""
    if bool(is_gaussian.all()):
        return 0.5 * (pred - obs) ** 2
    pc = pred.clamp(min=1e-10)
    poisson = pc - obs * torch.log(pc)
    if not bool(is_gaussian.any()):
        return poisson
    return torch.where(is_gaussian, 0.5 * (pred - obs) ** 2, poisson)


class MixtureBatch(nn.Module):
    """K binomial components for each of M measurements (same K, variable bins).

    The parameters ``log_A``, ``n_raw`` and ``logit_p`` are ``(M, K)`` tensors. Amplitudes are in
    the units of the observed data the model is evaluated against (the fit works in scaled
    units; see :mod:`hxex.fit`).
    """

    def __init__(
        self,
        batch: PaddedBatch,
        shape_w: torch.Tensor,
        log_A: torch.Tensor,
        n_raw: torch.Tensor,
        logit_p: torch.Tensor,
        noise_kind: str | Sequence[str] = "gaussian",
    ):
        super().__init__()
        self.register_buffer("is_gaussian", torch.zeros(log_A.shape[0], dtype=torch.bool))
        self.register_buffer("n_max", batch.n_max)
        self.register_buffer("mask", batch.mask)
        self.register_buffer("b", batch.b)
        self.register_buffer("shape_w", shape_w)
        self.log_A = nn.Parameter(log_A.clone())
        self.n_raw = nn.Parameter(n_raw.clone())
        self.logit_p = nn.Parameter(logit_p.clone())
        self.set_noise_kind(noise_kind)
        self.grid = integer_grid(batch.b, shape_w.shape[0])  # None for non-integer bins

    def set_noise_kind(self, noise_kind: str | Sequence[str]) -> None:
        """Change the noise model: one name for all measurements, or one per measurement."""
        m = self.log_A.shape[0]
        kinds = [noise_kind] * m if isinstance(noise_kind, str) else list(noise_kind)
        if len(kinds) != m or any(k not in NOISE_KINDS for k in kinds):
            raise ValueError(f"noise_kind must be one of {NOISE_KINDS}, one per measurement")
        self.noise_kinds = tuple(kinds)
        self.is_gaussian = torch.tensor([k == "gaussian" for k in kinds], device=self.log_A.device)

    @property
    def n_measurements(self) -> int:
        return self.log_A.shape[0]

    @property
    def n_components(self) -> int:
        return self.log_A.shape[1]

    @property
    def amplitudes(self) -> torch.Tensor:
        return torch.exp(self.log_A)

    @property
    def ns(self) -> torch.Tensor:
        return self.n_max.unsqueeze(1) * torch.sigmoid(self.n_raw)

    @property
    def ps(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_p)

    def predict(self) -> torch.Tensor:
        """Predicted intensity at every (padded) bin, ``(B_max, M)``."""
        conv = convolve(self.b, self.ns, self.ps, self.shape_w, self.grid, self.logit_p)
        return (self.amplitudes.unsqueeze(0) * conv).sum(dim=2)

    def components(self) -> torch.Tensor:
        """Predicted intensity of every component, ``(B_max, M, K)``."""
        return self.amplitudes.unsqueeze(0) * convolve(self.b, self.ns, self.ps, self.shape_w,
                                                       self.grid, self.logit_p)

    def nll(self, obs: torch.Tensor) -> torch.Tensor:
        """Loss per measurement, ``(M,)``, against ``obs`` of shape ``(B_max, M)``."""
        loss = noise_loss(self.predict(), obs, self.is_gaussian.unsqueeze(0))
        return loss.masked_fill(~self.mask, 0.0).sum(dim=0)

    @torch.no_grad()
    def parameters_numpy(self) -> dict[str, np.ndarray]:
        """``A``, ``n`` and ``p`` as ``(M, K)`` float64 arrays."""
        return {
            "A": self.amplitudes.cpu().numpy().astype(np.float64),
            "n": self.ns.cpu().numpy().astype(np.float64),
            "p": self.ps.cpu().numpy().astype(np.float64),
        }


def n_params(n_components: int) -> int:
    """Free parameters of a K-component model: A, n and p for each component."""
    return 3 * n_components


def bic(n_components: int, sample_size, nll):
    """BIC = ``3K ln(N) + 2 NLL`` (``N`` at least 2); works on scalars or arrays."""
    return n_params(n_components) * np.log(np.maximum(sample_size, 2.0)) + 2.0 * np.asarray(nll)


def aic(n_components: int, nll):
    """AIC = ``2 * 3K + 2 NLL``."""
    return 2.0 * n_params(n_components) + 2.0 * np.asarray(nll)
