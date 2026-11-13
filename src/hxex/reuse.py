# SPDX-License-Identifier: GPL-3.0-or-later
"""The previous results' fits reused by a run that replaces them (docs/PLAN.md Next steps 26,
"Applying"): ``hxex run --overwrite`` reads and picks every spectrum again, as always, and then
takes the fit, the population test, and the bootstrap of every spectrum whose input is the same
as in the bundle it replaces, fitting only the others. Everything that uses all the spectra at
once (the peak measure, the envelopes, the per-bin noise, the m/z check, the controls, the
quality flags) is worked out again, so the result is the one a full run gives, up to the
rounding a fit takes from the other measurements of its batch (HISTORY 46, 219).

A spectrum's input is the same when the bundle holds a measurement with its ``spectrum_id``
and the same bins, intensities, envelope, bound on n, charge, control role, random key and
scan cut, whose noise references are the same spectra and unchanged too, and whose series is
unchanged as a whole (every spectrum of it the same, none added or gone: the noise estimated
from replicates is per series); the bundle must have
been written by the same software (hxex and the numerical packages) with the same fit,
population-test, and bootstrap settings and the same per-bin noise (a curve that moved by
less than 1 % of its standard error is replaced by the bundle's: :func:`kept_curve`), and be
complete (no pending bootstrap). A spectrum whose K a review table chooses is reused only when
the bundle fitted that K for it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reused:
    """What a run takes from the previous bundle: ``rows`` (positions in the run's
    measurements) and, in that order, the bundle's ``fit``, population ``test`` (None without
    the test) and bootstrap (``boot``, the selected K's, and ``by_k``, the fixed-K ones of the
    run's ``[run.bootstrap] ks``; None without a bootstrap) of those spectra; ``made_with``:
    how the software that made them differs from the run's (:func:`differences`; the fitting
    code is the same), recorded in the provenance."""

    rows: np.ndarray
    fit: object
    test: object | None = None
    boot: object | None = None
    by_k: dict | None = None
    made_with: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)


def _same_arrays(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _same_cut(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a.weights, b.weights) and np.array_equal(a.noise, b.noise)


def _same_time(a, b) -> bool:
    a, b = float(a), float(b)
    return a == b or (np.isnan(a) and np.isnan(b))


def whole_series(pos: np.ndarray, new, old) -> np.ndarray:
    """``pos`` (:func:`same_inputs`) with -1 for every measurement of a series (``data_id``)
    that is not unchanged as a whole: each of its spectra the same, and the bundle's series
    holding exactly those spectra. The noise estimated from replicates, the noise references
    and the chunks of a run are per series."""
    pos = pos.copy()
    new_ids, old_ids = np.asarray(new.data_id), np.asarray(old.data_id)
    old_count = dict(zip(*np.unique(old_ids, return_counts=True), strict=True))
    for d in np.unique(new_ids):
        rows = np.flatnonzero(new_ids == d)
        mine = pos[rows]
        if ((mine < 0).any() or len(set(old_ids[mine].tolist())) != 1
                or old_count[old_ids[mine[0]]] != len(rows)):
            pos[rows] = -1
    return pos


def same_inputs(new, old) -> np.ndarray:
    """For each measurement of ``new`` (:class:`hxex.fit.Measurements`), its position in
    ``old`` when ``old`` holds the same spectrum with the same input (see the module
    docstring), else -1."""
    where = {str(s): j for j, s in enumerate(old.spectrum_id)}
    pos = np.full(len(new), -1, dtype=np.int64)
    cuts_new = new.scan_cut or (None,) * len(new)
    cuts_old = old.scan_cut or (None,) * len(old)
    role_new = new.role if new.role is not None else np.full(len(new), "")
    role_old = old.role if old.role is not None else np.full(len(old), "")
    charge_new, charge_old = new.charges(), old.charges()
    key_new = new.random_id if new.random_id is not None else new.spectrum_id
    key_old = old.random_id if old.random_id is not None else old.spectrum_id
    for i, sid in enumerate(map(str, new.spectrum_id)):
        j = where.get(sid)
        if j is None:
            continue
        if (np.array_equal(new.bins[i], old.bins[j])
                and np.array_equal(new.observed[i], old.observed[j])
                and np.array_equal(new.shapes[i], old.shapes[j])
                and float(new.n_max[i]) == float(old.n_max[j])
                and _same_time(new.time[i], old.time[j])
                and str(role_new[i]) == str(role_old[j])
                and int(charge_new[i]) == int(charge_old[j])
                and str(key_new[i]) == str(key_old[j])
                and _same_cut(cuts_new[i], cuts_old[j])):
            pos[i] = j
    # the noise references: the same spectra, themselves unchanged
    for i in np.flatnonzero(pos >= 0):
        a, b = new.references[i], old.references[pos[i]]
        if (a is None) != (b is None):
            pos[i] = -1
        elif a is not None and (
                len(a) != len(b)
                or any(pos[x] != y for x, y in zip(a, b, strict=True))):
            pos[i] = -1
    return pos


def _software(versions: Mapping | None) -> dict:
    out = dict(versions or {})
    out.pop("python", None)
    return out


def differences(made: Mapping | None, now: Mapping) -> list[str]:
    """What differs between the software that made a bundle's fits (``made``) and the
    software running (``now``), as ``torch 2.14.1+cu126 -> 2.14.1+cpu`` (the engine's
    fingerprint and Python left out: the first decides, the second is said by the packages)."""
    made = dict(made or {})
    out = []
    for key in sorted(set(made) | set(now)):
        if key in ("engine", "python", "mixed"):
            continue
        a, b = made.get(key, "unknown"), now.get(key, "unknown")
        if a != b:
            out.append(f"{key} {a} -> {b}")
    return out


def same_engine(made: Mapping | None, now: Mapping) -> tuple[bool, str | None]:
    """Whether fits made with ``made`` (a bundle's ``fitted_with``) can be kept by software
    ``now`` (:func:`hxex.io.bundle._versions`), and why not. The engine's fingerprint decides
    (:mod:`hxex._engine`): package versions and PyTorch's build (a graphics card's or the CPU's)
    do not, as fits agree across them to the rounding of the arithmetic (owner, 2026-11-11:
    fitted on a GPU, touched up on a laptop). A bundle made before the fingerprint was recorded
    needs the same software throughout, as before."""
    if made is None or made.get("mixed"):
        return False, ("it is not known which version of hxex made them (a resume by another "
                       "version)")
    if made.get("engine") is not None:
        if made["engine"] == now.get("engine"):
            return True, None
        return False, ("hxex's fitting code differs from the one that made them ("
                       + "; ".join(differences(made, now) or ["engine " + str(made["engine"])
                                                               + " -> " + str(now.get("engine"))])
                       + ")")
    if _software(made) == _software({k: v for k, v in now.items() if k != "engine"}):
        return True, None
    return False, ("another version of hxex or of its numerical packages made them, and they "
                   "were made before hxex recorded its fitting code ("
                   + "; ".join(differences(made, now)) + ")")


def fitted_with(manifest: Mapping) -> dict | None:
    """The software that made a bundle's fits, population test and bootstrap: its
    ``fitted_with``; for a bundle written before that was recorded, its ``versions`` when no
    resume rewrote it; None when not known."""
    if "fitted_with" in manifest:
        return manifest["fitted_with"]
    if (manifest.get("provenance") or {}).get("resumed"):
        return None
    return manifest.get("versions")


def resumed_fitted_with(manifest: Mapping) -> dict:
    """What a resume of the bundle with ``manifest`` records as ``fitted_with``: the bundle's,
    when the software resuming has the same fitting code (:func:`same_engine`: what it adds is
    made alike), else ``{"mixed": True}`` (fits of one version with additions of another, never
    reused)."""
    from .io.bundle import _versions

    before = fitted_with(manifest)
    if before is not None and same_engine(before, _versions())[0]:
        return dict(before)
    return {"mixed": True}


def _settings(config):
    """A fit, population-test or bootstrap configuration without where it ran (``backend``:
    the device and precision; a GPU's fits are kept on a CPU and the reverse, owner
    2026-11-11) and without ``keep_draws`` (what a bootstrap stores, not its values)."""
    import dataclasses

    if config is None or not dataclasses.is_dataclass(config):
        return config
    names = {f.name for f in dataclasses.fields(config)}
    if "backend" in names:
        config = dataclasses.replace(config, backend=None)
    if "keep_draws" in names:  # only what is stored (owner, 2026-11-13)
        config = dataclasses.replace(config, keep_draws=True)
    return config


CURVE_SHARE = 0.01  # a noise curve moved by less than this share of its standard error is kept


def _close(new: float, old: float, se: float) -> bool:
    if np.isnan(new) and np.isnan(old):
        return True
    return bool(np.isfinite(se) and abs(new - old) <= CURVE_SHARE * se)


def _same_number(a: float, b: float) -> bool:
    return a == b or (np.isnan(a) and np.isnan(b))


def kept_curve(path: str | Path | None, curve):
    """The per-bin noise a run that replaces the bundle at ``path`` fits with: ``curve`` (what
    the run measured), or the bundle's own when the two differ only in ``b1`` (and per sample),
    by less than :data:`CURVE_SHARE` of the bundle's standard error of it. Editing an
    undeuterated control's window moves ``b1`` by a tiny amount (on the private targeted
    dataset, by 0.0000005 of its standard error), which would otherwise change every
    spectrum's noise and leave no fit to keep; the earlier curve is as good a measurement.
    The same controls must remain (``n_controls``, ``n_series``), with the same onset and
    model error. Logged when the earlier curve is kept."""
    from .io.bundle import BundleError, bundle_manifest
    from .noise import ControlsNoise

    if path is None or curve is None or not Path(path).exists():
        return curve
    try:
        record = (bundle_manifest(Path(path)) or {}).get("noise_curve")
    except BundleError:
        return curve
    if not record:
        return curve
    old = ControlsNoise.from_dict(record)
    if old == curve:
        return curve
    samples_new = {s.sample: s for s in curve.by_sample}
    samples_old = {s.sample: s for s in old.by_sample}
    if (old.n_controls != curve.n_controls or old.n_series != curve.n_series
            or not _same_number(old.onset, curve.onset)
            or old.model_error != curve.model_error
            or old.model_error_at != curve.model_error_at
            or not _close(curve.b1, old.b1, old.b1_se)
            or set(samples_new) != set(samples_old)
            or any(samples_new[n].n_series != o.n_series
                   or not _close(samples_new[n].b1, o.b1, o.b1_se)
                   for n, o in samples_old.items())):
        return curve
    log.info("per-bin noise: b1 %.6g measured now, %.6g in the results being replaced (%.2g of "
             "its standard error): the earlier curve is kept, so the unchanged series keep "
             "their fits", curve.b1, old.b1, abs(curve.b1 - old.b1) / old.b1_se)
    return old


def reused_fits(path: str | Path | None, data, config, noise_curve,
                chosen: Mapping[str, int] | None = None) -> Reused | None:
    """The fits, population test and bootstrap of the bundle at ``path`` (the one the run
    replaces) that the run can take for its measurements ``data``, fitted with ``config``
    (:class:`hxex.pipeline.RunConfig`) and the per-bin noise ``noise_curve``; ``chosen``: the
    K a review table chooses per spectrum. None, with the reason logged, when nothing can be
    taken."""
    from .io.bundle import BundleError, _versions, open_bundle
    from .pipeline import RunConfig, _bootstrap_ks, _d2o_fraction

    if path is None or not Path(path).exists():
        return None
    try:
        with open_bundle(path) as b:
            if not (b.has("fit") and b.has("measurements")):
                return None
            stored = RunConfig.from_dict(b.manifest["configs"]["run"]) \
                if (b.manifest.get("configs") or {}).get("run") else None
            why = []
            made = fitted_with(b.manifest)
            same, reason = same_engine(made, _versions())
            if not same:
                why.append(reason)
            if stored is None:
                why.append("they do not record their run settings")
            else:
                for name, label in (("fit", "[run.fit]"),
                                    ("population_test", "[run.population_test]"),
                                    ("bootstrap", "[run.bootstrap]")):
                    if _settings(getattr(stored, name)) != _settings(getattr(config, name)):
                        why.append(f"{label} changed")
            if b.pending:
                why.append(f"they wait for their {', '.join(b.pending)}")
            fit = b.fit()
            if not why and _settings(fit.config) != _settings(config.fit):
                why.append("[run.fit] changed")
            if not why and fit.noise_curve != noise_curve:
                why.append("the per-bin noise measured from the undeuterated controls changed")
            test = b.population_test() if config.population_test is not None else None
            if not why and test is not None and test.d2o_fraction != _d2o_fraction(config):
                why.append("[run.controls] d2o_fraction changed (the population test uses it)")
            if why:
                log.info("fitting every spectrum: the previous results in %s cannot be reused "
                         "(%s)", path, "; ".join(why))
                return None
            pos = same_inputs(data, b.measurements())
            if chosen:
                for i, sid in enumerate(map(str, data.spectrum_id)):
                    k = int(chosen.get(sid, 0))
                    if pos[i] >= 0 and k and (k not in fit.k_values
                                              or not fit.fitted(k)[pos[i]]):
                        pos[i] = -1  # a K the review chooses, not fitted for it then
            pos = whole_series(pos, data, b.measurements())
            rows = np.flatnonzero(pos >= 0)
            if not len(rows):
                log.info("fitting every spectrum: none has the same input as in %s", path)
                return None
            old = pos[rows]
            boot = by_k = None
            if config.bootstrap is not None:
                boot = b.bootstrap().select(old)
                ks = _bootstrap_ks(config, fit)
                if ks:
                    by_k = {k: b.bootstrap(k).select(old) for k in ks}
            other = differences(made, _versions())
            if other:
                log.info("the previous results' fits are kept although made with other "
                         "software (%s): the fitting code is the same, so they agree with a new "
                         "fit to the rounding of the arithmetic", "; ".join(other))
            return Reused(rows=rows, fit=fit.select(old), made_with=tuple(other),
                          test=None if test is None else test.select(old),
                          boot=boot, by_k=by_k)
    except (BundleError, KeyError, ValueError) as exc:
        log.info("fitting every spectrum: the previous results in %s cannot be read for reuse "
                 "(%s)", path, exc)
        return None


def describe(reused: Reused | None, total: int) -> str:
    """``N of M spectra keep their fits from the previous results; K fitted`` for the log."""
    n = 0 if reused is None else len(reused)
    return (f"{n} of {total} spectra keep their fit"
            + (", population test" if reused is not None and reused.test is not None else "")
            + (" and bootstrap" if reused is not None and reused.boot is not None else "")
            + f" from the previous results (the same input); {total - n} fitted")
