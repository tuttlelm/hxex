# SPDX-License-Identifier: GPL-3.0-or-later
"""The previous results' fits reused by a run that updates them (docs/PLAN.md Next steps 26,
"Applying", and 40): ``hxex run`` takes the stored spectra or reads them again, and then takes
the fit, the population test, and the bootstrap of every spectrum whose input is the same as in
the bundle it updates, fitting only the others. Everything that uses all the spectra at once
(the peak measure, the envelopes, the per-bin noise, the m/z check, the controls, the quality
flags) is worked out again, so the result is the one a full run gives, up to the rounding a fit
takes from the other measurements of its batch (HISTORY 46, 219).

A spectrum's input is the same when the bundle holds a measurement with its ``spectrum_id``
and the same bins, intensities, envelope, bound on n, charge, control role, random key and
scan cut, whose noise references are the same spectra and unchanged too, and whose series is
unchanged as a whole (every spectrum of it the same, none added or gone: the noise estimated
from replicates is per series); the bundle must have been made by the same fitting code with
the same ``[run.fit]`` settings but ``k_max`` and the same per-bin noise (a curve whose b1
moved by less than :data:`CURVE_SHARE` is replaced by the bundle's: :func:`kept_curve`), and no
stage but the bootstrap may have stopped partway. The kept fits are then brought to the run's
other settings (:func:`hxex.pipeline.run`): the K missing fitted, the population test made again
when its settings, the D2O fraction or the K fitted changed, and the bootstrap made again
(stopped partway, or other settings) or completed (a fixed K newly listed). A spectrum whose K a
review table chooses is reused only when the bundle fitted that K for it.
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
    k_max: int | None = None
    retest: bool = False
    untest: bool = False
    rebootstrap: bool = False
    new_ks: tuple[int, ...] = ()

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


# a per-bin noise curve whose b1 moved by less than this share of itself is kept, and with it the
# fits of the unchanged spectra (HISTORY 433: refits at b1 x 0.98 ... 1.2 on two datasets;
# the fits move about in proportion to the move, with no level below which none does)
CURVE_SHARE = 0.01


def kept_curve(path: str | Path | None, curve):
    """The per-bin noise a run that updates the bundle at ``path`` fits with: ``curve`` (what
    the run measured), or the bundle's own when the fits would hardly differ: the same
    model-error term, and the bundle's ``b1`` (the one number of the curve the fits use; the
    onset and the per-sample values are only reported) within :data:`CURVE_SHARE` of the new
    one. Editing a control's window, or another set of undeuterated controls (a changed
    ``undeut_at``), moves ``b1`` a little, which would otherwise change every spectrum's noise
    and leave no fit to keep. Logged either way, with the move."""
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
    move = abs(curve.b1 - old.b1) / old.b1 if old.b1 else float("inf")
    if (old.model_error != curve.model_error or old.model_error_at != curve.model_error_at
            or not move <= CURVE_SHARE):
        log.info("per-bin noise: b1 %.6g measured now, %.6g in the results (moved %.2g %%, more "
                 "than %g %%): every spectrum is fitted with the new curve", curve.b1, old.b1,
                 100 * move, 100 * CURVE_SHARE)
        return curve
    log.info("per-bin noise: b1 %.6g measured now, %.6g in the results (moved %.2g %%, within "
             "%g %%): the earlier curve is kept, so the unchanged spectra keep their fits",
             curve.b1, old.b1, 100 * move, 100 * CURVE_SHARE)
    import dataclasses

    return dataclasses.replace(old, left_out=curve.left_out)  # what this run left out


def reused_fits(path: str | Path | None, data, config, noise_curve,
                chosen: Mapping[str, int] | None = None) -> Reused | None:
    """The fits, population test and bootstrap of the bundle at ``path`` (the one the run
    updates) that the run can take for its measurements ``data``, fitted with ``config``
    (:class:`hxex.pipeline.RunConfig`) and the per-bin noise ``noise_curve``; ``chosen``: the
    K a review table chooses per spectrum. None, with the reason logged, when nothing can be
    taken: other fitting code, other ``[run.fit]`` settings than ``k_max``, another per-bin
    noise, or a stage other than the bootstrap stopped partway.

    What the kept spectra still need is said on the result (docs/PLAN.md Next steps 40, an
    update): their fits brought to another ``k_max`` (``k_max``), the population test made
    again (``retest``: added, other settings or ``d2o_fraction``, or other K fitted), taken
    away (``untest``), the bootstrap made again (``rebootstrap``: stopped partway, added, or
    other values) or the fixed-K bootstraps ``[run.bootstrap] ks`` newly lists added
    (``new_ks``); the run does it (:func:`hxex.pipeline.run`)."""
    from .bootstrap import same_values
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
            elif not same_fit_settings(stored.fit, config.fit):
                why.append("[run.fit] changed")
            stopped = [s for s in b.pending if s != "bootstrap"]
            if stopped:
                why.append(f"they wait for their {', '.join(stopped)}")
            fit = b.fit()
            if not why and not same_fit_settings(fit.config, config.fit):
                why.append("[run.fit] changed")
            if not why and fit.noise_curve != noise_curve:
                why.append("the per-bin noise measured from the undeuterated controls changed")
            if why:
                log.info("fitting every spectrum: the previous results in %s cannot be reused "
                         "(%s)", path, "; ".join(why))
                return None
            k_max = config.fit.k_max if fit.config.k_max != config.fit.k_max else None
            test = b.population_test()
            testing = config.population_test is not None
            retest = testing and (
                test is None or k_max is not None
                or _settings(test.config) != _settings(config.population_test)
                or test.d2o_fraction != _d2o_fraction(config))
            untest = not testing and test is not None
            boot_settings = None if stored is None else stored.bootstrap
            rebootstrap, new_ks = False, ()
            if config.bootstrap is not None:
                rebootstrap = ("bootstrap" in b.pending or boot_settings is None
                               or b.bootstrap() is None
                               or not same_values(_without_ks(boot_settings),
                                                  _without_ks(config.bootstrap)))
            pos = same_inputs(data, b.measurements())
            fitted_k = config.fit.k_values if k_max is not None else fit.k_values
            if chosen:
                for i, sid in enumerate(map(str, data.spectrum_id)):
                    k = int(chosen.get(sid, 0))
                    if pos[i] >= 0 and k and (k not in fitted_k or (
                            k in fit.k_values and not fit.fitted(k)[pos[i]])):
                        pos[i] = -1  # a K the review chooses, not fitted for it then
            pos = whole_series(pos, data, b.measurements())
            rows = np.flatnonzero(pos >= 0)
            if not len(rows):
                log.info("fitting every spectrum: none has the same input as in %s", path)
                return None
            old = pos[rows]
            boot = by_k = None
            if config.bootstrap is not None and not rebootstrap:
                boot = b.bootstrap().select(old)
                held = set(b.bootstrap_ks())
                ks = [k for k in _bootstrap_ks(config, fit) if k in fitted_k]
                by_k = {k: b.bootstrap(k).select(old) for k in ks if k in held}
                new_ks = tuple(ks)  # those not held whole are completed by the run
            other = differences(made, _versions())
            if other:
                log.info("the previous results' fits are kept although made with other "
                         "software (%s): the fitting code is the same, so they agree with a new "
                         "fit to the rounding of the arithmetic", "; ".join(other))
            kept_fit = fit.select(old)
            if data.role is not None:  # the run's roles (the fits do not depend on them)
                import dataclasses

                kept_fit = dataclasses.replace(kept_fit, role=np.asarray(data.role)[rows])
            return Reused(rows=rows, fit=kept_fit, made_with=tuple(other),
                          test=None if test is None or untest else test.select(old),
                          boot=boot, by_k=by_k, k_max=k_max, retest=retest, untest=untest,
                          rebootstrap=rebootstrap, new_ks=new_ks)
    except (BundleError, KeyError, ValueError) as exc:
        log.info("fitting every spectrum: the previous results in %s cannot be read for reuse "
                 "(%s)", path, exc)
        return None


def same_fit_settings(a, b) -> bool:
    """Whether fits made with ``[run.fit]`` settings ``a`` can be kept for ``b``: the same
    settings but for ``k_max`` (the K missing are fitted, those above dropped: each K is
    fitted the same whatever ``k_max`` is) and where they run."""
    if a is None or b is None:
        return a is b

    def plain(c) -> dict:
        return {k: v for k, v in c.to_dict().items() if k not in ("k_max", "backend")}

    return plain(a) == plain(b)


def _without_ks(config):
    """A bootstrap configuration without what only adds to it (``ks``, ``set_k``)."""
    import dataclasses

    return dataclasses.replace(config, ks="selected", set_k=False)


def describe(reused: Reused | None, total: int) -> str:
    """``N of M spectra keep their fits from the previous results; K fitted`` for the log."""
    n = 0 if reused is None else len(reused)
    return (f"{n} of {total} spectra keep their fit"
            + (", population test" if reused is not None and reused.test is not None else "")
            + (" and bootstrap" if reused is not None and reused.boot is not None else "")
            + f" from the previous results (the same input); {total - n} fitted")
