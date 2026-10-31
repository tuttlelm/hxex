# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot of one spectrum, as independent layers: raw data, picked peaks and the fit.

Each layer needs its own source and any subset can be drawn: the raw spectra (``spectra``, or
the ones in the results), the picked table (``binned``, or the one in the results) and the fit
(``results``). Raw spectra and the predicted m/z of the picked bins are on the m/z axis; a fit
without the picked table can only be drawn against the deuterium bin.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..controls import correct_means
from ..picking import BinnedSet
from ..quality import FLAG_NAMES
from ..spectra import SpectrumSet
from .curves import component_curves
from .results import Results

LAYERS = ("raw", "picked", "fit")
# one colour per population, by rank of the mean (lowest deuteration first): Okabe-Ito colours,
# which stay distinguishable with red-green colour blindness
COLORS = ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#A6A100")
# after every colour has been used once the populations repeat the colours with another line
# style (and hollow markers in the population map), so any K stays readable
LINE_STYLES = ("--", ":", "-.", (0, (6, 2, 1, 2, 1, 2)))


def population_style(rank: int) -> dict:
    """Colour, line style and marker fill of population ``rank`` (1 = lowest mean): the colours
    of ``COLORS`` in turn, then the same colours again with the next line style and with hollow
    markers, so that populations stay distinct for any K."""
    cycle, c = divmod(rank - 1, len(COLORS))
    return {"color": COLORS[c], "linestyle": LINE_STYLES[cycle % len(LINE_STYLES)],
            "fillstyle": "full" if cycle % 2 == 0 else "none"}


# the total fit has a colour of its own (raw and picked data are grey), not one of the populations'
FIT_COLOR = "#D55E00"
RAW = "#555555"
RAW_DARK = "#222222"
PICKED_PALE = "#c4c4c4"
PICKED_BAR = "#8a8a8a"
MIN_WINDOW_LW = 1.2  # points: the least line width of a picked window, so it is always visible
MARKER_SIZE = 16  # area of the circle on top of a picked bin
STICK = "#444444"


def _break_gaps(mz: np.ndarray, y: np.ndarray, factor: float = 10.0):
    """The raw points with a NaN inserted wherever the m/z gap is more than ``factor`` times the
    median spacing, so that a line is not drawn across data that is not there (sparse or
    centroided spectra would otherwise show invented slopes between the peaks)."""
    mz, y = np.asarray(mz, dtype=float), np.asarray(y, dtype=float)
    if len(mz) < 3:
        return mz, y
    gaps = np.diff(mz)
    limit = factor * np.median(gaps[gaps > 0]) if (gaps > 0).any() else np.inf
    at = np.flatnonzero(gaps > limit) + 1
    return np.insert(mz, at, np.nan), np.insert(y, at, np.nan)


def _layers(show: Sequence[str]) -> tuple[str, ...]:
    show = (show,) if isinstance(show, str) else tuple(show)
    bad = [s for s in show if s not in LAYERS]
    if bad or not show:
        raise ValueError(f"show must be a non-empty subset of {LAYERS}, got {show!r}")
    return show


def _picked(spectrum_id: str, results: Results | None, binned: BinnedSet | None):
    """(m/z of the windows or None, n_deut bins, intensity) of the picked spectrum, or None.
    The windows sit at the predicted m/z, moved by ``window_shift_ppm`` where a bundle picked
    with the former ``recentre`` moved them (:mod:`hxex.picking.mz_check`)."""
    binned = binned if binned is not None else (results.binned if results else None)
    if binned is not None:
        try:
            i = binned.find(spectrum_id)
            mz, y, bins = binned.spectrum(i)
            centre = np.asarray(mz, float) * (1.0 + float(binned.window_shift_ppm[i]) * 1e-6)
            return centre, np.asarray(bins, float), np.asarray(y, float)
        except KeyError:
            pass
    if results is not None:
        try:
            i = results.find(spectrum_id)
        except KeyError:
            return None
        m = results.measurements
        return None, np.asarray(m.bins[i], float), np.asarray(m.observed[i], float)
    return None


def _charge_of(spectrum_id: str, results: Results | None, binned: BinnedSet | None) -> int:
    """The charge of a picked spectrum (1 when no table knows it)."""
    binned = binned if binned is not None else (results.binned if results else None)
    if binned is not None:
        try:
            return int(binned.charge[binned.find(spectrum_id)])
        except KeyError:
            pass
    return 1


def _window_label(cfg, recorded: dict) -> str:
    """How the picking window was set, for the legend: the ppm half-width, or the resolving
    power it follows (a setting, or the spectrum's recorded value)."""
    r = recorded.get("resolving_power")
    if cfg.uses_recorded and r is not None and np.isfinite(r) and r > 0:
        return f"window from the recorded R {r:g}"
    if cfg.resolving_power is None or cfg.uses_recorded:
        return f"±{cfg.resolution_ppm:g} ppm window"
    return f"window from R {cfg.resolving_power:g} at m/z {cfg.resolving_power_mz:g}"


def _recorded(spectrum_id: str, results: Results | None, spectra: SpectrumSet | None) -> dict:
    """The spectrum's recorded resolving power and reference m/z (NaN when unknown), for
    :meth:`hxex.picking.PickerConfig.half_width`."""
    for source in (spectra, results.spectra if results is not None else None):
        if source is not None:
            try:
                i = source.find(spectrum_id)
            except KeyError:
                continue
            return {"resolving_power": float(source.resolving_power[i]),
                    "resolving_power_mz": float(source.resolving_power_mz[i])}
    return {"resolving_power": None, "resolving_power_mz": None}


def spectrum_data(
    spectrum_id: str,
    results: Results | None = None,
    *,
    spectra: SpectrumSet | None = None,
    binned: BinnedSet | None = None,
    show: Sequence[str] = LAYERS,
    k: int | None = None,
    components: Sequence[int] | None = None,
    resolution_ppm: float | None = None,
) -> dict:
    """What :func:`plot_spectrum` draws, as arrays. The keys are the layers asked for that could
    be found (``raw``, ``picked``, ``fit``), each a dict:

    - ``raw``: ``mz``, ``intensity``;
    - ``picked``: ``mz`` (where the picker's windows were: the predicted m/z, moved for a series
      off its m/z; None without the picked table), ``bins`` (deuterium bins),
      ``intensity``, ``window`` (the half width in m/z of the picker's window around each bin,
      :meth:`hxex.picking.PickerConfig.half_width` of the results' picker settings with the
      spectrum's recorded resolving power, or ``mz * resolution_ppm / 1e6`` when that is given
      here; None when neither is known) and ``resolution_ppm`` (the settings' value);
    - ``fit``: ``k``, ``total`` (the whole fit), ``components`` (``(K, bins)``, ascending mean;
      ``shown`` lists the ranks, from 1, that were asked for), ``mean`` and ``population`` with
      ``mean_se`` and ``population_se`` (NaN without a bootstrap), ``corrected`` (back-exchange
      corrected means, or None without controls), ``band`` (``(2, bins)`` or None),
      ``component_band`` (``(K, 2, bins)``, the same band for each population, or None) and
      ``index`` (the position in the results).

    Missing layers are reported in ``missing`` (name to reason), not as errors, so that a GUI can
    ask for everything and draw what exists; :func:`plot_spectrum` raises only if nothing is left.
    """
    show = _layers(show)
    out: dict = {"spectrum_id": spectrum_id, "missing": {}}
    if "raw" in show:
        raw = results.raw_spectrum(spectrum_id) if results is not None else None
        if spectra is not None:
            try:
                raw = spectra.spectrum(spectra.find(spectrum_id))
            except KeyError:
                raw = None
        if raw is None:
            out["missing"]["raw"] = "no raw spectrum with this id"
        else:
            out["raw"] = {"mz": np.asarray(raw[0], float), "intensity": np.asarray(raw[1], float)}
    picked = _picked(spectrum_id, results, binned) if show != ("raw",) else None
    if "picked" in show:
        if picked is None:
            out["missing"]["picked"] = "no picked table or results with this id"
        else:
            ppm, window = resolution_ppm, None
            cfg = (results.settings.picker if results is not None
                   and results.settings is not None else None)
            label = None if ppm is None else f"±{ppm:g} ppm window"
            if ppm is not None and picked[0] is not None:
                window = picked[0] * ppm / 1e6
            elif cfg is not None:
                ppm = cfg.resolution_ppm
                recorded = _recorded(spectrum_id, results, spectra)
                label = _window_label(cfg, recorded)
                if picked[0] is not None:
                    window = cfg.half_width(picked[0], _charge_of(spectrum_id, results, binned),
                                            **recorded)
            out["picked"] = {"mz": picked[0], "bins": picked[1], "intensity": picked[2],
                             "window": window, "resolution_ppm": ppm, "window_label": label}
    if "fit" in show:
        if results is None:
            out["missing"]["fit"] = "no results"
        else:
            try:
                out["fit"] = _fit_data(results, spectrum_id, k, components)
            except KeyError:
                out["missing"]["fit"] = "the results do not contain this spectrum"
            else:
                if picked is not None and picked[0] is not None:
                    out["fit"]["mz"] = picked[0]
    return out


def _fit_data(results: Results, spectrum_id: str, k: int | None, shown) -> dict:
    i = results.find(spectrum_id)
    k = int(results.effective_K[i] if k is None else k)
    results.fit.require_fitted(i, k)
    shown = list(range(1, k + 1)) if shown is None else sorted({int(c) for c in shown})
    if any(c < 1 or c > k for c in shown):
        raise ValueError(f"components must be ranks from 1 to {k}, got {shown}")
    kf = results.fit.fits[k]
    bins = np.asarray(results.measurements.bins[i], dtype=float)
    order = np.argsort(kf.means[i])
    curves = component_curves(
        bins, kf.A[i][order], kf.n[i][order], kf.p[i][order], results.measurements.shapes[i]
    )
    mean, frac = kf.means[i][order], kf.populations[i][order]
    nan = np.full(k, np.nan)
    mean_se, population_se, band, component_band = nan, nan, None, None
    boot = results.bootstrap_for(i, k)
    if boot is not None:
        mean_se, population_se = boot.mean_se[i, :k][order], boot.population_se[i, :k][order]
        lo, hi = boot.curve_band(i)
        if len(lo) == len(bins):
            band = np.vstack([lo, hi])
        parts = boot.curve_components(i)
        if parts is not None and parts[0].shape == (k, len(bins)):
            component_band = np.stack([parts[0][order], parts[1][order]], axis=1)  # (K, 2, n)
    from .series import _control_row

    corrected = None
    if _control_row(results, int(results.fit.data_id[i])) is not None:
        corrected = correct_means(results.controls, [results.fit.data_id[i]], mean[None, :]).mean[0]
    return {
        "index": i, "k": k, "bins": bins, "total": curves.sum(axis=0), "components": curves,
        "shown": shown, "mean": mean, "population": frac, "mean_se": mean_se,
        "population_se": population_se, "corrected": corrected, "band": band,
        "component_band": component_band,
    }


def plot_spectrum(
    spectrum_id: str,
    results: Results | None = None,
    *,
    spectra: SpectrumSet | None = None,
    binned: BinnedSet | None = None,
    show: Sequence[str] = LAYERS,
    k: int | None = None,
    components: Sequence[int] | None = None,
    ax: Axes | None = None,
    rax: Axes | None = None,
    residual: bool = False,
    band: bool = True,
    component_band: bool = True,
    normalize: bool = False,
    corrected: bool | None = None,
    xaxis: str | None = None,
    picked_style: str = "auto",
    resolution_ppm: float | None = None,
    legend: bool = True,
    title: bool = True,
) -> Figure:
    """Draw one spectrum. Returns the figure; nothing is shown or saved.

    ``show`` picks the layers (a subset of ``'raw'``, ``'picked'``, ``'fit'``), each drawn if its
    source is available (see :func:`spectrum_data`); a layer that cannot be found is left out
    and a ValueError is raised only if nothing at all can be drawn. ``k`` defaults to the K in
    effect (the reviewer's choice, else the selected K). ``components`` lists the populations to
    draw as ranks of the mean from 1 (default: all); the total fit and the bootstrap band are
    always those of the whole fit. ``band`` and ``component_band`` switch the bootstrap bands of
    the total fit and of each drawn population (in its own colour). ``normalize`` divides by the
    tallest peak drawn. ``corrected`` (default: when the results have controls) adds the
    back-exchange-corrected mean to each population's label. ``picked_style`` is ``'window'``
    (each picked bin as a bar spanning the picker's window around the predicted m/z, from the
    results' picker settings (and the spectrum's recorded resolving power) or from
    ``resolution_ppm`` given here; it is shaded, not outlined, and never thinner than
    ``MIN_WINDOW_LW`` points, so it is visible at any zoom), ``'stick'`` (a thin line at the
    bin) or ``'auto'`` (a window when it is known and the axis is m/z, else a stick). Either
    way every picked bin has a circle on top. The fit line has its own colour, ``FIT_COLOR``,
    and the populations theirs
    (``COLORS``, then the same colours with another line style beyond ``len(COLORS)``
    populations, see :func:`population_style`; the population map uses the same ones).
    ``xaxis`` is ``'mz'`` (default when raw data is drawn or the picked table is available) or
    ``'n_deut'``. ``residual`` adds a panel of picked minus fit. With ``ax`` the spectrum is
    drawn there, and with ``residual`` its residuals in ``rax``, which must then be given. Every
    artist has a ``gid``: ``raw``, ``picked``, ``fit``, ``band``, ``component:<rank>``,
    ``component_band:<rank>`` and ``residual``.
    """
    if picked_style not in ("auto", "window", "stick"):
        raise ValueError("picked_style must be 'auto', 'window' or 'stick'")
    d = spectrum_data(spectrum_id, results, spectra=spectra, binned=binned, show=show, k=k,
                      components=components, resolution_ppm=resolution_ppm)
    layers = [name for name in LAYERS if name in d]
    if not layers:
        raise ValueError(f"nothing to draw for {spectrum_id}: {d['missing']}")
    fit, picked, raw = d.get("fit"), d.get("picked"), d.get("raw")
    if xaxis is None:
        mz_known = raw is not None or (picked is not None and picked["mz"] is not None) or (
            fit is not None and "mz" in fit)
        xaxis = "mz" if mz_known else "n_deut"
    if xaxis not in ("mz", "n_deut"):
        raise ValueError("xaxis must be 'mz' or 'n_deut'")
    if xaxis == "n_deut" and raw is not None:
        raise ValueError("raw spectra are on the m/z axis: use xaxis='mz' or leave raw out")
    if xaxis == "mz" and (
        (fit is not None and "mz" not in fit)
        or (picked is not None and picked["mz"] is None)
    ):
        raise ValueError("the m/z axis for picked peaks and the fit needs the picked table")
    if residual and (fit is None or picked is None):
        raise ValueError("residuals need both the picked and the fit layers")
    own = ax is None  # a figure of its own is laid out here; a caller's is the caller's to lay out
    if ax is None:
        fig = Figure(figsize=(5.5, 4.5 if residual else 3.5))
        if residual:
            ax, rax = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        else:
            ax, rax = fig.subplots(), None
    else:
        fig = ax.figure
        if residual and rax is None:
            raise ValueError("with ax given, residuals need their own axes: pass rax")
        rax = rax if residual else None
    peak = max([
        float(np.max(layer[key])) for layer, key in ((raw, "intensity"), (picked, "intensity"),
                                                     (fit, "total")) if layer is not None
    ] or [1.0])
    scale = 1.0 / peak if normalize and peak > 0 else 1.0
    both = raw is not None and picked is not None
    if picked is not None:
        # with the raw profile on top, the picked bins are pale so that both can be read
        x = picked["mz"] if xaxis == "mz" else picked["bins"]
        has_window = xaxis == "mz" and picked["window"] is not None
        if picked_style == "window" and not has_window:
            raise ValueError("the picking window needs the m/z axis and the picker's "
                             "resolution_ppm (not in the results' settings: pass resolution_ppm)")
        if has_window and picked_style in ("auto", "window"):
            shade = PICKED_PALE if both else PICKED_BAR
            # shading only: the edge has the colour of the fill, so it adds no outline but keeps
            # a window at least MIN_WINDOW_LW points wide at any zoom
            ax.bar(x, picked["intensity"] * scale, width=2 * picked["window"],
                   color=shade, edgecolor=shade, lw=MIN_WINDOW_LW,
                   label=f"picked ({picked.get('window_label') or 'window'})", gid="picked",
                   zorder=2)
        else:
            ax.vlines(x, 0, picked["intensity"] * scale, color=PICKED_PALE if both else STICK,
                      lw=5.0 if both else 2.0, label="picked", gid="picked", zorder=2)
        # a circle on top of every stick or window, as in the pyHXExpress plots
        ax.scatter(x, picked["intensity"] * scale, s=MARKER_SIZE, color=STICK, lw=0, zorder=4,
                   gid="picked")
    if raw is not None:
        rx, ry = _break_gaps(raw["mz"], raw["intensity"])
        ax.plot(rx, ry * scale, color=RAW_DARK if both else RAW,
                lw=0.9 if both else 1.0, label="raw", gid="raw", zorder=3)
    if fit is not None:
        x = fit["mz"] if xaxis == "mz" else fit["bins"]
        if band and fit["band"] is not None:
            ax.fill_between(x, fit["band"][0] * scale, fit["band"][1] * scale, color=FIT_COLOR,
                            alpha=0.18, lw=0, label="bootstrap band", gid="band", zorder=1)
        show_corr = fit["corrected"] is not None if corrected is None else corrected
        for rank in fit["shown"]:
            c = rank - 1
            style = population_style(rank)
            color = style["color"]
            label = _label(fit, c, show_corr and fit["corrected"] is not None)
            ax.plot(x, fit["components"][c] * scale, color=color, ls=style["linestyle"], lw=1.2,
                    label=label,
                    gid=f"component:{rank}", zorder=3)
            if component_band and fit["component_band"] is not None:
                ax.fill_between(x, fit["component_band"][c, 0] * scale,
                                fit["component_band"][c, 1] * scale, color=color, alpha=0.18,
                                lw=0, gid=f"component_band:{rank}", zorder=1)
        ax.plot(x, fit["total"] * scale, color=FIT_COLOR, lw=1.6, label="fit", gid="fit", zorder=5)
        if rax is not None:
            rax.axhline(0, color="grey", lw=0.8)
            rax.vlines(x, 0, (picked["intensity"] - fit["total"]) * scale, color=STICK, lw=2.0,
                       gid="residual")
            rax.set_ylabel("residual")
    ax.set_ylabel("normalised intensity" if normalize else "intensity")
    if title:
        ax.set_title(_title(results, d), fontsize=9)
    if legend:
        ax.legend(fontsize=7, frameon=False)
    (rax or ax).set_xlabel("m/z" if xaxis == "mz" else "deuterium level (bin)")
    if own:
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - layout is cosmetic
            pass
    return fig


def _label(fit: dict, c: int, corrected: bool) -> str:
    text = f"pop {c + 1}: {fit['mean'][c]:.2f}"
    if np.isfinite(fit["mean_se"][c]):
        text += f" ± {fit['mean_se'][c]:.2f}"
    text += f" D, {100 * fit['population'][c]:.0f}%"
    if corrected:
        text += f" (corr. {fit['corrected'][c]:.2f})"
    return text


def _title(results: Results | None, d: dict) -> str:
    notes = []
    fit = d.get("fit")
    if fit is not None:
        notes.append(f"K={fit['k']}")
        if fit["shown"] != list(range(1, fit["k"] + 1)):
            names = ", ".join(f"pop {r}" for r in fit["shown"])
            notes.append(f"showing {names} (total is the whole fit)")
        i = fit["index"]
        if results is not None and results.review is not None:
            status = str(results.review_status[i])
            if status != "unreviewed":
                notes.append(status)
        if results is not None and results.quality is not None and results.quality.flagged[i]:
            flags = [n for n in FLAG_NAMES
                     if getattr(results.quality, n)[i]]
            notes.append("flags: " + ", ".join(flags))
    return "\n".join([str(d["spectrum_id"]), "  ".join(notes)]) if notes else str(d["spectrum_id"])
