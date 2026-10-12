# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the hxex logo: ``python docs/images/make_logo.py`` writes ``hxex_logo.svg`` (for light
backgrounds) and ``hxex_logo_dark.svg`` (for dark ones) beside this file.

The h is a bimodal stick spectrum: two binomial populations (30 sites, 12 % and 34 % deuterium,
55:45) whose sum has the shape of an h (a tall stem, a shallow valley, an arch at x-height and a
leg to the baseline), with their fitted envelopes; the picked points stop where the spectrum
reaches the baseline and the fit runs on under the word. Then a math x, a Sigma for the e and
another x, as outlines of matplotlib's STIX math glyphs (SIL Open Font License), so the logo
needs no font. A halo in the background colour behind each glyph keeps overlapping letters
apart.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from scipy.stats import binom  # noqa: E402

THEMES = {  # ink (letters and fit), background (halos), sticks
    "hxex_logo": ("#1d2a44", "#ffffff", "#a8b3bf"),
    "hxex_logo_dark": ("#e6edf3", "#0d1117", "#57606a"),  # GitHub's dark background
}
POP1, POP2, DOT = "#2a9d8f", "#e76f51", "#d62839"

XH = 150.0  # x-height (data units): the x, the Sigma and the arch of the h
ASC = 1.75 * XH  # the stem of the h
H_WIDTH = 175.0  # the h, from the first stick to where the spectrum reaches the baseline
H_GAP = -0.42 * XH  # the h to the first x (overlapping)
GAP = -0.16 * XH  # the first x to the Sigma
TOPS_GAP = 0.08 * XH  # between the top of the Sigma and the top of the last x
PAST_FOOT = 0.04 * XH  # the fit ends this far past the foot of the last x
HALO = 4.0  # points
FONTSET = "stix"


def _glyph_path(tex):
    path = TextPath((0, 0), tex, size=100, prop=FontProperties(math_fontfamily=FONTSET))
    bb = path.get_extents()
    k = XH / bb.height
    return (path.vertices - [bb.x0, bb.y0]) * k, path.codes, bb.width * k


def _reach(tex, lo, hi):
    """The left and right reach of a glyph between ``lo`` and ``hi`` (fractions of its
    height), from its left edge."""
    v, _, _ = _glyph_path(tex)
    band = v[(v[:, 1] >= lo * XH) & (v[:, 1] <= hi * XH), 0]
    return float(band.min()), float(band.max())


def _populations(n_bins, floor=0.03):
    i = np.arange(n_bins)
    p1 = 0.55 * binom.pmf(i, 30, 0.12)
    p2 = 0.45 * binom.pmf(i, 30, 0.34)
    total = p1 + p2
    last = int(np.flatnonzero(total > floor * total.max()).max()) + 1  # first bin at baseline
    return i, p1, p2, last


def _smooth(x, y, k=14):
    """A Catmull-Rom curve through the points."""
    pts = np.vstack([[x[0], y[0]], np.column_stack([x, y]), [x[-1], y[-1]]])
    out = []
    for j in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[j - 1 : j + 3]
        for t in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(pts[-2])
    out = np.array(out)
    return out[:, 0], np.maximum(out[:, 1], 0)


def draw(name, ink, background, stick, directory):
    _, _, w_x = _glyph_path(r"$x$")
    _, _, w_sigma = _glyph_path(r"$\Sigma$")
    # the last x: its top starts TOPS_GAP after the end of the Sigma's top bar
    last_gap = _reach(r"$\Sigma$", 0.8, 1.0)[1] - w_sigma + TOPS_GAP - _reach(r"$x$", 0.8, 1.0)[0]
    x1 = H_WIDTH + H_GAP
    x_sigma = x1 + w_x + GAP
    x2 = x_sigma + w_sigma + last_gap
    fit_end = x2 + float(_glyph_path(r"$x$")[0][_glyph_path(r"$x$")[0][:, 1] < 0.15 * XH, 0].max())
    fit_end += PAST_FOOT

    i, p1, p2, last = _populations(200)
    pitch = H_WIDTH / last
    n = int(np.ceil(fit_end / pitch)) + 2  # bins out to the end of the fit
    i, p1, p2, total = i[:n], p1[:n], p2[:n], (p1 + p2)[:n]
    scale = ASC / total.max()
    x = i * pitch

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_aspect("equal")
    for xi, yi in zip(x, total * scale, strict=True):
        ax.plot([xi, xi], [0, yi], color=stick, lw=1.4, solid_capstyle="butt", zorder=1)
    for pop, colour in ((p1, POP1), (p2, POP2)):
        sx, sy = _smooth(x, pop * scale)
        keep = sx <= fit_end
        ax.fill_between(sx[keep], 0, sy[keep], color=colour, alpha=0.2, lw=0, zorder=2)
        ax.plot(sx[keep], sy[keep], color=colour, lw=2.4, zorder=3)
    sx, sy = _smooth(x, total * scale)
    keep = sx <= fit_end
    ax.plot(sx[keep], sy[keep], color=ink, lw=7, solid_capstyle="round", zorder=4)
    ax.scatter(x[: last + 1], total[: last + 1] * scale, s=20, color=DOT, lw=0, zorder=5)

    for z, (tex, left) in enumerate(((r"$x$", x1), (r"$\Sigma$", x_sigma), (r"$x$", x2))):
        v, codes, _ = _glyph_path(tex)
        path = MplPath(v + [left, 0], codes)
        ax.add_patch(PathPatch(path, facecolor=background, edgecolor=background, lw=HALO,
                               joinstyle="round", zorder=6 + 2 * z))
        ax.add_patch(PathPatch(path, facecolor=ink, lw=0, zorder=7 + 2 * z))

    ax.set_xlim(-25, x2 + w_x + 25)
    ax.set_ylim(-30, ASC + 30)
    ax.axis("off")
    with plt.rc_context({"svg.hashsalt": "hxex", "svg.fonttype": "path"}):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight", transparent=True,
                    metadata={"Date": None, "Creator": None})
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, (ink, background, stick) in THEMES.items():
        draw(name, ink, background, stick, here)
