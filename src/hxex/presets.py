# SPDX-License-Identifier: GPL-3.0-or-later
"""Presets: one setting, ``preset = "NAME"`` at the top of a project file, for how much work
``hxex run`` and ``hxex analyse`` do (docs/PLAN.md Next steps 28).

A preset is a set of project-file settings laid under the file's own: a setting written in the
file wins over the preset's, key by key into tables (:func:`expand`). Presets set effort only:
the numbers of populations tried, the population test's reference, the bootstrap, the ladder
rates, and which series get figures. What describes the experiment, the review, the statistical
policy (``alpha``, ``[analysis.filters]``, ``[analysis.choice]``, the control thresholds), and
the instrument settings are never in a preset, so the levels differ only by the evidence they
add. Without a preset every setting keeps its default, as before presets existed.

- ``quick``: K = 1 and 2, the population test with the chi-square reference (no simulations),
  no bootstrap. For the review, the retention windows, the controls, and a first look at the
  analysis; its results are provisional (:data:`PROVISIONAL`).
- ``bimodal``: K = 1 to 3, the simulated population test deciding one against two (the step to
  three only flags ``more_populations_possible``), a bootstrap with K = 2 for every spectrum
  (the K = 2 comparisons between samples need it) and each set's K (``set_k``); ladder rates
  for fragment ions.
- ``multimodal``: K = 1 to ``k_max`` (3 unless the file sets it), the simulated test at every
  step, a bootstrap at each spectrum's K and its set's (``set_k``); ladder rates.
- ``maximum``: ``multimodal`` with 100 resamples (for the 95 % intervals) and the figures of
  every series.

Analysis settings (``[analysis.fragments] fit_rates``) apply only to a file with an
``[analysis]`` table: a preset never switches the analysis on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# the settings of each preset, as a project file's tables (TOML data)
_COMPLETE_RUN = {"fit": {"k_min": 1, "k_max": 3},
                 "population_test": {"null": "simulated", "decide_up_to": False},
                 "bootstrap": {"ks": "selected", "set_k": True}}
PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "run": {"fit": {"k_min": 1, "k_max": 2},
                "population_test": {"null": "chi_square", "decide_up_to": False},
                "bootstrap": False},
    },
    "bimodal": {
        "run": {"fit": {"k_min": 1, "k_max": 3},
                "population_test": {"null": "simulated", "decide_up_to": 2},
                "bootstrap": {"ks": [2], "set_k": True}},
        "analysis": {"fragments": {"fit_rates": True}},
    },
    "multimodal": {
        "run": _COMPLETE_RUN,
        "analysis": {"fragments": {"fit_rates": True}},
    },
    "maximum": {
        "run": {**_COMPLETE_RUN, "bootstrap": {**_COMPLETE_RUN["bootstrap"], "n_boot": 100}},
        "analysis": {"fragments": {"fit_rates": True}},
        "figures": {"series": "all"},
    },
}
# the presets whose results are a first look, marked so in hxex info, the log and the figures
PROVISIONAL = ("quick",)
# one line each, for hxex check, the settings reference and the window
SUMMARIES = {
    "quick": "K = 1 and 2, the population test with the chi-square reference (no "
             "simulations), no bootstrap: a first look, provisional",
    "bimodal": "K = 1 to 3, the simulated population test deciding one or two (three only "
               "flagged), a bootstrap with K = 2 for every spectrum, ladder rates",
    "multimodal": "K = 1 to k_max (3), the simulated population test at every step, a "
                  "bootstrap at each spectrum's K and its set's, ladder rates",
    "maximum": "multimodal with 100 resamples and the figures of every series (a GPU "
               "recommended)",
}


@dataclass(frozen=True)
class Expansion:
    """What a preset did to a project file's data: ``name`` (None: no preset), ``data`` (the
    file's data with the preset laid under it, without the ``preset`` key), ``given`` (setting
    -> the preset's value, for every setting the preset sets and the file does not), and
    ``overridden`` (setting -> (the file's value, the preset's) where the file sets it
    otherwise). Settings are named ``table.key`` (``run.fit.k_max``); a table the preset turns
    on or off is named alone (``run.bootstrap``)."""

    name: str | None
    data: dict[str, Any]
    given: dict[str, Any] = field(default_factory=dict)
    overridden: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def provisional(self) -> bool:
        return self.name in PROVISIONAL

    def record(self) -> dict[str, Any] | None:
        """The preset as recorded with the results (None: no preset)."""
        if self.name is None:
            return None
        return {"name": self.name, "provisional": self.provisional,
                "overridden": {k: [_plain(a), _plain(b)] for k, (a, b) in self.overridden.items()}}


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


def names() -> tuple[str, ...]:
    return tuple(PRESETS)


def _lay(preset: Mapping, data: Mapping, prefix: str, given: dict, overridden: dict) -> dict:
    """``data`` over ``preset``, key by key into tables both have; ``given`` and ``overridden``
    collect what the preset set and what the file set otherwise."""
    out = dict(data)
    for key, value in preset.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in data:
            out[key] = value
            _collect(value, name, given)
        elif isinstance(value, Mapping) and isinstance(data[key], Mapping):
            out[key] = _lay(value, data[key], name, given, overridden)
        elif data[key] != value:
            overridden[name] = (data[key], value)
    return out


def _collect(value, name: str, given: dict) -> None:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            _collect(inner, f"{name}.{key}", given)
    else:
        given[name] = value


def expand(data: Mapping) -> Expansion:
    """The project file's ``data`` with its preset laid under it (:class:`Expansion`). Raises
    ValueError for a preset hxex does not know. ``preset = false``, or none, changes nothing."""
    name = data.get("preset")
    rest = {k: v for k, v in data.items() if k != "preset"}
    if name is None or name is False:
        return Expansion(None, rest)
    if not isinstance(name, str) or name not in PRESETS:
        raise ValueError(f"preset = {name!r}: expected one of "
                         + ", ".join(f'"{n}"' for n in PRESETS) + " (or leave it out for "
                         "the defaults)")
    preset = dict(PRESETS[name])
    if "analysis" not in data:  # a preset never switches the analysis on
        preset.pop("analysis", None)
    given: dict[str, Any] = {}
    overridden: dict[str, tuple[Any, Any]] = {}
    out = _lay(preset, rest, "", given, overridden)
    return Expansion(name, out, given, overridden)


NOT_SET = object()


def preset_value(name, setting: str):
    """The value preset ``name`` gives ``setting`` (``table.key``), or :data:`NOT_SET` (no such
    preset, or it leaves the setting alone). A table the preset turns off (``run.bootstrap =
    false``) gives its settings no value."""
    if name not in PRESETS:
        return NOT_SET
    value: Any = PRESETS[name]
    for part in setting.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return NOT_SET
        value = value[part]
    return value


def setting_names() -> frozenset[str]:
    """Every setting some preset sets (``table.key``; tables only by their settings)."""
    out: dict[str, Any] = {}
    for preset in PRESETS.values():
        for key, value in preset.items():
            _collect(value, key, out)
    return frozenset(out)


def value_text(value) -> str:
    """A setting's value as a project file writes it."""
    if value is True or value is False:
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(value_text(v) for v in value) + "]"
    if isinstance(value, Mapping):
        return "a table"
    return str(value)


def settings_text(name: str) -> list[str]:
    """Every setting of preset ``name`` as ``table.key = value`` lines (a table turned off as
    ``table = false``)."""
    given: dict[str, Any] = {}
    for key, value in PRESETS[name].items():
        _collect(value, key, given)
    return [f"{k} = {value_text(v)}" for k, v in given.items()]
