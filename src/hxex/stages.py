# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a project's results stand, stage by stage (docs/PLAN.md Next steps 40, step 2).

A bundle is made by a chain of stages: the **read** (the spectra from the input files), the
**pick** (the bins, the peak measure, the envelopes), the **noise** (the per-bin curve from the
undeuterated controls), the **fit**, the **population test**, the **correction** (the
back-exchange controls and the quality flags, from the K = 1 fits), and the **bootstrap**; the
analysis folders come after them all. What each stage depends on is recorded in the bundle:
the settings (``configs.run``, and the read plan in the read record, :mod:`hxex.stored_read`),
the identities of the data files, and the code that ran it (:func:`record`: the manifest's
``stages`` block, with the fitting code that :mod:`hxex.reuse` compares).

:func:`standing` compares a project with its bundle and gives each stage one state: ``done``,
``out of date`` (with each cause named: ``[run.controls] fulldeut_at: 27 h, was FD``, or "after
the fit" when a stage it comes after is out of date), ``not done`` (never made, or stopped
partway), or ``off`` (not asked for); and what ``hxex run`` would redo now
(:func:`run_plan`). ``hxex check`` and ``hxex info`` print them (:func:`lines`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

STAGES = ("read", "pick", "noise", "fit", "population_test", "correction", "bootstrap")
LABELS = {**{s: s.replace("_", " ") for s in STAGES}, "correction": "correction and flags"}
# the stage each one is made from (the chain; a stage is out of date after any stage before it)
AFTER = {"read": None, "pick": "read", "noise": "pick", "fit": "noise",
         "population_test": "fit", "correction": "fit", "bootstrap": "fit"}

DONE, OUT_OF_DATE, NOT_DONE, OFF = "done", "out of date", "not done", "off"

# the [run] settings of each stage, by the start of their dotted path (the longest first); the
# control settings are compared on their own (:func:`_control_causes`)
_RUN_STAGE = (
    ("fit.noise", "noise"),
    ("fit", "fit"),
    ("n_max_factor", "fit"),
    ("picker", "pick"),
    ("envelope_fallback_time", "pick"),
    ("envelope", "pick"),
    ("registry", "pick"),
    ("drop_no_signal", "pick"),
    ("population_test", "population_test"),
    ("quality", "correction"),
    ("bootstrap", "bootstrap"),
)
# where and how a computation runs, not what it gives (as hxex.reuse leaves them out), and what
# a bootstrap stores rather than its values
NOT_RESULTS = ("fit.backend", "memory_budget_bytes", "max_chunk", "bootstrap.keep_draws")
# the control settings that decide which spectra are controls
ROLE_KEYS = ("undeut_at", "fulldeut_at", "undeut_marked", "fulldeut_marked")
# the stages a change of each control setting reaches first (the others: the correction); the
# undeuterated controls give the envelopes' references, the peak measure and the noise, the
# fully deuterated ones are fitted at K = 1 and leave the population test's sets
_CONTROL_STAGES = {"undeut_at": ("pick", "noise"), "undeut_marked": ("pick", "noise"),
                   "fulldeut_at": ("fit", "correction"),
                   "fulldeut_marked": ("fit", "correction"),
                   "d2o_fraction": ("population_test", "correction")}


# --- the record --------------------------------------------------------------------------------

def current_codes() -> dict[str, str]:
    """The fingerprints of the code of each stage in this hxex (:mod:`hxex._engine`)."""
    from ._engine import (
        correction_fingerprint,
        engine_fingerprint,
        picking_fingerprint,
        reader_fingerprint,
    )

    engine = engine_fingerprint()
    return {"read": reader_fingerprint(), "pick": picking_fingerprint(), "noise": engine,
            "fit": engine, "population_test": engine, "correction": correction_fingerprint(),
            "bootstrap": engine}


def record(manifest: Mapping, codes: Mapping[str, str | None] | None = None) -> dict:
    """The ``stages`` block of a manifest: per stage, the ``code`` that made it (a fingerprint;
    None where not recorded) and its ``state`` (``done``, ``pending``: stopped partway, or
    ``off``). Worked out from the rest of the manifest: the read's code from the read plan, the
    noise's, fit's, population test's and bootstrap's from ``fitted_with``, the states from
    what the bundle holds and ``pending``; the picking's and the correction's code from
    ``codes`` (what made them in this bundle), else from the manifest's own ``stages`` block (a
    bundle written before 2026-11-20 has none: not recorded)."""
    configs = manifest.get("configs") or {}
    source = configs.get("spectra_source")
    plan = source.get("read_plan") if isinstance(source, Mapping) else None
    made = manifest.get("fitted_with") or {}
    engine = None if made.get("mixed") else made.get("engine")
    known = {k: (v or {}).get("code") for k, v in (manifest.get("stages") or {}).items()
             if isinstance(v, Mapping)}
    known.update({k: v for k, v in (codes or {}).items() if v is not None})
    pending = set(manifest.get("pending") or ())

    def entry(code, on: bool = True, stopped: bool = False) -> dict:
        return {"code": code, "state": "off" if not on else "pending" if stopped else "done"}

    return {
        "read": entry(plan.get("reader") if isinstance(plan, Mapping) else None),
        "pick": entry(known.get("pick")),
        "noise": entry(engine, manifest.get("noise_curve") is not None),
        "fit": entry(engine),
        "population_test": entry(engine, configs.get("population_test") is not None,
                                 "population_test" in pending),
        "correction": entry(known.get("correction"), configs.get("controls") is not None
                            or configs.get("quality") is not None),
        "bootstrap": entry(engine, configs.get("bootstrap") is not None,
                           "bootstrap" in pending),
    }


def recorded_code(manifest: Mapping, stage: str) -> str | None:
    """The fingerprint of the code that made ``stage`` of a bundle (None: not recorded)."""
    return record(manifest)[stage]["code"]


# --- words -------------------------------------------------------------------------------------

def under(path: str, prefixes) -> bool:
    return any(path == p or path.startswith(p + ".") for p in prefixes)


def at(record_, path: str):
    """The value at the dotted ``path`` of a nested record (None where absent)."""
    for part in path.split("."):
        if not isinstance(record_, Mapping):
            return None
        record_ = record_.get(part)
    return record_


def value_text(value) -> str:
    from .config import toml_value

    if value is None:
        return "unset"
    text = toml_value(value) if isinstance(value, (bool, int, float, str)) else json.dumps(
        value, default=str)
    return text if len(text) <= 40 else text[:37] + "..."


def setting_name(path: str) -> str:
    """``[run.controls] fulldeut_at`` for ``controls.fulldeut_at``; ``[run] controls`` for a
    whole table."""
    head, _, key = path.rpartition(".")
    return f"[run.{head}] {key}" if head else f"[run] {key}"


def role_text(key: str, value) -> str:
    from .io import format_time

    if key.endswith("_at"):
        return "unset" if value is None else format_time(float(value))
    return value_text(value)


def run_tables_given(project) -> set[str] | None:
    """The ``[run]`` keys a project file without ``[input]`` writes (its preset's among them),
    the only ones compared with the bundle's; None for a project with ``[input]`` (every
    setting compared: a run would apply each, defaults included)."""
    if project.input is not None:
        return None
    import tomllib

    try:
        with open(project.path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, TypeError, tomllib.TOMLDecodeError):
        return set()
    keys = set((data.get("run") or {}).keys()) if isinstance(data.get("run"), Mapping) else set()
    if project.preset is not None:
        keys |= {name.split(".")[1] for name in project.preset.given
                 if name.startswith("run.") and name.count(".") >= 1}
    return keys


def _normal(value):
    return json.loads(json.dumps(value, default=str))


def _stage_of(path: str) -> str:
    for prefix, stage in _RUN_STAGE:
        if under(path, (prefix,)):
            return stage
    return "fit"  # a setting added without a stage: counted where it costs the most


# --- the comparison ----------------------------------------------------------------------------

@dataclass
class Stage:
    """One stage's standing: ``state`` (:data:`DONE`, :data:`OUT_OF_DATE`, :data:`NOT_DONE`,
    :data:`OFF`) and its ``causes``, in words."""

    name: str
    state: str
    causes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LABELS[self.name]

    def text(self) -> str:
        return f"{self.label}: {self.state}" + (f" ({'; '.join(self.causes)})"
                                                 if self.causes else "")


@dataclass
class Standing:
    """Where a bundle stands against a project: the ``stages`` in order, ``notes`` (what could
    not be compared, said once), and ``run``: what ``hxex run`` would do now, in words (empty
    when there is no project to run)."""

    stages: list[Stage]
    notes: list[str] = field(default_factory=list)
    run: str = ""
    # how the run replaces the results: the option it needs, or the [output] switch that does it
    replacing: str = ""

    def __getitem__(self, name: str) -> Stage:
        return next(s for s in self.stages if s.name == name)


def _wanted(project) -> dict[str, bool]:
    """Which stages the project asks for."""
    run = project.run
    return {"read": True, "pick": True,
            "noise": getattr(run.fit.noise, "estimate", None) == "controls",
            "fit": True, "population_test": run.population_test is not None,
            "correction": run.controls is not None or run.quality is not None,
            "bootstrap": run.bootstrap is not None}


def standing(project, bundle: str | Path | None) -> Standing:
    """Where the bundle of ``project`` (``bundle``, default its ``[output] bundle``; ``project``
    None: the bundle alone, judged only on the code that made it) stands, stage by stage (see
    the module)."""
    from .io.bundle import BundleError, bundle_manifest

    if bundle is None and project is not None:
        bundle = project.output.bundle
    manifest = None
    if bundle is not None and Path(bundle).exists():
        try:
            manifest = bundle_manifest(Path(bundle))
        except BundleError:
            manifest = None
    if manifest is None:
        wanted = _wanted(project) if project is not None else dict.fromkeys(STAGES, True)
        return Standing([Stage(s, NOT_DONE if wanted[s] else OFF,
                               ["no results yet"] if wanted[s] else [])
                         for s in STAGES], run="does every stage" if project is not None
                        else "")
    rec = record(manifest)
    causes: dict[str, list[str]] = {s: [] for s in STAGES}
    notes: list[str] = []
    facts: dict = {}
    _code_causes(manifest, rec, causes, notes, facts)
    given = None
    wanted = {s: rec[s]["state"] != "off" for s in STAGES}
    if project is not None:
        given = run_tables_given(project)
        asked = _wanted(project)
        held_run = _held_run(manifest)
        if held_run is None:
            notes.append("the results do not record their run settings")
        else:
            _setting_causes(project, held_run, given, causes, facts)
            _control_causes(project, Path(bundle), held_run, given, causes, facts)
        for s in STAGES:  # a stage the file does not set keeps the results' choice
            if given is None or _stage_given(s, given):
                wanted[s] = asked[s]
        if project.input is not None:
            _input_causes(project, Path(bundle), manifest, causes, notes, facts)
        _review_causes(project, Path(bundle), causes)
    stages = []
    for s in STAGES:
        state = rec[s]["state"]
        own = causes[s]
        if state == "off" and not wanted[s]:
            stages.append(Stage(s, OFF, []))
        elif state == "off":
            if s == "noise" and not own:  # asked for, unchanged: too few controls to measure
                stages.append(Stage(s, OFF, ["no per-bin noise curve in the results: too few "
                                             "undeuterated controls to measure it"]))
            else:
                stages.append(Stage(s, NOT_DONE, ["the results have none"] + own))
        elif not wanted[s]:
            stages.append(Stage(s, OUT_OF_DATE, ["not asked for any more"]))
        elif state == "pending":
            stages.append(Stage(s, NOT_DONE, ["stopped partway: hxex resume continues it"]
                                + own))
        else:
            stages.append(Stage(s, OUT_OF_DATE if own else DONE, list(own)))
    by_name = {st.name: st for st in stages}
    for st in stages:  # after a stage that is out of date
        if st.state != DONE:
            continue
        before = AFTER[st.name]
        while before is not None:
            up = by_name[before]
            if up.state == OUT_OF_DATE or (up.state == NOT_DONE
                                           and "stopped partway" not in " ".join(up.causes)):
                st.state, st.causes = OUT_OF_DATE, [f"after the {up.label}"]
                break
            before = AFTER[before]
    out = Standing(stages, notes)
    if project is not None:
        out.run = run_plan(project, manifest, out, facts)
        output = project.output
        out.replacing = ("with [output] keep_previous = true (the results now kept as "
                         f"{Path(bundle).name}.N)" if output.keep_previous
                         else "with [output] overwrite = true" if output.overwrite
                         else "with --overwrite or --keep-previous")
    return out


def _stage_given(stage: str, given: set[str]) -> bool:
    keys = {"noise": ("fit",), "fit": ("fit",), "pick": ("picker", "envelope"),
            "population_test": ("population_test",), "correction": ("controls", "quality"),
            "bootstrap": ("bootstrap",)}.get(stage, ())
    return any(k in given for k in keys)


def _held_run(manifest: Mapping) -> dict | None:
    from .pipeline import RunConfig

    stored = ((manifest.get("configs") or {}).get("run"))
    if stored is None:
        return None
    return _normal(RunConfig.from_dict(stored).to_dict())  # a key added since: its default


def _code_causes(manifest, rec, causes, notes, facts) -> None:
    """Causes from the code: another hxex's fitting, picking or correction code."""
    from .io.bundle import _versions
    from .reuse import fitted_with, same_engine

    now = current_codes()
    same, why = same_engine(fitted_with(manifest), _versions())
    if not same:
        facts["engine_differs"] = True
        first = "noise" if rec["noise"]["state"] != "off" else "fit"
        causes[first].append(why.replace("them", "it") if why else "another version of "
                             "hxex's fitting code")
    unknown = []
    for stage, what in (("pick", "picking"), ("correction", "correction")):
        code = rec[stage]["code"]
        if rec[stage]["state"] == "off":
            continue
        if code is None:
            unknown.append(what)
        elif code != now[stage]:
            causes[stage].append(f"another version of hxex's {what} code made it")
    if unknown:
        notes.append(f"the {' and '.join(unknown)} code is not recorded (results made before "
                     "hxex recorded it): those stages are judged on their settings alone")


def _setting_causes(project, held: dict, given, causes, facts) -> None:
    """Causes from the ``[run]`` settings (the controls apart)."""
    from .io.bundle import settings_changed

    wanted = _normal(project.run.to_dict())
    changed = []
    for path in settings_changed({"run": held}, {"run": wanted}):
        if under(path, NOT_RESULTS) or under(path, ("controls",)):
            continue
        if given is not None and path.split(".")[0] not in given:
            continue
        stage = _stage_of(path)
        changed.append(stage)
        causes[stage].append(f"{setting_name(path)}: {value_text(at(wanted, path))}, was "
                             f"{value_text(at(held, path))}")
    facts["settings_changed"] = changed


def _control_causes(project, bundle: Path, held: dict, given, causes, facts) -> None:
    """Causes from ``[run.controls]``: the roles reach the stages that use them
    (:data:`_CONTROL_STAGES`), the rest the correction. ``controls`` left out (automatic) is
    decided from the bundle's picked spectra, as ``hxex resume`` decides it."""
    from .controls import AUTO, ControlConfig

    if given is not None and "controls" not in given:
        return
    mine, theirs = project.run.controls, held.get("controls")
    if mine == AUTO:
        if project.input is None:
            return
        from .io.bundle import BundleError
        from .pipeline import bundle_controls

        try:
            mine, _ = bundle_controls(bundle, AUTO)
        except (BundleError, ValueError, OSError) as exc:
            log.debug("controls not decided from %s: %s", bundle, exc)
            return
    if mine is None and not isinstance(theirs, Mapping):
        return
    if mine is None:
        causes["correction"].append("[run] controls: false, was a back-exchange correction")
        return
    if not isinstance(theirs, Mapping):
        causes["correction"].append("[run.controls]: a back-exchange correction, the results "
                                    "have none")
        return
    now = _normal(mine.to_dict())
    was = _normal(ControlConfig.from_dict(theirs).to_dict())
    roles = []
    for key in sorted(set(now) | set(was)):
        if now.get(key) == was.get(key):
            continue
        text = (f"[run.controls] {key}: {role_text(key, now.get(key))}, was "
                f"{role_text(key, was.get(key))}" if key in ROLE_KEYS else
                f"[run.controls] {key}: {value_text(now.get(key))}, was "
                f"{value_text(was.get(key))}")
        if key in ROLE_KEYS:
            text += " (which spectra are controls)"
            roles.append(key)
        for stage in _CONTROL_STAGES.get(key, ("correction",)):
            causes[stage].append(text)
    facts["roles_changed"] = roles
    facts["d2o_changed"] = now.get("d2o_fraction") != was.get("d2o_fraction")


def _input_causes(project, bundle: Path, manifest, causes, notes, facts) -> None:
    """Causes from ``[input]``: the read plan and the data files (the read), the time scale
    (applied to the read), ``[[input.exclude]]`` (the fit)."""
    from .stored_read import PLAN_KEY, changed_files, plan_differences, read_plan

    inp = project.input
    configs = manifest.get("configs") or {}
    source = configs.get("spectra_source") or {}
    facts["spectra_stored"] = "spectra.parquet" in (manifest.get("files") or {})
    if inp.spectra is not None:
        facts["read"] = f"takes the spectra stored in {Path(inp.spectra).name}"
    else:
        stored = source.get(PLAN_KEY) if isinstance(source, Mapping) else None
        if stored is None:
            notes.append("the results do not record what their read depended on (read before "
                         "2026-11-16): the read is not compared")
            facts["read_unknown"] = True
        else:
            now = read_plan(project)
            if now is None:
                causes["read"].append("what the read depends on could not be worked out from "
                                      "the settings")
            else:
                causes["read"] += plan_differences(stored, now)
            files = changed_files(project, source)
            if files:
                n = len(files)
                causes["read"].append(
                    f"{n} data file{'s' if n > 1 else ''} changed: "
                    + ", ".join(Path(f.removesuffix(' (not found)')).name
                                + (" (not found)" if f.endswith("(not found)") else "")
                                for f in files[:3]) + (f" and {n - 3} more" if n > 3 else ""))
    record_ = (configs.get("project_file") or {}).get("input")
    if isinstance(record_, Mapping):
        rules = _normal([r.to_dict() for r in inp.exclude])
        if rules != _normal(record_.get("exclude") or []):
            causes["fit"].append(f"[[input.exclude]]: {len(rules)} rule(s), was "
                                 f"{len(record_.get('exclude') or [])}")
        if "time_scale" in record_ and _normal(inp.time_scale) != _normal(
                record_["time_scale"]):
            causes["read"].append(f"[input] time_scale: {value_text(inp.time_scale)}, was "
                                  f"{value_text(record_['time_scale'])}")


def _review_causes(project, bundle: Path, causes) -> None:
    """A cause of the fit: spectra whose K the project's review table chooses that the results
    did not fit (the screen skipped it)."""
    from .review_table import table_choices

    try:
        chosen = table_choices(project.review)
    except Exception as exc:  # noqa: BLE001 - hxex check reports the table's problems itself
        log.debug("review table not read: %s", exc)
        return
    if not chosen:
        return
    from .io.bundle import BundleError, open_bundle
    from .pipeline import _unfitted_choices

    try:
        with open_bundle(bundle) as b:
            fit = b.fit()
    except (BundleError, OSError, KeyError, ValueError):
        return
    n = int((_unfitted_choices(fit, chosen) > 0).sum())
    if n:
        causes["fit"].append(f"{n} spectr{'a' if n > 1 else 'um'} the review table gives a K "
                             "the results did not fit (hxex resume fits them)")


def run_plan(project, manifest: Mapping, standing_: Standing, facts: Mapping) -> str:
    """What ``hxex run`` (replacing the results: ``--overwrite``) does now, stage by stage, as
    it decides: the stored spectra are taken when the read is unchanged (HISTORY 397), the
    stored picking when the spectra are taken and the picking, its settings, and the control
    roles are unchanged, the fits, population test and bootstrap of the spectra whose input is
    unchanged when the fitting code and their settings are (:mod:`hxex.reuse`; the per-bin
    noise must not move either), and the noise and the correction are always worked out
    again."""
    if project.input is None:
        return ""
    parts = []
    read = standing_["read"]
    if "read" in facts:
        parts.append(facts["read"])
        taken = True
    elif (read.state == DONE and facts.get("spectra_stored")
          and not facts.get("read_unknown")):
        parts.append("takes the stored spectra")
        taken = True
    else:
        parts.append("reads the input again")
        taken = False
    pick = standing_["pick"]
    if (taken and pick.state == DONE and not facts.get("roles_changed")
            and recorded_code(manifest, "pick") is not None):
        parts.append("keeps the picking")
    else:
        parts.append("picks again")
    if standing_["noise"].state != OFF:
        parts.append("measures the per-bin noise again")
    fitting = ("fit", "population_test", "bootstrap")
    changed = set(facts.get("settings_changed") or ()) & {"fit", "noise", *fitting}
    engine_cause = bool(facts.get("engine_differs"))
    pending = any(st.state == NOT_DONE and "stopped partway" in " ".join(st.causes)
                  for st in standing_.stages)
    d2o = bool(facts.get("d2o_changed")) and standing_["population_test"].state != OFF
    if changed or engine_cause or pending or d2o:
        tables = {"noise": "[run.fit]", "fit": "[run.fit]",
                  "population_test": "[run.population_test]", "bootstrap": "[run.bootstrap]"}
        named = sorted({tables[c] for c in changed})
        why = ("other fitting code" if engine_cause else "a stage stopped partway" if pending
               else " and ".join(named) + " changed" if changed
               else "[run.controls] d2o_fraction changed: the population test uses it")
        parts.append(f"fits every spectrum again ({why})")
    else:
        kept = ["fits"] + [LABELS[s] for s in ("population_test", "bootstrap")
                           if standing_[s].state != OFF]
        parts.append("keeps the " + (", ".join(kept[:-1]) + " and " + kept[-1]
                                     if len(kept) > 1 else kept[0])
                     + " of the spectra whose input is unchanged"
                     + (" (if the per-bin noise does not move)"
                        if standing_["noise"].state != OFF else ""))
    if standing_["correction"].state != OFF:
        parts.append("works out the correction and the quality flags again")
    return ", ".join(parts)


def lines(st: Standing, *, run: bool = True) -> list[tuple[str, str]]:
    """``(key, text)`` rows for ``hxex check`` and ``hxex info``: one per stage, the notes, and
    (``run``) what ``hxex run`` would do."""
    rows = [("" if i else "stages", s.text()) for i, s in enumerate(st.stages)]
    rows += [("", f"({note})") for note in st.notes]
    if run and st.run:
        rows.append(("hxex run", (f"{st.replacing}: " if st.replacing
                                  and st.run != "does every stage" else "") + st.run))
    return rows
