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

import numpy as np

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

    @property
    def up_to_date(self) -> bool:
        """Whether there is results and nothing for ``hxex run`` to do: every stage done or
        off."""
        return bool(self.stages) and all(s.state in (DONE, OFF) for s in self.stages)

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
        if rec["bootstrap"]["state"] == "done" and wanted["bootstrap"]:
            _bootstrap_causes(project, Path(bundle), causes)
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
            stages.append(Stage(s, NOT_DONE, ["stopped partway: hxex run continues it"]
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
    for stage, what in (("pick", "picking"), ("correction", "correction")):
        code = rec[stage]["code"]
        if rec[stage]["state"] == "off":
            continue
        if code is None:  # results made before the code was recorded: done again once
            causes[stage].append(f"made before hxex recorded its {what} code")
        elif code != now[stage]:
            causes[stage].append(f"another version of hxex's {what} code made it")


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
        if stage in ("fit", "noise") and path != "fit.k_max":
            facts["refit"] = True  # fits made with other settings: every spectrum fitted again
        if path.startswith("bootstrap.") and path not in ("bootstrap.ks", "bootstrap.set_k"):
            facts["rebootstrap"] = True
        causes[stage].append(f"{setting_name(path)}: {value_text(at(wanted, path))}, was "
                             f"{value_text(at(held, path))}")
    facts["settings_changed"] = changed


def _control_causes(project, bundle: Path, held: dict, given, causes, facts) -> None:
    """Causes from ``[run.controls]``: the roles reach the stages that use them
    (:data:`_CONTROL_STAGES`), the rest the correction. ``controls`` left out (automatic) is
    decided from the bundle's picked spectra, as ``hxex run`` decides it."""
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
    """Causes of the review: of the per-bin noise, undeuterated controls the review rejects
    now that the noise counted, or no longer rejects (owner, 2026-11-20: rejected controls are
    left out of it); of the fit, spectra whose K the project's review table chooses that the
    results did not fit (the screen skipped it)."""
    from .review_table import table_choices

    _rejected_controls_cause(project, bundle, causes)
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
                             "the results did not fit")



def _rejected_controls_cause(project, bundle: Path, causes) -> None:
    """A cause of the per-bin noise: the undeuterated controls the review rejects (the bundle's
    stored review, the project's review table, ``[[review.exclude]]``) differ from the ones the
    noise left out (its ``left_out``; none in results made before 2026-11-20)."""
    from .io.bundle import open_bundle
    from .review_table import rejected_spectra
    from .spectra import UNDEUT

    try:
        with open_bundle(bundle) as b:
            curve = b.fit().noise_curve
            if curve is None or not b.has("binned"):
                return
            binned = b.binned()
            stored = b.review() if b.has("review") else None
            source = b.spectra_source()
            rejected = rejected_spectra(project.review, binned, stored=stored, source=source)
    except Exception as exc:  # noqa: BLE001 - hxex check reports the table's problems itself
        log.debug("review rejections not worked out: %s", exc)
        return
    role = getattr(binned, "role", None)
    if role is None:
        return
    ids = np.asarray(binned.spectrum_id, dtype=str)
    now = set(ids[rejected & (np.asarray(role, dtype=str) == UNDEUT)])
    then = set(curve.left_out)
    added, back = len(now - then), len(then - now)
    if added or back:
        def controls(n: int) -> str:
            return f"{n} undeuterated control{'s' if n > 1 else ''}"

        parts = [f"{controls(added)} rejected in the review since it was measured"] if added \
            else []
        if back:
            parts.append(f"{back if added else controls(back)} no longer rejected")
        causes["noise"].append(" and ".join(parts))


def _bootstrap_causes(project, bundle: Path, causes) -> None:
    """Causes of the bootstrap: spectra whose K a review chooses without a bootstrap at it, and
    spectra of the population test's sets without one at their set's K (``[run.bootstrap]
    set_k``, with the project's ``[analysis]`` filters): the run adds both."""
    from .io.bundle import BundleError, open_bundle
    from .pipeline import _bundle_set_k_missing, _missing_choices, _review_choices
    from .review_table import merge_choices, table_choices

    try:
        chosen = table_choices(project.review)
    except Exception:  # noqa: BLE001 - hxex check reports the table's problems itself
        chosen = {}
    try:
        with open_bundle(bundle) as b:
            fit = b.fit()
            held = {k: b.bootstrap(k) for k in b.bootstrap_ks()}
            mine = merge_choices(_review_choices(b.review() if b.has("review") else None),
                                 chosen)
            wanted = _missing_choices(fit, held, mine) if mine else {}
            n = sum(len(r) for r in wanted.values())
            if n:
                causes["bootstrap"].append(f"{n} spectr{'a' if n > 1 else 'um'} shown at a K "
                                           "chosen in the review without a bootstrap there")
            if project.run.bootstrap is not None and _bundle_set_k_missing(
                    b, project.run.bootstrap, held, project.analysis):
                causes["bootstrap"].append("spectra of the population test's sets without a "
                                           "bootstrap at their set's K ([run.bootstrap] set_k)")
    except (BundleError, OSError, KeyError, ValueError) as exc:
        log.debug("bootstrap causes not worked out: %s", exc)

def run_plan(project, manifest: Mapping, standing_: Standing, facts: Mapping) -> str:
    """What ``hxex run`` does now to bring the results up to date, stage by stage, as it
    decides: the stored spectra are taken when the read is unchanged (HISTORY 397), the stored
    picking when the spectra are taken and the picking, its settings, and the control roles are
    unchanged, the fits of the spectra whose input is unchanged when the fitting code and the
    ``[run.fit]`` settings but ``k_max`` are (:mod:`hxex.reuse`; the per-bin noise must not
    move either), with the K missing fitted, the population test made again when its settings
    or the fits changed, and the bootstrap made again or completed; the noise and the
    correction are always worked out again."""
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
    changed = set(facts.get("settings_changed") or ())
    engine_cause = bool(facts.get("engine_differs"))
    stopped = {st.name for st in standing_.stages
               if st.state == NOT_DONE and "stopped partway" in " ".join(st.causes)}
    testing = standing_["population_test"].state != OFF
    if facts.get("refit") or engine_cause:
        why = ("other fitting code" if engine_cause else "[run.fit] changed")
        parts.append(f"fits every spectrum again ({why})")
    else:
        own = [c for c in standing_["fit"].causes
               if not c.startswith(("[run.", "after the"))]  # exclusions, a review K
        refit = (standing_["read"].state != DONE or standing_["pick"].state != DONE
                 or bool(own))
        parts.append("fits the spectra whose input changed, keeps the others' fits" if refit
                     else "keeps the fits")
        if standing_["noise"].state != OFF and standing_["noise"].state != DONE:
            from .reuse import CURVE_SHARE

            parts[-1] += (f" (if the per-bin noise's b1 moves by less than "
                          f"{100 * CURVE_SHARE:g} %)")
        if "fit" in changed:
            parts.append(f"fits the K missing (k_max {project.run.fit.k_max})")
        if testing and (standing_["population_test"].state != DONE or "fit" in changed):
            parts.append("makes the population test again")
        elif testing:
            parts.append("keeps the population test")
        if standing_["bootstrap"].state != OFF:
            if "bootstrap" in stopped:
                parts.append("makes the bootstrap (it stopped partway)")
            elif facts.get("rebootstrap") or standing_["bootstrap"].causes == [
                    "the results have none"]:
                parts.append("makes the bootstrap again")
            elif standing_["bootstrap"].causes == ["after the fit"]:
                parts.append("bootstraps the spectra whose fits or K changed, keeps the rest")
            elif standing_["bootstrap"].state != DONE:
                parts.append("adds what the bootstrap lacks and keeps the rest")
            else:
                parts.append("keeps the bootstrap")
    if standing_["correction"].state != OFF:
        parts.append("works out the correction and the quality flags again")
    return ", ".join(parts)


def lines(st: Standing, *, run: bool = True) -> list[tuple[str, str]]:
    """``(key, text)`` rows for ``hxex check`` and ``hxex info``: one per stage, the notes, and
    (``run``) what ``hxex run`` would do."""
    rows = [("" if i else "stages", s.text()) for i, s in enumerate(st.stages)]
    rows += [("", f"({note})") for note in st.notes]
    if run and st.run:
        rows.append(("hxex run", "nothing to do: the results are up to date"
                     if st.up_to_date else st.run))
    return rows
