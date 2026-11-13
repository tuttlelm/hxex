# SPDX-License-Identifier: GPL-3.0-or-later
"""The HDX data summary table that Masson et al. (2019, Nat. Methods 16:595, Table 1) ask every
HDX-MS study to report, one column per sample, as ``summary.csv`` and ``summary.md`` of
``hxex analyse``.

Masson's rows come first, in the paper's order and wording; *hxex* fills each from the analysis
(the peptides, times and replicates it kept, its coverage, its controls, the replicate scatter its
test uses and that test's threshold). What only the experimenter knows (buffer, pH, temperature,
quench, protease, LC/MS, what the replicates are) comes from ``[analysis.summary]``
(:attr:`hxex.analysis.AnalysisConfig.summary`): an item named like one of Masson's rows is added
to that row, any other item is listed first. *hxex*'s own rows follow Masson's: how the number of
populations was chosen, the peptide sets with two or more, the spectra the filters left out, the
differences found, and the software. Nothing is guessed: a row *hxex* cannot fill says so.

The cells hold values only; how each was made is said once, in numbered notes below the table
(the owner, 2026-11-13: the first version, an explanation in every cell, was a wall of text). The
item names carry the units and a note number (``Back-exchange, mean (IQR) ⁴``); in
``summary.csv`` the notes follow a blank row, one per row (``Note 4`` and its text), as
:func:`split_notes` reads them back.

The values are those of the analysis's other tables: the peptides and times are the labelled
time points kept after the filters (the controls are not time points), the repeatability is the
typical replicate standard deviation that ``delta_crit`` is made from
(:func:`hxex.analysis.compare._delta_crit`, robust: the median, not the plain pool), and the
back-exchange is that of the series whose fully deuterated control was measured (stand-ins left
out), from the controls the analysis uses.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

from ..analysis.residues import NoPeptideAtTime, coverage_table
from ..analysis.runs import RunSet, label, sample_setting
from ..analysis.species import _by_replicate, _records, _with_set_averages, analysis_controls
from ..io.times import format_time

log = logging.getLogger(__name__)

# Masson et al. 2019, Table 1, in its order (the protein states are the columns)
REACTION = "HDX reaction details"
TIME_COURSE = "HDX time course"
CONTROLS = "HDX controls"
BACK_EXCHANGE = "Back-exchange"
PEPTIDES = "Number of peptides"
COVERAGE = "Sequence coverage"
LENGTH = "Average peptide length / redundancy"
REPLICATES = "Replicates (biological or technical)"
REPEATABILITY = "Repeatability"
SIGNIFICANT = "Significant differences in HDX"
MASSON_ROWS = (REACTION, TIME_COURSE, CONTROLS, BACK_EXCHANGE, PEPTIDES, COVERAGE, LENGTH,
               REPLICATES, REPEATABILITY, SIGNIFICANT)
# a user's item named like one of these is added to that row (case and spacing ignored)
_ALIASES = {"replicates": REPLICATES, "average peptide length": LENGTH,
            "redundancy": LENGTH, "average peptide length/redundancy": LENGTH}

# hxex's own rows, after Masson's
POPULATION_RULE = "Number of populations chosen by"
TWO_OR_MORE = "Peptide sets with two or more populations"
LEFT_OUT = "Spectra left out by the analysis filters"
DIFFERENT = "Peptide time points different from the reference"
SOFTWARE = "Data analysis software"

# the item names shown: Masson's wording with the units of the short values
SHOWN = {
    BACK_EXCHANGE: "Back-exchange, mean (IQR; n)",
    PEPTIDES: "Number of peptides (charge states)",
    COVERAGE: "Sequence coverage, % of amides (residues)",
    REPLICATES: "Replicates (biological or technical), per peptide and time",
    REPEATABILITY: "Repeatability, median replicate sd",
    TWO_OR_MORE: "Peptide sets with two or more populations / judged",
    DIFFERENT: "Peptide time points different from the reference / tested",
}

_RULES = {
    "population_test": ("population test", "the population test of each peptide set (all "
                        "replicates and charge states of a peptide at one time; simulated null)"),
    "best_set": ("best set", "the information criterion pooled over each peptide set "
                 "(best_set_K)"),
    "bootstrap": ("bootstrap", "each spectrum's bootstrap (share and separation limits)"),
}
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_PLAIN = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
NOTE = "Note "  # the first cell of a note's row in summary.csv


def base_item(name: str) -> str:
    """The row an item name shown stands for (its note number and units taken off): a Masson
    row's name, one of *hxex*'s, or a user's item as written."""
    name = str(name).rstrip(" ⁰¹²³⁴⁵⁶⁷⁸⁹")
    for row, shown in SHOWN.items():
        if name == shown:
            return row
    return name


def _key(name: str) -> str:
    return " ".join(str(name).lower().replace("/", " / ").split()).replace(" / ", "/")


def _row_of(item: str) -> str | None:
    """The Masson row a user's item belongs to, or None for an item of its own."""
    key = _key(item)
    for row in MASSON_ROWS:
        if _key(row) == key:
            return row
    return _ALIASES.get(key)


def _user_value(value, state: tuple[str, str], shown: str) -> str:
    """A ``[analysis.summary]`` value for one sample: one text for every sample, or a table of
    sample (its name, or its display label) -> text; '' for a sample it does not name."""
    if isinstance(value, Mapping):
        for key in (shown, state[1]):
            if key in value and not isinstance(value[key], Mapping):
                return str(value[key])
        found = sample_setting(value, state[0], state[1])
        return "" if found is None or isinstance(found, Mapping) else str(found)
    return "" if value is None else str(value)


def _range(values: Sequence[int]) -> str:
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"{lo}"
    return f"{lo}-{hi} (median {float(np.median(values)):g})"


def _typical_sd(arrays) -> tuple[float, int]:
    """The typical replicate standard deviation of ``arrays`` as ``delta_crit`` takes it (each
    variance scaled by the median of its chi-square, then the median), and how many arrays had
    two replicates or more."""
    scaled = []
    for v in arrays:
        if len(v) >= 2:
            df = len(v) - 1
            scaled.append(float(np.var(v, ddof=1)) * df / stats.chi2.median(df))
    return (float(np.sqrt(np.median(scaled))) if scaled else np.nan), len(scaled)


def _count(n: int, noun: str) -> str:
    """``'1 time point'``, ``'4 time points'``."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _number(value: float, digits: int = 3) -> str:
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def _conditions_text(conditions) -> str:
    return (f"pH(read) {conditions.ph_read:g}, {conditions.temperature:g} °C, D2O fraction "
            f"{conditions.d2o_fraction:g}")


def _back_exchange(results, states: Sequence[tuple[str, str]], changes) -> dict:
    """Per state: the back-exchange (%) of each peptide series whose fully deuterated control
    was measured (its own replicates), from the controls the analysis uses, and the D2O
    fraction it was computed with."""
    runs = results.runs if isinstance(results, RunSet) else {"": results}
    out: dict[tuple[str, str], tuple[list[float], float]] = {s: ([], np.nan) for s in states}
    for name, r in runs.items():
        ctrl = analysis_controls(r, changes, quiet=True)
        if ctrl is None or r.binned is None:
            continue
        d2o = float(ctrl.config.d2o_fraction)
        b = r.binned
        where = {str(s): j for j, s in enumerate(b.spectrum_id)}
        first = {}
        for d, sid in zip(r.fit.data_id, r.fit.spectrum_id, strict=True):
            first.setdefault(int(d), str(sid))
        own = (np.asarray(ctrl.fd_measured) & (np.asarray(ctrl.n_fd) > 0)
               & ~np.array(["fd_other_charge" in str(f).split(";") for f in ctrl.flags]))
        for i, d in enumerate(ctrl.data_id):
            j = where.get(first.get(int(d), ""))
            if j is None or not own[i] or not np.isfinite(ctrl.retention[i]):
                continue
            sample = str(b.sample[j])
            state = results.state(name, sample) if isinstance(results, RunSet) else ("", sample)
            if state in out:
                values, _ = out[state]
                values.append(100.0 * (1.0 - float(ctrl.retention[i]) / d2o))
                out[state] = (values, d2o)
    return out


def dataset_summary(
    results,
    up,
    states: Sequence[tuple[str, str]],
    labels: Mapping[tuple[str, str], str],
    *,
    config,
    sequences=None,
    tables: Mapping[str, Mapping] | None = None,
    time_scale: str = "seconds",
    set_averages=None,
) -> dict[str, np.ndarray]:
    """The summary table's columns: ``item``, then one per state of ``states`` (the reference
    first) named by its display label in ``labels``. ``results`` are the peptide results the
    analysis read (a :class:`hxex.analysis.RunSet` or one run), ``up`` its species table,
    ``config`` its :class:`hxex.analysis.AnalysisConfig` (the populations rule decided),
    ``tables`` the tables computed so far (``compare_peptides``, ``left_out_spectra``) and
    ``set_averages`` the sets compared on their set-K average, as the comparison was made."""
    tables = tables or {}
    states = list(states)
    ks = (1, 2) if set_averages else (1,)
    groups, corrected, _ = _records(results, ks, config.exclude_flagged, config.filters,
                                    quiet=True, controls_changes=config.controls)
    groups = _with_set_averages(results, groups, set_averages, config.exclude_flagged,
                                config.filters, config.controls)
    per: dict[tuple[str, str], dict] = {s: {
        "peptides": {}, "charges": set(), "times": set(), "replicates": [], "fd": [],
        "d": [], "undeut": set(), "fulldeut": set(), "stand_in": set(), "amides": {}}
        for s in states}
    for key, recs in groups.items():
        run, sample, peptide, mod, start, end, point, k, r = key
        state = (run, sample)
        if state not in per or (k, r) != (1, 1):
            continue
        p = per[state]
        species = (peptide, mod, int(start), int(end))
        if point.role:
            p[point.role].add(species)
            continue
        p["peptides"][species] = None
        p["times"].add(float(point.time))
        p["charges"].update((*species, int(x[4])) for x in recs)
        p["amides"][species] = float(recs[0][5])
        if any(x[9] for x in recs):  # a stand-in control corrected it
            p["stand_in"].add(species)
        values, _ = _by_replicate(recs, 0)
        p["replicates"].append(len(values))
        p["fd"].append(values)
        p["d"].append(values * float(recs[0][5]))

    shown = [labels.get(s, label(*s)) for s in states]
    n_states = len(states)
    cells: dict[str, list[str]] = {}  # row -> one short value per state
    notes: dict[str, str] = {}  # row -> how its values were made (said once)

    reaction = []
    for state in states:
        cond = sample_setting(config.labelling, *state) if config.labelling else None
        reaction.append(_conditions_text(cond) if cond is not None else "")
    cells[REACTION] = reaction
    if config.labelling:
        notes[REACTION] = "pH(read), temperature, and D2O fraction from [analysis.labelling]."

    cells[TIME_COURSE] = [", ".join(format_time(t, time_scale) for t in sorted(per[s]["times"]))
                          or "none" for s in states]

    cells[CONTROLS] = []
    for state in states:
        p = per[state]
        n = len(p["peptides"])
        un = len(p["undeut"] & set(p["peptides"]))
        fd = len(p["fulldeut"] & set(p["peptides"]))
        text = f"UN {un}, FD {fd} of {n}"
        if not corrected.get(state, False):
            text += "; not corrected"
        elif p["stand_in"]:
            text += f"; {len(p['stand_in'])} stand-ins"
        cells[CONTROLS].append(text)
    notes[CONTROLS] = ("UN: undeuterated, FD: fully deuterated (maximally labelled) control: "
                       "how many of the peptides with labelled data have one. Stand-ins: "
                       "peptides corrected with a stand-in for a missing or rejected control; "
                       "not corrected: no back-exchange correction.")

    back = _back_exchange(results, states, config.controls)
    cells[BACK_EXCHANGE] = []
    fractions = set()
    for state in states:
        values, d2o = back[state]
        if not values:
            cells[BACK_EXCHANGE].append("not measured")
            continue
        fractions.add(f"{d2o:g}")
        q1, q3 = np.percentile(values, [25, 75])
        cells[BACK_EXCHANGE].append(f"{np.mean(values):.0f} % ({q1:.0f}-{q3:.0f} %; "
                                    f"{len(values)})")
    if fractions:
        notes[BACK_EXCHANGE] = (
            "1 - retention / D2O fraction of the labelling buffer ([run.controls] d2o_fraction: "
            f"{', '.join(sorted(fractions))}), over the n peptide charge states with their own "
            "fully deuterated control (stand-ins left out); mean and interquartile range.")

    cells[PEPTIDES] = [f"{len(per[s]['peptides'])} ({len(per[s]['charges'])})" for s in states]
    notes[PEPTIDES] = ("Peptides with labelled data (time points, not controls) after the "
                       "analysis filters, and their charge states.")

    coverage, unsequenced = _coverage(up, states, per, sequences)
    cells[COVERAGE] = [coverage[s][0] for s in states]
    notes[COVERAGE] = ("The share of the protein's amides (every residue but prolines and the "
                       "first) that those peptides cover.")
    if unsequenced:
        which = ("every sample" if len(unsequenced) == n_states
                 else ", ".join(shown[states.index(s)] for s in unsequenced))
        notes[COVERAGE] += (f" No protein sequence ([analysis] fasta) for {which}: the protein "
                            "is taken to end at the last covered residue, and residues no "
                            "peptide covers count as amides.")
    cells[LENGTH] = []
    for state in states:
        lengths = [end - start + 1 for _, _, start, end in per[state]["peptides"]]
        cells[LENGTH].append((f"{np.mean(lengths):.1f}" if lengths else "-")
                             + f" / {coverage[state][1]}")
    notes[LENGTH] = ("Mean peptide length in residues / mean number of peptides covering each "
                     "covered amide.")

    cells[REPLICATES] = [_range(per[s]["replicates"]) if per[s]["replicates"] else "none"
                         for s in states]
    notes[REPLICATES] = ("Fewest to most replicates (median) of a peptide at a time point; the "
                         "charge states of one replicate count once.")

    cells[REPEATABILITY] = []
    for state in states:
        sd_fd, n = _typical_sd(per[state]["fd"])
        sd_d, _ = _typical_sd(per[state]["d"])
        cells[REPEATABILITY].append(f"{_number(sd_fd)} fD ({_number(sd_d, 2)} D)" if n
                                    else "no replicates")
    notes[REPEATABILITY] = (
        "The median replicate standard deviation over the peptide time points with replicates, "
        "in fD and in deuterons: the scatter delta_crit is made from (each variance scaled to "
        "its degrees of freedom, then the median; Hageman and Weis 2019 pool them plainly).")

    cells[SIGNIFICANT] = _thresholds(states, per, tables)
    if any(c not in ("reference", "not tested") for c in cells[SIGNIFICANT]):
        notes[SIGNIFICANT] = (
            f"|difference| > delta_crit and Welch's t-test p < {config.alpha:g}, per peptide "
            "and time point, one population (the hybrid rule of Hageman and Weis 2019, with a "
            "robust replicate standard deviation); in brackets delta_crit in deuterons at the "
            "sample's median peptide.")
    elif not _k1_rows(tables.get("compare_peptides")):
        notes[SIGNIFICANT] = "Not tested: no reference sample ([analysis] reference)."

    # the user's items: added to their Masson row, or listed first
    own_items = []
    extra: dict[str, list[str]] = {}
    for item, value in (config.summary or {}).items():
        values = [_user_value(value, s, shown[i]) for i, s in enumerate(states)]
        row = _row_of(item)
        if row is None:
            own_items.append((str(item), values))
        else:
            extra[row] = [("; ".join(x for x in (old, new) if x)) for old, new in
                          zip(extra.get(row, [""] * n_states), values, strict=True)]
    rows: list[tuple[str, list[str]]] = list(own_items)
    for row in MASSON_ROWS:
        users = extra.get(row, [""] * n_states)
        values = ["; ".join(x for x in (b, u) if x) for b, u in zip(cells[row], users,
                                                                     strict=True)]
        if row == REACTION and not all(values):
            values = [v or "not given" for v in values]
            notes[REACTION] = ((notes.get(REACTION, "") + " ").lstrip()
                               + "Not given: write it in [analysis.summary] (or set "
                                 "[analysis.labelling]).")
        rows.append((row, values))

    # hxex's own rows
    rule = config.populations or "best_set"
    short, long = _RULES.get(rule, (rule, rule))
    rows.append((POPULATION_RULE, [short] * n_states))
    notes[POPULATION_RULE] = f"{long[0].upper()}{long[1:]}."
    rows.append((TWO_OR_MORE, _two_or_more(up, states, rule)))
    notes[TWO_OR_MORE] = ("Peptide time points (all replicates and charge states of a peptide "
                          "at one time) with two or more populations, of those judged.")
    rows.append((LEFT_OUT, _left_out(tables.get("left_out_spectra"), states)))
    notes[LEFT_OUT] = "Whole spectra the analysis filters left out (left_out_spectra.csv)."
    rows.append((DIFFERENT, _different(tables.get("compare_peptides"), states)))
    notes[DIFFERENT] = ("Peptide time points significantly different from the reference at "
                        "one population, of those tested (compare_peptides.csv).")
    from .. import version_text

    rows.append((SOFTWARE, [version_text()] * n_states))

    # the notes, numbered in the order of their rows
    columns: dict[str, list[str]] = {"item": [], **{name: [] for name in shown}}
    numbered: list[str] = []
    for row, values in rows:
        name = SHOWN.get(row, row)
        if row in notes:
            numbered.append(notes[row])
            name += " " + str(len(numbered)).translate(_SUPERSCRIPT)
        columns["item"].append(name)
        for col, value in zip(shown, values, strict=True):
            columns[col].append(value)
    if numbered:
        columns["item"].append("")
        for col in shown:
            columns[col].append("")
        for i, text in enumerate(numbered, 1):
            columns["item"].append(f"{NOTE}{i}")
            for j, col in enumerate(shown):
                columns[col].append(text if j == 0 else "")
    return {name: np.array(values, dtype=object) for name, values in columns.items()}


def note_number(item: str) -> int | None:
    """The note an item name shown points to (its trailing superscript number), or None."""
    digits = str(item).rstrip()[len(str(item).rstrip().rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹")):]
    return int(digits.translate(_PLAIN)) if digits else None


def split_notes(columns: Mapping[str, np.ndarray]) -> tuple[dict[str, list[str]], list[str]]:
    """A summary's columns split at the blank row: the table's rows (name -> values) and the
    notes' texts, in their order."""
    names = list(columns)
    items = [str(x) for x in columns["item"]]
    end = items.index("") if "" in items else len(items)
    table = {n: [str(x) for x in columns[n][:end]] for n in names}
    first = names[1] if len(names) > 1 else "item"
    notes = [str(columns[first][i]) for i in range(end + 1, len(items))
             if items[i].startswith(NOTE)]
    return table, notes


def _coverage(up, states, per, sequences):
    """Per state: the coverage and the redundancy values, from the peptides of the labelled
    time points (the amides: no proline, not the protein's first residues); and the states
    without a protein sequence."""
    times = sorted({t for s in states for t in per[s]["times"]})
    out = {s: ("-", "-") for s in states}
    unsequenced = [s for s in states
                   if sequences is None or sample_setting(sequences, *s) is None]
    if not times:
        return out, unsequenced
    try:
        cov = coverage_table(up, sequences=sequences, count="peptides", time=times)
    except NoPeptideAtTime:
        return out, unsequenced
    for state in states:
        rows = (cov.run.astype(str) == state[0]) & (cov.sample.astype(str) == state[1])
        amide = rows & ~np.asarray(cov.no_amide, dtype=bool)
        if not amide.any():
            continue
        covered = amide & np.asarray(cov.covered, dtype=bool)
        residues = cov.residue[rows]
        redundancy = cov.redundancy[covered]
        out[state] = (f"{100.0 * covered.sum() / amide.sum():.1f} % "
                      f"({int(residues.min())}-{int(residues.max())})",
                      f"{np.mean(redundancy):.1f}" if len(redundancy) else "-")
    return out, unsequenced


def _thresholds(states, per, tables) -> list[str]:
    """The comparison's threshold per sample: ``delta_crit`` in fD and, at the sample's median
    peptide, in deuterons."""
    rows = _k1_rows(tables.get("compare_peptides"))
    if rows is None:
        return ["not tested"] * len(states)
    cmp, keep = rows
    crit = np.asarray(cmp["delta_crit"][keep], dtype=float)
    crit = crit[np.isfinite(crit)]
    out = []
    for i, state in enumerate(states):
        if i == 0:
            out.append("reference")
        elif not len(crit):
            out.append("not tested")
        else:
            amides = np.array(list(per[state]["amides"].values()), dtype=float)
            in_d = f" ({crit[0] * np.median(amides):.2f} D)" if len(amides) else ""
            out.append(f"> {crit[0]:.3f} fD{in_d}")
    return out


def _k1_rows(cmp):
    if cmp is None or "comparison" not in cmp or not len(cmp["comparison"]):
        return None
    keep = np.asarray(cmp["comparison"]) == "K=1 fD"
    return (cmp, keep) if keep.any() else None


def _state_rows(table, state) -> np.ndarray:
    run = np.asarray(table.get("run", np.full(len(table["sample"]), "")), dtype=str)
    return (run == state[0]) & (np.asarray(table["sample"], dtype=str) == state[1])


def _different(cmp, states) -> list[str]:
    rows = _k1_rows(cmp)
    if rows is None:
        return ["not tested"] * len(states)
    cmp, keep = rows
    out = []
    for i, state in enumerate(states):
        if i == 0:
            out.append("reference")
            continue
        mine = keep & _state_rows(cmp, state)
        tested = mine & np.isfinite(np.asarray(cmp["p"], dtype=float))
        sig = mine & np.asarray(cmp["significant"], dtype=bool)
        out.append(f"{int(sig.sum())} / {int(tested.sum())}")
    return out


def _two_or_more(up, states, rule) -> list[str]:
    run, sample = up.run.astype(str), up.sample.astype(str)
    one = (up.K == 1) & (up.rank == 1) & (np.asarray(up.role, dtype=str) == "")
    count = (np.asarray(up.set_test_K) if rule == "population_test"
             else np.asarray(up.best_set_K))
    out = []
    for state in states:
        judged = one & (run == state[0]) & (sample == state[1]) & (count > 0)
        out.append(f"{int((judged & (count >= 2)).sum())} / {int(judged.sum())}"
                   if judged.any() else "not judged")
    return out


def _left_out(table, states) -> list[str]:
    if table is None or not len(table.get("sample", ())):
        return ["0"] * len(states)
    return [str(int(_state_rows(table, state).sum())) for state in states]


def summary_markdown(columns: Mapping[str, np.ndarray]) -> str:
    """The summary table as a Markdown table with its notes below, to paste into a manuscript
    or its supplement."""
    table, notes = split_notes(columns)
    names = list(table)

    def cell(text) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = ["# HDX data summary", "",
             "As recommended by Masson et al. (2019), Nat. Methods 16, 595-602, Table 1.", "",
             "| " + " | ".join(cell("" if n == "item" else n) for n in names) + " |",
             "|" + "---|" * len(names)]
    for i in range(len(table["item"])):
        lines.append("| " + " | ".join(cell(table[n][i]) for n in names) + " |")
    if notes:
        lines.append("")
        lines += [f"{i}. {text}" for i, text in enumerate(notes, 1)]  # not in the table
    return "\n".join(lines) + "\n"
