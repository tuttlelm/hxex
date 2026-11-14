# SPDX-License-Identifier: GPL-3.0-or-later
"""What every flag, control note, population-test flag, spectrum quality screen, and left-out
reason means, in one list: the dictionary behind ``hxex explain``, the guide's ``flags.md``
(written by :mod:`hxex._reference`), the ``flags.csv`` beside the tables and in each analysis
folder, and the window's tooltips.

Each :class:`Entry` names a word as *hxex* writes it, where it appears, what it means, what
*hxex* does about it, and what to look at. Thresholds are written as placeholders
(``{run.quality.min_intensity}``, a setting's path) and filled in by :func:`render` from the
defaults, a project's settings, or the settings results were made with, so the text cannot
fall behind the code; ``{screen:NAME:ACTION}`` gives a spectrum quality screen's thresholds
at each level. A test ties the list to the names the code writes (:func:`produced`), both
ways: a new flag cannot ship without a meaning, nor an entry outlive its flag.

The long evidence behind each check stays in the docstrings of the modules that make them
(:mod:`hxex.quality`, :mod:`hxex.controls`, :mod:`hxex.analysis.set_test`,
:mod:`hxex.picking.screen`, :mod:`hxex.analysis.species`).
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

# the families, in the order the guide and `hxex explain` list them: title, one of them, words
FAMILIES = {
    "quality": (
        "Quality flags", "quality flag",
        "Marks on single spectra (and on every spectrum of a series, for the checks of a whole "
        "series), in the `flags` column of `spectra.csv` and the overview. A flag alone leaves "
        "nothing out; the analysis acts on `off_mz` and `undeut_high` (see each)."),
    "controls": (
        "Control flags", "control flag",
        "Why a series' undeuterated or fully deuterated value is what it is, in `series.csv`: "
        "the `controls` column names the reasons when a stand-in was used (`control_fallback` "
        "true), `control_notes` the warnings that leave the control in use."),
    "population": (
        "Population test flags", "population test flag",
        "Signs that an extra population may be another species, or that more populations may "
        "be there, from the population test (`[run.population_test]`). Flags only: no number "
        "of populations changes."),
    "screen": (
        "Spectrum quality screens", "spectrum quality score",
        "Scores from 0 to 1, in the last columns of `spectra.csv`: the share of the intensity "
        "that looks like another species. `hxex analyse` acts on them by `[analysis.filters] "
        "spectrum_quality_filtering` (`loose`, the default, `strict`, or `none`)."),
    "left_out": (
        "Reasons a spectrum is left out", "left-out reason",
        "The `reason` column of `left_out_spectra.csv` in an analysis folder, and the window's "
        "\"left out\" column. Marking a spectrum `accepted` in the user review table keeps it "
        "against every one of these."),
}


@dataclass(frozen=True)
class Entry:
    """One word *hxex* writes: ``name`` (exactly as written), its ``family`` (a key of
    :data:`FAMILIES`), its ``level`` (spectrum, series, peptide and time), ``where`` it
    appears, a one-sentence ``short`` meaning (tooltips, ``flags.csv``), more of the
    ``meaning`` when needed, the ``action`` *hxex* takes, what to ``check``, and the ``guide``
    page that explains it (``page.md#anchor``). Text is Markdown with placeholders
    (:func:`render`)."""

    name: str
    family: str
    level: str
    where: str
    short: str
    action: str
    meaning: str = ""
    check: str = ""
    guide: str = ""

    @property
    def settings(self) -> tuple[str, ...]:
        """The settings its text names, as the guide writes them (``[run.quality]
        min_intensity``), in the order they appear."""
        out: list[str] = []
        for field in (self.short, self.meaning, self.action, self.check):
            for path in re.findall(r"\{([a-z0-9_.]+)\}", field):
                table, _, key = path.rpartition(".")
                name = f"[{table}] {key}"
                if name not in out:
                    out.append(name)
        return tuple(out)


_ENTRIES: list[Entry] = []


def _add(name, family, level, where, short, action, meaning="", check="", guide=""):
    _ENTRIES.append(Entry(name, family, level, where, short, action, meaning, check, guide))


SPECTRA_FLAGS = "`spectra.csv` `flags`"
SERIES_CONTROLS = "`series.csv` `controls`"
SERIES_NOTES = "`series.csv` `control_notes`"
FLAG_ONLY = "Flag only: nothing is left out or changed."
QUALITY_PAGE = "outputs.md#quality-flags"
CONTROLS_PAGE = "back-exchange-correction.md#how-hxex-judges-the-controls"
ORDER_PAGE = "back-exchange-correction.md#the-order-of-the-values"
STAND_IN = ("A stand-in (`control_fallback`): `hxex analyse` leaves the series out with "
            "`{analysis.filters.stand_in_controls}`")

# ---- quality flags (hxex.quality.FLAG_NAMES) ----------------------------------------------

_add("poor_fit", "quality", "spectrum", SPECTRA_FLAGS,
     "The fit is much worse than the other fits of its series.",
     FLAG_ONLY,
     "The misfit relative to the spectrum's own intensity is more than "
     "{run.quality.poor_fit_factor} times its series' median, and the deviance is above "
     "{run.quality.min_deviance}. Judged only in series with at least "
     "{run.quality.min_series_fits} fitted spectra.",
     "Look at the spectrum and its fit: another species, a spike, or a peak the picking cut.",
     QUALITY_PAGE)
_add("low_intensity", "quality", "spectrum", SPECTRA_FLAGS,
     "Far less signal than the other replicates at the same time point.",
     FLAG_ONLY,
     "The total intensity is below {run.quality.min_intensity} of the other replicates' mean, "
     "once each injection's overall brightness is divided out (an injection that is dim "
     "throughout is named in a warning and in `hxex info`, not flagged spectrum by spectrum). "
     "Spectra without replicates are not judged.",
     "A weak spectrum's fit is noisier: compare it with its replicates.",
     QUALITY_PAGE)
_add("noise_outlier", "quality", "spectrum", SPECTRA_FLAGS,
     "A replicate the replicate-based noise estimate left out.",
     "A control replicate is also left out of its control's average (the series then has "
     "`m0_replicate_left_out` or `fd_replicate_left_out`).",
     check="Compare it with its replicates.",
     guide=QUALITY_PAGE)
_add("mz_shift", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The series' peaks sit off their predicted m/z.",
     FLAG_ONLY,
     "The series' peaks are, as a rule, more than {run.quality.max_mz_offset_ppm} ppm from where "
     "the composition puts them, compared with the run as a whole. Judged only for series "
     "`off_mz` does not check: fragment ions, and series without an undeuterated control.",
     "A wrong sequence, modification, terminus, or ion type, or another species.",
     QUALITY_PAGE)
_add("off_mz", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The series' undeuterated peaks sit further from the run's calibration than its other "
     "series: most likely another species.",
     "`hxex analyse` leaves the series out (`{analysis.filters.off_mz}`; reason `off its m/z`), "
     "controls included.",
     "The limit is {run.picker.off_mz_sd} times the run's scatter. Often the retention time "
     "points at another peptide of almost the same mass.",
     "Check the retention time in the peptide table (`find_retention_time` can suggest one).",
     "picking-window.md#a-series-whose-peaks-sit-off-its-mz")
_add("rt_found", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The series was read at the retention time found for its exact m/z, not the peptide "
     "table's.",
     FLAG_ONLY,
     "Set by `find_retention_time = \"automatic\"` in `[input.scans]`.",
     "Check the time in `retention_suggestions.csv` and correct the peptide table.",
     "inputs.md#when-the-peptide-tables-time-is-wrong")
_add("drift_unclear", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The series was read from ion mobility files without a clear drift-time range.",
     FLAG_ONLY,
     "No undeuterated control had signal (every drift time was summed), or the undeuterated "
     "drift profile has a second peak the isotope spacing cannot tell apart, or the tallest "
     "drift peak sat off the peptide's m/z while another sat on it (that one was used; the log "
     "names both).",
     "Look at its spectra, or give the peptide its own `drift_start` and `drift_end`.",
     "inputs.md#ion-mobility-each-peptides-drift-time")
_add("elution_cut", "quality", "spectrum", SPECTRA_FLAGS,
     "A fragment spectrum whose elution peak reaches the first or last scan recorded for its "
     "precursor.",
     FLAG_ONLY,
     "The acquisition missed part of the peak (`find_fragment_peak`), so the spectrum is "
     "dimmer than it should be.",
     "Retention moved in that run: check the scheduled window.",
     "inputs.md#finding-each-fragment-channels-peak")
_add("precursor_off", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The precursor read in the fragment scans differs from its parent's MS1 spectrum by more "
     "than {run.quality.max_precursor_offset} D.",
     FLAG_ONLY,
     "Most likely another species in the isolation window shares its m/z.",
     "Treat the series' fragments with care.",
     "experiments.md#fragments")
_add("overlapping_ion", "quality", "spectrum", SPECTRA_FLAGS,
     "Another ion of the same fragment scans puts at least {run.quality.max_ion_overlap} of this "
     "fragment's picked intensity on its bins.",
     FLAG_ONLY,
     "Their peaks are too close for the instrument to separate, so the fragment's envelope is "
     "partly the other ion's. `overlapping_ion_with` in `spectra.csv` names it and its share. "
     "Ions that are not read (a, b, and y ions), and an ion on every one of the fragment's "
     "strong bins (it cannot be told apart), are not counted.",
     "Prefer another ion type or charge for the same residues.",
     "experiments.md#fragments")
_add("below_noise", "quality", "spectrum", SPECTRA_FLAGS,
     "A fragment spectrum fitted through the per-scan cut whose tallest peak is below "
     "{run.quality.min_signal_to_noise} times its scan's recorded noise.",
     FLAG_ONLY,
     "So many of its peaks were deleted scan by scan that even the fit through the cut cannot "
     "recover its deuteration. `signal_to_noise` in `spectra.csv` gives the size.",
     "Do not rely on its value.",
     "inputs.md#fitting-fragment-ions-through-the-cut")
_add("poor_envelope", "quality", "series, on each of its spectra", SPECTRA_FLAGS,
     "The fragment's ion-type envelope could not be measured well.",
     FLAG_ONLY,
     "Left with one ion type's theory for want of usable reference spectra, or the measured "
     "envelope does not describe the spectra: the deuteration can be off by up to a hydrogen.",
     guide="experiments.md#fragments")
_add("saturated", "quality", "spectrum", SPECTRA_FLAGS,
     "An undeuterated control whose peaks beside the tallest read higher than natural "
     "abundance predicts: the tallest peak reads low, the usual sign of detector saturation.",
     FLAG_ONLY,
     "Above {run.quality.saturated_ratio} times the prediction, in most of the series' control "
     "replicates; only the brighter half of the run's undeuterated controls is judged.",
     "Check the series' corrected uptake against the peptide's other charge states.",
     "outputs.md#very-bright-spectra-saturated-tails_low-and-bright")
_add("tails_low", "quality", "spectrum", SPECTRA_FLAGS,
     "An undeuterated control whose small isotope peaks read lower than natural abundance "
     "predicts.",
     FLAG_ONLY,
     "Below {run.quality.tails_low_ratio} times the prediction; seen in bright spectra, cause "
     "unknown.",
     guide="outputs.md#very-bright-spectra-saturated-tails_low-and-bright")
_add("bright", "quality", "spectrum", SPECTRA_FLAGS,
     "A labelled spectrum or fully deuterated control at least as bright as a control of its "
     "series flagged `saturated` or `tails_low`.",
     FLAG_ONLY,
     "Its own shape cannot be checked, so it is marked as likely distorted in the same way.",
     "When two populations appear only in the brightest replicate, look at the dimmer ones.",
     "outputs.md#very-bright-spectra-saturated-tails_low-and-bright")
_add("undeut_high", "quality", "spectrum", SPECTRA_FLAGS,
     "An undeuterated spectrum whose mean is above {run.controls.m0_max} deuterons: another "
     "species or a wrong pick.",
     "Left out of the control's mean when the series has a good undeuterated spectrum (the "
     "series is noted `m0_replicate_left_out`), and then out of the analysis's undeuterated "
     "values; when every one is above, the series is `m0_high`.",
     "An undeuterated peptide sits near 0. Judged spectrum by spectrum, so without replicates "
     "too.",
     "Look at the spectrum; mark it `accepted` to keep it.",
     CONTROLS_PAGE)

# ---- control flags (hxex.controls.CONTROL_FLAGS) ------------------------------------------

_add("uncorrected", "controls", "series", SERIES_CONTROLS,
     "A series the controls leave as measured, such as a fragment ion.",
     "No back-exchange correction; its values are in deuterons as measured.",
     guide=ORDER_PAGE)
_add("m0_missing", "controls", "series", SERIES_CONTROLS,
     "The series has no undeuterated control; its undeuterated value is taken as 0 (natural "
     "abundance).",
     "A stand-in (`control_fallback`), but a fair one: the analysis keeps the series.",
     guide=ORDER_PAGE)
_add("m0_high", "controls", "series", SERIES_CONTROLS,
     "The undeuterated control's mean is above {run.controls.m0_max} deuterons, so it is "
     "rejected and 0 is used.",
     STAND_IN + " (reason `stand-in control (undeuterated)`).",
     "Most often another species of almost the same m/z sits in the picking windows, usually in "
     "the labelled spectra too.",
     "Look at the undeuterated spectra; mark them `accepted` to keep the series.",
     CONTROLS_PAGE)
_add("m0_unreplicated", "controls", "series", SERIES_CONTROLS,
     "The undeuterated control has one replicate, so its value has no uncertainty.",
     "None: the control is used.",
     guide=CONTROLS_PAGE)
_add("fd_unreplicated", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control has one replicate, so its value has no uncertainty.",
     "None: the control is used.",
     guide=CONTROLS_PAGE)
_add("no_amides", "controls", "series", SERIES_CONTROLS,
     "The peptide has no exchangeable amides, so no fully deuterated value can scale it.",
     "A stand-in is used.",
     guide=ORDER_PAGE)
_add("fd_missing", "controls", "series", SERIES_CONTROLS,
     "The series has no fully deuterated control.",
     "It takes the next value in the order (another charge state, another sample, the map, or "
     "the run-wide value).",
     guide=ORDER_PAGE)
_add("retention_low", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control's retention is 0 or less, or below "
     "{run.controls.retention_min}: it is rejected.",
     "It takes the next value in the order.",
     check="Look at the control's spectra.",
     guide=CONTROLS_PAGE)
_add("retention_high", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control's retention is above {run.controls.retention_max}: it "
     "cannot be right, so it is rejected.",
     "It takes the next value in the order.",
     check="Look at the control's spectra: another species, or a wrong pick.",
     guide=CONTROLS_PAGE)
_add("fd_below_data", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control is below the highest time point of its series: the sample "
     "did not fully deuterate, so it is rejected.",
     "It takes the next value in the order.",
     "A little below is allowed ({run.controls.time_course_tolerance} D per amide); each time "
     "point counts by the median of its replicates.",
     guide=CONTROLS_PAGE)
_add("retention_outlier", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control's retention is far below the run's other controls, so it is "
     "rejected.",
     "It takes the next value in the order.",
     "More than {run.controls.outlier_sd} robust standard deviations below their median; made "
     "only with at least ten accepted controls and no `retention_min`.",
     guide=CONTROLS_PAGE)
_add("fd_off_model", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control misses the retention predicted from the peptide's sequence "
     "by more than {run.controls.kint_tolerance} per amide, so it is rejected.",
     "It takes the next value in the order.",
     "Only with the quench conditions given (`quench` in `[run.controls]`).",
     guide=CONTROLS_PAGE)
_add("fd_charges_combined", "controls", "series", SERIES_CONTROLS,
     "The series' fully deuterated value is the mean retention of the peptide's accepted "
     "controls over its charge states.",
     "Counts as measured: not a stand-in.",
     "Set by `{run.controls.combine_charges}`: retention belongs to the peptide, not its charge.",
     guide=ORDER_PAGE)
_add("fd_other_charge", "controls", "series", SERIES_CONTROLS,
     "The series has no good fully deuterated control of its own and takes the peptide's "
     "controls at its other charge states.",
     "Counts as measured when charge states are combined (the default); a stand-in with "
     "`combine_charges = false`.",
     guide=ORDER_PAGE)
_add("fd_other_sample", "controls", "series", SERIES_CONTROLS,
     "The series takes the same peptide's fully deuterated controls in the run's other samples.",
     STAND_IN + " (reason `stand-in control`); `other_sample` keeps these.",
     "Close to the peptide's own: the samples of one run share the peptide and the LC.",
     guide=ORDER_PAGE)
_add("fd_map", "controls", "series", SERIES_CONTROLS,
     "The series takes a retention map of the overlapping peptides with accepted controls.",
     STAND_IN + " (reason `stand-in control`).",
     guide=ORDER_PAGE)
_add("fd_median", "controls", "series", SERIES_CONTROLS,
     "The series takes the median retention of the run's accepted controls.",
     STAND_IN + " (reason `stand-in control`).",
     guide=ORDER_PAGE)
_add("fd_kint", "controls", "series", SERIES_CONTROLS,
     "The series takes the retention predicted from its sequence, fitted to the run's accepted "
     "controls.",
     STAND_IN + " (reason `stand-in control`).",
     "Only with the quench conditions given (`quench` in `[run.controls]`).",
     guide=ORDER_PAGE)
_add("fd_disagrees", "controls", "series", SERIES_NOTES,
     "The same peptide's fully deuterated controls in the other samples have a clearly "
     "different retention.",
     "A warning: the control is still used.",
     "By more than {run.controls.sample_tolerance}; in one run they should agree closely.",
     "Look at the control's spectra in each sample.",
     CONTROLS_PAGE)
_add("fd_multimodal", "controls", "series", SERIES_NOTES,
     "Most of the fully deuterated control's replicates were fitted best with more than one "
     "population.",
     "A warning: the control's average is still used.",
     "`series.csv` adds whether its retention agrees with the other samples' (`retention "
     "agrees`, `retention differs`), with its other charge states', or was not checked "
     "(`retention unchecked`). Differing can mean an incompletely deuterated sample.",
     "With `retention unchecked` or `retention differs`, look at its spectra.",
     CONTROLS_PAGE)
_add("fd_retention_agrees", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control's retention agrees with the same peptide's in the other "
     "samples.",
     "None: written as `fd_multimodal (retention agrees)` in `control_notes`.",
     guide=CONTROLS_PAGE)
_add("fd_charges_agree", "controls", "series", SERIES_CONTROLS,
     "The fully deuterated control's retention agrees with the peptide's other charge states.",
     "None: written as `fd_multimodal (retention agrees with its other charge states)` in "
     "`control_notes`.",
     guide=CONTROLS_PAGE)
_add("m0_multimodal", "controls", "series", SERIES_NOTES,
     "Most of the undeuterated control's replicates were fitted best with more than one "
     "population.",
     "A warning: the control is still used.",
     "An undeuterated sample has one population, so another species may be there.",
     "Look at the undeuterated spectra.",
     CONTROLS_PAGE)
_add("data_below_m0", "controls", "series", SERIES_NOTES,
     "A time point of the series reads below its undeuterated control.",
     "A warning: the control is still used.",
     "By more than {run.controls.time_course_tolerance} D per amide. The early time points are "
     "the hardest to measure, so it is the time point that is suspect.",
     "Look at that time point's spectra.",
     CONTROLS_PAGE)
_add("fd_charges_disagree", "controls", "series", SERIES_NOTES,
     "The peptide's charge states, whose fully deuterated controls are combined, differ in "
     "retention by more than {run.controls.sample_tolerance}.",
     "A warning: the combined value is still used.",
     guide=ORDER_PAGE)
_add("m0_replicate_left_out", "controls", "series", SERIES_NOTES,
     "An undeuterated control replicate was left out of the average.",
     "The control is the mean of the others.",
     "Above `m0_max` beside a good one (`undeut_high`), above the amide count, far from the "
     "other replicates ({run.controls.replicate_outlier} typical standard deviations), or a "
     "noise outlier.",
     guide=ORDER_PAGE)
_add("fd_replicate_left_out", "controls", "series", SERIES_NOTES,
     "A fully deuterated control replicate was left out of the average.",
     "The control is the mean of the others.",
     "Above the amide count, far from the other replicates ({run.controls.replicate_outlier} "
     "typical standard deviations), or a noise outlier.",
     guide=ORDER_PAGE)
_add("fd_deamidated", "controls", "series", SERIES_NOTES,
     "The fully deuterated control was lowered by its sample's deamidated share.",
     "The lowered value is the series' own, measured control.",
     "Only with `{run.controls.deamidation}`, for a peptide holding an Asn-Gly site; "
     "`deamidated_share` in `series.csv` gives the share.",
     guide="back-exchange-correction.md#deamidated-fully-deuterated-controls")

# ---- population test flags --------------------------------------------------------------

_add("narrow_component", "population", "spectrum", "`spectra.csv` `narrow_component`",
     "The fit at the spectrum's number of populations has a population narrower than deuterium "
     "allows.",
     FLAG_ONLY,
     "The narrowest width is set by the peptide's amides and the D2O fraction "
     "({run.controls.d2o_fraction}). Much narrower may be another species, or a spike.",
     "Look at the spectrum.",
     "population-test.md#two-flags-for-other-species")
_add("undeuterated_two", "population", "peptide, on every row in every sample",
     "`species_uptake.csv` `undeuterated_two`",
     "The peptide's undeuterated spectra read two populations: something else is there.",
     "Flag only, unless `{analysis.filters.undeuterated_cap}`: the peptide's labelled spectra "
     "then stay at one population.",
     "Whether the labelled spectra share the other species is not known.",
     "Look at the undeuterated spectra.",
     "population-test.md#two-flags-for-other-species")
_add("more_populations_possible", "population", "spectrum, and peptide and time",
     "`spectra.csv` and `species_uptake.csv` `more_populations_possible`",
     "The test's quick step above `decide_up_to` would add a population.",
     FLAG_ONLY,
     "Set with `{run.population_test.decide_up_to}`; a run that decides more populations tells.",
     guide="population-test.md#turning-it-on")

# ---- spectrum quality screens (hxex.picking.screen.SCREEN_SCORES) ------------------------

SCREENS_PAGE = "outputs.md#spectrum-quality-screens"
_add("screen_below_mono", "screen", "spectrum", "`spectra.csv` `screen_below_mono`",
     "Another species starting below the peptide's lightest peak, whose isotope peaks carry on "
     "into the peptide's.",
     "`hxex analyse` leaves the spectrum out from {screen:screen_below_mono:remove}.",
     "The share of the peptide's peaks that this carry-on explains. Empty when the spectrum does "
     "not reach far enough below the peptide.",
     guide=SCREENS_PAGE)
_add("screen_off_position", "screen", "spectrum", "`spectra.csv` `screen_off_position`",
     "Intensity of the five lowest peaks found away from where the composition puts them.",
     "`hxex analyse` leaves the spectrum out from {screen:screen_off_position:remove}.",
     "An overlapping species of slightly different mass does this.",
     guide=SCREENS_PAGE)
_add("screen_between", "screen", "spectrum", "`spectra.csv` `screen_between`",
     "Peaks between the isotope positions, inside or touching the peptide's envelope.",
     "`hxex analyse` leaves the spectrum out from {screen:screen_between:remove}.",
     "Another species, often of another charge.",
     guide=SCREENS_PAGE)
_add("screen_between_separate", "screen", "spectrum", "`spectra.csv` `screen_between_separate`",
     "Peaks between the isotope positions in a cluster of their own above the envelope.",
     "`hxex analyse` keeps the spectrum at one population from "
     "{screen:screen_between_separate:one_population}.",
     "The fit could take the cluster for a second, more deuterated population.",
     guide=SCREENS_PAGE)
_add("screen_beyond_reach", "screen", "spectrum", "`spectra.csv` `screen_beyond_reach`",
     "Intensity beyond the heaviest the peptide can be.",
     "`hxex analyse` warns from {screen:screen_beyond_reach:warn} and leaves the spectrum out "
     "from {screen:screen_beyond_reach:remove}.",
     "Those peaks belong to something else; when most of the intensity lies there, the spectrum "
     "is mostly another species.",
     guide=SCREENS_PAGE)

# ---- left-out reasons (hxex.analysis.species.LEFT_OUT_REASONS) ---------------------------

LEFT_OUT = "`left_out_spectra.csv` `reason`"
_add("off its m/z", "left_out", "series", LEFT_OUT,
     "The series is flagged `off_mz`: most likely another species.",
     "Left out by `{analysis.filters.off_mz}`, controls included.",
     guide="picking-window.md#a-series-whose-peaks-sit-off-its-mz")
for _score in ("screen_below_mono", "screen_off_position", "screen_between",
               "screen_beyond_reach"):
    _add(f"spectrum quality ({_score})", "left_out", "spectrum", LEFT_OUT,
         f"The spectrum's `{_score}` score reached the level's threshold.",
         f"Left out by `{{analysis.filters.spectrum_quality_filtering}}` (from "
         f"{{screen:{_score}:remove}}); a control left out is also left out of the correction.",
         guide=SCREENS_PAGE)
_add("kept at K = 1 (spectrum quality)", "left_out", "spectrum", "the window's \"left out\" column",
     "The spectrum's `screen_between_separate` score reached the level's threshold: it is kept "
     "with one population only.",
     "Not left out, and not counted as left out.",
     guide=SCREENS_PAGE)
_add("undeuterated control above m0_max", "left_out", "spectrum", LEFT_OUT,
     "An undeuterated spectrum flagged `undeut_high` that the controls left out of their mean.",
     "Left out of the analysis's undeuterated values.",
     guide=CONTROLS_PAGE)
_add("undeuterated replicate the controls left out", "left_out", "spectrum", LEFT_OUT,
     "An undeuterated replicate the controls left out of their mean (far from the others, above "
     "the amide count, or a noise outlier).",
     "Left out of the analysis's undeuterated values.",
     guide=ORDER_PAGE)
_add("stand-in control", "left_out", "series", LEFT_OUT,
     "The series' fully deuterated value is a stand-in, not a measurement.",
     "Left out by `{analysis.filters.stand_in_controls}`, except in a sample with no measured "
     "fully deuterated control at all.",
     "On a large dataset these were the series most often corrected above full deuteration.",
     guide=ORDER_PAGE)
_add("stand-in control (screened)", "left_out", "series", LEFT_OUT,
     "The spectrum quality screens left out every measured fully deuterated control of the "
     "series, so only a stand-in was left.",
     "Left out by `{analysis.filters.stand_in_controls}`.",
     guide=ORDER_PAGE)
_add("stand-in control (undeuterated)", "left_out", "series", LEFT_OUT,
     "The series' undeuterated control was rejected as too heavy (`m0_high`).",
     "Left out by `{analysis.filters.stand_in_controls}`, in every sample; `hxex analyse` warns "
     "when half a sample's series or more go this way.",
     "Replacing its value by 0 fixes the start, but the other species is still in its labelled "
     "spectra.",
     "A long peptide just over the limit can be good: mark its spectra `accepted` to keep it.",
     ORDER_PAGE)
_add("above the amide count", "left_out", "spectrum", LEFT_OUT,
     "The spectrum's one-population mean is above the peptide's amide count: impossible for the "
     "peptide alone.",
     "Left out by `{analysis.filters.above_amides}`.",
     guide="statistics.md#seeing-what-was-left-out")
_add("replicate outlier", "left_out", "spectrum", LEFT_OUT,
     "The spectrum lies far from its replicates at the same time point.",
     "Left out by `{analysis.filters.replicate_outlier}` (typical replicate standard "
     "deviations), with three or more replicates.",
     guide="statistics.md#seeing-what-was-left-out")

ENTRIES: tuple[Entry, ...] = tuple(_ENTRIES)
_BY_NAME = {e.name: e for e in ENTRIES}


def entry(name: str) -> Entry | None:
    """The entry of ``name`` as *hxex* writes it; also found ignoring case and spaces at the
    ends, and with an added note in brackets (``fd_multimodal (retention agrees)``)."""
    name = name.strip()
    if name in _BY_NAME:
        return _BY_NAME[name]
    folded = {n.lower(): e for n, e in _BY_NAME.items()}
    if name.lower() in folded:
        return folded[name.lower()]
    bare = re.sub(r"\s*\(.*\)$", "", name).lower()
    return folded.get(bare)


def near(name: str, n: int = 3) -> list[str]:
    """The names closest to ``name`` (a misspelt word)."""
    return difflib.get_close_matches(name.strip(), list(_BY_NAME), n=n, cutoff=0.6)


def of_family(family: str) -> list[Entry]:
    return [e for e in ENTRIES if e.family == family]


def produced() -> dict[str, tuple[str, ...]]:
    """Every name the code writes, by family, from the modules that write them."""
    from .analysis.species import LEFT_OUT_REASONS
    from .controls import CONTROL_FLAGS
    from .picking.screen import SCREEN_SCORES
    from .quality import FLAG_NAMES

    return {"quality": tuple(FLAG_NAMES), "controls": tuple(CONTROL_FLAGS),
            "population": ("narrow_component", "undeuterated_two", "more_populations_possible"),
            "screen": tuple(SCREEN_SCORES), "left_out": tuple(LEFT_OUT_REASONS)}


# ---- filling in the thresholds ------------------------------------------------------------

def _format(value) -> str:
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def setting_values(run=None, analysis=None) -> Callable[[str], object]:
    """A lookup from a setting's path (``run.quality.min_intensity``) to its value in ``run``
    (a :class:`hxex.pipeline.RunConfig`) and ``analysis`` (a
    :class:`hxex.analysis.AnalysisConfig`), each its defaults when None; a table left unset
    (``[run.controls]`` without a table, say) gives its class's defaults."""
    from .analysis import AnalysisConfig
    from .config import _config_class, _hints
    from .pipeline import RunConfig

    roots = {"run": run if run is not None else RunConfig(),
             "analysis": analysis if analysis is not None else AnalysisConfig()}

    def value(path: str):
        first, *rest = path.split(".")
        obj = roots[first]
        for part in rest:
            if dataclasses.is_dataclass(obj) and part not in {f.name for f in
                                                              dataclasses.fields(obj)}:
                raise KeyError(path)
            nxt = getattr(obj, part)
            if nxt is None and part != rest[-1]:  # a table not set: its defaults
                cls = _config_class(_hints(type(obj))[part])
                if cls is None:
                    raise KeyError(path)
                nxt = cls()
            obj = nxt
        return obj

    return value


def _screen(score: str, action: str, level: str | None) -> str:
    from .analysis.screening import THRESHOLDS

    parts = []
    for name, table in THRESHOLDS.items():
        found = [t for s, t in table.get(action, ()) if s == score]
        if found:
            mark = " (the level used)" if name == level else ""
            parts.append(f"{found[0]:g} at `{name}`{mark}")
    return " and ".join(parts) if parts else "no level"


def render(text: str, values: Callable[[str], object] | None = None, *,
           level: str | None = None) -> str:
    """``text`` with its placeholders filled in: ``{run.quality.min_intensity}`` becomes
    ```min_intensity` (0.3)`` (the value from ``values``, :func:`setting_values`; the
    defaults when None), and a placeholder already inside backticks becomes ``key = value``;
    ``{screen:NAME:ACTION}`` the screen's threshold at each level (``level``: the level in
    use, marked)."""
    values = values or setting_values()

    def setting(m: re.Match) -> str:
        quoted, path = m.group(1) is not None, m.group(1) or m.group(2)
        key = path.rsplit(".", 1)[-1]
        shown = _format(values(path))
        return f"`{key} = {shown}`" if quoted else f"`{key}` ({shown})"

    text = re.sub(r"\{screen:([a-z_]+):([a-z_]+)\}",
                  lambda m: _screen(m.group(1), m.group(2), level), text)
    return re.sub(r"`\{([a-z0-9_.]+)\}`|\{([a-z0-9_.]+)\}", setting, text)


def plain(text: str) -> str:
    """Markdown made plain for a terminal, a CSV cell, or a tooltip: no backticks or
    asterisks."""
    return text.replace("`", "").replace("*", "")


def describe(e: Entry, values=None, *, level: str | None = None) -> dict[str, str]:
    """An entry's fields rendered (Markdown), with the settings it names."""
    return {"name": e.name, "family": FAMILIES[e.family][0], "level": e.level, "where": e.where,
            "meaning": render(" ".join(filter(None, (e.short, e.meaning))), values, level=level),
            "short": render(e.short, values, level=level),
            "action": render(e.action, values, level=level),
            "check": render(e.check, values, level=level),
            "settings": ", ".join(e.settings), "guide": e.guide}


def names_in(text: str) -> list[str]:
    """The names a cell holds (``off_mz;poor_fit``, ``fd_multimodal (retention agrees)``),
    those the dictionary knows."""
    return [part.strip() for part in str(text).replace(", ", ";").split(";")
            if part.strip() and entry(part) is not None]


def tooltip(names: Iterable[str]) -> str:
    """One line per name, the name in bold (``**name**``, for
    :func:`hxex.gui.tips.rich_tip`) with its short meaning; empty when none is known."""
    lines = []
    for name in names:
        e = entry(name)
        if e is not None:
            lines.append(f"**{name}**: {plain(render(e.short))}")
    return "\n".join(lines)


# ---- the flags.csv tables -----------------------------------------------------------------

def table(counts: Mapping[str, int], values=None, *, level: str | None = None,
          units: Mapping[str, str] | None = None) -> dict:
    """The ``flags.csv`` table: one row per name in ``counts`` with a count above 0, in the
    dictionary's order (names it does not know at the end), with ``family``, ``count`` and
    its ``unit``, ``meaning``, ``action``, ``check``, ``settings`` and ``guide``, as plain
    text with this run's thresholds (``values``)."""
    import numpy as np

    units = units or {}
    names = [e.name for e in ENTRIES if counts.get(e.name, 0) > 0]
    names += [n for n, c in counts.items() if c > 0 and n not in _BY_NAME]
    cols: dict[str, list] = {c: [] for c in ("name", "family", "count", "unit", "where",
                                             "meaning", "action", "check", "settings", "guide")}
    for name in names:
        e = _BY_NAME.get(name)
        d = describe(e, values, level=level) if e is not None else {}
        cols["name"].append(name)
        cols["family"].append(d.get("family", ""))
        cols["count"].append(int(counts[name]))
        cols["unit"].append(units.get(name, "spectra"))
        for col in ("where", "meaning", "action", "check", "settings", "guide"):
            cols[col].append(plain(d.get(col, "")))
    return {c: np.array(v, dtype=np.int64 if c == "count" else object).astype(
        np.int64 if c == "count" else str) for c, v in cols.items()}


def run_counts(results) -> tuple[dict[str, int], dict[str, str]]:
    """How many spectra carry each quality and population-test flag, how many series each
    control flag, and how many spectra each screen acts on at the default level, in a run's
    results; with the unit of each."""
    import numpy as np

    from .analysis.screening import THRESHOLDS
    from .quality import FLAG_NAMES

    counts: dict[str, int] = {}
    units: dict[str, str] = {}
    q = results.quality
    if q is not None:
        for name in FLAG_NAMES:
            counts[name] = int(np.asarray(getattr(q, name)).sum())
    ctrl = results.controls
    if ctrl is not None:
        for flags in ctrl.flags:
            for f in {t for t in str(flags).split(";") if t}:
                counts[f] = counts.get(f, 0) + 1
                units[f] = "series"
    from .report.tables import _screen_columns, _test_columns

    for name in ("narrow_component", "more_populations_possible"):
        column = _test_columns(results).get(name)
        if column is not None:
            counts[name] = int(np.sum(column))
    for name, column in _screen_columns(results).items():
        limits = [t for action in THRESHOLDS["loose"].values() for s, t in action if s == name]
        if limits:
            values = np.nan_to_num(np.asarray(column, dtype=float), nan=-1.0)
            counts[name] = int(np.sum(values >= min(limits)))
    return counts, units


def analysis_counts(tables: Mapping[str, Mapping]) -> tuple[dict[str, int], dict[str, str]]:
    """How many spectra each left-out reason took out, and how many peptides (each sample's)
    carry ``undeuterated_two`` and peptides and times ``more_populations_possible``, in an
    analysis's tables (``left_out_spectra``, ``species_uptake``); with the unit of each."""
    import numpy as np

    counts: dict[str, int] = {}
    units: dict[str, str] = {}
    left = tables.get("left_out_spectra")
    if left is not None and len(left.get("reason", ())):
        for reason, n in zip(*np.unique(np.asarray(left["reason"]).astype(str),
                                        return_counts=True), strict=True):
            if reason:
                counts[str(reason)] = int(n)
    up = tables.get("species_uptake")
    if up is not None:
        n = len(up.get("sample", ()))
        run = np.asarray(up["run"]).astype(str) if "run" in up else np.full(n, "")
        ident = list(zip(run, *(np.asarray(up[c]).astype(str) for c in
                                ("sample", "peptide", "modification", "start", "end")),
                         strict=True))
        for name, unit, with_time in (("undeuterated_two", "peptides", False),
                                      ("more_populations_possible", "peptides and times", True)):
            if name in up:
                flagged = np.asarray(up[name], dtype=bool)
                keys = {(*ident[i], str(up["time"][i]) if with_time else "")
                        for i in np.flatnonzero(flagged)}
                counts[name] = len(keys)
                units[name] = unit
    return counts, units
