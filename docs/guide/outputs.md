# Outputs: tables and figures

`hxex run` writes a results **bundle** (`[output] bundle`, such as `results`: a folder, or a `.zip`,
of Parquet tables with a `manifest.json`) and CSV tables beside it (`results_tables`).
`hxex report` writes the tables again from the bundle (with your review applied), and with
`--figures` also draws the figures (`results_figures`); `hxex analyse` writes the post-fitting analysis to a folder named after the bundle and the
reference: `results_analysis_WT` for comparisons with WT (`results_analysis` without a
reference), so analyses with different references never replace each other.

With the bundle at its usual name, `results`, and analyses with WT and Mut as the reference, a
project folder looks like this:

```text
my_protein/
├── hxex.toml                    your project file (never written by hxex)
├── user_review_table.csv        your review decisions, if you make some (see Reviewing results)
├── results/                     the bundle (hxex run)
│   ├── manifest.json            what the bundle holds, and the settings of the run
│   ├── run_settings_used.toml   every setting of the run, written out (see below)
│   ├── spectra.parquet          the raw spectra
│   ├── binned.parquet           the picked deuterium sticks
│   ├── fit.parquet, ...         the fits, controls, quality flags, bootstrap
│   └── review.parquet           the review decisions (once there are any)
├── results_tables/              the tables (hxex run, hxex report)
│   ├── series.csv
│   ├── spectra.csv
│   ├── components.csv
│   ├── uptake.csv
│   ├── review.csv               (once there are review decisions)
│   └── hxex_listing.json        the files hxex wrote here
├── results_figures/             the figures (hxex report --figures)
│   ├── overview.pdf
│   ├── WT_0012_0025_LVPRGS_z_2.pdf    one file per series, named by its label
│   ├── ...
│   └── hxex_listing.json
├── results_analysis_WT/         the analysis with WT as the reference (hxex analyse)
│   ├── left_out_summary.csv           what was left out, step by step: read it first
│   ├── species_uptake.csv, compare_peptides.csv, ...
│   ├── woods_Mut_1min.pdf, ...
│   ├── analysis_settings_used.toml    every setting of the analysis, written out
│   ├── run_settings_used.toml         the run settings of the results it read
│   └── hxex_listing.json
├── results_analysis_Mut/        the analysis with Mut as the reference
├── results.1/                   an earlier generation (hxex run --keep-previous),
├── results.1_tables/            with its tables, figures, and analyses
├── results.1_figures/
├── results.1_analysis_WT/
└── results_figures.1/           earlier figures (hxex report --figures --keep-previous)
```

Which tables and figures appear depends on the settings and the data; they are listed below.
`hxex_listing.json` is how *hxex* knows which files in a folder it wrote, so it can replace
them later and leave the rest alone. The bundle's tables are described in
[BUNDLE.md](../BUNDLE.md); the CSV tables hold the same results for spreadsheets.

These folders, and their earlier generations (`results.1`, `results.1_tables`,
`results_figures.1`, ...), belong to *hxex*: every command refuses to write over one unless
given `--overwrite` or `--keep-previous` (see [Running again](#running-again) below). In all of
them, *hxex* removes only the files it wrote itself. Keep your own files elsewhere. The
bundle's own format is described in [BUNDLE.md](../BUNDLE.md).

## Running again

A command that would write into a folder that already holds results stops before it writes
anything, and says what to do. There are three ways on:

- `--overwrite`: replace the earlier results in that folder (for the bundle, your review
  decisions are carried over);
- `--keep-previous`: first move the folder to the next free name (`results_figures.1`,
  `results_analysis_WT.1`, ...; the bundle moves to `results.1` together with its tables,
  figures, and analyses);
- another folder: `[output] bundle` (or `--bundle`) for `hxex run`, `--tables DIR` or
  `--figures DIR` for `hxex report`, `--out DIR` for `hxex analyse`.

`overwrite = true` or `keep_previous = true` in the `[output]` part of the project file does
the same for every command, so you need not type the option each time.

With `--keep-previous`, `hxex run` moves the earlier results before it reads anything, so you
know at once whether it can: on Windows a folder cannot be moved while one of its files is open
(a table open in a spreadsheet program, for example), and *hxex* then stops with every folder
where it was and tells you to close the file. If the run fails later, the earlier results are
whole at `results.1`. Names of the form `results.N` beside the bundle are kept for these
generations: do not give another bundle such a name, since *hxex* would list it as one.

| Command | Writes | Afterwards out of date |
|---|---|---|
| `hxex run` | `results`, `results_tables` | `results_figures`, `results_analysis_...` |
| `hxex resume` | `results` (adds to it), `results_tables` | `results_figures`, `results_analysis_...` |
| `hxex report` | `results_tables` | |
| `hxex report --figures` | `results_tables`, `results_figures` | |
| `hxex analyse` | `results_analysis_...` | |

`hxex resume` adds to the bundle and writes its tables again without asking: they belong to the
same results (a tables folder made from other results is left as it is, with a warning; `hxex
report --overwrite` writes it). `hxex report` likewise writes the tables of the same results
again in place, so `hxex report hxex.toml --figures` right after `hxex run` needs no
`--overwrite` for them; the figures folder, and tables made from other results, follow the rule
above. One exception keeps your work safe: if you changed one of the tables after *hxex* wrote
it (opened `spectra.csv` and saved it, say), `hxex report` stops and names the file. Copy what
you changed out of the folder first (a review table belongs next to `hxex.toml`), or give
`--overwrite` to replace it (*hxex* says which files it replaced) or `--keep-previous` to keep
the folder as `results_tables.N`. Its last line names both folders, for example `5 table(s) in results_tables (review: 1
rejected); 12 figure file(s) in results_figures`. After `hxex run` or `hxex resume`, *hxex* warns about every figures and analysis
folder made from other results, and `hxex info results` lists each folder as `up to date` or
`made from earlier results`. To bring them up to date:

```bash
hxex report hxex.toml --figures --overwrite
hxex analyse hxex.toml --overwrite
```

`hxex check hxex.toml` lists these folders under `outputs`: which exist, whether each was made
from the results in the bundle now, and which command writes each, with one closing line on
`--overwrite` and `--keep-previous`. They are information, not problems: the check still ends with `ok`, since each
command stops by itself, before doing any work.

Rerunning with more or fewer spectra can change results even for spectra you did not touch: a
few values (stand-in controls, outlier limits, the comparisons' critical difference) come from
the whole project. [Analysing part of a dataset](part-of-a-dataset.md) lists them.

Deleting only the `results` folder does not start afresh: its tables, figures, and analysis
folders stay beside it, and the next `hxex run` stops at the tables folder. To start afresh,
delete every folder whose name starts with `results`, or use `--overwrite` or
`--keep-previous`.

## Words used in the tables

- **series**: one peptide (or fragment) at one charge in one sample, followed over time.
  `data_id` numbers it.
- **K**: the number of populations fitted to a spectrum; **rank** 1 is the least deuterated.
- **mean**: deuterons of a population (or of the spectrum at K=1); **fD**: mean over the
  exchangeable amides, 0 to 1. `_corrected` values are back-exchange corrected from the
  controls. **population**: a component's share of the spectrum.
- **role**: `undeut` or `fulldeut` for a control, empty for a time point. **time**: seconds; empty
  for a control.
- **_se**: a standard error (from the bootstrap when there is one).

## Tables from `hxex run` and `hxex report`

Numbers are written with six significant digits (and at least four decimals from 1 up, so an
m/z keeps its 0.0001), whole numbers without `.0` (a 30 min time is `1800`), and an empty cell
where there is no value. The bundle keeps every digit.

| File | One row per | Holds |
|---|---|---|
| `series.csv` | series | sample, peptide, range, charge; spectra counted (`n_flagged` with a quality flag, `n_rejected` rejected or excluded, `n_multi` with more than one population, rejected spectra not counted, as in the analysis); `controls` (measured; `uncorrected` for a series the controls leave as measured, such as a fragment ion; or why a fallback was used), `control_notes` (control warnings that leave it in use: `fd_disagrees`, `fd_multimodal` with whether its retention agrees with the other samples' (`retention agrees`, `retention differs`), or else with its other charge states' (`retention agrees with its other charge states`), or `retention unchecked`, `m0_multimodal`, `data_below_m0`, `fd_charges_disagree` (the peptide's charge states, whose fully deuterated controls are combined, differ by more than 0.1), `m0_replicate_left_out` and `fd_replicate_left_out` (a control replicate was left out of the average: far from the other replicates, above the amide count, or a noise outlier), `fd_deamidated` (the fully deuterated control was lowered by its sample's deamidated share, see [Deamidated fully deuterated controls](back-exchange-correction.md#deamidated-fully-deuterated-controls))), `retention` (deuterium kept), `deamidated_share` (with `deamidation = true`, for a peptide holding an Asn-Gly site: its sample's deamidated share; empty otherwise), `control_fallback`, `noise_reference` (where the series' noise level came from: `per bin` with per-bin noise, the default, see [The noise model](noise-model.md); with one noise level per spectrum, `undeuterated control`, `own spectrum` when no undeuterated control of the series was fitted, so each spectrum set its own and may show too many populations, see [Analysing part of a dataset](part-of-a-dataset.md), or `run's scans` for kinetics), `rt` |
| `spectra.csv` | spectrum | the spectrum's identity (time, role, replicate, scan target for scan formats), its retention time `rt` and, for LC-MS scan files, `rt_start` and `rt_end` (the times of the first and last scan it summed, minutes), `review_status`, `review_K` (a number of populations chosen in the review), `note` (the review's note), quality `flags`; selected K, `centroid` (and corrected), each rank's mean and population, the [spectrum quality screens](#spectrum-quality-screens) (`screen_*`), and the series' m/z check: `envelope_offset_ppm` (how far its undeuterated peaks sit from the run's calibration), `envelope_offset_limit_ppm` (the limit for `off_mz`; both empty where not checked) and `window_shift_ppm` (0, except in results made with the former `recentre`; [details](picking-window.md#a-series-whose-peaks-sit-off-its-mz)). With the [population test](population-test.md): `criterion_K` (the number the fit's score chose; K is then the test's), `test_p_1` and `test_p_2`, how each step's null was made (`test_null_1`, `test_null_2`: `simulated`, `chi_square`, `screened`, `flat` when every simulation gave the same value so no chance can be worked out, or empty when the step was not tested), and the flag `narrow_component` (the fit at the spectrum's K has a population narrower than deuterium allows). A copy of it is the starting point of the [user review table](project-file.md#the-user-review-table) |
| `components.csv` | spectrum and population | K, rank, mean, mean_se, corrected mean, population and their errors (the same values as the `mean_<rank>`, `mean_se_<rank>`, `mean_corrected_<rank>`, `mean_corrected_se_<rank>`, `population_<rank>` and `population_se_<rank>` columns of `spectra.csv`, one row per population instead of one column set per rank) |
| `uptake.csv` | series and time (or control), then K and rank | the replicates averaged, rejected spectra left out. Each time starts with a `centroid` row (in the `description` column): the average of every replicate's centroid (`centroid_corrected` of `spectra.csv`, or `centroid` for a series the controls leave uncorrected) whatever number of populations each replicate chose, with `n_K1`, `n_K2`, and `n_K3` counting the replicates at each K, and `K`, `rank`, and `population` empty. The rows after it average only spectra at the same K, one row per population, described as `K=1`, `K=2 low-D`, `K=2 high-D` (and `K=3 low-D`, `K=3 mid-D`, `K=3 high-D`; beyond three by rank, `K=4 rank 1` and so on); their `n_K` columns are empty. Every row has the series' `start`, `end`, and `charge`, the time in seconds (`time`) and as a label (`time_label`: `30 min`, or `UN` and `FD` for the controls), `role`, `n_amides` (the peptide's exchangeable amides), `n_replicates`, `mean` (in deuterons, back-exchange corrected for a series the controls correct; as measured for one they leave uncorrected, such as a fragment ion), `mean_se` (the error of that average, `replicate_sd` divided by the square root of `n_replicates`; empty with one replicate, as is `replicate_sd`: it holds no fit error), `replicate_sd`, `fD` (`mean / n_amides`, the fraction deuterated; empty for an uncorrected series), and `population` |
| `review.csv` | reviewed spectrum (only when there are any) | status, the number of populations chosen (`review_K`), note, date, stale, and source: the `review table` (`user_review_table.csv`), an `exclusion rule` of the project file, or a `review` decision stored in the bundle |
| `retention_suggestions.csv` | series searched (only for scan files read with `find_peak`) | where the peptide's exact m/z elutes in the undeuterated controls, for each series that sits off its m/z or has no undeuterated envelope: `searched_because`, the peptide table's `table_rt`, the time it was read at (`read_rt`), the `verdict` (`here`, `elsewhere`, `uncertain`, `ambiguous`, or `not found`), the `proposed_rt` (minutes), whether it was `applied`, and every candidate time with its height ([details](inputs.md#when-the-peptide-tables-time-is-wrong)) |

## Figures from `hxex report --figures`

`hxex run` draws no figures. `hxex report hxex.toml --figures` draws them from the results
(and writes the tables again, as `hxex report` always does), with the settings of the
`[figures]` part of the project file; changing those settings needs
no new run, only `hxex report` again. (Given the bundle instead, `hxex report results
--figures` uses the usual settings and its own options.)
[Figures](figures.md) lists them with the settings that choose and change each.

- `overview.pdf`: which K was selected and how often (per time, before and after review); the
  controls (retention against its limits, measured against fallback); the quality flags. Each
  dashed limit line is named beside it (for example `outlier limit 0.23` or `max 1.1`), and
  the bars say in a few words what they count, close to the flag names of `series.csv` and
  `spectra.csv` (`UN two populations` is `m0_multimodal`, `low intensity` is
  `low_intensity`). On the controls page, bars marked `(note)` (light grey) count controls
  that are used but carry a note, some of them good news (`agrees with other samples`); the
  orange bars say why a control was missing or not used. The quality page's middle panel
  plots each spectrum's misfit (relative to the typical one of its series) against its
  intensity (relative to its replicates): the poor fits worth a look are those above the
  dashed line but right of the vertical one, poor fits that are not merely weak spectra.
- One file per series, named by its label (sample, range, peptide or fragment, charge,
  condition), with a page per kind. Each page has one legend, at the top or on the right, that
  names every symbol on it.
  - **spectra**: raw data, picked values and the fit, the picking window shaded. Each panel's
    title gives the time, the replicate, the number of populations drawn, the review status
    when there is one, and the quality flags, for example `30 min, rep 2 (K=1, rejected;
    low_intensity)` or `4 s, rep 1 (K=2; bright)`. The marker on top of each picked value says
    what the bins hold: a circle for peak heights, an up-pointing triangle for peak areas. An
    area is turned into a height-equivalent number (see [A peak's height or its
    area](picking-window.md#a-peaks-height-or-its-area)), so a triangle can sit below the top
    of a narrow, bright peak or above a wide, weak one: that is by design, not a mispick. The
    legend says which: "picked values: areas (height-equivalent)" or "picked values: heights";
    the panels are laid out a time point per row, its replicates side by side (a single
    replicate's time course fills rows instead), at most `[figures] page_rows` rows (8) of
    `page_columns` (6) a page: more replicates of a time point go on over the next row, and a
    series with more go on over further pages (more pages in the PDF; numbered files,
    `<series>_spectra_1.png`, `_spectra_2.png`, ..., for PNG and SVG). A long time course
    (a kinetics run's hundreds of scans) is drawn at no more than `[figures] max_points` time
    points (48), spread over log time with the first and last kept (fewer when early scans lie
    closer together than the spacing: each is drawn once); the page header says how many of
    how many are shown. Every spectrum is still fitted and in every table; `max_points = false`
    draws them all, over as many pages as they take;
  - **populations**: each population's mean and share, one panel per spectrum (titled as on the
    spectra page), with a last column that overlays the replicates (one marker shape per
    replicate); the star is the population-weighted mean of the spectrum. The axis reaches the
    largest population drawn, so a population above the amide count is shown, not cut off.
    The page is left out when every spectrum has K=1;
  - **uptake**: deuteration against time (or index with `time_scale = "order"`), corrected by
    default when there are controls. The marker's area grows with the population; each
    replicate has its own colour (and, beyond six replicates, its own marker as well). A
    rejected or excluded spectrum is drawn hollow and grey. On a time axis the series' control
    spectra are drawn at the ends of the axis, the undeuterated ones before the first time and
    the fully deuterated ones after the last, labelled `UN` and `FD` on the axis itself (no
    time is marked beyond the last time point, so they are not read as times), hollow and grey
    when rejected or left out by the quality screens. The dotted lines mark the levels the correction uses (0
    and the amide count when corrected; else the undeuterated and fully deuterated means); when
    a level came from a stand-in rather than the series' own control, the legend names it, for
    example "FD level (amide count): median stand-in". With `x = "index"` the controls are
    drawn as the first and last points. On a logarithmic time axis (`scale = "log"`) a time
    point labelled at 0 s cannot be drawn.

A series' file name keeps the letters of any alphabet in its label (as the analysis folder's
names do) and replaces everything else with `_`; a sample name longer than 30 characters is
shortened, never the range, peptide, and charge that follow it. When two series would get the
same name, `_id` and the series' `data_id` are added (`WT_0001_0010_PEPTIDE_z_2_id7`).

By default only the flagged series get figures (a quality flag, a rejection or a control
fallback); `series = "all"` in `[figures]`, or `hxex report hxex.toml --figures --series all`,
draws them all. `format` is `pdf` (pages in one file), `png` or `svg` (one file per page).

Each spectrum is drawn with its own number of populations: the one the fit chose, or the one
chosen in the review. To see every series at one number of populations instead, for example to
compare one and two populations side by side, set `k` in `[figures]` (or give `--k`):

```bash
hxex report hxex.toml --figures --k 1              # every series at one population
hxex report hxex.toml --figures results_k2 --k 2   # at two, into a folder of its own
```

`k` must be one of the numbers the run fitted (`k_min` to `k_max`). It changes only the
figures: the tables and the analysis keep each spectrum's own choice. A spectrum for which that
number was not fitted (the fit's screen skipped it) is left out of the populations and uptake
pages, and its panel on the spectra page says so.

## Quality flags

Flags mark spectra for review; nothing is dropped. They appear in the `flags` column of
`spectra.csv` and in the overview.

| Flag | Meaning |
|---|---|
| `poor_fit` | the misfit is many times the series' median (`poor_fit_factor`) and the deviance is high |
| `low_intensity` | far less signal than the other replicates at the same time |
| `noise_outlier` | a replicate the replicate-based noise estimate left out; a control replicate is also left out of its control's average (its series gets `m0_replicate_left_out` or `fd_replicate_left_out`) |
| `mz_shift` | the series sits off its predicted m/z: a wrong sequence, modification, terminus or ion type (series that `off_mz` does not check: fragment ions, and series without an undeuterated control) |
| `off_mz` | the series' undeuterated controls put its peaks further from the run's calibration than the run's other series: most likely another species (often the retention time points at another peptide of almost the same mass); the analysis leaves the series out ([details](picking-window.md#a-series-whose-peaks-sit-off-its-mz)) |
| `rt_found` | the series was read at the retention time found for its exact m/z, not the peptide table's (`find_retention_time = "automatic"`; [details](inputs.md#when-the-peptide-tables-time-is-wrong)) |
| `poor_envelope` | the fragment envelope could not be measured well |
| `saturated` | an undeuterated control whose peaks beside the tallest one read higher than natural abundance predicts: the tallest peak reads low, the usual sign of detector saturation |
| `tails_low` | an undeuterated control whose small isotope peaks read lower than natural abundance predicts; seen in bright spectra, cause unknown |
| `bright` | a labelled spectrum or fully deuterated control at least as bright as a control of its series flagged `saturated` or `tails_low` |

### Very bright spectra: `saturated`, `tails_low`, and `bright`

An undeuterated control has a known shape: the natural isotope pattern of the peptide. *hxex*
compares each undeuterated control with that pattern, with the height set by its tallest peak.

- **`saturated`**: the peaks beside the tallest one read more than 1.15 times what the pattern
  predicts (the setting `saturated_ratio`). The tallest peak is then too low, which is how a
  detector behaves when it saturates.
- **`tails_low`**: the small isotope peaks, those the pattern puts at 1 to 10 % of the tallest,
  read less than 0.7 times their prediction (the setting `tails_low_ratio`). This is seen in the
  brightest spectra of several datasets and instruments. Its cause is not known, so the flag
  names what is seen, not a mechanism.
- **`bright`**: the labelled spectra and fully deuterated controls of the same series that are at
  least as bright as a flagged control. Their own shape cannot be checked, because it is not
  known in advance, so they are marked as likely to be distorted in the same way.

A control is flagged only when most of its series' undeuterated replicates show the sign, and
only the brighter half of a run's undeuterated controls is judged: the peak ratios of a weak
spectrum are mostly noise. Set either setting to `false` in `[run.quality]` to switch its flag
off.

**What they mean for the results.** Measured on four datasets, with about 400 to
3,000 undeuterated controls each:

- A `saturated` undeuterated control reads 0.07 to 0.17 deuterons too high (about 1 to 2 % of
  the peptide's amides). Its value is subtracted from every time point of the series, so the
  corrected deuterium uptake of that series reads low by about as much, unless its labelled
  spectra are saturated in the same way. On the dataset with the most saturated controls they
  were also fitted with more than one population more often (22 % against 14 % of unflagged
  controls of similar brightness).
- A `tails_low` spectrum is narrower than the pattern predicts. It does not make extra
  populations more likely: flagged controls chose more than one population less often than
  unflagged ones (2 % against 5 %). The mean of a labelled spectrum is expected to read slightly
  low, by an amount the controls cannot show, because a control's value cannot fall below zero.
- A `bright` spectrum chooses more than one population more often than its dimmer replicates at
  the same time point (12 to 23 percentage points more). The same happens in series with no
  flagged control at all, and grows with how much brighter one replicate is than another (5 to 7
  points at up to 1.5 times brighter, 15 to 38 points at three times or more). So most of it is
  not the distortion these flags detect, but the noise level: these numbers were measured with
  one noise level per series, set from its undeuterated control (the default before 31 October
  2026), which makes the error bars of a spectrum much brighter than that control too small.
  Its mean differs little from its replicates' (a median of 0.01 to 0.02 deuterons). The
  per-bin noise, now the default ([The noise model](noise-model.md)), gives each bin a noise
  that grows with its intensity, which addresses this cause; its effect on bright spectra alone
  has not been measured.

What to do: none of these flags removes or changes anything. When a result rests on a
`saturated` series, check its corrected uptake against the peptide's other charge states. When
two populations are found only in the brightest replicate of a time point, flagged or not, look
at the dimmer replicates before trusting the second population.

## Spectrum quality screens

When *hxex* picks a spectrum, it also looks at the raw spectrum around the peptide for signs of
another species: a second peptide, or a contaminant, whose peaks overlap the peptide's. Such a
spectrum can shift the measured deuteration or show a second population that is not there.
Five scores, each a share of the intensity from 0 (nothing seen) to 1, are in the last
columns of `spectra.csv`.

**`screen_below_mono`: another species starting below the peptide.** *hxex* looks one and two
isotope steps below the peptide's lightest (monoisotopic) peak. Peaks there belong to something
else, and that species' own isotope peaks carry on into the peptide's lowest peaks. The score
is the share of the peptide's peaks that this carry-on explains. It moves the measured
deuteration (on a large test dataset, by one to three times the replicate scatter), so
`hxex analyse` leaves the spectrum out. The score is empty when the spectrum does not reach far
enough below the peptide to look.

**`screen_off_position`: low peaks in the wrong place.** The five lowest peaks should sit where
the peptide's composition puts them. The score is the share of the intensity in those peaks
found more than half the picking window away, after allowing for the series' own offset (the
calibration, measured on its undeuterated control). An overlapping species of slightly
different mass does this. Such spectra disagree with their replicates, so `hxex analyse` leaves
the spectrum out.

**`screen_between`: peaks between the isotope positions, in the envelope.** A peptide's peaks
come one isotope step apart; peaks well between those positions (outside the picking window,
at least a quarter step from any position) belong to another species, often of another charge.
The score is their share of all the peaks, counted where they lie inside or touching the
peptide's envelope. These spectra also disagree with their replicates, so `hxex analyse` leaves
the spectrum out.

**`screen_between_separate`: a separate cluster above the envelope.** The same kind of peaks,
but in a cluster of their own above the peptide's envelope, still in the range the peptide's
deuterium could reach. It does not move the one-population result, but the fit can take the
cluster for a second, more deuterated population. `hxex analyse` keeps the spectrum with one
population only.

**`screen_beyond_reach`: intensity past the peptide's reach.** The share of the picked intensity
beyond the heaviest the peptide can be (every amide deuterated, plus its natural isotope
envelope). Those peaks belong to something else. A little of it made no measurable difference
to the fits, so `hxex analyse` keeps such a spectrum and only warns, with a count. When most of
the picked intensity lies there, the spectrum is mostly another species: on a Q-TOF dataset
more than half of those spectra sat over a deuteron away from their replicates, and on a
crowded Orbitrap digest, where the other species comes back in every replicate, many sat far
from the expected uptake. `hxex analyse` leaves those out.

What `hxex analyse` does is set by `spectrum_quality_filtering` in `[analysis.filters]`:
`"loose"` (the default), `"strict"`, or `"none"` (the scores are still in `spectra.csv`, but
nothing is done with them). The levels were set from fits on a large dataset: with and without
the interference taken out, and how well replicates agree.

| Score | `loose` | `strict` | What happens |
|---|---|---|---|
| `screen_below_mono` | 0.10 | 0.05 | left out |
| `screen_off_position` | 0.20 | 0.10 | left out |
| `screen_between` | 0.40 | 0.20 | left out |
| `screen_between_separate` | 0.20 | 0.10 | one population only |
| `screen_beyond_reach` | 0.10 | 0.10 | warning |
| `screen_beyond_reach` | 0.50 | 0.30 | left out |

A spectrum left out is listed in `left_out_spectra.csv` with the score that caught it (for
example `spectrum quality (screen_below_mono)`). On a large intact-peptide dataset `loose` left out
about 2 % of the spectra and `strict` about 4 %.

- **Controls.** A control the screens leave out is also left out of the back-exchange
  correction. `hxex run` already leaves out the controls the `loose` level removes, so
  `series.csv` shows the controls the default analysis uses; with `strict` or `none` the
  analysis works the correction out again for its own level. When that removes every measured fully
  deuterated control of a series, the series has only a stand-in left and is left out too
  (`stand-in control (screened)` in `left_out_spectra.csv`; see
  [Back-exchange correction](back-exchange-correction.md)).
- **Your review wins.** A spectrum marked `accepted` in the
  [user review table](project-file.md#the-user-review-table) is never screened out.
- **Fragments are not screened.** The levels were set on intact peptides; fragment spectra
  have their scores in `spectra.csv` but nothing is done with them.
- **Not assessed.** An empty score means *hxex* could not look: the spectrum had no picked
  intensity, the data were picked outside *hxex* (a table of picked peaks, not spectra), or
  the results were made by an earlier version of *hxex* (run again to get them).

The tolerances follow the picking window (`[run.picker]`), so the scores mean the same on
instruments of different resolving power. The run itself does nothing with the scores: its
tables and figures show every spectrum.

## Tables from `hxex analyse`

Every analysis table starts with `run` and `sample` (the run is blank for a single bundle).

**Peptides and residues**

| File | Holds |
|---|---|
| `species_uptake.csv` | per species (peptide or fragment, charges and replicates combined), time, K and rank: mean, fD and their errors, population, amides, `charge_trend`, `control_fallback` (corrected with a stand-in control: such series are left out unless `stand_in_controls = "include"` or the sample has no measured control), `fd_other_charge` (true only with `combine_charges = false`, where such a series stays a stand-in: the only stand-in was a fully deuterated control from the peptide's other charge states, which the protection-factor fit treats as measured; with `combine_charges = true`, the default, such a series counts as measured and this column is false); and the number of populations chosen for the species at that time from all its spectra together: `best_set_K`, `best_set` (true on the rows of that description: filter on it for one description per peptide and time), `best_set_reason`, and `set_delta_criterion` (the evidence: negative favours two populations); with the [population test](population-test.md), `best_set_K` comes from the test, with its p-values `set_p_1` (one against two) and `set_p_2` (two against three), and `set_test_K`, the test's own number (`best_set_K` follows it only up to the largest K analysed, `[analysis] ks`; 0 for a set the test never saw, every spectrum of it left out by the filters), and the flag `undeuterated_two` (the peptide's undeuterated set reads two populations in some sample: another species); `average_from_k2` (on every row of a species and time whose set has two populations at least 2 deuterons apart: its K=1 values are the two populations' share-weighted average, not the one-population fit, which there settles on one of them; see [Two populations far apart](statistics.md#two-populations-far-apart)) |
| `left_out_spectra.csv` | the spectra the analysis left out whole (`[analysis.filters]`), with `reason`: `off its m/z` (its series is flagged `off_mz`, controls included; [details](picking-window.md#a-series-whose-peaks-sit-off-its-mz)), `replicate outlier` (far from its other replicates), `above the amide count` (more deuterium than the peptide can hold, before the back-exchange correction), `stand-in control` (its series has no measured fully deuterated control), `stand-in control (screened)` (it had one, but the [spectrum quality screens](#spectrum-quality-screens) left it out), `stand-in control (undeuterated)` (its undeuterated control was too heavy to be the peptide alone, `m0_high`; [details](back-exchange-correction.md#how-hxex-judges-the-controls)), or `spectrum quality (<score>)` (another species in the spectrum: the score that caught it). To keep one, mark it `accepted` in the [user review table](project-file.md#the-user-review-table) |
| `left_out_summary.csv` | what was left out on the way from the spectra read to the analysis, one row per step, with a count of 0 where nothing was: before fitting (the `[[input.exclude]]` rules, spectra without peaks), the review (rejected, a chosen number of populations), the controls (control spectra screened out, missing or rejected controls, stand-ins), and the analysis (the whole spectra of `left_out_spectra.csv` by reason, and the two-population descriptions set aside by each filter, the spectrum kept at one population). Columns: `stage`, `what`, `count`, `unit` (spectra, series, or peptides and times), and `setting` (where it is set). Read it first: it shows where a dataset loses its spectra |
| `residues.csv` | per residue, time, K and rank: fD from the overlapping peptides, redundancy, resolution and the block the residue lies in (`block_start`, `block_end`: the stretch the peptides cannot divide, whose length is the resolution), coverage; with the [population test](population-test.md), on the K=1 rows: `population_tier`, `test_strength_2`, `test_agreement_2` (and `_3`) |
| `coverage.csv` | per residue: redundancy, resolution, proline, covered |
| `populations.csv` | per residue and labelling time (the controls take no part): the K=1 and the two K=2 values, whether two populations are kept and, if not, why |
| `regions.csv` | per region and labelling time (the controls take no part): fD over the region's residues |
| `compare_regions.csv` | with regions and a reference: per region and time, whether each sample differs from the reference there: `delta` (the mean of its residues' own differences), `delta_shared` and `delta_shared_se` (the difference and error the test used), `z`, `significant`, `peptide_support`, `region_test_only`, `n_residues` and `coverage` (the region's residues compared), and `delta_crit` (see *Reading the region significance map* below) |

**Comparisons** (with a `reference`)

| File | Holds |
|---|---|
| `compare_peptides.csv` | per peptide and time: the difference from the reference, its error, t, p, the critical difference (`delta_crit`) and `significant`; `n_reference` and `n`, the replicates on each side (a replicate's charge states count once); for the population, the `separation` of its two populations, and `reason` says why a peptide was not compared (for example `not separated (reference)`: its two populations are too close to tell apart; `one population (sample)`: that side's set is best described by one population; `no K=2 (sample)`: that side has no two-population description left, for example reviewed to one population), or, for a peptide compared at K = 2 although a side has three populations, says so (`three populations (sample): K = 2 compared`). A sample with no labelled time point (controls only) is left out of the analysis, with a warning |
| `compare_residues.csv` | per residue and time: difference, error, z, `significant` (at K=2, only where both samples keep two populations), and whether each side keeps two (`two_reference`, `two`); the values and `delta` are each sample's own, while `z` and `significant` come from the peptides measured in both samples at that time, whose difference and error are `delta_shared` and `delta_shared_se` (see *Reading the residue overlay* below) |
| `compare_population_counts.csv` | per peptide and time: whether each side is best described by one or two populations (`K_reference`, `K`), the evidence (`delta_criterion`: negative favours two), `differs`, and `borderline` (a difference with weak evidence); with the [population test](population-test.md), `p` and `p_reference` (each side's p-value where they differ), which then decide `borderline` |
| `compare_populations.csv` | per residue and time: the low-D population against the reference's, with the critical difference (`delta_crit`) and `significant` (judged from the peptides measured in both samples, `delta_shared`; empty without a bootstrap at K = 2, see [the statistics page](statistics.md#what-the-bootstrap-adds)). The table keeps the low-D share, as `compare_peptides.csv` does; the figures draw the high-D share (one minus it, so `delta` has the opposite sign there), so that higher is more deuterium |
| `compare_residues_K2.csv` | with a bootstrap at K = 2 only: per residue, time, and `rank` (1 the low-D population's position, 2 the high-D one), the difference from the reference where both keep two populations, laid out as `compare_residues.csv` |
| `compare_choices.csv` | per residue: whether each side keeps two populations, and whether they agree |

**Runs** (several bundles): `run_settings.csv` (settings that differ between the runs, and
whether they matter) and `run_identity.csv` (species whose position or amide count differs).

**Protection factors** (`protection = true`): `protection.csv` (per peptide, K and rank:
`log10_protection` with its error, bounds, misfit, the direction), `protection_by_time.csv` (the
apparent factor at each time) and, with a reference, `compare_protection.csv`.

**Fragments**

| File | Holds |
|---|---|
| `fragment_uptake.csv` | per fragment spectrum: the K=1 mean, its error, amides, condition, parent |
| `ladder_rates.csv` | per one-residue ladder step: rate and amplitude with errors, `resolved`, `precise`; with `kint`, intrinsic rate and protection |
| `ladder_residues.csv` | per residue and N-/C-terminal ion pair: the combined rate, misfit, `agreement` |
| `closure.csv` | per complementary pair and split: the miss c + z - M, flagged beyond the tolerance |

## Figures from `hxex analyse`

[Figures](figures.md) lists every figure `hxex analyse` draws, with its file names, what it
shows, the settings it needs, and the settings that change it.

`[analysis] format` sets the file type (`pdf`, `png`, `svg`).

A sample keeps its colour in every figure that colours samples (the residue overlays, the
every-sample residue figure, the region bars, and redundancy and resolution), whichever samples
a figure shows and whichever
sample is the reference: the first sample black, the others in their order (that of
`[analysis] samples`, which also sets the order of every figure's rows and panels; else
alphabetical). `[analysis.colours]` chooses them yourself, one line per
sample (`D109H = "orange"`, or a hex code such as `"#1f77b4"`); a sample you do not name keeps
a default colour that none of yours uses.

**Every sample at once.** `residues_all_<time>` draws every sample's fDavg residue by residue
on one axis, with none of them a reference: each sample in its colour and marker, a line broken
where no peptide measures two residues together, and a light band of ±½ δcrit, half the
smallest difference the peptide comparisons call significant (one value for every pair of
samples, named above the plot): where two samples' bands are apart, they differ by more than
that; where they overlap, by less (without a reference sample there is no δcrit, and the bands
are ±1 standard error). Every time's figure has the same y axis, so they compare at a glance.
fDavg is
the analysis's average deuteration, as on the overlays: the K=1 fit's, or, where a peptide's
spectra have two populations at least 2 D apart (`average_from_k2` in `species_uptake.csv`),
the K=2 population-weighted mean, which the one-population fit misses. It is the figure to
look at first for a run with several samples; the Woods and overlay figures then say which
differences are significant.

**Reading the Woods plot.** Each panel shows, for every peptide, a bar over the residues it
covers at its difference from the reference (sample minus reference), with its standard error as
a vertical line. A bar is coloured where the difference is significant (red: the sample is
higher; blue: lower): Welch's t-test on the replicate spectra gives p below `alpha`, and the
difference is larger than the critical difference, drawn as the dashed lines. The legend above
the panel names the colours and the dashed lines. On every panel red means more deuterium in
the sample and blue less, which is why the population panel shows the share of the high-D
population: a larger high-D share is more deuterium.

The plot comes in two figures. The first, `woods_<sample>_<time>`, has the average deuteration
(K=1 fD, one population) and, below it, **Two Populations Selected**: a grey bar over each
peptide whose spectra are best described by two populations in one sample but not the other
(`<sample> only` above the line, `<reference> only` below), and the peptides where both have
two (`both`). A bar is outlined, not filled, where the evidence for the difference is weak
(borderline). A change in the number of populations is a finding in itself: one sample has
two states where the other has one.

The second figure, `woods_K2_<sample>_<time>`, compares the two populations: the fD of the
low-D population, of the high-D one, and their populations. It compares only peptides that
have two populations in both samples: elsewhere one sample's second "population" is a small
tail of a single one, and comparing it with the other's second state would compare different
things.

The population panel shows the change in the share of the high-D population (the low-D
share's change with its sign turned round, since the two shares add up to one), and not
directly but multiplied by the peptide's separation: the difference in fD between its two
populations (the mean of the two samples'). The product is the deuterium, as a fraction of the
amides, that moved from one state to the other, so the panel is in the same units as the fD
panels and has one critical difference, the K=1 one. This is the same test as comparing the
change in share with the critical difference divided by the separation: a small change of share
matters in a peptide whose two states are far apart, and little where they are close.
`compare_peptides.csv` keeps the change in the low-D share itself (`delta`, the comparison
`K=2 rank 1 (low D) population`: the opposite sign of the panel's), with the `separation` and
the threshold it was judged by (`delta_crit`, the K=1 critical difference divided by the
separation).

**Reading the residue overlay.** `overlay_residues_<time>` draws each sample against the
reference residue by residue, each in its colour. The average
deuteration (K=1, fDavg) is small circles on a line with a shaded ±1 standard-error band; the
line joins two residues only where one peptide measures both, so a gap means the peptides do not
connect them. Where a sample has two populations, fDlow is an up triangle and fDhigh a down
triangle, with their error bars; the axis is fD, and the legend names the shapes (fDavg,
fDlow, fDhigh) in grey beside the samples' colours. A residue has two populations for a sample when the peptides
whose spectra are best described by two populations carry at least half of its weight (and the
two values are separated, precise, and on either side of the average). Its fDlow and fDhigh, like
its average, come from every peptide covering it, so the three always describe the same
peptides. Residue values change only where a peptide starts or ends, so the
average runs in steps: the width of a step is the resolution the peptides allow.
`overlay_populations_<time>` is drawn the same way, with the share of the high-D population
(`pop_high`) as down triangles, the legend showing each sample's triangle in its colour (the
high-D share, as on the Woods plot: higher is more deuterium): each sample where it has two populations, whether or not the other
sample has. The by-residue figure can show any of these values (`overlay` in `[analysis]`:
`low`, `high`, `avg`, `pop_low`, `pop_high`).

The values in these figures are each sample's own. Whether a residue's difference is
significant (`significant` in `compare_residues.csv` and `compare_populations.csv`) is judged
differently: from both samples' values rebuilt from only the peptides measured in both samples
at that time. A peptide that only one sample has, because a filter left it out of the other or
because the other never measured it, changes that sample's values alone, so the difference
would partly show which peptides were averaged rather than a change in exchange. The
difference the test used is in `delta_shared` (with `delta_shared_se`); where it is much
smaller than `delta`, the samples' peptides differ there. A residue that only such peptides
cover is not tested (`delta_shared` empty); `hxex analyse` says how many.

A residue's difference also counts as significant only where at least one peptide covering it
shows the difference itself: that peptide is significant in `compare_peptides.csv`, at the same
time, in the same direction (`peptide_support` true). This is a safeguard. The residue test
divides each difference by its error, and if those errors are too small (for example when the
noise model does not suit the data) it calls differences that are not there; the peptide test
uses the replicates instead. The residue test can also add up several overlapping peptides that
each fall just short, so a residue it calls without any single peptide carrying the difference is
kept visible rather than dropped: `residue_test_only` is true, `significant` is false, and the
significance maps draw it grey with dots in the colour of its direction (red for more deuterium
than the reference, blue for less), labelled "significance not confirmed at peptide level". Look at those residues with that in mind: with suitable
error bars some are real small differences that only the combination finds; with error bars that
are too small most are not. In the K = 2 tables the same rule uses the K = 2 peptide comparisons.

**Reading the residue significance map.** `residue_significance` shows, for every sample and
time, which residues differ significantly from the reference, laid out like `population_tiers`:
one row per sample and time, one cell per residue. A significant residue is red where the sample
holds more deuterium than the reference and blue where it holds less, in three shades by the
size of the difference: 1 to 2, 2 to 4, or more than 4 times the threshold (`delta_crit`). The
size is the difference the test judged (`delta_shared`), so the colour and the decision come from
the same comparison. Grey cells were compared and are not significant; grey cells with red or
blue dots passed the residue test, but no significant peptide confirms the difference
(`residue_test_only`, see above: "significance not confirmed at peptide level"), so they are not
counted as significant, and the dots' colour gives the direction; cross-hatched cells could not be tested, because no peptide measured in both samples covers them; white cells were not
compared (a residue not covered in one of the samples, a proline, or the first residue of the
protein). The shades show size, not how sure the test is: the residue errors come from the fits,
and grading by them would give the deepest colours where the errors are most understated.

Thin dark outlines mark the residue blocks: stretches between the ends of the peptides both
samples measured at that time, which the peptides cannot divide (`block_start` and `block_end`
in `compare_residues.csv`; their length is the resolution). Every residue in a block has the
same value, so a block coloured as one tells you only that the difference lies somewhere in it.
A long block means a call that cannot be placed more finely; narrow blocks place it nearly to the
residue. The blocks come from the pair of samples in that row, so they can differ between rows.

`residue_significance_K2` does the same at K = 2, in up to three sections. The first,
*populations*, shows where only the sample keeps two populations (hatched one way) or only the
reference does (hatched the other way); white means one population in both. Where both keep two,
the cell is grey, or, with a bootstrap at K = 2, coloured when the share of the low-D population
differs significantly: blue where the sample holds more of the low-D population (so less
deuterium overall), red where it holds less. The other two sections, drawn only with a bootstrap
at K = 2, show whether the low-D and the high-D population each sit at a different deuterium
level from the reference's, coloured as above.

**Which peptides decide one or two populations on this map.** Each sample's own results
(`compare_choices.csv`, the `choice_` figures, and `population_tiers`) decide one or two
populations from all of that sample's peptides. This map decides it again for each pair of
samples, from only the peptides both samples measured at that time, the same peptides its test
and its outlines use. The two can disagree when one sample has a peptide the other lacks (one
left out by a filter, or not measured): that extra peptide counts in the sample's own decision
but not on the map.

A yellow triangle in a block's corner marks a block where they disagree. The block's colour or
texture is always the map's decision, from the shared peptides; the triangle says that at
least one sample's own peptides decide differently there. For example:

- **Grey with a triangle:** on the shared peptides both samples have two populations, and their
  shares do not differ significantly. But one sample has an extra peptide there that reads one
  population, so its own result (its `choice_` figure) shows one population, and
  `compare_choices.csv` lists two populations in the other sample only.
- **Hatched with a triangle:** on the shared peptides only one sample has two populations, but
  the own peptides decide otherwise: both, or neither, or the other sample.
- **White with a triangle:** on the shared peptides neither sample has two populations, but an
  extra peptide gives one of them two in its own result.

To see what the own peptides decided at a triangle, look up that residue and time in
`compare_choices.csv` (`two_reference` and `two`) or in each sample's `choice_` figure. A block
with a triangle is one where the answer depends on which peptides you count: treat the
difference in the number of populations there as uncertain.

**Reading the region significance map.** With regions defined (`[[analysis.regions]]`),
`region_significance` shows whether each sample differs from the reference over each region:
one row per sample and time, one column per region, in order of where the regions start. A
region's difference is the mean of its residues' differences on the peptides both samples
measured, and its error the mean of their errors (the residues share peptides, so averaging them
does not make the region more precise than its residues). It is significant when the difference
is large compared with that error and larger than the threshold (`delta_crit`), and when a
significant peptide in the same direction lies at least half inside the region. The colours are
those of the residue map: red for more deuterium than the reference, blue for less, in three
shades by size; grey not significant; grey with red or blue dots where the region test passed
but no significant peptide lying mostly inside the region confirms it (`region_test_only`:
"significance not confirmed at peptide level", the dots' colour giving the direction); cross-hatched where none of the region's residues could be
compared. A region is not called different just because one residue in it is: tried on simulated
data, that rule called about a third of the regions next to a real change different.

## The settings used

Two files record every setting that was used, including the ones you left at their usual
values, so you can see later exactly how a result was made (for the methods section of a
paper, say) without keeping the project file in step:

- `run_settings_used.toml`, in the bundle: the settings of `hxex run` (what was read, the
  fit, the bootstrap, the controls, ...). A setting you did not write in `hxex.toml` shows its
  usual value; a step that was switched off shows as, for example, `bootstrap = false`.
- `analysis_settings_used.toml`, in each analysis folder: the settings of `hxex analyse`.
  Beside it is a copy of the run's `run_settings_used.toml` (with an analysis of several runs,
  one per run: `run_settings_used_A.toml`, ...). The analysis folder therefore still says how
  its results were made after `hxex run --overwrite` has replaced the bundle.

`hxex run` leaves the analysis folders as they were, so for a while they describe the earlier
results. `hxex info results` lists each analysis folder as `up to date` or `made from earlier
results`, and `hxex analyse` warns about the other analysis folders of the same results that
are out of date. Run `hxex analyse --overwrite` again (with that folder's `--reference`) to
bring one up to date.

Both are ordinary project files: open them in a text editor to read them, or copy one out of
its folder and use it as a project file to repeat the run. *hxex* writes them with the
results and replaces them with the results, and never reads them back; editing them changes
nothing. File paths are written as `record_paths` in `[input]` says: with `record_paths =
"name"`, only file names appear, which is useful when you send results to someone.

## Sharing results

To send your results to someone, zip the results folder (`results`) in the usual way: on
Windows, right-click it and choose **Send to > Compressed (zipped) folder**; on macOS,
right-click it and choose **Compress**. The person receiving it does not need to unpack it:
every *hxex* command reads the zip as it is, for example `hxex report results.zip --figures`.
Include `results_tables`, `results_figures`, and the analysis folders (`results_analysis_WT`)
as well if they should see
those without running *hxex*.

*hxex* can also write the results as a zip from the start: give the bundle a name ending in
`.zip` (`bundle = "results.zip"` in `[output]`, or `hxex run hxex.toml --bundle results.zip`).
The tables and figures are still written as ordinary folders beside it, named after the whole
file name: `results.zip_tables` and `results.zip_figures` (and `results.1.zip_tables` for an
earlier generation).

A zip cannot be changed in place, so review decisions cannot be saved into one; unpack it
first if you need to.
