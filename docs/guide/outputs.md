# Outputs: tables and figures

`hxex run` writes a results **bundle** (`[output] bundle`, such as `results`: a folder, or a `.zip`,
of Parquet tables with a `manifest.json`) and CSV tables beside it (`results_tables`).
`hxex report` draws the figures (`results_figures`) from the bundle, and writes the tables again;
`hxex analyse` writes the post-fitting analysis to a folder named after the bundle and the
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
│   ├── species_uptake.csv, compare_peptides.csv, ...
│   ├── woods_Mut_60s.pdf, ...
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

| Command | Writes | Afterwards out of date |
|---|---|---|
| `hxex run` | `results`, `results_tables` | `results_figures`, `results_analysis_...` |
| `hxex resume` | `results` (adds to it), `results_tables` | `results_figures`, `results_analysis_...` |
| `hxex report` | `results_tables` | |
| `hxex report --figures` | `results_figures` | |
| `hxex analyse` | `results_analysis_...` | |

`hxex resume` adds to the bundle and writes its tables again without asking: they belong to the
same results. After `hxex run` or `hxex resume`, *hxex* warns about every figures and analysis
folder made from other results, and `hxex info results` lists each folder as `up to date` or
`made from earlier results`. To bring them up to date:

```bash
hxex report hxex.toml --figures --overwrite
hxex analyse hxex.toml --overwrite
```

`hxex check hxex.toml` lists these folders under `outputs`: which exist, whether each was made
from the results in the bundle now, and which command would need `--overwrite` or
`--keep-previous`. They are information, not problems: the check still ends with `ok`, since each
command stops by itself, before doing any work.

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

| File | One row per | Holds |
|---|---|---|
| `series.csv` | series | sample, peptide, range, charge; spectra counted (flagged, rejected, multimodal); `controls` (measured, or why a fallback was used), `control_notes` (control warnings that leave it in use: `fd_disagrees`, `fd_multimodal` with whether its retention agrees with the other samples' (`retention agrees`, `retention differs`), or else with its other charge states' (`retention agrees with its other charge states`), or `retention unchecked`, `m0_multimodal`, `data_below_m0`, `fd_charges_disagree` (the peptide's charge states, whose fully deuterated controls are combined, differ by more than 0.1), `m0_replicate_left_out` and `fd_replicate_left_out` (a control replicate was left out of the average)), `retention` (deuterium kept), `control_fallback`, `rt` |
| `spectra.csv` | spectrum | the spectrum's identity (time, role, replicate, scan target for scan formats), `review_status`, `review_K` (a number of populations chosen in the review), `note` (the review's note), quality `flags`; selected K, `centroid` (and corrected), and each rank's mean and population. A copy of it is the starting point of the [user review table](project-file.md#the-user-review-table) |
| `components.csv` | spectrum and population | K, rank, mean, mean_se, corrected mean, population and their errors |
| `uptake.csv` | series, time (or control) and rank | the replicates averaged: `n_replicates`, `mean` (corrected when there are controls), `replicate_sd`, population. Rejected spectra are left out |
| `review.csv` | reviewed spectrum (only when there are any) | status, the number of populations chosen (`review_K`), note, date, stale, and source: the `review table` (`user_review_table.csv`), an `exclusion rule` of the project file, or a `review` decision stored in the bundle |

## Figures from `hxex report --figures`

`hxex run` draws no figures. `hxex report hxex.toml --figures` draws them from the results,
with the settings of the `[figures]` part of the project file; changing those settings needs
no new run, only `hxex report` again. (Given the bundle instead, `hxex report results
--figures` uses the usual settings and its own options.)

- `overview.pdf`: which K was selected and how often (per time, before and after review); the
  controls (retention against its limits, measured against fallback); the quality flags.
- One file per series, named by its label (sample, range, peptide or fragment, charge,
  condition), with a page per kind:
  - **spectra**: raw data, picked sticks and the fit, the picking window shaded;
  - **populations**: each population's mean and share over time (left out when every spectrum has
    K=1);
  - **uptake**: deuteration against time (or index with `time_scale = "order"`), corrected by
    default when there are controls. A rejected or excluded spectrum is drawn hollow and grey.

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
| `noise_outlier` | a replicate the replicate-based noise estimate left out |
| `mz_shift` | the series sits off its predicted m/z: a wrong sequence, modification, terminus or ion type |
| `poor_envelope` | the fragment envelope could not be measured well |

## Tables from `hxex analyse`

Every analysis table starts with `run` and `sample` (the run is blank for a single bundle).

**Peptides and residues**

| File | Holds |
|---|---|
| `species_uptake.csv` | per species (peptide or fragment, charges and replicates combined), time, K and rank: mean, fD and their errors, population, amides, `charge_trend`, `control_fallback` (corrected with a stand-in control: such series are left out unless `stand_in_controls = "include"` or the sample has no measured control), `fd_other_charge` (with `combine_charges = false` only: the only stand-in was a fully deuterated control from the peptide's other charge states, which the protection-factor fit treats as measured); and the number of populations chosen for the species at that time from all its spectra together: `best_set_K`, `best_set` (true on the rows of that description: filter on it for one description per peptide and time), `best_set_reason`, and `set_delta_criterion` (the evidence: negative favours two populations) |
| `left_out_spectra.csv` | the spectra the analysis left out whole (`[analysis.filters]`), with `reason`: `replicate outlier` (far from its other replicates), `above the amide count` (more deuterium than the peptide can hold, before the back-exchange correction), or `stand-in control` (its series has no measured fully deuterated control). To keep one, mark it `accepted` in the [user review table](project-file.md#the-user-review-table) |
| `residues.csv` | per residue, time, K and rank: fD from the overlapping peptides, redundancy, resolution, coverage |
| `coverage.csv` | per residue: redundancy, resolution, proline, covered |
| `populations.csv` | per residue and time: the K=1 and the two K=2 values, whether two populations are kept and, if not, why |
| `regions.csv` | per region and time: fD over the region's residues |

**Comparisons** (with a `reference`)

| File | Holds |
|---|---|
| `compare_peptides.csv` | per peptide and time: the difference from the reference, its error, t, p, the critical difference (`delta_crit`) and `significant`; for the population, the `separation` of its two populations, and `reason` says why a peptide was not compared (for example `not separated (reference)`: its two populations are too close to tell apart) |
| `compare_residues.csv` | per residue and time: difference, error, z, `significant` (at K=2, only where both samples keep two populations), and whether each side keeps two (`two_reference`, `two`) |
| `compare_population_counts.csv` | per peptide and time: whether each side is best described by one or two populations (`K_reference`, `K`), the evidence (`delta_criterion`: negative favours two), `differs`, and `borderline` (a difference with weak evidence) |
| `compare_populations.csv` | per residue and time: the low-D population against the reference's, with the critical difference (`delta_crit`) and `significant` |
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

| Figure | Shows |
|---|---|
| `coverage_<sample>` | one bar per peptide, coloured by fD, for every time and rank |
| `redundancy_resolution` | redundancy and resolution per residue |
| `choice_<sample>_<time>` | the residue profile: K=1, the K=2 ranks, the chosen description and the population (`profile_...` when there is no K=2) |
| `compare_residues_<time>`, `compare_populations_<time>` | each sample against the reference, by residue (see *Reading the residue comparison* below) |
| `woods_<sample>_<time>` | peptide differences from the reference at K=1, and where the number of populations differs (see *Reading the Woods plot* below) |
| `woods_K2_<sample>_<time>` | the two populations compared, where both samples have two |
| `butterfly_<sample>` | reference above, sample below, a line per time, differences underneath |
| `regions_<time>` | the regions' fD |
| `protection_<sample>`, `protection_difference_<sample>` | peptide protection factors, and their differences from the reference |
| `ladder_<parent>_<ion>_z<charge>` | the precursor, then for every ladder step the two fragments and their difference with its fit |
| `closure_<parent>_<n ion>+<c ion>` | the closure miss over time per split |
| `residue_rates_<sample>_<range>` | per-residue rates, amplitudes and, with `kint`, protection |

`[analysis] format` sets the file type (`pdf`, `png`, `svg`).

**Reading the Woods plot.** Each panel shows, for every peptide, a bar over the residues it
covers at its difference from the reference (sample minus reference), with its standard error as
a vertical line. A bar is coloured where the difference is significant (red: the sample is
higher; blue: lower): Welch's t-test on the replicate spectra gives p below `alpha`, and the
difference is larger than the critical difference, drawn as the dashed lines.

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

The population panel does not show the change in the share of the low-D population directly,
but that change multiplied by the peptide's separation: the difference in fD between its two
populations (the mean of the two samples'). The product is the deuterium, as a fraction of the
amides, that moved from one state to the other, so the panel is in the same units as the fD
panels and has one critical difference, the K=1 one. This is the same test as comparing the
change in share with the critical difference divided by the separation: a small change of share
matters in a peptide whose two states are far apart, and little where they are close.
`compare_peptides.csv` keeps the change in share itself (`delta`), with the `separation` and the
threshold it was judged by (`delta_crit`, the K=1 critical difference divided by the
separation).

**Reading the residue comparison.** `compare_residues_<time>` draws each sample against the
reference residue by residue, the reference in black and the sample in its colour. The average
deuteration (K=1, fDavg) is small circles on a line with a shaded ±1 standard-error band; the
line joins two residues only where one peptide measures both, so a gap means the peptides do not
connect them. Where a sample has two populations, fDlow is an up triangle and fDhigh a down
triangle, with their error bars. A residue has two populations for a sample when the peptides
whose spectra are best described by two populations carry at least half of its weight (and the
two values are separated, precise, and on either side of the average). Its fDlow and fDhigh, like
its average, come from every peptide covering it, so the three always describe the same
peptides. Residue values change only where a peptide starts or ends, so the
average runs in steps: the width of a step is the resolution the peptides allow.
`compare_populations_<time>` is drawn the same way, with the share of the low-D population
(`pop_low`) as up triangles: each sample where it has two populations, whether or not the other
sample has. The by-residue figure can show any of these values (`overlay` in `[analysis]`:
`low`, `high`, `avg`, `pop_low`, `pop_high`).

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
The tables and figures are still written as ordinary folders beside it.

A zip cannot be changed in place, so review decisions cannot be saved into one; unpack it
first if you need to.
