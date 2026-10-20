# Outputs: tables and figures

`hxex run` writes a results **bundle** (`[output] bundle`, such as `results`: a folder, or a `.zip`,
of Parquet tables with a `manifest.json`), CSV tables beside it (`results_tables`) and, when asked,
figures (`results_figures`). `hxex report` writes the tables and figures again from a bundle;
`hxex analyse` writes the post-fitting analysis to a folder named after the bundle and the
reference: `results_analysis_WT` for comparisons with WT (`results_analysis` without a
reference), so analyses with different references never replace each other.

These folders, and their earlier generations (`results.1`, `results.1_tables`,
`results.1_analysis_WT`, ...), belong to *hxex*: a later run refuses to write over them unless given
`--overwrite` (replace them; the review is carried over) or `--keep-previous` (move them to the
next generation first). Running `hxex analyse` again with the same reference replaces the
earlier analysis in its folder. In all of them, *hxex* removes only the files it wrote itself. Keep your own files
elsewhere. The bundle's own format is described in [BUNDLE.md](../BUNDLE.md).

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
| `series.csv` | series | sample, peptide, range, charge; spectra counted (flagged, rejected, multimodal); `controls` (measured, or why a fallback was used), `retention` (deuterium kept), `control_fallback`, `rt` |
| `spectra.csv` | spectrum | the spectrum's identity (time, role, replicate, scan target for scan formats), `review_status`, `review_K` (a number of populations chosen in the review), quality `flags`; selected K, `centroid` (and corrected), and each rank's mean and population |
| `components.csv` | spectrum and population | K, rank, mean, mean_se, corrected mean, population and their errors |
| `uptake.csv` | series, time (or control) and rank | the replicates averaged: `n_replicates`, `mean` (corrected when there are controls), `replicate_sd`, population. Rejected spectra are left out |
| `review.csv` | reviewed spectrum (only when there are any) | status, the number of populations chosen (`review_K`), note, date, stale, and source: an `exclusion rule` of the project file, or a `review` decision stored in the bundle |

## Figures from `hxex run` and `hxex report --figures`

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
fallback); `[run.plots] series = "all"` or `hxex report --figures --series all` draws them all.
`format` is `pdf` (pages in one file), `png` or `svg` (one file per page).

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
| `species_uptake.csv` | per species (peptide or fragment, charges and replicates combined), time, K and rank: mean, fD and their errors, population, amides, `charge_trend` |
| `residues.csv` | per residue, time, K and rank: fD from the overlapping peptides, redundancy, resolution, coverage |
| `coverage.csv` | per residue: redundancy, resolution, proline, covered |
| `populations.csv` | per residue and time: the K=1 and the two K=2 values, whether two populations are kept and, if not, why |
| `regions.csv` | per region and time: fD over the region's residues |

**Comparisons** (with a `reference`)

| File | Holds |
|---|---|
| `compare_peptides.csv` | per peptide and time: the difference from the reference, its error, t, p, the critical difference and `significant` |
| `compare_residues.csv` | per residue and time: difference, error, z, `significant` |
| `compare_populations.csv` | per residue and time: the low-D population against the reference's |
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
| `compare_residues_<time>`, `compare_populations_<time>` | each sample against the reference, by residue |
| `woods_<sample>_<time>` | peptide differences from the reference |
| `butterfly_<sample>` | reference above, sample below, a line per time, differences underneath |
| `regions_<time>` | the regions' fD |
| `protection_<sample>`, `protection_difference_<sample>` | peptide protection factors, and their differences from the reference |
| `ladder_<parent>_<ion>_z<charge>` | the precursor, then for every ladder step the two fragments and their difference with its fit |
| `closure_<parent>_<n ion>+<c ion>` | the closure miss over time per split |
| `residue_rates_<sample>_<range>` | per-residue rates, amplitudes and, with `kint`, protection |

`[analysis] format` sets the file type (`pdf`, `png`, `svg`).

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
