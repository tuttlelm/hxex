# The hxex results bundle (schema version 1)

A bundle holds everything needed to replot and review a run without refitting. It is a
directory (or a read-only zip of it) with a `manifest.json` and Parquet files (zstd). Every
number is stored exactly: float64, int64, bool or string, with list columns for ragged data
(`list<double>` and so on), so reading a table gives back the values the run produced. The
tables can be read with any Parquet reader (`pyarrow`, R `arrow::read_parquet`, Julia
`Parquet2.jl`); only the Python reader (`hxex.io.open_bundle`) is tested.

Written with `hxex.io.write_bundle(path, measurements=..., fit=..., ...)`, read with
`hxex.io.open_bundle(path)`. Writing is atomic (nothing is left behind on a failure), an
existing bundle is only replaced with `overwrite=True`, and `Bundle.verify()` checks the SHA-256
of every file. A bundle directory is replaced file by file: the files its manifest lists are
removed, the new ones moved in and the manifest written last, so files of your own in the
directory (notes, figures) stay; one named like a file of the new bundle is refused before
anything changes. An interrupted replacement leaves `manifest.json.replacing` in place of the
manifest: the directory does not open until the bundle is written again.

## What a results folder holds, and what a later run does with it

The bundle and the folders named after it (`results_tables`, `results_figures`, and the
earlier generations `results.1`, `results.1_tables`, `results.1_figures`, or `results.1.zip`
for a zip) are hxex's: a later `hxex run` replaces or renames them. Three kinds of content
live there and are treated differently:

- **Computed results** (the bundle's tables, the CSV tables, the figures) are replaced. A
  bundle directory is replaced file by file as above. A tables or figures folder keeps a
  listing of the files hxex wrote, `hxex_listing.json`; writing again (`hxex run`, `resume`,
  `report`, `write_tables`, `write_figures`) removes the listed files and writes the new ones,
  so the folder holds what the current bundle gives. A folder with files but no listing (one
  written before the listing existed) is written only if no file in it is named like a new
  one.
- **Annotations** (the review layer: your work, keyed by `spectrum_id`) are carried over to
  the new bundle, never silently dropped: an entry the new run cannot apply (its spectrum was
  dropped, its chosen K was not fitted) is kept with `stale = true`, its status and note on
  record and its chosen K ignored until you change the entry (see "The review layer").
- **Your own files** in these folders are never touched, but one named like a file hxex is
  about to write stops the run before anything changes; keep your own files elsewhere, or
  under other names.

`hxex run --overwrite` (or `[output] overwrite = true`) replaces in place; `--keep-previous`
(`keep_previous = true`) first moves the bundle, with its `_tables` and `_figures` folders, to
the next generation (`results.N`, `results.N_tables`, `results.N_figures`; N is one more than
any number ever used beside the bundle, so a name is never reused; `hxex.io.keep_previous`).
A tables or figures directory named explicitly in `[output]` is not moved (its hxex files are
replaced). `hxex info` lists the generations (`hxex.io.generations`) and the provenance below.

## Keys and conventions

- `spectrum_id` is the key of a spectrum in every table: `PROJECT:LOCAL_ID` (for example
  `myproj:0001-0006-PEPTIDE/60s-2-z2.csv`, or `bimodal:Angio_2_HI.xlsx#3`). It is unique within a
  bundle. `measurement` is the row number (0-based) of a measurement in `measurements.parquet`
  and joins the per-measurement tables.
- Time is in seconds (0 undeuterated, the user-set time for the fully deuterated control).
  `data_id` identifies a series (one peptide at one charge with its controls and time points).
- Intensities, amplitudes `A` and the noise scale are in the units of the data. The optimiser's
  own parameters (`log_A`, `n_raw`, `logit_p`) are in the noise-scaled units the fit works in.
- Per-measurement values that depend on K are lists of K values in the order of the original
  fit's components. Where a table pads to the largest K tried (`k_max`), the padding is NaN.
  A K the multimodality screen (`fit.screen`) did not fit for a measurement has NaN in every
  value of that K (its loss and criteria included) and is never its selected K.
- `mean` is `n * p` in deuterons and `population` (a component's share) is `A_k / sum(A)`.

## manifest.json

`kind` (`hxex-results-bundle`), `schema_version`, `created` (UTC), `project`, `notes`,
`versions` (python, hxex, numpy, scipy, torch, pyarrow, pyteomics), `k_values` (the K tried),
`n_measurements`, `pending` (the stages the run has still to add: `["bootstrap"]` for the
bundle a run writes as soon as its fits are done, which it rewrites complete at the end, or
`hxex.run.resume_bootstrap` completes if the run stopped during the bootstrap; empty or absent
when complete), `provenance` (null for a bundle written where there was none; for one that
replaced a bundle: `replaced` (its path), `replaced_created` and `replaced_hxex` (its stamp
and version), `at`, `moved_to` (the generation it was moved to with `keep_previous`, else
null), `settings_changed` (the dotted paths of the run settings that differ, `fit.k_max`,
and `spectra_source` when other files were read or read differently) and `review`
(`carried`, `stale`: the review entries carried over and how many are stale);
`hxex.io.provenance_of` builds it, and a resumed bundle keeps it), `configs` (`fit`, `bootstrap`, `picker`, `controls`, `quality`, `envelope`, `run` (the whole run configuration, including the modification registry): the settings as written by the
`to_dict` of each config; the fit config includes the noise settings and the device;
`spectra_source`: how the raw spectra were read, recorded whether or not they are stored, or
null: for `read_mzml` the reader, project, runs (path, sample, rep, time, time offset, and
the scan conditions the run sets instead of the targets'),
targets (entry, MS level, activation, precursor, retention window, filter text), mzML settings,
picker settings, modification registry, `series` (each `data_id` with its sample and target
position, which ties a result to its precursor and activation), `record_paths` (`path` as
given, `relative` to a data root, or `name` only, for results to be shared) and `files` (each
run's file `name`, `size`, `checksum` and `algorithm`: the indexed mzML file's own SHA-1 when
it has one, else SHA-256); `read_thermo` records the same with `reader` `thermo`, the
RawFileReader version as `thermo_reader`, and each `.raw` file's SHA-256.
`hxex.io.read_mzml_source` (or `read_thermo_source`) reads it again, finding the files by
path, relative to a data root, or by name in a directory; a file whose size or checksum
differs is read with a warning, or refused with `on_change="error"`, and its new entry in
`files` keeps the old identity under `previous`; for `read_spectra` (HX-Express, SpecExport,
pooled) the format, project, data directory, entries, filters, control times and `files` (each
entry's input: an HX-Express file, or a SpecExport folder identified as a whole), repeated by
`hxex.io.read_spectra_source`; `hxex.io.read_source` repeats either kind). A scan-file
record also holds `windows`, the m/z range each target's scans were cut to (the picker's
predicted bins plus `mz_margin`; a rerun from stored spectra whose picker reaches beyond it
warns). Spectra taken from an earlier bundle or spectra file (`[input] spectra`) keep that
read's record with `stored_in`, the path they were taken from. Then
`files`: for each file its `rows`, `bytes`, `sha256` (and `derived: true` for the tidy table).

## Tables

| file | one row per | content |
|---|---|---|
| `measurements.parquet` | measurement | what was fitted: `measurement`, `spectrum_id`, `data_id`, `time`, `n_max`, `bins` (n_deut), `observed` (intensity), `shape` (the natural-abundance envelope used), `references` (indices of the noise-reference spectra, null = the spectrum itself) |
| `fit.parquet` | measurement | `measurement`, `spectrum_id`, `data_id`, `time`, `selected_K`, `obs_scale` (the noise scale the spectrum was divided by), `sample_size` (the N of the BIC), `n_max`, `noise_kind` (`gaussian` or `poisson`), `score` (selection criterion for each K tried, inf where the population filter rejected it) |
| `fit_k.parquet` | measurement and K | `measurement`, `fit_K`, `A`, `n`, `p` (lists of K), `log_A`, `n_raw`, `logit_p` (the optimiser's parameters), `nll`, `bic`, `aic`, `passes_min_pop` |
| `noise_estimate.parquet` | measurement | when the noise was estimated from replicates: `gain`, `estimated`, `reason` (why a series was not estimated), `score` (consistency with replicates), `outlier` (flagged), `n_replicated` |
| `bootstrap.parquet` | measurement | at the selected K: `selected_K`, `n_success`; `A_se`, `n_se`, `p_se`, `mean_se`, `population_se` (lists of `k_max`); `A_ci`, `n_ci`, `p_ci`, `mean_ci`, `population_ci` (lists of `k_max * 2`: low and high of each component); `curve_lo`, `curve_hi` (the band of the total fitted curve, one value per bin); `component_lo`, `component_hi` (the same band for each population, `K * bins` values: population by population in the order of the original fit; absent in bundles written before they were added) |
| `bootstrap_draws.parquet` | measurement | optional individual fits at the selected K: `A`, `n`, `p` (lists of `n_boot * k_max`, draw-major), `ok` (list of `n_boot`) |
| `bootstrap_k.parquet` | measurement and K | the fixed-K bootstrap (`BootstrapConfig.ks`): the columns of `bootstrap` plus `fit_K`; every measurement has a row for each K requested, and `selected_K` equals `fit_K` |
| `bootstrap_k_draws.parquet` | measurement and K | the draws of the fixed-K bootstrap when kept: the columns of `bootstrap_draws` plus `fit_K` |
| `binned.parquet` | spectrum | the peak-picked table: the index columns below plus `mz` (predicted), `intensity`, `n_deut`, `peak_mz` (where the peak was found, NaN for an empty bin) (lists per bin), `mz_offset_ppm` (the spectrum's intensity-weighted median peak offset; both NaN in bundles written before hxex recorded them), `env_width`, `env_symm`, `max_namides`, `threshold`, and the flags `empty`, `no_signal`, `capped` |
| `spectra.parquet` | spectrum | optional raw spectra: the index columns plus `mz`, `intensity` (lists) |
| `review.parquet` | reviewed spectrum | `spectrum_id`, `status` (`accepted` or `rejected`, or empty), `k_override` (0 = none), `note`, `updated` (UTC time), `stale` (the entry does not apply to this fit; absent in bundles written before 2026-10-17, meaning false) |
| `quality.parquet` | measurement | when quality flags were assessed: `deviance` (reduced deviance of the selected-K fit), `deviance_rel` (to the series median), `misfit` (`sum((obs - fit)^2) / sum(obs^2)` of that fit) and `misfit_rel` (to the series median; the `poor_fit` statistic; absent in bundles written before it), `intensity` (total), `intensity_rel` (to the other replicates at the same time), `mz_offset` (the series' median m/z offset minus the median over all series, ppm), NaN where not assessed; `poor_fit`, `low_intensity`, `noise_outlier`, `mz_shift`, `poor_envelope` (the series' ion-type envelope was not measured well, `envelopes.parquet`; absent in bundles written before it) and `flagged` (any of them) |
| `envelopes.parquet` | series (`data_id`) | when the run used an ion-type mixture or an inferred envelope: `mode`, `source` (`reference`, `shared` (one envelope for the conditions of one ion, `envelope.share`), `fallback` or `theory`), `weights` (of the ion types, lightest first), `residual` (on the bins the envelope is used with), `n_reference` (the series' own reference spectra with signal), `good`, `fit_shape` (the envelope of the fit; the same shape is in the series' `measurements` rows), and for shared groups `own_residual` (the shared envelope against the series' own references; NaN without any) and `agrees` (whether its references agree with the other conditions'; both kept when the group then fell back; absent in bundles written before them) |
| `controls.parquet` | series (`data_id`) | when controls were given: `n_amides`, `n_undeut`, `n_fd` (replicates used), `m0_measured`, `mfd_measured` (replicate means of the K=1 fits, NaN if missing), `m0`, `mfd` (the values used) with `m0_se`, `mfd_se` (NaN: one replicate; 0: theoretical; for a theoretical back-exchange fallback the fit's scatter per amide times `n_amides`), `retention`, `fallback`, `flags` (`;`-separated: `m0_missing`, `m0_high`, `fd_missing`, `retention_low`, `retention_high`, `no_amides`, `m0_unreplicated`, `fd_unreplicated`, `fd_median`, `fd_off_model`, `fd_kint`); with `fd_fallback = "kint"` the fitted model (`a`, `t0`, `b_rt` and `median_rt` when it used retention times, else null, `n_series`, `n_rejected`, `sd_per_amide`, `quench`) is JSON in the table's schema metadata (`fd_model`) |
| `components.parquet` | measurement, K and component | the tidy table of every fit with its intervals (from `bootstrap` for the selected K and `bootstrap_k` for the others), for other tools; derived (not read back). `mean` is the uncorrected mean (deuterons beyond natural abundance). With controls it also has `mean_corrected`, `mean_corrected_se`, `mean_corrected_ci_lo`, `mean_corrected_ci_hi` (the controls' uncertainty included), `mean_corrected_control_se` (that part alone) and `control_fallback` |

Index columns of `binned` and `spectra`: `spectrum_id`, `sample`, `peptide` (modX labels such as
`acK`), `peptide_range`, `file`, `label`, `modification`, `start_seq`, `end_seq`, `charge`,
`rep`, `data_id`, `time`, `rt` (the chromatographic retention time in minutes, NaN when unknown;
from `[input] retention_times`, a peptide table's `rt` column or the scans of a fixed-time run;
absent in bundles written before 2026-10-15, read as unknown).

## The review layer

`review.parquet` is the only table that changes after writing (`Bundle.set_review`, directory
bundles only; the manifest is updated with the new checksum). It never changes the fit: use
`ReviewTable.effective_K(fit)` and `status_of(fit)` for the reviewed view. A chosen K must be
one that was fitted. A run that replaces a reviewed bundle carries the review over and marks
the entries it cannot apply `stale` (the spectrum is not in the new fit, or the chosen K was
not fitted): the status and note stay on record (`status_of` still reports the status when
the spectrum is there), `effective_K` ignores a stale entry's chosen K, and `hxex info` counts
them. `ReviewTable.update` makes an entry current again, so a stale entry you change is
checked like any other when stored (an entry for a spectrum the bundle does not have is
refused; clearing it removes it). `ReviewTable.mark_stale(fit)` recomputes the marker.

## Versioning

`schema_version` is an integer (2 since 2026-10-13: a component's share is `population`, its
bootstrap columns `population_se` and `population_ci`, where version 1 had `frac_se` and
`frac_ci`; hxex reads both). New optional tables and columns are additions and do not change
it; a reader ignores tables and columns it does not know. An incompatible change raises it, and
`open_bundle` refuses a bundle with a higher version than it knows, with a message to upgrade.
Control identification and back-exchange values (Phase 3) will be added as further tables.

## Raw spectra without a bundle

`hxex.io.save_spectra(spectra, path)` writes a `SpectrumSet` to one Parquet file and
`hxex.io.load_spectra(path, ids=None)` reads it back exactly, with no fit and no peak picking.
The layout is that of `spectra.parquet` above (the same columns and types), so the two are
interchangeable: `load_spectra` also reads the `spectra.parquet` of a bundle. The Parquet file
metadata (keys prefixed `hxex.`) records `kind` (`hxex-spectra`), `schema_version`, `project`,
`created`, `hxex_version`, `n_spectra` and `content_sha256` (a hash of the ids, times and every
m/z and intensity, checked on a full load; a partial load by `ids` cannot be checked) and,
when the spectra carry one, `source` (JSON, as `spectra_source` in the bundle manifest).
Writing is atomic and replaces an existing file only with `overwrite=True`. These two functions
need numpy and pyarrow only (not torch). `spectra_info(path)` returns the metadata.
