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
of every file.

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
- `mean` is `n * p` in deuterons and `population` (a component's share) is `A_k / sum(A)`.

## manifest.json

`kind` (`hxex-results-bundle`), `schema_version`, `created` (UTC), `project`, `notes`,
`versions` (python, hxex, numpy, scipy, torch, pyarrow, pyteomics), `k_values` (the K tried),
`n_measurements`, `pending` (the stages the run has still to add: `["bootstrap"]` for the
bundle a run writes as soon as its fits are done, which it rewrites complete at the end, or
`hxex.run.resume_bootstrap` completes if the run stopped during the bootstrap; empty or absent
when complete), `configs` (`fit`, `bootstrap`, `picker`, `controls`, `quality`, `envelope`, `run` (the whole run configuration, including the modification registry): the settings as written by the
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
`hxex.io.read_spectra_source`; `hxex.io.read_source` repeats either kind), and
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
| `review.parquet` | reviewed spectrum | `spectrum_id`, `status` (`accepted` or `rejected`, or empty), `k_override` (0 = none), `note`, `updated` (UTC time) |
| `quality.parquet` | measurement | when quality flags were assessed: `deviance` (reduced deviance of the selected-K fit), `deviance_rel` (to the series median), `misfit` (`sum((obs - fit)^2) / sum(obs^2)` of that fit) and `misfit_rel` (to the series median; the `poor_fit` statistic; absent in bundles written before it), `intensity` (total), `intensity_rel` (to the other replicates at the same time), `mz_offset` (the series' median m/z offset minus the median over all series, ppm), NaN where not assessed; `poor_fit`, `low_intensity`, `noise_outlier`, `mz_shift` and `flagged` (any of them) |
| `envelopes.parquet` | series (`data_id`) | when the run used an ion-type mixture or an inferred envelope: `mode`, `source` (`reference`, `fallback` or `theory`), `weights` (of the ion types, lightest first), `residual`, `n_reference`, `good`, `fit_shape` (the envelope of the fit; the same shape is in the series' `measurements` rows) |
| `controls.parquet` | series (`data_id`) | when controls were given: `n_amides`, `n_undeut`, `n_fd` (replicates used), `m0_measured`, `mfd_measured` (replicate means of the K=1 fits, NaN if missing), `m0`, `mfd` (the values used) with `m0_se`, `mfd_se` (NaN: one replicate; 0: theoretical), `retention`, `fallback`, `flags` (`;`-separated: `m0_missing`, `m0_high`, `fd_missing`, `retention_low`, `retention_high`, `no_amides`, `m0_unreplicated`, `fd_unreplicated`) |
| `components.parquet` | measurement, K and component | the tidy table of every fit with its intervals (from `bootstrap` for the selected K and `bootstrap_k` for the others), for other tools; derived (not read back). `mean` is the uncorrected mean (deuterons beyond natural abundance). With controls it also has `mean_corrected`, `mean_corrected_se`, `mean_corrected_ci_lo`, `mean_corrected_ci_hi` (the controls' uncertainty included), `mean_corrected_control_se` (that part alone) and `control_fallback` |

Index columns of `binned` and `spectra`: `spectrum_id`, `sample`, `peptide` (modX labels such as
`acK`), `peptide_range`, `file`, `label`, `modification`, `start_seq`, `end_seq`, `charge`,
`rep`, `data_id`, `time`.

## The review layer

`review.parquet` is the only table that changes after writing (`Bundle.set_review`, directory
bundles only; the manifest is updated with the new checksum). It never changes the fit: use
`ReviewTable.effective_K(fit)` and `status_of(fit)` for the reviewed view. A chosen K must be
one that was fitted.

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
