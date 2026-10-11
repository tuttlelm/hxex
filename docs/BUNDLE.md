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
  `tubB5:0001-0006-MRECIS/60s-2-z2.csv`, or `bimodal:Angio_2_HI.xlsx#3`). It is unique within a
  bundle. `measurement` is the row number (0-based) of a measurement in `measurements.parquet`
  and joins the per-measurement tables.
- Time is in seconds (0 undeuterated, the user-set time for the fully deuterated control).
  `data_id` identifies a series (one peptide at one charge with its controls and time points).
- Intensities, amplitudes `A` and the noise scale are in the units of the data. The optimiser's
  own parameters (`log_A`, `n_raw`, `logit_p`) are in the noise-scaled units the fit works in.
- Per-measurement values that depend on K are lists of K values in the order of the original
  fit's components. Where a table pads to the largest K tried (`k_max`), the padding is NaN.
- `mean` is `n * p` in deuterons and `population fraction` is `A_k / sum(A)`.

## manifest.json

`kind` (`hxex-results-bundle`), `schema_version`, `created` (UTC), `project`, `notes`,
`versions` (python, hxex, numpy, scipy, torch, pyarrow, pyteomics), `k_values` (the K tried),
`n_measurements`, `configs` (`fit`, `bootstrap`, `picker`, `controls`: the settings as written by the
`to_dict` of each config; the fit config includes the noise settings and the device), and
`files`: for each file its `rows`, `bytes`, `sha256` (and `derived: true` for the tidy table).

## Tables

| file | one row per | content |
|---|---|---|
| `measurements.parquet` | measurement | what was fitted: `measurement`, `spectrum_id`, `data_id`, `time`, `n_max`, `bins` (n_deut), `observed` (intensity), `shape` (the natural-abundance envelope used), `references` (indices of the noise-reference spectra, null = the spectrum itself) |
| `fit.parquet` | measurement | `measurement`, `spectrum_id`, `data_id`, `time`, `selected_K`, `obs_scale` (the noise scale the spectrum was divided by), `sample_size` (the N of the BIC), `n_max`, `noise_kind` (`gaussian` or `poisson`), `score` (selection criterion for each K tried, inf where the population filter rejected it) |
| `fit_k.parquet` | measurement and K | `measurement`, `fit_K`, `A`, `n`, `p` (lists of K), `log_A`, `n_raw`, `logit_p` (the optimiser's parameters), `nll`, `bic`, `aic`, `passes_min_pop` |
| `noise_estimate.parquet` | measurement | when the noise was estimated from replicates: `gain`, `estimated`, `reason` (why a series was not estimated), `score` (consistency with replicates), `outlier` (flagged), `n_replicated` |
| `bootstrap.parquet` | measurement | at the selected K: `selected_K`, `n_success`; `A_se`, `n_se`, `p_se`, `mean_se`, `frac_se` (lists of `k_max`); `A_ci`, `n_ci`, `p_ci`, `mean_ci`, `frac_ci` (lists of `k_max * 2`: low and high of each component); `curve_lo`, `curve_hi` (the band of the total fitted curve, one value per bin) |
| `bootstrap_draws.parquet` | measurement | optional individual fits at the selected K: `A`, `n`, `p` (lists of `n_boot * k_max`, draw-major), `ok` (list of `n_boot`) |
| `binned.parquet` | spectrum | the peak-picked table: the index columns below plus `mz` (predicted), `intensity`, `n_deut` (lists per bin), `env_width`, `env_symm`, `max_namides`, `threshold`, and the flags `empty`, `no_signal`, `capped` |
| `spectra.parquet` | spectrum | optional raw spectra: the index columns plus `mz`, `intensity` (lists) |
| `review.parquet` | reviewed spectrum | `spectrum_id`, `status` (`accepted` or `rejected`, or empty), `k_override` (0 = none), `note`, `updated` (UTC time) |
| `controls.parquet` | series (`data_id`) | when controls were given: `n_amides`, `n_undeut`, `n_fd` (replicates used), `m0_measured`, `mfd_measured` (replicate means of the K=1 fits, NaN if missing), `m0`, `mfd` (the values used) with `m0_se`, `mfd_se` (NaN: one replicate; 0: theoretical), `retention`, `fallback`, `flags` (`;`-separated: `m0_missing`, `m0_high`, `fd_missing`, `retention_low`, `retention_high`, `no_amides`, `m0_unreplicated`, `fd_unreplicated`) |
| `components.parquet` | measurement, K and component | the tidy table of every fit with its intervals, for other tools; derived (not read back). `mean` is the uncorrected mean (deuterons beyond natural abundance). With controls it also has `mean_corrected`, `mean_corrected_se`, `mean_corrected_ci_lo`, `mean_corrected_ci_hi` (the controls' uncertainty included), `mean_corrected_control_se` (that part alone) and `control_fallback` |

Index columns of `binned` and `spectra`: `spectrum_id`, `sample`, `peptide` (modX labels such as
`acK`), `peptide_range`, `file`, `label`, `modification`, `start_seq`, `end_seq`, `charge`,
`rep`, `data_id`, `time`.

## The review layer

`review.parquet` is the only table that changes after writing (`Bundle.set_review`, directory
bundles only; the manifest is updated with the new checksum). It never changes the fit: use
`ReviewTable.effective_K(fit)` and `status_of(fit)` for the reviewed view. A chosen K must be
one that was fitted.

## Versioning

`schema_version` is an integer. New optional tables and columns are additions and do not change
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
m/z and intensity, checked on a full load; a partial load by `ids` cannot be checked).
Writing is atomic and replaces an existing file only with `overwrite=True`. These two functions
need numpy and pyarrow only (not torch). `spectra_info(path)` returns the metadata.
