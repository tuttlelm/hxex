# hxex history: findings and measurements

The dated record of findings and measurements moved verbatim from `docs/PLAN.md`; sections cited
here but not found are there. Read the relevant item when a task needs the evidence for a decision.

## Plan header (replaced)

Status: draft, revised through clarifications. Phase 0 scaffold created 2026-10-09 (see
"Phase 0 status").

## Decisions so far

- **Name, repository and hosting:** the package is called **`hxex`** (`pip install hxex`,
  `import hxex`), in a **new repository**, hosted at github.com/tuttlelm/hxex. On
  2026-10-09 the name was not registered on PyPI and the GitHub repository did not exist
  publicly (a GitHub 404 could also mean a private repository); the PyPI name should be
  reserved early in Phase 0. Existing pyHXExpress notebooks use `import hxex_updating as
  hxex`; code that mixes the two should rename that alias to avoid confusion with the new
  package. The prototype (the single-file fitting script `fit_multimodal_binomial.py` and its
  test notebook, in a separate working folder) stays where it is, with its Python 3.10
  environment, Jupyter kernel and project notes untouched, as the reference and regression
  source. The new repository is in a **different parent folder** from the prototype
  (confirmed), so no rename was needed. This plan lives in the new repository as
  `docs/PLAN.md`.
- **Minimum Python version: 3.11** (`requires-python = ">=3.11"`), with CI on 3.11 to 3.14
  (and 3.15 once it is final). Python 3.10 reaches end of life in October 2026, and PyTorch
  2.15 (scheduled for late October 2026) requires 3.11 or newer. Phase 0 must confirm that
  `pyteomics`, `pyarrow`, the isotope-distribution library and PyTorch all install on 3.11 to
  3.14, and the current development environment (`.venv`, Python 3.10) needs rebuilding on
  3.11 or newer.

## Evidence behind the "new package, port selectively" choice

Read from pyHXExpress (`hxex_updating.py`, 2,398 lines, 32 top-level functions):

- The fitting is one function, `run_hdx_fits` (about 870 lines), fitting one spectrum at a
  time with `scipy.optimize.curve_fit` or `differential_evolution`, interwoven with
  plotting, bootstrap and file output.
- Configuration is a global `config` module (194 references) with about 35 `config_*.py`
  variants in the folder.
- There is no version control history, and the folder holds six or more forked copies of the
  main module. Only a minimal test setup exists.
- The readers, peak picker, mzML helpers, fragments module and plotting already exist and
  encode instrument-specific knowledge.

Reader and picker behaviour (from reading the code, not yet ported):

- `read_hexpress_data`: Excel or CSV, header row of time labels (`undeut...`, `TD...`,
  `60 s`, `5 min`, ...), one (m/z, intensity) column pair per time point; time units and
  replicate counting are parsed from the labels.
- `read_specexport_data`: one CSV per spectrum; time, a `Non` / `Full` prefix and the
  replicate are encoded in the file name.
- `peak_picker`: predicted m/z per deuterium count (ExMS isotope spacing), a +/-ppm window,
  tallest stick in the window with shoulder rejection, a noise threshold (absolute or a
  percentage of the maximum), and zero-filling to stop. It produces the binned sticks that
  were fitted in the tests.
- All of them read global `config`, build results by repeated DataFrame concatenation, and
  mix file parsing with peak picking. The HX-Express reader also has a test-data branch that
  reads a solutions file. A port should separate these concerns.

## Results bundle and replotting

Requirement (added after review): it must be possible to read in previous results and
generate plots of the raw data and the fits, without refitting.

What pyHXExpress does today (from `hxex_updating.py` and an example output folder):

- `plot_spectrum(deutdata, rawdata, fit_params, ...)` redraws a single spectrum (raw data,
  picked peaks, fit, optional residuals) from three tables.
- A run writes: `data_fits<date>.csv` (one row per spectrum: overall centroid, number of fit
  populations, p-value, and per population `centroid_k`, `Dabs_k`, `Dabs_std_k`, `pop_k`,
  `pop_std_k`), `fitparamsAll_asrun<date>.csv` (fit parameter vector, `ncurves`, `nboot`,
  `rss`), `alldeutdata<date>.csv` (the binned sticks), `allrawdata<date>.zip` (raw m/z and
  intensity), `metadf_asrun<date>.csv` (peptide table), the parameter file
  `hdxms_params<date>.py`, and PDFs.
- The fitted **curves are not stored**: they are recomputed from the parameters and the
  natural-abundance shape rebuilt from the sequence.

Design for the new package:

- A **results bundle** (a directory or zip of open-format files plus a JSON manifest with a
  schema version) containing everything needed to replot:
  - the binned data and, optionally, the raw spectra (see "Raw spectrum storage");
  - peptide metadata: sequence with PTMs, charge, ion type, modifications, replicate, time;
  - the fitted components per measurement and per K (amplitude in intensity units, n, p, mean,
    population fraction), the selected K, and the BIC/AIC values for each K (see "Saving all
    fits");
  - the noise setting that was used (model and sigma per measurement), the control
    identification, the back-exchange values and the fallback flags;
  - bootstrap summaries (standard errors and CIs on A, n, p, mean and fraction) and the
    per-bin CI band of the total fit, so error bars and bands can be redrawn without the
    bootstrap samples; the individual bootstrap fits are an option (see "Saving all fits");
  - the natural-abundance shape used for each measurement (small), so replotting does not
    depend on the chemistry code version;
  - the configuration and package versions;
  - a **review layer**: per-spectrum flags and notes (accept or reject, comments, manual
    overrides such as a chosen K) that are stored with the bundle and need no refitting; see
    "GUI".
- A **loader** returns the same in-memory objects the fit produces, and the **`report`
  module plots from those objects only**, so a fresh fit and a loaded bundle give identical
  plots. Options include per-sample or per-spectrum selection, raw spectrum versus binned
  sticks, residuals, normalisation and error bars.
- Round-trip tests: fit, save, load and plot must reproduce the direct plot's data exactly.
- Only bundles written by the new package need to be readable at first. A loader for
  pyHXExpress's own outputs is a later project goal (see "Later goals").

### Raw spectrum storage

pyHXExpress keeps raw spectra in a pandas DataFrame (one row per m/z point, with the text
metadata repeated on every row) and saves them as a zipped CSV. Measured on the real raw data
of the test set (566 spectra, 496,708 points, median 893 points per spectrum):

| Layout | Size | Load all | One spectrum | One sample |
|---|---|---|---|---|
| A. pandas CSV in a zip (current) | 5.70 MB | 0.35 s | 384 ms | 358 ms |
| B. Parquet long table, all metadata columns | 1.37 MB | 0.07 s | 25 ms | 31 ms |
| C. Parquet long table, minimal plus index | 1.08 MB | 0.00 s | 2.6 ms | 4.0 ms |
| D. Parquet, one row per spectrum (list columns) | 1.01 MB | 0.01 s | 3.3 ms | 4.0 ms |
| E1. NumPy npz, ragged, compressed | 0.97 MB | 0.01 s | 13.8 ms | 117 ms |
| E2. NumPy .npy, ragged, memory-mapped | 7.95 MB | 0.00 s | 0.3 ms | 6.2 ms |
| F. HDF5, ragged, gzip | 1.67 MB | 0.02 s | 0.7 ms | 15.5 ms |

In memory the same data take 174 MB as a pandas DataFrame and 8 MB as flat NumPy arrays with
offsets. Intensity as float32 would change 7.2% of the values (maximum relative error
4.8e-8), so full precision stays the default and float32 is an explicit option. Timings were
taken on one machine with pyarrow 25.0 and h5py 3.16; they show relative differences, not
guarantees.

Decision (confirmed):

- **In memory:** a lightweight spectrum set made of flat NumPy arrays (m/z, intensity) with an
  offsets array and a small typed index (sample, peptide, charge, time, replicate, ...). No
  pandas DataFrames in the core; pandas, if used at all, only at the edges for convenience.
- **On disk:** Parquet (zstd), **one row per spectrum with list columns** (layout D): metadata
  is stored once per spectrum, files are about 6 times smaller than the zipped CSV, a
  spectrum loads in a few milliseconds, row groups allow chunked writing for very large runs,
  and the format is open and readable from R and Julia as well as Python. The same format is
  used for the binned table and the results tables, so a single dependency (pyarrow) covers
  the bundle.
- **Considered and not chosen:** ragged `.npy` arrays plus a JSON index (layout E2). Fastest
  random access and no new dependency, but 8 times larger, and not language-neutral or typed.
- At 10,000 spectra of this size the compressed raw data would be about 18 MB (extrapolated
  from the 566-spectrum measurement).
- The bundle stores extracted spectra windows, not whole LC-MS runs; very large profile data
  would use the same layout but may call for a size limit or a path reference.

### Saving all fits

The bundle supports an option to save **all** fits:

- **All K:** the parameters of every fitted model K = 1 .. K_max per measurement, with their
  NLL, BIC and AIC, a flag for the selected K, and whether it passed the minimum-population
  rule. The engine already computes all of these. This is small and should be saved by
  default.
- **Bootstrap fits (option):** the individual bootstrap fits per measurement (component
  parameters for each bootstrap sample, plus a success flag), not only the summaries and
  bands. The bootstrap refits each measurement at its **selected K only** (decided), so
  these are the fits at the selected K. For scale: 10,000 measurements x 100 bootstrap
  samples x up to 3 components x 3 parameters is about 9 million numbers (about 72 MB at
  float64 before compression; computed, not measured).

### Later goals

- A loader for pyHXExpress's own outputs (`alldeutdata`, `allrawdata`, `data_fits`,
  `fitparamsAll`, `metadf`), so older results can be plotted with the new plotting.
- **HX-Express output** (`hxex.io`): a writer for the HX-Express format, as in pyHXExpress's
  `export_to_hxexpress(rawdata, metadf, save_xls, removeUNTDreps)`. For this output the spectra
  written are the **peak-picked** spectra (a `BinnedSet`: the predicted m/z and intensity of
  each bin); that is a property of this format's output, not a rule for exports in general
  (other outputs are defined separately). pyHXExpress's function has the layout but writes the
  raw spectra. One spreadsheet per sample, peptide and charge, with an (m/z, intensity) column
  pair per spectrum, labelled `undeut<rep>`, `TD<rep>` and `<time> sec`, where a replicate is
  encoded as +0.01 s per replicate (`1.01 sec` is the second spectrum at 1 s); columns are
  ordered undeuterated, then times, then fully deuterated, and a `removeUNTDreps` option keeps
  only the first undeuterated and fully deuterated spectrum (HX-Express does not use the
  replicates). pyHXExpress writes the file name `SAMPLE` + range + `-PEPTIDE` + `z` + charge,
  which is not the name its own HX-Express reader expects
  (`SAMPLE_START-END-PEPTIDE-zCHARGE-text.xlsx`); the new writer should use the name
  `read_hxexpress_file` and `scan_hxexpress` read, so a round trip works and is tested. The
  fully deuterated time comes from the user's setting. Flagged or dropped spectra are not
  written (drop them with `BinnedSet.select` first). Writing the fitted results in
  pyHXExpress's CSV layouts (`data_fits`, `fitparamsAll`) is a separate item that goes with the
  loader above.

## Sequence syntax and PTMs

Differences found in the existing code, to be settled (compatibility versus correctness):

1. **Amide count with modified residues.** `count_amides` counts uppercase letters for the
   residue count but slices the sequence by **characters** for the N-terminal proline rule.
   With `Nterm_subtract = 1` it agreed with a residue-based count on every case tested. With
   `Nterm_subtract = 2` and a modified residue among the first residues it differs (for
   example `acKPEPTIDE`: 4 by `count_amides`, 5 by a residue-based count).
2. **Fragment generation with modified residues.** `hxex_fragments.py` predates PTM support
   and has not been updated for it (confirmed by the author), so this is an expected gap, not
   a regression. `get_fragments` slices by characters. Verified by running it on `PEPacKTIDE` (8 residues, 10 characters): b4 gives
   `PEPa` (should be `PEPacK`) and y5 gives `KTIDE` (should be `acKTIDE`), while the range
   columns use residue counts. Plain sequences are unaffected.

The new code uses residues. Where that changes a result relative to pyHXExpress it is a
documented, deliberate difference, and the golden tests against `hxex_updating.py` cover the
cases where the two agree.

## Ion types (fragment ions)

What pyHXExpress does today (from reading `hxex_updating.py` and `hxex_fragments.py`):

- `peak_picker` already uses the ion type: the undeuterated m/z comes from
  `pyteomics.mass.calculate_mass(sequence, ion_type=..., charge=...)`. The ion type is carried
  in the modification string (for example `ion_type:c`), and the default is `M`.
- `hxex_fragments.py` generates fragment entries from a parent peptide with a spec such as
  `b-NH3:1,2,7 x:ALL y:1,2:1,2` (ion type, fragment lengths, optional charges). Each ion type
  is classed as parent, N-terminal or C-terminal for truncation; the fragment charge is
  limited to the parent charge minus 1 by default. Its ion list is taken from pyteomics'
  `std_ion_comp`.
- `get_na_isotope` **ignores the ion type**: it removes `ion_type` from the composition and
  builds the shape from the fragment's sequence with default termini.
- `count_amides` **has no ion-type argument**.

Measured (RDKVQKEYAL as an example fragment, pyteomics compositions, shape from the
elemental composition): using the ion-type composition instead of the default-termini one
changes the natural-abundance shape mean by `a` -0.020, `x` +0.015, `b` -0.005, `z` -0.004,
`c` -0.001, `y` and `M` 0; the largest single-bin change is 0.008. These are small next to the
fit errors seen so far (about 0.4 D), but they are systematic. The effect on the exchangeable
amide count was **not** measured.

## CPU and scale measurements (this machine, 48 cores, PyTorch using 24 threads, float64)

| Run | CPU | GPU (RTX A4500), same data |
|---|---|---|
| Fit, 542 measurements | 62 s | about 16-19 s |
| Fit, 1,084 measurements | 109 s | not measured |
| Bootstrap, one iteration at 1,084 | 24 s | about 8 s at 542 (not measured at 1,084) |

- CPU cost is roughly 100 s per 1,000 fits and grows about linearly.
- Extrapolated, not measured: 10,000 fits in about 15-20 minutes on this CPU; a bootstrap of
  N = 20 at that size on the order of an hour.
- A laptop-class CPU was not measured and will be several times slower.
- On the GPU, run time grew roughly linearly beyond a few thousand measurements, and the
  share of unconverged fits changed little with batch size (tested up to about 10,000,
  using replicated test data).

## GUI

Technology options, to be decided when the GUI starts (general trade-offs, not tested here):

- **Qt desktop (PySide6):** the best interactivity (zooming, picking, linked plots). Qt for
  Python is available under the LGPL, which fits GPL v3. It is a large dependency, so it would
  be an optional extra (`pip install hxex[gui]`).
- **Browser-based (Panel, NiceGUI, Streamlit and similar):** faster to build and well suited to
  running on a remote GPU machine.
- **Notebook widgets:** the lightest option and the closest to the existing notebook workflow.

Integrating the GUI from the start would be better only if the primary workflow is interactive
manual curation during analysis (for example hand-placing peaks or re-picking controls); then
the review state would shape the core. If most users run batch analyses and then inspect the
results, library-first is right.

## Phase 0 status

Created 2026-10-09 (branch `main`; public at github.com/tuttlelm/hxex, first commit `a2c0ce7`):
source layout (`src/hxex` with a placeholder for each planned module), `pyproject.toml`
(hatchling, Python >= 3.11, GPL-3.0-or-later), `LICENSE` (the GPL v3 text from pyHXExpress's
`LICENSE.txt`), `README.md`, `.gitignore`, smoke tests, a CI workflow and this plan.

Verified:

- The package installs and the 17 smoke tests and `ruff check` pass on Python 3.11, 3.12, 3.13
  and 3.14 (Linux x86-64, CPU-only PyTorch 2.14.1, NumPy 2.4.6, SciPy 1.17.1, pyarrow 26.0.0,
  pyteomics 5.0.1, brain-isotopic-distribution 1.5.19, matplotlib 3.11.2), using `uv`-managed
  interpreters. `hxex --version` works.
- **GitHub Actions CI passes on all 11 jobs** (commit `5c3c50d`): Linux on Python 3.11, 3.12,
  3.13 and 3.14; Windows and macOS on Python 3.11, 3.12 and 3.14; and the minimum-versions job
  (Python 3.11, lowest allowed dependency versions). CPU PyTorch comes from the PyTorch CPU
  index on Linux and from PyPI on Windows and macOS.
- The repository is public and GitHub detects the license as GPL-3.0. The history contains one
  commit, authored with the GitHub no-reply address, and no local paths or private contact
  details.
- **Dependency lower bounds:** installing the lowest allowed version of every dependency (NumPy
  1.26.0, SciPy 1.11.1, PyTorch 2.5.0, pyarrow 15.0.0, pyteomics 4.7, brain-isotopic-distribution
  1.5.1, matplotlib 3.8.0) works on Python 3.11 and the smoke tests pass. This checks installation
  and imports only; it becomes meaningful as code starts using these libraries' APIs.
- **Licenses:** every runtime dependency is permissive and compatible with GPL v3: NumPy
  (BSD-3-Clause), SciPy (BSD), PyTorch (BSD-style), pyarrow (Apache-2.0), pyteomics (Apache-2.0),
  brain-isotopic-distribution (Apache-2.0, from its license text), matplotlib (PSF).
- **SPDX identifier decided: `GPL-3.0-or-later`** (changed from an initial `GPL-3.0-only`, which
  matched pyHXExpress's plain GPL-3.0 declaration). It is stated in `pyproject.toml`, the
  README and a `SPDX-License-Identifier` header in every source and test file.
- **Packaging:** the sdist and wheel build and pass `twine check`; the wheel contains the
  package and the license, with the intended metadata.
- **GPU:** a separate `.venv-gpu` (PyTorch 2.14.1 with CUDA 12.6) sees the RTX A4500 on this
  machine (driver 535), runs float32 and float64 GPU operations, and the smoke tests and lint pass
  in it. `.venv` stays CPU-only to match CI. The README documents the GPU install.

Prepared but not yet run:

- `.github/workflows/publish.yml` (PyPI trusted publishing, checks that the release tag matches
  the version) and `docs/RELEASING.md` (the setup steps) are written but have never run.

Not done:

- The PyPI name `hxex` is not reserved. This needs a PyPI account; the steps are in
  `docs/RELEASING.md`. A name is only held once a first release is uploaded.

## Phase 1 status: `hxex.chem`

Implemented 2026-10-09 (commits a9973ea, c1496d9): `modifications` (private,
immutable `ModificationRegistry`, `acK`/`pS`/`pT`/`pY` built in, `parse_sequence` tokeniser,
error classes), `ions` (ion types are pyteomics' `std_ion_comp`), `amides` (`count_amides`;
`exchangeable_sites` = amides plus the user's `Hex`, any `namide_scale` is left to the fitter),
`composition` (`composition`, `neutral_mass`, `mz`, `natural_abundance`) and `spec` (the
metadata `modification` string: `H:1 Hex:1 ion_type:c undeut_mz:...`). `Nterm_subtract` is an
explicit argument (`nterm_subtract`, default 1), not global config.

Tests (`tests/test_chem.py`, 279 pass in the suite): golden values from `hxex_updating.py`
(committed in `tests/golden/chem_reference.json`, regenerated by
`tests/golden/make_chem_golden.py`, which extracts the reference functions with `ast` so the
reference module and its edits to pyteomics' global tables are never imported); m/z and
composition for every pyteomics ion type; shapes against an independent `brainpy` calculation
from pyteomics compositions; PTMs; no global pyteomics state.

Deliberate differences from `hxex_updating.py` (all other tested values agree to 1e-9 or better):

1. **Amide count with a modified residue in the first residues** and `Nterm_subtract = 2`
   (residue-based, as already listed above): `acKPEPTIDE` 5 vs 4, `pTPEpYPEPK` 4 vs 3.
2. **m/z with a composition modification (`H:1`, ...).** The reference adds
   `calculate_mass(composition=mod, charge=z)`, which includes `z` protons a second time, so
   its m/z is too high by one proton mass (about 1.0073; checked: 466.2459 vs 465.2387 for
   `PEPTIDEK`, z = 2, `H:1`). `hxex` adds the composition once. Not seen in the golden data,
   which has no composition modifications.
3. **Shape with a composition modification.** In `get_na_isotope` the final
   `{**pepcomp, **mod_dict}` overwrites the summed counts with the modification's own values,
   so `H:1` sets H to 1 (the shape has 7 peaks instead of 10 for `PEPTIDEK`) and `Hex:1`
   replaces the amide count instead of adding to it. `hxex` adds both.
4. **Shape follows the ion type** (planned above, open question 3). The reference uses the
   default-termini composition. Identical for `M` and `y`; differences for other types are the
   small ones measured earlier.

Still open: how the amide count depends on ion type. `AmideRules` is a serialisable per-ion-type
offset table passed to `count_amides`, `exchangeable_sites` and `natural_abundance`; it is empty
by default, so every ion type uses the pyHXExpress rule until offsets are set; which terminal
modifications are needed; whether modified residues should change the amide count. Note:
pyteomics 5.0 changed the `z+1`, `z+2`, `z+3` compositions and warns on access, and the lower
bound in `pyproject.toml` is 4.7, so those ion types differ between pyteomics versions (tests
compare against whichever is installed). pyteomics 4.7 (the lower bound) cannot compute
`calculate_mass(composition=..., charge=...)` (a keyword clash inside pyteomics), so `chem` uses
`Composition(...).mass(charge=...)`, which works on 4.7 and 5.x. Not done in `chem`:
fragment-table generation (Phase 6), peak picking (next in Phase 1).

## Phase 1 status: readers (`hxex.io`, `hxex.spectra`, `hxex.hooks`)

Implemented 2026-10-09 (commits 4ec4ed3, 3a0e5ed). Readers return raw spectra only;
peak picking is a separate stage (next).

- `hxex.spectra`: `PeptideEntry` (a row of pyHXExpress's `metadf`: file, sample, peptide, range,
  charge, `modification`, other columns kept in `extra`), `Spectrum`, and `SpectrumSet` (flat
  NumPy `mz`/`intensity` plus `offsets` and parallel index arrays; no pandas).
- **Spectrum identity:** every spectrum has a `spectrum_id` built from where it was read, so it
  is unique within a data directory and independent of read order, filtering and the fully
  deuterated time: `FILE#COLUMN` (1-based column of the label) for HX-Express,
  `SAMPLE/PEPTIDE_DIR/FILE` for SpecExport, `PEPTIDE_DIR/FILE` for pooled folders.
  `read_spectra` requires an explicit `project` name (non-empty, no `:`) and prefixes it, giving
  `PROJECT:LOCAL_ID`, so ids stay unique when sets from different directories are combined;
  `split_spectrum_id` takes it apart. `SpectrumSet.find(id)` gives the position. Replicate numbers may have gaps (SpecExport takes
  them from the file name); (data set, time, rep) is not assumed unique, the id is. The ids
  are meant to key the results bundle and the review layer.
- `hxex.hooks`: `progress(done, total)` and `cancel() -> bool` hooks and `Cancelled`; used by
  `read_spectra`.
- `hxex.io`: `scan(data_dir, fmt)` lists entries, `read_spectra(entries, data_dir, fmt)` reads
  them, `read_peptide_table` reads a `metadf` CSV. Format-specific functions are in
  `io.hxexpress` (`.xlsx` or `.csv`, one file per sample, peptide and charge, one time label per
  column pair) and `io.specexport` (`SAMPLE/START-END-SEQ/TIME-REP-zCHARGE.csv` plus
  `SAMPLE.fasta`). Time labels are parsed in `io.times`. The fully deuterated time is a user
  argument (`fulldeut_time`, default 1e6 s).
- New dependency, agreed with the owner: `openpyxl>=3.1` (MIT; its only dependency `et-xmlfile`
  is MIT). The lower bound 3.1.0 was tested on Python 3.11.

- **Pooled folders** (`hxex_pooled_utils.py`): a SpecExport folder holding peptides from more
  than one protein, `DATA_DIR/START-END-SEQUENCE/...`, with no sample level and no FASTA file to
  cut from (a residue range is ambiguous when proteins share the folder). `scan(dir, "pooled",
  sample=...)` takes the sequence from the directory name, tokenised with the `chem` registry so
  modX labels work. `match_candidates(entries, pooled_fasta, classes=[...])` returns, for each
  entry, the FASTA records containing its peptide (modifications ignored) and the requested
  class labels that appear in those record ids. Not ported from that file: `update_datafits`
  (resets D-uptake to the fitted n x mu and adds a dummy total-deuteration row; it reads fit
  output and belongs to Phase 3, controls and results).

Verified: on the Bimodal_HDX_Data test set the 24 spreadsheets give 566 spectra and 496,708
points, **identical** (m/z, intensity, time, replicate) to pyHXExpress's `allrawdata`. On the
Chris_B4 SpecExport example the entries (sample, range, charge, sequence from the FASTA) equal
pyHXExpress's `metadf_asrun`, and the (time, replicate) set equals its `alldeutdata`. On two
real pooled folders (tubulin, 318 entries each) the entries, candidate lists and classes equal
pyHXExpress's `metadf` columns `sample_matches` and `tub_class`. Those three tests read their
data from `HXEX_BIMODAL_DATA`, `HXEX_SPECEXPORT_EXAMPLE` and `HXEX_POOLED_EXAMPLE` and are
skipped without them; the other tests use generated files.

Deliberate differences from `hxex_updating.py`:

1. Readers do not pick peaks and do not contain the `Test_Data` branch (reading
   `bimodal_solutions2.txt`); that belongs to the test code.
2. An unknown or missing time unit raises `ReadError`. The reference computes
   `60 ** ('smh'.find(unit))`, so a label such as `60` or `5 x` silently becomes 1/60 of its
   value. Units and the `undeut`/`TD` prefixes are case-insensitive here.
3. SpecExport charges of 10 or more are read (the reference reads only a single digit), and
   spectra are ordered by time, replicate and file name (the reference uses `os.listdir` order).
4. Filters on sample and peptide range are exact matches (the reference's HX-Express filter on
   peptides is a substring match). Files whose names do not parse, and directories that are not
   `START-END-SEQ`, are skipped with a warning or debug message instead of stopping the scan.
   `.xls` is not read (the reference scanned only `.xlsx`).
5. The spectrum for a label is read from the label's own column and the next, not from column
   `2 * i` (identical for the standard layout). With several FASTA records the last is used,
   as in the reference, with a warning.

6. Pooled matching: a class is reported if its text appears in any single matching record id (the
   reference searched the `;`-joined string, so a label could match across two ids), and the
   `class_suffix` argument is not ported (matches and classes are returned as tuples, not
   joined strings). Sequences in pooled directory names that do not parse are skipped with a
   warning.

7. Times are computed as decimals and rounded once, so `4.1m` is exactly 246 s (the reference's
   `4.1 * 60` gives 245.99999999999997) and equal times written differently, such as `4.1m` and
   `246s`, compare equal. Values that are exact in floating point are unchanged.

Kept from the reference, on purpose: digits after the decimal point of an HX-Express time of 1
or more are replicate markers (`1.2 min` is 60 s, second replicate), not fractions.

## Peak picker: reference behaviour (`peak_picker` in `hxex_updating.py`, read 2026-10-10)

What it does, per spectrum (m/z sorted, intensity):

1. Predicted bin centres: `undeut_mz` (from `chem.mz`, or the user's `undeut_mz`), then bin `i`
   is `undeut_mz + sum(dm_k) / z` with the ExMS steps 1.003355, 1.003355, 1.004816, 1.004816 and
   then 1.006277 per bin. The number of bins is
   `max(count_amides, min_pts) + len(natural_abundance) // 2 + Zero_Filling (+ Hex)` with
   `min_pts = 3 * Max_Pops + 2`.
2. Noise threshold: `setNoise` if non-zero, else `Y_ERR / 100 * max(intensity)` of that spectrum.
3. For each bin, the window is `pred * (1 +/- ppm / 1e6)`. With fewer than 10 points in the window
   the bin intensity is the window maximum; with 10 or more the maximum counts only if it is not
   the first or last point of the window (shoulder rejection), else the bin is 0.
4. A bin is a "zero" if it is empty, or below the threshold, **and** its predicted m/z is above
   the mean of all predicted m/z. Picking stops at the first bin `i >= min_pts` where the count
   of zeros so far reaches `Zero_Filling`.
5. Output per bin: the **predicted** m/z (not the observed position), the window maximum (kept
   as is when below the threshold), `n_deut`, plus `env_width` (charge times the m/z width of
   the envelope at `Env_threshold` times the maximum, by linear interpolation) and `env_symm`
   (`1 + left-edge intensity / max`, so 1.1 unless the first bin is above the envelope height),
   and `max_namides` (backbone count, without `Hex`).
6. The SpecExport reader drops spectra whose picked intensities sum to zero; the HX-Express
   reader does not. `time_idx` is numbered over the times that remain.

Findings (numbers from the October 2026 tubulin runs, 4,235 spectra, and the other test sets):

- **The 19Feb2024 `alldeutdata` is not an oracle for the current picker.** It was made by an
  older version (constant spacing 1.006227 per bin and a different bin count). The oracle is the
  October 2026 pooled runs, which come with `allrawdata` and use the current spacing: the
  reference function reproduces `batchrun_1pop_06Oct2026/TBA_run` exactly (2,232 spectra, six
  columns, 1e-9) with `Nterm_subtract=2, Zero_Filling=3, setNoise=50, Max_Pops=1,
  Peak_Resolution=50, Env_threshold=0.1`. A numpy prototype of the same logic also matches
  TBA_run and TBB_run exactly (0 of 4,235 differ).
- **The zero count is cumulative, not consecutive.** Zeros anywhere to the right of the mean
  m/z add up, so a valley inside a multimodal envelope can stop picking early. A consecutive
  rule would pick more bins in 21 % of spectra, and in 7 % the extra bins contain intensity
  above the threshold (whether that is signal or noise has not been examined).
- **The picked data depend on `Max_Pops`** (through `min_pts`), a fitting parameter.
- **Below-threshold intensities are kept** (10 % of bins hold a non-zero value under the
  threshold) while being counted as zeros; 32 % of spectra run to the bin cap instead of
  stopping on zeros.
- **Shoulder rejection and the 10-point rule have never been exercised by real data here:** in
  every dataset tested (Bimodal, B4, tubulin) no window has 10 or more points, so these data
  are centroided or sparse. The branch has no oracle and needs synthetic tests; treat it as
  unvalidated.
- The reference sorts the window with an unstable quicksort, so a tie for the maximum is broken
  arbitrarily (only relevant to the shoulder test; no ties occurred).
- The `try/except` around the envelope calculation hides errors; the envelope columns are
  missing (NaN) instead. For modified peptides the bin count and `undeut_mz` inherit the two
  `chem` differences listed above.

## Phase 1 status: picker (`hxex.picking`)

Implemented 2026-10-10. `pick_spectra(spectra, PickerConfig)` turns a `SpectrumSet` into a
`BinnedSet` (flat arrays: predicted m/z, intensity, `n_deut`, plus per spectrum `env_width`,
`env_symm`, `max_namides`, the threshold used, and the flags `empty`, `no_signal`, `capped`; the
index arrays and `spectrum_id` are carried over). `PickerConfig` is serialisable
(`to_dict`/`from_dict`) and holds the settings pyHXExpress took from `config`: `resolution_ppm`
(50), `zero_filling` (4), `noise` (absolute; None or 0 means use `noise_percent` of the spectrum
maximum, 1 %), `env_threshold` (0.1), `nterm_subtract`, `shoulder_min_points` (10) and the
amide rules. Per-spectrum modification strings (`Hex`, `ion_type`, `undeut_mz`, composition)
are honoured through `chem`.

Deliberate differences from `peak_picker` (decided with the owner, 2026-10-10):

1. **Picking ends `zero_filling` bins after the last bin that reaches the threshold**, so a
   valley inside a multimodal envelope never ends picking early. pyHXExpress stopped after a
   cumulative count of below-threshold bins to the right of the mean m/z. On the October
   tubulin runs (TBA 2,232 and TBB 2,003 spectra) the old rule left out bins above the
   threshold in 20 to 25 % of spectra, more than 10 % of the spectrum maximum in about 5 %, and
   in a few cases the maximum itself.
2. **The bin limit is the physical range**: `amides + Hex + (natural-abundance peaks - 1) +
   zero_filling`. pyHXExpress used `max(amides, min_pts) + peaks // 2 + zero_filling`, which cut
   the natural-abundance tail of strongly deuterated spectra (the limit was reached in 32 to
   41 % of spectra, now 18 of 2,233 and 2 of 2,003; `capped` flags them).
3. **No `min_pts`.** pyHXExpress needed at least as many points as fit variables, and derived the
   minimum from `Max_Pops`, so the picked data depended on a fitting setting. The picker has no
   such minimum; the engine in this project has no point-count requirement (3K parameters per
   model, BIC), and capping K by the number of bins is a fit-stage decision (Phase 2).
4. **Nothing is dropped.** Spectra with all-zero picked intensity are flagged `empty`
   (pyHXExpress dropped them in the SpecExport reader only) and `no_signal` (no bin reaches the
   threshold, a broader flag). The run stage drops flagged spectra with `BinnedSet.select`.
5. **`time_idx` is not stored**; `time_idx()` is derived from the spectra in the set. In
   pyHXExpress the SpecExport reader numbered the picked table and the raw table separately, so
   dropping a whole time point shifted the picked numbering against the raw one, and
   `run_hdx_fits` selects raw spectra by the picked index (silently the wrong spectra). Refer to
   spectra by `spectrum_id`.
6. A tie for the window maximum goes to the first point in m/z order (pyHXExpress used an
   unstable sort). Errors in the envelope calculation are not swallowed: a spectrum without a
   crossing gets NaN `env_width` and `env_symm`.
7. A trailing zero fill of exactly `zero_filling` bins replaces pyHXExpress's variable tail (it
   kept picking until the zero count to the right of the mean m/z was reached, so undeuterated
   spectra had many more zero bins). If baseline bins are wanted for noise estimation, raise
   `zero_filling`.

Kept from `peak_picker`: the ExMS spacing, the inclusive +/- ppm window, the 10-point rule and
shoulder rejection, sub-threshold intensities kept as they are, predicted (not observed) m/z in
the table, the envelope formulas, `max_namides` without `Hex`.

Verified against pyHXExpress output from the October 2026 pooled runs (SpecExport raw spectra
with the picked table from the same code): in TBA, TBB, TBA_B5 and TBB_B5, every bin both
pickers produced is identical (m/z and intensity, exactly), every bin only pyHXExpress kept is
below the threshold, the envelope columns are equal wherever the bin counts are equal, and the one
all-zero spectrum is flagged `empty`. A golden file of 11 spectra covering
the cases above (made by `tests/golden/make_picker_golden.py`) is kept outside the repository
because the data is unpublished; the test reads it from `HXEX_PICKER_GOLDEN`, and the whole-run
check reads a run directory from `HXEX_PICKER_ORACLE`. Both skip when unset. The 19Feb2024 Bimodal `alldeutdata` came from
an older picker (constant bin spacing) and is not an oracle.

**Unvalidated:** shoulder rejection and the 10-point rule. In every dataset available (Bimodal,
B4, tubulin) no window has 10 or more points, so they are covered by synthetic tests only.
Modified peptides (`Hex`, `H:1`, PTM labels) are also covered only by synthetic tests, because no
oracle run uses them.

## Phase 2 plan: the engine (proposed 2026-10-10; decisions marked OPEN)

Read: the whole prototype, `fit_multimodal_binomial.py` (2,034 lines) and `run_test.ipynb`.

**What the prototype has.** Core model (about 450 lines): real-valued binomial pmf, convolution with
the natural-abundance shape, a batched K-component mixture (`log A`, `n = n_max * sigmoid`,
`p = sigmoid`; optional mean bounds). Fit: shape-corrected k-means start, nested K start by
splitting the heaviest component, Adam with cosine learning rate then L-BFGS, per-measurement
NaN protection, best restart per measurement, BIC/AIC selection with a minimum-population
filter. Noise: constant-sd Gaussian or Poisson, scaled from a reference spectrum (the series'
undeuterated control). Bootstrap: parametric resampling, warm-started refits at the selected K,
Hungarian matching of components, SE and percentile CI on A, n, p, mean and fraction, plus a CI
band of the total curve. Output: a pandas table and matplotlib plots.

**Problems that shape the design.**
1. Device and dtype are module globals set at import (`torch.set_default_dtype`): no global state
   is allowed, so they become an explicit engine setting (default as now: float32 on GPU,
   float64 on CPU).
2. The bootstrap keeps an iteration only if *every* measurement in the set succeeded, so one
   degenerate spectrum discards that resample for all of them. It never bit on the 542-spectrum
   set (all 20 iterations succeeded for all), but with 10,000 fits it would lose most iterations.
3. A second, single-measurement code path (`BinomialMixture`, `fit_mixture`, `fit_mixture_auto`,
   about 400 lines) duplicates the batch path; the notebook only uses the batch one.
4. `mean_bounds` (pinning controls to expected means) is implemented but unused: the notebook
   handles bad controls by a back-exchange fallback instead.
5. Errors are `print`ed, `plt.show()` is called, and the k-means start is a Python loop over
   measurements (slow at 10,000).

**Proposed mapping.** `model`: the math and the mixture, given an explicit device and dtype.
`noise`: the noise model and choosing each measurement's reference spectrum (a series is one
entry, that is `data_id`; the undeuterated control is the spectrum at the user-set time).
`fit`: `FitConfig`, initialisation, optimiser, K selection, returning a `FitResult` made of
NumPy arrays (per K: A, n, p; grids of NLL, BIC, AIC; selected K; population filter flags;
noise scale; the configs). `bootstrap`: `BootstrapConfig` and `BootstrapResult`. No pandas
(tables are columns of arrays that go to Parquet in the bundle). Plotting moves to `report`
in Phase 4 and reads only a loaded result. Input is a `BinnedSet`: bins are `n_deut`, shapes
come from `chem.natural_abundance`, and `n_max` is a factor times `chem.exchangeable_sites`.

**Regression baseline (frozen from the prototype, 2026-10-10).** The 542 measurements of the
Bimodal test set (23 series, Bradykinin_Lo excluded) as the notebook sets them up: binned
sticks from `alldeutdata_19Feb2024.csv` (the engine takes binned data, so the older picker does
not matter here), Gaussian noise at 3 % of the undeuterated control peak, `n_max = 1.5 x
amides`, K_max 3, bootstrap N = 20. Prototype results:

| | CPU float64 | GPU float32 |
|---|---|---|
| fit | 67.7 s | 18.9 s |
| bootstrap (N=20) | 299 s | 177 s |
| K correct | 69.2 % | 69.0 % |
| selected K = 1 / 2 / 3 | 180 / 279 / 83 | 180 / 278 / 84 |

CPU and GPU agree on the selected K for 541 of 542 spectra. On those, the fitted means differ
by a median of 0.0002 D (90th percentile 0.04 D, maximum 1.6 D) and the fractions by a median of
0.000 (90th percentile 0.005, maximum 0.42): the large differences are different local optima of
the same non-convex fit, so CPU/GPU parity has to be a statistical criterion (selected K, median
and percentile differences, NLL no worse), not an element-wise one. All bootstrap iterations
succeeded for all 542 measurements on both. The refactored engine on CPU float64 with the same
seeds is expected to reproduce the CPU baseline much more closely, and how closely will be
measured and recorded.

**Order of work (1 to 3 done, see the status below).** (1) `model` and `noise`, tested numerically against the prototype's values;
(2) `fit`, checked against the CPU baseline; (3) `bootstrap`; (4) the result types and the
results-bundle schema (spectrum ids are the keys); (5) CPU/GPU parity tests.

**Decisions (2026-10-10).** Drop the single-measurement path (batch with one measurement covers
it). The bootstrap accepts iterations per measurement (a deliberate difference; identical when
nothing fails). `n_max` is a user setting with default 1.2 times the exchangeable sites.
`mean_bounds` (bounds and a starting value for the mean of every component of a measurement, to
pin control spectra near their expected means) is dropped: a hard bound hides a bad control,
while the notebook's fallback flags it. It was confined to the parameter transform and the
initialisation, so it can be added later if a use appears.

**K range.** The user sets the range of K to test (`k_min` to `k_max`): 1 to 3, 2 to 3, only 1,
only 2. The nested start uses the (K-1) fit when that K was fitted and k-means otherwise (for
example the first K of a range that starts at 2). With one K there is no model selection and
the minimum-population filter is only reported as a flag. The bootstrap always runs at each
measurement's selected K and is optional, including with a single K.

## Phase 2 status: `model`, `noise` and `fit` (2026-10-10)

Done so far, with the bootstrap and the results bundle (later subsections); CPU/GPU parity is
next. Implemented: `hxex.model`
(`Backend` for device and dtype, the real-valued binomial pmf, shape
convolution, `PaddedBatch`, `MixtureBatch`, BIC/AIC), `hxex.noise` (`NoiseConfig`,
`noise_scale`, `control_references`, `sample_size`) and `hxex.fit` (`FitConfig`, `Measurements`
with `from_binned` and `from_arrays`, `fit_models` returning a `FitResult` of NumPy arrays with a
tidy `table()`, k-means and split starts, the optimiser). No global state, no pandas, errors are
logged or raised, hooks for progress and cancellation. `k_min` to `k_max` is a setting (1-3, 2-3,
only 1, only 2); with one K there is no selection and the population filter is only reported;
the nested start uses the previous K when it was fitted and k-means otherwise.

**Verified against the prototype** (the 542 measurements, CPU float64, tests in
`tests/test_model.py`, `test_noise.py`, `test_fit.py`, goldens from `tests/golden/`):

- The noise scale matches the prototype to 1e-12. The pmf matches for every bin up to `n` (1e-12)
  and predictions and losses match at `n` just above an integer (1e-6); the one bin above `n`
  differs by design (see the next subsections).
- With the prototype's torch version, a port (made before the model change below) that keeps its
  global gradient clipping, L-BFGS and random streams reproduces the prototype's K = 1 fits for 98.7 % of the spectra to 1e-9 (all to
  4e-8), selects the same K for 541 to 542 of 542, and is not slower (58.6 s against 67.7 s). With
  torch 2.14 instead of 2.5.1 the same K = 1 agreement is 78 %: different torch versions give
  different roundings, and the non-convex fit amplifies them.
- The prototype is **not reproducible**: it never seeds torch's generator and its warm starts
  add `torch.randn` noise, so two runs differ for about a third of the K >= 2 fits (selected K
  agrees for 541 of 542). Agreement with it can only be statistical.
- The prototype's measurements are **not independent**: `clip_grad_norm_` over the whole batch
  (and the summed L-BFGS objective) let one spectrum change the step of all others. A change of
  1e-10 to one spectrum changed the fits of 20 to 82 % of the other 541 (loss differences up to
  46 %). Results therefore depended on what else was in the batch, which would break chunking
  by device (Phase 4).

**Deliberate differences** (1, 2 and 4 are settings; 3 and 5 are not):
1. `grad_clip='per_measurement'` instead of the whole-batch norm.
2. `lbfgs_steps=0` (replaced by the Levenberg-Marquardt polish): L-BFGS gave no measurable gain (315 against 317 fits better than the
   prototype with and without it) and is the second source of coupling.
3. Every random number of a measurement comes from a generator seeded by `seed` and a key from
   its `spectrum_id` (CRC-32), so runs are reproducible and a spectrum's starting points do not
   depend on its position or the batch size.
4. `n_max` is `n_max_factor` times the exchangeable sites (user setting, default 1.2).
5. The model's pmf masks at `b < n + 1` instead of `b <= n` (see the next subsection).

With these, a spectrum's fit depends only on its own data, its noise reference and the seed: a
1e-10 change to one spectrum leaves 95.9 % of the others bit-identical (the rest are its series,
whose noise scale comes from it), and a subset fitted alone agrees with the full-batch fit to
1e-9 for 93 to 99 % of its spectra (the remainder is summation order with different padding,
amplified by the fit; pad to one global width when chunking). Against the prototype's baseline
(`HXEX_FULL_REGRESSION=1`, about a minute): selected K identical for 97.6 % (13 of 542), correct K
68.5 % against 69.2 % (variants of the optimiser spanned 68.5 to 69.4 %), median loss difference
about 0 (slightly lower at K = 2 and 3), fit 51.5 s against 67.7 s. CPU/GPU: the prototype's own
GPU run selects the same K as its CPU run for 541 of 542; the parity test for the new engine is
still to be written.

**Convergence and the cut-off of the binomial (investigated and settled 2026-10-10).** On clean
synthetic bimodal spectra 10 to 17 % of K = 2 fits ended well above the best loss, in every
optimiser variant (more Adam steps or restarts did not help). Comparing 80 synthetic spectra
(one, two and three populations) with a second-order multi-start solver gave two causes:

- **The cut-off of the real-valued binomial.** The prototype masks bins above `n` (`b <= n`;
  pyHXExpress's `nCk_real` does the same), so the loss drops by a jump each time `n` rises through
  an integer that carries probability, a large one when `p` is high (late time points, fully
  deuterated controls). A gradient method that stops just below an integer cannot see that
  crossing helps, Adam overshoots back over the edge, and the optimum can sit on the edge itself.
  (`n` is a free real number throughout; the best fits include n = 14.43 and 16.23.) The mask
  also loses up to 15 % of the probability between integers at p = 0.9 and puts the centre of the
  component up to 0.28 D below `n p`, so the reported mean `n p` was not the component's centre.
- **The wrong basin.** A spectrum of two populations fitted with one often ends in a broad
  low-mean solution with `n` at its upper bound; harmless for selection (BIC rejects it).

**Decision: the model uses the exact gamma-function continuation, masking at `b < n + 1`
instead of `b <= n`.** `Gamma(n - b + 1)` is finite and positive for `n < b < n + 1` and its
reciprocal falls to 0 as `n` decreases to `b - 1`, so the pmf is continuous in `n`, identical at
integer `n`, keeps the mass within 2 % of 1 and the centre within 0.05 D of `n p` (checked for
13 <= n <= 14, p = 0.9). It is the only model: the prototype's hard mask is not an option
(removed on the owner's instruction), and the escape stage that was written to cross the cliffs
of the hard mask was removed with it. **This is a deliberate difference from the prototype and
pyHXExpress:** at non-integer `n` the bin just above `n` has probability, and the fitted means of
spectra at high `p` can move (on the Bimodal set the median shift against the hard mask was 0.005
D, the 90th percentile 0.18 D, with no systematic direction). The model test with the prototype's
values compares every bin up to `n` exactly (1e-12) and predictions and losses at `n` just above an
integer (1e-6); the one extra bin is checked structurally.

A **Levenberg-Marquardt polish** per measurement (`polish_steps=40`; `hxex.fit.polish.lm_polish`)
follows the restarts: damped Gauss-Newton on the Jacobian from `torch.func`, only steps that lower
the measurement's loss are kept, each step limited to 2 per parameter (without the limit the first
step from a poor start could leap to a distant optimum with `n` pinned at its bound), measurements
that converge or stall are dropped from the batch, Fisher weights for Poisson noise. It lowers the
loss by a small amount in about 85 % of the fits (more than one loss unit in 1 to 2 of 140) at
almost no cost, and it works per measurement, so independence is kept.

| | hard mask, Adam only | hard mask + polish + escape | smooth, Adam only | smooth + polish (final) |
|---|---|---|---|---|
| synthetic set A (80 spectra): correct K | 93.8 % | 97.5 % | 100 % | 100 % |
| synthetic set B (60 spectra, own integer n of 6-14 and p of 0.3-0.95 per component): correct K | 83.3 % | 90.0 % | 96.7 % | 98.3 % |
| set B: error of the means, median / 90th pct | 0.071 / 0.53 D | 0.028 / 0.35 D | 0.024 / 0.34 D | 0.027 / 0.33 D |
| Bimodal test set (542): correct K (prototype 69.2 %) | | 69.6 % | 69.2 % | 69.0 % |
| Bimodal test set: fully recovered / median mean error | | 30.8 % / 0.342 D | 31.0 % / 0.324 D | 30.3 % / 0.326 D |
| Bimodal test set: fit time (prototype 67.7 s) | | 63.2 s | 55.7 s | 60.4 s |

On your real set the model change is neutral (the two models pick the same K for 526 of 542
spectra) and it is clearly better where the truth is known and varied. Selected K equals the
prototype's for 94.8 % of the spectra; the regression tests require the fits to be not worse than
the prototype's (median loss difference below 1e-3, as good or better for 70 % of fits), because
the models now differ slightly. Still open: a spectrum fitted with too few populations can end in
the wrong basin (harmless for selection).

**Noise model (looked at 2026-10-10; decision: keep both Gaussian and Poisson).** In the
replicate data of the tubulin runs (TBA and TBB, the peak-picked stick heights of 12,000 bins
each, 3 or more replicates at the same peptide, charge, time and bin) the noise is not constant:
the replicate sd grows with the intensity with a log-log slope of 0.82 (0.64 after each spectrum
is normalised to its own total, which removes whole-spectrum scale differences), and the
variance divided by the intensity is roughly constant (about 1.0 to 1.3e-3) for shares of the
spectrum between 0.005 and 0.77, which is Poisson-like with about 1,000 effective counts per
spectrum. Caveats: these are window maxima (biased up at low intensity) and replicate-to-replicate
variability includes deuteration differences, not only measurement noise. On the Bimodal set
against the known solutions the constant-sd Gaussian (3 % of the undeuterated peak) did better
than Poisson at every effective-count value tried (correct K 69.0 % against at most 68.5 %, fully
recovered 30.3 % against at most 28.0 %, median mean error 0.326 D against at least 0.381 D), but
that set's noise may not look like real replicate noise, so the two results do not settle which
model is right; real noise probably has a floor plus an intensity-dependent part (for example
variance = a^2 + b I). Both models stay. A heteroscedastic Gaussian that contains both as special
cases is a candidate. The test that decides is the Phase 5 validation: which model gives
calibrated confidence intervals on replicates and charge-state pairs. The k-means start rescales
a Gaussian-noise spectrum to `KMEANS_PSEUDO_COUNTS` (1000) so the clustering does not depend on
the intensity of the data; the fits did not change for values from 30 to 100,000. The Poisson
path does not rescale (it would see flattened peaks if its scaled intensities exceeded 2,000 per
bin, which the usual `n_eff` setting avoids).

**Bootstrap (`hxex.bootstrap`, 2026-10-10).** A parametric bootstrap per measurement at its
selected K (optional, also with a single K): the fitted curve is resampled with the noise model
of the fit (Gaussian with unit sd on the noise-scaled spectrum, or Poisson), refitted from the
original fit (first start with jitter, two random starts, Adam at a lower learning rate, the
polish), and the components are matched to the original fit by their mean `n p` (all
permutations vectorised up to K = 4, Hungarian beyond). Reported at the selected K: standard
errors and percentile intervals of the amplitude, `n`, `p`, the mean and the population fraction,
and a per-bin band of the total curve. `BootstrapConfig` sets `n_boot` (200), `ci` (95), the refit
settings, `keep_draws` (the individual fits of the selected K, off by default) and `curve_band`.
`BootstrapResult.table(fit)` adds the interval columns to the fit's table.

Differences from the prototype: a resample that fails for one measurement is dropped for that
measurement only (the prototype discarded the iteration for all of them; `n_success` counts the
good ones per measurement), a measurement whose own fit failed is not resampled, and every random
number comes from a generator seeded by the seed, the iteration and the spectrum id (reproducible
and independent of the batch). Checks: with the noise level of the generator the standard errors
match the spread of the fits over 40 independent noise realisations (ratio 1.06 and 0.91 for the
two means, 0.97 for the fractions; test under `HXEX_FULL_REGRESSION=1`); on the 542 Bimodal
measurements with N = 20 all resamples succeed, it takes 284 s on the CPU (prototype 299 s), and
against the prototype's own bootstrap the standard error of the mean is a median 0.93 times its
(middle half 0.74 to 1.12), of the fraction 1.02 times, and the interval width of the mean 0.91
times, as expected for estimates that each carry about 16 % error from 20 resamples. The
intervals are only as good as the noise model: they use the configured noise, not the data's.
Small batches are slow because each resample runs the optimiser steps one after the other
(about 1.5 s per resample for 3 spectra); the cost per spectrum falls with the batch size.

**Noise estimated from replicates (2026-10-10; `NoiseConfig(estimate='replicates')`, off by
default).** A set is one series: a peptide at one charge with its controls and time points
(`data_id`); nothing is known about the instrument or sample preparation, so nothing is pooled
across series. The noise of the tubulin series (peak-picked stick heights) is variance =
gain x intensity within a series (log-log slope of variance against intensity 1.11, middle half
1.00 to 1.19, 73 series), with a gain that differs between series (median 54, 10th to 90th
percentile 29 to 103; the effective counts per series 99 to 1,266 and rising with the series'
total intensity), so the estimate is made per series and no pooled variance function is used.
Replicates of a time point are rescaled to a common total (the amplitude is a free parameter of
the fit), the variance of each bin of enough intensity (`min_share` of the total) is divided by
its mean, and the series gain is the median over its time points of the median over bins,
each divided by the median of the chi-square distribution of that ratio, so that a bad
spectrum cannot inflate it (a plain pooled estimate was a median 1.13 times, 90th percentile 2.5
times and at most 12 times the robust one). A series is fitted with the Poisson model and that
gain as its scale (`obs / gain` are then counts). It needs at least `min_replicated` (3) time
points with two or more replicates; **a series without that falls back to the configured noise
settings**, so the engine now takes the noise model per measurement (`MixtureBatch`, the polish,
the starting points, the sample size of the BIC and the bootstrap draws all work per
measurement; `FitResult.noise_kind` records it). On the tubulin data 158 of 168 series can be
estimated (16 have one replicated time point, 5 none).

**Safeguard against bad spectra.** Each spectrum of a time point with three or more replicates
gets a score: the mean over bins of its squared difference from the mean of the other replicates
divided by the variance the gain predicts for that difference (`gain x mean x (1 + 1 / k)`),
about 1 for a consistent spectrum (tubulin: median 0.93, 90th percentile 2.3; 3.5 % flagged in
50 series, 99th percentile above 10). While three or more replicates are active the worst is
removed if its score exceeds `outlier_score` (4) and the others are scored again, so one bad
spectrum does not make its good replicates look bad; removed spectra are left out of the gain
(estimated again) and flagged, never dropped, and `fit_models` logs a warning. A flag means
"inconsistent with its replicates" (it can be a real difference in deuteration, not only bad
data), and a pair of replicates gives a gain but no score. `FitResult.noise_estimate` keeps the
gains, scores, flags and the reasons a series was not estimated, for the review layer. Not
implemented, on purpose: an estimate from the residuals of the fit when there are no replicates
(it was 1.98 times the replicate gain, middle half 1.43 to 3.07, Spearman 0.82, because it
includes the misfit of the model); other screens of a spectrum (fit chi-square, total intensity
against the series) belong to the quality flags of Phase 3.

**The results bundle (`hxex.io.write_bundle` and `open_bundle`, 2026-10-10).** A directory (or a
zip, read only) with `manifest.json` and zstd Parquet tables, schema version 1, documented for
readers in other languages in `docs/BUNDLE.md`. It keeps the fit inputs (`measurements`: bins,
intensities, the natural-abundance shape, `n_max`, noise references), the fit (the selected K,
the noise scale and model per spectrum, every K tried with its amplitudes, `n`, `p`, losses, BIC,
AIC and the optimiser's own parameters), the noise estimate, the bootstrap (summaries, the band
of the total curve, optional draws), the picked table, optional raw spectra, the configs and
package versions, and a tidy derived `components` table. Loading returns the objects the run
produced **bit for bit** (tested for every table, from a directory and a zip), so a loaded bundle
and a fresh fit give identical plots. Writing is atomic and refuses to overwrite a bundle unless
asked (never a directory that is not a bundle); `verify()` checks the SHA-256 of every file;
newer schema versions are refused and unknown tables ignored. `spectra(ids)` reads only the rows
asked for (Parquet row groups of 1,000 spectra). The **review layer** (accept or reject, a chosen
K, a note, per `spectrum_id`) is the one mutable table; it needs no refit, is validated against
the fit, and `effective_K` and `status_of` give the reviewed view. On the Bimodal set (566
spectra, 542 fits, a bootstrap with draws) the whole bundle is 2.35 MB, of which the raw spectra
are 1.07 MB (the plan estimated 1.01 MB for this layout), it writes in 0.13 s and loads in
0.03 s, and one spectrum by id takes about 9 ms. Tested with pyarrow 15.0.0 (the minimum) and
26. The controls and back-exchange values are bundle tables (Phase 3, item 27; additive, no schema version change). `hxex.io` loads the bundle lazily, so reading spectra does not import torch.

27. **Controls and back-exchange correction (Phase 3, `hxex.controls`; owner's decisions of
    2026-10-11).** Controls are the spectra of a series at the user-set undeuterated and fully
    deuterated times, replicates averaged (a replicate flagged as a noise outlier is left out).
    The control means `m0` and `mFD` come from the **K=1 fits** of the control spectra, in the
    model's units (deuterons beyond natural abundance); the raw envelope centroid is not used,
    because it contains the natural-abundance offset (0.35 to 1.16 Da on the Bimodal peptides)
    and is noisier (the centroid and the K=1 mean differ by more than 1.2 D for the worst 10 %
    of fully deuterated controls). The correction is `(mean - m0) / (mFD - m0) * n_amides`, not
    below 0; fractions are unchanged and the uncorrected means are kept beside the corrected
    ones. Each control **falls back on its own** to the theoretical value (`m0` = 0, `mFD` =
    `n_amides`) with a flag: `m0` when missing or above 0.5 D; `mFD` when missing or when the
    retention `(mFD - m0) / n_amides` is below 0.5 or above 1.1 (the limits are settings of
    `ControlConfig`, default 0.5, 0.5 and 1.1). A series with no usable control uses the
    theoretical values, flagged; retention is not borrowed from other series. The uncertainty of
    measured controls (SE over replicates, flagged when there is one replicate) is added to the
    SE (delta method) and widens the interval of each corrected mean (in quadrature). A deliberate
    difference from hxex_updating.py, which has no back-exchange correction. On the 23 Bimodal
    series only pep122_LO_z2 falls back (`m0_high`, a K=1 fit of 13.0 D on its undeuterated
    control), and retention is 0.81 to 1.00.

28. **Quality flags (Phase 3, `hxex.quality`).** Flags only; nothing is dropped here (the run stage
    drops spectra the picker marked `empty`, and the review layer holds manual decisions).
    `poor_fit`: the reduced deviance of the selected-K fit (squared residual for Gaussian noise,
    Poisson deviance for Poisson, noise-scaled, over `bins - 3K`) is more than 10 times the
    median of its series (at least 5 assessed spectra) **and** above 1. The absolute value alone
    is unusable: on the tubulin subset (40 peptides, 371 spectra, noise estimated from
    replicates) the median is 4.9, because the model is never exactly right, and on the Bimodal
    data with the 3 % noise setting it is 0.17, because that noise is far above the real one; the
    floor of 1 removed 11 of 12 false flags on Bimodal (fits better than the noise model, flagged
    only because their series was cleaner) and left the one real bad fit (series 14, 7 s,
    deviance 44). `low_intensity`: total intensity below 0.3 of the mean of the other replicates
    at the same time (not assessed without a replicate: the total changes along a time course,
    and against the series median the 5th percentile on tubulin is already 0.31). `noise_outlier`:
    the replicate-consistency flag of the noise estimate. On the tubulin subset: 9, 11 and 23
    spectra (36 flagged of 371), overlap poor-fit with outlier 2, low-intensity with outlier 5, so
    the screens are largely complementary; on Bimodal: 1, 0 and 0. Thresholds are defaults from
    these two data sets, not validated against known bad spectra (there is no labelled set).

29. **Run stage (Phase 3, `hxex.run`).** `run(spectra_or_binned, RunConfig, out=...)`: pick, drop
    the spectra flagged `empty` (and `no_signal` unless `drop_no_signal=False`; the picked table
    in the bundle keeps every spectrum and its flags), fit and bootstrap in chunks, controls,
    quality flags, bundle (with the run settings in `configs.run`). A chunk is made of whole
    series, because the noise estimate and the noise references work within a series; it is
    sized from a memory budget (default half the free GPU memory, 4 GiB on the CPU; `max_chunk`
    caps it). The memory of a fit was measured on the GPU: the peak grows linearly with the
    chunk, at 5.2 to 5.9 times the size of 12 pmf arrays for 94 to 542 Bimodal measurements
    (float32), so the estimate uses 75 arrays (the bootstrap needs less). Chunking does not change
    the result: on the 542 Bimodal measurements in 12 chunks of whole series against one chunk
    (CPU, float64) the selected K is the same for 542 of 542, the controls and flags are
    identical, and the means of the populations agree to 0.0017 D at the 99th percentile and
    0.037 D at most (the order of the populations in a fit is arbitrary, so they were compared
    sorted; 5 measurements differ by more than 0.01 D, all with nearly equal loss). Small chunks
    cost time (12 chunks took 198 s against 63 s for one on the CPU), so the default budget is
    large. Progress is reported in 1000 units with the fit first and then the bootstrap, and a
    cancel writes nothing. The bootstrap is made at each measurement's selected K only.

30. **Bootstrap at fixed K (`BootstrapConfig.ks`).** pyHXExpress was usually run as separate fixed
    1-population and 2-population fits, each with bootstrap statistics. `ks='selected'` (the
    default) is the bootstrap at each measurement's selected K; `'all'` or a tuple such as
    `(1, 2)` also bootstraps every measurement at each of those K values
    (`bootstrap.bootstrap_by_k`; `RunResult.bootstrap_by_k`; the `bootstrap_k` bundle tables).
    The fit already holds every K of a run, so only the bootstrap is added; its cost grows with
    the number of Ks, about the same as the separate runs without the second picking and fit. A
    measurement's resamples depend on its id and the draw only, so the result at its selected K
    is the one the selected-K bootstrap gives, and the selected-K result is taken from the
    fixed-K ones when they cover it (no repeated work; Ks that were selected but not requested
    are bootstrapped anyway, because the selected-K result needs them). Intervals at a K that
    fits badly are still computed; `passes_min_pop` and the quality flags say so. **A K in a run of
    several is not worse than the same K fitted alone** (`FitConfig.kmeans_after_warm`, now the
    default). Before, K=2 in a K=1..2 run started from the K=1 fit only, and a separate fixed-2
    run from k-means: on the Bimodal data (542 measurements, CPU) 25 differed by more than 0.01
    loss units (9 better nested, 16 better separate; the largest gaps 5.7 and 122), and the
    sorted means by more than 0.1 D for 34 measurements. A K that has a nested start is now also
    fitted from the k-means start, as one restart more (the random restarts are unchanged, so
    the run contains the separate fit), and the best restart is kept. Result: the K=2 loss is
    at most 0.045 worse than the separate fit for 3 of 542 measurements (the best restart is
    chosen before the final polish) and better, by up to 5.7, for 5; K=1 and the selected K do
    not change (0 of 542 differ); the cost is one restart per K above the first (the fit took
    32 s against 39 s in this measurement, which is within the timing noise); the regression
    comparison with the prototype still passes.

31. **Reports and plots (Phase 4, `hxex.report`; owner's decisions of 2026-10-11).** Everything
    draws from `Results` (a run's objects or an open bundle, `Results.from_run` /
    `from_bundle`), so a fresh run and a loaded bundle give the same data (tested on the arrays
    behind each plot). matplotlib only, with `Figure` objects (no pyplot, nothing shown or
    saved by the plot functions); the curves are plain NumPy and checked against the PyTorch
    model. **Plots:** `plot_spectrum` with independent layers `raw`, `picked` and `fit` (any
    subset, each from its own source, so raw-only and picked-only work without a fit; m/z is
    the default axis, as the predicted m/z of the picked bins, with the deuterium bin as the
    fallback when the picked table is absent), a choice of populations by rank, residuals,
    normalisation, bootstrap band and the K in effect (the reviewer's choice); the series plots
    `plot_series_spectra` (time by replicate grid; a single-replicate time course wraps into a near-square grid of at most 4 columns, or `columns=` sets it), `plot_populations` (fraction against
    deuterium level with draws, standard errors and centroid, replicate overlay) and
    `plot_uptake` (population means along the time course, marker area by fraction, `x='time'`
    linear or log, or `x='index'` for the test sets, the fully deuterated control as a line or as
    the last labelled point, **back-exchange corrected by default** with raw as an option);
    `Reference` / `load_reference` overlay known populations (the Bimodal solutions file). Every
    plot has a `*_data` function returning its arrays, every artist a `gid` (`component:2`,
    `band`, ...) and `Results.with_review` replaces the review layer without copying anything,
    so a GUI can redraw, toggle and review without changes here. **Tables** (CSV, no pandas): per series (`series_table`: sample, peptide, charge, counts of
    spectra, flagged, rejected and multi-population spectra, control status and retention; with
    `find_series` to select series by sample, range, charge or flags, and `format_table` to print
    any table as plain text),
    per spectrum (`data_fits` style, wide), per spectrum and population (tidy) and the
    replicate-averaged uptake (rejected spectra left out). **Files:** `PlotConfig` (series
    `flagged` by default, `all` or a list; kinds; layers; axis; format; `residual` panels under
    the spectra, off by default; `dpi` 300) and `write_figures` write
    one multipage PDF per series (or SVG/PNG per kind), made and released one series at a time;
    `RunConfig.plots` (off by default) writes them as part of a run. **Populations are
    identified by the rank of the mean at each time point, sorted independently**; tracking a
    population along the time course (linking each to the nearest mean at the next time) could
    be done entirely from the results as a post-processing step and is a possible later option
    (it fails when populations appear, vanish, cross or change K, so rank stays the default).
    **Experiments run the other way** (deuterated at the start, then exchanged into H2O, so the
    fully deuterated control comes first) are handled by the control times alone: the controls
    and the correction never assumed an order, and the plots now recognise a control by its time
    (within `atol`) rather than by being the latest; the undeuterated control is labelled `UN`.
    The readers take `undeut_time` as well as `fulldeut_time` for the control labels
    (`undeut`/`Non`, `TD`/`Full`), so a real 0 s time point can be kept apart from the
    undeuterated control (for example `undeut_time=-1`); a clash is logged as a warning, and equal
    control times are refused. In such an experiment `m0_max` may need raising if the
    undeuterated control is the last time point (it keeps the deuterium of the residual D2O).
    **Colours.** Okabe-Ito colours (distinguishable with red-green colour blindness; checked by
    simulating deutan and protan vision) for the populations (`COLORS`, by rank of the mean) and
    the replicates (beyond six populations the colours repeat with the next line style and hollow
    markers, `population_style`, so K has no cap for plotting); the fit line has a colour of its own (`FIT_COLOR`, not one of the
    populations'), because raw and picked data are grey; the populations keep their colours in
    the population map and its replicate overlay (there the replicate is the marker shape).
    The bootstrap now also stores the band of each population (`component_lo`/`component_hi`,
    additive columns of the `bootstrap` tables), drawn in the population's colour like the band
    of the total fit. **Settings are stored with the results** (the bundle manifest, `configs`): every stage's settings
    and the whole `RunConfig`, now including the modification registry (`RunConfig.registry`,
    serialisable; registries compare by mass, not only by label), the package versions, the
    creation time, project and notes. `Results.settings` (`Settings`) reads them back for plots
    (typed `picker`, `run`, `registry`); `write_bundle` warns when it is given no picker or run
    settings. Display choices (which populations are drawn, zoom) are deliberately not stored
    with the results; the picked bins are drawn as the picker's window (predicted m/z times
    1 +/- `resolution_ppm`) when that setting is known (`picked_style='auto'`), the replicate
    overlay of the population map carries the same guides at 0 and at the number of amides as
    the single cells, and the raw line is broken across large gaps in m/z. Not stored: the
    input file paths or checksums (the ids hold only file names; an open question, since a path
    can reveal a local layout).
    **K comparison and run overviews:** `plot_k_comparison` draws one spectrum at every K tried
    (loss, the selection score relative to the best, the minimum-population rule, the selected
    and the reviewer's K); `plot_controls_overview` (retention and undeuterated means with their
    limits, fallback reasons), `plot_quality_overview` (flag counts; fit against intensity as the
    share of flagged spectra per grid cell, since overlapping markers cannot show which kind
    dominates where thousands of spectra pile up, with sparse cells as points; the thresholds;
    flagged share per sample) and `plot_k_selection` (selected K, and the K in
    effect at each time point) summarise a run; `write_figures` writes them as `hxex_overview`
    (`PlotConfig.overview`, on by default). Not built yet: the sequence-coverage uptake map and
    state comparison (butterfly) plots.

32. **Large-dataset benchmark (Phase 4 target, 2026-10-11).** An unpublished SpecExport dataset of
    26,333 spectra (five samples, 1,372 series, six time points including both controls, up to
    four replicates), kept outside the repository. Settings as in its pyHXExpress runs (picker:
    zero filling 3, noise 50, N-terminal subtraction 2, 50 ppm, envelope threshold 0.1); fit
    K = 1 to 3 with noise estimated from replicates; bootstrap N = 20 at fixed K = 1 and 2;
    controls and quality flags. 152 spectra were dropped (empty or no signal), 26,181 fitted.

    | Stage | GPU (RTX A4500, float32) | CPU (Threadripper 3960X, 24 cores, float64) |
    |---|---|---|
    | Read and pick | 31 s | 31 s |
    | Fit, K = 1 to 3 | 4.9 min | 77.4 min (16x slower) |
    | Bootstrap, controls, quality | 41.6 min | not finished (stopped after about 3 h on one fifth of the data) |
    | Bundle write (332 MB with raw spectra) | 6 s | |
    | Peak memory | 7.1 GB GPU, 3.9 GB RAM | |

    The whole GPU run took about 48 minutes, so the Phase 4 target (10,000 fits in a measured
    time on stated hardware) is met. CPU and GPU select the same K distribution (K = 1/2/3:
    21,950/3,953/278 on CPU, 21,903/4,008/270 on GPU). On the CPU PyTorch used 24 threads at about
    20 cores' load; part of the time is single-threaded Python (k-means starts, bookkeeping), and
    float32 would roughly halve the arithmetic: both are possible speed-ups, not done. The CPU is
    fine for fits but not for full bootstraps at this size. **Controls:** 283 of 1,372 series fell
    back to theoretical values, 186 because the series has no fully deuterated spectrum; the
    retention of the series that passed was 0.64 (0.56 to 0.72 for the middle 80 %), much lower
    than on the Bimodal set (0.81 to 1.00), so the default lower limit of 0.5 is close to this
    dataset's normal range (44 series below it). The limits are settings; whether to derive them
    per dataset is open. Quality flags: 1,497 spectra (5.7 %; poor fit 264, low intensity 633,
    replicate outlier 835).

33. **mzML reader (`hxex.io.mzml`; owner's decisions of 2026-10-12, needed before fragment data).**
    A small mzML 1.1 reader written with the standard library (no lxml, so no new dependency):
    plain or indexed files, uncompressed or zlib 32/64-bit arrays, profile or centroid scans;
    numpress files are refused with a clear message. Thermo `.raw` files are converted outside
    hxex (ThermoRawFileParser, the command is in the module docstring); reading `.raw` directly
    stays a Phase 6 item. `MzmlFile` gives the scan table (MS level, time in seconds, filter
    string, precursor m/z and charge, activation, profile or centroid; no arrays) and single
    scans through the file's index. `read_mzml(runs, targets)` gives a `SpectrumSet` for targets
    (peptides, the intact protein or fragment ions, selected by MS level, activation such as
    ETD, precursor and retention window) in **both layouts**: a run without a fixed time makes
    each scan, or each group of scans (every N, or time windows), a time point at the mean scan
    time plus an offset (continuous labelling, as in the top-down examples); a run with a fixed
    time (an LC-MS file per labelling time, or a control file) sums each target's scans within
    its retention window into one spectrum. Summing is built in (no more pre-summed files):
    profile scans are interpolated onto the union of their m/z points without bridging the
    stretches the vendor left out, centroided peaks are merged within a ppm tolerance. A series
    is a sample and a target; replicate runs share it. Checked on a real Thermo file (Orbitrap,
    885 scans, ETD MS2 with precursor 1012 at charge 20): scan table in 0.5 s, one scan by index
    in 0.2 s, the intact protein (charge 20) as 23 sums of 20 MS1 scans in 2.7 s, picked in every
    spectrum (204 bins), with the deuteration falling along the run (a back-exchange
    experiment). **Kinetics runs (owner, 2026-10-12):** the time course data run from D2O to
    H2O; the scan times may need adjusting for kinetics fits (an offset exists, `time_offset`);
    early time points are poor in some datasets and need a way to be excluded; large files
    should be sampled (some time points, for example spaced in log time) rather than read whole.
    **The fixed-time (LC-MS) layout on real data:** one file of an unpublished bottom-up ETD
    dataset, 18 peptides with retention windows from a list: all found (44 to 139 MS1 scans
    summed per window, m/z equal to the list's), with plausible envelopes. **Summing checked
    against OpenMS** (the same file summed 10 MS1 scans at a time by
    OpenMS SpectraMerger, 68 sums; our sums rebuilt from the same component scans, compared in
    the intact protein's window): the first version summed on the union of the scans' m/z points,
    which was 7 times denser than any scan, because scans of one acquisition have the same
    spacing but grids offset by up to half a step (0.1 step at the median); profile scans are now
    resampled onto one grid with the scans' own spacing (the densest scan's points plus the
    others' where it has no data), like OpenMS. Result: same time (the mean scan time), area
    within 0.3 to 0.5 %, profile correlation 0.9999, picked bins with the median ratio 1.00 (0.95
    to 1.01 for the middle 80 % of bins above 5 % of the maximum), envelope centre within 0.03 to
    0.11 bins. **Picker finding:** on these summed intact-protein spectra (charge 20, 20 ppm
    window, about 8 points per window) the shoulder rule (a window of 10 or more points whose
    maximum is at its edge is dropped) removed 18 to 23 genuine isotope peaks per OpenMS sum and
    1 to 5 per ours, moving the envelope centre by up to 1.4 bins; the rule was unvalidated (no
    earlier dataset had windows that full) and needs a decision for high-resolution profile data
    (`PickerConfig.shoulder_min_points` turns it off for now). Also fixed: OpenMS writes the
    filter string as a userParam. Not done yet: fragment target lists (Phase 6, with the
    unverified fragment amide counts).

34. **Peak picker: the composition ladder (owner's decision of 2026-10-12; method in
    docs/PEAK_PICKING.md).** The shoulder rule turned out to be a symptom: on an intact protein it
    removed genuine apexes because the predicted positions had drifted from the peaks. The ExMS
    ladder (a fixed 1.003355 / 1.004816 / 1.006277 Da) does not match real isotope clusters, which
    mix deuterium (1.00628 Da) with natural heavy isotopes (about 1.0025 Da per step); measured
    steps were 1.001 to 1.0025 Da for undeuterated and 1.0044 to 1.0054 Da for deuterated
    peptides. **Default now (a deliberate difference from pyHXExpress):** each bin at the
    abundance-weighted mass of its cluster, the natural isotope clusters of the composition
    (brainpy) convolved with a binomial deuterium distribution whose level comes from a first pass
    with the ExMS ladder and is estimated twice (`PickerConfig.ladder`, `exms` keeps the old
    ladder; the pyHXExpress oracle tests pin it). No peak is assigned to a bin to set the positions
    (the owner's objection to recalibrating spectra from their own peaks). Results: intact ions in
    high-resolution scans within about 1 ppm (ExMS up to +2.2 ppm), fragments 95 % inside their
    windows (ExMS 50 %), the intact protein's real peaks lost to the shoulder rule 339 to 17,
    peptide SpecExport data and the Bimodal accuracy unchanged. An apex rule (tallest local maximum
    in the window) was tried and dropped: on coarse peptide data it zeroed real peaks lying outside
    their windows. Open: the shoulder rule; per-spectrum offsets (diagnostic planned, calibration
    from the undeuterated control possible); constant offsets of -14 to -49 ppm for fragment ion
    types (`c-1`, `z-dot`, `M-NH3`), to be settled with the fragment ion verification.

35. **Ion-type mixtures for fragments (owner's decisions of 2026-10-12; `hxex.picking.envelopes`,
    method in docs/PEAK_PICKING.md).** Fragment spectra hold several ion types one hydrogen apart
    (c-1, c′, c+1; z, z•, z′); pyHXExpress handled this with `NA_envelope` `mixedN` (and `infer`).
    The fragment offsets of item 34 came from this: with the right type, fragments of an
    undeuterated reference sit within 1 to 3 ppm. hxex fits the weights of `components` types (the
    lightest given) to the series' undeuterated references (non-negative least squares, three rounds
    of picking and fitting), falls back to the spectra at `envelope_fallback_time` (the end-state
    run; the owner: a fallback when the undeuterated control is missing or poor) and then to the
    theory, both marked; the mixture sets the picker's positions and the fit's envelope (`infer`
    uses the reference envelope in the fit). `RunConfig.envelope`, the `envelopes` bundle table.
    Checked on a 12-residue peptide (ETD, back-exchange kinetics, one replicate, private data):
    short fragments almost pure c′ and z•, the longest about one third c′ (z•) and two thirds c+1
    (z′); misfit below 3 % with three types against 80 to 97 % with one; deuteration of c fragments
    within 0.07 D of pyHXExpress for 10 of 11, z fragments 0.05 to 0.23 D higher (most likely
    pyHXExpress's, item 41). The mzML reader gained `filter_excludes` and `filter_contains` on
    targets (scan types such as with or without in-source activation, Thermo's `sid=`; activation
    short names now match exactly, so `ETD` no longer also selects EThcD scans), `min_time` (leave
    out early time points) and `max_spectra` (time points sampled evenly in log time, only those
    scans read), for the kinetics runs. Next: the z offset, the other fragmentation types and
    peptides, fragment target generation (done, item 36), and exchange-rate fitting.

36. **Fragment targets and terminal groups (`hxex.fragments`, `hxex.chem`; 2026-10-12).**
    Sequences may carry terminal groups in modX style (`Ac-PEPTIDE`, `PEPTIDE-NH2`); the
    groups known are `H-`, `Ac-`, `-OH` and `-NH2`, and they change the composition but not the
    residues or the amide count. The m/z with `-NH2` equals pyteomics' for every ion type tested;
    pyteomics reads `Ac` as actinium, so `Ac-` is checked against its composition instead.
    `fragment_entries` turns a parent entry and a list in pyHXExpress's syntax
    (`'c-1:ALL z-dot:2-6 M'`) into one entry per fragment and charge, sliced by residue, N-terminal
    ions keeping the parent's N-terminal group and C-terminal ions its C-terminal group; `extra`
    names the parent and the fragment, and is kept in the mzML read record. `fragment_targets`
    makes the MS2 targets. On all 100 pyHXExpress fragment lists of two peptides of the private fragment dataset
    (back-exchange kinetics of five peptides) every row is among the generated entries (same sequence with
    terminal group, range, charge and ion type). **Deliberate differences** from
    `hxex_fragments.py`: residues are tokens (it slices characters); `ALL` and open ranges are
    lengths 1 to the number of residues (it used `start_seq` to `end_seq`, the same only when the
    parent starts at 1); a parent `modification` other than `ion_type:M` is refused rather than
    copied to every fragment (a composition change, extra sites or an explicit m/z cannot be
    assigned to fragments). The default charge limit is the parent's charge minus one, as in
    pyHXExpress; some of its lists were made without it (`limit_charge=False`).
    **Unresolved (the owner cannot check the sample; that peptide is set aside and the others are
    used instead):** the spectra of one peptide (11 residues, a C-terminal amide in its metadata) match the C-terminal free acid, not the
    amide given in the metadata (MS1 of the equilibrium run: no peak at the amide's monoisotopic
    m/z, and the series fits the acid within 5 ppm against -10 to -23 ppm as the amide; the
    isolation windows hold the whole envelope, item 38). In the MS2 spectra the unreacted
    precursor is at the acid's m/z (intensity 9,795 against 9 at the amide's at one charge, 7,358
    against 134 at another), two z ions match the acid (-0.5 and -11 ppm) and no amidated one is
    present, while the c ions (holding both glutamines) fit without a shift, so it is not a glutamine deamidation. With `-NH2` the z bins are 0.984 Da light, about
    one deuterium.

37. **Where the peaks were found: the m/z screen (2026-10-12; docs/PEAK_PICKING.md).** The picker
    records each picked peak's position (`BinnedSet.peak_mz`, the apex refined by a parabola on
    profile data) and each spectrum's intensity-weighted median offset from the predictions
    (`mz_offset_ppm`); the bundle keeps both. A new quality flag, `mz_shift`, marks a series whose
    median offset is more than `QualityConfig.max_mz_offset_ppm` (15; None turns it off) from the
    median over all series, which removes the calibration. It was prompted by the peptide of item 36:
    the ion-type mixture fitted the wrongly composed z ions well enough, while their peaks sat 23
    to 41 ppm off (within 3 ppm as the acid; fragments of the 12-residue peptide within 2 ppm). On 3,000 spectra of
    the intact-peptide benchmark the calibration offset is +5.5 ppm and 2 % of series are flagged,
    mostly with a consistent offset (spread 3 to 8 ppm), which points to a wrong assignment rather
    than noise; what they are was not examined.

38. **Precursor isolation windows (2026-10-12).** The scan table now has each MS2 scan's
    isolation window (`isolation_lower`, `isolation_upper`, from mzML's isolationWindow). A window
    that cuts the precursor's isotope envelope would change the fragments' envelopes (long
    fragments more), and as the envelope moves with deuteration the cut would change along a time
    course. Measured on the private fragment dataset ( the fraction of the precursor's envelope inside
    the window, natural isotopes and binomial deuteration of the backbone amides): the kinetics
    files (+/- 5 m/z) keep it whole at every level for every peptide and charge, except the 24-residue
    peptide at 3+ above about 75 % deuteration (0.94 at 75 %, 0.64 at 90 %) and, at 90 %, two
    others at 2+ (0.97 and 0.99). The undeuterated controls (+/- 1 to 2.5 m/z, different centres) keep their
    undeuterated envelope whole except the 8-residue peptide's (+/- 1 m/z: 0.98 at
    3+, 0.89 at 2+), whose
    reference envelopes therefore lose some heavy isotopes. So isolation does not explain the
    heavier ion types of the long fragments of item 35. Not modelled; to revisit if data with
    narrow windows on deuterated samples come up.

39. **The 12-residue peptide, all fragmentation types against pyHXExpress (2026-10-12; private
    fragment dataset).**
    hxex read the undeuterated control and both kinetics replicates per type, generated
    pyHXExpress's fragment list (`fragment_targets`), fitted K = 1 with ion-type mixtures (three
    components) and was compared with pyHXExpress's uncorrected deuteration at the same scans
    (pyHXExpress settings: no N-terminal subtraction, 40 ppm, noise as a percentage). The m/z screen
    flagged the precursor-related species (`M-NH3`, full-length `z`) at about -30 ppm, and weak
    series without a good mixture.
    - ETD 3+: c fragments within 0.07 D (lengths 2 to 9), z fragments 0.05 to 0.23 D higher (as in
      item 35).
    - ETD 4+: 2+ c fragments within 0.03 D, z fragments 0.0 to 0.25 D higher; 1+ c6 to c9 0.35 to
      0.6 D **lower**. For the same fragment from the 3+ and the 4+ precursor, hxex agrees with
      itself (c8, c9 within about 0.15 D at every time) while pyHXExpress's 4+ values are 0.3 to
      0.6 D above its own 3+ values and level off near 0.7 D instead of decaying: the difference is
      pyHXExpress's.
    - Both programs see c4 higher (0.45 against 0.14 D at the start) and z6, z8 about 0.6 D lower
      from the 4+ precursor than from the 3+: a property of the data (for example hydrogen
      scrambling depending on the precursor charge), not of either program. The owner: it may
      well depend on the charge; noted, no action.
    - EThcD 2+: the undeuterated file records charge 3 for its 2+ precursor, so a
      charge filter selected nothing; `fragment_targets(precursor_charge=None)` selects by the
      other conditions. Then z7, z8 0.11 D higher. Only 8 fragments in pyHXExpress's list.
    - UVPD 3+: a1 to a4 within 0.07 D; a5 to a7 (2+) 0.1 to 0.2 D higher with a wide spread; few
      x fragments, mostly without a good mixture.

40. **Three more peptides against pyHXExpress (2026-10-12; private fragment dataset).**
    As item 39, seven acquisition types ("Low", no in-source activation): 8 residues: ETD 3+,
    EThcD 2+, UVPD 3+; 14 residues: EThcD 2+, UVPD 2+; 24 residues: ETD 5+, EThcD 3+. Every fragment of pyHXExpress's
    lists was generated (238). The equilibrium ("Final") run, summed, served as the fallback
    reference where the undeuterated one failed or did not exist (UVPD of the 8- and 14-residue peptides). For
    usable fragments (a good mixture, no m/z flag), the median difference from pyHXExpress's
    uncorrected deuteration was within 0.03 D for c, a and z ions of the 8- and 24-residue peptides (c: 82 to
    100 % of fragments within 0.1 D); the 14-residue peptide's z ions were 0.13 D higher (0 to
    0.23), like the ETD 3+ z ions of item 39 (the open z offset, item 35). Larger single differences: the 24-residue peptide's c13 to c20 0.5 to 0.7 D
    lower on few matched points (6 to 15; pyHXExpress fitted 16 sampled times), the precursor ions
    (`M`) of the 8- and 24-residue peptides (-0.5 to -0.7 D, +0.2 D), and one x ion (-0.9 D, fallback
    reference). Not examined further. Data quirks met: another undeuterated file records 3+ for its 2+
    precursor (as in item 39), and two peptides' undeuterated files have `sid=1.00` on every scan while the kinetics files mark in-source activation with
    `sid=`, so one `filter_excludes='sid'` cannot serve both: the comparison read them separately.
    Since then a run can replace the targets' scan conditions for its file (`MzmlRun`:
    `activation`, `precursor_mz`, `precursor_charge`, `filter_contains`, `filter_excludes`; the
    default `'target'` keeps the target's, None removes the condition; recorded with the read):
    one read with `filter_excludes=None` on the control gives the same spectra.

41. **The z-fragment offset: a closure test (2026-10-12; private fragment dataset).** In ETD every
    atom of the precursor ends up in the c or the z fragment, so at each scan the deuteration of
    c_i plus that of its complement z_(n-i) equals the precursor's, whatever hydrogen scrambling
    or transfer happens. This checks fragments without a reference program. With the unreacted
    precursor (charge-reduced species need not carry the same deuteration): on the 12-residue
    peptide (ETD 3+), where hxex's z fragments were 0.05 to 0.23 D above pyHXExpress's, hxex closes
    within 0.10 D for every 1+ pair (median over scans -0.01 to -0.10) and pyHXExpress misses by
    up to +0.21 D (its precursor is 0.26 D below hxex's, its z fragments lower, its c fragments the
    same). Where the two programs' z fragments agree, they close alike: 8-residue peptide ETD 3+
    -0.07 (hxex) and +0.04 (pyHXExpress), 24-residue peptide ETD 5+ -0.19 and -0.16. So the z
    offset of items 35 and 39 is most likely pyHXExpress's (its envelope reference and ion-type
    set); the 14-residue peptide's z offset (item 40) cannot be tested (no usable c fragments).
    The 2+ z fragments that the m/z screen flagged do not close in either program. A closure check
    could become a fragment-level quality measure (open).

42. **Read records for every format, target columns, and the m/z-flagged series (2026-10-12).**
    - `read_spectra` (HX-Express, SpecExport, pooled) now records its read like `read_mzml`: the
      format, project, data directory (`record_paths` `'path'` or `'name'`), entries, filters,
      control times, and each entry's input identity (`hxex.io.identity`: an HX-Express file's
      SHA-256; a SpecExport folder as a whole, its file count, size and one SHA-256 over every
      file's relative name and contents). `read_spectra_source` repeats it, warning about (or,
      with `on_change='error'`, refusing) a changed input and keeping the old identity under
      `previous`; `read_source` repeats a read of any format. On the intact-peptide benchmark
      (1,376 entries, 26,333 spectra) hashing costs about 1 s of a 32 s read and the record is
      0.46 MB.
    - Result tables of mzML reads (`spectra_table` and the others built on the index columns)
      add `fragment`, `ms_level`, `activation`, `precursor_mz` and `precursor_charge` (0 or NaN
      where the target set none), from the read record's series map; other formats are unchanged.
    - The m/z screen on the whole intact-peptide benchmark: 35 of 1,376 series (2.5 %; run
      calibration +5.2 ppm), 27 of them 2+ (half the dataset is 2+), weak (median at the 26th
      intensity percentile) and noisy (offset spread within the series 16.6 ppm against 4.8). 25
      have different offsets in the undeuterated and the deuterated spectra (an overlapping
      species or noise, which deuteration moves relative to the peptide); 10 have one offset
      throughout (a wrong composition or assignment). 8 peptide ranges are flagged in two or more
      series (of 51 series of those ranges), which points at the peptide rather than a spectrum.
      Individual series were not examined.

43. **Thermo `.raw` files read directly (`hxex.io.thermo`; owner's decisions of 2026-10-12).**
    - Approach: Thermo's own RawFileReader (.NET assemblies `ThermoFisher.CommonCore.Data` and
      `.RawFileReader`) called through pythonnet, as in the Apache-2.0 project
      wilburnlab/constellation (which has no format decoding of its own). No `.exe` and no
      conversion step. pythonnet is the optional extra `hxex[thermo]` (lower bound 3.0.3, checked
      to load the assemblies); the runtime is pythonnet's choice (Mono on Linux and macOS, the
      .NET Framework on Windows, .NET 8+ with `PYTHONNET_RUNTIME=coreclr`; Mono 6.12 tested here,
      and the owner confirmed Windows 11 with both the built-in .NET Framework and .NET 8 on
      2026-10-13). The assemblies are the user's (they come with ThermoRawFileParser), found through
      `thermo_dir` or `HXEX_THERMO_DIR`; hxex never bundles or downloads them. A pure-Python
      decoder of the format was rejected as too large and fragile for now.
    - API: `ThermoFile` mirrors `MzmlFile` (`scans()`, `scan()`, `read_scans()`); `read_thermo`
      takes the same `MzmlRun`, `MzmlTarget` and `MzmlConfig` as `read_mzml` and shares its code
      (the per-run loop and the repeat are now format-independent in `hxex.io.mzml`). The record
      has `reader: "thermo"` plus `thermo_reader` (the RawFileReader version); `read_source` and
      the result tables' target columns accept it. Unit tests use a stand-in for the
      RawFileReader; `HXEX_THERMO_EXAMPLE` (a `.raw` with its ThermoRawFileParser `.mzML` beside
      it) plus `HXEX_THERMO_DIR` compare a real file.
    - Matching ThermoRawFileParser 1.4.5 (`-p`) needed three details: the arrays are the scan's
      segmented data read *with* the scan statistics (without them the m/z values move by about
      0.2 ppm); `IncludeReferenceAndExceptionData` must be on, as ThermoRawFileParser sets it
      (else lock-mass and reference peaks are zero: up to 145 points per scan); and a symbolic
      link must be resolved before opening (the RawFileReader otherwise fails with an overflow).
      A NumPy view of a .NET array (buffer protocol) crashes Mono at interpreter exit, so arrays
      are copied.
    - Checked against ThermoRawFileParser's mzML on four Orbitrap files: two top-down HDX files
      (885 and 105 scans, ETD MS2), an infusion file with ETD, EThcD and MS3 (2,147 scans) and a
      data-dependent top-down file with ion-trap HCD and MS3 (7,264 scans). Every scan's id, MS
      level, filter, precursor m/z and charge, isolation window and centroid flag agree (times
      to 1e-12 relative, as the mzML writes minutes as text), and the arrays of 120 scans per
      file are identical in m/z and agree in intensity within the mzML's precision. Scan table:
      0.5, 1.0 and 2.4 s against 0.4, 0.4 and 1.0 s from the mzML.
    - One deliberate difference: the activation of MS3 scans is that of the last stage (the one
      that made the scan, for example ETD after HCD; EThcD as ETD plus supplemental beam-type
      CID). ThermoRawFileParser writes the last stage in the infusion file but the first stage
      (HCD) for the 1,517 data-dependent MS3 scans, so the two disagree there; MS1 and MS2 agree
      everywhere.
    - License: the owner added a GPL section 7 permission to link or combine hxex with the
      RawFileReader and convey the result (`LICENSE-EXCEPTION`, shipped with `LICENSE`; the
      SPDX expression stays `GPL-3.0-or-later`, as SPDX has no identifier for a custom
      exception). Open: Thermo's license forbids "reflection" (clause 2.1), which is how
      pythonnet calls .NET. Alternatives if that matters: a compiled C# bridge called through
      .NET hosting (as the Rust `thermorawfilereader` does), a helper process, or decoding the
      format; or a written answer from Thermo.

44. **Progress display (2026-10-12).** The hooks reported a fraction only, with fixed shares of
    the 1000 units per stage (fit 30, bootstrap 65), so a display could neither name the stage
    nor estimate the time left: on the intact-peptide benchmark the CPU fit took 77 min and the
    bootstrap over 3 h (item 32). `hxex.hooks.start_stage` now tells a hook that has a
    `start_stage(name, start, end)` method which units are which stage (plain functions are
    unchanged); `run` names `pick`, `fit`, `bootstrap`, `controls and quality`, `write` and
    `figures`, and keeps the last 10 units for the stages after the bootstrap, so the total is
    reached only at the very end. `hxex.hooks.ProgressLog` is such a hook: through `logging`
    (logger `hxex.progress`) it logs each stage's start and end, and at most every 30 s the
    stage's percent, time taken and time left, estimated from that stage's own rate.
    - Results were written only at the very end, so a run stopped or failed during the bootstrap
      lost its fits (77 min on the CPU benchmark). The controls and quality flags need only the
      fits, so a run with a bundle and a bootstrap now goes fit, controls and quality, writes the
      bundle with `pending: ["bootstrap"]` in the manifest (`Bundle.pending`), bootstraps, and
      rewrites it complete (bundle writes were already atomic: a temporary directory moved into
      place). Stopped during the bootstrap, the fit bundle stays and the log says so;
      `hxex.run.resume_bootstrap(path)` bootstraps it with the recorded run settings (or new
      bootstrap settings) and rewrites it, keeping the stored spectra, notes and review, with the
      same result as an uninterrupted run (tests/test_run.py). Stopped before the fits are done,
      nothing is written, as before. `FitResult.select` (the inverse of `merge`) splits a stored
      fit into chunks. The bundle is written twice (6 s on the benchmark).

45. **Command line and project file (`hxex.cli`, `hxex.config`; owner's decisions of 2026-10-12).**
    - One TOML project file (no YAML, so no new dependency: `tomllib` reads it, the template is
      written as text): `project`; `[input]` with the format (`specexport`, `hxexpress`,
      `pooled`, `mzml`, `thermo`), the folder and filters, or for scan files a peptide table
      (pyHXExpress's metadf) with `[input.target]` scan conditions, optional `[input.fragments]`
      (the fragment syntax of `hxex.fragments`, expanded by `fragment_targets`), `[input.scans]`
      (`MzmlConfig`) and `[[input.runs]]`; a folder format may also take the table instead of
      finding files by name (the Bimodal set needs it); `[run]`, the `RunConfig` sections laid
      over their defaults key by key; `[output]` (bundle, tables, figures, notes). A key left
      out keeps its default; `false` stands for None (TOML has no null), and a `[run.bootstrap]`,
      `[run.controls]` or `[run.plots]` table turns its stage on. Relative paths are relative
      to the project file. Unknown keys are refused, as by the configs' own `from_dict`.
    - Commands: `init` (a template with every default, per format; it reads back as the
      defaults), `check` (the settings, and the entries or, for scan files, each run's scans per
      target, as `read_mzml` chooses them: `hxex.io.mzml.target_scans`), `run` (read, run, tables;
      `--bundle`, `--overwrite`, `--device`), `resume` (`resume_bootstrap`), `report` (tables
      and figures from a bundle) and `info` (what a bundle holds, `--verify`). Messages and
      progress (`ProgressLog`, every 30 s or `--progress-every`) go to stderr, results to stdout;
      exit status 0, 1 (an error, as a message), 2 (usage), 130 (stopped). Ctrl-C stops after
      the current step; a second one aborts.
    - Checked on the Bimodal set (566 spectra, K = 1 to 2, bootstrap N = 5, CPU): 1 min 33 s in
      all (read 3 s, pick 1 s, fit 32 s, bootstrap 55 s), selected K 193/373. Ctrl-C during the
      bootstrap stopped within 5 s and left the fit bundle; `hxex resume` completed it with the
      same K distribution. The per-entry "reading" message of the folder readers is now debug.

46. **CPU speed (2026-10-13; intact-peptide benchmark, Threadripper 3960X, 24 cores).** A fit of
    100 series (1,847 measurements, K = 1 to 3, the benchmark's settings) took 272 s on the CPU in
    float64 with PyTorch's default 24 threads, 20 cores busy throughout. Profiled, the binomial
    convolution was most of it: the pmf was evaluated (two `lgamma`, an `exp`) at every shape
    point and bin, an `(L, B, M, K)` array per step.
    - **Integer convolution** (`hxex.model.integer_grid`, `IntegerGrid`): picked bins are whole
      numbers, so `b - j` takes only `B` values. The pmf is evaluated once per integer and the
      shifts are windows of one padded array (`unfold`) when every measurement's bins are 0, 1,
      2, ... (all picked data), or gathered otherwise; other bins keep the direct evaluation.
      The values are the same (bitwise on the benchmark subset; tests check values to 1e-13 and
      gradients to 1e-9 for all three paths); padded positions keep their bin-0 value.
      `MixtureBatch` builds the grid once. A batched matrix product and a grouped 1-D
      convolution were tried and were slower than the gather.
    - **Polish**: the Levenberg-Marquardt Jacobian came from `vmap(jacfwd)` per measurement on
      the direct convolution; it is now 3K forward-mode passes of the batched model (one per
      parameter column; each measurement depends on its own row only), with the integer path.
    - Timings of the 100 series: 272 s, then 143 s (gather), 130 s (batched polish), 86 s
      (windows), all float64 with 24 threads, losses equal to 6 digits; float32: 56.5 s (24
      threads), 52.5 s (12). Threads in float64: 8: 79 s, 12: 73 s, 24: 86 s, 48: 174 s (the
      48 are hardware threads of 24 cores).
    - **float32 by default on the CPU** (`Backend.dtype = "auto"` is now float32 everywhere; the
      prototype used float64 on the CPU). On the 100 series it changed the selected K of 3
      measurements (1,386/438/23 to 1,389/438/20).
    - **Chunk memory**: the run stage sized CPU chunks with a factor measured for the direct
      convolution (75 pmf-sized arrays per measurement). Measured peaks with the integer
      convolution: 9.2 on the CPU (945 and 3,727 measurements, float32) and 7.6 to 8.5 on the
      GPU, so `WORK_FACTOR` is 12 (75, `WORK_FACTOR_DIRECT`, for non-integer bins): about six
      times larger chunks for the same budget.
    - **The whole benchmark** (26,181 measurements) through `analyse` on the CPU with the
      defaults: fit in 17.1 min (77.4 before), selected K 21,906/4,008/267 (GPU run:
      21,903/4,008/270; old CPU run: 21,950/3,953/278), the same K as the GPU run for 99.75 %.
      The full test suite, the slow regression tests and the GPU parity tests pass.
    - Threads: `threads = N` in the project file or `--threads N` (`run`, `resume`, `report`;
      the flag wins) sets PyTorch's CPU threads for that command and restores them afterwards.
      The library does not set them (a process-wide setting); the default stays PyTorch's.
    - Not done: the k-means starts (scipy, per measurement in Python, about 10 % of a fit).

47. **Nomenclature (owner's decision of 2026-10-13, following published precedent).** A component's
    share of the molecules is its `population` (was `fraction`): `KFit.populations`,
    `BootstrapResult.population_se` and `population_ci`, the tables' `population`,
    `population_se`, `population_<r>` and `population_se_<r>`, the tidy tables' `population`
    (was `population_fraction`), and the plot data and labels. Deuteration is `mean`, the absolute
    deuterium in deuterons (back-exchange corrected when there are controls), and `fD`, the
    fractional deuteration `mean / n_amides` (0 to 1), of a component or of any grouping. Components
    are named by K and rank, not slow and fast, and the values mean the same for forward
    labelling and reverse exchange. The bundle schema is version 2 (bootstrap columns
    `population_se` and `population_ci`, version 1 `frac_se` and `frac_ci`); hxex reads both
    (checked on the intact-peptide benchmark bundle, version 1). `min_population` kept its name.
    The full table of names and their equivalents in pyHXExpress and pyHDX is in
    `docs/ANALYSIS.md` ("Nomenclature").

48. **Post-fitting analysis, phase 1 (2026-10-13; design in docs/ANALYSIS.md).**
    - `hxex.analysis.species_uptake(results, ks)`: one row per sample, peptide (or fragment),
      time, fixed K and rank, with `mean` (deuterons, back-exchange corrected), `fD`
      (`mean / n_amides`), `population` and their standard errors, replicates and charge states
      combined (`sqrt(max(mean(se_i^2), s^2) / n)`), a charge-trend flag, the run's
      `nterm_subtract` and undeuterated time; rejected spectra left out.
    - `residue_map` (per time) and `coverage_table` (per sample, at the undeuterated control by
      default or chosen times; redundancy counted in peptides or spectra), optionally checked
      against a FASTA sequence per sample.
    - Checked against pyHDX 0.4.3 (run from its own code on 192 peptides of one sample at 4 s of
      the intact-peptide benchmark, K = 1): residue fD and its standard error agree to 2e-16,
      redundancy and resolution are identical at every exchanging residue. pyHDX leaves 33 more
      residues of 173 empty, every one among the first `nterm_subtract` residues of some other
      peptide (it decides exchange once for the protein; hxex per peptide). The resolution of
      the owner's earlier figures (11 at residues 153 to 165 for most samples) came from
      pyHXExpress's `plot_resred`, which took the first state's residue numbers for every state;
      pyHDX and hxex give 4 to 6 there.
    - `hxex.report.coverage`: `plot_coverage` and `coverage_figure` (pyHDX's wrapped layout,
      one bar per peptide, gouldian colours embedded with attribution, CC BY 4.0) and
      `plot_redundancy_resolution` (the owner's overlay style). On the benchmark: 16,626 rows in
      2 s; 70 species flagged for a charge trend; 52 of 5,542 K = 1 values above fD 1.05, mostly
      at late times (series whose fully deuterated control fell back or is low).

49. **The K=2 components depend on the noise model (2026-10-13; intact-peptide benchmark, WT at
    4 s, 1,034 spectra).** Residue profiles of the K=2 description put the high-deuteration
    population at fD 1.0 to 1.6, against 0.4 to 0.9 in the owner's pyHXExpress analysis (K=1 and
    the K=2 low population agreed: correlation 0.96 and 0.89 with the owner's residue values;
    the high population 0.11). Peptide by peptide (uncorrected, main fits): hxex's high component
    had a median population of 0.06 (below 0.05 in 495 spectra) and fD 0.86 (above 1 in 422),
    with n at its bound (1.2 times the amides, the same bound as pyHXExpress's `Nex_Max_Scale`)
    in 780; pyHXExpress's had population 0.34 and fD 0.45. Scored by least squares on hxex's
    spectra, pyHXExpress's solution was better in 905 of 1,034, which first looked like an
    optimiser trap in hxex; **under hxex's own loss it is not**: the run estimated the noise from
    replicates, so 1,024 of these spectra were fitted with the Poisson loss, under which the
    stored fit beats pyHXExpress's solution in 1,025 of 1,034, and hxex's optimiser started from
    pyHXExpress's solution moves away from it in 1,006 and ends at the stored loss (median
    difference 0). The Poisson loss weights the low-intensity tail more than least squares
    (pyHXExpress, `Y_ERR = 1`) and is best served by a small component reaching into the high
    mass tail; that many starts end at the same loss also shows the K=2 split is poorly
    determined here. **Refitted with the Gaussian noise model** (every WT spectrum, K = 1 and 2,
    191 s on the CPU) hxex agrees with pyHXExpress: high component fD 0.40 and population 0.40
    (pyHXExpress 0.45 and 0.34), fD above 1 in 46 (54), within 0.1 in both fD and population of
    pyHXExpress in 638 of 1,034 (232 with Poisson), correlation 0.75 (0.23). The selected K
    barely moved (K=1/2: 696/338 against 702/300 and 32 at K=3). Also from pyHXExpress's code:
    its reported `Dabs_k` and `pop_k` are means over the bootstrap draws whose populations lie
    within `Pop_Thresh` and `1 - Pop_Thresh`, not the main fit. Open for the owner: the noise
    model for K=2 decompositions (the Phase 5 choice). Decided (owner): the fit's bound on `n`
    stays a user setting, and the analysis filters K >= 2 descriptions (`SpeciesFilters`): per
    spectrum, populations within 5 to 95 % and no component above fD 1.1; per peptide and time,
    fD error below 0.25. With them, the WT 4 s residue values against the owner's
    (`B5phospho_fitsdf_byres`): stored Poisson fits, 106 peptides keep K=2, correlation 0.96
    (K=1), 0.94 (rank 1), 0.84 (rank 2; medians 0.63 and 0.62); Gaussian refit, 181 peptides,
    0.98, 0.96 and 0.93 (rank 2 medians 0.56 and 0.62). (A first version filtered on the fitted
    `n` above 1.1 times the amides, which kept only 50 peptides and correlated 0.70; the owner
    meant `n * p`, the reported deuteration.) The K=1-between-ranks filter (per peptide and time, K=1 value
    between the K=2 ranks) removes 388 descriptions over the whole benchmark (stored fits) and
    100 over WT (Gaussian refit), but at WT 4 s only one and none, so the correlations above do
    not change. (Setting `SpeciesFilters.k1_between_ranks`.)

50. **The one-or-two-population choice (phase 2; 2026-10-13).** `choose_populations` applies the
    tests of `choose_fits` per residue (separation at 50 % confidence, both errors below 0.25,
    the K=1 value between the ranks, lever-rule population within 5 to 95 %), with the
    population error propagated from all three values (checked against numerical derivatives);
    `plot_population_choice` draws the owner's `rfufits` figure (curves solid where chosen and
    dashed elsewhere, chosen markers sized by population, K=1 as open circles, a population
    panel). Against the owner's choice (`B5phospho_fitsdf_byres`, WT, 156 residues), from the
    stored fits (Poisson loss, fixed-K bootstraps): two populations at 126, 117, 98 and 100
    residues (4 s, 60 s, 1,800 s, 72,000 s) against 90, 104, 94 and 93, the same choice at 120,
    123, 100 and 105, populations correlated 0.5 to 0.6 (the K=2 split differs with the loss,
    item 49). The WT Gaussian refit had no bootstrap: its errors came from the scatter of
    replicates only and two populations passed at 150 to 155 of 156 residues, so the choice
    needs the bootstrap (`species_uptake` now warns without one).

51. **Comparisons within a run (phase 3; 2026-10-13).** `compare_peptides` (hybrid rule:
    Welch's test on the replicate spectra and a global `delta_crit`), `compare_residues`,
    `region_summary` and `Annotations` (user-defined regions and sites); plots: reference
    against each sample residue by residue (the owner's `vsWT_byresidue`), the Woods plot, and
    region bars (`RFUbarplots`). `species_uptake` was split into the per-spectrum records (which
    the Welch test needs) and their combination, with identical output on the benchmark. On the
    intact-peptide benchmark against WT (K=1): `delta_crit` 0.112 fD (4 replicate spectra
    typically); significant peptides at 4 s, 60 s, 1,800 s and 72,000 s: S19D 10, 13, 13, 11 of
    about 175; S45D 8, 9, 16, 18; S59D 11, 11, 12, 14; D3 20, 19, 31, 52 of about 142, mostly
    higher uptake, in the N-terminal region; at 4 s the S59D residue differences cluster at
    residues 59 to 65. Region values at 4 s match the owner's `RFUbarplots` (FR-A about 0.35,
    the beta regions 0.2 to 0.3, L9,CTR about 0.8; regions from the owner's notebook, used only
    for the private examples).
    Population-resolved, D3 against WT: at K=2, only 39 to 49 peptides have a description that
    passes the filters in both samples at a time, and the replicate scatter of each rank's fD is
    larger (`delta_crit` 0.24 and 0.23 against 0.11 at K=1), so 0 to 7 are significant. For 17
    N-terminal peptides (ending by residue 60) with K=2 in both, D3's average fD is higher by a
    median 0.14 at 4 s while the low-D state's fD differs by only 0.03 and its population is
    lower by 0.17 (in 13 of 17): fewer molecules in the protected state rather than more
    deuterium in either; by 20 h no difference. Compared with the hybrid rule, though, the
    populations are too uncertain for single peptides: their pooled replicate scatter gives a
    `delta_crit` of 0.48 (no peptide significant; median shift -0.09 at 4 s), and per residue 1
    of about 120 is significant at 4 s (median -0.07). A tendency, not a per-peptide result,
    with these (Poisson-loss) fits.

52. **`hxex analyse` and the `[analysis]` settings (2026-10-13).** `hxex.analysis.AnalysisConfig`
    (reference sample and order, times, K values, FASTA per sample, coverage times, redundancy
    count, filters, choice limits, alpha, regions and sites, figure format, output directory),
    read from the project file's `[analysis]` section (`[[analysis.regions]]`,
    `[[analysis.sites]]`, `filters = false`), and `hxex.report.write_analysis`, which writes the
    tables (species uptake, residues, coverage, populations, the peptide, residue and population
    comparisons, regions) and the figures (coverage grids, redundancy and resolution, choice
    profiles per sample and time, residue and population comparisons, stacked Woods plots per
    sample, region bars). `hxex analyse PROJECT` (or a bundle with the default settings;
    `--bundle`, `--out`, `--reference`, `--format`, `--threads`). On the intact-peptide
    benchmark (5 samples, 5 time points): 8 tables and 66 figures in 38 s on the CPU.

53. **Comparing separate runs (phase 4; 2026-10-13).** `hxex.analysis.RunSet` holds several runs'
    results by name and relabels their samples `run:sample`; `species_uptake` and
    `compare_peptides` (and so the residue map, the choice, the comparisons, regions and
    `write_analysis`) take it like one run. Also `compare_settings` (where the recorded run
    settings differ, and which differences matter), `check_identity` (species whose position or
    amide count differs between runs), `compare_choices` (residues whose one-or-two choice
    differs; also written within a run), `plot_butterfly`, `[analysis.runs]` and
    `hxex analyse NAME=BUNDLE ...`. Figure file names replace characters a file system refuses
    (the `:`) with `_`. Check on the intact-peptide benchmark loaded twice as runs A and B:
    `A:WT` against `B:<variant>` reproduces the within-run WT against each variant exactly (3,872
    peptide-time rows; deltas, p values, significance and `delta_crit` 0.1119 identical), with
    no identity or settings differences; 5.6 s against 2.5 s within the run. Only one run of the
    benchmark exists, so differences between genuinely separate runs are not yet measured.
    Undecided: the rule for one population in one run and two in the other (ANALYSIS.md
    "Later parts", recommended: K=1 always, K=2 where both keep it, disagreements reported).

54. **By-residue comparison plot from the owner's reference (2026-10-13).** The owner's
    current reference for comparing samples residue by residue is
    `B5_phospho_2pop_vsWT_byresidue_72000s_26Aug2026` (one panel per variant against WT in
    black, points with capped error bars, no lines, both K=2 values); the owner asked for any of
    low, high and average to be selectable. Built as `plot_residue_overlay` and
    `residue_overlay_figure` (`show`, and `[analysis] overlay`, default `["low", "high"]`; the
    K=1 average as open points); `hxex analyse` now draws the fD `compare_residues_<time>`
    figures with it (the earlier K=1 comparison with significance bands stays for the
    population comparison). The owner also said that many plots so far followed older versions
    of their figures, so every new plot type now starts from a reference the owner supplies, and
    confirmed the choice figure (one sample per figure) as it is. On the benchmark the style
    matches the reference; the K=2 values differ because the fits differ (item 49). The dashed lines of
    regions and sites in the comparison plots are grey (owner), not purple.

55. **Why the 4 s by-residue comparison differed from the owner's figure (2026-10-13;
    intact-peptide benchmark, owner's `2pop_vsWT_byresidue_4s` against hxex's).** In order of
    effect: (1) **The noise model of the stored fits.** The run estimated the noise from
    replicates, so most spectra were fitted with the Poisson loss (item 49), which often puts the
    second component in the high-mass tail (fD 1.4 to 2.4, population 2 to 10 %); the filters then
    drop that peptide's K=2 description. Around residue 40 this removed most of D3's peptides
    (12 of about 20 between residues 33 and 50), and the residue values came from the few left,
    which are lower: over residues 35 to 50 the median D3 - WT difference was +0.07 (low) and
    -0.01 (high) against the owner's +0.10 and +0.11. Refitted with the Gaussian loss (picked
    spectra at 0 s, 4 s and the fully deuterated control, every sample, K = 1 and 2, 20-draw
    bootstrap; 7 min on the GPU; `hxex_private/HSPB5_benchmark/gauss_4s`), the peptide fits match
    pyHXExpress's (for example 33-47 D3: 0.46 and 0.94 against 0.42 and 0.90), the peptides are
    kept, and the difference is +0.08 and +0.06, the increase in both populations visible again.
    (2) **Which residues are drawn.** The owner's figure shows only residues `choose_fits` calls
    bimodal (both errors and the lever-rule population's error below 0.25, population 0.10 to
    0.90); hxex's overlay shows every residue with K=2 values (about 148 per sample against
    about 100), most visibly residues 118 to 175. (3) **Error sizes**: the owner's K=2 high
    errors are about twice hxex's (median 0.08 to 0.11 against 0.04 to 0.05): pyHDX gets the
    per-spectrum `Dabs_std` with replicates and charge states as separate rows, while hxex gives
    the standard error of their mean (about 4 spectra per peptide). Larger errors fail the
    bimodal test more often. (4) Smaller: some short peptides (33-38, 34-38, 34-48 for D3) still
    split differently after the refit; pyHXExpress reports the bootstrap mean within `Pop_Thresh`
    rather than the main fit (item 49); the notebook drops spectra with fD below 0.1 after time
    0 and those on the owner's manual quality list. Back-exchange correction is not a source:
    the amide counts are equal, and where both fits agree the values agree.

56. **Decided: the Gaussian loss (owner, 2026-10-13).** After item 55, the owner chose the
    Gaussian loss (least squares on the scaled spectrum, as pyHXExpress) definitively. It is
    already the default (`NoiseConfig(model='gaussian', percent=3.0)`, no estimate); the
    benchmark had used `estimate='replicates'`, which fits every series with replicates with the
    Poisson loss. The benchmark is refitted with the Gaussian loss
    (`hxex_private/HSPB5_benchmark/gauss_full`) and the examples regenerated from it.

57. **The replicate noise estimate keeps the Gaussian loss; overlay filter (owner, 2026-10-13).**
    `NoiseConfig(estimate='replicates')` used to fit every series it could estimate with the
    Poisson loss (its gain as the scale), which is what put the benchmark on the Poisson loss.
    Owner's choice (of: keep it Gaussian, warn, or remove it): it now only sets the Gaussian
    level, `sd = sqrt(gain x the spectrum's mean bin intensity)` (variance = gain x intensity,
    so the expected sum of squared residuals is the estimated one), and it refuses the Poisson
    model. Under a constant sd per spectrum the fitted values do not depend on the sd; it sets
    the BIC choice of K and the size of the bootstrap noise. Outlier flagging is unchanged.
    The by-residue overlay can draw the K=2 values only where the choice keeps two populations
    (`choice=` on `plot_residue_overlay` and `residue_overlay_figure`; `[analysis]
    overlay_two_only`, default false), as in the owner's figure; the average is never filtered.
    `write_analysis` now draws a population comparison panel only for samples with values at
    that time (it raised before when one had none).
    Under the Gaussian loss the `poor_fit` flag caught undeuterated controls: at one noise level
    per series a spectrum's deviance grows with its peak height, and the controls are the most
    peaked spectra, so a small relative misfit gave them a large deviance against the series
    median (306 of the 320 flagged at 0 s in the 4 s refit were K=1 fits). Decided (owner): judge
    the fit relative to the spectrum's intensity. `poor_fit` now compares the misfit
    `sum((obs - fit)^2) / sum(obs^2)` (raw intensities, independent of the noise model) with the
    series median (factor 10, as before), and keeps the absolute `min_deviance` guard; the
    deviance and its ratio stay as statistics, and `QualityFlags` and the bundle gain `misfit`
    and `misfit_rel` (NaN when read from older bundles; the quality overview plots `misfit_rel`).
    On the benchmark, Gaussian refit (24 min on the GPU): flagged at 0 s 1.4 % (deviance rule
    13.0 %), at the other times 0.4 to 0.8 % (0.7 to 1.6 %); 183 in all against 711, 152 of them
    flagged by both. On the stored Poisson fits the two rules flag about as many (303 and 264).
    The test fixtures drew their undeuterated controls as a Gaussian near bin 0, which no
    binomial produces (misfit 3 % against 0.1 to 0.3 %); they now use the peptide's natural
    isotope envelope.

58. **The overlay filter is on by default; the one-vs-two rule by example (owner, 2026-10-13).**
    `overlay_two_only` defaults to true (the user can turn it off). The filter applies per
    sample: each sample's K=2 points are drawn only where its own choice keeps two
    populations. Example (Gaussian refit, residue 63 at 60 s): WT keeps two populations (low 0.44,
    high 0.80, average 0.68); D3's average 0.47 lies outside its K=2 values (0.30, 0.46), so it
    keeps one; the figure shows WT's two points and D3's average, the K=1 values are compared
    (D3 lower by 0.22 +- 0.02) and `compare_choices` records the disagreement. The peptides over
    residue 63 contain the S59D site, so their sequences differ between WT and D3 and they are
    not compared as peptides; only the residue level compares that region. The region and site lines of
    every residue plot (profiles, choice, redundancy and resolution, comparisons) are now the
    same grey dashed lines (`hxex.report.coverage.MARK_LINE`; owner).

59. **Back-exchange controls on the benchmark; the buffer's D2O fraction (2026-10-13).** Around
    residue 60 at 4 s the K=2 values of S59D and D3 were about half the owner's (high 0.45 to
    0.6 against 0.8 to 0.88). The peptide fits agree where both use the same correction; the
    difference is the control: several series over the S59 site (55-62, 55-64, 55-65 in S59D
    and D3, 49-65 in D3) have a measured retention of 0.475 to 0.50, below `retention_min`
    (0.5, item 27), so hxex fell back to full retention, i.e. no correction, while pyHXExpress
    uses the measured value (`UN_TD_corr`, no lower limit) and, when the control is missing, its
    `Dfrac` (0.9, the buffer's D2O fraction). Over the benchmark 250 of 1,372 series (15 % of
    the spectra) fell back: 186 without a fully deuterated control, 53 below the limit (median
    measured retention 0.47; 31 in 0.45 to 0.5), 1 above, 10 other; accepted retention has a
    median of 0.64 (5 to 95 %: 0.54 to 0.73). Added (owner): `ControlConfig.d2o_fraction`
    (default 0.9, pyHXExpress's `Dfrac`), the deuterium fraction of the labelling buffer; a
    missing or rejected fully deuterated control now falls back to `m0 + d2o_fraction *
    n_amides` instead of `n_amides`. Median |hxex - owner| of the K=1 residue values (Gaussian
    refit): S59D and D3, residues 55 to 68 at 4 s: 0.203 with the old fallback, 0.185 with
    `d2o_fraction` 0.9, 0.010 with 0.9 and `retention_min` 0.3; every residue, sample and time:
    0.023, 0.020 and 0.015. Retention measures the D2O fraction times what survives
    back-exchange, so here it centres at 0.64, below the buffer's fraction.

60. **Peptides compared by residue range by default (owner, 2026-10-13).** Peptide comparisons
    matched identical sequences only, so a peptide containing a mutation (S19D, S45D, S59D, D3)
    was never compared with the reference's; only the residue level covered those stretches.
    `compare_peptides(match='range')`, now the default (`[analysis] peptide_match`), pairs
    peptides over the same residues and time and records `peptide_reference`,
    `modification_reference` and `same_sequence`; where a sample has more than one species over
    a range (a modified and an unmodified form), only identical sequences are paired.
    `match='sequence'` keeps identity matching. On the benchmark (Gaussian refit, against WT)
    this adds 76, 116, 54 and 229 peptide-time comparisons for S19D, S45D, S59D and D3 (13, 20, 9
    and 39 mutated peptides; 5, 28, 17 and 61 of them significant). PTMs, as built: a modified
    residue is written in the modX sequence, so a modified peptide is its own species, compared
    only with the same modified form; at the residue level both forms of a stretch are averaged
    into the sample's values (open: whether to keep them apart).

61. **Control fallbacks: limit, median option, marks; modified forms as states (owner,
    2026-10-13).** `ControlConfig.retention_min` defaults to half of `d2o_fraction` (0.45 at
    0.9; `min_retention` is the limit in use), and `fd_fallback='median'` falls back to the
    median retention of the run's accepted controls instead of `d2o_fraction` (flagged
    `fd_median`). Because samples can be corrected differently (one series measured, the other
    a fallback), the analysis now carries `control_fallback` (species: any of its spectra; residue
    map and population choice: any covering peptide; comparisons: `control_fallback_reference`,
    `control_fallback` and `corrected_differently`), and the comparison plots (by-residue
    overlay, residue and population comparisons, Woods, butterfly) mark such residues or
    peptides with a `*` at the top. Modified peptides at the residue level: `residue_map(
    modified='separate')`, the default, makes each modified form a state of its own labelled by
    its sites (`'WT pS59'`; `modified_state`), so a sample's values come from its unmodified
    peptides; `'combined'` keeps them together (`[analysis] modified_peptides`). The benchmark
    examples use `HSPB5_benchmark/gauss_full_v2`, the Gaussian refit with controls and quality
    recomputed under these defaults: 219 fallbacks instead of 250, and around residue 60 at 4 s
    the K=2 values of S59D and D3 close to the owner's.

62. **The remaining K=2 low-population gap is a filter difference; hxex's filtering kept (owner,
    2026-10-13).** After items 59 to 61, hxex's K=2 low population around residue 60 at 4 s was
    still about 0.14 below the owner's (WT, S59D, D3). On the same fits (`gauss_full_v2`), only
    the filtering of K=2 descriptions was varied, against the owner's residue values (median
    hxex - owner, low; residues 55-68 at 4 s / all residues, samples and times): hxex's filters
    (a spectrum's whole K=2 description is dropped when a component fails) -0.142 / -0.038; no
    filters -0.070 / -0.028; the owner's notebook (each population row dropped on its own when
    its fD is above 1.1 or below -0.1, so the low row of a spectrum whose high component fails is
    kept) -0.082 / -0.028; the notebook with its further rule dropping every row with fD below
    0.1 after time 0 -0.046 / -0.004 (high population about -0.05 / -0.01 throughout). So the
    gap comes from the filters: per-row filtering accounts for about 0.06 and the fD < 0.1 rule,
    which removes the lowest low-population values, for about 0.04; the remaining -0.05 around
    residue 60 matches the peptide-level fit differences of a few short peptides (55-62, 58-65).
    Owner: hxex's filtering stays as it is, being justified (a description with an unphysical
    component is not trusted for either component, and genuinely protected states with fD below
    0.1 are not removed).

63. **Recomputing and combining (post-fitting analysis, phase 5; 2026-10-13).** pyHXExpress's
    `recalc_uptake` recomputes the correction after bad spectra are removed; in hxex the uptake
    already left out the spectra the review rejected, but the controls were those of the run.
    `analysis_controls(results, changes)` now recomputes them from the run's K=1 fits when the
    review rejected a control spectrum (rejected spectra and noise outliers left out) or when
    control settings are changed (`species_uptake(controls=...)`, `compare_peptides(controls=
    ...)`, `[analysis.controls]`); otherwise the stored controls are used. Recomputing the
    Gaussian refit's controls under the current defaults reproduces `gauss_full_v2` exactly (219
    fallbacks, identical mFD). Combining runs (`combine_batch_data`): `RunSet(merge={'A:WT':
    'WT', 'B:WT': 'WT'})` and `[analysis.merge]` pool samples of separate runs (the same run
    merged with itself gives the same values with twice the spectra). Run identity is kept
    (owner): every spectrum's record carries its run, the species table gains `runs` ('A,B'),
    `n_spectra_by_run` ('4,4') and `run_offset` (a run differing from the first by more than 2
    combined standard errors in the same direction at more than half of the shared times, and at
    two at least; a warning counts them). Without the flag a batch effect would only widen the
    standard errors (the replicate scatter) and enter the Welch test as replicate noise. The
    export to pyHDX's DynamX-like format (`hdexa_to_pyhdx`) is not needed (owner).

64. **Fragments tied to their parent; MS3 stages (2026-10-14).** In a private bottom-up dataset
    (an LC-MS file per labelling time, targeted peptides fragmented in MS2 and MS3), each
    precursor is acquired as plain ETD or EThcD MS2, as MS2 after in-source activation (`sid=`),
    and as MS3 that heats the precursor with HCD (energies 5 to 35), isolates the same m/z again
    and fragments it with ETD or EThcD (`ms3 313.0100@hcd15.00 313.0000@etd25.00`). The parent of
    such an MS3 scan is the peptide itself. Two gaps were found. (1) Fragment targets had no
    precursor m/z by default, so in a file with several precursors a peptide's fragments were
    read from the scans of every precursor of the same charge (single-peptide infusion hid it).
    (2) The readers kept the last stage only, so an MS3 scan's pre-activation and parent were
    lost. Now every scan keeps its stages (`Scan.stages`; the scan table gains `parent_mz`,
    `parent_lower`, `parent_upper`, `preactivation`, `preactivation_energy` and `energy`), and
    `MzmlTarget` gains `parent_mz` (the parent ion's m/z range, undeuterated to fully deuterated,
    which the parent's isolation window must overlap, or which its m/z must be within the
    tolerance of when no window is stated) and `preactivation` (`'HCD'`, `'none'`).
    `fragment_targets` sets `parent_mz` from the parent by default (`'parent'`) and takes
    `ms_level`; so does `[input.fragments]`. An isolation window need not contain the parent's
    monoisotopic m/z (in the fragment kinetics files it is centred up to 3 m/z above it), hence
    the overlap with the whole envelope rather than a tolerance around one m/z. MS3 series are
    labelled `-ms3-pre:HCD` and the results tables gain `preactivation`, so they stay separate
    conditions. ThermoRawFileParser's mzML writes only the last stage of an MS3 scan; the earlier
    stages are then read from the filter string (without their isolation windows, which the
    Thermo reader has). On one file of the dataset the two readers agree on every new column
    except those windows (5,570 scans). The fragment kinetics files select the same scans as
    before. Not built: an MS3 fragment of an MS2 fragment (a fragment entry as a parent).

65. **Fragment amide counts by position in the parent (2026-10-14).** Until now every ion type
    used the peptide rule on the fragment's own sequence (pyHXExpress's), and `nterm_subtract`
    removed a C-terminal fragment's first residues as if they were the parent's N-terminus.
    By structure a fragment holds the backbone N-H of its residues, and at the cleavage site c
    ions also hold the next residue's (N-Calpha cut: c = b + NH3), x and y their own first
    residue's, and a, b and z not (z = y - NH3). The pyteomics compositions agree: with no
    side-chain nitrogen, c ions have L + 1 nitrogens, a, b, x and y L, z ions L - 1. So c, x
    and y counted one amide too few, and with the owner's `Nterm_subtract = 0` z ions one too
    many. Now amides are placed in the parent (`chem.amide_positions`): prolines and the
    parent's first `nterm_subtract` residues are left out (`nterm_subtract` stays a parent
    setting, owner); C-terminal fragment entries carry `parent_pos` in their modification so
    every count (picking, shapes, fits) agrees; `AmideRules` sets the cleavage-site amide per
    ion type (in place of the integer offsets). `fragments.fragment_amides` lists the residues
    a fragment holds and `amide_difference` those a difference measures (c_6 - c_5: residue 7;
    z_6 - z_3: residues n - 4 to n - 2; the parent minus c_k: the complementary z ion), refusing
    differences that are not a set of residues. Fragment lists stop at n - 1 residues and leave
    out c ions before a proline and z ions starting with one. Tested for every ion type and
    length against the compositions, on four parents with prolines and with `nterm_subtract` 0
    to 2. Evidence in the private fragment dataset (the 8-residue scrambling probe, low-energy
    ETD, first 15 s of back-exchange kinetics): complementary c and z ions sum to the precursor
    (2.95 to 3.37 D against 3.33 D), so the ladders fix the differences but not which residue
    each belongs to; with the structural assignment the c and z ladders give the same
    per-residue values (after 15 s within about 0.15 D) and the fast and slow residues fall
    where the probe was designed, the first slow residue one after the His/Asp block, as in the
    12-residue probe. The MAX runs there are back-exchanged to about 0 D (the experiment
    deuterates, then dilutes into H2O), and some early time points are spurious (owner).
    Unverified: neutral-loss variants (follow their base ion), the hydrogen moved in c-1/z+1,
    oxazolone b ions (owner: keep L - 1).

66. **Neutral losses, the c-1 hydrogen and b ions (owner, 2026-10-14).** A count changes the
    fractional deuteration (one amide in five or six is 17 to 20 %) and which residue a
    difference names, not the measured deuterium; the fit cap and the shape barely. (1) A lost
    NH3 takes a counted amide only when it is the cleavage-site nitrogen: from a side chain
    (K, R, N, Q) or the N-terminal amine it takes back-exchanged hydrogens. Neutral-loss
    variants follow their base ion, except `c-NH3`, whose NH3 is most plausibly the c ion's
    terminal NH2 (it is then a b ion, and its composition says so); M-NH3 (ETD's charge-reduced
    precursor) would hold one amide fewer only with `nterm_subtract = 0` and an NH3 from the
    N-terminal amine. Neutral-loss ions go into residue differences only after checking. (2) The
    c radical (`c-1`) has given a hydrogen atom to its z partner (z+1). Were it the cleavage-site
    amide's, a c-1 ion would lose that residue's deuterium. In the private fragment dataset (the
    8-residue probe, low-energy ETD, first 15 s) c-1 plus its z-dot partner equals the precursor
    (3.29, 3.33 and 3.17 D against 3.33 D where the cleavage-site residue holds 0.56, 0.95 and
    0.98 D; the sums would be about 2.77, 2.38 and 2.35), so the hydrogen moved is not a
    deuterated amide's, for the ion measured as c-1 (it may mix c-dot and c'). z+1 is untested
    (the fits use z-dot). (3) a = b - CO: both hold the nitrogens of residues 1 to L, which
    supports L - 1, but the linear (acylium) and oxazolone b ions have the same formula. In the
    oxazolone (the usual structure of collisional b ions) residue L's nitrogen is in the ring
    with a double bond to the previous carbonyl carbon and carries the mobile proton (from the
    solvent) instead of its own hydrogen: L - 2. Kept: L - 1 (`AmideRules({"b": False})` for
    L - 2); collisional b and y ions are scrambled and are not for residue values, and UVPD a
    ions (a direct Calpha-C cut) keep residue L's N-H.

67. **Fragment ladder kinetics, the simple way (2026-10-14).** `hxex.analysis.fragments`:
    `fragment_uptake` (K = 1 deuteration of every fragment-ion spectrum of an mzML or Thermo
    run, with its condition, parent, fragment label, charges and amide count), `ladder_steps`
    (one row per step of an ion series in residue order, with the residues it measures, or why
    a residue has none: block, proline, N-terminal, no ion), `difference_trace` (two fragments
    matched scan by scan within a condition, errors in quadrature), `fit_decay`
    (`A exp(-k t) + C` from several starting rates, `resolved` False when `k t_last < 0.1` or
    `k t_first > 3`: a decay too slow for the range is fitted as a straight line with a tiny
    rate and huge amplitude and offset), `fit_residues` (bounded least squares over all
    fragments, with which residues the fragments isolate). `hxex.report.ladder_kinetics_figure`
    follows the owner's reference layout: precursor ions on top, one row per step (fragments
    and difference; the difference with its fit). Projects take several `[[input.fragments]]`
    tables, each with a `label` naming its condition (gentle and harsh scans of one file
    against one undeuterated control). An ion without exchangeable amides (z1) is fitted with
    the cap of one site. On the 8-residue probe of the private fragment dataset (ETD 3+, 12,910
    spectra, 32 s on the GPU), the c-1 ladder resolves residues 4 to 8 in both conditions (gentle:
    D4 2.8e-3, I5 6.0e-3, V6 1.9e-3, I7 6.3e-4, K8 2.1e-3 per s); the z-dot ladder agrees for
    D4, I5 and K8-ish but gives V6 and I7 unresolved in one condition each; least squares at
    100 s (gentle): D4 0.03, I5 0.42, V6 0.83, I7 0.89, K8 0.58 D per amide. The owner's
    reference figure assigns c_(k-1) -> c_k to residue k; its rates equal these, one residue
    lower (c3 -> c4 6.2e-3 against I5 6.0e-3, and so on), so the difference is the residue
    assignment (item 65); the owner confirmed that the reference figure assigned the residues
    wrongly and hxex's assignment is right. Later the same day: `closure_trace` and
    `closure_table` make item 41's closure test a function (c and z, a and x, b and y, scan by
    scan against the unreacted precursor by default, flagged beyond 0.25 D), and the ladder
    figure takes `right_traces` (one's own traces in the right column, for the owner's private
    analyses). On the probe all 12 c-1 + z-dot pairs close within -0.08 to +0.03 D (spread
    0.02 to 0.05 D) in both channels; against the charge-reduced 1+ precursor -0.03 to -0.15 D.
    The closure holds at any level of scrambling, so it checks the fragments' fits, not
    scrambling; hxex has no scrambling test (it belongs to the owner's method, tabled).

68. **Fragment outputs in `hxex analyse` (2026-10-14).** For a run with fragment series the
    analysis writes `fragment_uptake.csv`, `ladder_rates.csv` (`ladder_rates`: the decay fit
    of every single-residue step per condition, ion type and fragment charge) and `closure.csv`,
    and per parent (`<sample>_<range>_z<charge>`) the ladder kinetics figure per ion type and
    fragment charge, a closure figure per complementary pair (c + z - M against time, a panel
    per condition, the tolerance band) and a per-residue rate figure (rates on a log axis with
    their errors, ion types by marker and conditions by colour, unresolved fits open along the
    bottom; fitted amplitudes below). The owner had no reference for the last two and asked for
    them to be designed. `[analysis.fragments]` (`FragmentAnalysis`: ion types, fragment charges,
    `min_time`, condition labels, closure tolerance and precursor charge) tunes them and
    `fragments = false` turns them off. Series read from fragment lists, precursor ions
    included, are left out of the peptide analysis, which a fragment-only run skips (it would
    otherwise draw residue maps of precursor ions at every scan time). Several runs (a
    `RunSet`) get no fragment outputs yet (a warning). On the 8-residue probe: 3 tables and 4
    figures in 10 s.

69. **Two clocks and a per-file usable start (owner, 2026-10-14).** Times in a kinetics run
    are on two clocks: the acquisition clock (scan times from the start of acquisition) and the
    exchange clock (spectrum times: scan time plus the run's `time_offset`, when exchange began
    relative to acquisition). `MzmlRun.usable_from` (`[[input.runs]] usable_from`; seconds on
    the acquisition clock, set by the user for each file, never suggested by hxex) leaves out
    the file's scans before it, for example spurious points while an infusion settles; it does
    not move the time axis (`time_offset` does). `[input.scans] min_time` and `windows` are on
    the acquisition clock too (for every run); the analysis `min_time` (`fit_decay`,
    `ladder_rates`, `closure_table`, the figures, `[analysis.fragments]`) is on the exchange
    clock and only keeps points out of the fits. Recorded with the read only when set. On the
    probe, `usable_from = 20` in place of the analysis cut gives the same rates (12,678 instead
    of 12,910 spectra fitted).

70. **Intrinsic exchange rates in H2O/D2O mixtures: `hxex.kint` (owner, 2026-10-14).** For
    theoretical back-exchange references and protection factors. Source: Grimaldi, Stofella and
    Paci, J. Phys. Chem. B 130, 2493 (2026), code pacilab/hdx-rates-mixtures (MIT; the notice
    is kept in the module). It was ported rather than used as a dependency (owner): it is not
    a package (no `pyproject.toml`, not on PyPI, flat-module imports), uses pandas, takes
    one-letter sequences only and has no terminal groups beyond the free ones; the logic is
    about 150 lines and two tables. Unlike `hdxrate` (pyHDX; pD = pH_read + 0.4 x), it treats
    the mixture: the ionic product (Sweeton 1974, Gold and Lowe 1967), the D fractions of L+ and
    OL- (fractionation factors 0.69 and 0.47) and the four isotope pairs, giving a forward
    (H to D) and a back (D to H) rate per amide. In a mixture an unprotected amide relaxes at
    `kforw + kback` to the D fraction `kforw / (kforw + kback)` (about 1.13 x at x = 0.1), so
    that sum is the rate compared with observed decays. Inputs: `ExchangeConditions(ph_read,
    temperature, d2o_fraction, reference)`, temperature in degrees Celsius (a value in kelvin is
    refused), reference `PDLA` (default) or `3Ala`; nothing in the fragment dataset's metadata
    records them, so the user sets them.
    - **Check against the reference code** (`tests/golden/make_kint_golden.py`, 192 cases: four
      sequences, pH_read 2.5 to 8.5, 0 and 25 degrees, x = 0 to 1, both references): identical
      to about 1e-12 once the reference's intermediate rounding (rates to 1e-6 /s, factors to
      0.01) is stripped; with its rounding, within about 2 % (median 0.8 %).
    - **Differences from the reference code:** modX sequences with terminal groups; `Ac-` gives
      residue 1 an amide with the acetyl's factors (Bai's `Ac` row, unused there) and takes
      residue 2's N-terminal factor away; `-NH2` has no C-terminal factor (Bai's is the
      carboxyl's) and marks the last residue `approximate`; a two-residue peptide gets both
      terminal factors (the reference applies only the N-terminal one); no rounding; NaN, not
      0, where there is no amide hydrogen (residue 1 with a free amine, prolines).
    - **PTMs.** Bai et al. give no factors for pS, pT, pY or acK, and none have been published.
      Moroco et al., MCP 24, 100904 (2025; 100 pS/pT and 100 acK peptide pairs, pH_read 3,
      20 degrees, 88 % D2O): once normalised to fully deuterated controls, S/T phosphorylation
      had no measurable effect on uptake, observed rate or protection factor; lysine
      acetylation slowed exchange significantly (up to 2 D), with no correction factors given.
      pY and other PTMs were not tested. A phosphomimetic S to D is covered by Bai's Asp
      factors; phosphohistidine's slowed exchange is the imidazole C-H, not an amide. So
      (owner): a modified residue takes its standard residue's factors and is marked
      `approximate` (its amide and the next, which its side chain also affects), unless
      `ExchangeConditions.factors` gives its own `(AL, AR, BL, BR)`.
    - **Uses (opt-in, owner).** Nothing is computed unless the user opts in
      (`[analysis.fragments] kint = true`) and gives the samples' conditions
      (`[analysis.fragments.exchange]`, one table for every sample or one per sample,
      `[analysis.fragments.exchange.<sample>]`); conditions alone change nothing, and `kint`
      without them is refused. `ladder_rates(..., conditions=)` then adds `kint` (the residue's
      `kforw + kback`, from the parent's sequence, in which the kinetics happen; NaN for a
      sample without conditions, with a warning) and `protection` (`kint / rate`); without
      conditions those columns do not exist. `expected_uptake(rates, residues, times, initial)`
      is the theoretical reference: the deuterium of unprotected amides relaxing from
      `initial` (`ExchangeConditions.initial_fraction`, for back-exchange the labelling
      solvent's D fraction). Figures (designed by hxex at the owner's request): the ladder
      figure's right column adds the theoretical curve, dashed black, of the step's residue
      (and of all the parent's amides in the precursor row; amides without a rate, such as a
      free N-terminal amine, are left out) when `initial_fraction` is set; the residue-rates
      figure adds each residue's intrinsic rate as a black bar and a third panel of protection
      factors. Peptide-level (MS1) protection factors and theoretical controls for the quench
      and LC are not built.
    - **Can the conditions be inferred from the data? (2026-10-14, the private fragment
      dataset's 8-residue peptide, ETD, gentle and harsh, resolved steps.)** The D fraction can:
      the decays' offsets (about 0.03 D per amide) and the final equilibrium run give the
      solvent's equilibrium, about 1.13 x for small x. pH and temperature cannot: both scale
      every rate together (the acid branch with pH and temperature, the base branch the opposite
      way with pH), so fitting them makes the median protection factor 1 by construction, and
      the fit is poor anyway (pH free, any temperature: best at the grid's edge, pH 1.0, rms
      0.35 to 0.7 log units). With the reported pH_read of the H2O quench buffer, 2.8 (owner),
      the median protection factor is 1 only near 33 degrees; at 20 degrees it is about 0.15
      (residues 5 to 8: 0.02 to 0.15, faster than unprotected, which an amide in solution cannot
      be) while D4 is about 12 (H3 unresolved). The pattern along the sequence is close to the
      reverse of Bai's at pH 2.8 (D4 and H3 predicted fastest, K8 slowest from the carboxyl
      factor; an amidated C-terminus would speed K8 up only 9 times). So the infusion conditions
      differ from an aqueous pH 2.8 solution (temperature, organic solvent, pH after mixing), or
      the fragment rates hold more than solution exchange (exchange or scrambling during ion
      transfer); open, with the owner.

71. **Theoretical back-exchange against measured fully deuterated controls (2026-10-14).**
    Published HSPB5 data (WT, D109H, R120G; SpecExport; owner: need not stay private), K = 1
    fits with controls; 127 series of 48 peptides with replicated controls and a retention
    time (HDExaminer's results table; SpecExport has none). Models of the fully deuterated
    control's deuterium `D = mfd - m0`, 5-fold cross-validation grouped by peptide, 10 random
    splits, quench assumed pH_read 2.5 and 0 degrees (1.6 % formic acid, frozen, thawed and
    digested at the run): hxex's fallback `0.9 n_amides` RMSE 5.75 D (measured retention is
    low here, median 0.49 per amide); one fitted retention `a n_amides` 1.04 D (`a n + c`
    1.05); each amide at its intrinsic rate (`hxex.kint`) for one fitted effective time,
    `a sum_i exp(-k_i t0)`, 0.52 D; with `t0 + RT` 0.49 D (`t0 + b RT` 0.49). The sequence
    halves the error; the retention time adds about 6 % (the sequence model's residual per
    amide correlates with RT, r = 0.35). Fitted `a` 0.87 to 0.91, not given to the fit,
    matches the labelling buffer's expected 0.85 to 0.9 D2O (owner): an independent check;
    `t0` about 1 h at pH 2.5 and 0 degrees, more than the LC, so the thaw, digestion and
    source dominate. pH 2.3 or 2.8 and 2 degrees change neither
    ranking nor errors (in the acid branch they rescale every rate, which `t0` absorbs).
    Is the shortfall all back-exchange (owner)? Out-of-fold residuals per amide by domain
    (by peptide midpoint; NTD 1-66, ACD 67-150, CTD 151-175) are small (s.d. 0.045) and mostly
    retention time (NTD peptides elute late, CTD early): with `t0 + b RT`, NTD +0.014, ACD
    -0.004, CTD -0.018; within one RT band the NTD is about 0, and the structured ACD holds
    slightly more than predicted, the opposite of incomplete labelling. So the shortfall is
    consistent with back-exchange. The remaining residual is peptide-specific and
    reproducible across variants (WT against D109H r = 0.97, against R120G 0.96) and not
    additive over overlapping peptides (80-86 -0.13, 87-96 +0.09, 80-96 -0.01): a
    peptide-level effect (termini, factor tables, source, measurement), not residue
    labelling (`residuals_by_region.py` in the example).
    Not built into hxex yet. Noticed on the way: with `[run.controls]` on, only its own
    `undeut_time` identifies the undeuterated controls; `[run] undeut_time` does not, so
    every series is `m0_missing` unless it is set (left as is; for the owner).

72. **Overwriting a bundle directory keeps files of your own (owner, 2026-10-14).** It used to
    remove the whole directory, so notes kept in it (a `summary.md`) went with every rerun.
    Now only the files the old manifest lists are removed; the new files are moved in and the
    manifest is written last. Readers trust only manifest-listed, checksummed files, so nothing
    stale can be read as part of the new results; a table the new run does not write is
    removed with the rest. A file of your own named like one of the new bundle's files is
    refused before anything changes. The replacement is no longer a single rename: an
    interruption leaves `manifest.json.replacing`, the directory does not open (a clear
    error) and the next write with `overwrite` completes it. New bundles and zips are written
    as before (one rename).

73. **Theoretical back-exchange as the fully deuterated fallback (owner, 2026-10-14).** From
    item 71: `ControlConfig(fd_fallback="kint", quench=ExchangeConditions(...))` (opt-in, like
    every use of `hxex.kint`; `[run.controls] fd_fallback = "kint"` with
    `[run.controls.quench]` ph_read, temperature, d2o_fraction). Every series of a parent
    ion (not fragments, not with extra `Hex` sites) gets the intrinsic rates of the amides it
    counts at the quench conditions (`series_rates`; a counted N-terminal amine keeps
    nothing). The run fits `a` and `t0` of `D = a sum exp(-k_i t0)` to its measured FD
    controls (robust soft-L1, then refitted without the rejected ones); a measured control is
    rejected when it misses the prediction by more than `kint_tolerance` (0.15 D per amide,
    about 3 times the scatter on the B5 data; `fd_off_model`), which replaces the retention
    floor (`retention_min` unset means none in this mode; an explicit one still applies). A
    missing or rejected control becomes `m0 + prediction` (`fd_kint`) with standard error
    `sd_per_amide * n_amides` instead of 0. Fewer than `kint_min_series` (10) usable controls:
    nothing is fitted, the `d2o_fraction` fallback stays (logged). The model is kept in the
    bundle (`controls.parquet` schema metadata) and recomputed at analysis time from the
    bundle's picked spectra (`analysis_controls`). Retention times are not used (SpecExport
    has none; they added about 6 %). On the B5 data (pH_read 2.5, 0 degrees, x 0): `a` 0.853,
    `t0` 3,393 s from 194 controls, none rejected, scatter 0.045 D per amide; the run's
    fallbacks went from 39 of 195 (38 good controls under the 0.45 floor, given
    `0.9 n_amides`, about twice what they hold) to 1 (no FD control: 9.71 +- 0.81 D, retention
    0.53, instead of 16.3 D). Default (owner): `fd_fallback` unset means `kint` when quench
    conditions are given and `d2o` without; an explicit value wins, and the unset value is
    stored as such, so conditions added later (`[analysis.controls]`) switch it too
    (`ControlConfig.fd_mode`). Are FD controls still needed? Yes, a few: with `a` fixed at the
    buffer's 0.875, a guessed `t0` of 600, 1,800, 3,400 or 7,200 s gives an RMSE of 0.26,
    0.14, 0.05 and 0.14 D per amide (the effective time is set by the whole workflow and
    cannot be predicted from the LC alone); fitted to 3, 5, 10 or 20 peptides' FD controls,
    the rest are predicted to 0.055, 0.051, 0.048 and 0.046 D per amide. A measured FD
    control is about 15 times more precise (median standard error 0.0026 D per amide), so
    the model replaces controls where they are missing or bad, not where they are good.

74. **Series spectra figures three times faster (2026-10-15).** Drawing every series' spectra
    (150 series, about 14 spectra each, PDF) took about 9 minutes. Profiled: 60 % was
    `tight_layout`, which `plot_spectrum` ran on the whole page after every panel (each call
    measures every panel already drawn, so the cost grew with the square of the panels), and
    15 % was reading the raw spectra from the bundle once per panel. Now `plot_spectrum` lays
    out only a figure of its own (the series page and the K comparison lay out theirs once),
    and `plot_series_spectra` reads a series' raw spectra in one read. Pages are pixel-identical;
    the 150 series and the merged PDF took 178 s. The rest is mostly writing the PDF.

75. **Residue comparisons carry each side's one-or-two choice (owner, 2026-10-15).** The open
    question (item 53) was whether `compare_residues` at K=2 should leave out residues where a
    sample's choice keeps one population. Owner: whether one sample supported K=2 and another
    K=1 is worth knowing, so no row is left out; instead each row records both choices.
    `compare_residues(..., choice=)` adds `two_reference` and `two` (False where a side has no
    K=1 value, as in `compare_choices`), and `ResidueComparison.choices_differ` marks the
    disagreements; `compare_populations` always adds them (the lever rule can apply where the
    choice keeps one population), and `hxex analyse` passes its choice, so `compare_residues.csv`
    has the columns whenever K=2 was analysed. Without a choice the columns are `None` and the
    values are unchanged.

76. **Waters `.raw` input: tabled (owner, 2026-10-15).** Looked at on two undeuterated
    Cyclic IMS MSe runs (functions: low-energy MS1, 1,900 scans with ion mobility; high-energy;
    lock mass at 785.84; m/z not lock-mass corrected in the file). Open-source readers:
    - OpenWRaw 4.0.0 (Apache-2.0, Rust with Python wheels, first release 2026-05): opens a run
      in 0.3 s and reads every function, but decodes no ion mobility and applies no lock mass.
      Against DynamX's exported spectrum of a 13-residue peptide (2+), the peaks were 33 ppm low
      and the isotope pattern was wrong even within +-0.04 min (1, 0.98, 0.92, 0.83 against
      DynamX's 1, 0.81, 0.28, 0.08, with a second overlapping 2+ species): summed over all drift
      times, where DynamX selects the peptide's drift window.
    - rainbow-api 1.5.3 (LGPL-3.0): refuses these files (only the 2, 4, 6 and 8 bytes-per-point
      formats; "HRMS ... not supported"); no ion mobility.
    Options left: the Waters MassLynx Raw SDK supplied by the user (as Thermo's RawFileReader;
    licence by request from Waters, ion mobility, Linux unconfirmed), OpenWRaw with hxex's own
    lock-mass correction (no drift selection), a decoder of our own, or msconvert to mzML with
    drift times. Recommended if resumed: the SDK.

77. **Kinetics runs take their noise reference from their own scans (owner, 2026-10-15).** The
    noise sd of a series is a percentage of its reference spectrum's tallest peak, the
    undeuterated control. In a kinetics run (an mzML or Thermo run without a fixed `time`) the
    control is a separate file, on the private fragment dataset about 2 % of a kinetics scan's
    intensity, so the kinetics spectra looked almost noiseless: harmless for the K=1 values,
    but it inflates the case for K=2 and would make bootstrap resamples far too narrow. Of
    the three options (each spectrum its own reference; one reference per series from its
    kinetics scans; noise estimated from neighbouring scans) the owner chose the second:
    `noise.kinetics_references` gives every kinetics spectrum of a series and replicate the
    one kinetics spectrum whose tallest peak is their median (the lower one for an even count),
    so the series keeps a common scale at the kinetics file's intensity. Which spectra are
    kinetics is read from the read record (`io.kinetics_spectra`: the runs without a `time`,
    matched by the spectrum ids hxex makes); `Measurements.from_binned(kinetics=...)`, passed
    by `run`. Controls and every other format keep their references.
    On the probe (ScramPep ETD 3+, gentle and harsh, 12,616 kinetics spectra and 62 controls,
    K = 1 and 2): the kinetics scales rose by a median factor of 47 (10th to 90th percentile 29
    to 60), the controls' were unchanged, and K=2 was selected for 1,048 kinetics spectra
    against 8,849 before. Of the K=1 fits, 659 move by more than 0.05 D: M-NH3 301 of 1,686,
    z7 and z8 286 of 1,133, c-1 66 of 4,772, and M and z2 to z6 6 of 5,087. The loss is flat
    there: the rescaled loss is the same
    (median change 0, 99th percentile 6e-5 relative). The ladder rates agree within their
    standard errors (largest: harsh z K8 0.0088 to 0.0063 s^-1, 1.3 SE; gentle c D4 0.0028 to
    0.0034 s^-1, 0.9 SE). The noise estimated from neighbouring scans (the third option) stays
    a possible opt-in.

**CPU and GPU (2026-10-10, RTX A4500, torch 2.14.1 with CUDA 12.6; `tests/test_gpu_parity.py`,
skipped without CUDA).** On the 542 Bimodal measurements, CPU float64 against GPU float32:
the selected K is identical for **542 of 542** spectra (the prototype: 541 of 542). The loss at
each K agrees to within 1e-3 for 100, 98 and 91 % of the fits (K = 1, 2, 3; median relative
difference below 1e-6); the fitted means of spectra with the same K differ by a median of
0.0000 D, 90th percentile 0.002 D, 99th percentile 0.05 D and at most 0.70 D (more than 0.1 D in
0.9 % of the spectra, more than 0.5 D in 0.2 %, the prototype's worst was 1.6 D), and the
fractions by at most 0.07 (the prototype 0.42). GPU float64 reproduces the CPU almost exactly (same
K for all 542, median loss difference 1e-15, means within 0.02 D). A GPU float32 run twice is
bit-identical, and fitting a subset alone agrees with the full batch, as on the CPU. Fit
time for the 542: CPU 67.5 s, GPU 18.9 s (float32) and 22.0 s (float64) (the prototype's GPU
fit took 16 to 19 s); bootstrap with N = 20: CPU 284 s, GPU 184 s, all resamples succeeding.
The mixed noise models, the estimate and the polish (`torch.func` on CUDA) run on the GPU, and
a bundle written from a GPU run opens without CUDA. Float32 is the GPU default; float64 is
available and costs little on this card.

**Phase 2 exit criteria** (reproduces the frozen results, passes CPU/GPU parity, save/load round
trip exact) are met.

**Raw spectra without a fit (`hxex.io.save_spectra` and `load_spectra`, 2026-10-11).** The readers
return raw spectra and do no picking; a `SpectrumSet` can now be saved to and loaded from a single
Parquet file without a fit or a bundle (the same layout and types as the bundle's
`spectra.parquet`, so the two are interchangeable, documented in `docs/BUNDLE.md`), exactly,
atomically, with a content checksum in the file metadata and partial reads by spectrum id. They
need numpy and pyarrow only. The shared Arrow helpers moved to `hxex.io._arrow`. Still to do:
the HX-Express writer (of the peak-picked spectra, see "Later goals"), and writers for SpecExport
CSVs or mzML if wanted.

78. **Code audit: the binomial components are normalised; owner decisions (2026-10-15).** A full
    audit (six reviewers over the whole package; every finding verified against the code, the
    main ones reproduced). The fixes are in this commit, each with a test. The findings that
    changed behaviour or settled a question:
    - **The real-valued binomial was not normalised.** Item 2's "mass within 2 % of 1" held only
      where it was checked (`13 <= n <= 14`, `p = 0.9`). At non-integer `n` the bin above `n`
      carries `(1 - p)^(n - b)` with a negative exponent, so the mass grows without bound as `p`
      nears 1: measured 1.02 at `n = 13.5, p = 0.9`, 1.20 at `p = 0.97`, 1.70 at `p = 0.99`,
      4.8 at `p = 0.999`, and 1.7 at `n = 0.5, p = 0.9`. The populations `A_k / sum A` were then
      not shares (on the Bimodal baseline 29 of 542 spectra had a selected component with mass
      above 1.1, 18 above 1.3, and a population off by up to 0.18), and the back-exchange
      correction inherited it through the fully deuterated controls. `model.log_binom_norm` now
      gives `log` of each component's mass over its valid bins `0 .. ceil(n)` (a logsumexp on
      an integer grid, differentiable), and `log_binom_pmf` and `convolve` subtract it, so every
      component sums to exactly 1 for any real `n` and `p`; the amplitudes are areas. The log
      terms `log p` and `log (1 - p)` come from the logit (`logsigmoid`) wherever the logit is
      known (the model and the polish), so in float32 the pmf stays finite and normalised where
      `sigmoid(logit_p)` rounds to 1 (the `1e-10` clamp used to take over there and the bin
      above `n` carried a mass of about 1.5e4). The reported mean stays `n p`: for the
      normalised continuation the pmf's centre is `n p` within 0.05 D at `p <= 0.9` and within
      0.4 D at `p = 0.99` and half-integer `n` (integer `n`: exact). An alternative continuation
      that is exact in both mass and mean, the mixture `(1 - f) B(floor n, p) + f B(ceil n, p)`
      with `f = n - floor(n)`, is an open option for the owner (it has kinks in `n` at the
      integers but no cliffs). The prototype comparisons in `tests/test_model.py` add the mass
      back before comparing. The fit regressions are unchanged (same selections and losses
      within their tolerances).
    - **A letter-only formula in `ModificationRegistry.with_modification`** (`'O'` for
      oxidation) was read by pyteomics as a peptide sequence (pyrrolysine plus water, 255 Da
      heavier instead of 16). Now `Composition(formula=...)`. The TOML path (dicts) was fine.
    - **Terminal groups on fragment ions (owner: raise).** `composition()` applied the far
      terminus's group to a fragment that does not contain it (`PEPTIDE-NH2` as a b ion was
      0.984 Da low, `Ac-PEPTIDE` as a y ion 42 Da high); it is now a `ChemistryError`.
      `fragment_entries` already writes each fragment with the terminus it has.
    - **`false` in the project file.** The "false means unset" rule was applied to every
      top-level `[run]` scalar (`drop_no_signal = false` became None and crashed the run) and
      never inside `[run.*]` tables (a fixed noise `sd` could not be set, `fd_time = false`
      meant time 0). It is now type-aware: `false` is None only for settings whose type admits
      None, in every table.
    - **`[run.controls]` times (owner, closes item 71's trap):** when not set, `undeut_time` is
      the input's undeuterated time (the one the reader labels the controls with) and `fd_time`
      the input's fully deuterated time; `false` means no control of that kind.
    - **A fully deuterated control at or below the undeuterated level** (`mfd <= m0`) passed in
      kint mode (no retention floor) and made every corrected mean of the series NaN without a
      flag; one such NaN then made every residue of that sample and time NaN in the residue
      map (`0 * NaN`), and the Woods plot crashed on NaN limits. A non-positive span is now
      `retention_low` whatever the floor; non-finite values are masked out of the residue
      design; unknown (NaN) errors stay unknown instead of becoming 0.
    - **Hex sites count as amides (owner):** they stand for exchangeable amides on a
      modification, so `n_amides` of the controls is the site count (amides plus Hex), now
      passed as such instead of `n_max / 1.2`.
    - **Range matching across modified forms is intended (owner):** `compare_peptides` with
      `match='range'` pairs an unmodified reference with the sample's modified form when each
      side has one species over the range; documented there.
    - **Noise reference and bootstrap resampling** differ from pyHXExpress as the owner intends
      (now listed in PLAN's deliberate differences).
    - **Overwriting a bundle keeps its review (owner).** `hxex run --overwrite` on a reviewed
      bundle used to drop `review.parquet`; the review is carried over (the ids are stable).
    - Readers: an empty zlib-encoded scan no longer aborts an mzML read; time units are whole
      words (`ms` is milliseconds; unknown units raise); duplicate spectrum ids (same sample and
      file name in sibling directories, or windows selecting the same scans) raise instead of
      colliding in the bundle; Windows-recorded run paths are found by name on POSIX; overwrite
      never removes a file that is not a bundle; review writes are atomic.
    Open for the owner (from the audit): the kint-mode retention floor for series without
    rates; the `[input.scans]` blanket `false` rule; whether `ms` should be accepted; the
    SpecExport FASTA, which is sliced by characters (a plain-residue FASTA is the precondition;
    a modX FASTA would be mis-sliced).

79. **Kint mode: series without rates keep the retention floor (owner, 2026-10-15).** With
    `fd_fallback = 'kint'` the floor was none for every series, but the prediction only judges
    the series that have rates (parent-type ions without `Hex`), and only once `kint_min_series`
    usable controls exist; fragment ions and `Hex` series had no check at all (any level from
    0 to full exchange was accepted). They now keep the d2o-mode floor (half of `d2o_fraction`)
    and fall back to `d2o_fraction` below it. The owner's wider point, to be designed: the
    back-exchange correction may belong at the parent level, applied to the fragments in a way
    still to be decided (PLAN, "Risks and open decisions").
    **Sequence files and modifications (owner):** FASTA has no syntax for the modX labels, so
    modifications are given in the peptide list (`peptides.csv`, pyHXExpress's metadf) and
    `read_specexport_dir` slices the plain FASTA by characters, which is correct for it. A
    sequence file that carries modifications would need a format decision: a future
    consideration, not a bug.
    **Errors without a bootstrap (owner's question):** the aggregation already gives them.
    `species_uptake` combines the records of a species (replicates and charge states) with the
    larger of their own errors and the scatter between them (`_combine`), so without a
    bootstrap the peptide-level standard error is the scatter's, and is unknown only for a
    species with a single record. The residue level propagates those, so residues covered only
    by single-record species are the ones without an error. The scatter of overlapping
    peptides about a residue's value is not used as an error (an option, not built).

80. **Fragment outputs for several runs (owner, 2026-10-15).** The owner asked to compare
    activations and ladders across runs. `fragment_uptake` (and so `write_analysis` and
    `hxex analyse`) takes a `RunSet`: the fragments of every run read from scan files, in one
    table with a `run` column, samples labelled as in the peptide comparison (`run:sample`, or
    the `merge` label). In a merged sample each run's conditions become `run:condition`, so
    runs are drawn side by side and never pooled, unlike merged peptide samples: each run has
    its own exchange clock (item 69), and scans are matched within a run. Runs that count
    fragment amides differently (`nterm_subtract`, amide rules) are refused. The three tables
    start with the run. Runs of one activation usually differ in precursor charge (EThcD of the
    2+, ETD of the 3+), so the residue-rate figure now holds every precursor charge of a parent,
    a colour per condition and charge (`Z3` in the legend; file `residue_rates_<sample>_<range>`
    without a charge when there are several); the ladder and closure figures stay per precursor
    charge, since their precursor row is charge-specific. On the 8-residue peptide of the private
    fragment dataset (ETD 3+ and EThcD 2+ fitted as two projects from one kinetics file;
    gentle condition), the same ion type agrees between activations wherever both are resolved
    (z: I5 6.85e-3 and 6.80e-3 /s; c: I7 6.2e-4 and 5.9e-4, K8 2.07e-3 and 2.20e-3; D4, V6
    within about one standard error). The c- against z-ladder disagreement of ETD at I7 and K8
    cannot be checked on EThcD: its z step at I7 is unresolved and its K8 z error is wide. One
    EThcD harsh c-1 step (I5) fits a 40 D amplitude for one residue, an unphysical fit that
    nothing flags yet. Noticed: a project file with only `[analysis]` (to compare bundles with
    `merge` and labels) is refused for lacking `[input]`; the comparison used the Python API.
    Closure figures (owner): the splits had shades of one colour map and one marker and could
    not be told apart; each now has a colour of its own and a marker (10 colours, 7 markers).

81. **MS1 protection factors, peptide level (owner, 2026-10-15).** Design and decisions in
    docs/ANALYSIS.md ("MS1 protection factors"). Built: `hxex.analysis.protection`
    (`peptide_protection`, `compare_protection`, `peptide_rates`), `[analysis] protection =
    true` with `[analysis.labelling]`, the tables `protection.csv`, `protection_by_time.csv`,
    `compare_protection.csv` and the figures `protection_<sample>` and
    `protection_difference_<sample>`; `intrinsic_rates(free_termini=)` for stretches of a
    protein. Naming (owner): always `protection`, never `P` or `p` (the binomial parameter and
    the comparisons' p-value); not "heterogeneous" (it suggests conformational states), but
    `protection_drift` and `one_factor_fits`. Checked on synthetic uptake from known factors
    (recovered to 1e-4 in log10; unequal amides drift; bounds at the thresholds).
    On the published B5 disease data (WT, D109H, R120G; K = 1, measured FD controls) with
    ASSUMED labelling conditions (pH_read 7.0, 25 degrees, 0.9 D2O; the real ones were not
    at hand), 63 peptides per sample: the C-terminal tail is least protected and the ACD most,
    as expected, but no peptide fits one factor (median reduced chi-square 935) and the
    apparent factor rises with time by a median of 4.8 log10 units over 0.1 s to 20 h (WT
    medians 0.8, 1.5, 2.1, 3.1, 4.4, 5.8 at 0.1 s, 0.8 s, 4 s, 1 min, 30 min, 20 h). Either
    the amides of each peptide span a wide range of protection, or the millisecond points were
    labelled under other conditions. Owner: the millisecond points are unreliable (the
    experimental setup); without them the rise remains (WT medians 2.1 to 5.8 from 4 s to
    20 h, drift median 3.5, still no peptide fits one factor). Owner: B5 is a polydisperse
    oligomer whose exchange is likely multimodal; in many regions two populations describe the
    data better than one (not "a bimodal oligomer"), and later times are more sensitive to
    subunit dissociation, so a protection factor that changes with time is expected there: the
    K = 1 average mixes populations and slow events that no single EX2 factor describes. On
    such data `protection_drift` and `one_factor_fits` report the behaviour, not a fault;
    a K = 2 rank's apparent factor (a K = 2 bundle) is the closer look, and a residue-level
    EX2 fit of K = 1 averages would meet the same limits, and should also take a K = 2 rank
    (the per-population states pyHXExpress gave pyHDX). No example dataset suited to checking
    a residue-level fit is at hand (owner): synthetic data first.
    `[analysis] exclude_times` (owner): times left out of the whole analysis (species table
    and so every comparison, residues, protection factors, fragment uptake), as rejected
    spectra are (`Results.without_times`; the bundle's fits are kept); a control's time is
    refused. The example leaves out 0.1 and 0.8 s.
    Example: hxex_private/analysis_examples/protection_factors/b5_disease.
    Butterfly plot (owner): the "corrected differently" note sat over the middle of the figure;
    it is now an entry of the time legend, as in the residue overlay.

82. **Residue-level protection factors (owner asked, 2026-10-15).** Design in docs/ANALYSIS.md
    ("Residue level"). `residue_protection` fits one `log10_protection` per exchanging residue
    of each state (a sample at K = 1, or a K = 2 rank: apparent) to all its peptides at once:
    weighted squared residuals of `fD` (errors floored at 0.01) plus `r1` times pyHDX's
    total-variation regulariser in log10 units (prolines bridged, uncovered stretches not);
    L-BFGS in float64; errors from the inverse Hessian scaled by the misfit above 1;
    `resolved` where the peptides' coverage vectors span the residue's unit vector;
    `compare_residue_protection` (no significance where either side is not resolved); figures
    `protection_residues` and `protection_residue_comparison_<sample>`; `[analysis]
    protection_r1` (default 0.1). Progress and cancellation hooks.
    - **The loss is not convex.** The exponentials saturate and neighbouring residues can trade
      values: plain L-BFGS from the peptide-level starting values stopped at chi-square 2.05
      per point where the true values give 1.01 (synthetic, 80 residues, 70 peptides, 9
      times, noise 0.01). Continuation over `r1`, single-residue grid sweeps and pair swaps
      did not escape; basin hopping did: windows of 2 to 5 neighbouring residues get random
      values (half of them at the worst-fitting peptide), a 30-iteration descent follows, and
      the move is kept if the loss drops; 40 moves without a gain end it; seeded. A descent
      never returns a point worse than its start (a failed line search can). Values are held
      between -3 and 14 by a smooth wall.
    - **Synthetic check** (six seeds): on residues resolved and inside the time window, the
      error is 0.06 to 0.16 log10 units with misfit 1.02 to 1.09 and 79 to 100 % of true
      values within two standard errors; about 9 s per 80-residue state on the CPU. `r1` from
      0.03 to 0.3 performs alike, 1 slightly worse: default 0.1.
    - **pyHDX 0.4.3 as a reference** (imports only with a stub for proplot, which the
      installed matplotlib breaks; hxex's intrinsic rates substituted, its factors converted
      from `kint / (1 + P)`): on the same data and residues it ends 0.27 to 0.46 log10 units
      from the truth (r1 0.1 or 1, patience raised to 2,000 epochs; 80 s per fit) with a
      residual error several times the noise: SGD stops in a poorer local minimum of the same
      problem. So pyHDX is not a tight reference here; the known truth is.
    - **Figures (owner):** on B5 some residues had errors of hundreds to thousands of log10
      units (flat data: the factor is undetermined), which made the residue plot useless.
      `protection_max_se` leaves them out of the figures, with a count; the panel title gives
      the misfit. Default 5 log10 units (owner: about half the axis range, so it does not
      distort the plot; at 1, 440 of 468 B5 residue values were left out, the misfit of 170 to
      270 scaling the errors by its square root). The plots are drawn as the residue
      comparison plots (a line with standard-error bars; owner).
    - **Owner's assessment:** for data like B5 (multimodal exchange) the residue-level factors
      do not provide much information. The peptide-level factors make sense, and the
      residue-level fD seems valid, but the protection factors do not translate well to single
      residues. So the residue-level fit is opt-in on its own (owner): `[analysis]
      protection_residues = true`, which needs `protection`; `protection = true` alone gives
      the peptide level.

83. **Retention times kept with the spectra (owner, 2026-10-15).** Mainly to keep the metadata
    together (owner), and used by the theoretical back-exchange fallback when available.
    `rt` (minutes, NaN unknown) on `Spectrum`, `SpectrumSet` and `BinnedSet`, stored with the
    index columns of `spectra` and `binned` (older bundles read as unknown); in the results
    tables (per spectrum, and per series as the median; `retention` there stays the
    back-exchange retention) and the species table (median of its spectra). Sources, the first
    that gives one: `[input] retention_times` (a CSV: `rt` and any of `sample`, `start_seq` and
    `end_seq` or `peptide_range`, `peptide`, `charge`; the row matching on the most columns
    wins; any format, SpecExport included), the peptide table's `rt` column (checked when
    read), and for mzML or Thermo runs with a fixed time the intensity-weighted mean time of
    the scans summed (a kinetics run's scan time is its exchange time: no RT). The fallback
    (`fd_fallback = "kint"`) fits `a sum exp(-k (t0 + b RT))` (`b` not below 0) when at least
    `kint_min_series` controls have an RT; the others are taken at their median RT
    (`fd_model` `b_rt`, `median_rt`). On the B5 data with HDExaminer's Actual RT (144
    sample-sequence rows, 4,042 of 5,261 spectra; 45 of 194 controls without one, at 8.70 min):
    `a` 0.891, `t0` 2,510 s + 2.59 RT (without RT: 0.853, 3,393 s); in-sample scatter 0.045 D
    per amide either way, as item 71's cross-validated gain of about 6 % suggests. Example:
    hxex_private/analysis_examples/quench_backexchange/b5_disease (`make_rt_table.py`,
    `hxex_kint_rt.toml`).

84. **A project file for analysis only (owner, 2026-10-15; item 80's gap).** `hxex analyse`
    accepts a project file without `[input]` when it has an `[analysis]` table, for comparing
    bundles (`[analysis.runs]`, `merge`, labels): `load_project(path, analysis_only=True)`
    gives `input = None`. `hxex run`, `check` and `resume` still need `[input]`, and their
    error says that a file with only `[analysis]` is for `hxex analyse`. The ScramPep ETD
    against EThcD comparison is now such a file (`scrampep_compare/hxex.toml`).


85. **CPU speed, part two, and an exact multimodality screen (owner asked, 2026-10-15).**
    Profiled on 100 series of the intact-peptide benchmark (1,889 measurements, K = 1 to 3, the
    benchmark's settings, CPU float32, 12 threads): 87 s, of which Adam 65 s, the polish 12 s
    and the k-means starts 10 s.
    - **Model step** (`hxex.model.mixture`): the pmf at the integers comes from the ratio of
      successive terms (`sum_{i<u} log((n - i) / (i + 1))`, one `log` and a cumulative sum
      instead of an `lgamma`), evaluated once for both the normaliser and the bins; and the
      prediction convolves the amplitude-weighted mixture once (`predict_mixture`; the shape is
      the same for every component, so `sum_k A_k (B_k * D) = (sum_k A_k B_k) * D`) instead of
      each component. Forward plus backward 7.9 to 3.8 ms per step; values equal to 1e-13
      (float64, the prototype goldens) and about 1e-5 relative in float32.
    - **Bug found and fixed:** terms above the support, `(1 - p)^(n - b)`, overflowed float32
      where `p` is near 1 before `torch.where` masked them, so the gradient was NaN (0 x inf);
      `_clean_grads` zeroed it, and Adam stopped moving `n` and `p` of that measurement. 134 of
      the 1,889 had NaN gradients at their fitted parameters. Masking to `-inf` before any `exp`
      removes it (test). float64 was not affected.
    - **k-means starts** batched (`kmeans_starts`): a k-means on the bins weighted by their
      rounded intensities (the same clusters as clustering that many copies, as before, without
      making them), vectorised over the measurements, with centroids starting at the weighted
      quantiles (deterministic; scipy's random points cannot be reproduced). 10 s to under 1 s.
    - Together: 87 to 42 s; selected K 1,508/319/62 against 1,509/318/62 (one measurement).
      The whole benchmark (26,181) through `analyse`: 10.5 min (17.1 in item 46, under the model
      of that time).
    - **The screen** (`FitConfig.screen`, off by default; `hxex.fit.screen`). Every K-mixture
      predicts the shape convolved with a non-negative distribution on `0 .. ceil(n_max)`, so
      the loss of the best such distribution (non-negative least squares, 0.3 s for 7,741
      spectra) is a floor under the loss of every K. With the Gaussian loss, K cannot be
      selected where its penalty plus twice the floor is worse than the best criterion of the
      Ks already fitted (with a tolerance of 1e-3 + 1e-4 relative for float32), so it is not
      fitted. Exact by construction: it skips only fits that cannot be selected. No floor for
      the Poisson loss (it can be negative) or bins that are not whole numbers: never skipped.
      Skipped fits are NaN (parameters, loss, criteria; failed fits stay `inf`) with an `inf`
      score; `FitResult.fitted(k)` and `require_fitted` tell them apart; the components table
      leaves them out; a reviewer's K override, a spectrum figure or table at a skipped K
      raises; population maps at a fixed K leave the panel empty; the species table logs and
      drops them; the fixed-K bootstrap has no values for them (a warning; it already does not
      resample a measurement without a fit). It needs `k_min = 1`. Off by default because the
      analysis's K=2 descriptions and the one-or-two choice use the K=2 fit of every spectrum.
    - Screen measured on 400 series (7,741 measurements; reference: the full fit on the GPU,
      K 6,256/1,174/311): the floor was below the fitted loss of every K for all of them;
      K = 2 and 3 skipped for 74.9 %, K = 3 for 93.0 %, no multimodal measurement missed. The
      floor of 0 alone (2 nll1 <= 3 ln N) skips 65.7 %. Approximate screens (a quick K = 2 from
      the split and k-means starts, LM only or 50 to 100 Adam steps, a BIC margin) skipped
      only 6 to 10 % more and missed 64 to 353 multimodal measurements: not built.
    - With the screen on the CPU: 7,741 measurements 193.6 to 75.6 s, the whole benchmark 10.5
      to 3.6 min (K = 2 fitted for 26.3 %, K = 3 for 7.2 %). Selected K identical for 7,741 of
      7,741 and 26,180 of 26,181: the one is a K = 3 fit that found another minimum in the
      smaller batch (the rounding of a reduction depends on a measurement's position in the
      batch: 4.4 % of measurements differ in the last bit between two batches at the same
      parameters, padding to the same width does not change that; chunking has the same effect,
      item 30), so its K = 3 criterion was 82.7 against 66.2 and it selected K = 2.
    - Measured, not changed (results would change; owner to decide): K = 1 hardly needs its
      restarts (one restart matches the six within 0.1 % of the loss for 98.4 %, six of 200
      Adam steps for 99.8 % at half the cost), K = 2 does (one to three restarts or 200 steps
      changed the K = 1 against K = 2 decision of 24 to 75 of 7,741). With the screen on, K = 1
      is about three quarters of the fit. Batches up to about 4,000 rows cost less per row
      (2.15 ms per 1,000 rows at 1,889, 1.35 at 7,556), so fitting the restarts of a K side by
      side would help small projects (not done).
    - GPU parity: one test asked for the same K on a tie (criteria 0.02 apart); float32 on the
      GPU now reaches a K = 2 fit 0.02 better than float64 on the CPU, so the test checks the
      measurements whose choice is not a tie. Test suite on this machine: 7:50 before, 5:27 after.
    - On the GPU (RTX A4500, float32; measured after item 86): 7,741 measurements 69 to 31 s
      with the defaults; the screen did not help there (34 s: a step costs about the same for
      2,000 rows as for 8,000), with `k1_adam_steps = 200` 25 s. The whole benchmark (26,181)
      1.03 min with the defaults, 0.61 with the screen, 0.56 with both.

86. **Fewer Adam steps for K = 1 (owner asked, 2026-10-15; item 85's measurement).**
    `FitConfig.k1_adam_steps` (None, the default: `adam_steps`) sets the Adam steps of the K = 1
    fit only; the larger K keep theirs, as they need them (item 85). Whole intact-peptide
    benchmark (26,181 measurements, CPU, 12 threads, controls as configured there), 400 against
    200 steps for K = 1:
    - Time 10.46 to 9.23 min; with the screen (item 85) 2.43 min (3.6 with 400 steps).
    - Selected K differs for 2 of 26,181 (one 2 to 3, one 3 to 2), 3 with the screen.
    - K = 1 loss worse by more than 0.1 % for 0.18 %, by more than 0.5 for 9; never better.
      K = 2 losses (from the split of the K = 1 fit): 8 worse and 7 better by more than 0.5;
      24 of 4,349 spectra selecting K = 2 in both moved a mean by more than 0.1 D.
    - K = 1 means: median change 3e-6 D, 99th percentile 0.022 D, 118 above 0.05 D (maximum
      0.74). Controls: `m0` changed by more than 0.01 D in 142 of 1,372 series (maximum 0.075
      D), `mfd` in 16 (maximum 0.093 D); fallbacks unchanged.
    - **Finding:** the undeuterated spectra whose K = 1 mean moved by more than 0.01 D (281 of
      3,248) have the same loss (median change 0, 95th percentile 3e-6): the K = 1 mean of an
      undeuterated spectrum lies in a flat valley of the loss, so it is set by the data only to
      a few hundredths of a deuteron (up to 0.07 D here), and the optimiser's path decides
      where in it the fit stops. Neither value fits better. This uncertainty of `m0` is not in
      its reported error.
    - Owner's decision (2026-10-15): the default stays 400; `k1_adam_steps = 200` with
      `screen = true` is the recommended setting when speed matters (README).
