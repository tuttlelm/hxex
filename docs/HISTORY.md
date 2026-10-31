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
real pooled folders (the private pooled dataset, 318 entries each) the entries, candidate lists and classes equal
the pooled `metadf` columns `sample_matches` and the protein-class column. Those three tests read their
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

Findings (numbers from the October 2026 runs of the private pooled dataset, 4,235 spectra, and the other test sets):

- **The 19Feb2024 `alldeutdata` is not an oracle for the current picker.** It was made by an
  older version (constant spacing 1.006227 per bin and a different bin count). The oracle is the
  October 2026 pooled runs, which come with `allrawdata` and use the current spacing: the
  reference function reproduces the first of the two pooled runs exactly (2,232 spectra, six
  columns, 1e-9) with `Nterm_subtract=2, Zero_Filling=3, setNoise=50, Max_Pops=1,
  Peak_Resolution=50, Env_threshold=0.1`. A numpy prototype of the same logic also matches
  both pooled runs exactly (0 of 4,235 differ).
- **The zero count is cumulative, not consecutive.** Zeros anywhere to the right of the mean
  m/z add up, so a valley inside a multimodal envelope can stop picking early. A consecutive
  rule would pick more bins in 21 % of spectra, and in 7 % the extra bins contain intensity
  above the threshold (whether that is signal or noise has not been examined).
- **The picked data depend on `Max_Pops`** (through `min_pts`), a fitting parameter.
- **Below-threshold intensities are kept** (10 % of bins hold a non-zero value under the
  threshold) while being counted as zeros; 32 % of spectra run to the bin cap instead of
  stopping on zeros.
- **Shoulder rejection and the 10-point rule have never been exercised by real data here:** in
  every dataset tested (Bimodal, B4, the private pooled dataset) no window has 10 or more points, so these data
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
   private pooled runs (run A 2,232 and run B 2,003 spectra) the old rule left out bins above the
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
with the picked table from the same code): in the four private pooled runs (A, B and their B5 counterparts), every bin both
pickers produced is identical (m/z and intensity, exactly), every bin only pyHXExpress kept is
below the threshold, the envelope columns are equal wherever the bin counts are equal, and the one
all-zero spectrum is flagged `empty`. A golden file of 11 spectra covering
the cases above (made by `tests/golden/make_picker_golden.py`) is kept outside the repository
because the data is unpublished; the test reads it from `HXEX_PICKER_GOLDEN`, and the whole-run
check reads a run directory from `HXEX_PICKER_ORACLE`. Both skip when unset. The 19Feb2024 Bimodal `alldeutdata` came from
an older picker (constant bin spacing) and is not an oracle.

**Unvalidated:** shoulder rejection and the 10-point rule. In every dataset available (Bimodal,
B4, the private pooled dataset) no window has 10 or more points, so they are covered by synthetic tests only.
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
replicate data of the private pooled runs (A and B, the peak-picked stick heights of 12,000 bins
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
across series. The noise of the private pooled dataset's series (peak-picked stick heights) is variance =
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
measurement; `FitResult.noise_kind` records it). On the private pooled dataset 158 of 168 series can be
estimated (16 have one replicated time point, 5 none).

**Safeguard against bad spectra.** Each spectrum of a time point with three or more replicates
gets a score: the mean over bins of its squared difference from the mean of the other replicates
divided by the variance the gain predicts for that difference (`gain x mean x (1 + 1 / k)`),
about 1 for a consistent spectrum (private pooled dataset: median 0.93, 90th percentile 2.3; 3.5 % flagged in
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
    is unusable: on the private pooled subset (40 peptides, 371 spectra, noise estimated from
    replicates) the median is 4.9, because the model is never exactly right, and on the Bimodal
    data with the 3 % noise setting it is 0.17, because that noise is far above the real one; the
    floor of 1 removed 11 of 12 false flags on Bimodal (fits better than the noise model, flagged
    only because their series was cleaner) and left the one real bad fit (series 14, 7 s,
    deviance 44). `low_intensity`: total intensity below 0.3 of the mean of the other replicates
    at the same time (not assessed without a replicate: the total changes along a time course,
    and against the series median the 5th percentile on the private pooled subset is already 0.31). `noise_outlier`:
    the replicate-consistency flag of the noise estimate. On the private pooled subset: 9, 11 and 23
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
      where in it the fit stops. Neither value fits better.
    - By deuterium level (K = 1 mean as a share of the exchangeable sites; spectra whose loss
      did not change): moved by more than 0.01 D for 238 of 3,126 below 0.02, 44 of 126 at
      0.02 to 0.05 (99th percentile 0.19 D), 26 of 613 at 0.05 to 0.10, 35 of 18,123 at 0.10
      to 0.80 (99th percentile at most 0.002 D) and 4 of 104 above 0.80. So it is an effect of
      the ends of the range, not of the controls: 74 of the 347 that moved are deuterated
      samples with little or almost full uptake.
    - **Data, not the optimiser** (2026-10-16; 325 series, 6,255 spectra: every series with a
      moved spectrum plus 40 at 0.2 to 0.6): a long fit (float64, 3,000 Adam steps, 300 LM
      iterations) lands at a third point in the valley (median 0.014 D from the 400-step fit
      below 0.05, 0.034 D at 0.05 to 0.8) without a lower loss (median change -1e-6, at most
      0.004): more steps do not settle it. The shift between 400 and 200 steps is about one
      standard error: below 0.05 a median of 0.024 D against a curvature (Laplace) standard
      error of 0.023 and a bootstrap one of 0.031 (shift / bootstrap SE 0.86); at 0.05 to 0.8,
      0.041 against 0.031 and 0.072 (0.62). Unmoved spectra at 0.2 to 0.6: curvature and
      bootstrap errors agree (0.075 and 0.074 D). At the bound (the mean pressed to 0, p or n
      saturated) the curvature error is 0 and not usable; the bootstrap's is.
    - The controls' errors cover it mostly: `m0_se` is the scatter of the undeuterated
      replicates, each fitted alone, so where in the valley each lands is part of it. Of the
      142 series whose `m0` changed by more than 0.01 D, 140 have replicates; the change is a
      median 0.89 of their standard error (0.021 D; 0.007 D over all series), above 2 for
      21 % (two or three replicates estimate a standard error poorly, and replicates fitted
      the same way may stop alike). Earlier in this item `m0`'s error was said not to include
      this: wrong, corrected here.
    - Owner's decision (2026-10-15): the default stays 400; `k1_adam_steps = 200` with
      `screen = true` is the recommended setting when speed matters (README).

87. **Shared envelopes for the conditions of one ion, and the `poor_envelope` flag (owner,
    2026-10-16).** A fragment's envelope is a mixture of ion types (`c-1` and `c`, `z-dot` and
    `z+1`) whose proportions the theory cannot predict, so it has to be measured, normally from
    the undeuterated control (item 35, `[run.envelope] mode = "mixed"` or `"infer"`).
    - **Owner's evidence** (August 2025, private): the undeuterated envelopes of every fragment
      of the 11-residue peptide of the private fragment dataset compared at low and high
      in-source energy, for ETD (3+), EThcD (2+) and UVPD (3+). Conclusion: they agree unless
      something else is wrong with the data. In that comparison the ETD and EThcD `c-1` and
      `z-dot` ions and the precursors agree closely; five UVPD `a`/`x` ions differ in shape, and
      five ions are seen at one energy only. The mixture differs between activations (EThcD's
      `c-1` ions carry more of the lighter type than ETD's), so an envelope is shared across
      energies, not across activations.
    - **Built:** `EnvelopeConfig.share` (opt-in, `[run.envelope] share = true`). The series of
      one sample, ion, charge, parent, activation, pre-activation and precursor that differ only
      in their scan filters or list label (`hxex.io.envelope_groups`) get one envelope from all
      their references. The conditions are compared pairwise first: two agree when the envelope
      fitted to each one's references fits the other's within `max_residual` of its own fit,
      both ways. The shared envelope is fitted to those that agree with the most others; the
      rest (`agrees` False) are flagged; with no agreement at all (two conditions that
      disagree) it is fitted to all and all are flagged. A condition without references of its
      own takes the shared envelope. The envelope table gains `own_residual` and `agrees`
      (older bundles: NaN and True); `n_reference` is the series' own count with `share`.
      The first version compared each condition with the pooled fit, which an outlier pulls
      towards itself (the synthetic test caught it); the second compared absolute residuals,
      which marked as disagreeing two conditions with the same references whenever the mixture
      fitted neither.
    - **Flag `poor_envelope`** (all runs with `mixed` or `infer`, shared or not): the series was
      left with the theory (no usable reference), its envelope does not describe the spectra
      it was fitted to (`good` False), or, with `share`, its references disagree with the other
      conditions' or the shared envelope does not describe them. Previously these were only
      counted in the log.
    - `[run] envelope_fallback_time` was already readable from the project file (not a gap,
      contrary to what I told the owner at first).
    - **Real data** (the private ScramPep ETD example: gentle and harsh scans of one
      undeuterated file, 3 components): with `share` the weights are identical to the
      per-condition ones (the references are the same spectra), no condition disagrees, and the
      run takes the same time. 36 of 68 series are flagged `poor_envelope` (3,686 of 12,678
      spectra), as before the change but now visible: 6 with no reference at all (`c-1:1` at
      1+ and 2+, `z-dot:1` at 2+, each condition), 28 whose mixture does not fit (residual
      0.16 to 1.0 against 0.15; 22 of them 2+ fragments, 4 1+, 2 3+) and 2 (one ion, both
      conditions) whose mixture fit at 0.105 by `estimate_envelope`'s residual but at 0.22
      when the reference was picked again with the final weights' bins (fixed in item 88:
      34 flagged since). In most of the 2+ references the 2+ ion is
      not there to speak of (`c-1:4` 2+: the picked bins hold scattered noise; the strongest
      nearby peaks belong to another ion), so their kinetics fits rest on no measured envelope.
      Not yet checked on data where the conditions have separate undeuterated files.

88. **The envelope's residual on the bins it is used with (2026-10-16; item 87's finding).**
    `estimate_envelope` alternates picking the references with the current weights' bins and
    fitting new weights, but reported the misfit of the last weights to the references picked
    with the previous weights' bins, and used those weights' bins in the run. On a weak
    reference the rounds do not settle: `c-1:2` 2+ of the ScramPep ETD example cycles with
    period four (weights (0, 0, 1), (0, 0, 1), (0.10, 0, 0.90), (0, 0.12, 0.88); misfit on
    their own bins 0.105, 0.155, 0.22, 0.199), so after the default three rounds it reported
    0.105 for weights whose own misfit is 0.22. Now every set of weights (the equal starting
    ones and each fitted set) is scored on the references picked with its own bins, at the
    best scale, and the best is kept; with `infer` the fit envelope is the average picked with
    the kept weights' bins (it was the previous round's). No extra picking: each round's
    picking scores the weights it was made with, plus one at the end. `residual` now equals
    `reference_residual` of the series' own references (tested). On that example: weights
    changed in 4 of 68 series (two ions, both conditions), `good` in none, and `poor_envelope`
    36 to 34 series (the cycling ion now keeps the weights that fit at 0.105). A run with
    `mixed` or `infer` can give slightly different weights and residuals than before.

89. **A peptide's own retention window (owner asked, 2026-10-16).** A project file had one
    `rt_window` for every peptide (`[input.target]`, `[[input.fragments]]`), so an LC-MS file
    per labelling time could not have its peptides read from MS1: without a window each
    peptide's MS1 spectrum is summed over the whole run. The peptide table now takes optional
    `rt_start` and `rt_end` columns (minutes, like `rt`; `entry_rt_window`, checked when the
    table is read: both or neither, numbers, end after start). They set the window (in seconds)
    of the peptide's target and of its fragments, in place of the project's; rows without them
    keep the project's. On a private bottom-up dataset (item 64's: an LC-MS file per labelling
    time, 18 targeted precursors, windows from the owner's list; example
    hxex_private/analysis_examples/fragment_ladders/targeted_lcms): every MS1 parent was
    found in all 17 runs (two undeuterated, one fully deuterated, 14 times from 4 s to 27 h),
    and its centroid was compared with that of the precursor M in the peptide's own isolated
    MS2 scans, run by run. For 11 of 18 they agree (median difference 0.01 to 0.09 D; single
    runs up to about 1 D, and one MS2 M fitted at 0 D at 27 h); for two the median is 0.4 to
    0.5 D (22 residues 4+, and 20 residues 3+, whose undeuterated MS1 centroid is 4 D with
    K = 2 in 12 of 17 spectra: interference in its 2.5 min window); for five 2+ precursors
    fragmented by EThcD the MS2 M values do not follow the labelling time (6, 14, 15, 17 and
    20 residues), while the MS1 values do: the MS2 M is the suspect measure there (an 8-residue
    2+ EThcD precursor agrees).
    Finding: the fully deuterated control is below the 27 h point for most
    peptides (owner: it was meant to be fully deuterated).

90. **A run from the raw spectra of an earlier bundle (owner asked, 2026-10-16).** Reading
    raw files can take most of a run (about 20 min for 17 Thermo files of item 89, against
    under a minute for picking and fitting on the GPU), so a project can now take the raw
    spectra a bundle stored instead: `[input] spectra = "path"` (a bundle directory or `.zip`,
    or a spectra file of `save_spectra`), or `hxex run|check PROJECT --spectra PATH`. With the
    rest of `[input]` present, the same project file reads the input or reruns from a bundle:
    that part is checked as usual, and the read-stage settings that differ from the defaults
    are logged as not used (the stored spectra are already summed and cut to each target's m/z
    window: retention windows, margins, scan filters and targets cannot change). With
    `spectra` alone (and `retention_times`) the format and a folder format's control times come
    from the stored read record. The record is kept (envelope sharing and the kinetics noise
    reference read it) with `stored_in` added. Refused: a missing path, a bundle without raw
    spectra (`keep_spectra = false`), and (CLI) writing the results over the bundle the spectra
    come from. Spectrum ids keep the original project's prefix.

91. **Ladder rates of forward labelling at fixed times (owner asked, 2026-10-16).** The fragment
    analysis (item 67) was built on back-exchange kinetics (one continuous run); item 89's
    dataset is the first forward labelling with an LC-MS file per time (14 times, 4 s to 27 h)
    and MS1 parents read with the fragments. Two fixes were needed. (1) `hxex analyse` kept
    fragment series out of the peptide analysis by (sequence, modification, range); a fragment
    target carries its parent's, so every peptide also read from MS1 was dropped with them.
    They are now left out by series (`Results.without_series`, `excluded_series`, like
    `excluded_times`): the MS1 parents stay, the precursor M read from MS2 scans does not
    (it would otherwise be pooled with the MS1 parent into one species). (2) The fully
    deuterated control (the run's `fd_time`) entered every ladder step as the last point of the
    kinetics; `fragment_uptake` now leaves it out. `fit_decay` already fits a rise (negative
    amplitude), and the ladder figure's time axis is logarithmic, so nothing else changed. On
    that dataset (no intrinsic rates: conditions unknown): 738 single-residue steps in four
    conditions, 468 resolved, 108 residues with at least one resolved rate. Where the ladders
    are dense and the signal strong the conditions agree: in an 11-residue 3+ precursor, ETD and
    ETD after HCD pre-activation give 0.037 and 0.040 /s, 0.039 and 0.054 /s for two
    consecutive residues and 1 to 9e-3 /s for the next two, and the 2+ EThcD ladder of the same
    peptide matches. Elsewhere one residue's rates often scatter over two orders of magnitude
    between conditions and ion types: single replicates, and a difference of two fragments
    carries both their errors. The shortest fragments (two or three residues) are noisy, and
    the 2+ precursors' EThcD M did not follow the time for five peptides (item 89).

92. **Series figure titles (owner, 2026-10-16).** The series title was placed at the default
    fraction of the figure height, which on a tall grid (a 17-time series is 48 in) put it about
    1 in down, over the first row; it is now placed in inches inside the band kept at the top.
    A fragment series' title and file name now name the fragment (`z-dot:14 z=3, parent 51-72
    z=4, ETD`; file `..._59_72_z_dot_14_z3_parent_z4_ETD`): before, they showed only the
    fragment's range, sequence and charge, so the same fragment in two conditions had the same
    name. The file name leaves out the sequence so that the fragment survives the 60-character
    limit. Peptide series keep their names.

93. **One m/z range for every panel of a series (owner, 2026-10-16).** The panels of a series'
    spectra figure each scaled to what they drew, so their m/z ranges differed with each
    spectrum's raw window and picked bins. Every panel now has the series' expected range, the
    lowest to the highest predicted m/z of the picked bins of all its spectra (undeuterated to
    beyond fully deuterated), padded by 10 % of its width on each side (`padding`); without a
    picked table, the panels' own ranges together.

94. **Reading scan files: where the time went (owner, 2026-10-16).** Reading the 17 Thermo
    files of a targeted LC-MS dataset (about 7,300 scans each, 3,118 targets: 18 precursors in
    four fragmentation conditions, three ion types) took about 23 min, about 80 s per file.
    Profiled on one file (107 s under the profiler): 66 s in `ScanTable.select`, whose
    activation test split and compared every scan's activation string in Python, once per
    target; 29 s in `ThermoFile.read_scans`, which masked each scan's points against each of the
    3,118 target windows in turn; 5 s summing scans; 2 s building the scan table through
    pythonnet; under 1 s converting the arrays. pythonnet was not the cost. Now the text tests
    of `select` (activation, pre-activation, filter) run once per distinct value (a file has a
    handful) and are cached on the table, and the windows are merged once into sorted, disjoint
    ranges tested with one `searchsorted` per scan (`merge_windows`, `in_windows`; the mzML
    reader shares both). One file: 80.6 to 8.7 s, its spectra identical (m/z, intensities, ids,
    times, retention times); all 17 files: 138 s. The warning for a target without scans in a
    file is now one line per file (`no scan for N of M targets, for example ...`; each target
    at debug level): in a targeted method each precursor is fragmented in only some
    conditions, and the 1,550 lines per file buried the log.

95. **Audit of 2026-10-17, Session A: the bug fixes (2026-10-17).** The items of
    `docs/AUDIT_2026-10-17.md` marked for Session A, each with a test, in the order the document
    gives. Validation: a run's `time` is a number of seconds or absent and `rep` a whole number
    of 1 or more, with errors naming the run (`input.runs #2 (file.mzML)`); `false` for a
    table-valued `[analysis]` key means none; a retention-time row with fewer or more cells than
    the header, or naming no spectrum, is a read error with its line; `hxex check` prints the
    summary first, then every problem on its own line (missing peptide table, data directory,
    retention-time table, Thermo directory or run file, an existing bundle, a peptide that does
    not tokenise, named by file, line and token with the modX hint), and accepts an analysis-only
    file. Controls: with `fd_fallback = "kint"` and too few usable or agreeing controls, the
    rated series get the d2o floor (half of `d2o_fraction`) like the series without rates (four
    peptides with a fully deuterated level of 0.1 per amide gave a tenfold inflated correction);
    reruns from stored spectra take the format and both control stamps from the stored record
    whenever `spectra` is set (`--spectra` included; the inherited control times follow; one
    warning when the file says otherwise), and a record that cannot be read yet leaves the
    file's settings until the spectra are read (`hxex check` reports it; `hxex run` refuses it
    before anything else); `exclude_times` is checked against the controls' undeuterated time.
    Protection: an unknown `fD_se` takes the median of the peptide's (residue fit: the state's)
    known errors, none known gives an unweighted fit (a NaN error got the 0.01 floor, five times
    the weight of a good point); peptides without a finite value leave the residue fit (logged;
    `n_times = 0` in `protection.csv`); `modified_peptides` applies to the residue fit, each
    modified form a state of its own with its sample's sequence and conditions
    (`residue_protection(modified=)`); `compare_protection` pairs reference peptides as
    `compare_peptides` does (`compare._pairs`), so two forms over one range no longer collapse;
    a sample whose peptides mostly lose deuterium with time is refused with a message naming
    the two-direction model. Fragments and envelopes: a RunSet skips runs whose reader is not a
    scan reader instead of losing every fragment output; once fragment series exist, a
    ValueError is raised, not swallowed, and a missing K = 1 has its own message; the closure
    warning fires for a single run again; shared envelopes with three or more conditions
    re-admit the conditions the core describes within `max_residual` and refit once,
    `n_reference` is each series' own references with signal, and `own_residual` and `agrees`
    stay on record when the group falls back; the envelope stage has progress and cancellation
    hooks (stage `envelope`, 3 % of the units before `pick`); the species retention time is the
    median over the spectra the tables use (not rejected, excluded or, with `exclude_flagged`,
    flagged); in an LC run a fragment takes its parent's retention time (the parent read before
    it, as a project orders its targets), in a kinetics run no spectrum has one. Engine and
    plumbing: the tables and figures of `hxex run` and `resume` use the carried review
    (`RunResult.review`, `Results.from_run`); `Measurements` requires whole-number bins and the
    fractional convolution path is gone (`integer_grid` raises; `_convolve_direct` stays as the
    reference the fast path is tested against; `WORK_FACTOR_DIRECT` removed); the screen logs
    the measurements whose NNLS floor did not converge and computes no floor when only K = 1
    is tried; `write_bundle(review=)` is checked as `set_review` is (`_check_review`);
    `resume --figures` writes figures with the default plot settings when the run had none,
    without changing the run's record; the read's m/z windows are recorded (`windows` in the
    scan-file record) and a rerun from stored spectra whose picker reaches beyond them warns
    (`stored_window_excess`); the "not used" line no longer lists a default `data_root`;
    `describe()` says "(none: analysis only)". Docs: BUNDLE.md (`poor_envelope`, `shared`,
    `own_residual`, `agrees`, `stored_in`, `windows`, the NaN rows of screened Ks), ANALYSIS.md
    (error weighting, direction, modified forms, prolines bridged). No table column was renamed;
    `n_reference` of shared groups changed meaning (own references with signal, not the count
    of own reference spectra). Low item 3 (resolve the stored spectra lazily) was reconciled with
    high item 3: the record is read at load time when it can be (the manifest only, to settle the
    format and the control times), and otherwise when the spectra are read. GPU parity with the
    screen: `tests/test_gpu_parity.py` (run on the GPU build). Sessions B (the overwrite
    contract) and C (readability) and the design items remain.

96. **Audit of 2026-10-17, Session B: the overwrite contract (2026-10-17).** Three kinds of
    content live in or beside a results folder and are now treated by kind: computed results
    are replaced, annotations are carried and never silently dropped, the user's own files are
    never touched. `hxex run` on an existing bundle still refuses; the message names
    `--overwrite`, `--keep-previous` and another `output.bundle` (`hxex check` likewise). The
    review layer carried over with `--overwrite` keeps every entry: those the new run cannot
    apply (spectrum dropped, chosen K not fitted) get `stale = true` in `review.parquet`
    (`ReviewTable.stale`, `mark_stale`, `n_stale`; old bundles read as all current), their
    status and note stay on record and count in `status_of`, `effective_K` ignores their
    chosen K, `update` makes an entry current again and `_check_review` then checks it as any
    other; one log line gives the counts (before: entries the run could not hold were left
    out with a warning). The manifest gets a `provenance` block (`hxex.io.provenance_of`;
    `write_bundle(provenance=)` records it as given, a resumed bundle keeps it): the path
    replaced, its stamp and hxex version, when, where it was moved to, `settings_changed`
    (the dotted paths of the run settings that differ, `settings_changed`; the spectra source
    counts once, its read stamps ignored) and the review counts; `hxex info` shows it, the
    review counts and the generations. `--keep-previous` (`[output] keep_previous`,
    `run(keep_previous=)`) moves the bundle to `name.N` (`name.N.zip` for a zip) with its
    `name_tables` and `name_figures` folders to `name.N_tables` and `name.N_figures`
    (`hxex.io.keep_previous`, sizes logged; `generations` lists them), just before the new
    bundle is written; N is one more than any number ever used beside the bundle (a folder or
    file starting `name.N` counts), so a name is never reused; an explicit `output.tables` or
    `output.figures` directory is not moved (logged). Something at the path that is not a
    bundle is refused before anything is fitted, with `overwrite` too (before: fitted, then
    refused at the write). The tables and figures folders keep a listing of hxex's files,
    `hxex_listing.json` (`hxex.io.replace_outputs`, `output_listing`): `write_tables` and
    `write_figures` remove the listed files, refuse a file of yours named like a new one
    before anything changes (`OutputError`; a folder without a listing, written before this
    item, is refused only on such a clash) and write the listing even when they fail
    part-way, so a stopped write is still known as hxex's. The owner's decisions: `_tables`
    and `_figures` stay siblings of the bundle and generations keep the "bundle name plus
    suffix" rule (`results.1_tables`), so every command finds a generation's folders as it
    finds the current ones; the docs and setup say plainly that `<bundle>*` folders are
    program folders (README, the template's `[output]` comments, `hxex init`'s closing hint,
    BUNDLE.md, the refusal message). Tests: `tests/test_outputs.py` (new), the review, provenance
    and generation tests in `tests/test_bundle.py`, `tests/test_run.py` and `tests/test_cli.py`.

97. **Audit of 2026-10-17, Session C: the readability pass (2026-10-17).** The package read as
    a new user would read it. Template: every generated `[run.*]` and `[input.scans]` line
    carries its meaning and unit as a trailing comment (`hxex.config._COMMENTS`, wrapped at the
    comment column; a setting too long for one goes first and its comment follows), each table
    a comment of its own (`_SECTION_COMMENTS`; the empty `[run.picker.amide_rules]` and
    `[run.registry]` tables explained), the `[[input.fragments]]` block explains the
    `TYPE:LENGTHS:CHARGES` syntax, the `[analysis]` scalars come before its sub-tables (the
    `protection*` lines moved up), the `quench` table has an example inline value, and settings
    that share a name elsewhere cross-refer (`min_population` in the fit, the filters and the
    choice; `d2o_fraction` of the labelling buffer, the kinetics' solvent and the controls;
    `reference` as a sample or a rate set; `min_time` on the scan and the exchange clocks).
    Project files: TOML values are type-checked where they are read (`_check_types`, from the
    config classes' annotations), so `[run.fit] k_max = "3"` is reported as `[run.fit] k_max =
    "3": expected a whole number`, a sub-table's own value errors name the table (`[run.fit]:
    need 1 <= k_min <= k_max`), `[analysis]` and `[input.scans]` likewise; `ci` below 1 is
    refused as a fraction where a percentage is meant; a file read for `hxex analyse` only may
    leave `project` out; `hxex analyse` with `coverage_times` naming times no peptide was
    measured at (a project without time-0 spectra) falls back to every time with a warning
    (`NoPeptideAtTime`). `[run.picker] noise` is `noise_threshold`; `noise` stays as an alias
    (`PickerConfig(noise=)`, and in bundle manifests written before). Messages: `hxex check`
    says what an off stage means (controls: no back-exchange correction); `hxex run` prints
    where the tables and figures went; the analysis warnings name their TOML keys and the
    filter summary lists only non-zero counts. Tables (old -> new): `status` ->
    `review_status` (spectra, components), `fallback` -> `control_fallback` (series), `n` ->
    `n_replicates` and `mean_sd` -> `replicate_sd` (uptake); `centroid`, `review_status` and
    `control_fallback` are defined in `hxex.report.tables`; the residue and coverage tables
    get `no_amide` (a proline or the protein's first residue(s): the row is empty whatever the
    coverage). Figures: `hxex_overview` -> `overview`; a series' file is its label alone
    (`figure_stems`; the `data_id` is added only where two series would share a stem); the
    populations page is left out of a series with one population throughout (`series_kinds`).
    Docstrings: the config classes read as user text, one line per field with its unit, and the
    developer asides (HISTORY numbers, owner dates, pyHXExpress and prototype comparisons) left
    the docstrings of 30-odd modules; evidence pointers moved to `#` comments. The package
    docstring describes the package and `hxex` exports its entry points lazily (`hxex.run`,
    `load_project`, `read_input`, `open_bundle`, `Results`, `write_tables`, `write_figures`,
    `write_analysis`, ...). README: quick start, inputs (the five formats in a line each, the
    peptide-table columns, the control convention, the units), a Python example and the CLI
    route for Thermo files. No private script reads the renamed columns (checked with the
    owner in the audit).

98. **A picking window from the instrument's resolving power (2026-10-18; audit design item
    4).** The picker's window was a constant relative half-width, `resolution_ppm` (50 ppm,
    pyHXExpress's `Peak_Resolution`): a mass tolerance with no relation to the peak width, and
    at high charge wider than the isotope spacing (at m/z 1000 and z = 20 a 50 ppm half-width
    is 50 mDa, the whole spacing, so each window held both neighbours and the shoulder rule
    zeroed real peaks). `PickerConfig.resolving_power` (R quoted at `resolving_power_mz`, 200
    by default) now sets the window from the peak width instead: `R(m) = R0 (m0 / m) **
    resolving_power_exponent` (0.5, Orbitrap; 0 TOF; 1 FT-ICR), half-width `window_fwhm`
    (1.0) peak widths `m / R(m)`, never below `mass_accuracy_ppm` (10; the calibration error)
    and never above half the isotope spacing `1.00235 / (2 z)` (`half_width`,
    `window_edges`; the ppm window keeps its exact multiplicative edges and no cap, so every
    earlier result is unchanged). For R = 60,000 at m/z 200: 29 ppm at m/z 600, 46 ppm at m/z
    1500, 25 mDa instead of the 37 mDa peak width at m/z 1000 with z = 20. Owner's decisions:
    Orbitrap scaling and one FWHM as defaults; the floor as its own key; the cap in the new
    mode only, its effect on the ppm window to be measured on the private pooled and intact-peptide
    data before any change there. `resolving_power = "recorded"` takes each spectrum's value
    from the scan files: Thermo raw files record it per scan in the trailer (`Orbitrap
    Resolution:` on the Tribrids and Exploris, `FT Resolution:` before; the example file, an
    Orbitrap Ascend, has 120,000 at MS1 and 30,000 at MS2) and name the instrument model,
    which says whether R is quoted at m/z 200 or 400 (`hxex.picking.reference_mz`: 400 for
    the Elite, Velos, XL, Discovery and LTQ Orbitrap generation, 200 otherwise); mzML files
    from ThermoRawFileParser carry neither. The readers record them per spectrum
    (`Spectrum.resolving_power`, `resolving_power_mz`; `SpectrumSet` columns, in
    `spectra.parquet` and spectra files, NaN when unknown, absent in files written before),
    per scan (`ScanTable.resolution`) and per file in the read record (`source["instruments"]`:
    model, reference m/z, distinct values per MS level; None for mzML). A spectrum without a
    recorded value keeps the ppm window (one warning with the count). `hxex check` prints the
    instrument and the recorded resolving powers of each raw file and the setting that would
    use them. The spectrum figures draw the window actually used (`spectrum_data` `window`,
    `window_label`: "±50 ppm window", "window from R 60000 at m/z 200" or "window from the
    recorded R 120000"); the envelope references carry their recorded values too. Template
    comments and PEAK_PICKING.md describe the settings. Tests: `tests/test_picking.py`
    (`test_window_from_the_resolving_power`), `tests/test_thermo.py` (recorded values through
    the readers, the read record and `hxex check`), `tests/test_spectra_file.py` (round trip,
    old files read as unknown), `tests/test_report.py` (the drawn window).

99. **The half-spacing cap on the plain ppm window: measured (2026-10-18).** The cap of item 98
    (half-width at most half the isotope spacing, `1.00235 / (2 z)`) binds only where the ppm
    half-width exceeds it, that is where m/z times charge exceeds 10,023 at 50 ppm. On the two
    bottom-up datasets it never does, so applying it to `resolution_ppm` would change nothing
    there, exactly: the intact-peptide benchmark reaches m/z times charge 4,950 (26,333 spectra,
    charges up to 6, m/z up to 1,777) and the private pooled dataset 3,844 (318 entries,
    charges up to 5). On the top-down example file (an intact protein at charge 20, m/z 1,008;
    23 sums of 20 MS1 scans) a 50 ppm half-width is 50.4 mDa against an isotope spacing
    of 50.1 mDa, so every window holds both neighbouring peaks (21 points per window), and the
    shoulder rule then zeroes every bin whose neighbour is taller: a median of 27 of the 95
    real envelope bins per spectrum (the apex and flat stretches survive, the flanks go), the
    kept tail beyond bin 110 is the neighbours' flanks picked at 20 to 30 ppm offsets, and
    the envelope centre is biased by about 9 bins (59.0 against 50.1 on one spectrum; a median
    shift of 8 bins over the run). With the cap (25 mDa, 10 points per window), with the 20
    ppm window the earlier notes used, with the shoulder rule off, or with
    `resolving_power = "recorded"` (120,000 at MS1) the pick is the same within a bin (centre
    49.4 to 50.9, 98 strong bins). Conclusion: the plain window is wrong exactly where the cap
    binds, and nowhere else; applying the cap to `resolution_ppm` as well would leave every
    bottom-up result unchanged and repair top-down picking at the default settings. Decision
    left to the owner (a deliberate difference from pyHXExpress if taken). Scratch script in
    the session scratchpad only; the numbers are from the stored picked table of the benchmark,
    the pooled folder's entries, and the top-down example file.

100. **The half-spacing cap applies to the ppm window too (2026-10-18; owner's decision after
    item 99: "if pyHXExpress was wrong we do not need to copy it").** `PickerConfig.half_width`
    and `window_edges` cap every window, the `resolution_ppm` one included, at half the
    isotope spacing `1.00235 / (2 z)`; where the cap does not bind the ppm window's edges are
    the exact products `mz * (1 -/+ ppm / 1e6)` as before, so every bottom-up pick is unchanged
    (the picker golden and oracle tests pass bit for bit). On the top-down example at charge 20
    the default settings now give the capped pick of item 99: 98 strong bins instead of 73,
    the envelope centre within a bin of the 20 ppm and recorded-resolving-power picks, no
    spectrum of the 23 differs from the scratch capped variant. A deliberate difference from
    pyHXExpress (PLAN); PEAK_PICKING.md, the template comment on `resolution_ppm` and the
    user guide draft describe the cap. Test: `tests/test_picking.py`
    (`test_window_from_the_resolving_power`).

101. **Controls by role, not by time; times with units (2026-10-18; audit design item 1,
    session 1).** A `role` column (`undeut`, `fulldeut` or empty) on every per-spectrum table
    (spectra, picked, measurements, fit, bundle) replaces the control times: the readers set it
    from the formats' labels (`Non`/`Full`, `undeut`/`TD`), `role` in `[[input.runs]]` marks a
    control file (summed into one spectrum without a time), and `undeut_at`/`fulldeut_at` in
    `[run.controls]` and `[analysis.controls]` make the spectra at a time serve as a control as
    well (`apply_control_roles`; the match is exact through the shared parser), and
    `undeut_marked`/`fulldeut_marked = false` make those spectra the control instead of the
    marked files (needed by the private targeted LC-MS example, whose analysis replaces a
    fully deuterated run that reads 0.12 D per amide low with the 27 h plateau: with the
    additive `fulldeut_at` alone the two were averaged; the owner confirmed the 27 h spectra
    serve alone, the TD file being under-deuterated). A control's time is NaN unless it is also a time point. Removed: `[input] undeut_time`/`fulldeut_time`,
    `[run] undeut_time`, `[run.controls] undeut_time`/`fd_time`/`atol`, the 1e6 sentinel, the
    collision warning (a real 0 s point beside the undeuterated control needs nothing now) and
    `RunConfig.reference_time`; the old keys are refused with a message naming what replaced
    them, and bundles written before the roles derive them from their recorded control times
    at load (`modernise_manifest`, `legacy_roles`; the file is left as written, so their
    controls keep the recorded times 0 and 1e6 in `time` until the bundle is rerun; checked on
    the intact-peptide benchmark bundle: 3,248 undeuterated and 2,274 fully deuterated
    measurements recognised). Grouping by
    point: `time_point` (a time, or a control role) and the role-aware `time_index` (the
    undeuterated control first, the fully deuterated last) replace equality on the time in
    the noise estimate, the quality checks, the species, residue and uptake tables and the
    series figures. The analysis's per-time figures and comparisons are over time points
    only; coverage is judged on the undeuterated controls by role. Times: `parse_time`
    accepts a number of seconds or text with a unit (`"5 min"`, `"20 h"`, `"1h21m"`) in
    every time setting of the project file (runs, `[run.controls]`, `envelope_fallback_time`,
    `[input.scans]`, `[analysis]` lists); `format_time` labels figures (`20 h`); the
    HX-Express replicate-marker rule stays in that reader. `[input] time_scale`
    (`seconds`/`order`) is parsed and stored; its effect on the plots and analyses is part of
    session 2, with `hxex check`/`info` reporting the roles and the user-guide section's
    figures. Golden picker and engine fixtures unchanged (roles do not enter the picker or the
    fits; the noise references are the same spectra). Private project files updated to the new
    keys.

102. **Times that only order the points, roles in `hxex check` and `info` (2026-10-18; audit
    design item 1, session 2).** `[input] time_scale` now acts: the loader records it in the
    spectra's read record (`source["time_scale"]`; the scan-format branch had dropped the
    setting in session 1, fixed), `Results.time_scale` and `RunSet.time_scale` (which must
    agree) read it back, and a rerun from stored spectra keeps the stored scale unless the
    project file sets one. With `"order"`: `plot_uptake` draws by index when `x` is unset
    (the new default, None, in `PlotConfig` too), refuses `x='time'`, and `write_figures`
    turns a configured `"time"` into the index axis with one warning; `format_time(t,
    "order")` is the bare number, the series, overview and analysis figures label such a
    point `point 2` and name their files `profile_WT_2`; the protection analysis and the
    fragment ladder kinetics refuse with a message naming the setting. Every figure function
    that prints a time took a `time_scale` keyword (the analysis dispatcher passes the
    results'). The figure-label review found the remaining raw seconds in `hxex check`'s run
    line and the `undeut_at`/`fulldeut_at` error messages, now `format_time`; the tables stay
    in seconds. `hxex check` counts the control files per sample of a SpecExport or pooled
    folder from the file names (HX-Express labels are inside the sheets: it says so), names a
    scan-format run's role, and `describe` (its summary) spells out `undeut_at`,
    `fulldeut_at` and `*_marked = false` and shows the time scale; `hxex info` gained
    `roles` (control spectra and series per role, with the times they also carry) and
    `times`. User guide: the controls section got the uptake figures of one series of the
    published B5 data (`docs/images/guide_uptake_*.svg`, a three-peptide scratch run), the
    `check`/`info` excerpts and a section on `time_scale = "order"`. Tests: order-scale plots
    and labels, the analysis refusals, the CLI roles and scale lines, the SpecExport control
    counts, the stored-scale rule.

103. **The run as a variable in the analysis (2026-10-18; audit design item 2).** A
    `RunSet`'s tables no longer name a sample `run:sample`: every analysis table (species
    uptake, residues, coverage, populations, the peptide, residue, population and choice
    comparisons, regions, the protection tables and their comparisons, the fragment uptake,
    ladder rates and closure) has a `run` column before `sample`, and a species is identified
    by `(run, sample, ...)` throughout (`where(run=..., sample=...)`; the comparison tables
    carry `reference_run` beside `reference`). `run` is blank for an analysis of one bundle
    and for a merged sample (its spectra pool several runs, listed in `runs`); fragment rows
    keep their run, with a `merged` flag, because fragment series are never pooled: a merged
    sample is one state (`state_run` '') whose series are `(run, condition)`, drawn side by
    side, and `condition` stays as recorded (the `run:condition` prefix is gone, so
    `[analysis.fragments.labels]` keys match again). `run:sample` is a display label
    (`display_labels`): figure titles, legends and file stems use it only where the same
    sample name occurs in several runs of the table, and the sample's name otherwise; the
    figures are drawn from copies of the tables whose `sample` is that label, so the plot
    functions did not change. Settings name a sample, with the run as a qualifier
    (`resolve_sample`): `reference`, `samples` and `--reference` take a sample name, which
    must be in one run of the set (an error lists the `run:sample` choices otherwise), or
    `run:sample`; a merged sample or a one-bundle sample is always named by its name alone.
    The per-sample tables `[analysis.fasta]`, `[analysis.labelling]`,
    `[analysis.fragments.exchange]` and `[analysis.fragments.labels]` take a run's own entries
    under the run's name (`[analysis.labelling.A.WT]`; `sample_setting` looks up the run's
    entry, then the sample's), and a nested table whose name is not a run of `[analysis.runs]`
    is refused. `merge` is unchanged. The display rule (prefix only where ambiguous) and the
    blank `run` of merged species rows were confirmed by the owner (2026-10-18). The private
    two-run fragment example's labels moved to the per-run form. Tests: the run column and
    labels, the resolution and its errors, the nested settings (fasta, labelling, exchange,
    labels) and their refusal, the merged and unmerged fragment states; the former
    `run:sample` assertions updated. Design note in docs/ANALYSIS.md ("The run as a variable").

104. **Protection factors in both directions, peptide level (2026-10-19; audit design item
    3).** The forward-only refusal of falling series is replaced by one two-direction model,
    `fD(t) = mean_i(e_i + (fD0 - e_i) exp(-kint_i t / protection))`, `kint_i = kforw + kback`
    and `e_i = kforw / (kforw + kback)` of the exchange solvent (`hxex.kint`). Owner's
    decisions of 2026-10-19: the direction comes from the labelling conditions (no direction
    setting, as decided 2026-10-13): `initial_fraction` above `d2o_fraction` is a
    back-exchange; on the corrected scale `e_i` is the equilibrium over `initial_fraction`
    (the fully deuterated reference made in the labelling solvent); back-exchange samples
    without correction are fitted on the absolute scale (`fD = mean / n_amides`, `e_i` the
    equilibrium); peptide level only this session. As decided on 2026-10-18, `fD0` is fitted
    (bounded by the largest `e_i` and 2, three starting factors) unless the series has a
    measured fully deuterated control (corrected, no control fallback), which fixes it at 1.
    Forward labelling is the same model with `fD0 = 0` and `e = 1`, computed as before: the
    forward tests are unchanged. Per-time apparent factors and bounds use the progress
    `(fD0 - fD) / (fD0 - mean e)`. A fitted start with every point at equilibrium (judged
    from the largest start possible) is an upper bound with `fD0` NaN; fewer than three
    times, no value (logged). A sample going against its conditions' direction is refused
    in both directions, with the setting to change. New columns: `protection.csv`
    `direction`, `corrected`, `fD0`, `fD0_se`, `equilibrium`; `protection_by_time.csv`
    `progress`. The residue fit leaves back-exchange states out with a warning.
    - **Errors on synthetic data** (one 11-residue peptide in 10 % D2O at pH 7, 20 degrees,
      `initial_fraction` 0.9, start 0.8 fitted, seven times 10 s to 10,000 s, fD error 0.01,
      200 repeats): at protection 1,000 and 10,000 (the decay inside the range) the scatter
      matches the reported errors (log10 protection 0.023 against 0.022 and 0.017 against
      0.018; start 0.010 against 0.011 and 0.006 against 0.006) without bias. At protection
      100 (fD already 0.46 of 0.8 at 10 s) the start is poorly determined and its error
      understated (0.15 against 0.09; log10 protection 0.096 against 0.081).
    - Not tried on real data: no back-exchange dataset with known conditions yet.

105. **Residue-level protection factors from fragment ladders (owner, 2026-10-19).** The
    owner judged the peptide-derived residue factors poor (B5) and chose fragment ladders,
    where they resolve single residues with low error, for residue-level factors; the
    peptide-derived residue fit is not extended to back-exchange (then removed, item 106). Built on `ladder_rates`
    (whose free `A exp(-k t) + C` already fits either direction): `log10_protection` and its
    se (`rate_se / (rate ln 10)`), `precise` (resolved and `rate_se / rate <= max_rate_error`,
    new `[analysis.fragments] max_rate_error`, default 0.5), and `ladder_residues`
    (`ladder_residues.csv`): the N- and C-terminal ladders' estimates combined per residue by
    weighted mean in log10, se scaled by the misfit above 1, `agreement` from the chi-square
    at alpha 0.05. Owner's pooling rules: conditions (gentle, harsh) never pooled; fragment
    charges kept separate; ion types of one terminus not combined (c-type ions are mixtures,
    c and c-1 read from the same peaks), so each (N ion, C ion) pair is a row. Check on the
    synthetic ladder (8 residues, noise 0.02 D, 60 seeds): each step's scatter in log10 rate
    matches its reported se (0.011 to 0.031 against 0.011 to 0.029), no bias above 0.006,
    the combined se halves the variance as expected, and 94 % of two-ladder residues agree
    at alpha 0.05. Not tried on real data: the fragment-kinetics infusion conditions (needed
    for kint) are still awaited.
    - **Error ranges in the fragment figures (owner, 2026-10-19).** The ladder kinetics and
      closure figures draw +-1 se bars on every point (light, alpha 0.35, no caps; the trace's
      `mean_se`). The residue-rates figure's rate panel already had them; the amplitude panel
      now has them from the new `amplitude_se` (`DecayFit`, `ladder_rates.csv`: propagated
      from the fit's covariance, `A = A' exp(k t_first)`; on the synthetic ladder 0.012
      scatter against 0.011 reported), the protection panel the +-1 se range of
      `log10_protection`. The combined per-residue values are not drawn (a figure change
      waits for the owner's reference).
    - **A band on the fitted curves (owner, 2026-10-19).** On real kinetics data the points
      have no `mean_se` (K = 1 fits without a bootstrap, one replicate per time), so the
      point bars do not show; the owner chose a band on the fitted curves over a bootstrap
      refit. `DecayFit` keeps the fit's covariance (`covariance`, `t_first`) and gives
      `curve_se(t)`; the ladder figure fills +-1 se around every resolved fit (unresolved
      fits stay dashed, no band). Coverage on the synthetic ladder (200 repeats, residue 16):
      the true curve inside the band 62 to 70 % of the time at 10, 50, 200 and 800 s. On the
      private fragment example (about 150 points per trace) the band is about as wide as the
      line except near the ends of the range: the curves are well determined.

106. **Residue-level protection factors from peptides removed (owner, 2026-10-19).** The
    regularised residue fit of item 82 (`residue_protection`, `compare_residue_protection`,
    `ResidueProtection`, `ResidueProtectionComparison`, the figures
    `residue_protection_figure` and `residue_protection_comparison_figure`, the outputs
    `protection_residues.csv`, `compare_protection_residues.csv`, `protection_residues_*` and
    `protection_residue_comparison_*`, the settings `protection_residues`, `protection_r1`,
    `protection_max_se`) is gone. Owner: the factors were unreliable (item 82's B5
    assessment), and an option invites users to take them for reliable; residue-level rates
    and protection factors come from fragment ladders (item 105). Never released (the only
    tag, v0.0.1.dev0, predates it); the code is in a6cb28f and later commits up to 4c3a293. A
    project file with any of the three settings is refused, saying why and where
    residue-level values come from. The peptide-level factors' `_prepare` lost the
    modified-forms split, used only by the residue fit (modified forms stay peptides of their
    sample, as the peptide level always did). The residue-level fD map is unaffected. The
    owner's private protection example script no longer asks for the residue level.
    Comparing fragment-ladder residues between samples (`compare_ladder_residues`) is a
    separate item, for when a dataset with two samples of one parent arrives.

107. **The user guide begun (owner, 2026-10-19; audit design item 5, PLAN Next steps 7).**
    Owner's choices: plain Markdown in `docs/guide/` (no site generator, no new dependency);
    a settings reference generated from the code with a test; the guide starts from the
    input formats (HX-Express, SpecExport separate or pooled, mzML, Thermo raw) and the
    experiment types (bottom-up, top-down, kinetics, each with optional fragments). Pages:
    README (index), project-file, inputs, experiments, outputs, settings (generated), and the
    five sections of the former `USER_GUIDE_DRAFT.md` as pages of their own (controls,
    picking-window, comparing-runs, back-exchange-protection, fragment-residues; the draft is
    removed). `hxex._reference` walks the config classes the parser checks keys against
    (InputSettings, MzmlRun, FragmentSettings, MzmlConfig, `_TARGET_KEYS`, OutputSettings,
    RunConfig's sections, `ANALYSIS_KEYS` with the analysis classes, ExchangeConditions);
    descriptions from `config._COMMENTS` and, for the tables the template writes by hand,
    `_reference._DESCRIPTIONS` (these duplicate the template's hand-written comments; making
    the template read them is possible later). Tests: every key described, the page current,
    every `[analysis]` key listed. Not yet: a worked tutorial on a publishable dataset.
    Behaviours found while documenting (documented as they are; for the owner): (a)
    `record_paths = "relative"` passes `hxex check` for folder formats but `hxex run` refuses
    it; (b) a SpecExport time token such as `5min` is refused with a message that lists `min`
    and `ms` as units (the token takes one letter, s, m or h); (c) the pooled scan uses the
    default modification registry, so a folder whose sequence uses a `[run.registry]` label is
    skipped (inferred from the call chain, not tested); (d) for the scan formats the peptide
    table's `file` and `sample` columns are required but unused (every target is searched in
    every run, and a peptide and charge listed under two samples clash); (e) the
    `FragmentSettings.limit_charge` docstring says charges stay at or below the parent's,
    while `ALL` stops at the parent's charge - 1 and charges written out are not capped.
    All five fixed in item 108.

108. **Five input behaviours found while writing the guide, fixed (owner, 2026-10-19; item
    107).** (a) `record_paths` is checked when the project file is read: `path` or `name` for
    the folder formats (`relative` refused, saying it is for the scan formats), `path`,
    `relative` or `name` for mzml and thermo. (b) A SpecExport time token that is not `Non`,
    `Full` or a number with one letter (`s`, `m`, `h`; `0.8s` as in the published B5 export)
    gets a message saying so, instead of one listing `min` and `ms`. The tokens accepted are
    unchanged. (c) The pooled scan takes the project's `[run.registry]` (`scan(...,
    registry=)`, from `folder_entries`). (d) Owner's decision: for the scan formats the peptide
    table's `sample` ties a row to the runs of that sample (empty or no column: every run);
    `file` and `sample` are optional there (`SCAN_REQUIRED`). `scan_peptides` refuses a
    table sample that no run has and a peptide (sequence, range, charge, modification) listed
    twice for the same runs, naming the line. `read_mzml`/`read_thermo` apply the tie
    (`target_applies`) and record `samples_tie_targets`; a record without it (older bundles)
    is repeated as it was read, every target from every run. `hxex check` counts each run's
    own targets and names a run with no peptide of its sample. The four private scan-format
    projects have one sample, the same in table and runs: their targets are unchanged. Test
    fixtures that labelled rows `S` with runs `WT` now leave the sample empty. (e) The
    `FragmentSettings.limit_charge` docstring and the reference say what the code does.

109. **A generated command-line reference (owner, 2026-10-19).** `docs/guide/commands.md` is
    written from `hxex.cli.build_parser` by the same generator as the settings page
    (`python -m hxex._reference docs/guide` now writes both). Every positional argument got a
    help text (`PROJECT`, `BUNDLE`; also shown by `--help`), and `init --force` says that
    without it an existing file is refused. Tests: the page is current, and every argument of
    every command has a help text.

110. **The guide for users who know no Python (owner, 2026-10-19).** Owner: users are likely
    not computer savvy, and are not expected to know Python. New pages: `install.md` (uv, then
    `uv tool install` from the GitHub zip, so no git, Python, or pip knowledge is needed; tested
    on Linux, including the CPU-only PyTorch variant, updating with `--reinstall --refresh`,
    and the `[thermo]` extra; Windows and macOS untested), `first-project.md` (opening a
    terminal in a folder, editing the project file in a plain text editor, writing Windows
    paths, and the commands in order), and `python.md` (the Python material moved out of the
    user pages, and how to enter review decisions, which has no command yet). Fixes found on
    the way: the build-system bound `hatchling>=1.25` let a resolver choose a hatchling that
    cannot read the `license-files` list (it happened with PyTorch's index first, which carries
    an old `packaging`): now `>=1.27`. brainpy's invalid escape sequences printed a
    SyntaxWarning on the first run under Python 3.12 and later: silenced around its import. A
    Windows path with backslashes between double quotes gave TOML's bare "Invalid hex value":
    the message now says to use forward slashes or single quotes.

111. **Bundles zipped by the operating system are read (owner, 2026-10-19).** hxex writes a
    zip bundle with `manifest.json` at the zip's top (`bundle = "results.zip"`); Windows'
    "Compressed (zipped) folder" and macOS "Compress" put the folder inside the zip, which
    hxex refused as not a bundle, so a user who zipped results to share them sent an
    unreadable file. `_zip_root` finds the bundle at the top or in the zip's one folder
    (ignoring `__MACOSX/` and `.DS_Store`); the zip store, `bundle_manifest`, and the
    replaceable-bundle check use it. A zip with several folders is still not a bundle. The
    guide's outputs page has a "Sharing results" section. Fixed after CI (2026-10-19): the test zipped its whole
    temporary folder into a zip inside that folder, so the zip read itself as it grew; on
    Windows (Python 3.12) it grew to the zip size limit and filled the runner's disk, and the
    later tests failed with "No space left on device". The two-folder zip is now made from a
    folder of its own and written outside it.

112. **Lists of tables forgive single brackets (owner, 2026-10-19).** A list of tables with one
    item written with single brackets, as most headings are (`[input.runs]`,
    `[analysis.regions]`, `[analysis.sites]`, `[run.registry.residues]`), is read as a list of
    one, as `[input.fragments]` already was (`config._items`). Before, `[input.runs]` said
    "give at least one [[input.runs]]", regions and sites gave a Python error, and the registry
    said "expected a list". A single-bracket heading written twice (TOML refuses it) now gets a
    hint: double brackets for the list tables (`config.LIST_TABLES`), one heading for any
    other. The guide's first project and the settings reference explain double brackets.

113. **The analysis folder treated like the tables and figures folders (owner, 2026-10-19).**
    `write_analysis` writes through `replace_outputs` (the `hxex_listing.json` of item 96): a
    second `hxex analyse` removes the files of the first (a figure it no longer draws goes),
    keeps the user's own files, and refuses one of theirs named like a new file. `_analysis`
    joins the bundle's sibling folders, so `--keep-previous` moves `results_analysis` to
    `results.N_analysis`; an analysis directory named by `[analysis] out` or `--out` is not
    moved. `hxex run` does not write the analysis: after `--overwrite` it warns that
    `results_analysis` still describes the replaced results. No other output folder lacked
    this: tables and figures (from `run`, `resume`, and `report`) already had it, and the
    multi-run default `runs_analysis` and explicit directories now get the listing too. As for
    the tables in item 96, an analysis folder written before the listing (nine in the private
    examples) is written again only if none of its files is named like a new one; move them,
    or write elsewhere, once.

114. **`peptide_range` optional and checked (owner, 2026-10-19).** The range label is used only
    as text (the `peptide_ranges` selection, figure titles and file names, table columns, the
    input's sort order, retention-time tables); residues always come from `start_seq` and
    `end_seq`, and nothing checked that the two agreed. In a peptide or file table it is now
    optional: left out or empty, it is `peptide_range_text(start, end)`, zero-padded as the
    folder formats' file names (`0001-0010`); one whose numbers differ from `start_seq` and
    `end_seq` (or that is not `START-END`) is a warning naming the line. No table in the test
    data or the private examples triggers it. Next: the `peptide_ranges` selection, which
    compares the label's text exactly ("1-10" does not select "0001-0010").

115. **Peptides selected by their residues; a region selection (owner, 2026-10-19).**
    `[input] peptide_ranges` compared the range label's text exactly ("1-10" did not select
    "0001-0010"); it now compares residues (`parse_range`, leading zeros ignored) with
    `start_seq` and `end_seq`, and an entry that is not `START-END` is refused when the project
    file is read. Owner's choice: an entry stays one exact peptide, and a new
    `[input] peptide_region = [first, last]` selects every peptide wholly within those residues;
    with both, a peptide that matches either is read. Folder formats only, as `peptide_ranges`
    (a scan format's peptide table says what to read). Recorded in the read record
    (`peptide_region`) so a repeated read selects the same; `find_series(peptide_ranges=)`
    compares residues too.

116. **Excluding spectra from the project file (owner, 2026-10-19; PLAN "Excluding spectra").**
    `hxex.exclusions`: rules (`Exclusion`: sample, peptide_range by residues, charge, time,
    role, rep, file, spectrum_id, note) in `[[input.exclude]]` and `[[analysis.exclude]]`; a
    spectrum matching every key of one rule is excluded, a rule must name a key, and one that
    matches nothing is a warning. Run level: `run(exclude=)` sets the matched spectra aside
    after the control roles are assigned (not picked, fitted, used as controls, noise
    references, or envelope references) but stores them with the raw spectra; the manifest's
    `excluded` records the rules, their counts, and the ids (`hxex info`; kept by `resume`); a
    rerun with `--spectra` and without the rule fits them. Analysis level:
    `exclude_from_results` adds rejections to an in-memory copy of the review layer (note
    "excluded by [[analysis.exclude]]: ..."), so every consumer of the review honours them:
    `hxex analyse` (each run of a RunSet), and the tables and figures of `hxex run` and of
    `hxex report`, which now also takes a project file (its `[output] bundle` and rules). The
    bundle is not changed. `hxex check` lists the rules (it reads no spectra, so it does not
    count them for the folder formats; the run logs the counts). The review layer stays; both
    apply. A `file` is the run file (scan formats), the HX-Express spreadsheet, or the
    SpecExport spectrum file. Found on the way: the uptake figure drew rejected spectra like
    any other (review rejections too); owner's choice: visible, but hollow and grey
    (`uptake_data` gives `rejected`; the legend says "rejected / excluded"), and the population
    figure labels the panel "(rejected)" and draws its overlay point hollow and grey.

117. **The controls always fitted at K = 1 (owner, 2026-10-19).** The back-exchange correction
    reads the controls' K = 1 fits, and `RunConfig` refused `[run.controls]` unless
    `fit.k_min = 1`. Owner: the controls are fitted from `k_min` to `k_max` like the data (which
    shows up problems with them), and at K = 1 as well, always, since that fit is the one the
    correction uses. `fit_models` now adds K = 1 for the measurements with a role when `k_min`
    is above 1 (the others get NaN for it, as for a K the screen skipped), and chooses every
    measurement's K within `k_min` to `k_max` only (an extra K = 1 is never chosen; with every
    K rejected, `k_min` as before). It is added whether or not the run corrects, so
    `[analysis.controls]` can correct later. The bootstrap's `ks = "all"` is the K chosen from,
    not the extra K = 1. `hxex info` says when the controls' K = 1 was added. The refusal is
    gone; the screen keeps its `k_min = 1` rule. Test: a `k_min = k_max = 2` run fits K = 1 for
    the controls only, chooses K = 2 everywhere, and gives the controls of a `k_min = 1` run.
    Checked on the owner's question: a time point promoted only afterwards
    (`[analysis.controls]` `undeut_at` or `fulldeut_at`, after a `k_min` above 1 run) has no
    K = 1 fit, and its missing mean slipped through `fit_controls` (an undeuterated one fell
    back without a flag; a fully deuterated one passed the retention limits, as every
    comparison with NaN is false, and made the series' corrected means NaN). `fit_controls`
    now refuses spectra serving as controls without a K = 1 fit, saying to promote them in
    `[run.controls]` or run with `k_min = 1`.

118. **A bootstrap added to, or redone on, a complete bundle (owner, 2026-10-19).** `hxex
    resume` finished only a bootstrap that a run had begun (the bundle's `pending`), so a run
    made without `[run.bootstrap]` could take one only by refitting from its stored spectra.
    `resume_bootstrap(add=True, bootstrap=...)` now bootstraps a complete bundle: one without
    a bootstrap gets one, one with a bootstrap has it replaced (the bundle's run settings
    record the new one; the fits, the review, the exclusions and the provenance are kept).
    `hxex resume PROJECT` does it with the project's `[run.bootstrap]` (refused without the
    table) on its `[output] bundle`, writes the tables (and figures) as `[output]` asks, with
    `[[analysis.exclude]]` applied. `hxex resume BUNDLE` of a complete bundle is refused,
    naming the project-file form.

119. **One analysis folder per reference (owner, 2026-10-19).** Asked whether several
    `[[analysis]]` tables (references) are supported: they were not, and the double-bracket
    form crashed with a Python error; and `hxex analyse --reference Mut` without `--out` wrote
    into the same `results_analysis` as the analysis with WT, replacing it. Owner's choice
    (of: a reference recorded in the folder's listing and a guard; dated or numbered kept
    analyses; the reference in the name): the default folder is named after the bundle and
    the reference, `analysis_folder_name` (`results_analysis_WT`, `results_analysis` without a
    reference, characters a folder name should not hold made `_`), so analyses against
    different references never meet; the same reference replaces its folder (the listing).
    `--keep-previous` moves every analysis folder of the bundle (`analysis_folders`) to its
    generation (`results.1_analysis_WT`); `run --overwrite` warns about each. `[[analysis]]`
    is refused, saying to run `hxex analyse --reference NAME`. An `[analysis] out` or `--out`
    is used as given. Existing `results_analysis` folders of analyses made with a reference
    stay where they are: a new analysis writes beside them, in the new name.

120. **`n_boot` default 200 to 20 (owner, 2026-10-19).** Owner: 200 is far too many for
    general use (20 is fine on its merits; pyHXExpress's procedure differed, so its value is no
    argument). Measured cost (items 32, 46): one resample refits at the selected K from the fit,
    with 3 starts against the fit's 6 over every K, so about 1/6 of the fit with K = 1 to 3; at
    20 the bootstrap takes about 3 times the fit (7 times with `ks = [1, 2]`; the benchmark's
    ratio was 8.5 with its K = 3 extras). Twenty give usable standard errors; the 95 % percentile
    intervals at 20 are near the range of the draws, so the comments and the guide say to raise
    it when the intervals matter. Of the private project files only one has a bootstrap, and it
    sets its own `n_boot`.

121. **The review shown to users: `review.csv` and `review_K` (owner, 2026-10-19).** Owner:
    users cannot read Parquet, and the guide described the review layer as if they could see
    it. Only `review_status` (spectra.csv), the figures' "rejected", and a count in `hxex info`
    reached them; notes, chosen K, dates, and stale entries were only in the bundle.
    `write_tables` now writes `review.csv` when the review layer (stored, or with
    `[[analysis.exclude]]` rules applied) has entries: one row per entry with the spectrum's
    sample, peptide, charge, time, role, replicate, the status, `review_K`, note, date, stale,
    and `source` (`review` or `exclusion rule`, from the rules' note). Without entries the
    file is not written (and a listed old one goes). The index columns gain `review_K` (the
    K the review chose; the `K` column already used it), a whole number or empty. The guide's
    "Reviewing results" is rewritten around what a user can do. Choosing K without Python (an
    `[[analysis.choose_k]]` rule) is the next item if the owner wants it.

122. **The settings used, written out (owner, 2026-10-20).** Owner asked whether the settings
    left at their defaults are saved with a bundle. They were, for the run: the manifest's
    `configs.run` is the whole `RunConfig` and `spectra_source` the reader's full settings;
    but not the `[input]`/`[output]` tables as given, nor any `[analysis]` setting (an analysis
    folder held only results), and only as JSON for the program. Owner chose TOML (a format
    users already write and can reuse) over JSON, the names `run_settings_used.toml` (in the
    bundle) and `analysis_settings_used.toml` (in each analysis folder), and, for analyses of
    several runs, a copy of each run's file (`run_settings_used_RUN.toml`) beside the
    analysis's own, so every file covers one thing. Built: the manifest gains
    `configs.project_file` (`config.project_record`: name, threads, `[input]` with every
    setting, `[output]`, with the command line's `--bundle`, `--overwrite`, `--keep-previous`
    applied; paths as `[input] record_paths` records them: `name` and `relative` leave no
    directory); `write_bundle` writes `run_settings_used.toml` from the manifest
    (`config.run_settings_text`), so it cannot disagree with it, and owns the name like
    `manifest.json` (not in `files`, not checksummed, never read back); a resumed bundle
    keeps `project_file`. `write_analysis` writes `analysis_settings_used.toml`
    (`config.analysis_settings_text`) and the run file(s) from the results' manifests, so old
    bundles' analyses get one too, and the analysis folder keeps the settings of the results
    it was made from after `run --overwrite` (which leaves the analysis folders in place).
    Written as a project file: a None whose default is not None as `false`, a stage that was
    off as `name = false`, a None default as the template's commented line; a test reads each
    template, a project with every kind of table, and an analysis back as the same settings.
    `[analysis]` paths follow the project's `record_paths` too. The guide's outputs page has
    the files in its folder tree and a section on them. Then (owner: "add the check") an
    analysis folder's listing records the results it read (`made_from`: run -> the bundle's
    `created` stamp; in the listing, not parsed from the settings file's comments);
    `hxex info` lists each analysis folder of the bundle as up to date, made from earlier
    results (with their stamp), or not known (older folders); `hxex analyse` warns about the
    bundle's other analysis folders made from earlier results. The stamp has one-second
    resolution: two bundles written in the same second would look alike, which a run cannot
    do. Analyses of several runs (`runs_analysis...`) are not checked: they belong to no one
    bundle.

123. **Figures out of the run; `[figures]` and a fixed K for drawing (owner, 2026-10-20).**
    Asked how a user would choose K without Python, the owner first asked what exists:
    verified that the run chooses K spectrum by spectrum (criterion over k_min to k_max, no
    peptide- or residue-level K) and that the analysis ignores that choice (fixed K = 1 and
    K = 2 fits, one-or-two chosen per residue and time, which in practice needs the fixed-K
    bootstrap for the K = 2 errors); and that the plot functions already took `k=`, from
    Python only. The plot settings were `[run.plots]`, honoured only by `hxex run` (and
    `resume`): `hxex report hxex.toml --figures` read the bundle's recorded plot settings, so an
    edited `[run.plots]` did nothing until a full refit. Owner: pull figure generation out of
    the run, for clarity (`hxex run` never draws figures; one more command rather than two ways
    in). Built: top-level `[figures]` (`PlotConfig`; `Project.figures`), read by `hxex report`
    given the project file (given a bundle: defaults and options); `RunConfig.plots`,
    `run(figures_dir=, reference=)`, `resume_bootstrap(figures_dir=, reference=)`,
    `[output] figures` and `hxex resume --figures` removed; `[run.plots]` and `[output] figures
    = true`/a folder refused with where things went; `[output] figures = false` (written by
    every template until now) accepted with an info line; old bundles' `configs.run.plots`
    ignored when read. `PlotConfig.k` / `hxex report --k`: every series drawn at one K (a K
    the run did not fit is refused; a spectrum the screen skipped is left out of the
    populations and uptake pages and labelled on the spectra page); tables and the analysis are
    unchanged. `RunResult.write_figures` stays for Python. The guide's "Running it all again"
    is three commands now (run, report, analyse). Choosing K for the run's tables or for the
    analysis's residue-level choice was discussed and not built (owner to decide after trying
    `k`). No private project file used `[run.plots]` or asked the run for figures (checked).

124. **The user review table: K and rejections from a spreadsheet (owner, 2026-10-20).** Owner:
    users work in Excel, so choosing K should be a column they fill in, per spectrum, in a file
    named `user_review_table.csv`, documenting which columns not to touch. Concern raised by
    the owner: a K forced without a bootstrap at it has errors not comparable with the other
    spectra's (without `ks`, K = 2 errors exist only where K = 2 was selected; the rest are
    replicate scatter or none). Owner asked for an easy "fit the missing bootstraps". Built:
    `hxex.review_table` reads a copy of `spectra.csv` (BOM, `;` or tab separators, capitals,
    `2.0`, and `unreviewed` = no decision accepted; a bad value names its row; other columns
    ignored), named in `[review] table`, and lays it over the review layer in memory
    before the exclusion rules (`apply_view`; `hxex run` tables, `report`,
    `resume` tables, `analyse`). Its entries replace the stored review's for the same spectrum;
    `review.csv` says `review table` (an in-memory `ReviewTable.source`, not stored). A chosen
    K the run did not fit for the spectrum is refused. Errors: when the results have a
    bootstrap, `hxex report` refuses a chosen K without one at it, naming `hxex resume`;
    `resume_bootstrap(chosen_k=)` (and `run(chosen_k=)`, with the carried review's choices)
    bootstraps exactly those spectra at their K (`_chosen_part`) and stores them as a partial
    fixed-K bootstrap (`bootstrap.scatter`: the other rows empty, `n_success` 0;
    `BootstrapResult.covers`; `Results.bootstrap_for` and `interval_table` fall back to the
    selected-K bootstrap for empty rows). The bootstrap does not use the noise references, so
    a subset is taken with `Measurements.select(drop_references=True)`; each spectrum is
    seeded from its own id, so the values equal a full fixed-K bootstrap's. `hxex resume
    hxex.toml` now keeps a bootstrap that has the project's settings (it used to redo it
    every time) and adds only what the chosen K lack; nothing to add is said, not an error.
    `hxex check` summarises the table and refuses one it cannot read. The analysis honours
    the table's rejections; a chosen K does not change it (it uses the fixed K = 1 and K = 2
    fits). Guide: "Reviewing results" rewritten around the table, with which columns to edit
    and which to leave; `hxex run`'s refusal and the overwrite description no longer mention
    figures. Then (owner: "top-level review will be more transparent") the after-fitting
    decisions got a top-level `[review]` table, as the figures got `[figures]`: `table` (the
    review table; any name, the guide suggests `user_review_table.csv`) and
    `[[review.exclude]]` (the rules, formerly `[[analysis.exclude]]`, still accepted with a
    warning, refused beside `[[review.exclude]]`). `Project.review`
    (`review_table.ReviewSettings`); `write_analysis(review=)`; the analysis record writes
    every rule under `[review]` (none under `[analysis]`). The template's `[review]` block
    replaces a `review_table` line that had sat under `[analysis.choice]` by mistake.

125. **`best_set_K`: one number of populations per peptide and time from all its spectra (owner,
    2026-10-20).** Discussed while settling how users choose K: the run chooses per spectrum
    (replicates can disagree), the analysis's species table reports K = 1 and K = 2 without
    choosing, and only the residues are chosen (per residue and time, from the bootstrap
    errors). Owner asked whether a peptide-level K could be supported statistically where
    spectra conflict; per time point only (owner: never one K for the whole time course, as
    populations appear and merge). Of two methods (the residue tests on combined values,
    needing the bootstrap; or pooled evidence) the owner agreed to pooled evidence plus the
    population check, and to the column names. Built (`analysis.species.best_set`,
    `_set_evidence`): the fits' raw criterion summed per K over the set's spectra (those the
    tables use, fitted at every K analysed; not the selection score, whose infinity for a K
    that failed the run's population rule would veto the set); the smallest sum wins; a K the
    filters left out, or with a combined population outside `[analysis.choice]`
    `min_population`, is passed over for the next. Columns in `species_uptake.csv`:
    `best_set_K`, `best_set`, `best_set_reason`, `set_delta_criterion`. Nothing else uses it
    yet (the comparisons stay at K = 1; open, to decide on real data). docs/ANALYSIS.md has
    the design.

126. **Judging the controls without a fixed floor; the median as the default fallback (owner,
    2026-10-20).** Found while planning the tutorial (published B5 data, the 0.1 s and 0.8 s
    points excluded, 3,874 spectra, 85 % D2O): 19 of 195 fully deuterated controls were
    rejected (`retention_low`) at 0.34 to 0.42, just under the default floor (half of
    `d2o_fraction`, 0.425), and replaced by `d2o_fraction` (0.85), about 1.7 times the
    accepted median (0.51). They were real: the same peptide and charge agree across the three
    samples to 0.015 in retention (median; 0.044 at most), while peptides range from 0.34 to
    0.64, so retention belongs to the peptide (sequence, time on the column) and to the
    workflow, not to noise; no fully deuterated control was below its 20 h point by more than
    0.37 D (median 0.63 D above). Owner: the fallback default is the median retention of the
    run's accepted controls (`fd_fallback` unset: `kint` with quench conditions, else
    `median`; before, `d2o`), and the floor should change; on my recommendation, built:
    without a set `retention_min` there is no fixed floor. A fully deuterated control is
    rejected when its retention is 0 or less or above `retention_max`, when it is below the
    highest time point of its series by more than `time_course_tolerance` (0.1 D per amide;
    `fd_below_data`, the under-deuteration check, which needs no assumption and holds for
    either labelling direction), or when it is more than `outlier_sd` (4) robust standard
    deviations below the median of the run's controls (`retention_outlier`; from at least
    five controls; with `kint` only for the series without rates, or for all when the model
    could not be fitted, replacing the old `d2o_fraction / 2` floor there). Each time point
    counts by the median of its replicates (one bad replicate does not decide). An
    undeuterated control above the lowest time point is only flagged (`data_below_m0`): owner
    pointed out that the fastest time points are the hardest to measure, so when they
    disagree the time point is suspect, not the control, which `m0_max` already judges on an
    absolute scale (a first version rejected the control, `m0_above_data`). Flags that only
    warn: `fd_disagrees` (retention more than `sample_tolerance`, 0.1, from the group median
    of the same peptide, modification and charge in the other samples or conditions; from
    the other one in a pair; `series_groups`) and `fd_multimodal` / `m0_multimodal` (most of
    the control's replicates chose K above 1), and `data_below_m0`. The run's limit is stored with the controls
    (`ControlTable.limits`, JSON metadata `limits`; old bundles have none and the overview
    then draws their old floor). `series.csv` gains `control_notes`; the overview shows only
    the reasons that occurred and the warnings in grey. On the tutorial data: all 194
    measured controls accepted, limit 0.25 (median 0.50, robust sd 0.065), one series without
    a fully deuterated control at the median; no `fd_disagrees`; `fd_multimodal` and
    `m0_multimodal` on 13 and 14 series (the broad-envelope peptides, where every replicate
    chooses two populations; with "any replicate" it was 42 series, too many to be useful).
    Noise estimated from the replicates (`estimate = "replicates"`) was also tried there: it
    doubled the spectra choosing K = 2, controls included, so the tutorial keeps the default
    noise. Open (PLAN Next steps 8): whether retention also depends on the charge state.

127. **The tutorial: a guide page, a notebook, and a download (owner, 2026-10-20).** Owner's
    plan: a Markdown walk-through for people who do not use Jupyter and a notebook of the same
    steps, starting from the published B5 SpecExport data (the tutorial dataset), the
    quench-flow points (0.1 s, 0.8 s) left out as a tutorial step (`[[input.exclude]]`). The
    download is the zipped SpecExport folders (39 MB; a bundle's `spectra.parquet` is 49 MB
    and tied to hxex's layout), with the notebook, a README (citation: Woods et al., PNAS
    2025, e2425061122; the paper's labelling conditions) and the licence (CC BY 4.0 for the
    data, owner); hosted as an asset of a GitHub release `tutorial-data-N` (owner), which
    `publish.yml` now leaves alone (PyPI only for `v*` tags). Jupyter comes with the optional
    `notebook` extra (`notebook>=7.2`, BSD; the lowest installs on 3.11); `uv tool install
    "hxex[notebook] @ ..." --with-executables-from notebook,jupyter-core` puts `jupyter` on
    the path. `docs/guide/tutorial.md`: init, the edits, check, run (no bootstrap: 1 min 14 s
    here, about two on a laptop), the tables, the figures of three WT peptides (87-96 one
    population, 42-54 two at 4 s and 1 min, 68-75 a weak 30 min spectrum), the review table
    (K = 1 for two 30 min spectra of 42-54, one rejection), the bootstrap added by `hxex
    resume` (`ks = [1, 2]` for the analysis: 12 min 53 s here; one full run with it took
    14 min, too long for step 6), and `hxex analyse` (R120G takes up more deuterium than WT
    over residues 1 to 80, most at early times, as in the paper). Not used: a peptide subset
    (the full dataset is fast enough), `screen` (its skipped fits break `--k 2` and
    `best_set_K`). `docs/tutorial/make_tutorial.py` builds the zip and, with `--run`,
    executes the notebook on the data and draws the page's figures from its results
    (`docs/images/tutorial_*.png`), so page, notebook and figures come from one run;
    `tests/test_tutorial.py` checks that the page and the notebook give the same commands and
    settings and that the zip holds what the page lists (working through it: opt-in,
    `HXEX_TUTORIAL_DATA`). The notebook is committed without outputs. New for it:
    `hxex.load_results(path)` (a project's results with its `[review]`, as `hxex report` sees
    them) and `hxex.show_series(results, sample, residues, charge, k=, kinds=)` (shown as PNG,
    because hxex draws without pyplot and no inline backend is set). Found on the way:
    `spectra.csv` had no `note` column although the guide told users to fill it in a copy
    (added, with the review's notes); the control limits (item 126).

128. **Populations compared by the lever rule; charge states averaged within replicates (owner,
    2026-10-20).** Owner, on the tutorial's Woods plot: K=1 is significant over residues 1 to
    100 for R120G against WT, but no K=2 panel is, while a shift in the average has to show in
    the K=2 positions or the population ("you may lose significance in fD or population but
    not both"). Two causes. Each K=2 number was tested on its own although the three trade off
    within a spectrum (about 15 % of K=2 comparisons reached p < 0.05, 69 % at K=1); and the
    pooled thresholds were large (population 0.49, low fD 0.22, high fD 0.20, against 0.036 at
    K=1), dominated by poorly separated descriptions: two thirds to three quarters of the K=2
    differences with p < 0.05 failed them. Pooling only where `best_set_K` = 2 lowered them
    only somewhat (population 0.43 -> 0.35). Owner proposed the residue level's rule at the
    peptide level: K=1 as the proxy of the average deuteration (valid while the K=1 fit spans
    the envelope). Measured (scratch, tutorial data, D109H and R120G against WT): the lever rule
    with each spectrum's own K=2 positions barely helps (replicate sd 0.214 -> 0.210, 9 -> 12
    significant: the scatter is in the positions); with the positions averaged over the
    sample's replicates the population is as consistent as K=1 (p < 0.05 for 175 of 321 pairs)
    but the pooled threshold (0.40) still blocks most (29 significant); with the K=1 threshold
    divided by the separation, 151. Built (owner: replace the fitted share; keep the samples'
    positions separate, "part of what we are looking for is whether underlying states are the
    same with differently sampled populations, or whether the underlying states themselves
    appear to be different"): `compare_peptides(quantity='population')` as in docs/ANALYSIS.md,
    with the choice's tests (`reason` column) and per-row thresholds (`threshold`, written as
    `delta_crit`); `compare_populations` per residue with the same threshold rule; the Woods
    plot draws a population's thresholds as a short dashed line over each peptide and counts
    only the compared pairs. Then (owner, asking whether the comparison pools charge states):
    the charge states of one replicate are averaged before the replicates are combined
    (`species._by_replicate`), in the species table (so the residue map, the choice and the
    residue comparisons too) and in every peptide comparison and its pooled scatter; pooling
    them had counted one sample's charge states as independent replicates. In the tutorial
    data only peptide 1-23 has several charges (n 12/9 -> 4/3). Outputs compared (kept outside
    the repository; the tutorial's bundle analysed three ways): fitted share 9
    significant of 363; lever rule 121 of 245 compared; with charges averaged 120 (R120G
    1-23 at 4 s lost significance, p 0.003 -> 0.074), residue level 6 -> 127; owner: "analysis
    3 looks good, let's make it the standard". The residue comparison plot (`compare_populations_*`)
    now looks like `compare_residues_*` (points with capped error bars, the legend above) and
    draws no significance bands (owner: they reused the samples' colours and confused; the
    tables hold the significance). Open: the population test treats each sample's averaged
    positions as exact; only 25 of the 121 also pass |delta| / delta_se > 1.96 with their error
    included (owner to revisit; a joint bootstrap of replicates and positions is the other
    option).

129. **A missing fully deuterated control takes the other charge states (owner, 2026-10-21).**
    Owner asked whether the controls are per peptide and charge (they are: one series is one
    peptide at one charge in one sample) and whether the tutorial's one series without a fully
    deuterated control has other charge states. It has: R120G 1-23 at 3+ has no `Full` file,
    while its 4+ and 5+ have one each (retention 0.533 and 0.533, single replicates); it fell
    back to the run median, 0.504. The measured charge states of that peptide agree within
    each sample (WT 0.523/0.517/0.521, D109H 0.509/0.511/0.508). Owner: "let's try
    fd_other_charge". Built: `[run.controls] fd_other_charge` (default true; also in
    `[analysis.controls]`): a series whose fully deuterated control is missing or rejected
    takes the mean retention of the same peptide's accepted measured controls at its other
    charges in the same sample, with its own m0 (flag `fd_other_charge`, replacing
    `fd_median` or `fd_kint`; `fallback` stays true; `mfd_se` from the donors' standard
    errors, NaN when they are all single replicates). It runs after the median or kint
    fill-in, so the kint fit and the run's limits never see the borrowed values. The peptide
    is (sample, sequence, modification without `undeut_mz`); fragment ions are left out
    (their ion type, or `"fragment"` in their scan target: gentle and harsh conditions are
    never pooled). On the tutorial bundle (recomputed in memory) R120G 1-23 3+ is the only
    series that changes: retention 0.504 -> 0.533, its uptake about 6 % lower, in line with
    its 4+ and 5+. Bundles keep their stored controls until rerun (or recomputed by an
    `[analysis.controls]` table).

130. **Charge state does not change the deuterium level; the private fragment dataset has no
    fully deuterated control (owner, 2026-10-21).** Owner asked whether the charge state affects
    back-exchange, using the private fragment dataset (infusions of deuterated peptide diluted
    into H2O, scan time = exchange time), and what deuteration level its files named "full"
    have. Every file has MS1 full scans, so all charge states of a parent are recorded in the
    same scans: a difference between them could only arise in the spray or the gas phase.
    Measured without a fit or controls (scratch script, kept in the private examples): per MS1
    scan and charge, the mean mass shift of the intensity at the isotope-peak positions over
    the envelope's contiguous extent, minus the theoretical shift of the undeuterated ion,
    / 1.00628. Paired per-scan medians from 20 s on, against each peptide's strongest charge
    state: 8-mer 2+ - 3+ +0.024 D (levels 0.8 to 3.4 D), 11-mer 2+ - 3+ +0.032 (0.6 to 6.7),
    12-mer 3+ and 5+ - 4+ +0.03 to +0.05 (0.6 to 3.3, two replicates), 24-mer 3+, 4+, 6+, 7+ -
    5+ within +-0.03; the end-state runs agree within 0.04 D (the 14-mer is almost all 2+).
    At most a few hundredths of a deuterium: no charge dependence that matters. In an LC
    experiment the charge states form after the column, so this supports `fd_other_charge`
    (item 129; owner: "evidence supporting" the other-charge fallback). Owner then agreed that
    such a series counts as measured for the back-exchange start of the protection fit. Built:
    `ControlTable.measured` (own controls measured, or the fully deuterated one from the other
    charge states with a measured undeuterated one); species column `fd_other_charge` (every
    fallback behind the species is that one); `_start` fixes fD0 = 1 when the peptide's
    fallbacks are all of that kind. `control_fallback` stays true for them in every table.
    A first attempt with a wide centroid window showed 0.4 to 0.6 D offsets between charges:
    unrelated peaks up to ~37 Da above the monoisotopic peak, an artefact; the stick-based
    measure removes it (envelope overlays confirmed it). The two "full" files are
    undeuterated (24-mer 3+ to 6+ -0.04 D, 14-mer 2+ -0.01 D; envelopes match the
    natural-abundance pattern): protiated MS1 references, never deuterated (owner), not
    fully deuterated controls. The end-state runs are near 0 D, so an effect proportional to
    the deuterium present would hardly show there; the kinetics runs (up to 3 to 7 D) bound
    it at about 1.5 % of the deuterium for the 8- and 11-mers, a few % for the 12-mer. The
    end-state runs keep 0.13 to 0.35 D, about 1 to 1.5 % of the labile hydrogens (a rough
    estimate of the D fraction left after the dilution). So this kind of data has no fully
    deuterated control and no LC back-exchange to correct; the owner suggested analysing the
    fragments relative to the parent's deuterium content instead (open; ties to the PLAN risk
    "Back-exchange correction of fragment ions"). Owner: test as close to full deuteration as the data allow,
    after checking where each kinetics run becomes reliable. Start of a run = the first scan
    whose strongest charge state has at least 25 % of its run-median signal and whose D does
    not rise over the next three scans (the first scans are empty or carry ~0.3 D carryover;
    in one 12-mer run D stays flat for ~20 s while the sample is still mixing). Over the 60 s
    from the start (up to 86 % of the amides deuterated, the 11-mer): 8-mer 2+ - 3+ +0.034 D
    (+1.0 %), 11-mer +0.091 D (+1.5 %, larger than over the whole run, so it scales with D),
    24-mer within about +-1 % (per-scan scatter +-0.2 D). The 12-mer's 4+ is 3.5 to 4.5 %
    below both its 3+ and 5+, which agree with each other, in both replicates and independent
    of the 4+ intensity: peaks at half-integer positions under the 4+ envelope (2 to 3 % of its
    signal) point to an interfering species, not a charge effect. So charge changes the
    deuterium level by at most about 1.5 % of the deuterium present.

131. **The kint fallback's 10 controls, the median below them, and uncorrected fragments
    (owner, 2026-10-21).** Owner asked how `kint_min_series` (10) was chosen: it had no
    measurement behind it (set with item 73; the code refuses fewer than 3). Measured on the
    B5 example (194 accepted fully deuterated controls, 149 with retention times; scratch
    scripts with the private examples): 400 random subsets of n controls, the model fitted as
    the code does (fit, reject beyond `kint_tolerance`, refit) and judged on the held-out
    controls (rms retention error per amide), against the subset's median retention.
    Clean subsets: kint 0.060 vs median 0.068 at n = 3 (kint better in 80 % of subsets), 0.048
    vs 0.065 at 8 and 10 (96 %), 100 % from 15 (full model 0.045); with retention times the
    same. With one control per subset reading low (retention x 0.6): kint better in 44 % at 3,
    67 % at 5, 87 % at 8, 93 % at 10, 98 % at 15; the rejection catches the bad one 28 %, 61 %,
    74 %, 80 %, 88 % of the time. The fitted scatter (the stand-ins' `mfd_se`) is too small at
    small n (median 0.035 at 3, 5th percentile 0.004; 0.044 at 10). So 10 stands, now with a
    reason: one bad control is the risk, and from 10 the prediction beats the median in more
    than nine cases out of ten. The real defect was below it: the series kint could not fill
    took `d2o_fraction` (rms error 0.35 per amide here, five to seven times the median's).
    Owner: fall back to the median retention of the accepted controls. Built: in kint mode the
    series without rates (extra `Hex` sites), and every series when the model is not fitted,
    take `fd_median` (`d2o_fraction` only when no control is accepted, as in median mode;
    `fd_fallback = "d2o"` unchanged). The rated controls the run's limit rejects when no
    model is fitted now take it too (their rejection was not passed on before, so they kept
    `d2o_fraction`).
    Owner also asked whether fragments get any correction ("I think we decided that none of
    the back-exchange corrections apply"): the record had only the open point of item 79
    (the correction may belong at the parent level); the code corrected a fragment series by
    its own controls or a fallback. Owner: fragments uncorrected for now. Built: the fragment
    series (ion type other than `M`, or a fragment scan target, `controls.fragment_ids`) get a
    control row flagged `uncorrected` (measured means kept, `m0` 0, `mfd` `n_amides`,
    retention NaN, no fallback), take no part in the run's limits, median, prediction or
    `fd_disagrees`; `correct_means` passes their values through; their species rows say
    `corrected = false`; the components table leaves their corrected columns empty; series
    figures draw them uncorrected (a forced `corrected = true` skips them instead of failing).
    Bundles keep their stored controls until rerun or recomputed. The guide's controls page
    now lists the order of the values as numbered steps.

132. **A modified peptide in a project with fragment ions stays refused (owner, 2026-10-21).**
    Since b329373 a peptide-table row with a `modification` is refused in any project with
    `[[input.fragments]]` (one row is both the peptide and the fragments' parent), and `hxex
    check` lists every such row. The alternative, reading the peptide at MS1 and skipping only
    its fragments, was left open; owner: keep the refusal, since its message says how to work
    around it (modified residues as modX labels in the sequence, or two projects: one without
    `[[input.fragments]]` for the MS1 spectra, one with the unmodified peptide for the
    fragments; guide inputs.md says the same).

133. **`hxex check` counts what each exclusion rule matches (owner, 2026-10-21).** Item 116
    left the counts to the run (`hxex check` read no spectra), so a mistyped rule (a range or
    sample written differently from the data) was found only after the fit. Owner: count them
    in the check, and a rule that matches nothing fails it. Built without reading any spectrum:
    `io.spectrum_index` gives the spectra a folder read would make (SpecExport and pooled from
    the file names, `specexport_points`, now shared with `read_specexport_dir`; HX-Express
    from each file's first row, `read_hxexpress_header` and `hxexpress_columns`, shared with
    `read_hxexpress_file`), and `io.mzml.scan_index_rows` those of a scan file from its scan
    table (`_plans`, the planning step of `_read_run`, so kinetics grouping, `min_time`,
    `usable_from` and `max_spectra` are the read's). Tests compare both with the reads (ids,
    times, roles, replicates, labels). `hxex check` applies the `undeut_at`/`fulldeut_at`
    roles first, as the run does, then prints one `matched` line per rule (`N of M spectra`;
    `[[review.exclude]]` rules among the spectra the input rules leave) and makes a rule that
    matches nothing a problem, naming each value no spectrum has with the values present
    (`exclusions.why_nothing`), or saying that only the combination matches nothing. A
    project reading stored spectra is not counted. The published B5 folder (5,261 spectra):
    2.6 s.

134. **Every output folder follows the bundle's rule (owner, 2026-10-21).** Owner, rerunning the
    tutorial notebook, was confused by results figures and tables folders left from earlier
    results. Found: `hxex run` refused only an existing bundle; a bundle deleted by hand left
    its tables folder (silently rewritten), its figures and analysis folders (no warning:
    the stale warnings came only with `--overwrite`); `hxex report` and `hxex analyse` always
    replaced their folders, removing every figure they had drawn before (drawing three series
    after all of them left three), with no way to keep an earlier set; tables and figures
    folders did not record the bundle they came from. Owner's decisions: `report` and
    `analyse` consistent with `run`; one switch, `[output] overwrite` / `keep_previous`,
    covers every folder; a folder kept on its own goes to its own `NAME.N`
    (`results_figures.1`, not `results.1_figures`); the tables follow the same rule. Built:
    `io.prepare_output` (a folder whose listing names files is refused unless overwrite, or
    moved to `NAME.N` with keep_previous, N never reused) used by `hxex run` for its tables
    (before anything is fitted; a bundle kept as a generation still takes its tables along),
    `hxex report` (`--overwrite`, `--keep-previous`; tables and figures) and `hxex analyse`
    (the same options); `hxex resume` rewrites the tables of the bundle it adds to without
    asking. The tables and figures listings record `made_from` (the bundle's `created`
    stamp), so `hxex info` lists `outputs` (tables, figures) as up to date or not, and after
    `hxex run` and `hxex resume` every figures or analysis folder made from other results is
    a warning naming the command (`report --figures --overwrite`, `analyse --overwrite`).
    Guide: outputs.md "Running again" (the three ways, the table of what each command
    writes and makes out of date, deleting only `results` is not a fresh start), first-project,
    project-file, tutorial (`--overwrite` on the second and third `report --figures`, and how to
    start the tutorial again); README; BUNDLE.md; template comments. The tutorial notebook
    changed (two commands): the download needs a new zip. Owner: replace the zip on
    tutorial-data-1 (not yet announced, so no link has to stay as it was); a new tag only once
    the tutorial is announced (docs/RELEASING.md, "The tutorial data").

135. **The Woods plot draws a population scaled by its separation (owner, 2026-10-21).** Owner:
    on the tutorial data's K=2 population Woods panel the dashed lines moved from peptide to
    peptide, unlike the fD panels'. By design (item 128): a population's threshold is the K=1
    `delta_crit` divided by the peptide's separation (tutorial: separations 0.12 to 0.94,
    median 0.40, so thresholds from about 0.04 to 0.3). The Woods plot is for judging
    significance by region, so owner chose (from an example, scratch) to draw the population
    difference times the separation, in fD units (the deuterium that moved between the
    states), against one dashed line, the K=1 `delta_crit`: the same test (|delta| >
    delta_crit / separation), the same peptides significant (R120G at 1 min: 18 of 38 in both),
    and poorly separated peptides no longer look large. Built:
    `PeptideComparison.separation` (population rows; NaN for fD), a `separation` column in
    `compare_peptides.csv` (which keeps the unscaled `delta`), `scaled_by_separation()` (delta
    and delta_se times the separation, threshold delta_crit, `scaled`), used for the Woods
    stack (title "... population × separation"); the scaled panel shares the fD panels' y
    range and is labelled "Δpopulation × separation". Guide outputs.md "Reading the Woods
    plot" describes the method; tutorial text updated. The per-residue population figure is
    unchanged. `docs/images/tutorial_woods.png` needs redrawing (`make_tutorial.py --run`).

136. **`hxex check` lists the output folders as information (owner, 2026-10-21).** In the
    tutorial's step 9 (`check` after the run) the existing bundle was "1 problem(s) found",
    though it only concerns a new `hxex run`. Owner chose (over a `check --for report` flag) an
    `outputs` block of plain lines: the bundle, tables, figures, and analysis folders that
    exist, whether each was made from the results in the bundle now (or is left from a bundle
    that is gone), and which command would need `--overwrite` or `--keep-previous` (or what
    `[output] overwrite` / `keep_previous` will do). Not problems: each command refuses by
    itself before doing any work, and the check ends with `ok` unless something else is wrong.

137. **K=2 comparisons only where both samples have two populations; the number of populations
    compared; the Woods plot in two figures; the residue figure restyled (owner, 2026-10-21).**
    Owner asked how to read regions with significant and non-significant peptides, and why
    fDlow/fDhigh looked different where K=1 did not. A Fable-model agent found (tutorial data):
    the K=2 rank and population comparisons paired one sample's second state with the other's
    small tail of a single population (D109H 113-134: WT two states, D109H one plus a tail;
    FDQFFGEHLLE 24-34: the rank-2 difference opposite to K=1); 36 of the 43 significant rank
    rows had `best_set_K = 1` on a side; the per-residue choice (confidence 0.5) kept two
    populations at 88 % of residues. A scratch test of gating on `best_set_K = 2` (peptide
    level) and a weighted vote (residue level; "any" 1,361, "half" 436, "all" 61 of 1,872
    residue-sample-times) led to the owner's choices: (1) peptide K=2 comparisons (positions
    and population) only where both sides have `best_set_K = 2`, else NaN with `reason`
    `one population (sample|reference|both)` (`compare_peptides(best_sets=)`,
    `analysis.best_sets`); (2) residue K=2 values from the two-population peptides only, every
    covering peptide voting with the residue weights, two populations only where they carry at
    least half (`residue_map(two_population_peptides=True)`, `two_population_weight`,
    `choose_populations` reason `one-population peptides`), and residue K=2 differences
    significant only where both sides keep two; (3) where the number of populations differs is
    shown, not hidden: `compare_population_counts` (`compare_population_counts.csv`;
    `borderline` unless the two-population side's summed criterion is at least 10 below 0 and
    the other's above 0); (4) the Woods plot as two figures: `woods_<sample>_<time>` (K=1 fD
    and "Two Populations Selected": sample only, reference only, both; grey bars, outlined when
    borderline, stacked in lanes, a "K=2" column header) and `woods_K2_<sample>_<time>` (the
    gated K=2 panels, short titles, "Δpop × sep", one note line); (5) the residue figure: K=1
    as small circles on a line with a ±1 se band, joined only between residues one peptide
    measures together (`ResidueMap.linked_next`; prolines bridged inside a peptide), fDlow and
    fDhigh as up and down triangles, fills and K=2 error bars 65 % opaque with solid outlines,
    the legend naming the samples only, `overlay` default now includes `avg`. Also: dashed
    thresholds and zero lines drawn above the grid (a threshold on a gridline looked faint), the
    Woods zero line grey. Tutorial data: rank comparisons 363 -> 92 (significant 23 -> 3 and
    20 -> 4), population 245 -> 85 (120 -> 36); 122 peptide-times differ in the number of
    populations, 31 borderline. Tutorial page, notebook (one cell: the second Woods figure) and
    images updated; the download zip changed again (replace it on tutorial-data-1). Then
    (owner): the population figure (`compare_populations_<time>`) had its own plotter and kept
    the old look; it is now the residue figure with the overlay values `pop_low` and
    `pop_high` (the populations from the choice, each sample where it keeps two, not requiring
    both; up and down triangles by rank, the high-D one the complement), sharing the markers,
    fills, legend of sample colours, axes and time label (`_rank_points`, `_residue_axes`,
    also used by `plot_residue_comparison`); the corrected-differently mark stays on by
    default, `mark_corrected_differently=False` in Python turns it off. Open: the
    K=2 ranks' `delta_crit` is pooled over every K=2 description; the threshold of the
    borderline evidence (10) is fixed, not a setting.

138. **A multimodal fully deuterated control says whether its retention agrees (owner, 2026-10-22).**
    On the intact-peptide benchmark one sample's fully deuterated controls are called two
    populations in 36 % of spectra (the others 4-14 %), yet their K=1 retention matches the most
    similar sample's within 0.005 (r 0.90), also for the bimodal series: the molecules kept
    different amounts of deuterium around the same average, and the correction, which uses the
    average, is sound. The guide's wording ("can mean an incompletely deuterated or a mixed
    sample") suggested the control was unusable. Built: `_flag_disagreement` adds the warning
    token `fd_retention_agrees` to an `fd_multimodal` fully deuterated control whose retention agrees
    with the same peptide's in the other samples (`CONTROL_NOTES`); `series.csv`
    `control_notes` writes `fd_multimodal (retention agrees)`, `(retention differs)` (with
    `fd_disagrees`) or `(retention unchecked)` (nothing to compare with). Guide reworded. Not built:
    comparing with the same peptide's other charge states when a run has one sample (owner's
    point 2, to design), and marking these on the overview figure. Bundles show the new note once
    their controls are recomputed or the run is repeated.

139. **`[analysis] populations`: best_set_K or the bootstrap rule, and what decides it (owner,
    2026-10-22).** The owner is not settled on how one or two populations are chosen, so both
    stay available: `populations = "best_set"` (the default: the HISTORY 137 gate, K=2 values
    and comparisons where the set is best described by two populations, the half-weight vote
    per residue) or `"bootstrap"` (the per-residue choice of `[analysis.choice]` alone, every
    peptide's K=2 fit used). Evidence gathered on the intact-peptide benchmark (rerun with the
    current hxex; private notes): (1) calibration against known answers (simulated spectra
    with hxex's noise, the pyHXExpress bimodal test set): the confidence-0.5 residue rule calls
    two populations in 44-91 % of true single populations, best_set_K in 0-2 % (real
    undeuterated controls 4-7 %), a parametric-bootstrap likelihood-ratio test about 1 %; small
    or close second populations are missed by every rule (a 10 % population 0.3 fD away found
    21-33 % of the time with three replicates); best_set_K's power does not grow with
    replicates (one penalty per spectrum), the test's does. (2) Envelope width against one
    binomial, independent of the fit: wider at 4 s and 1 min in the N-terminal region, not in
    the central or C-terminal regions, not at late times; agrees with best_set_K at 91 % of
    residues and with the bootstrap rule (confidence 0.95, 10 %) at 27 %; blind to close
    components (each narrower than a binomial). (3) Why the bootstrap rule and the owner's
    earlier notebook differ: the same tests, the same positions (within 0.04 fD); the
    notebook's residue errors were in deuterons where its values were fractions (larger by the
    peptide's amide count, so its z = 1 test was strict for long peptides), and the errors of a
    forced split shrink with averaging (replicates, charges, overlapping peptides) because the
    split lands reproducibly: overlapping peptides agree on the forced K=2 positions as well
    where the set says one population as where it says two (leave-one-out residuals 0.06 and
    0.05 fD), so precision and consistency do not show a second state. (4) `nterm_subtract` 1
    against 2 changes 2.2 % of spectra's K (mostly through the K=1 n cap); the residue weighting
    (exponent 0, 1, 2) moves the residue-level decision within the regions best_set_K marks,
    not between regions. (5) In best_set mode the K=2 curves came from the two-population
    peptides and K=1 from all, so 357 residue-times showed rank curves that do not bracket K=1:
    a construction artefact (owner's observation); not yet changed (proposed: K=2 curves from
    every peptide, the gate deciding only which residues are labelled two). Proposed, not
    built: a three-level per-peptide verdict from the calibrated test's p-value (two,
    undecided, no sign of two), carried to residues by the vote.

140. **Controls combined over charge states; spectra left out whole; stand-in controls out of
    the analysis; a robust `delta_crit` (owner, 2026-10-22).** On the intact-peptide benchmark
    the hybrid rule's `delta_crit` (0.136 fD) was set by a few series whose replicates read a
    different envelope (a 5-amide peptide at 5.5-5.7 D in two of four replicates beside 1 D;
    the fit's n at its cap, fD near 2 after the correction): the top 1 % of series carried
    71-74 % of the pooled variance, the typical replicate sd was 0.017 (same at every exposure
    time; replicates are separately prepared samples), and series with a stand-in fully
    deuterated control carried 38 % (116 of the 133 series-times whose corrected fD is
    consistently above 1.1 used one). The owner's pyHXExpress practice (drop fD < 0.1 or > 1.1
    at the exposure times, per component too) was tried: the upper limit halves `delta_crit`,
    but of the values below 0.1, 176 of 187 agree with their replicates (real low uptake at
    4 s and 1 min), and of those above 1.1, 133 agree with their replicates (a stand-in control
    too low for the peptide); a floor on components removes protected low-D populations.
    Owner: an empirical rule that needs justifying. Built instead: (1) `[run.controls]
    combine_charges` (default true): every charge state of a peptide in one sample takes the
    plain mean retention of its accepted measured fully deuterated controls (charge states
    agree: median spread 0.015, 90 % within 0.05 on the benchmark; HISTORY 130); a charge state
    without its own counts as measured (`fd_other_charge`, not `fallback`, so never "corrected
    differently"); disagreeing charges (span above `sample_tolerance`) are combined and warned
    (`fd_charges_disagree`); the undeuterated control stays per charge (owner: FD only, plain
    mean); `fd_other_charge` keeps its old meaning only with `combine_charges = false`.
    (2) `[analysis.filters]` leaves out whole spectra (listed with the reason in
    `left_out_spectra.csv`; a spectrum accepted in the review is kept): `replicate_outlier = 6`
    (more than 6 typical replicate sd from the median of its series' replicates at that time,
    three or more replicates; the typical sd is the run's own, the median of the
    chi-square-scaled replicate variances; measured from the group median because the median
    of the others is pulled halfway to an outlier when there are three), `above_amides = true`
    (a K=1 mean above the amide count before correction), and `stand_in_controls = "exclude"`
    (series whose fully deuterated control is the median, kint or `d2o_fraction` stand-in,
    `ControlTable.fd_measured`; kept in a sample with no measured control at all). Charge states
    are not compared here (`charge_trend` does that: about half the spectra a cross-charge
    check flagged agreed with their own charge's replicates). (3) `delta_crit` from the median
    of the chi-square-scaled replicate variances instead of their plain pool (a deliberate
    difference from Hageman and Weis; PLAN). (4) `best_set_reason` says when the set favoured a
    K the filters had removed ("set evidence for two, but K = 2 not kept by the filters: one
    used"). The `min_fD` component floor tried during the session was removed. Benchmark, with
    the controls recomputed: 148 replicate outliers, 19 above the amide count, 2,742 spectra of
    191 stand-in series left out (no residue loses coverage); `delta_crit` 0.136 -> 0.027;
    significant one-population peptide differences 185 of 2,951 -> 795 of 2,467; residue
    differences 90 -> 746. Bundles keep their stored controls until rerun or until
    `[analysis.controls]` changes a setting (then they are recomputed).

141. **Residue K=2 values from every covering peptide in best_set_K mode (owner, 2026-10-22).**
    With the HISTORY 137 gate the residue K=2 values came from the two-population peptides and
    K=1 from all, so the rank curves of the choice figures did not bracket K=1 at 290 of 3,500
    residue-times on the intact-peptide benchmark (the owner's observation: "non-physical on
    aggregation"). Now every covering peptide's K=2 description enters the residue values, as
    in bootstrap mode, and the two-population peptides' vote only decides which residues have
    two populations (owner's option 1: one set of K=2 numbers per residue, not the
    two-population peptides' values at the residues decided as two). Benchmark: K=1 outside
    the ranks 290 -> 48 residue-times; residue-times with two populations unchanged (465);
    where both constructions decide two, the positions differed by a median of 0.02 fD. A
    residue covered only by one-population peptides now has K=2 values and the reason
    `one-population peptides` instead of `no K=2`. Peptide comparisons stay gated.

142. **`[analysis.choice]` defaults follow the populations rule (owner, 2026-10-22).** Owner:
    the bootstrap rule's defaults should be confidence 0.95 and a 10 % minimum population (the
    calibration of HISTORY 139 found 0.5 / 5 % calling two populations in 44-91 % of single
    ones), and linking them to the best_set_K mode, where the same numbers only check the fits'
    comparison (and `min_population` also enters `best_set_K`'s own check), would be arbitrary.
    Built: `choice_defaults(populations)` (`'bootstrap'`: 0.95, 0.10; `'best_set'`: 0.5, 0.05,
    unchanged); `AnalysisConfig.choice` unset by default and resolved by `choice_limits`, so
    changing `populations` changes the limits; keys set in `[analysis.choice]` apply in either
    mode, the others take the mode's defaults. Bundles analysed in bootstrap mode before this
    used 0.5 / 5 % unless the project set them (the private benchmark analyses set 0.95 / 10 %).

143. **Control replicates checked before they are averaged (owner, 2026-10-22).** The checks of
    HISTORY 140 left whole spectra out of the analysis but not out of the controls, so a
    replicate reading another envelope still entered its control's mean (on the benchmark an
    undeuterated replicate carried more deuterium than the peptide's amides). Built in
    `fit_controls` (`_control_replicates_left_out`): a control replicate whose K=1 mean is
    above the amide count is always left out, and one more than `[run.controls]
    replicate_outlier` (6) typical replicate standard deviations from the median of the
    series' replicates of that control (three or more) too; the typical sd is the run's own,
    from its time-point replicates (deuterons per amide, the median of the chi-square-scaled
    variances). Flags (warnings in `control_notes`): `m0_replicate_left_out`,
    `fd_replicate_left_out`. A retention above `retention_max` (1.1) with an undeuterated value
    near 0 is now caught by the amide check first (the control is then `fd_missing` with
    `fd_replicate_left_out`). Benchmark (nterm_subtract 1, charges combined): 13 series had an
    undeuterated replicate left out, none a fully deuterated one; 3 fewer series fall back
    (190 -> 187); retentions change by at most 0.006.

144. **The population test takes the positions' error; the K=2 threshold from two-population
    sets; the borderline evidence a setting (owner, 2026-10-22).** Three items left open in
    HISTORY 128 and 137. (1) The peptide population comparison tested the replicates'
    lever-rule populations with Welch's test, holding the sample's averaged K=2 positions
    fixed, which overstated significance (tutorial data: 25 of 121 significant differences
    remained once the positions' error was counted). Now `t = delta / delta_se` with
    `delta_se` from each side's whole error (`lever_population`: K=1, low and high), `p` its
    two-sided normal value; the threshold is unchanged (K=1 `delta_crit` over the
    separation). (2) With the best_set_K gate, the K=2 ranks' `delta_crit` pools only the sets
    best described by two populations (the others' K=2 fits are splits of one population, and
    those are no longer compared). (3) `[analysis] borderline_evidence` (default 10): the
    margin `compare_population_counts` requires before a difference in the number of
    populations is not `borderline`.

145. **A multimodal fully deuterated control checked against its other charge states (owner,
    2026-10-22; HISTORY 138 point 2).** When no other sample has the peptide (a run with one
    sample), an `fd_multimodal` control was `retention unchecked`. Now `_flag_disagreement`
    compares its own measured retention with the median of the same peptide's measured
    controls at its other charge states in the same sample (`charges`): within
    `sample_tolerance` it gets `fd_charges_agree`, and `control_notes` reads `fd_multimodal
    (retention agrees with its other charge states)`. The comparison with the other samples
    comes first; one that disagrees there stays `retention differs`.

146. **The checks of HISTORY 140-145 on other data; `false` in `[analysis.controls]` (2026-10-22).**
    On the published tutorial data the checks barely act: 2 replicate outliers (one peptide's
    2+ series, fD 1.24 and 1.02 where its sample's combined values are 1.06 and 0.91), nothing
    above the amide count, and no stand-in series once charges are combined; the robust
    `delta_crit` is 0.028 against 0.035 from the plain pool, and the significant
    one-population peptide differences go from 309 to 327 of 504. The known-answer test set
    has one spectrum per peptide, charge and time, so the replicate check cannot apply there.
    Found while doing this: `[analysis.controls]` is a table of changes to `[run.controls]`,
    so its `false` (unset) was not converted; it now follows `[run.controls]`' false rule
    (`replicate_outlier = false` turns the check off there too).

147. **The owner's manual screen against hxex, and contaminant shoulders in sparse profiles
    (2026-10-22; not adopted).** The owner's earlier manual screen of the intact-peptide
    benchmark (460 entries: missed or no peak, missed point in peak picking, contaminating
    peptide, low intensity) maps onto hxex's spectra through pyHXExpress's series ids. The
    "missed point" entries described the December 2024 picks (93 % had a hole inside the
    envelope); the 2025 pyHXExpress picks and hxex's do not (1 %). Prototype cause detectors
    caught about half of the "contaminating peptide" spectra (intensity outside the envelope,
    with the picker's floor kept; a second species in the undeuterated control), and much of
    what they flagged beyond the screen was real interference too, often outside the fitted
    bins. Contaminant shoulders: on sparse profile data (a Q-TOF at R ~22,000, one point per
    ~18 ppm) a +-50 ppm window holds 4-5 points, so the shoulder rule (10 points or more) never
    applies and a neighbour's flank is picked. A test zeroing an edge maximum when the intensity
    rises to a peak beyond 1.5 half-widths changed 3,546 spectra (5,043 bins, mostly small tail
    bins); narrower windows and recentring by each spectrum's measured offset did harm. Owner:
    not better enough to change the method; nothing built. Found on the way: m/z offsets are
    mostly a property of the series (the undeuterated control's offset puts 92 % of spectra
    within 10 ppm), and a peptide whose deuterated peaks sit 0.07 Da from the predicted
    positions while its undeuterated control is on position is another species, not a shift.

148. **A residue-level confidence for the number of populations (design item 3a; 2026-10-22/23;
    scratch only, design note not yet written).** Built per residue from the covering peptide
    sets (charges and replicates pooled): *strength*, a weighted Stouffer combination of each
    set's z (the tail of a gamma fitted to its simulated null, B = 100; z clipped at +-3.09: the
    calibration study's clipping of p to 1e-6 made a p = 1 set count -4.75 and the strongest +2.33,
    biasing its Stouffer rows towards "no"), and *agreement*, the weighted share of the covering
    sets with two populations (calibrated likelihood-ratio test p < 0.01, minor population >= 10 %).
    Owner's choice: four tiers, two localised (strong, agreement >= 0.5), two (strong, 0.33-0.5),
    undecided (one criterion), no sign. Agreement grades localisation, not existence: on the
    simulation the share of strong residues inside the true region falls steadily from 0.95
    (agreement >= 0.67) to 0.48 (below 0.2), and strong residues far from the region are nearly
    absent above 0.2. Why 1/3: a truly two-state residue is covered by peptides that reach outside
    it and lose power (agreement median 0.52, lower quartile 0.33), so a majority rule loses half of
    them; on the intact-peptide benchmark the owner's dimer interface is marked at 1/3 in every
    sample from 4 s to 30 min, and intermittently at 0.5. Undeuterated control sets have 9 % with
    two populations but no residue at "two": ion artefacts do not survive the combination.
    Considered and dropped (owner): decoding blocks with exoneration (spans whose bimodality appears
    only when slow and fast parts are measured together would be cleared) and minor-fraction
    agreement. Owner: multimodality from exchange differences within a peptide or from several
    conformations is signal; fit evidence takes precedence over eyeballed verdicts. Found: an
    isolated residue at "two" in one region came from a second species under one family of ions
    (a z = 2 series below the monoisotopic m/z, 40-70 ppm off the peptide's isotope grid, moving
    with a substitution in the sequence; not carry-over, which would sit on the grid); the K = 1
    gain of the central domain between 30 min and 20 h follows the owner's measured
    subunit-exchange order of five samples exactly, mostly as both populations moving, not as a
    transfer between them.

149. **Spectrum quality screens for interfering species (2026-10-23; prototypes, levels agreed,
    not built).** None of the earlier detectors looked at the raw m/z profile. Four fit-free
    scores from the raw spectrum and the picked bins, with tolerances from the picker's window:
    R1, a species below the monoisotopic m/z extrapolated into the lowest positions; R2, low picked
    bins off their predicted m/z beyond half the window (after the series' undeuterated offset);
    R3, peaks between the isotope positions (outside the window and a quarter step from every
    position) inside the reachable range, split into R3t (up to the envelope's top edge) and R3s
    (separate clusters above it); R4, picked intensity beyond the reachable range (the zero bins
    there are kept: real information). Profile and stick (centroided) paths; about 0.3 ms per
    spectrum. Dead ends: the fitted envelope as the region (fit-dependent), picked bins at 10 % or
    connected bins as the region (a cluster between positions leaves little in the picked bins),
    the picking window alone as "between" (counted peptides' own split maxima). Across four
    datasets (two Q-TOF, a published one at 6 ppm point spacing, an Orbitrap) the scores mean the
    same; flag rates differ with the samples (0.5-12 % at 0.10). The owner's blind scoring of 128
    banded spectra, then fit evidence (counterfactual refits with the interference taken out, and
    replicate agreement): R1 biases the K = 1 mean (median 0.8-3.4 replicate sd), R2 and R3t give
    heavy replicate tails, R3s leaves the K = 1 mean alone but creates false populations (K > 1
    78 % -> 36 %), R4 has no measurable effect. Agreed: one setting, `spectrum_quality_filtering`
    (none / loose / strict, default loose), with three actions: loose removes R1 >= 0.10,
    R2 >= 0.20, R3t >= 0.40, keeps R3s >= 0.20 at K = 1 only, warns at R4 >= 0.10; strict removes
    R1 >= 0.05, R2 >= 0.10, R3t >= 0.20, K = 1 only at R3s >= 0.10. On the benchmark loose removes
    2.1 % and strict 4.4 %. A review of `accepted` keeps a spectrum; the same setting will hold the
    weak-spectrum rule (weak spectra at K = 1 only), to be sized after the noise model. Also:
    `review_K` explained in the guide (picks one of the fits already made; status may stay empty).

150. **Noise: weak spectra and a per-bin noise model (2026-10-23; prototype on a branch).** Weak
    spectra give false extra populations: undeuterated control sets judged two-population by the
    calibrated test, 6 % with tops above 20,000, 11 % at 5,000-20,000, 25 % at 1,300-5,000, 40 %
    below. The fixed 3 % of the series' undeuterated top is wrong in three ways: too low for the
    Q-TOF data (4-5 % at the top of the envelope, 7-11 % for weak spectra), too high for the
    Orbitrap (about 1 %), and constant across a spectrum where the noise grows with each bin's
    height. Measured per run from the undeuterated controls (the natural-abundance envelope is
    known: one height fitted, the residuals are the noise; HDExaminer exports have a flat baseline,
    so the baseline cannot be used), which needs no replicates and so applies to the known-answer
    set; checked against consecutive-scan scatter on Orbitrap infusions (33,788 bin pairs): the
    residual method reads 5-40 % high, the shape's own error. One sd per spectrum from that curve
    brackets the answer from both sides (at each spectrum's own top: known-answer correct K 79.1 %
    against 74.9 % now but benchmark labelled K > 1 still 22 %; at the series' top: K > 1 down to
    6.5-10.7 % but known-answer 71.3 %): a deuterated envelope spreads the same ions over more bins.
    Per-bin weights (branch `perbin-noise`, uncommitted; `NoiseConfig.estimate = "controls"`,
    variance a2 + b I + c2 I^2 per run; default fits unchanged): known-answer 80.6 % (K = 1 / 2 / 3:
    93 / 81 / 69 %), labelled K > 1 21 -> 8 % on a noisy mostly unimodal dataset and 24 -> 17 % on the
    benchmark, with a hump at mid intensity in the undeuterated controls. Not yet in the bootstrap
    or the screen. Next to try: weights from predicted intensities (iterative), the floor, the
    curve's model-error share.

151. **Spectrum quality screens, part 1: the scores (2026-10-23).** The five scores of item 149
    are computed at picking (`hxex.picking.screen`; `raw_scores` per spectrum inside
    `pick_spectrum`, `off_position_scores` over the picked set in `pick_spectra`, since R2 needs
    the series' undeuterated offset), kept on `BinnedSet`, in the bundle's binned table and at
    the end of `spectra.csv`: `screen_below_mono` (R1), `screen_off_position` (R2),
    `screen_between` (R3t), `screen_between_separate` (R3s), `screen_beyond_reach` (R4). Every
    path that picks computes them (a run, a rerun from stored spectra); a resume keeps the
    bundle's; a set picked elsewhere and bundles written before them read NaN (not assessed).
    The reachable range uses the plan's exchangeable sites (amides plus `Hex`) and its isotope
    clusters (the ion-type mixture when there is one), where the prototype took the backbone
    amides and the fitted shape: the same on the benchmark. Checked against the v4 prototype
    on the benchmark (26,211 spectra, re-picked from the stored spectra): all five identical,
    picked bins unchanged; picking 23.6 s -> 29.2 s (0.2 ms per spectrum). Nothing acts on the
    scores yet: part 2 is the `spectrum_quality_filtering` setting (PLAN Next steps 9).

152. **Spectrum quality screens, part 2: `spectrum_quality_filtering` (owner, 2026-10-23).** One
    `[analysis.filters]` setting, `"loose"` (default), `"strict"` or `"none"`, with the levels
    of item 149 (`hxex.analysis.screening`): remove (left out whole, reason `spectrum quality
    (<score>)` in `left_out_spectra.csv`), one population only (the spectrum's K >= 2 records
    dropped where the other K >= 2 filters act; the analysis never reads `review_K`, so this is
    the same route), or a logged warning. Decided (owner, on my recommendations): controls are
    screened, and a removed control counts as rejected, so `analysis_controls` recomputes the
    correction without it (`screened`); a series whose measured fully deuterated controls all
    go becomes a stand-in and, with `stand_in_controls = "exclude"`, is left out as
    `stand-in control (screened)`; fragment series are not screened (levels set on intact
    peptides; on a private ETD example loose would remove 6 %, mostly R2 on c-1 ions, i.e.
    hydrogen-transfer mixtures, not contaminants); accepted spectra and NaN scores are never
    touched; screened spectra do not count in the replicate-outlier scatter. The run is
    unchanged (an analysis setting acts in the analysis). On the benchmark (scores injected
    into the existing bundle): loose leaves out 555 spectra by the screens plus 146 through
    the stand-in cascade, keeps 110 at one population and warns about 465; strict 1,144 + 389,
    174. Replicate outliers fall from 148 to 103 at loose (45 were screened spectra). Each
    analysis takes seconds. Bundles made before item 151 have no scores, so nothing changes
    for them until they are run again.

153. **Per-bin noise: refinements of V4 tried, none adopted (2026-10-23; branch `perbin-noise`,
    still uncommitted).** The undeuterated controls' residual scatter at one bin height grows
    with the spectrum's top: for tops of 5,000 and more it is a function of the height relative
    to the top alone (about 0.2 % of the top at empty bins, 2 % at 3-10 %, 4 % at 10-30 %, 6 % at
    30-60 %, on the intact-peptide benchmark and the noisy unimodal dataset alike), i.e.
    variance ~ kappa T I, better described by the top T than by the total intensity. Refitted
    with that curve (kappa 1.6e-3 / 0.9e-3 / 0.45e-3: 4 / 3 / 2 % of the top at the tallest bin),
    the choice of K got much worse: known-answer correct K 80.6 % (V4) -> 68.7 %, labelled K > 1
    on the noisy set 8 % -> 43 %, on the benchmark 17 % -> 45 %. Weights at the fit's own
    prediction (one reweighting pass) gave 82.2 % on the known-answer set but 18 % on the noisy
    set (its weak undeuterated controls 38-69 % K > 1); V4's curve fitted to relative errors was
    mixed; the floor (0.2-1 % of the top) made no difference. Reason: these give the small bins
    small variances, so the binomial model's mismatch in the flanks and tail of a labelled
    envelope reads as significant and a second component absorbs it (the Poisson failure of
    item 55). The controls' residuals are noise around the right shape; labelled fits also carry
    model error, largest at small bins. A noise model is judged by the K it chooses (known
    answers, undeuterated controls), not by how well it matches the controls' residuals. V4
    (absolute curve fit, smoothed-observed weights) stays the candidate; the benchmark's
    mid-intensity rise of undeuterated K > 1 (9 % at tops 5,000-20,000) is not a matter of the
    curve's form. The Poisson loss under the current code (owner: item 55's conditions did not
    rule it out): at the 3 % scale (about 1,100 counts at the series' undeuterated top), at the
    count measured from the controls (1/kappa), scaled by each spectrum's own top, with an
    absolute counting term in the gain (up to 7 % + 30) and with an offset (variance = prediction
    + 1 to 30 counts), it stays worse than the current Gaussian on the unimodal data (labelled
    K > 1 23-31 % against 21 % on the noisy set, 18-30 % against 24 % on the benchmark; weak
    undeuterated controls 50-56 %), the offset bringing the known-answer set to 78.6 % (V4 80.6,
    now 74.9). The weak controls' second component (median 9 %, about 6 D above the main one,
    always heavier) fits the tail: picked noise is the window maximum, so an empty bin reads
    small and positive (1-8 % of the top in weak spectra), and a loss that scales every spectrum
    to the same counts makes that tail significant. V4 copes through its absolute variance
    floor. Decided (owner): the noise-model work stops here; V4 stays shelved on the branch
    (not built into the bootstrap or the screen), to come back to only if the calibrated
    set-level tests show that the spectrum-level noise still matters. Next: the weak-spectrum
    rule with the current noise. Scripts and tables: private noise_model README.

154. **No weak-spectrum rule; weak spectra should weigh less through their errors (owner,
    2026-10-23).** Weak spectra (total picked intensity below 0.05-0.2 of the run's median
    undeuterated total) give false extra populations on three of five datasets (undeuterated
    controls 22-73 % at K > 1 against 3-7 % typical; their labelled K > 1 carry no information:
    stronger siblings at another charge or replicate are no more often at K > 1), not on the
    tutorial data, and the known-answer set's weak spectra are mostly truly multimodal. A rule
    calibrated per run from the controls works (private evaluation, weak_spectra README) but needs
    about 100-200 undeuterated controls, too many to be feasible (owner); a sibling check and a
    fixed threshold (0.05: a point or two fewer false K > 1, no cost on the known-answer set)
    were judged too complicated or not worth a rule. The existing `low_intensity` flag (below
    0.3 of the replicates' mean) is a different thing: it finds a dropout replicate and flags
    only 1-4 % of the weak spectra at K > 1, which are weak together with their replicates.
    Decided: no rule. Direction (owner): the size of the errors of weak spectra should lower
    their weight in the aggregate fits. Today replicates and charge states are combined by a
    plain mean (the errors only set the combined standard error), and the noise sd is 3 % of the
    series' own undeuterated top, so a weak series' errors are not larger than a strong one's;
    weighting by the errors needs both an inverse-variance combination and errors that grow as
    a spectrum weakens (an absolute noise floor, as V4's, item 153). To be evaluated before
    anything is built (owner): whether the errors grow with weakness today, and whether weighting
    by them improves the combined values against replicates and known answers.

155. **The Gaussian choice reopened: a t loss on per-bin weights, and calibrated errors (2026-10-24;
    scratch on branch `perbin-noise`, uncommitted; private noise_model README).** Owner: the
    Gaussian decision of item 56 "made sense with the information at the time"; another loss is
    worth considering if it can be justified. Two directions item 153 never tried, scored as there
    (the K chosen on the known-answer set and on undeuterated controls): a model-error share of
    the intensity added to V4's curve in quadrature, and a Student-t loss (0.5 (nu + 1) log(1 +
    w r^2 / nu) per bin, IRLS weights in the polish, BIC on that loss) with 4 or 10 degrees of
    freedom on the current noise (V0t) and on V4's per-bin weights (V4t). Correct K on the
    known-answer set: V0 74.9 %, V4 80.6, V4t4 81.7, V4t10 81.9, model error 79.6 / 77.5 (K = 3
    recovery falls, as with every variant that loosens the small bins), V0t4 76.7. Labelled K > 1
    on the noisy unimodal dataset: V0 20.9 %, V4 8.0, V4t4 5.6, V4t10 6.9 (undeuterated controls
    10.0 / 3.2 / 1.4 / 1.7); on the intact-peptide benchmark 23.7 / 16.8 / 15.0 / 15.9
    (undeuterated 7.1 / 5.8 / 4.5 / 5.1; the mid-intensity hump shrinks but stays). Candidate:
    V4t4, best or within 0.2 points of best on every line, at V4's cost. Errors (benchmark, K = 1
    means with K = 1 bootstrap errors against replicate sets, which include labelling-time
    variation, and charge sets of one replicate, which include any charge offset; median
    observed-over-bootstrap ratio normalised to 1 for Gaussian noise): today's errors are
    miscalibrated monotonically with brightness, 2-5x too large below 0.3 of the undeuterated
    control's top and 3-4x too small above it; they grow with relative weakness but hardly with
    the absolute top. V4's errors are flat across brightness and bracket 1 (replicates 0.84,
    charges 1.06) and grow 10x from the brightest to the weakest spectra; V4t4's are 1.2-1.5x
    too large because the bootstrap resamples t noise (variance 2x the curve's): resampling
    Gaussian noise at the per-bin sd under the t-loss fit restores V4's calibration (V4t4g:
    replicates 0.83, charges 1.06, flat across brightness; median error 0.071 D, 0.044 at
    tops above 100,000 and 0.66 below 1,300). Weighting
    by the errors (leave-one-out, inverse-variance against the plain mean): never helps for
    replicates (separately prepared samples; V0 hurts by 16 %, V4 neutral) and helps for charge
    states of one replicate (6-8 %, 13-19 % where the errors differ most) once the errors are
    calibrated. The per-bin fits' K = 1 means are also more reproducible: the median squared
    replicate difference falls 34 % and the charge difference 40 % under V4 (30 / 36 % under
    V4t4). Decision pending (owner): adopt V4t4 as the default loss and noise (per-bin variance
    from the run's undeuterated controls, V4's absolute curve and floor, t loss nu = 4, Gaussian
    resamples at the per-bin sd), the Gaussian resample for its bootstrap, inverse-variance
    combination of charge states within a replicate and a plain mean across replicates. The
    build would need the weights in the screen (its NNLS floor is a least-squares bound: off
    under the t loss, or a t-consistent bound), the calibrated null simulations under the same
    model, new regression goldens, and a deliberate-difference entry in PLAN.md.

156. **Envelope membership (owner's idea) evaluated: strays do not drive the fits; bright spectra's
    isotope tails fall below natural abundance (2026-10-24; private envelope_membership README).**
    Owner's framing: two issues, off-target intensity in the picked peaks (the greater risk) and
    the error bar on each reliable peak. The test: every envelope of the peptide, whatever its
    deuterium distribution, is a non-negative combination of the natural-abundance shape shifted
    along the grid, so a non-negative deconvolution (lenient, NNLS) or the largest envelope that
    fits under the sticks (strict, a linear programme with a tolerance of 2 sd or 5 % of the
    top) leaves the intensity no envelope can produce. On the intact-peptide benchmark the
    lenient unexplained share at 5 % flags 8.7 % of spectra and catches 58 % of the owner's
    "contaminating peptide" marks (the earlier connectivity detector 13 % at the same
    precision); flagged spectra choose K > 1 twice as often (41 against 21 %). But setting the
    excess bins to the explainable intensity and refitting (K 1-3, V0 noise) changes almost
    nothing: labelled K > 1 23.7 -> 23.6 %, K changed in 4-6 % of the flagged spectra,
    known-answer recovery 74.9 -> 74.9-75.7 %, replicate agreement of the K = 1 means unchanged;
    the same on the noisy unimodal dataset. The mixture fit already ignores isolated strays; the
    extra K > 1 of flagged spectra comes from intensity consistent with an envelope (shoulders on
    the flank), which no shape test separates from a population: only replicate, charge-state
    and control evidence can. Not built. Side finding: with the height fixed at the monoisotopic
    peak, the undeuterated controls' isotope peaks at 1-10 % of the top read 0.6-0.8 of natural
    abundance in bright spectra (tops above 20,000), following the height relative to the top,
    not the absolute intensity; in the raw exported profiles (sticks equal the raw apex heights,
    areas agree), on all three datasets (their profiles are TOF-like: resolving power flat in
    m/z; the known-answer set's resampled points do not say). The tutorial data are from a
    Waters Synapt G2-Si (the paper's methods). The natural-abundance shape is wider than bright
    spectra's real envelopes: part of the small-bin model error of item 153. Also in the
    published GSK3 data (Waters, DynamX export), so not one export program; the tutorial run's
    brightest spectra show the opposite (the monoisotopic top reads low: saturation); the
    Orbitrap targeted project zeroes small peaks outright (instrument noise threshold); on the
    benchmark the deficit grows as the charge falls (M+3 at 0.54 of prediction at z1, 1.06 at
    z4). Cause unknown to the owner (candidates: the dead-time correction, thresholding before
    scans are summed); a per-run response curve from the controls is the possible remedy, not
    tested.

157. **Quality flags `saturated`, `tails_low` and `bright` (owner: flag only; 2026-10-24).** The
    two signs of item 156 measured on every undeuterated control, whose true shape is the natural
    abundance of its composition, the height fixed at the tallest predicted peak
    (`quality.control_shape_ratios`): `saturated` when the peaks beside the tallest one (predicted
    at 30 to 100 % of it) read above `saturated_ratio` (1.15) times their prediction, that is the
    tallest peak reads low, the classic sign of detector saturation, which is the only sign the
    owner's question about saturation nomenclature fits; `tails_low` when the tail peaks
    (predicted at 1 to 10 % of the top, at least two) read below `tails_low_ratio` (0.7), the
    deficit of item 156, named for what is seen because its cause is unknown. A control is flagged
    only when most of its series' control replicates show the sign (as `m0_multimodal`) and only
    the brighter half of the run's controls is judged (tallest stick at or above their median):
    without that guard 6 % of the controls below 5,000 read "saturated" on two datasets, noise.
    `bright` marks a series' spectra without a role at least as bright as a flagged control: likely
    distorted the same way, though their own shape cannot be checked. `[run.quality]`
    `saturated_ratio` and `tails_low_ratio` (false: off); `quality.parquet` columns (old bundles
    read False); flags, spectra.csv and the overview as the other flags. Counts: intact-peptide
    benchmark, controls 13 % tails_low (37 % above 300,000) and 4 % saturated, 2.3 % of labelled
    spectra bright; the tutorial run, 27 % saturated (47 % above 300,000), 11 % bright; the
    published GSK3 data, 10 % tails_low (29 % above 300,000); the noisy unimodal dataset 2 % each.
    No fit changes. `bright` also covers the fully deuterated controls, which feed the correction
    (benchmark 46, tutorial 27). Risks measured (private envelope_membership/flags): a saturated
    undeuterated control's K = 1 mean reads 0.07-0.17 D high (0.9-2 % of the amides) against 0 for
    unflagged controls of similar brightness, so the series' corrected uptake reads low by about
    as much; on the tutorial run saturated controls chose K > 1 more often (22 against 14 %).
    tails_low controls chose K > 1 less often (1.9 / 2.4 against 4.5 / 5.1 %); their mean bias
    cannot be read from controls (the K = 1 mean is bounded at 0). Bright spectra choose K > 1
    12-23 points more often than their dimmer replicates at the same time, but series without
    any flagged control show the same, growing with the brightness ratio (brightest minus
    dimmest replicate: +5-7 points at up to 1.5x, +15-38 at 3x or more): mostly the fixed noise
    level per series (errors of spectra brighter than their undeuterated control 3-4x too
    small, item 155), not the distortion the flags detect; their K = 1 means differ from their
    replicates' by a median of 0.01-0.02 D. Documented with this statement in the guide's
    outputs page.
    A grouping of every flag and screen by what goes wrong, proposed and kept here only (owner,
    2026-10-24: "model mismatch" could confuse users; to revisit as the flags are refined):
    interference, another species adds intensity (the five screens; a second species in the
    undeuterated control); envelope distortion, the peptide's own pattern is recorded wrongly
    (`saturated`, `tails_low`, with `bright` as at risk); model mismatch, the theory cannot
    predict the true shape (`poor_envelope`, `poor_fit`); weighting, the error bars are the
    wrong size (`low_intensity`, the bright-replicate effect). Each calls for a different
    remedy: remove intensity, correct the shape, a better model, better errors. It also
    explains item 156: removing stray intensity is the interference remedy, and interference
    was not the main source of error. The guide keeps the flat flag table.

158. **The tutorial download was stale; `hxex resume` redid every bootstrap with a list setting
    (2026-10-24).** Owner: the tutorial "currently throws errors". The committed notebook,
    executed end to end with the committed code (1874fb0) from a fresh unpack, ran without an
    error and reproduced the page's numbers (3,874 measurements; K 1/2/3 = 2,912/839/123). The
    errors come from the download: the zip on `tutorial-data-1` was built on 2026-10-20, before
    `hxex report` refused to replace existing figures (item 134), and its notebook's report cells
    at steps 9 and 10 lack `--overwrite`. A rebuilt zip (the owner replaces the asset; the
    tutorial is not yet announced) carries the current notebook. Running the notebook a second
    time in the same folder showed the refusals the introduction promises for `init` and `run`,
    and also for `report` and `analyse`: the introduction now says so, and how to start again.
    It also showed that the bootstrap cell redid the whole bootstrap (14 minutes): `hxex resume
    PROJECT` keeps a bootstrap only when the manifest's settings equal the project's, and the
    manifest's JSON returns `ks = [1, 2]` as a list while the project holds a tuple, so they never
    matched. Fixed by comparing the project's settings in the manifest's JSON form; regression
    test in `test_resume_adds_or_redoes_a_bootstrap_from_the_project_file`. Page text updated: the
    peptide and charge without a fully deuterated control now takes its other charge's control
    (item 129), the controls notes include retention agreement, the analysis writes about 40
    figures; the page's figures redrawn from this run (butterfly, controls, residues, both Woods
    figures changed; the spectra and uptake figures are identical). Benchmark rerun with the same
    commit for the owner (private: results_1874fb0 beside the earlier bundles).

159. **A fast route without the bootstrap evaluated (owner's question, 2026-10-24; private
    fast_route README).** `best_set_K` is computed from the fits alone (the K = 1 and K = 2
    criteria summed over a peptide's spectra at one time, and the population limit). Each
    project's analysis written with and without its bootstrap (committed code 1874fb0): best_set_K
    identical for 98.8 % (tutorial) and 99.5 % (intact-peptide benchmark) of species and times;
    adding the bootstrap moves the other 9 and 15 from two populations to one (owner: best_set_K
    comes first, the bootstrap after), because the K = 2 precision filter (`max_fD_se`) drops
    their K = 2 description once its error is known, and lets an unknown error pass. The peptide comparisons are unchanged (K = 1 and
    K = 2 positions: identical significance), because the hybrid rule uses only the replicate
    spectra (Welch's t, `delta_crit`); only `delta_se` shrinks (0.4-0.8 of the bootstrap error).
    Without the bootstrap, the residue comparisons agree 97.5 / 88.8 % (they test |delta| / se), the
    K = 2 population comparison has unknown errors for most peptides, and the error bars are too
    small. The tutorial's bootstrap (K = 1 and 2, 3,874 spectra) takes 13.5 minutes on the CPU
    (PyTorch's default 24 threads; 12 are faster here, item 46) and 3.8 on the GPU, against 1.3
    minutes for the fit; the tutorial page and notebook now say about four minutes on a GPU
    (they said a minute or two). Workflows (owner): a quality assessment of a dataset needs only
    the fits and best_set_K; error bars, residue comparisons and the population comparison need
    the bootstrap at K = 1 and K = 2 for every spectrum (`ks = [1, 2]`; not `selected`, which
    leaves each spectrum without the other K, and not K = 3, which the analysis never uses).
    Not built: a fits-only route would need the precision filter to treat unknown errors as
    failing; a bootstrap at K = 2 only for the sets with best_set_K = 2 (28 % of the tutorial's
    species and times, 23 % of the benchmark's) would save about a third; decisions for the owner.
    Guide page started (owner): docs/guide/statistics.md, what the fits alone give, what the
    bootstrap adds, and the roles of replicates and controls, with the tutorial's numbers.

160. **`left_out_summary.csv` in every analysis folder (owner, 2026-10-24).** What was left out
    between the spectra read and the analysis was only on screen (the analysis log) or spread
    over `left_out_spectra.csv` (whole spectra only), the manifest (`[[input.exclude]]`) and
    `series.csv` (controls). The new table has one row per step, a count of 0 included, with
    `stage` (before fitting, review, controls, analysis: whole spectra, analysis: two-population
    descriptions), `what`, `count`, `unit` (spectra, series, peptides and times) and `setting`
    (`analysis.left_out_summary`). It reuses the analysis' own counts: the whole-spectra rows
    count `left_out_spectra`'s reasons, and `_records` hands back the K >= 2 filters' counts
    (`counts=`); `Settings.excluded` carries the manifest's exclusion record (empty for older
    bundles without rules). On the tutorial and the intact-peptide benchmark (1874fb0 bundles)
    every count equals the analysis log's. Guide: outputs.md (table list, folder tree) and
    statistics.md ("Seeing what was left out").

161. **The set test needs very little resampling (owner's question, 2026-10-24; private
    population_calibration/resampling README).** The calibrated set test (a likelihood-ratio
    test of one against two populations per peptide set, against a simulated null from the K = 1
    fits; design item 3a) was run with B = 100 null resamples per spectrum, 2.39 million fits
    (72 minutes on the GPU) for the intact-peptide benchmark. Subsampled offline (4,724 sets):
    the empirical p cannot go below 1 / (B + 1), so per set it needs B >= 100; a gamma tail or an
    independent recombination of each spectrum's draws agrees with B = 100 on 97.4-97.6 % of sets
    at B = 20; a null pooled over the run's spectra (summed over n draws for a set of n) agrees on
    97.7-97.9 % (kappa 0.95, as close as the two B = 100 methods are to each other) from as few as
    2,400 null fits (5 % of the spectra, 2 resamples each; seconds on the GPU). A chi-square with
    3 degrees of freedom per spectrum did as well here, without any simulation, but is not
    guaranteed. The undeuterated sets read two populations 8-10 % of the time whatever the
    method or B: that is the noise model and contaminants, not resampling. Against today's
    best_set_K (summed BIC) the set test finds two populations in 36 against 27 % of labelled sets
    at the same false rate on the undeuterated sets (agreement 91 %). Consequence for the
    workflow (owner: fits with an estimated K first, the bootstrap after for firmer statistics):
    a pooled-null set test fits in the first step, at a few per cent of the fits' cost, and does
    not need or change the bootstrap; the residue-level tiers (item 148) built on it would then
    also come from the fits alone. Not built; to check first on other datasets and within-run
    heterogeneity of the null.

162. **The set test on public Orbitrap data; a crowded digest's interference (2026-10-24; private
    resampling README and the local project's README).** The owner pointed to MassIVE
    MSV000094510 (Moroco et al., Mol Cell Proteomics 2025, doi 10.1016/j.mcpro.2025.100904; CC0;
    owner: may be named, it is from a public repository): an Orbitrap HX study of an unstructured
    Jurkat whole-cell digest and synthetic PTM peptides. The digest's 45 Q Exactive HF runs
    (MS1 at 45,000; 563 peptides; triplicates of 13 times; both controls) were read as Thermo raw
    files with a peptide table built from the paper's supplement. Findings:
    - Interference: 58 % of the undeuterated controls chose K > 1 (51 % with a +-10 ppm window in
      place of the +-30-40 ppm the recorded resolving power gives). Their second component sits
      about 8 D above the first with about 20 % of the intensity: other peptides' peaks landing on
      the deuterium grid over the 8-24 positions a peptide can reach. The loose spectrum quality
      screens remove 11-14 % of the spectra and leave this untouched (it is on the grid). After the
      back-exchange correction hxex and the paper's HDExaminer values both sit on the
      intrinsic-rate prediction (median difference 0.00), hxex with more outliers (12 % off by
      more than 0.1, HDExaminer 7 %); the outliers carry interference (62 % K > 1, 11 % of the
      intensity beyond reach). A centroid of the same sticks is worse than the K = 1 fit, so the
      difference is in the sticks: HDExaminer finds each peptide's peak in each run, reads around
      the envelope and was curated; hxex used one retention time per peptide +-0.15 min (my guess:
      the paper gives no integration range, only a 0.8 min scheduling window for its ETD runs).
      Owner: retention windows found from the data, as well as given ones (PLAN Next steps 10).
    - Resampling (item 161) on these data: a per-set gamma tail at B = 20 agrees with B = 100 on
      98.8 % (benchmark 97.8 %), robust on both; the pooled null is unstable here (single draws
      90-96 %; the null grows more with peptide size: mean 2.5 to 4.6 by size quarter), so it is
      not the safe default item 161 suggested; chi-square 3 df 97.8 % but drifting with size.
      The undeuterated sets read two populations in 49 % whatever the null: interference, not
      calibration.
    Item 161's workflow note stands with one change: the set test's null is best a per-set
    simulation at about 20 resamples (the cost of a bootstrap), not a pooled one.

163. **Unknown K = 2 errors failing the precision filter: tried and reverted (2026-10-24; private
    fast_route, `*_unknown_fails`).** Item 159 found that the fits-only route keeps a few
    two-population descriptions the bootstrap later sets aside, mostly where a set's error is
    unknown (one spectrum and no bootstrap at K = 2: 14 of 18 such rows in the tutorial, 20 of 30
    on the benchmark). Owner approved making an unknown error fail `max_fD_se`. Measured without
    the bootstrap against with it: best_set_K identical for 97.4 % (tutorial) and 98.5 %
    (benchmark), against 98.8 and 99.5 % before; the over-calls fell (tutorial 9 -> 2) but 18 and
    41 sets now read one population where the bootstrap keeps two, because many single-spectrum
    sets have precise K = 2 errors once bootstrapped. Reverted: unknown errors pass as before;
    the fits-only route's small over-call stays, documented in the guide's statistics page.

164. **Retention windows found from the data: prototype on the Orbitrap digest (2026-10-24;
    PLAN Next steps 10; local project README, section "Retention windows found from the data").**
    Per run and peptide, a chromatogram summed over the peptide's predicted bins (the profile's
    tallest point within +-10 ppm), over +-0.5 min of the given time, smoothed over 2.4 s; the
    local maximum nearest the given time; window apex +- a multiple of its half-maximum width,
    ended at the valleys; spectra summed again with hxex's own summing and refitted. The found
    peaks sit a median 0.8 s from HDExaminer's per-run peak times (97 % within 6 s); nearest, not
    tallest, matters (a taller species often elutes 20 s away); the first try with 3-scan smoothing
    split the peaks (MS1 every 0.27 s). With windows of +-0.5 width (median 6.5 s, against the
    fixed 18 s) the undeuterated controls at K > 1 fall from 51 to 35 % and the labelled spectra
    from 30 to 20 %; the corrected uptake against the intrinsic-rate prediction does not change
    (12 % more than 0.1 off). Those misses concentrate in 9 % of the peptides, which sit 2.9 ppm
    off their predicted m/z (0.7 ppm for the rest): co-eluting near-isobaric species, the
    remaining interference (next step). The ambiguity rule as written flags half the spectra and
    needs rethinking. So the windows are worth building for the populations; the uptake
    discrepancy with HDExaminer is not a retention-time matter. Smoothing (owner asked how
    dataset-dependent it is): flat from 1.3 to 4 s (5 to 15 scans; every measure within a point
    or two, the apex changing in 2-3 % of the spectra); only 0.8 s failed. An automatic rule,
    half the run's median half-maximum peak width over its 50 tallest chromatograms (at least 3
    scans), chose 11-12 scans (about 3 s) in all 45 runs: the proposed default, with a setting in
    seconds as the override, since peak widths (LC) and scan spacing (method) vary by dataset.

165. **Retention windows found from the data, built (owner: "build it", 2026-10-24; PLAN Next
    steps 10).** `[input.scans] find_peak` (default false: existing projects read as before):
    in an LC-MS run with a fixed time, each MS1 peptide whose table row gives `rt` and no
    `rt_start`, `rt_end` has its elution peak found (`hxex.io.mzml.find_peaks`): a chromatogram per
    scan, the sum over the peptide's predicted bins of the profile's tallest point within the
    picker's own half-width (`PickerConfig.half_width`, so finding and picking agree), within
    `peak_search` (30 s) of `rt`; the charge states of a peptide summed into one chromatogram
    (retention belongs to the peptide; charge states read from the same scans); a running mean of
    `peak_smoothing` s, by default half the run's median half-maximum width over its 50 tallest
    chromatograms (at least 3 scans); the local maximum (>= 10 % of the tallest) nearest `rt`;
    the window apex +- `peak_width` (0.5) half-maximum widths, ended at the valleys. A peptide
    without signal gets no spectrum in that run; the run log gives each run's measured width and
    smoothing. A found spectrum's `rt` is its scans' signal-weighted time (not the table's).
    Every spectrum of a fixed-time scan run now records `rt_start`, `rt_end` (its first and last
    scan, minutes) through the spectra, the picked table, the bundle (absent in older bundles:
    NaN) and `spectra.csv`. The peptide table takes a `run` column (a run's file name): such a
    row is read from that run only, once per run, and a peptide's general row steps aside for
    the runs that have their own (`not_runs` on its entry), so a run's row overrides it (its own
    window, say); `hxex check` refuses a run name that is no run's file, and with `find_peak` a
    peptide with neither `rt` nor a window. With `find_peak`, `hxex check`'s spectrum index
    plans a found target over its search range (the peak is not known before the scans are
    read). Not built: the ambiguity flag (the prototype's rule flagged half of a crowded
    digest's spectra), finding for fragment ions. Defaults from the prototype on the Orbitrap
    digest (HISTORY 164; smoothing flat from 1.3 to 4 s). Guide: inputs.md ("Finding each
    peptide's peak", the run column, the four settings), outputs.md (spectra.csv), BUNDLE.md.
    On the Orbitrap digest with a peptide table of retention times only (a table with
    `rt_start`, `rt_end` keeps them and reads as before: the same fits exactly): peaks found for
    all 563 peptides in all 45 runs, measured peak width 6.2 s, smoothing 11 scans (3.0 s);
    every window within 1 s of the prototype's (median 0.1 s); K > 1 in the undeuterated controls
    34.0 % (fixed window 51.4 %), labelled 21.0 % (30.3 %); uptake against the prediction 12.6 %
    more than 0.1 off (12.0 %).

166. **Near-isobaric species: the picker's m/z check (owner, 2026-10-25; PLAN Next steps 11;
    private notes in the Orbitrap project's `mz_offset/` and `analysis_examples/series_offset/`).**
    Step 2 of the owner's order: on the public Orbitrap digest (MSV000094510; HISTORY 162, 165)
    57 peptides missed the intrinsic-rate prediction (corrected K = 1 fD more than 0.1 off at
    half their time points or more), HDExaminer fine on 40 of them. Findings:
    - 14 of the 57 were cysteine peptides read as carbamidomethylated (from the DDA
      identifications): HDExaminer's masses are 57.02 Da per cysteine lower and the unmodified
      m/z peaks at the table's retention times, so the HDX samples were not alkylated. An input
      error, corrected in the local peptide table (misses 57 -> 53; > 0.1 off 12.6 -> 11.6 %).
    - Not the calibration: per-run offsets +0.5 to +1.5 ppm, flat in m/z and time; per-scan
      offsets do not drift across the elution peak (no window in time avoids the interference).
    - `mz_offset_ppm` is measured inside the picking window, so a +-10 ppm window hides larger
      offsets (and `mz_shift` at 15 ppm can never fire). Measured from the peak tops of M, M+1,
      M+2 within +-40 ppm, 91 % of undeuterated envelopes sit within 10 ppm (median 0.9); 9 % sit
      10 to 40 ppm away, by the same amount in both undeuterated runs (r = 1.00), whatever the
      m/z, time or intensity: near-isobaric species merged with the peptide's peaks at R 45,000
      (peaks 30 to 40 ppm wide). The +-10 ppm window then reads their flank.
    - A per-bin rule (a bin whose window maximum leads uphill to a top outside the window set to
      0, or left out of the fit's loss) cut K > 1 (labelled 20 -> 13 %) but made uptake worse
      (> 0.1 off 11.6 -> 14.3 to 14.7 %; misses 53 -> 77 to 80): it contested 37 % of the bins,
      whole far-off envelopes included. Not adopted; zero against left out not fairly tested.
    - Series level: flagging a series whose undeuterated offset is beyond 6 times its run's
      robust scatter isolates bad series on both instruments (Orbitrap: 27 % of time points off
      against 9.6 %; Q-TOF benchmark: the flagged series with other charge states disagree with
      them by 1.1 to 1.5 D, typical 0.08 D; known-answer set: none flagged). A fixed 10 ppm
      limit flags 30 % of the known-answer series (their scatter is 8 ppm); the picking window
      as the limit flags nothing in a +-50 ppm window. Inconsistent undeuterated envelopes (M to
      M+2 disagreeing) are bad on the Orbitrap (32 % off) but not on the Q-TOF: not used.
    - Leaving flagged series out (owner's question) costs good values: HDExaminer is within 0.1
      on 86 % of the flagged peptides' time points, hxex on 73 %; leaving them out drops hxex's
      good values from 5,969 to 5,400 (HDExaminer 6,258). Moving their windows by the measured
      offset recovers most: 82 % (5,969 -> 6,034 in all; K > 1 labelled 20.2 -> 17.7 %, fully
      deuterated 19.6 -> 17.2 %). Moving every series gains nothing (the rest lose 13 values).
      On the Q-TOF benchmark (+-50 ppm) moving changes nothing (median 0.000 D, largest 0.27 D)
      and the flagged series still disagree with their other charges: moving does not rescue a
      foreign species, so the flag stays.
    Built (owner: on by default; the limit a setting, 6 by default; document how the settings
    change a run): `hxex.picking.recentre`; `[run.picker] off_mz_sd` (6; false: no check) and
    `recentre` (true; false: flag only). A run is one undeuterated injection (sample and
    replicate); its calibration a Theil-Sen line in m/z (the Q-TOF's drifts by about +1 ppm per
    100 m/z within an injection: with a constant calibration the benchmark's scatter was 2.9
    ppm and two of its four flagged series fell below the limit), its scatter the robust sd
    about it; fewer than 10 controls: all the controls'. Quality flag `off_mz` (statistic
    `envelope_offset`); `mz_shift` now only for series the check does not assess (fragment ions,
    series without undeuterated controls), since it still catches wrong fragment compositions.
    `binned.parquet` and `spectra.csv`: `envelope_offset_ppm`, `envelope_offset_limit_ppm`,
    `window_shift_ppm` (absent in older bundles: not assessed, not moved); `mz` stays the
    predicted m/z; spectrum figures draw the windows where they were. Flagged at 4 / 6 / 8 / 10
    sd: Orbitrap 74 / 66 / 61 / 59 of 527, Q-TOF 22 / 4 / 2 / 1 of 1,320, known-answer 0. The
    built run on the Orbitrap digest reproduces the experiment exactly (6,034 good values).
    Guide: picking-window.md "A series whose peaks sit off its m/z" (with "Changing these
    settings"), outputs.md; PEAK_PICKING.md; BUNDLE.md.

167. **The remaining gap to HDExaminer, and undeuterated controls above `m0_max` (owner,
    2026-10-25; private notes in the Orbitrap project's `gap/`).** On the Orbitrap digest
    (bundle with the m/z check, item 166) the paper's predicted uptake is a rough yardstick: it
    cannot be reproduced with `hxex.kint` at the stated pH 3 and 20 C (best fit at an apparent pH
    ~2.4 leaves a median 0.044 fD), whether or not k_back is included (the spreadsheet
    convention of pure-D2O forward rates; at 88 % D2O the correct relaxation rate is only ~5 %
    faster, moving the normalised prediction by at most 0.015 fD), and both programs miss the
    same mid-curve points (rate errors; the paper itself reports 97 % within two-fold). Compared
    directly instead: HDExaminer's deuterons are 1.145 times hxex's everywhere, controls included
    (about 1/0.88: it probably divides by the D2O fraction; cancels in fD); the fully deuterated
    ranges agree (median ratio 0.999); the programs differ by > 0.1 fD at 240 of 6,751 peptide-
    times, 197 of them in 24 peptides. Those 24 have undeuterated controls choosing K > 1 (median
    share 1.0 against 0.0), differences of several deuterons, `data_below_m0` in 9 (0 of the 514
    peptides that never differ) and stand-in controls in 22 (`m0_high` in 14). Side by side, the
    peptide's own envelope is small or absent in the undeuterated control and another species
    fills the reachable range (+5 to +12 Da, a 5+ across a 2+, one 47 ppm off); hxex's K = 1 fit
    takes it for the peptide. For IACLDFSLQK HDExaminer's values are the uptake of the species
    47 ppm off (windows moved onto it reproduce them within 0.03-0.07); nothing is at the exact
    m/z at HDExaminer's retention time, and the weak signal at the exact m/z at 4.29 min is no
    peptide envelope (M+1/M 0.2 against 0.62): the free-cysteine peptide is not seen (its DDA
    identification was of the alkylated form). A guard on recentring (move only below one peak
    width) has no support: 5 of the 66 off_mz series are 0.85 peak widths or more off, none with
    a separate peak at the exact m/z, and moving improved agreement with both yardsticks there;
    the 40 ppm search is about one peak width on this Orbitrap.
    `hxex analyse`'s `stand_in_controls = "exclude"` already left out 20 of the 24 (series whose
    fully deuterated control is a stand-in). Built (owner: option 2, `m0_max` as the limit, the
    existing reason kept, no sample exception, one switch): it also leaves out the series whose
    undeuterated control read above `[run.controls] m0_max` (`m0_high`), reason `stand-in
    control (undeuterated)`, with a warning when a sample loses half its series or more; a
    missing undeuterated control (`m0_missing`) is not a reason; `accepted` in the review and
    `stand_in_controls = "include"` keep them. Effect: digest 22 of the 24 left out, 11 of the
    25 that differ at 1-3 times, 2 of the 514; among the series kept the programs differ at 38 of
    6,334 peptide-times (0.6 %, from 0.9 %). Q-TOF benchmark: 6 series newly left out, five
    plainly another species (70-78 2+/3+ in three samples, a 107-117 2+; means 3.5-7 D, K > 1)
    and one borderline (a 90-113 4+ at 0.59 D, K = 1: the kind to accept by hand). Known-answer
    set: one series (13.0 D), the one the owner had already excluded for its +13 Da envelope. Tutorial data: none.
    Guide: back-exchange-correction.md, outputs.md.

168. **The set test on the cleaned Orbitrap bundle; the population test designed (owner,
    2026-10-25; docs/ANALYSIS.md "The population test"; private notes in the Orbitrap project's
    `set_test_offmz/`).** Re-measured after items 165-167 (B = 20 per-set null, 505,940 fits,
    15 min on the GPU; p < 0.01, minor share >= 0.10; the spectra hxex analyse keeps): undeuterated
    sets read two in 28.4 % (49.4 % on the old bundle; best_set_K 35.2 %, 59.1 %), labelled sets
    17.5 % (36.2 %); the rules agree on 95.5 % of labelled sets. The undeuterated sets read as two
    hold another species, not a calibration error: the second component a median 7.9 D above the
    first, share 0.16, the heavier in 99 %; screen_beyond_reach >= 0.10 in 40 % (16 % of those read
    as one); no intensity trend. Labelled sets read two in 42 % of series whose undeuterated set
    reads two and 7 % where it reads one (65 % of all labelled two-population sets are in the
    former); their K = 1 fD stays close to HDExaminer's (1.8 % against 0.3 % of peptide-times more
    than 0.1 apart). CPU cost of the null: 9.7 ms per spectrum and simulation (24 threads),
    against 1.8 ms on the GPU. Owner's decisions for the design: one statistic from spectrum to
    residue (the set test replaces best_set_K; "best_set" and "bootstrap" modes kept for now);
    significance 0.01 and a 10 % minor population, user settings; a series whose undeuterated set
    reads two is kept at one population by default (the route of item 152); the bootstrap gives
    errors only; the stage is opt-in (`[run.population_test]`; the owner rejected `[run.null]` as
    meaningless to users), and with it on each spectrum's K follows the test, otherwise BIC or AIC;
    the test steps beyond two (K against K + 1), and a residue can hold more than two
    conformations (owner: more than two states at a residue is the heterogeneity measured), so the
    residue tiers beyond two are to be tested on simulated data; `[run.fit] screen` is the owner's
    quick-look route, whose desired output resembles the tiers, so their interaction is to be
    evaluated (verdicts and tier figures with the screen on and off).

169. **The population test's four evaluations (2026-10-25; docs/ANALYSIS.md "The population test";
    private notes in the population-test examples folder).** Step-1 null of the intact-peptide
    benchmark at B = 20 (524,240 fits, 17 minutes on the GPU); sets of the spectra hxex analyse
    keeps; p < 0.01 and a 10 % added population.
    - **The screen.** `[run.fit] screen` skips K = 2 for 73 % of the benchmark's spectra: it is exact
      for one spectrum's BIC, but the set test adds many gains below that penalty. With T = 0 for
      skipped spectra 200 of 1,093 labelled two-population sets are lost and the residue tiers keep
      79 % of their cells (the "two" tier 10 -> 2 % of residues); leaving skipped spectra out of
      their sets biases the other way (280 gained). The screen's loss floor bounds a skipped
      spectrum's statistic (T <= 2 (nll1 - floor), held for all), so a screen made for the test
      skips K = 2 only where that bound is below 1: 31 % of the fits, 99.8 % of labelled-set
      verdicts unchanged, 99.1 % when those spectra also take a chi-square(3) null in place of
      their simulation (31 % of the simulations saved). A chi-square(3) null for every spectrum
      with every fit (no simulation) agrees on 95.6 % of sets and 89 % of residue cells: a usable
      quick look, but item 162 found it drifting with peptide size on the Orbitrap.
    - **Beyond two (simulation).** Overlapping peptides over 60 residues, three replicates. Three
      equally spaced states 0.35 fD apart are not resolvable: BIC picks K = 3 in 13 % of the
      clearest peptides and the test in 7 %; the step-2 null itself is well behaved (per spectrum
      mean 2.5, 99 % 9.6). With states 0.05 / 0.50 / 0.95, 35 % of the three-state peptides read
      three and 77 % of the true residues reach a three tier (60 % localised), none far away; but
      two independent two-state regions two residues apart put 23 % of their residues at a three
      tier (15 % localised), all at the junction, where most peptides span both regions and show
      three or four components: there agreement cannot tell them from a three-state region, as the
      design feared. How sets that read one enter step two hardly matters (z = 0 is the simplest;
      leaving them out adds stray marks). One two-state region never reaches a three tier.
    - **The undeuterated rule.** Known-answer set: no undeuterated set reads two, so it removes
      nothing; each spectrum's K from the test equals BIC's within two points (73.1 against 74.9 %
      correct; K = 3 57 against 56 %). Benchmark: 8.0 % of undeuterated sets read two (second
      component 6.2 D above, share 0.15); capping their series removes 16 % of the labelled
      two-population sets and a quarter to a third of the residue cells at two at the early times.
      Unlike the Orbitrap (item 168), these peptides read two in their other samples too, where
      the undeuterated set reads one (53-62 % against 32 % overall), and 15 of the 31 peptides read
      two in one sample's undeuterated set only: on this Q-TOF the rule mostly removes evidence
      other samples confirm.
    - **CPU cost.** CPU-only build pinned to 8, 4, and 2 cores: 12.7, 14.6, and 20.6 ms per spectrum
      and simulation, about five times the run's own K = 1 to 3 fits at B = 20 whatever the cores
      (more cores help little beyond four): 2.1 hours on four cores for 26,000 spectra, one step.

170. **The undeuterated rule examined; decisions on the population test (owner, 2026-10-25).**
    Decisions: the test-specific screen of item 169 replaces the BIC screen when the population
    test is on, with the chi-square null as the quick look; the residue tiers go up to three for
    now. The undeuterated rule, examined on the intact-peptide benchmark: its sets reading two are
    not carry-over (the extra component sits at a labelled level of the same peptide no more often
    than chance) but a narrow cluster on the isotope grid (7 D above, share 0.20, width 0.11 D
    against about 1.1 D for populations in labelled spectra; the Orbitrap's signature too). They
    are the dim ones (share reading two 15 % in the dimmest intensity quartile, 3 % in the
    brightest; dimmer than the same peptide in other samples in 76 %), with a cluster of similar
    height across samples: a species of fixed abundance shows only under a weak peptide, so capping
    the series of the sample where it shows is arbitrary. A physical bound separates such clusters
    from deuterated populations: a population of mean mu D whose amides each hold at most the D2O
    fraction r has a variance of at least mu (1 - r). Below 0.3 of that width (components of at
    least 1 D and 10 %) are 78 % of the undeuterated and 16 % of the labelled spectra reading two on
    the benchmark, and 1-2 % of the known-answer set's real two-population mixtures.

171. **The width check and the undeuterated cap, by the data (owner, 2026-10-25: the data decide,
    not the eyeballed review).** The owner's blind review of 70 benchmark spectra (picked as
    current hxex picks them; the sparse-profile shoulder problem of item 147 is unchanged) did not
    tell the width check's narrow components from passing ones (interference seen in 7 of 30 and
    5 of 30). The data: narrow components are single-bin in 52 % (Orbitrap 63 %), above the most
    deuterium possible in 25 % (passing 6 %), present in the undeuterated spectrum in 23 % (5 %);
    neither kind moves the K = 1 values (distance to the replicates' median identical to
    one-population spectra). Undeuterated sets (two is a known false result): the width check
    lowers their two from 8.0 to 2.9 % on the Q-TOF and from 25.3 to 9.1 % on the Orbitrap, at the
    cost of 158 and 236 labelled two-population sets. The undeuterated cap cannot change that rate,
    but the fully deuterated sets of capped peptides read two twice as often as the rest (Q-TOF 29
    against 15 %, Orbitrap 29 against 14 %), and as often when the cap is widened to the peptide in
    every sample (15 % of the FD sets instead of 8 %): the species is there in every sample, so item
    170's reading (the cap removes evidence other samples confirm) was wrong. Width check with the
    peptide-wide cap: Q-TOF UN / FD / labelled two 8.0 / 17.0 / 34.8 -> 2.9 / 9.7 / 22.2 %, Orbitrap
    25.3 / 17.5 / 16.2 -> 9.1 / 9.4 / 5.7 % (an unstructured digest, where two should be rare).

172. **Peak picking: only apexes count (owner, 2026-10-25: picking first; the window maximum may
    be a habit carried over from pyHXExpress, and rules on top of it are patches).** The window
    maximum takes a neighbour's flank whenever a brighter peak sits just outside a window (the
    owner's example: another species ~90 ppm below the predicted positions, its falling flank
    picked in every window). Prototypes on stored spectra, judged by known-false results
    (undeuterated sets reading two), fully deuterated sets reading two, labelled sets reading
    two, and replicate agreement: two patches on the window maximum (zero an edge maximum;
    only when it rises into a neighbour beyond 1.5 half-widths) changed little on the Q-TOF
    benchmark (undeuterated two 7.2 -> 6.2-6.7 %). Taking the tallest local maximum whose apex
    lies inside the window (a flank is never a peak) did as well there (6.6 %, values unchanged)
    and much better on the Orbitrap digest with its +-10 ppm window (undeuterated two 25.3 ->
    15.4 %, fully deuterated 17.1 -> 14.0 %, labelled 15.7 -> 12.1 %, replicate agreement
    unchanged; 22,293 spectra changed); the known-answer set's K recovery was unchanged (74.9 %).
    A tolerance tighter than the window, centred on each series' offset and sized by undeuterated
    peaks (+-16 ppm on the Q-TOF, where they sit within 16 ppm), cut real deuterated peaks: these
    scatter +-20 to 40 ppm about the predicted positions on the Q-TOF (blended deuterium and
    carbon-13 isotopologues sampled every 23 ppm), so the window's reach is not arbitrary there
    (undeuterated two 3.3 % but fully deuterated 22.0 %, replicate outliers doubled).
    Built (owner, 2026-10-25): `[run.picker] pick`, `"peak"` the default (a local maximum of the
    raw spectrum, a gap in the data counting as lower; every point in centroided data, fewer
    than four points per isotope step as in the screens; the apex from the parabola through the
    point and its neighbours; peaks found once per spectrum over the windows' reach), and
    `"window_max"` for pyHXExpress's picks (the golden file and the whole-run oracle still match
    with it); the shoulder rule applies to `"window_max"` only. A documented difference from
    pyHXExpress (docs/PLAN.md); guide picking-window.md "What a bin takes: a peak". Results made
    before keep their sticks until run again (`hxex run --spectra`).

173. **The width check and the undeuterated cap on the new picks (2026-10-25).** Re-measured after
    item 172 with the full simulated null (B = 20). Picking alone took the undeuterated sets
    reading two from 8.0 to 7.0 % on the Q-TOF benchmark and from 25.3 to 15.8 % on the Orbitrap
    digest. The width check (a K = 2 component narrower than 0.3 sqrt(mu (1 - D2O fraction)),
    >= 1 D and >= 10 %, keeps the spectrum at one population) then brings both to about 3 % (2.8
    and 3.2 %), at 139 and 218 labelled sets. The undeuterated cap, widened to the peptide in every
    sample of the run, still finds deuterated sets reading two far more often in the capped
    peptides (labelled 62 against 30 % on the Q-TOF, 33 against 9 % on the Orbitrap, an
    unstructured digest; fully deuterated 32 against 16 % on the Q-TOF, but 18 against 14 % on the
    Orbitrap, where most of item 171's enrichment had been flank picks). Both together: labelled
    sets reading two 33.8 -> 22.9 % (Q-TOF) and 12.5 -> 5.9 % (Orbitrap), fully deuterated
    18.2 -> 11.1 % and 14.4 -> 9.6 %.

174. **The population test, built (owner: "continue with the population test things",
    2026-10-25; docs/ANALYSIS.md "The population test"; guide population-test.md; private notes
    in the population-test examples folder, `built_test/`).** `[run.population_test]`
    (`simulations` 20, `significance` 0.01, `min_population` 0.10, `null` simulated or
    chi_square, `width_check`, `seed`; needs `k_min = 1`, `k_max >= 2`, the Gaussian loss):
    `hxex.population_test` simulates each spectrum's null at every step (K against K + 1, the
    run's own fit settings, each draw seeded by the spectrum id so chunking does not matter),
    keeps the null's mean and variance, applies the width check (the HISTORY 173 rule, r from
    `[run.controls] d2o_fraction`, its default 0.9 without controls; a spectrum leaves the step
    where its K + 1 fit has a narrow component and every step after) and sets each spectrum's K
    from its own test (the bootstrap, figures and `spectra.csv` follow; `criterion_K`,
    `test_p_<K>`, `narrow_component` in `spectra.csv`; `population_test.parquet` in the bundle,
    carried by `hxex resume`). With `[run.fit] screen` the run uses the test's screen (K + 1
    fitted only where the loss floor leaves T room to reach 1; skipped spectra T = 0 with the
    chi-square(3) null); `population_test()` refuses a fit made with the criterion's screen.
    `hxex.analysis.set_test`: sets of the spectra the analysis keeps (left-out spectra, the
    screens' one-population spectra and `review_K = 1` leave; accepted spectra are exempt from
    the width check and the cap), summed statistics and null moments, a gamma tail;
    `[analysis.filters] undeuterated_cap` (default true) keeps a peptide's labelled and fully
    deuterated spectra at one population, in every sample, when one of its undeuterated sets
    reads two judged *before* the width check (as HISTORY 173 evaluated it; judged after, only
    peptides with a wide extra population were capped: labelled two 24.3 against 22.1 %).
    `[analysis] populations = "test"` (the default when every run has the test, else
    `best_set`; the settings file records the rule used) fills `best_set_K`, `best_set_reason`
    and the new `set_p_1`, `set_p_2`; downstream the test is treated as `best_set` (the
    residue gate, the comparisons). Residues (K = 1 rows of `residues.csv`):
    `test_strength_2/3`, `test_agreement_2/3` and `population_tier` (six tiers; sets without a
    p-value at a step enter with z = 0, a step no set reached is NaN), and the figure
    `population_tiers` (the HISTORY 148 tier map; the two "three" tiers a rose pair, every pair
    of tiers at least 9.9 apart in OKLab under simulated protan, deutan, and tritan vision).
    One change from the evaluation: a set adds up its spectra's null means and variances
    instead of summing the draws elementwise (deterministic, exact for the chi-square null;
    same rates on the benchmark, 98.6 % of set verdicts equal on the Q-TOF and 99.5 % on the
    Orbitrap, the differences symmetric at the boundary). Benchmark check (Q-TOF, K = 1 and 2,
    13.5 min on the GPU, the null 12.3 min = 1.4 ms per spectrum and simulation): sets reading
    two UN / FD / labelled 3.2 / 10.9 / 22.1 % against the prototype's 2.8 / 11.1 / 22.9 % on the
    same fits, 97.6 % of the 4,731 sets equal one by one (two independent 20-draw nulls);
    spectra reading two 13.5 % by the test against 19.5 % by BIC; the width check stops 10.0 %.
    Tutorial data (published; CPU, default threads of the 48-core workstation, K = 1 to 3):
    fits 73 s, step one 13.2 min (10.2 ms per spectrum and simulation), step two 22.5 min
    (17.4 ms): 11 and 18 times the fits, not the "about five times" of item 169 (measured with
    pinned cores; the progress weights and the guide use these). Spectra: 3,092 one, 705 two,
    77 three (BIC: 2,912 one, 840 two); width check 221. Sets: 13 labelled sets read three,
    15 of 174 undeuterated sets two (10 peptides capped); residue cells two (localised) 223,
    two 99, three, localised 3 (R120G at residue 144, the long peptide the owner expects rate
    heterogeneity in, and WT at 24). Found on the way: `best_set_K` follows the test only up to
    the largest K the analysis describes (`ks`, (1, 2) by default), so the residue levels read
    the test's own number, the new column `set_test_K`. Open: a null draw far above the rest (a K = 1 refit stuck in a poor optimum)
    inflates its spectrum's null variance (0.06 % of the Q-TOF spectra, 0.43 % of the
    Orbitrap's; the evaluation's numbers include them); `hxex resume` adding the test to an
    existing bundle; the test for fragment series (not evaluated).

175. **The population test's two rules become flags; borderline from the test; why null refits
    get stuck (owner, 2026-10-26).** Decisions: the width check is a flag only (not yet
    justified as a rule: nothing showed the labelled spectra it stopped were wrong):
    `narrow_component` in `spectra.csv`, the `width_check` setting removed, each spectrum's and
    set's test unchanged. The undeuterated cap is a flag by default (`undeuterated_two` in
    `species_uptake.csv`, on every set of a peptide whose undeuterated set reads two in some
    sample of the run; the tier figure's undeuterated rows show it) and a switch
    (`[analysis.filters] undeuterated_cap`, default false): the owner, "the potential
    contaminant could move out of the measured bins", and item 173's enrichment does not show
    that the labelled spectra's second populations are the species. With the test on, a
    difference in the number of populations between samples (`compare_population_counts.csv`,
    the Woods plot's outlines) is borderline unless the side with more populations has p below
    `significance / 10` and the other above `significance x 10`, at the step where they differ
    (columns `p`, `p_reference`); `borderline_evidence` acts only without the test.
    Stuck refits (private notes `built_test/stuck/`): the null draws far above the rest are K = 1
    refits stuck on a tiny envelope at 0-5 D while the data sit at 8-21 D: weak (peak 4-49 against
    unit noise), highly deuterated spectra, four of six fully deuterated controls (6 spectra,
    0.02 %, on the new picks of the Q-TOF benchmark; reproduced exactly from their seeds). Cause:
    the K = 1 start (`hxex.fit.init.kmeans_starts`) is a method-of-moments estimate from the
    weighted bins; noise spread over 30-45 bins makes the variance exceed the mean, p is clipped
    to 0.02 and n to its cap, so the start's mean collapses to about 0.5 D wherever the data are,
    and Adam does not recover from an envelope that does not overlap them (the random restarts
    did not either). A start that keeps the cluster mean (p raised to mean / n when the estimate
    loses it) refits all six to the loss of the generating curve (statistics 64-5,276 -> 1-8).
    The run's own fits share the start: 21 of 26,164 benchmark spectra (0.08 %; 16 labelled, 5
    undeuterated) have a stuck K = 1 fit (K = 1 mean 1.5 D, 8.4 D with the kept-mean start); their
    statistics stay above 20 (verdicts barely move), but their K = 1 values are wrong. Proposed,
    not built (it changes every fit): keep the mean in the start.

176. **The k-means start keeps the mean (owner: "fix the stuck fits", 2026-10-26; item 175).**
    `hxex.fit.init.kmeans_starts`: where the method-of-moments `n` is clipped to its range, `p`
    is set to `mean / n`, so the start keeps the cluster's mean instead of collapsing to about
    0.5 D (p clipped to 0.02 when noise makes the spread exceed the mean). Every K's k-means start
    uses it (the nested starts follow from K = 1). Refits of the two private bundles at their own
    settings (`built_test/stuck/start_fix_eval.py`): Q-TOF benchmark, K = 1 losses lower in 21
    spectra (19 by more than 10: the stuck fits of item 175), higher in none; K = 2 lower by more
    than 1 in 112 (20 by more than 10), higher in 10 (none by more than 10); K = 1 means moving
    by more than 0.5 D: 26; the criterion's K changed for 47 of 26,164 spectra (42 up, as their
    K = 2 fits improved). Tutorial data, K = 1 to 3: K = 1 unchanged, K = 2 lower in 6 and higher
    in 2, K = 3 lower in 14 and higher in 4 (none worse by more than 10); the criterion's K changed
    for 3 of 3,874. The six stuck null draws refit to the loss of their generating curve. Test:
    `test_kmeans_start_keeps_the_mean_of_a_weak_spectrum`. The prototype regression
    (`HXEX_FULL_REGRESSION`) and GPU parity pass. Found by the chunking test with the new start:
    a spectrum fitted exactly at K and K + 1 (an undeuterated test spectrum of 9 bins) had a
    statistic and null draws of rounding noise (1e-15), whose p-value was arbitrary (0.78 in one
    chunking, 0 in the other, reading two). A statistic is now 0 below the precision of the
    losses, the screen's tolerance (`2 (1e-3 + 1e-4 |nll_K|)`, `statistic_of`), for the observed
    statistic and every null draw: on the benchmark 5 % of the statistics and 2 % of the draws
    are below it (median 0.0024), and no spectrum's or set's K changes.

177. **The residue choice follows the tiers; `populations = "population_test"` (owner,
    2026-10-26).** The rule's name is `"population_test"` (was `"test"`: too little information,
    owner), matching `[run.population_test]`. With it, `choose_populations(..., tiers=True)`
    decides each residue by its tier: two populations where the tier is "two" or above
    (`[analysis.choice] tier = "two"`, the default the owner chose: strong evidence and a third
    of the covering weight) or "two, localised" or above (`"localised"`), and the residue has
    K = 2 values; the bootstrap tests of the choice (separation, precision, K = 1 between,
    population range) no longer apply in this mode (owner: "if those rationales are now
    obsolete, then they can go"; they stay in the "best_set" and "bootstrap" modes). The K = 2
    values and their error bars come from the bootstrap; without one they are drawn without
    error bands, and the residue profile and choice figures say "no error bands: they need a
    bootstrap ([run.bootstrap], ks = [1, 2])" (owner: clear communication that reliable error
    bars need the bootstrap). `reason` names the tier that fell short (`tier: undecided`, `tier:
    no sign`, `tier: two` with `"localised"`) or `no K=2`; a "three" tier is drawn as two
    (`tier: three, drawn as two`; K = 3 residue values are not drawn yet). Without a bootstrap
    the bands come from the replicates' scatter only (too small: `statistics.md`), and the
    figures say "no bootstrap: error bands from the replicates' scatter only, too small". The
    private reruns with the start of item 176 (`results_start`, `analysis_start_WT` in both
    population-test folders): at the time points, 1,032 of the benchmark's 1,238 residue-times
    at tier two or above are drawn as two and 206 (17 %) have no K = 2 values (the filters left
    out every covering peptide's K = 2 description; 3,139 spectra have a component above fD 1.1,
    `max_fD`); the tutorial data 570 of 577 (7 without K = 2 values).

178. **K = 2 fits above full deuteration (2026-10-26; private notes `built_test/above_fd/`).** Why
    17 % of the benchmark's tier-two residue-times had no K = 2 values (item 177): the `max_fD`
    filter leaves out K = 2 fits whose higher component is above corrected fD 1.1, 16 % of the kept
    K = 2 fits (3,810), all above the series' own fully deuterated control (measured median 0.72 of
    the amides). The share grows with deuteration: undeuterated 8.7 %, 4 s 9.1 %, 60 s 9.5 %, 30 min
    15.4 %, 20 h 22.0 %, fully deuterated controls 43.4 %: nearly fully deuterated envelopes (K = 1
    fD close to 1; not detector saturation) split into a main component and a minor (median 0.24),
    often narrow component beyond full deuteration; the series' retention is ordinary (0.56). The
    test calls 2,988 of these spectra one population and 822 two (higher component median corrected
    fD 1.40, 90 % 2.11; 38 % carry `narrow_component` against 18 % of all spectra tested two). A
    second deuterated population of the peptide cannot sit there, so for those the test's "two" says
    the envelope is not one binomial (a heavy tail, a broader envelope, another species), not that
    there are two deuterated populations. Open (owner to decide; flags first): a flag on sets and
    residues whose two populations have no physical K = 2 description; why the heavy end of nearly
    fully deuterated envelopes is heavier than a binomial. The threshold of `max_fD` (1.1), measured
    (`above_fd/threshold.py`): genuine single fully deuterated populations, the K = 1 corrected fD
    of every fully deuterated control spectrum and of the 20 h points (measured controls only),
    reach q99 1.07 / 1.07 and q99.5 1.08 / 1.10 on the benchmark (controls / 20 h) and q99 1.03 /
    1.10, q99.5 1.04 / 1.13 on the tutorial data; the control part of the correction's error is far
    smaller (se median 0.007, q99 0.04-0.06), so a per-series threshold from it would cut real
    values; the K = 2 high components have no valley to place a threshold in (a continuous tail from
    1.0 to above 2). So 1.1 sits at about the 99th to 99.5th percentile of genuine fully deuterated
    single populations on both datasets.

179. **Spectra above full deuteration leave the population test (owner, 2026-10-26; item 178).**
    With the population test, a spectrum whose K + 1 fit has a population above corrected fD
    `[analysis.filters] max_fD` (1.1; uncorrected series: deuterons over amides) leaves the set test
    from that step on, at one population (as the screens' one-population spectra; the review's
    accepted spectra excepted); no new setting, on fits alone (owner: no bootstrap needed in the
    current workflow). `left_out_summary.csv` counts them ("held out of the test: a population above
    fD 1.1 in the two-population fit"). Benchmark (`results_start`): 4,583 spectra held out; sets
    reading two undeuterated 7.0 -> 3.0 % (a known false result), fully deuterated 18.6 -> 10.0 %,
    labelled 33.0 -> 28.2 %; residue-times at tier two or above 1,032 -> 895. The threshold is the
    existing one, about the 99th to 99.5th percentile of genuine fully deuterated single
    populations (item 178). Not built: widening it by the component's own error with a bootstrap,
    `1 + sqrt(0.1^2 + (z se)^2)` (the two errors in quadrature: subtracting `z se` and keeping 1.1
    would count the scatter twice); a bootstrap-free error from the fit's curvature is possible but
    least reliable for exactly these minor, narrow components.

180. **What drives two-population fits above full deuteration (2026-10-26; items 178-179).** The
    owner declined widening the threshold by the component's error. Measured on the Q-TOF benchmark
    (`results_start`) and the tutorial run. (1) Most of these fits are the null's behaviour, not a
    physical heavy tail: K = 2 fits of spectra drawn from each spectrum's own K = 1 fit plus the
    run's Gaussian noise (three draws each, refitted at 1 and 2) put a population above corrected
    fD 1.1 as often as the data do (fully deuterated controls 41 vs 40 % benchmark, 51 vs 51 %
    tutorial; at every time point the simulation is at or above the data). They are splits of
    the main envelope (high component median 0.2 to 0.5 sd above the K = 1 mean; 74 % benchmark,
    93 % tutorial), common for short peptides because one sd of the envelope is 0.25 to 0.4 of the
    undeuterated-to-fully-deuterated span, so half a sd above a fully deuterated envelope is
    already above 1.1. Not the n bound (K = 1 at its n cap is not more common among them), not
    overdispersion (variance ratio no higher than the rest). The test calls most of them one
    population. (2) Those the test calls two (benchmark 822, tutorial 184) are of two kinds: another
    species beyond 3 sd of the envelope (benchmark 227, mostly undeuterated and early points; the
    heavy species of items 168-170), and splits of spectra brighter than their undeuterated
    reference: among the splits, the test calls two 74 % (benchmark) and 84 % (tutorial) of those
    more than twice as bright as their series' undeuterated controls and 0 to 5 % of those dimmer.
    The brightness effect is general, not limited to these: fully deuterated controls (one
    population) more than twice as bright as their undeuterated controls test two in 51 %
    (benchmark) and 62 % (tutorial), those dimmer in 0 to 2 %; labelled spectra 63 and 68 % vs 1 to
    8 %. Cause: the Gaussian noise sd is fixed at 3 % of the reference's tallest peak, so a relative
    misfit of the envelope's shape (tutorial: flattened tops of the brightest spectra, log10
    intensity 6.7 vs 5.9; benchmark: envelopes broader than a binomial) grows with brightness, and
    the null, drawn with the same constant noise, does not contain it. The hold-out of item 179
    removes some of these (those above 1.1), not the cause. Scripts:
    `population_test/built_test/above_fd/drivers` (private, README).

181. **Peak heights vs integrals, and per-bin noise curves on both (2026-10-26/27; items 150-153,
    180).** Five datasets: the Q-TOF benchmark, the tutorial run, the private noisy unimodal
    dataset, the known-answer set (all Q-TOF exports), and the published Orbitrap digest. (1)
    Picking: on the Q-TOF exports weak peaks are wider than tall ones (1.15 to 1.22x at 5 to 15 % of
    the spectrum's top in three datasets, 1.04x overall in the noisy one; the monoisotopic peak
    included, so not isotopic fine structure; not baseline: valleys 0.3 % of their neighbours), so
    `pick = "peak"` heights under-read the weak, heavy bins (benchmark: -13 % three bins past the
    tallest). Integrals over +-1.5 FWHM, the FWHM measured from the run's tallest peaks (a power law
    in m/z, capped below half the isotope spacing) and rescaled per spectrum to the heights'
    maximum, remove that deficit and halve the undeuterated controls' residuals; integrating to the
    valleys overshoots (rejected); 1.5 FWHM suits every dataset (2 is the same on the benchmark).
    The Orbitrap shows no width effect and keeps its heavy-side deficit (-8.6 -> -7.3 %): not
    picking. (2) Integrals with the constant 3 % noise: fully deuterated controls at K > 1 15 -> 9 %
    (benchmark; 8.8 -> 5.2 % without its one outlying sample, whose fully deuterated controls read
    K > 1 in 38.7 % with heights against 4 to 13 % in the other four, and are much brighter relative
    to their undeuterated controls: 49 of the 70 more than twice as bright as their reference), 8.5
    -> 6.6 % (tutorial), 27 -> 20 % (noisy); known-answer correct K 74.9 -> 78.3 %; the robust
    replicate scatter of K = 1 means falls by a quarter to a half and the charge scatter by a fifth
    to a half (the bootstrap errors do not change, so they now over-state); the Orbitrap is slightly
    worse (+1 to +3 points). (3) Per-bin curves `a2 + b I + c2 I^2`: the proportional term is
    required (without it fully deuterated K > 1 17 to 92 %; benchmark heights without the outlying
    sample 24 %, against 4 % with the term). Heights with the prototype's controls curve are the
    best combination on the Q-TOF data (fully deuterated K > 1 5 / 5 / 9 %, the benchmark without
    the outlying sample 4.0 %, which the controls curve also brings from 38.7 to 9.2 %; the
    brightness effect of item 180 84 -> 10 % on the benchmark, 86 -> 14 % without the outlying
    sample; robust error ratios 0.72 to 0.88 replicate, 1.07 to 1.50 charge), but the prototype's
    fit is decided by the brightest group and over-states weak bins' noise 3 to 7x; on integrals it
    set the low terms to 0 and under-stated them. A relative fit that follows the controls' noise at
    every intensity chooses K > 1 more often in the controls (benchmark fully deuterated heights 6.8
    %, integrals 17 %; Orbitrap 74 %). Reading: the undeuterated controls' residuals are noise only
    (their shape is exactly natural abundance), whereas deuterated envelopes also carry shape error
    (broader than one binomial, items 178-180); the more exact the noise and the sticks, the more
    that shape error reads as a second population. A curve pooled from the bundle's own fits (no
    controls) is worse still (the K = 2 and 3 fits absorb the shape error). (4) The noise floor
    (owner's question: do instruments and exports set it differently?). Measured from each bundle's
    undeuterated controls (bins where the natural-abundance envelope is below 1e-3 of the top;
    trimmed RMS), the floor grows with the spectrum's brightness as about top^0.5 to 0.8 on every
    dataset, the same four or more bins past the envelope as next to it (co-eluting species and
    background, not the envelope's tail). Exports differ: Q-TOF exports 37 to 56 counts in their
    dimmest controls with 25 to 33 % exact zeros (one tutorial-protocol export about 200); the
    Orbitrap 76 % exact zeros (censored) and the rest other species at 2 to 7 % of the top. Refits
    with the prototype's floor (sd at least 0.5 % of the top), a measured constant, or a measured
    power law of the top: on the Q-TOF data the floor moves K > 1 by at most 1 to 3 points (not the
    limit there); on the Orbitrap the measured power law is what makes per-bin noise work, better
    than the constant sd for the first time (undeuterated K > 1 21.2 -> 15.1 %, fully deuterated
    13.7 -> 13.4 %, labelled 12.9 -> 11.9 %), where the 0.5 % share had made it worse (25 %). The
    measured floor is instrument-agnostic (from the run's own controls), but on the Orbitrap it is
    not a noise estimate: those spectra are reduced profile data close to sticks (about 9 points per
    resolved peak, 44 % exact zeros, sub-threshold signal removed), so heights and areas carry the
    same information and there is no noise floor to measure; its gain comes from down-weighting bins
    that hold other species' signal. The principled form for such data is counting variance for
    reported peaks with zeros as censored, and off-target signal handled as interference (owner's
    point, 2026-10-27). The fully deuterated K > 1 shares are a yardstick, not false-positive rates:
    heterogeneous back-exchange narrows an envelope (a Poisson-binomial) and cannot by itself make a
    mixture choose K > 1, but some fully deuterated controls may truly hold two populations; the
    conclusions rest on what does not need that assumption (the brightness gradient, which real
    populations cannot follow, the undeuterated controls, the mostly unimodal set, the known
    answers, and error calibration). (5) What drives the misfit (the fully deuterated controls need
    not be single populations, so nothing is learnt from them): K = 1 residuals divided by a noise
    model matching the undeuterated controls grow with the spectrum's absolute brightness in every
    role, the undeuterated controls included (benchmark heights chi2/dof 0.31 -> 2.88 from the
    dimmest to the brightest fifth), so the error scales with the spectrum's amplitude at every bin
    and can be measured from the undeuterated controls alone; relative brightness adds a
    flattened-top (saturation) signature on the tutorial run only. On the Orbitrap detection is a
    probability (undeuterated isotope peaks at 1-3 % of the top reported 45 %, at 3-10 % 76 %; Q-TOF
    exports 96-100 % above 0.3 %); 35 % of its undeuterated controls have a dropped peak inside the
    envelope, these are fitted with K > 1 more than twice as often, and they are half of its
    undeuterated controls at K > 1. (6) Item 156's tail test (`quality.control_shape_ratios`)
    repeated with integrals: the tail deficit and its charge dependence go (benchmark tails at 1-10
    % of the top 0.73 -> 0.97 at tops 100,000-300,000, the noisy set 0.77 -> 1.05), so `tails_low`
    mostly detects the height artefact (item 156's areas used a window of half the peak width);
    left: the benchmark's brightest controls (0.81 above 300,000) and the tutorial run's saturation
    (peaks beside the tallest 1.18, unchanged). In bright undeuterated controls the remaining misfit
    sits in the bins at 1-30 % of the top, with a consistent sign, and repeats from replicate to
    replicate (brightest third: median series error 11.8 % against 5.5 % replicate scatter, Spearman
    0.74, benchmark integrals; dimmest third: equal, 0.23 to 0.36): a per-series shape error, not
    noise, dominant once counting noise is small; as variance it would be treated as independent
    between replicates. Nothing built; owner (2026-10-27): area picking waits on the noise model.
    Open for the owner: integrals with the constant sd now; heights with the controls curve; or
    first a shape-error term estimated from the fully deuterated controls (cross-validated). Scripts
    and tables: `noise_model/height_vs_area`, `noise_model/areas` (private, README).

182. **Random numbers keyed by what a spectrum is, not its id; analysing part of a dataset
    (2026-10-27; owner).** Every random number of a measurement (fit restarts and k-means
    jitter, bootstrap draws, population-test nulls) was seeded by the stage's seed and the
    CRC-32 of its spectrum id, `PROJECT:LOCAL`, so the same spectrum read under another project
    name, or with another file layout, got other draws. Now `Measurements.random_id` (default
    the spectrum id) keys them, and `Measurements.from_binned` sets it to
    `fit.measurements.measurement_identity`: sample, peptide, modification, residue range,
    charge, replicate, time, role and label, with `#1`, `#2`, ... for spectra alike in all of
    these, numbered in spectrum-id order. A bundle recomputes it from its picked spectra when
    read (nothing new is stored), so `hxex resume` and the population test draw as the run did.
    The seeds themselves were already settings (`[run.fit]` 0, `[run.bootstrap]` 42,
    `[run.population_test]` 0) and are written in `run_settings_used.toml` and the manifest.
    Existing bundles give slightly different bootstrap errors and nulls when rerun. Also from
    the owner's question about users fitting their own batches: a read-only audit listed the
    values pooled across series or the whole run (stand-in fully deuterated retention, the
    outlier floor of 5 or more controls, kint with 10 or more, the replicate-outlier limits,
    `stand_in_controls`, `delta_crit` over every peptide, time and sample, the m/z calibration
    per injection with its fallback below 10, the brightness-relative flags) and the silent
    fallback of a series without its undeuterated control (each spectrum its own noise
    reference: a smaller sd, more populations; no warning yet). Owner: what matters most is that
    users know what analysing part of a dataset does; new guide page
    `docs/guide/part-of-a-dataset.md`, linked from the index and from outputs.md "Running again".
    Not built: warnings for the silent fallbacks, storing every run-level value, a
    batch-invariance test.

183. **A warning for series fitted without their undeuterated control (2026-10-27; item 182).**
    `hxex run` now warns when series have no undeuterated control (each of their spectra its own
    noise reference: the sd set from its own tallest peak, usually lower than the series', so
    the number of populations can be over-estimated), naming up to five series, or once when no
    series has one; not with a fixed `[run.fit.noise] sd`. `series.csv` gains `noise_reference`
    (`undeuterated control`, `own spectrum`, or `run's scans` for kinetics;
    `hxex.noise.noise_reference_kinds`), and `hxex info` counts the series of each kind. The
    guide page `part-of-a-dataset.md` points to them. Not built: the same in `hxex check` (it
    would need the reference rules before fitting).

184. **Inferred envelopes and the bright-spectrum misfit (2026-10-27; items 180-181).** The
    reproducible residual of bright undeuterated controls is mostly specific to each peptide
    (a response shared by share of the top and brightness explains R² 0.13-0.34 on integrals;
    a co-eluting species at a fixed m/z adds some: rank correlation of undeuterated and fully
    deuterated excesses at the same bins +0.30 on the intact-peptide benchmark, +0.39 on the
    private noisy unimodal dataset). A peptide-specific error of the natural-abundance shape
    would reach the deuterated spectra by convolution, which `[run.envelope] mode = "infer"`
    already models, so four Q-TOF datasets were refitted with it (constant noise, heights and
    integrals). It helps slightly and harms nothing measured: labelled spectra with two or
    more populations down 0.6-3 points (most on the noisy unimodal dataset), the known-answer
    set's correct K up 0.3-1.3 points, the replicate-to-bootstrap ratio on the benchmark
    1.46 to 1.37 (heights) and 1.43 to 1.36 (integrals). Fully deuterated controls more than
    twice as bright as their undeuterated reference still read two or more populations in
    58-89 %. Those controls are not bright in absolute terms on the Q-TOF datasets (median top
    about 25,000 against an undeuterated reference of about 8,000; 4-6 % above the undeuterated
    controls' 90th percentile), and by absolute top neither control reads two populations more
    often when brighter (bright undeuterated controls 0 % above the 95th percentile). So on
    those data the excess comes from the noise reference, not from brightness: the sd is 3 % of
    a dim reference's tallest peak, too small for a spectrum two or more times taller. On the
    tutorial data it is brightness (undeuterated controls above about 800,000 read two
    populations in 22-63 % before inference: saturation, item 181). Noise that scales with the
    spectrum's own intensity, not the reference's, is the next candidate. Undeuterated controls are their own fit shape
    with inference, so their rate of two populations (0-1 %) is no evidence. No code change.

185. **Areas when the run shows the width effect (2026-10-27; items 181, 184).** Owner: the
    stick measure is judged by how well the undeuterated controls reproduce their
    natural-abundance envelope (the one ground truth, free of any noise model), so the choice
    does not wait on the noise model; rule agreed: areas when the run's weak peaks are wider
    than its tall ones, heights otherwise. Built as `[run.picker] measure` (`"auto"`, the
    default; `"height"` as pyHXExpress; `"area"`) and `area_fwhm` (1.5), in
    `hxex.picking.measure`: the run's undeuterated controls are picked once with heights, the
    FWHM of each one's tallest peak and of its weak peaks (5-30 % of the tallest) measured from
    the raw profile, and areas taken when the median weak / tallest ratio over the brighter half
    of the controls is at least 1.06, from at least 20 controls (fewer, or centroided spectra:
    heights). Dim spectra give unreliable widths: over all controls the noisy unimodal dataset
    read 1.07, over the brighter half 1.14. The four Q-TOF datasets read 1.12-1.19, the Orbitrap
    digest 1.02 (with only 20 controls, 5-95 % ranges 1.04-1.25 and 1.00-1.03). The width at an
    m/z is a Theil-Sen power law from the tallest peaks (flat on the Q-TOF data, +0.50 on the
    Orbitrap, as its resolving power predicts). An area is the integral of the linearly
    interpolated profile (gaps count as zero) over +/- 1.5 widths of the apex (at most 0.45 of
    the isotope spacing), divided by the width times the median area per unit height and width
    of the controls' tallest peaks, so a peak of the run's width keeps its height on average
    whatever its line shape (a Gaussian's factor assumed instead read 4 % low, because linear
    interpolation reads a peak sampled at 2.6 points per width a few per cent wide). The
    choice is the same for envelope inference and picking, recorded in the manifest
    (`peak_measure`) and shown by `hxex info`. Also found: at 2.6 points per width a height
    depends on where the points fall on the peak (the tallest point up to a tenth below the
    apex), an area does not. Refits of the four Q-TOF datasets with the built picker reproduce
    the scratch integrals (known answer correct K 74.9 % with heights, 78.3 % scratch, 79.6 %
    built; benchmark fully deuterated controls reading two or more populations 14.9, 8.8, and
    8.6 %; labelled 22.9, 19.1, and 20.2 %); the mean replicate-to-bootstrap ratio rose on the
    benchmark and the noisy dataset (1.43 to 1.54, 1.62 to 1.95) with the robust ratio unchanged,
    because labelled spectra now read brighter against their undeuterated reference and the
    constant noise (3 % of its top) is relatively smaller: the noise model's problem of item
    184. Owner on the next step, the noise from replicate differences: labelled replicates can
    differ in their real exposure (timing at short times), so only undeuterated and fully
    deuterated controls (and adjacent scans, where the scans are read) can measure noise that
    way. Guide: picking-window.md "A peak's height or its area" and part-of-a-dataset.md;
    PEAK_PICKING.md, PLAN.md (deliberate difference), BUNDLE.md.

186. **Are the sequence-based natural-abundance shapes right? (2026-10-28; items 181, 184.)**
    Owner: be confident in the theoretical shapes before attributing differences from them to
    anything else. (1) The calculation: brainpy's shapes equal a direct convolution of the NIST
    isotope abundances (pyteomics `nist_mass`) to 1e-15, and every stored shape in five bundles
    (the intact-peptide benchmark, the tutorial run, the private noisy unimodal dataset, and the
    known-answer set, area sticks; the published Orbitrap digest, heights) rebuilds exactly from
    its sequence. (2) The composition: an error in an abundance or a composition rule grows with
    the number of atoms, but the 13C abundance that best fits each undeuterated control (bins at
    least 5 % of the tallest; middle three brightness quintiles) does not change with carbon count
    within one charge (benchmark 2+ rho -0.04, n 890; 3+ -0.14, n 586; noisy set 2+ and 3+ within
    0.02). (3) The abundance: fitted 13C 1.065-1.09 % on the Q-TOF datasets at 2+ and above,
    against IUPAC's 1.070 % (d13C -32.6 permil; biological material -10 to -30 permil is
    1.068-1.095 %, a difference of a few per cent at heavy bins), with residuals at the bins that
    carry the signal within +-1.5 % (benchmark), +-3 % (noisy set), +-1 % (tutorial, dimmest
    quintile only), and +-3 % at the first three bins of the Orbitrap (later bins lose weak peaks,
    item 181(5)). (4) Charge pairs of one injection agree less than replicate pairs of one charge
    (residual Spearman +0.17 vs +0.45 benchmark, -0.09 vs +0.30 noisy set; tutorial +0.91 vs +0.90,
    saturation shared): the composition is the same at both charges, so the reproducible pattern
    of item 181(6) belongs to the ion, not to the sequence-based shape. Item 184's "peptide-
    specific error of the natural-abundance shape" is withdrawn: what repeats across replicates
    is driven by brightness (replicates of one peptide share it) and by the ion. What departs
    from the shapes: brightness (the tutorial run's compression from tops of about 275,000, lower
    than the 600,000-800,000 of item 181; the benchmark's heavy-side deficit above about 210,000,
    -16 to -26 % at bins 3-5), a tilt with charge, the picked floor at bins
    below 5 % of the top in dim spectra, and the Orbitrap's dropped weak peaks. The IUPAC value
    stays (small next to these). Not tested: sulfur on its own. The tilt with charge: across
    spectra 1+ ions read 13C 1.035-1.050 % (d13C about -60 permil, below any biological value)
    and 3+ and above 1.08-1.10 %, but 90-99 % of the 1+ undeuterated controls carry another
    species (a picker screen above 0; chemical background in electrospray is mostly singly
    charged) and the few clean ones read 1.072 % (benchmark, n 18); paired within one injection
    1+ to 2+ is +0.013 (share higher 0.56). From 2+ to 4+ a small tilt remains: paired +0.02 to
    +0.03 points of 13C per charge step (2+ to 3+ higher in 70 % of 327 pairs, 3+ to 4+ 72 % of 134;
    noisy set 2+ to 3+ +0.025, 63 %), the same in spectra no screen flags and with heights, and
    unrelated to m/z or brightness within one charge (|rho| <= 0.1). Peak geometry cannot cause it:
    the Q-TOF widths are a constant share of m/z (49 ppm, exponent 0.05), so width over isotope
    spacing is the same at every charge of a peptide. Untested candidate: a detector response that
    depends on the ion's charge (impact energy, single-ion pulse height) acting on weak signals.
    Its size: about 3 % of the M+1/M ratio per charge step, a shift of the natural envelope's mean
    of about 0.015 Da for 50 carbons, negligible for uptake. Heights read 13C 0.03-0.05 points
    lower than areas (benchmark 2+ 1.000-1.050 %, d13C to -95 permil): the height deficit of weak
    peaks of item 181, which areas remove. Scripts:
    `noise_model/shape_check` (private, README).

187. **The noise curve from replicate differences of the controls (2026-10-28; items 180, 184,
    186; step 1 of the noise plan).** Noise measured where the signal is clean: two replicates of
    one series and role (undeuterated or fully deuterated), tops within 1.5x, every screen below
    its loose threshold, tops below the run's distortion onset (the first band whose
    undeuterated controls' flattening leaves 0.9-1.1), bins at least 5 % of the pair's tallest.
    The difference d = y1 - r y2 (r from the sums) has E[d²] = a²(1 + r²) + b I (1 + r) + 2c² I²
    under a² + b·I + c²·I² per replicate, so replicate differences give the curve without a
    shape model and cancel any misfit the replicates share; it is compared with the same
    curve from the same undeuterated controls' residuals against natural abundance (what the
    per-bin prototype V4 fits). Area sticks of the step-2 bundles of item 185 (the
    intact-peptide benchmark, the tutorial run, the private noisy unimodal dataset; the
    known-answer set has no replicates) and heights of the published Orbitrap digest.
    (1) The counting term agrees between pairs and residuals (b 23 vs 21, 40 vs 51, 16-23 vs 22
    in the bundles' units) and the floor is about 0. (2) The proportional term is smaller from
    pairs: tutorial 0 vs 2.1 %, benchmark 1.9 vs 2.9 %, noisy set 3.0 vs 3.1 %, the same with tops
    within 1.15x. So the residual curve's c mixes noise with the reproducible residual (the residual
    beyond the replicate-pair noise, i.e. the part the replicates share; any of categories 2-5:
    lost or distorted signal, added signal present in every replicate, own processing, physics),
    by a share that differs per run (all of it on the tutorial, where the compression explains
    it; about half on the benchmark, cause not shown; little on the noisy set). On the
    benchmark c grows with brightness below the flattening onset (pairs with tops under 100,000
    0.5 %, 100,000-400,000 2.1 %). (3) Fully deuterated pairs differ more
    (benchmark 4.0 %, tutorial 1.3 %) because the replicates differ in deuterium content: with a
    per-pair shift removed they agree with the undeuterated pairs within the datasets' spread.
    (4) At matched intensity the noise grows with charge (benchmark, 3,000-10,000: 2+ 6.8 %, 3+
    7.2 %, 4+ 9.5 % per replicate; fully deuterated 1+ 4.6 % to 5+ 10.2 %), roughly as a counting
    term whose signal per ion grows with charge: consistent with item 186's untested
    detector-response candidate. (5) Tails: 3.9 % of the benchmark's undeuterated pair bins lie
    beyond 5 robust sd (in 10 % of the pairs), 1.6 % noisy set, 0.4 % tutorial: added or lost
    signal in one replicate that the loose screens pass. (6) The bottom: counting holds down to
    0.2 % of the tallest on the benchmark and reads 1.2-1.7x higher below 2-5 % on the tutorial
    and the noisy set; bins present in only one replicate pass 5 % below about 1-2 % of the
    tallest (the lost-signal zone). (7) Orbitrap: a large floor and counting term in its own
    units (relative sd 24 % at 3e5, 0.8 % at 4e7, no proportional term in the undeuterated
    pairs), fully deuterated pairs noisier at the same intensity even with the shift removed;
    adjacent-scan differences not measured (they need the scans re-read). (8) Owner: is part of
    the reproducible residual another noise source, e.g. the Gaussian term? Noise independent
    between replicates, of any form, is already in the pair differences (and the floor is about
    0), so only noise the replicates share can sit there. Pairs of undeuterated controls of one
    peptide sequence and charge from different samples (chemically identical) give c 1.04 %
    tutorial (0 within a sample; 0 with the per-pair slope removed: a tilt), 2.63 % benchmark
    (1.94 within), 3.11 % noisy set (2.95 within), against residuals 2.14 / 2.94 / 3.13 %: about a
    quarter, two thirds, and most of a small remainder of the reproducible residual varies between
    samples (session or preparation: the bundles cannot tell). The rest, shared across samples,
    is the brightness-dependent distortion. Not yet used in any fit. Scripts: `noise_model/pairs`
    (private, README).

188. **Refits with the replicate-pair noise curve (2026-10-28; item 187).** The per-bin prototype
    (items 150-153) ported onto main in a scratch worktree; K 1-3 refits and K = 1 bootstraps of
    the item-185 area bundles with three curves: V4 (the prototype's residual curve), P (item
    187's undeuterated replicate-pair curve: noise alone) and a fallback for runs without
    replicates (the residual curve's floor and counting term, no proportional term), against
    today's constant sd (V0). With P the fully deuterated controls choose K > 1 in 26-33 % (V0 6-21
    %) and labelled spectra in 25-61 % (V0 16-26 %), and the share grows with brightness (benchmark
    fully deuterated 16 % below a top of 30,000, 81 % above 1.2 M; V0 5-10 % throughout): the
    signature of a misfit of constant relative size that the noise, relatively smaller in brighter
    spectra, no longer covers. Undeuterated controls barely change on the benchmark and the noisy
    set (their misfit is a heavy-side deficit, which no mixture can use) but split above the
    tutorial's onset (its compression broadens the envelope). The same misfit sets the real
    precision: noise-only errors of the K = 1 mean (median 0.012-0.032 D against 0.06-0.07 D) are
    2-3x smaller than the scatter between charge states of one injection (robust ratio 2.65
    benchmark, 3.06 noisy set, 8.8 tutorial). So the reproducible residual of deuterated envelopes
    has to be modelled in the fit (a term of the variance or of the model), not only marked; the
    marks of the noise plan (brightness onset, lost-signal zone, outliers in one replicate) still
    apply. V4 on area sticks stays worse than V0 (its fit is ruled by the brightest groups), and
    the fallback behaves like P (known-answer set: correct K 77.5 % against 79.6 % with V0 and
    82.7 % with V4; true one-population spectra kept at K = 1: 76 % against 98 %). The charge
    yardstick assumes that a replicate label names one injection at every charge (the pair curve
    does not): a replicate's deviation from its charge's mean correlates with the same label's at
    another charge (+0.85 tutorial, +0.65 noisy set, benchmark Spearman +0.52; different labels
    -0.06 to -0.34), so it holds; the noisy set's charge sets scattering more than its replicate
    sets is a systematic difference between charge states, not mixed-up labels. Scripts:
    `noise_model/pairs` (private, README, step 1b).

189. **Where the model-error term should come from (2026-10-28; item 188).** Item 188 left the
    reproducible residual of deuterated envelopes to be modelled, as a term (m I)^2 added to the
    replicate-pair noise. Owner: an m estimated over a bundle depends on what was submitted, and
    a stability test on these data would only show that they are homogeneous. (A) Per series, from
    its own clean controls (robust; sampling error from half splits of the bins), m is about as
    uncertain as its value (medians 2-5 %, sampling sd 1.6-4.7 %), series do differ beyond that,
    and a series' undeuterated m hardly predicts its fully deuterated m (Spearman 0.07-0.25): no
    per-series route. The noise level itself grows with charge, consistently (benchmark 1+ 0.46
    to 6+ 2.79 times the bundle curve; noisy set 0.71 to 2.04; tutorial 1.06 to 3.86), which a
    per-bundle curve ignores. (B) One fixed m for every dataset, 0-15 %, refitted on the
    intact-peptide benchmark, the tutorial run, the private noisy unimodal dataset, the
    known-answer set, and the published Orbitrap digest: about 10 % (7-10 %) calibrates the
    replicate errors on all four datasets with replicates, two Q-TOF groups and the Orbitrap alike
    (robust ratio 0.82-0.94; today's constant sd 0.60-0.97), with fully deuterated K > 1 at or below
    today's on the Q-TOF data (4.0 / 17.0 / 7.5 % against 6.1 / 21.1 / 8.6 %), undeuterated K > 1 on
    the tutorial 3.9 against 11.6 %, and the known-answer set 82.9 against 79.6 % correct K; the
    Orbitrap's fully deuterated K > 1 stays above today's (21.5 against 13.7 %). Replicate sets
    include labelling-time scatter, so 10 % is an upper bound; charge sets (one injection) stay
    under-covered (1.4-2.1) by systematic differences between charge states. 10 % is far above
    the per-series misfit: the fixed term also covers the heavy tails and the spread between
    series, and it dominates every bin above about b / m^2 (a few thousand counts on the Q-TOF
    exports), so the bundle's pair curve matters only for weak bins. Not built. Scripts:
    `noise_model/pairs` (private, README, step 1c).

190. **The noise as a + z b1 I, one instrument scale per bundle, and b1 by sample (2026-10-28;
    item 189).** The per-series noise level divided by charge is flat (benchmark 0.43-0.48 from 1+
    to 6+), as for counting with a signal per ion that grows with charge, so the noise variance
    was written a² + z b1 I with one scale b1 per bundle (estimated from the undeuterated controls'
    residuals: no replicates needed), plus item 189's fixed (m I)² at 10 %. b1 does not change with
    sample beyond its error on any dataset (benchmark, five samples: 7.0-8.6 ± 0.9-1.5 from the
    undeuterated residuals, 5.9-8.1 ± 0.9-1.1 from the fully deuterated K = 1 residuals, including
    a sample whose fully deuterated control the owner suspects), and a b1 per sample changes no
    result by more than 0.5 points. Against the replicate-pair curve with the same m: equal on the
    tutorial, better on the noisy set (labelled K > 1 13.2 against 16.2 %, fully deuterated 15.8
    against 17.0 %, charge-set ratio 1.87 against 2.11), slightly worse on the benchmark (fully
    deuterated 8.1 against 7.5 %), known-answer set 83.5 against 82.9 % correct K; replicate errors
    calibrated alike (0.91-0.96). Correction to item 189: the benchmark's lower fully deuterated
    K > 1 with the fixed m came from the suspect sample alone (20.9 to 8 %); in the other four
    samples today's constant sd chooses K > 1 less often (5.4 against 7.4-8.1 %). Not built.
    Scripts: `noise_model/pairs` (private, README, step 1d).

191. **Per-bin noise from the controls built as an option, and checked (2026-10-28; items
    187-190).** Built on main as `[run.fit.noise] estimate = "controls"` with `model_error` (percent,
    default 9: the value at which item 189's sweep calibrated the replicate errors on the three Q-TOF
    datasets, 9.1 / 8.9 / 9.0 interpolated; the Orbitrap would want about 6.7). Each bin's variance
    is z b1 I + (m I)^2 on the bin's smoothed observed intensity (three bins), floored at (0.5 % of
    the spectrum's top)^2; `b1` is measured once per bundle (`hxex.noise.controls_noise`, called by
    `run.bundle_noise`) from the undeuterated controls that pass every loose screen and whose tops
    sit below the run's distortion onset, with a standard error from 20 resamplings of the series
    and a value per sample (`not measured` with fewer than five series or no counting term; a
    warning when a sample differs by more than three standard errors). The fit's loss, the LM
    polish, the screen's loss floor (weighted NNLS), the bootstrap's resampling and the population
    test's null all take the per-bin variances; the bundle's manifest records the curve
    (`noise_curve`) and `hxex info` prints it. Estimator details, each found by a synthetic test:
    bins grouped by intensity within each charge (mixing charges biased b1 low); the residual's
    expected variance through the projection of the least-squares height (the tallest peak's
    residual is far below its noise); robust group variances standardised by their expected
    variance and iterated (a heteroscedastic group's MAD runs low); the relative fit divides by
    the fitted, not the observed, group variance; the floor a^2 dropped (measured about 0 on every
    dataset, and it traded off against b1). About +5 % bias remains. Bug found by the checks: the
    known-answer set's dimmest controls read flattened (picked noise in their small heavy-side
    peaks), so the onset was put at 0 and no control was used; a departing band now counts only
    above a dimmer band that matches. Checks on the step-2 bundles of item 185 (GPU; K 1-3 refits,
    K = 1 bootstraps with n_boot 20, the population test at B = 20; today's constant sd = V0, the
    built per-bin noise = P9; b1 tutorial 31.2 ± 20, noisy set 11.5 ± 1.3, benchmark 8.16 ± 0.49,
    known-answer set 9.35 ± 14 from 21 controls; every sample agreed with its bundle): the screen
    picks the same K for 100 % of spectra on all four; the slow regression comparisons and GPU parity
    pass. K > 1, V0 against P9, undeuterated / fully deuterated controls, with the population test:
    tutorial 8.1 / 5.0 against 0.8 / 3.3 %, noisy set 5.1 / 14.8 against 4.1 / 10.2 %, benchmark
    4.2 / 5.4 against 3.6 / 4.4 % (without the suspect sample 3.9 / 3.4 against 3.4 / 4.3 %),
    known-answer set 13.0 / 13.0 against 8.7 / 8.7 % (23 controls each), correct K 77.8 against
    84.8 %. With the criterion alone: tutorial 11.6 / 6.1 against 4.3 / 4.2 %, noisy set 8.3 / 21.1
    against 10.6 / 19.4 %, benchmark 6.0 / 8.6 against 7.0 / 8.6 % (without the suspect sample
    fully deuterated 5.4 against 8.5 %), known-answer set 13.0 / 17.4 against 17.4 / 34.8 % (8
    against 4 of 23), correct K 79.6 against 83.5 %; labelled spectra split more often (benchmark
    20.2 against 24.7 %, tutorial 26.4 against 30.3 %). Replicate-set calibration (robust ratio)
    0.60 / 0.97 / 0.82 against 0.89 / 1.04 / 1.04 (tutorial, noisy set, benchmark); charge sets
    0.42 / 2.89 / 1.23 against 1.97 / 2.38 / 1.58 (both under-cover differences between charge
    states). The population test took 0-36 % longer with P9 (benchmark 1,134 against 1,544 s).
    So the option passes with the population test and is mixed with the criterion alone. Owner
    (2026-10-28): not the default until it is more reliable; it stays opt-in. Open: more false
    splits of controls than today with the criterion alone (two datasets' undeuterated controls,
    the known-answer set's fully deuterated ones) and on the benchmark without its suspect sample,
    the Orbitrap, and steps 2-3 of the noise plan (lost-signal zone, outliers in one replicate). Scripts: `noise_model/pairs` (private,
    `built_check.py`, `built_check/summary.txt`).

192. **Audit of 2026-10-28, Session A: wrong results (2026-10-29; docs/AUDIT_2026-10-28.md).**
    Each with a test that fails on the old code. High 1 and Medium 11: `best_set_K`'s set evidence
    now sums the criterion over the spectra the species table keeps (`_add_evidence`, filled by
    `_spectrum_records`): a spectrum the filters leave out whole adds nothing, as if rejected, and
    one whose K description the table does not keep (not fitted, left out by the K >= 2 filters,
    kept at one population by the screens or by the review) adds its criterion at the K below, so
    it cannot vote for more populations than the table gives it (the difference between K sums is
    unchanged for unfitted K, which the old sums left out). `review_K = 1` now also acts in the
    analysis, by the screens' one-population mechanism (owner, 2026-10-29; supersedes item 152's
    "the analysis never reads review_K"): the K >= 2 descriptions leave the species table, the
    lever-rule comparisons and the set evidence, and left_out_summary.csv counts them. High 2: the
    bundle stores the spectra with the readers' roles; `[run.controls] undeut_at`/`fulldeut_at`
    act again on every run, so a rerun from stored spectra follows the project file as it is
    (bundles made with those settings before this keep the promoted roles in their spectra). High
    6: series that are not corrected (fragment ions) leave the rates of the theoretical
    back-exchange fit. Medium 9: a control replicate whose K = 1 fit failed in every restart (loss
    inf, placeholder parameters) is not used by the controls or the time course (logged). Question
    9: the run's controls leave out the control spectra the loose screens remove
    (`screening.control_removal`; their ids in `ControlTable.screened`, JSON `screened` in
    controls.parquet), so series.csv shows the controls the default analysis uses;
    `analysis_controls` reuses the stored table when its own level removes the same spectra and
    recomputes it otherwise (`screened=None` now means "as the run"). Question 16: sets never
    tested at step one take no part in the residue strength and agreement. Medium 21: a
    population-test null without spread leaves the spectrum untested at that step (logged; it gave
    p = 0 for any statistic above it), and the null's variance is the sample variance (ddof 1; the
    gamma null was 5 % narrow at B = 20). Recomputed from the stored draws of the private
    population-test bundles: benchmark 1 flat cell, 57 of 26,164 spectra (0.22 %) and 13 of 4,721
    sets move, all from two to one; tutorial 4 flat cells, 17 of 3,874 spectra (14 two to one, 1
    three to one, 2 three to two) and 1 of 1,134 sets; p-values below 0.05 rise by a median 1.4
    (spectra) and 1.7-2.0 (sets). Medium 8: `screen_below_mono` continues a lone peak below with
    the peptide's shape scaled to its monoisotopic peak, not its tallest (a 41-residue peptide's
    same-shape contaminant at half height: 0.062 before, 0.211 now, true share 0.318; the rest is
    the score's three-bin reach). Benchmark repicked from its stored spectra with both rules
    (26,333 spectra): loose removes 406 (1.54 %) before and 413 (1.57 %) now, strict 959 (3.64 %)
    and 966 (3.67 %); 772 scores moved, at most +0.10. Not material: the thresholds stay. Medium
    20: the per-bin noise smooths the first and last bins over the two bins there are (the zero
    padding read the first bin of 1000, 300, 50 at 433 instead of 650, which gave that bin, the
    tallest of an undeuterated control, 2.25 times its weight). Re-check of item 191's yardsticks with the
    fix (P9 refits, K = 1 bootstrap n_boot 20, population test B = 20 on the four step-2 bundles;
    V0 unchanged): same K for 98.5-99.3 % of spectra (criterion) and 98.2-99.5 % (test).
    Before / after: replicate-set calibration (robust) tutorial 0.89 / 0.88, noisy set 1.04 /
    0.99, benchmark 1.04 / 1.00; charge sets 1.97 / 1.97, 2.38 / 2.28, 1.58 / 1.54. K > 1 with the
    criterion, labelled / undeuterated / fully deuterated: tutorial 30.3 / 4.3 / 4.2 against 29.1
    / 3.7 / 4.2 %, noisy set 16.9 / 10.6 / 19.4 against 16.8 / 10.7 / 18.4 %, benchmark 24.7 / 7.0
    / 8.6 against 23.4 / 7.0 / 8.3 %; with the test, undeuterated / fully deuterated: tutorial 0.8
    / 3.3 against 0.4 / 3.3 %, noisy set 4.1 / 10.2 against 4.1 / 9.2 %, benchmark 3.6 / 4.4
    against 3.6 / 4.2 %; known-answer correct K 83.5 against 84.0 % (criterion) and 84.8 against
    84.2 % (test). The calibration moved by 0.05 at most, so `model_error` stays 9. Scripts:
    `noise_model/pairs` (private, `built_check_edges.py`, `built_check_edges/summary.txt`).

193. **Audit of 2026-10-28, Session B: crashes and data mix-ups (2026-10-29;
    docs/AUDIT_2026-10-28.md, with the owner's Session B design decisions).** Each with a test
    that fails on the old code. High 3 and Medium 5: a K a review table chooses is honoured: the
    run's screen never skips it (`fit_models(required_k=)`; the K and every K between 2 and it
    are fitted for that spectrum, the selected K unchanged), so it is bootstrapped like every
    chosen K; a table changed after the run that names a K the screen skipped is reported by
    `hxex report` (the remedy named), and `hxex resume hxex.toml` fits it first (`run._fit_chosen`:
    the spectrum's whole series fitted again without the screen, only the missing fits taken; the
    same values as a run with the table), also without `[run.bootstrap]`; a K the run does not try
    at all is refused by `hxex check`, `run` and `resume` before anything is read. A carried-over
    bundle review choosing a K the screen skipped is marked stale (`ReviewTable.mark_stale` tests
    the spectrum's own fit; it crashed at the checkpoint write). Question 11: `review_K = 0` clears
    a K chosen before (`review_table.CLEARED`, `merge_choices`); `inf`, `nan` and `1e400` are
    refused with their row. High 4: where one input file and charge (an HX-Express file: the file)
    is read by several rows of the whole peptide table, their local ids end with `#seriesN` (the
    data_id); every other id is unchanged and a filter never changes an id; `spectrum_index`
    agrees. High 5: `record_paths = "name"` is refused when two runs read files of one name from
    different folders, and a repeated read in which runs that read different files resolve to
    one file is refused. Medium 15: runs without `rep` are numbered 1, 2, ... per sample and
    point (time, control role, or kinetics) in the order listed (`io.mzml.number_replicates`,
    logged); the same `rep` twice at one fixed point is refused in the project file and at the
    Python interface; the series figures fall back to the time-ordered grid (warned) when two
    spectra still share a time and replicate. High 7: uptake.csv carries the measured mean of a
    series the controls leave uncorrected. High 8: `corrected = true` draws such a series
    uncorrected, its axis labelled "uncorrected series". High 9: `load_results` of a zip bundle
    reopens the zip for each raw-spectra request; a loader failure is noted in the spectra panels
    instead of stopping the figure. Medium 1: `[output]` and `[input] parents` are type-checked
    (`tables = "true"` refused with the hint). Medium 2: `hxex report` and `analyse
    --keep-previous` move the earlier folder only once nothing more can refuse the command
    (`write_analysis(before_writing=)`). Medium 3: a project file with a byte-order mark is read.
    Medium 4: `hxex report hxex.toml` writes the tables where `[output] tables` says (`false`:
    none). Medium 6: mzML `referenceableParamGroupRef` parameters are read (also for single scans
    through the index). Medium 7: the m/z check reads centroided spectra by the nearest stick (it
    climbed to a taller isotope stick and gave no offset), and the profile climb is bounded by the
    spectrum's median point spacing. Medium 12: `plot_k_comparison` draws a K the screen skipped
    as the picked spectrum, titled "not fitted (screened)". Question 10: a series taking its other
    charge states' fully deuterated retention (combine_charges or fd_other_charge) that the
    value would put below its own time course is `fd_below_data` with the run's median
    retention; a series whose own control passed its checks keeps it there instead of the
    combined one (owner, 2026-10-29). Question 12: `hxex resume` fills keys the
    stored bootstrap settings lack with their defaults before comparing (logged). Question 13:
    `limit_charge = false` stays a switch. Question 14: a rerun from stored spectra skips the read
    (logged so) and records in `run_settings_used.toml` the read settings of the stored spectra's
    record (`config.stored_read_settings`; the peptide table and targets from the project file),
    with a warning when the project file's read settings differ from the record; and it takes the
    bundle's picked table (with its peak measure and envelopes) instead of picking again when the
    picker, envelope and registry settings, the hxex version, the spectra picked (in order) and
    their control roles all match (`run._stored_picking`; otherwise it picks again and logs what
    differed). Question 17: the population rule travels with the tables: `species_uptake` and
    `compare_peptides` default to the results' rule, `SpeciesUptake.populations` records it,
    `residue_map` follows it and `choose_populations` uses the tiers whenever the residue map has
    them. Question 24: a fragment target that finds no scan only because the scans do not state
    their precursor charge is warned about, once per run. Question 25: a kinetics run whose scans
    lack a scan time is refused.

194. **Audit of 2026-10-28, Session C: tables and figures (2026-10-29; docs/AUDIT_2026-10-28.md).**
    Each with a test that fails on the old code. Medium 10: the controls take no part in any
    per-residue choice or comparison (`ResidueMap.time_points` drops their rows before
    `choose_populations`, `compare_residues`, `compare_populations` (through the choice) and
    `region_summary`); their fits stay in the species and residue tables (ANALYSIS.md "Ground
    rules"). Before, a control's NaN time split each control residue into three partial choice
    rows and gave empty region rows. Medium 13: spectra.csv `narrow_component` is the width
    check's flag for the fit at the spectrum's K (the step that added its last population; never
    set at K = 1), not the last step's. Medium 14: series spectra and population titles follow
    `time_scale = "order"` ("point 60", not "1 min"). Medium 16: uptake.csv has one row per
    series, point, `K` and rank (new column `K`): spectra fitted at different K are no longer
    averaged together. Medium 17: series.csv `controls` reads `uncorrected` for a series the
    controls leave as measured (fragment ions), and the controls overview counts such series in
    their own bar, not in "both measured". Medium 18: `ControlTable` caches its row index and
    `uncorrected` (computed once per table; `row_of`), so the report's per-spectrum controls
    lookups no longer rebuild a map of every series: 0.066 ms per spectrum for the three lookups
    and one `correct_means` at 3,000 series (the audit measured 1.4 ms per lookup); a vectorised
    `correct_means` over all spectra was not needed. Medium 19: the residue comparison, overlay
    and region-bar figures raise their top to 1.05 times the highest point drawn (value plus its
    error) when that is above 1.05, as `plot_residue_profile` does; fD is not clipped at 1.

195. **Audit of 2026-10-28, Session D: the Low list, the acetylated N-terminus, the kint check
    and two measurements (2026-10-29; docs/AUDIT_2026-10-28.md).** Each change in behaviour has
    a test.
    - Settings and command line: `[input] data_dir = false` is refused as missing ("needed for
      FORMAT") instead of a traceback in `hxex check`; `peptide_region = false` unsets it as
      the other list keys do; the template's stale header clause on inherited control times is
      gone; `hxex analyse NAME=results.zip` keeps the name `NAME` (the single-bundle shortcut is
      for a bare path); `hxex check`'s outputs block looks for `runs_analysis[_REF]` beside the
      first run with `[analysis.runs]`, as `hxex analyse` writes it, with its state from each
      run's stamp; a quoted review-table note may hold a line break; a spectra file whose read
      record is not JSON is a ReadError (ProjectError in a project) naming the file;
      `hxex run` refuses an `output.bundle` that exists but is not a bundle before the read
      (it was refused after it); `hxex resume --tables` applies the bundle's rule to a tables
      folder holding other results' tables (before the bootstrap; the bundle's own tables are
      brought up to date as before).
    - Readers and bundles: manifests, listings and spectra-file metadata are strict JSON (a
      non-finite number, such as `resolving_power_mz` of an unknown instrument, is written as
      null; older files with NaN still load); a zip bundle is staged as `NAME.zip.tmp-*.zip`
      (it was `NAME.zip.zip`, replacing a file of that name); `set_review` writes back the
      manifest as read with its files updated (a legacy bundle keeps its control-time keys);
      `parse_time` takes NumPy numbers (not NumPy booleans); the HX-Express label `.5 min` is
      30 s (pyHXExpress fails on it: `float('')`), so it left the test's error cases;
      `ThermoFile` after `close()` raises ThermoError; output listings write `/` between
      folders on every system (and read the `\` of older Windows listings), and a path outside
      the folder is logged and left out of the listing rather than raised from the `finally`
      that writes it; the retention-time clash names both rows; a SpecExport time that does not
      parse names its file.
    - Big mzML files: a finished `spectrum` or `chromatogram` element is removed from its list
      as well as cleared, a spectrum's metadata is parsed once (its arrays decoded only when the
      scan is kept), and `MzmlFile.read_scans` reads through the file's index when it has one
      (each spectrum read up to the next one's offset). Measured (private top-down files;
      warm cache; spectra byte-identical before and after): a 1.1 GB file without an index
      reads its scan table in 1.63 s before and 1.64 s after, 210 scans in 2.45 vs 2.46 s,
      1,260 in 6.65 vs 6.67 s, peak memory 172 MB both (the XML parse dominates; `clear()`
      alone already freed the content); a 149 MB indexed file reads 68 scans in 0.73 s before,
      0.51 s after.
    - Picking, chemistry, fragments: `PickerConfig` refuses NaN for `mass_accuracy_ppm` and
      `resolving_power_exponent`, booleans for the integer settings and 1.5 for
      `shoulder_min_points` (NumPy integers accepted); a zero threshold (`noise_percent = 0`,
      or an all-zero spectrum) no longer makes empty bins signal, so `capped` is not set on
      every spectrum and an all-zero spectrum is `no_signal` as well as `empty` (default
      settings: unchanged; picker golden and oracle tests pass); non-finite intensities are read
      as 0 before picking (both pick modes); a charge-1 parent whose fragment charges `ALL`
      leave none (limit_charge) is a warning, not silence.
    - Controls and run: the `fd_multimodal`/`m0_multimodal` majority counts the replicates
      used (a replicate `replicate_outlier` left out no longer counts); a control replicate left
      out as a noise outlier is noted `m0_replicate_left_out`/`fd_replicate_left_out` as the
      owner decided (outputs.md's series.csv and noise_outlier rows say both causes); a spectrum
      without a bin is dropped as `no_bins`, not `no_signal` (run and resume, the log line).
    - Analysis: `difference_trace` pairs fragments of one parent only (key gains `parent`,
      `parent_start`); the population comparison (`_lever_sides`) takes each replicate's K = 1
      over every charge state that has one, as compare_peptides.csv does, and the K = 2
      positions over the charge states with both (before, a charge state without a K = 2 fit
      left the K = 1 side too, so the two tables' K = 1 could differ; population comparisons
      change slightly where charge states differ in K); `PeptideComparison.n` is replicates
      (docstring and outputs.md); a time point promoted by `[analysis.controls] undeut_at`/`fulldeut_at`
      leaves the labelled points of the species and residue tables (and the replicate-outlier
      filter's time points), not only the control fit.
    - Report: the figures folder's pre-check lists the overviews' own names
      (`overview_k_selection.png`, ...; `overview_files`), so a file of yours of that name is
      refused before anything is written; the k-selection axis reads "time point";
      `residue_comparison_figure` without a sample besides the reference raises as the overlay
      does; `show_series(k=K)` and `series_figures` refuse an unfitted K before drawing.
    - Engine: `fit_models`' progress total counts a k-means warm start only for a K warmed
      from a K that is chosen from (not from the controls' extra K = 1), so it ends at its
      total; `ControlsNoise` compares NaN as equal (and hashes so), so merging chunks with
      separate copies works; a noise scale that is not positive and finite (a reference
      without intensity) is a FitError naming the spectra, in `fit_models` and `prepare`,
      instead of a fit that is NaN in every restart. GPU parity and the full regression rerun.
    - Acetylated N-terminus (owner's decision): `nterm_subtract` applies to a free amine only.
      After `Ac-` residue 1 keeps its amide and none is subtracted (also with `nterm_subtract`
      2 or more), in `amide_positions`, `count_amides` (peptides and N-terminal ions), and
      everything built on them (bins, shapes, kint's `series_rates`, which now uses kint's rate
      for residue 1), `exchanging_residues`, the ladder notes, and the residue tables' `no_amide`
      (residue 1 has an amide where an `Ac-` peptide starts the protein). A C-terminal fragment
      of an `Ac-` parent carries the cap in its modification (`parent_nterm:Ac-`, beside
      `parent_pos`; written only for such fragments, so other strings and ids are unchanged),
      so with `nterm_subtract` 2 it holds residue 2's amide as its parent does. The evidence
      (asked for by the owner: `nterm_subtract` is meant to leave off residues almost certainly
      back-exchanged): kint at pH 2.5, 0 C, 5 % D2O over 2,000 random 8-mers gives residue 2
      after a free amine a median half-life of 4 min (19 % of its D kept after 10 min), an
      interior residue 95 min (93 %), residue 1 after `Ac-` 62 min (89 %) and residue 2 after
      `Ac-` 98 min (93 %): the fast loss comes from the free NH3+ and goes with the cap, so
      nothing is subtracted after it (owner agreed; values other than 1 and 2 are not used in
      practice). PLAN.md "Deliberate differences" records it.
    - Scans without a scan start time (owner, 2026-10-29, after Session B's question 25): refused
      in every mzML run, not only kinetics runs; a run with a fixed time used to drop such scans
      by its retention window (or keep them without one) without a word. Guide inputs.md says
      so. No private project has an `Ac-` peptide, so no existing result
      changes; the template's and settings.md's `nterm_subtract` line says it.
    - Question 15: `hxex check` counted every row of a folder format's peptide table; it now
      filters them by `samples`, `peptide_ranges` and `peptide_region` as the read does.
      Question 26: SpecExport cuts each peptide from the FASTA by residues (modX), keeping a
      terminal group of the FASTA sequence on the peptide at that end; a letter outside the
      registry (an `X` in the FASTA, say) falls back to the cut by letters, so plain FASTA
      reads as before. Question 27: kept; PLAN.md's reader bullet cites it, and a test reads a
      sheet with a spacer column.
    - Question 23, kint verified against the sources (web research, nothing sent but searches):
      every terminal rule agrees. Bai et al. 1993 Table II gives the free N-terminus AR -1.32,
      BR 1.62 on residue 2 (measured against an acetyl-blocked dipeptide, so leaving it out
      after `Ac-` is right) and the C-terminal carboxyl AL 0.05 (COOH) / 0.96 (COO-) and BL
      -1.8; the Englander lab's rate spreadsheet blends the acid factor with Glu's pKc and
      adds both terminal factors to a dipeptide, as hxex does (pacilab and HDXrate add only one).
      The acetyl factor is in no paper: the spreadsheet's row "H3CCO-" has AR 0.293, BR -0.197
      as residue 1's preceding group (hxex rounds to 0.29, -0.20; 0.003 log units). No factor
      exists for a C-terminal amide; the spreadsheet's N-methylamide row is a separate
      reference, so "no factor, approximate" stands. hxex's Gly BL -0.03 and Asp/Glu (COO-) BL
      0.10/-0.11 are the later (Mori 1997, Nguyen 2018) values, not Bai's. Golden test:
      `test_terminal_groups_take_the_published_factors` solves the conditions' acid and base
      coefficients from a free peptide and checks the acetyl, N-terminal and C-terminal-amide
      rules on `Ac-AAAA-NH2` and `Ac-GG-NH2` to 1e-9.
    - Question 20, measured (private top-down example file; private
      analysis_examples/topdown_widths): the question's premise does not hold there. The file's
      recorded resolving power is 120,000 at m/z 200 (about 53,400 at m/z 1,010), so peaks at
      charge 20 are 2.7 FWHM apart and resolved; `peak_fwhm` reads 18.2-18.5 ppm against 18.7
      expected, and the half-height walk crosses no valley on the tallest peaks (0 of 443 scans;
      weak peaks 1 in 26,399 walks). `auto` chooses heights (the file has no undeuterated
      controls; with every spectrum taken as one the width ratio is 1.02 on single scans and
      1.09 on sums of 20, the rise from the scan-to-scan m/z scatter of weak peaks, not merging);
      heights and areas differ by 1.5 % in envelope shape (+0.2 D centroid). Two equal peaks
      merge at half height below about 1.4-1.5 FWHM spacing (R below about 30,000 at m/z 1,000
      for charge 20). No change and no guide sentence: no unresolved data to test it on.
    - Question 22, measured (private analysis_examples/bic_sample_size; 11 bundles, recomputed
      from the stored nll, reproducing every stored BIC choice): leaving the leading and
      trailing zero bins out of N (the trailing ones are `zero_filling`; median 0-22 % of N, q90
      11-44 %) changes the BIC's K in 0.1-1.0 % of spectra, always upward (a smaller penalty):
      benchmark 0.36-0.52 %, tutorial 0.08-0.15 %, noisy unimodal set 0.68 %, Orbitrap
      0.46-1.04 %; controls 0-1.4 % (Orbitrap FD 31 of 2,228). The known-answer set gains
      slightly (79.6 -> 80.1 % and 74.9 -> 75.7 % correct K on two bundles). Then tried in the
      code (owner: "try it"): three projects rerun from their stored spectra with N trimmed
      (identical fits; only the choice moves, as recomputed): labelled K > 1 known answer
      77.6 -> 78.0 %, tutorial 26.4 -> 26.5 %, benchmark 20.2 -> 20.6 %; undeuterated controls
      13.0 -> 13.0, 11.6 -> 11.8 and 6.0 -> 6.2 %; known-answer correct K 79.6 -> 80.1 % (2 of
      387). REVERTED (owner, 2026-10-29): the zero bins at the ends have meaning (the fit must
      explain them), so they count in N; the gain is within noise and the controls split
      slightly more often.

196. **Fully deuterated controls: width, deamidation, and better stand-ins (2026-10-30; private
     analysis_examples/rho_fd, README with every number; noise simulation phase 2 in
     analysis_examples/noise_sim).**
    - Noise-simulation phase 2 (scratch): baseline zeroing and the saturation form tried change
      nothing; picked noise in empty bins raises false two in dim spectra under the per-bin
      noise (5 -> 24 %); envelopes wider than a binomial are the largest effect (intra-class
      correlation 0.03: 58-74 % of single-population peptides with 16+ amides split, the
      population test keeping most; 0.01: almost nothing).
    - Measured on real FD controls (PLAN 15; n fixed at the site count, p and rho fitted; the
      method recovers rho 0 / 0.01 / 0.03 on simulated controls): the typical FD envelope is as
      wide as the binomial (median rho 0, upper quartile ~0.01) on four Q-TOF datasets; 5-27 %
      of controls are over 10 % wider; the width belongs to the peptide (repeats across
      replicates and charges, grows with charge); on an Orbitrap digest the envelopes are
      narrower (median 0.85). UN controls match their natural shape. Wider FD controls go with
      more two-population readings in labelled spectra, at every deuteration level, so not by
      width spread over the elution.
    - Deamidation: in one protein (two datasets, different protocols) the wide FD controls are
      the peptides with its single Asn-Gly site. Its FD controls are heated (urea, 90 C,
      phosphate pH 7.5) and N -> D adds 0.984 Da, inside a deuterium bin. A +1-bin share fits
      0.64 / 0.58 in every such peptide and charge, ~0 elsewhere; confirmed by mass (the FD
      controls of those peptides read 3.0 / 3.1 ppm below their UN controls, expected 4.3 / 2.9);
      the protein's other Asn (followed by Leu) normal. Their FD retention is overstated (~5 %).
      The shift share alone is no detector (it fires on width: a merely wider envelope reads
      0.6 in simulation; its baseline is +0.35 in another dataset). A charge-linked M+1 tilt in
      UN controls (HISTORY 186) briefly read as stock deamidation: withdrawn. Design (opt-in,
      relative share per sample and mass, correct the own control by the share; not built):
      docs/ANALYSIS.md "Deamidated fully deuterated controls".
    - The run-median stand-in overcorrects peptides that retain more than the median (nested
      pairs: up to 1.5 D per extra amide). Leave-one-out on every accepted FD retention, median
      |error|: run median 0.040-0.054; the same peptide in the run's other samples 0.006-0.012; a
      retention map of overlapping peptides (sequence-checked, built locally per series)
      0.022-0.032 for a peptide measured in no sample. BUILT (owner: default on, switchable,
      this order): own -> other charges -> `fd_other_sample` -> `fd_retention_map` (`fd_map`)
      -> kint -> median, each stand-in checked against the series' time course;
      `hxex.controls.series_residues`; neither new stand-in counts as measured (owner: only
      expand that pool if needed). Effect: benchmark 182 median stand-ins -> 71 other-sample, 98
      map, 13 median (retention changed by 0.093 at the median); noisy set 19 -> 15 / 3 / 1;
      tutorial none. The map's place before kint is not tested (no dataset with quench
      conditions).
    - Both stand-ins are still stand-ins for the analysis filter (`stand_in_controls`), so by
      default those series stay out of the analysis. New level (owner, option 1):
      `stand_in_controls = "other_sample"` keeps the `fd_other_sample` series (as close as the
      peptide's charge states) and leaves out the map and the run-level stand-ins; the default
      stays "exclude". The protection fit's start is unchanged (every stand-in fits fD0).

197. **Deamidated fully deuterated controls built, from the mass alone (2026-10-30; private
     analysis_examples/rho_fd, README "Build check").**
    - Correction to 196: the "+1-bin share 0.64 / 0.58" is not a deamidated share. A binomial of
      n sites mixed with a share p of itself one bin up is exactly a binomial of n + 1 sites, so
      the shift-share fit returns phi = p whenever one more site fits better (any envelope wider
      than a binomial); the evaluation's optimiser started at p 0.6. Searching every p, the
      built fit gives other optima (benchmark 0.12-0.19 relative share, with uniform or per-bin
      weights). The envelope share is dropped; evaluations 1-2's share column was this artefact.
      The mass test stands.
    - The share from the mass: per series, the FD minus UN picked-peak offset (each relative to
      the other peptides of its injection) over the shift of a fully deamidated copy
      (-0.022 Da / M); per sample, the Asn-Gly series' median minus the other series'. Benchmark
      0.33-0.49 (se 0.06-0.13), tutorial 0.37-0.57 (se 0.08-0.13), noisy set -0.10 to +0.07,
      Orbitrap digest 0.00 (se 0.02); other series' median within 0.02 of 0 everywhere; a
      same-size null's 99th percentile 0.02-0.23. Nested pairs (evaluation 3) with the mass
      share removed: 20 h 0.67-0.91, none above 1 (within 0.03 of the envelope version).
    - BUILT (owner, 2026-10-30): `hxex.deamidation`; `[run.controls] deamidation` (default
      false) and `deamidation_min_share` (0.25); candidates Asn-Gly, or a C-terminal Asn with
      Gly next in an overlapping peptide of the same sample (not FASTA: the run has none);
      a deamidated sample's Asn-Gly FD control spectra are lowered by the share before every
      check, replicate checks included (a control just above its amide count only because of
      the deamidated share is kept), as whole bins (the fit's unit; 0.984 would leave 1.6 % of
      the shift); flag `fd_deamidated` (a note, still measured); `series.csv`
      `deamidated_share`; the assessment in the controls table's metadata; also in
      `[analysis.controls]`. Not assessed: tables picked elsewhere (no offsets), samples with
      fewer than two Asn-Gly or two other series with both controls. Dropped from the design:
      `deamidation_min_mass` and `deamidation_not_confirmed` (one measure now). On the
      benchmark every Asn-Gly series of the five samples is corrected (median retention 0.67
      -> 0.65), the others unchanged except five map stand-ins (under 0.0001); tutorial 0.60
      -> 0.58; 0.4 s per bundle.

198. **`pick = "peak"` keeps the ppm window; a resolving power goes with `window_max` only
     (2026-10-30; private reference_runs/48abc2c, README "Orbitrap: recorded window").**
    - The rerun of the Orbitrap digest with `resolving_power = "recorded"` (R 45,000 at m/z 200)
      against `resolution_ppm = 10`: empty spectra 167 -> 18, but undeuterated controls with
      K > 1 21 -> 48 %, labelled spectra 13 -> 27 %, K = 3 2.3 -> 6.9 %; K = 1 uptake against
      the predicted values r 0.902 -> 0.802, more than 0.1 off 11.6 -> 13.5 %.
    - The arithmetic is right (R read per scan, m/z 200 for the model, FWHM m / R(m), square-root
      scaling). Two things are not what a user who knows R would expect. The recorded R is the
      acquisition setting: 735 tall undeuterated peaks were 1.52-1.54 times narrower than it
      implies at every m/z (21.5 ppm at m/z 300-500 to 34.0 ppm at 1000-1400); the run's own
      width measurement (`peak_measure`, 32.8 ppm at m/z 1000) agrees with the peaks. And
      `window_fwhm = 1` is a half-width of one FWHM, a window two FWHM wide: with the nominal R,
      tops up to 1.5 real peak widths away were accepted, which are resolved neighbouring peaks.
    - One FWHM (item 98) was sized for `window_max`, which needs the window to cover the peak.
      With `pick = "peak"` (item 172) the window only accepts or rejects an apex, so it is a
      mass tolerance. The apex scatter about the predicted m/z, after each file's median offset,
      does not follow m/z: robust sd for peaks at 30-100 % of the tallest, Orbitrap 2.3-2.8 ppm
      undeuterated and 1.6-2.3 labelled, Q-TOF 6.0-6.9 and 5.7-7.0, flat from m/z 300 to 1600
      although the Orbitrap's FWHM grows by half; peaks at 5-30 % scatter 8-20 ppm (part of it
      neighbours caught by the wide windows). On the Q-TOF most of the 6 ppm is the series' own
      offset (within an envelope 1.5-3 ppm), which the m/z check (item 166) handles.
    - BUILT: `PickerConfig` refuses `resolving_power` with `pick = "peak"` (the message names
      `window_max`); a stored record without `pick` that sets a resolving power reads as
      `window_max` (records before item 172 were picked that way; one with neither still reads
      as the defaults); `hxex check`'s hint and the template comment name `window_max`.
      Unchanged: the window of `pick = "peak"` and of every result made without a resolving
      power. Guide picking-window.md (the ppm window is the peak pick's window, with the
      measured scatter; the resolving-power sections marked `window_max` only, with the
      reasons), experiments.md (top-down: the cap is enough with `peak`), install.md, inputs.md;
      PEAK_PICKING.md, PLAN. A bundle made with the combination (the private recorded-window
      rerun) no longer loads.
    - Open: whether the m/z check's M/M+1/M+2 agreement tolerance (5 ppm) should follow the
      measured peak width, so the 18 Orbitrap peptides left empty by the ±10 ppm window
      (15-40 ppm off) are recentred.

199. **A high `screen_beyond_reach` removes the spectrum (owner: "try the build", 2026-10-30;
     private reference_runs/48abc2c, README "screen_beyond_reach (R4): sizing a removal
     level").** Until now R4 (picked intensity beyond the peptide's reach) only warned (0.10 at
     both levels, item 152: no measurable effect on the benchmark's fits). The owner's review of
     an Orbitrap digest's time courses found series whose late spectra and fully deuterated
     control are mostly another species (R4 0.71-0.79), kept with a warning, the control still
     setting the series' correction.
    - Share of spectra: benchmark >= 0.3 0.5 %, >= 0.5 0.1 %; tutorial 0.2 / 0.1 %; noisy set and
      known-answer set 0; Orbitrap digest 4.1 / 2.2 % (>= 0.7 1.2 %).
    - High values go with wrong numbers. Benchmark labelled spectra, centroid more than 1 D from
      the median of its replicates: R4 < 0.1 1.5 %, 0.3-0.5 20 %, 0.5-0.7 58 % (n 12), >= 0.7
      78 % (n 9). Orbitrap, corrected K = 1 more than 0.2 of FD off the paper's predicted uptake:
      2.9 %, 10.9 %, 13.6 %, 40.2 % (n 17,573 / 385 / 199 / 239); replicates there do not show it
      (the other species co-elutes in every replicate). The tutorial has too few high values.
    - BUILT: `screen_beyond_reach` removes at 0.5 (`loose`) and 0.3 (`strict`), the warning from
      0.10 kept below that. A removed control is rejected (stand-in), and `hxex run` leaves out
      the controls loose removes, as for the other scores; existing bundles take it in the
      analysis (the screened controls differ from the stored set, so the correction is
      recomputed). Removed at loose: benchmark 26 spectra (0.10 %, 3 controls), tutorial 4,
      noisy and known-answer none, Orbitrap 514 (2.05 %, 74 controls); strict 105 / 6 / 0 / 0 /
      735. Guide outputs.md (table and paragraph), ANALYSIS.md, PLAN.
    - Also measured that day (not kept): find_peak's retention trace from peak tops in the
      window (as `pick = "peak"`) instead of the tallest point. It fixes a series whose runs
      flipped between two co-eluting species 40 ppm apart (every run then takes the one at the
      prediction), but runs agree less overall (series whose chosen retention spans more than
      0.3 min across 45 runs 7 -> 25), mostly series the m/z check moves 10-36 ppm: the old trace
      followed their envelope's flank consistently, the new one finds nothing real at the
      prediction. Which species a series follows is the identity question (a fragment-ion check
      on the dataset's DDA runs proposed).
