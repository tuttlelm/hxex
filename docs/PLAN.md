# Project plan: a successor to pyHXExpress with a batched binomial-mixture engine

Status: draft, revised through clarifications. Phase 0 scaffold created 2026-10-09 (see
"Phase 0 status").

## Goal and scope

A standalone, pip-installable package that goes from spectra (raw or exported) to binned
sticks to binomial-mixture fits with uncertainties and reports. Its fitting engine is the
one built and tested in this repository (`fit_multimodal_binomial.py`). It is meant to
replace pyHXExpress for new users while staying comparable with it.

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
- **Standalone.** The package does not depend on pyHXExpress. Functions are rebuilt case by
  case, with the existing code as specification and test oracle.
- **CPU is a first-class target.** The GPU is an optional speedup.
- **First readers:** HDExaminer SpecExport and HX-Express formats, as implemented in
  pyHXExpress. Possible later readers: mzML and Thermo raw.
- **Adoption goals:** a good install story, documentation, tests and stable outputs.
- **pyHXExpress is mainly the author's tool,** so compatibility is wanted for comparison
  and migration, not to protect an installed user base.
- **Saved results:** a self-contained results bundle, readable only for bundles written by the
  new package at first (a legacy pyHXExpress loader is a later goal).
- **Raw spectrum storage:** Parquet (zstd), one row per spectrum with list columns, with flat
  NumPy arrays plus a typed index in memory and no pandas DataFrames in the core. `pyarrow`
  is the dependency for the bundle (see "Raw spectrum storage").
- **License: GNU GPL v3 or later (`GPL-3.0-or-later`).** The `LICENSE` file is the standard
  GPL v3 text of 29 June 2007, copied from pyHXExpress's `LICENSE.txt`. pyHXExpress itself
  declares plain GPL-3.0 (its README makes no statement and it has no per-file headers), so
  hxex differs from it by the author's choice: the "or later" option is added in
  `pyproject.toml`, the README and an SPDX header in every source file. Every runtime
  dependency was checked and is compatible with GPL v3 (see "Phase 0 status"). Thermo's
  raw-file reader, if added later, is under Thermo's own license terms, which need review
  before it is bundled or required.
- **Minimum Python version: 3.11** (`requires-python = ">=3.11"`), with CI on 3.11 to 3.14
  (and 3.15 once it is final). Python 3.10 reaches end of life in October 2026, and PyTorch
  2.15 (scheduled for late October 2026) requires 3.11 or newer. Phase 0 must confirm that
  `pyteomics`, `pyarrow`, the isotope-distribution library and PyTorch all install on 3.11 to
  3.14, and the current development environment (`.venv`, Python 3.10) needs rebuilding on
  3.11 or newer.
- **Saving all fits:** every K = 1 .. K_max fit is saved by default; individual bootstrap
  fits are an option and are kept for the **selected K only** (see "Saving all fits").

## Reference implementation

`hxex_updating.py` is the reference for the behaviour of existing functions (readers, peak
picker, chemistry, metadata handling). The other copies in the pyHXExpress folder
(`pyhxexpress.py`, dated backups, `hxex_updating_26mar2025.py`, ...) are not references.
`hxex_fragments.py` is the reference for fragment-table generation.

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

## Requirements

### Functional

- **Readers** return spectra (m/z, intensity, plus sample, peptide, charge, time, replicate,
  file). Time labels and file-name conventions are parsed in one isolated, tested place.
- **Peak picking** converts a spectrum into binned deuterium sticks at predicted positions
  (ExMS spacing, ppm window, shoulder rejection, zero-filling).
- **Chemistry:** m/z, elemental composition, exchangeable amide count and natural-abundance
  shape from sequence (including PTMs such as `acK`, `pS`, `pT`, `pY`; see "Sequence syntax
  and PTMs"), charge, modifications **and ion type**. The default ion type is the
  intact peptide (`M`); other types (a, b, c, x, y, z and the variants pyteomics defines,
  such as `c-dot`, `z+1` and neutral losses) are selectable. pyteomics is the reference for
  compositions and masses (see "Ion types" below).
- **Fitting:** K = 1 to 3 populations with BIC selection, user-set sigma (absolute or as a
  percentage of a reference peak), a cap on exchanging amides, robust handling of
  non-finite fits.
- **Controls:** user-set undeuterated and fully deuterated times, replicate averaging, and
  flagged fallbacks when a control is missing or bad.
- **Uncertainty:** bootstrap CIs on means and population fractions.
- **Outputs:** tidy result tables with quality flags, and per-sample plots with error bars.
- **Saved results and replotting:** a run saves a self-contained results bundle. Previous
  results can be read back to generate plots of the raw data and the fits (and the results
  tables) without refitting; see "Results bundle and replotting".

### Non-functional

- Runs unmodified on CPU or GPU.
- Reproducible: every run records its configuration and package versions.
- Pure functions with explicit config objects, no global state.
- Documented and tested, with example data.
- **GUI-ready from the start** (the GUI itself comes later; see "GUI"): serialisable
  configuration, progress and cancellation hooks on long operations, `logging` and structured
  exceptions instead of `print`, plot functions that return figures, and lazy loading of
  results.

## Architecture

### Data contracts (the stable interfaces between stages)

- `Spectrum`: arrays plus metadata.
- `BinnedSpectrum` / the binned table: one row per bin, using the same columns as
  pyHXExpress's `alldeutdata`, so results remain comparable.
- Results tables: components, per-measurement quality flags and CIs.

### Modules

- `chem`: amide counting and isotope shapes.
- `io`: the readers behind a small plug-in interface (path and metadata in, spectra out),
  so new formats are additions; plus the results-bundle writer and loader.
- `picking`.
- `model`: the mixture, loss and shapes.
- `fit`: initialization, optimizer, nested K start, NaN protection.
- `noise` and `controls`: sigma reference selection and the back-exchange correction.
- `bootstrap`.
- `run`: chunking by device and orchestration.
- `report`: tables and plots.
- `cli`, plus a TOML or YAML configuration file.

The binned table is the interchange point. Each stage can run independently, so users can
start from their own sticks.

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

## Sequence syntax and PTMs

Sequences may contain modified residues written in pyteomics modX style (lowercase prefix
plus residue letter). `hxex_updating.py` supports:

- `acK`: acetylated lysine (K + C2H2O)
- `pS`, `pT`, `pY`: phosphorylated serine, threonine, tyrosine (base residue + HPO3)

How pyHXExpress does it: at import it edits pyteomics' **global** residue table
(`mass.std_aa_comp['acK'] = ...`). Without that edit pyteomics rejects such sequences
(verified).

How the new package should do it (verified to work with pyteomics):

- A **private, extensible modification registry** (label -> composition change), with
  `acK`, `pS`, `pT`, `pY` built in and a way for users to add their own. It is passed to
  pyteomics through its `aa_comp` argument (composition, mass and m/z all work), so the
  package never edits pyteomics' global state.
- **Residues are tokenised, not counted as characters,** everywhere: amide counting,
  fragment slicing, N-terminal exclusion. (`pyteomics.parser.parse` already tokenises
  `acK`, `pS` as single residues.)
- The metadata `modification` column of pyHXExpress (composition changes such as `H:1`, a
  `Hex` count, `ion_type:c`, an explicit `undeut_mz`) is a second, separate mechanism. It is
  kept with the same syntax at first for compatibility.

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

Open questions:

1. Which PTMs and terminal modifications are needed in the first release beyond `acK`,
   `pS`, `pT`, `pY` (for example N-terminal acetylation)?
2. Should modified residues change the exchangeable amide count? pyHXExpress currently does
   not (the side-chain term is off by default).

## Ion types (fragment ions)

Added after review: the chemistry must depend on the fragment ion type, not only the peptide
sequence.

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

Plan:

- All of `chem` takes an ion specification (sequence, charge, ion type, modifications), so the
  mass, composition, amide count and shape are consistent with each other.
- pyteomics is the reference: tests compare composition and m/z for every supported ion type
  against pyteomics, and shapes against an independent calculation.
- Fragment-table generation (the `hxex_fragments.py` behaviour) is a separate module that
  produces metadata rows; it can follow the core chemistry.
- Binned-table metadata carries `ion_type`, the parent peptide and the fragment index, as
  pyHXExpress's `parent` and `frags` columns do.

Open chemistry questions for the domain owner:

1. How the exchangeable amide count depends on ion type. **Unverified; see "Risks and open
   decisions".** c and z ions, for example, differ from b and y in which backbone amide
   hydrogens are present, and the N-terminal subtraction rule (`Nterm_subtract`) may apply
   differently to N- and C-terminal fragments. pyHXExpress currently does not distinguish them.
2. Which ion types are needed for the first release.
3. Whether the shape should follow the ion-type composition (as planned), or stay compatible
   with pyHXExpress's current behaviour by default.

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

## Testing and validation

1. **Golden tests:** compare the new readers, picker and shapes with `hxex_updating.py`
   outputs on the Bimodal_HDX_Data test sets, plus sequences with PTMs (`acK`, `pS`, `pT`,
   `pY`) where the two agree. For ion types, compare compositions and m/z with
   pyteomics for every supported type; where pyHXExpress ignores the ion type (shape, amide
   count), the new behaviour is checked against the chemistry, not against pyHXExpress.
2. **Regression tests:** freeze the current 542-measurement results as a regression suite
   for the engine, within tolerance.
3. **Simulation tests:** known mixtures, NaN injection, CPU/GPU parity.
4. **Real-data validation:** replicates and charge-state pairs give a direct check on
   interval calibration. If the CIs are right, replicate means should agree within them
   about 95% of the time.

## GUI

A GUI is wanted at some point. Decision (recommendation, to be revisited): **build the library
and the command-line tool first and add the GUI later**, as another client of the same
library. The stages already have stable data contracts (spectra, binned table, results bundle,
a `report` module that plots from loaded results only), and the hard parts of the project are
the chemistry, picking and fitting engine, which do not depend on a GUI. Starting from a GUI
would slow that work and tends to bake interface assumptions into the data model.

Rules adopted now so that adding a GUI later is easy:

1. No global state; explicit config objects (already planned).
2. A serialisable configuration (TOML or JSON, generated from dataclasses) that a GUI can edit
   and save (already planned).
3. **Progress and cancellation hooks** on every long operation (a callback or iterator with a
   cancel flag). A GUI cannot freeze for many minutes with no stop button, and retrofitting
   hooks later means touching every long-running function.
4. Logging through the `logging` module and structured exceptions; no bare `print`.
5. Plot functions return figures and never call `show()`, so they embed in any window or save
   headless (already planned for `report`).
6. Lazy loading of results (Parquet row groups) so a GUI can browse thousands of fits without
   loading everything.

**Review layer.** In HDX-MS the main value of a GUI is usually interactive review and
curation. The results bundle therefore carries per-spectrum flags and notes (accept or
reject, comments, manual overrides such as a chosen K) that need no refitting. The
command-line tool and notebooks use the same layer, and it is cheap to add to the schema now
and awkward to retrofit.

Technology options, to be decided when the GUI starts (general trade-offs, not tested here):

- **Qt desktop (PySide6):** the best interactivity (zooming, picking, linked plots). Qt for
  Python is available under the LGPL, which fits GPL v3. It is a large dependency, so it would
  be an optional extra (`pip install hxex[gui]`).
- **Browser-based (Panel, NiceGUI, Streamlit and similar):** faster to build and well suited to
  running on a remote GPU machine.
- **Notebook widgets:** the lightest option and the closest to the existing notebook workflow.

Sequence: the first GUI is a **results viewer and review tool** (open a bundle, browse spectra
and fits, set review flags), after Phase 4. A run launcher (choose files, set the configuration,
start and monitor a run) comes after that.

Integrating the GUI from the start would be better only if the primary workflow is interactive
manual curation during analysis (for example hand-placing peaks or re-picking controls); then
the review state would shape the core. If most users run batch analyses and then inspect the
results, library-first is right.

## Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Setup | New repository under version control, packaging, CI, license and naming, compatibility contract for pyHXExpress files | Installs and tests run on CPU |
| 1. Input | `chem` (with ion type, default `M`), picker and the two readers (SpecExport and HX-Express) | Binned tables match pyHXExpress on the test sets; ion-type chemistry matches pyteomics |
| 2. Engine | The current script split into modules with explicit config objects; the results-bundle schema | Reproduces the frozen results; passes CPU/GPU parity; save/load round trip is exact |
| 3. Controls and results | Controls, noise reference, the back-exchange method, quality flags | Method decided on real data; flags documented |
| 4. Scale and reporting | Chunking by device, bootstrap, plots from saved results, tables | A 10,000-fit run within a measured time on stated hardware; replots from a saved bundle match direct plots |
| 5. Validation and release | Real-data replicate check, docs, examples | Release 0.1 |
| 6. Extra input | Fragment-table generation, mzML, then Thermo raw; better peak picking | Each has golden tests |
| 7. GUI (after Phase 4) | Results viewer and review tool; later a run launcher | Opens a saved bundle, browses fits, saves review flags without refitting |

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

Implemented 2026-10-09 (not committed at the time of writing): `modifications` (private,
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
compare against whichever is installed). Not done in `chem`: fragment-table generation (Phase
6), peak picking (next in Phase 1).

## Risks and open decisions

- **UNVERIFIED: exchangeable amide count for fragment ions.** Every ion type uses the pyHXExpress
  rule (`AmideRules` offsets are all 0), which is only expected to hold for `M`, `b` and `y`.
  The values for `c`, `z` and the other types are unconfirmed and must not be treated as
  validated. Planned checks: derive the exchangeable N-H groups per ion type from structure
  and record them as tests; check the ETD/ECD-HDX literature conventions; and compare the
  predicted and measured maximum deuteration of fragments against a fully deuterated control
  (needs fragment data; none is in the test set, which is intact peptides only). Deferred by
  the owner until the ETD data is ready to be examined.
- **Back-exchange correction is unresolved.** On the simulated test data the intermediate
  points looked less back-exchanged than the fully deuterated control. It needs real data
  before it becomes a default.
- **Sigma guidance.** Sigma is user-supplied, but users will need a recommended way to
  estimate it, probably from baseline bins. The peak picker's noise threshold should
  connect to it.
- **Dependency footprint.** PyTorch is large. CPU-only wheels help, but an install story for
  Windows and Mac users needs testing.
- **Thermo raw** needs a vendor library or a third-party reader, with licensing to check.
- **CPU time for large bootstrap runs** may need user guidance (smaller N, fewer restarts).
- **Memory and plots at 10,000+ fits:** figures must be selective by default.

## Decisions needed

1. The remaining Phase 0 items above (initial commit, creating the GitHub repository, reserving
   the PyPI name, confirming the SPDX identifier), and what to start in Phase 1. (Name,
   repository, hosting, license and minimum Python version are decided.)
