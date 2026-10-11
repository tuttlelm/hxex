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
all-zero spectrum is flagged `empty`. `tests/golden/picker_reference.npz` (11 spectra covering
the cases above, made by `tests/golden/make_picker_golden.py`) is committed; the whole-run check
reads a run directory from `HXEX_PICKER_ORACLE`. The 19Feb2024 Bimodal `alldeutdata` came from
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

## Next steps

1. **Phase 3 (controls and results), in progress: controls, the correction, the quality flags
   and their bundle tables are done (items 27 and 28); the run integration remains.** Identify the undeuterated and fully deuterated controls by
   the user-set times (replicates averaged), the back-exchange correction with flagged fallbacks
   for missing or bad controls (the notebook's rule: undeuterated mean above 0.5 D, or retention
   outside 0.5 to 1.1, falls back to the theoretical controls), quality flags (the
   replicate-consistency score is a start; the fit chi-square and the total intensity against the
   series are the other screens), and the control tables for the bundle. On the Bimodal
   unfolding-like spectra the correction raised "both populations recovered" from 22 % to 71 %, so
   it matters for the reported means.
2. **Run stage** (`hxex.run`): drop flagged spectra (`empty`, a decision on `no_signal`), chunking
   by device memory (results do not depend on the chunk, see the independence tests; pad to one
   global width), progress and cancellation across the whole pipeline.
3. Open design items: how K is capped by the number of bins; whether to add bounds on the means
   (per component, with decisions on K above the number of bounds) or a fixed-population
   (fractions only) mode for unfolding studies; the choice of noise model for the Phase 5
   validation; fragment-ion amide counts (see "Risks and open decisions").

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
