# Project plan: a successor to pyHXExpress with a batched binomial-mixture engine

Status: Phases 0 to 3 are done and Phase 4 is done except the sequence-coverage uptake map (see
"Status"). The dated findings and measurements are in `docs/HISTORY.md`: the "... status"
sections and the numbered items (item 27 and so on) cited in this plan are there.

## Goal and scope

A standalone, pip-installable package that goes from spectra (raw or exported) to binned
sticks to binomial-mixture fits with uncertainties and reports. Its fitting engine is the
one built and tested in this repository (`fit_multimodal_binomial.py`). It is meant to
replace pyHXExpress for new users while staying comparable with it.

## Decisions so far

- **Name, repository and hosting:** the package is called **`hxex`** (`pip install hxex`,
  `import hxex`), in a new repository hosted at github.com/tuttlelm/hxex. Existing pyHXExpress
  notebooks use `import hxex_updating as hxex`; code that mixes the two should rename that alias
  to avoid confusion with the new package. The prototype (the single-file fitting script
  `fit_multimodal_binomial.py` and its test notebook, in a separate working folder with its own
  Python 3.10 environment) stays untouched as the reference and regression source of the engine.
  The PyPI name is not reserved yet (see "Status").
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
  2.15 requires 3.11 or newer.
- **Saving all fits:** every K = 1 .. K_max fit is saved by default; individual bootstrap
  fits are an option and are kept for the **selected K only** (see "Saving all fits").

## Reference implementation

`hxex_updating.py` is the reference for the behaviour of existing functions (readers, peak
picker, chemistry, metadata handling). The other copies in the pyHXExpress folder
(`pyhxexpress.py`, dated backups, `hxex_updating_26mar2025.py`, ...) are not references.
`hxex_fragments.py` is the reference for fragment-table generation.

The evidence behind the "new package, port selectively" choice (how `hxex_updating.py` is built,
and what its readers and picker do) is in `docs/HISTORY.md`.

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
generate plots of the raw data and the fits, without refitting. The bundle as built (schema
version 1) is specified for readers in other languages in `docs/BUNDLE.md`. The original design
text, pyHXExpress's output files and the storage measurements are in `docs/HISTORY.md` under
this heading.

Design decisions:

- A **results bundle** (a directory or zip of open-format files plus a JSON manifest with a
  schema version) holds everything needed to replot: the binned data and, optionally, the raw
  spectra; peptide metadata (sequence with PTMs, charge, ion type, modifications, replicate,
  time); the fitted components per measurement and per K, the selected K and the BIC/AIC of each
  K; the noise setting, the control identification, the back-exchange values and the fallback
  flags; bootstrap summaries and the per-bin CI band of the total fit; the natural-abundance
  shape used for each measurement, so replotting does not depend on the chemistry code version;
  the configuration and package versions; and a **review layer** (per-spectrum accept or
  reject, comments, manual overrides such as a chosen K) that needs no refitting (see "GUI").
  pyHXExpress does not store the fitted curves (it recomputes them from the parameters and a
  shape rebuilt from the sequence).
- A **loader** returns the same in-memory objects the fit produces, and the **`report` module
  plots from those objects only**, so a fresh fit and a loaded bundle give identical plots.
  Round-trip tests: fit, save, load and plot must reproduce the direct plot's data exactly.
- Only bundles written by the new package need to be readable at first.
- **Raw spectrum storage (confirmed):** in memory, a lightweight spectrum set of flat NumPy
  arrays (m/z, intensity) with an offsets array and a small typed index, and no pandas
  DataFrames in the core. On disk, Parquet (zstd) with **one row per spectrum and list
  columns**; the same format is used for the binned table and the results tables, so a single
  dependency (pyarrow) covers the bundle. Considered and not chosen: ragged `.npy` arrays plus a
  JSON index (8 times larger, not language-neutral or typed). The bundle stores extracted
  spectra windows, not whole LC-MS runs.
- **Saving all fits:** the parameters of every fitted model K = 1 .. K_max per measurement, with
  NLL, BIC and AIC, the selected-K flag and the minimum-population rule, are saved by default.
  The individual bootstrap fits are an option, kept for the **selected K only** (the bootstrap
  refits each measurement at its selected K).

Later goals:

- A loader for pyHXExpress's own outputs (`alldeutdata`, `allrawdata`, `data_fits`,
  `fitparamsAll`, `metadf`), so older results can be plotted with the new plotting.
- **HX-Express output** (`hxex.io`): a writer for the HX-Express format, as in pyHXExpress's
  `export_to_hxexpress`, but writing the **peak-picked** spectra (a `BinnedSet`) rather than the
  raw ones, under the file name that `read_hxexpress_file` and `scan_hxexpress` read
  (pyHXExpress writes a name its own reader does not expect), so a round trip works and is
  tested. The full specification (layout, labels, `removeUNTDreps`) is in `docs/HISTORY.md`
  under this heading. Writing the fitted results in pyHXExpress's CSV layouts (`data_fits`,
  `fitparamsAll`) is a separate item that goes with the loader above.

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

Where a residue-based result differs from pyHXExpress (the amide count with a modified residue
among the first residues, fragment slicing), the difference is deliberate; see "Deliberate
differences from pyHXExpress". The original findings are in `docs/HISTORY.md` under this heading.

Open questions:

1. Which PTMs and terminal modifications are needed in the first release beyond `acK`,
   `pS`, `pT`, `pY` (for example N-terminal acetylation)?
2. Should modified residues change the exchangeable amide count? pyHXExpress currently does
   not (the side-chain term is off by default).

Status: question 1 is partly answered, since terminal groups (`H-`, `Ac-`, `-OH`, `-NH2`) are
supported (item 36); question 2 is still open ("Phase 1 status: `hxex.chem`").

## Ion types (fragment ions)

Added after review: the chemistry must depend on the fragment ion type, not only the peptide
sequence.

What pyHXExpress does with ion types (carried in the modification string; a fragment generator;
a shape and an amide count that ignore the ion type) and the measured effect of the ion-type
composition on the shape are in `docs/HISTORY.md` under this heading.

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

Status: the plan above is implemented (`chem` takes the ion type; fragment entries name their
parent and fragment, items 36 and 42). Question 3 is settled: the shape follows the ion-type
composition ("Phase 1 status: `hxex.chem`"). Fragment spectra are fitted with ion-type mixtures
(item 35). Questions 1 and 2 are open; question 1 is the first entry of "Risks and open
decisions".

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

Sequence: the first GUI is a **results viewer and review tool** (open a bundle, browse spectra
and fits, set review flags), after Phase 4. A run launcher (choose files, set the configuration,
start and monitor a run) comes after that.

The technology options (Qt desktop, browser-based, notebook widgets) and when integrating the GUI
from the start would be better are in `docs/HISTORY.md` under this heading.

## Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0. Setup | New repository under version control, packaging, CI, license and naming, compatibility contract for pyHXExpress files | Installs and tests run on CPU |
| 1. Input | `chem` (with ion type, default `M`), picker and the two readers (SpecExport and HX-Express) | Binned tables match pyHXExpress on the test sets; ion-type chemistry matches pyteomics |
| 2. Engine | The current script split into modules with explicit config objects; the results-bundle schema | Reproduces the frozen results; passes CPU/GPU parity; save/load round trip is exact |
| 3. Controls and results | Controls, noise reference, the back-exchange method, quality flags | Method decided on real data; flags documented |
| 4. Scale and reporting | Chunking by device, bootstrap, plots from saved results, tables | A 10,000-fit run within a measured time on stated hardware; replots from a saved bundle match direct plots |
| 5. Validation and release | Real-data replicate check, docs, examples | Release 0.1 |
| 6. Extra input | Fragment-table generation, Thermo raw (mzML is done, item 33); better peak picking | Each has golden tests |
| 7. GUI (after Phase 4) | Results viewer and review tool; later a run launcher | Opens a saved bundle, browses fits, saves review flags without refitting |

## Status

A summary of the "... status" sections and items in `docs/HISTORY.md`, which hold the details.

- **Phase 0 (setup): done.** Repository, packaging (hatchling, `GPL-3.0-or-later`, SPDX
  headers), CI on 11 jobs (Linux, Windows and macOS, and a minimum-versions job at the
  dependency lower bounds), a CPU-only `.venv` and a separate GPU `.venv-gpu`. Not done: the
  PyPI name is not reserved, and the publish workflow (`docs/RELEASING.md`) has never run.
- **Phase 1 (input): done.** `hxex.chem` (modification registry, ion types, amide rules empty by
  default), the readers (HX-Express, SpecExport, pooled folders; spectrum ids; progress and
  cancellation hooks) and the picker (`hxex.picking`), checked against pyHXExpress output. The
  shoulder rule and the 10-point rule are unvalidated (synthetic tests only).
- **Phase 2 (engine): done; the exit criteria are met.** `model` (exact gamma-function
  continuation of the binomial), `noise` (Gaussian, Poisson, an optional estimate from
  replicates with an outlier safeguard), `fit` (Levenberg-Marquardt polish, measurements
  independent of each other), `bootstrap`, the results bundle, raw spectra saved without a fit,
  and CPU/GPU parity.
- **Phase 3 (controls and results): done** (items 27 to 29): controls and the back-exchange
  correction, quality flags, and the run stage with chunks of whole series.
- **Phase 4 (scale and reporting): done except the sequence-coverage uptake map** (item 31),
  which moves to the post-fitting analysis (see "Next steps"). Bootstrap at fixed K (item 30);
  the large-dataset benchmark met the 10,000-fit target (item 32). HX-Express-style output of a
  run is open.
- **Command line:** `hxex.cli` is still a placeholder (`hxex --version` works).
- **Phase 5 (validation and release):** no work recorded yet; its replicate check is to decide
  the noise model.
- **Phase 6 (extra input): partly done, ahead of order.** The mzML reader with scan summing
  (item 33), the composition ladder in the picker (item 34), ion-type mixtures for fragments
  (item 35), fragment targets and terminal groups (item 36), the m/z screen (item 37), precursor
  isolation windows (item 38), fragment comparisons with pyHXExpress and the closure test (items
  39 to 41), and read records for every format (item 42). Open: fragment-ion amide counts, the
  z-fragment offset, the shoulder rule, reading Thermo `.raw` directly.
- **Phase 7 (GUI): not started.** The review layer, the `*_data` functions, the artists' `gid`s
  and `Results.with_review` are in place for it (item 31).

## Deliberate differences from pyHXExpress

hxex matches `hxex_updating.py` (and `hxex_fragments.py` for fragments) except where listed here.
Each entry points to the evidence: a section of this plan, or a section or item of
`docs/HISTORY.md` ("HISTORY" below). Add every new difference here.

Project and design:

- License `GPL-3.0-or-later`; pyHXExpress declares plain GPL-3.0 (author's choice; "Decisions
  so far", HISTORY "Phase 0 status").
- No global `config` module: explicit, serialisable config objects; `Nterm_subtract` is an
  explicit argument (`nterm_subtract`, default 1) (HISTORY "Evidence behind the ..." and "Phase 1
  status: `hxex.chem`").
- No pandas in the core; raw spectra are stored as Parquet with one row per spectrum, not as a
  zipped CSV with one row per m/z point ("Results bundle and replotting"; HISTORY "Raw spectrum
  storage").
- Results are saved as a results bundle (every K, the shapes used, bootstrap summaries, a review
  layer) instead of `data_fits`, `fitparamsAll` and the other files ("Results bundle and
  replotting").
- Modifications live in a private registry passed to pyteomics through `aa_comp`; pyHXExpress
  edits pyteomics' global residue table ("Sequence syntax and PTMs").

Chemistry (HISTORY "Phase 1 status: `hxex.chem`"):

- Residues are tokenised, never sliced as characters: with `Nterm_subtract = 2` and a modified
  residue among the first residues the amide count differs (`acKPEPTIDE` 5 against 4; HISTORY
  "Sequence syntax and PTMs", item 1).
- m/z with a composition modification (`H:1`, ...) adds the composition once; the reference adds
  `z` protons a second time and is one proton mass too high.
- The shape with a composition modification adds it to the counts; the reference overwrites them
  (`H:1` sets H to 1, `Hex:1` replaces the amide count instead of adding to it).
- The natural-abundance shape follows the ion-type composition; the reference uses the
  default-termini composition (identical for `M` and `y`; "Ion types", HISTORY "Ion types").

Readers (HISTORY "Phase 1 status: readers"):

- Readers do not pick peaks and have no `Test_Data` branch (reading the solutions file).
- An unknown or missing time unit raises `ReadError` (the reference silently scales such a label
  to 1/60 of its value); units and the `undeut`/`TD` prefixes are case-insensitive.
- SpecExport charges of 10 or more are read (the reference reads one digit); spectra are ordered
  by time, replicate and file name (the reference uses `os.listdir` order).
- Sample and peptide-range filters are exact matches (the reference's HX-Express peptide filter
  is a substring match); files and directories whose names do not parse are skipped with a
  message instead of stopping the scan; `.xls` is not read.
- A label's spectrum is read from the label's own column and the next, not from column `2 * i`;
  with several FASTA records the last is used, as in the reference, but with a warning.
- Pooled matching: a class must appear in a single matching record id (the reference searched
  the `;`-joined string); `class_suffix` is not ported (tuples, not joined strings); sequences
  in pooled directory names that do not parse are skipped with a warning.
- Times are computed as decimals and rounded once (`4.1m` is exactly 246 s and equals `246s`).
- Kept on purpose: digits after the decimal point of an HX-Express time of 1 or more are
  replicate markers (`1.2 min` is 60 s, second replicate).

Peak picker (HISTORY "Phase 1 status: picker" and item 34; `docs/PEAK_PICKING.md`):

- Bins sit at the composition ladder (natural isotope clusters convolved with a binomial
  deuterium distribution), not the ExMS ladder; `PickerConfig.ladder='exms'` keeps the old
  ladder, and the pyHXExpress oracle tests use it (item 34).
- Picking ends `zero_filling` bins after the last bin that reaches the threshold, instead of
  after a cumulative count of below-threshold bins right of the mean m/z, so a valley inside a
  multimodal envelope never ends picking early.
- The bin limit is the physical range, `amides + Hex + (natural-abundance peaks - 1) +
  zero_filling`, not `max(amides, min_pts) + peaks // 2 + zero_filling`; `capped` flags spectra
  that reach it.
- No `min_pts`, so the picked data no longer depend on `Max_Pops`, a fitting setting.
- Nothing is dropped by the picker: all-zero spectra are flagged `empty` and `no_signal`
  (pyHXExpress dropped them in the SpecExport reader only); the run stage drops flagged spectra.
- `time_idx` is not stored (in pyHXExpress a dropped time point could make `run_hdx_fits` select
  the wrong raw spectra); spectra are referred to by `spectrum_id`.
- A tie for the window maximum goes to the first point in m/z order (pyHXExpress used an
  unstable sort); envelope errors are not swallowed (NaN `env_width` and `env_symm`).
- A trailing zero fill of exactly `zero_filling` bins replaces pyHXExpress's variable tail.
- Kept on purpose: the inclusive +/- ppm window, the 10-point rule and shoulder rejection,
  sub-threshold intensities kept as they are, predicted (not observed) m/z in the table, the
  envelope formulas, `max_namides` without `Hex`.
- Fragment ion-type mixtures: hxex fits the weights of the given ion types to the series'
  undeuterated references, with marked fallbacks; pyHXExpress used `NA_envelope` `mixedN` (and
  `infer`) (item 35).

Fitting (HISTORY "Evidence behind the ...", "Phase 2 plan" and "Phase 2 status"):

- A batched binomial-mixture engine fits a user-set range of K with BIC selection, instead of
  `run_hdx_fits` fitting one spectrum at a time with `curve_fit` or `differential_evolution`.
  Fixed-K bootstraps (`BootstrapConfig.ks`) cover pyHXExpress's separate fixed 1- and
  2-population runs (item 30).
- The real-valued binomial is masked at `b < n + 1` (exact gamma-function continuation), not
  `b <= n` as in pyHXExpress's `nCk_real` and the prototype: the pmf is continuous in `n`, and
  means at high `p` can move (Bimodal: median 0.005 D, 90th percentile 0.18 D).
- Back-exchange correction from the controls (control means from K = 1 fits, per-control
  fallbacks with flags); `hxex_updating.py` has no back-exchange correction (item 27). Uptake
  plots are corrected by default (item 31).

Fitting, differences from the prototype (the engine's regression source; HISTORY "Phase 2 plan"
and "Phase 2 status"):

- Device and dtype are explicit engine settings, not module globals; the single-measurement code
  path and `mean_bounds` are dropped.
- `n_max` is `n_max_factor` times the exchangeable sites (default 1.2).
- Gradient clipping per measurement, no L-BFGS (`lbfgs_steps=0`; a per-measurement
  Levenberg-Marquardt polish instead), so a spectrum's fit does not depend on the rest of the
  batch.
- Every random number of a measurement comes from a generator seeded by `seed` and its
  `spectrum_id`, so runs are reproducible and independent of batch position and size.
- A K with a nested start is also fitted from the k-means start (`FitConfig.kmeans_after_warm`),
  so a K in a run of several is not worse than the same K fitted alone (item 30).
- Bootstrap: a failed resample is dropped for that measurement only (the prototype discarded the
  iteration for all), a measurement whose own fit failed is not resampled, and resamples are
  seeded per iteration and spectrum id.

Fragments (`hxex_fragments.py`; HISTORY item 36):

- Fragments are sliced by residue (it slices characters, so `PEPacKTIDE` b4 gave `PEPa`; HISTORY
  "Sequence syntax and PTMs", item 2).
- `ALL` and open ranges mean lengths 1 to the number of residues (it used `start_seq` to
  `end_seq`, the same only when the parent starts at 1).
- A parent `modification` other than `ion_type:M` is refused rather than copied to every
  fragment.
- Kept: the default charge limit is the parent's charge minus one.

Known differences in results (findings, not rules):

- z fragments: hxex's deuteration is up to 0.25 D above pyHXExpress's on several peptides; a
  closure test (c plus the complementary z against the precursor) favours hxex, so the offset is
  most likely pyHXExpress's (items 35, 39 to 41).
- 1+ c fragments from a 4+ precursor: pyHXExpress's values are 0.3 to 0.6 D above its own values
  from the 3+ precursor while hxex agrees with itself; the difference is pyHXExpress's (item 39).

Planned, not built:

- The HX-Express writer will write the peak-picked spectra under the file name the reader
  expects (pyHXExpress writes the raw spectra under a name its own reader does not read; "Results
  bundle and replotting", HISTORY "Later goals").

## Next steps

1. **Phase 3 (controls and results): done** (items 27 to 29: controls and the back-exchange
   correction, quality flags, their bundle tables and the run stage). On the Bimodal
   unfolding-like spectra the correction raised "both populations recovered" from 22 % to 71 %, so
   it matters for the reported means.
2. **Phase 4 (reports and plots): done except the sequence-coverage map**, which moves to the
   post-fitting analysis below (item 31; the large-dataset benchmark, item 32, met the Phase 4
   target). **Run stage** (`hxex.run`): done (item 29). HX-Express-style output of the run is
   still open.
3. **Post-fitting analysis, together with the sequence-coverage map (planned; not designed yet).**
   Analyses that start from the fitted results and work on the protein rather than on one
   spectrum. The owner will give further references when this starts; pyHXExpress's
   `hxex_utils.py` (reference copy, see CLAUDE.local.md) already has some of it:
   - coverage map: peptides as bars along the sequence, coloured by uptake (or a population's
     uptake) at each time and state (`plot_coverage`);
   - residue-level values from overlapping peptides: a per-residue relative uptake (RFU)
     averaged over the peptides that cover it, with the redundancy (how many peptides cover a
     residue) and the resolution (the length of the shortest one) (`plot_rfu_residue`,
     `plot_resred`); for two populations it assumes the more protected populations of
     overlapping peptides belong together, which needs thought beyond two;
   - choosing between the one- and two-population description of each peptide and time from
     the fixed-K fits and their errors (a population fraction above a minimum, standard errors
     below a limit at a stated confidence), falling back to the one-population value
     (`choose_fits`); here this would build on the selected K, the fixed-K bootstraps and the
     review layer rather than replace them;
   - comparing states (samples) per residue and time, and averaging over user-defined regions
     of the sequence (`plot_comparison`, `group_fits`);
   - **comparing separate runs** (a common workflow; owner, 2026-10-12): the same comparisons
     across bundles as across samples within one. Possible today only by hand (open each
     bundle, `find_series` by sample, range and charge, draw with `ax=` side by side). Needed:
     matching series across runs by identity (sequence with modifications, charge; never by
     `data_id`, which is a position within its run) with a check that they agree; overlays
     with one colour per run (the uptake plot colours by replicate); differences per peptide and
     time with propagated uncertainty, and difference or butterfly plots; a comparison of the
     runs' settings (manifests) that warns where they differ in ways that matter (picker, noise
     model, control times and limits); and a rule for a peptide and time fitted with one
     population in one run and two in another (shared with the one-or-two-population choice);
   - recomputing uptake without spectra marked bad (`recalc_uptake`; here the review layer's
     rejections) and combining runs (`combine_batch_data`);
   - export of the uptake table in the format other tools read (`hdexa_to_pyhdx` writes the
     DynamX-like table pyHDX expects).
   - **exchange rates from time courses** (owner, 2026-10-12): for continuous-labelling or
     back-exchange kinetics with the scan time as the exchange time (fragment-level data such as
     c and z ions, see CLAUDE.local.md), fit an exponential decay (or rise) of each fragment's
     deuterium over time, with a constant model as the alternative, and turn consecutive
     fragments into per-residue rates; the reference is pyHXExpress's kinetics notebooks.
   **Decided (owner, 2026-10-12):** these tools are rewritten to fit hxex (its result types, no
   pandas in the core, explicit configs), and the parts pyHXExpress does through pyHDX
   (residue-level averaging, the uptake-table conversion) are rewritten too, so neither pyHDX nor
   proplot becomes a dependency; plots use matplotlib, like the rest of `hxex.report`. Rejected or flagged spectra, the back-exchange correction and the
   population identity by rank all carry into these analyses and need decisions when this is
   designed.
4. Open design items: how K is capped by the number of bins; whether to add bounds on the means
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
  Update: the method is decided and implemented (item 27, the owner's decisions of 2026-10-11)
  and the uptake plots are corrected by default (item 31); whether to derive the retention limits
  per dataset is open (item 32).
- **Sigma guidance.** Sigma is user-supplied, but users will need a recommended way to
  estimate it, probably from baseline bins. The peak picker's noise threshold should
  connect to it.
  Update: the noise can be estimated per series from replicates
  (`NoiseConfig(estimate='replicates')`, HISTORY "Phase 2 status"); the choice of noise model is
  left to the Phase 5 validation.
- **Dependency footprint.** PyTorch is large. CPU-only wheels help, but an install story for
  Windows and Mac users needs testing.
  Update: CI installs and tests the CPU wheels on Windows and macOS (HISTORY "Phase 0 status").
- **Thermo raw** needs a vendor library or a third-party reader, with licensing to check.
  Update: `.raw` files are converted outside hxex (ThermoRawFileParser) and read as mzML; reading
  `.raw` directly stays a Phase 6 item (item 33).
- **CPU time for large bootstrap runs** may need user guidance (smaller N, fewer restarts).
  Update: confirmed by the benchmark (item 32): the CPU is fine for fits but not for full
  bootstraps at that size; float32 and less single-threaded Python are possible speed-ups, not
  done.
- **Memory and plots at 10,000+ fits:** figures must be selective by default.
  Resolved: `PlotConfig` draws the `flagged` series by default and `write_figures` makes and
  releases one series at a time (item 31).

## Decisions needed

1. The remaining Phase 0 items above (initial commit, creating the GitHub repository, reserving
   the PyPI name, confirming the SPDX identifier), and what to start in Phase 1. (Name,
   repository, hosting, license and minimum Python version are decided.)
   Resolved except reserving the PyPI name: the repository exists and is public, the SPDX
   identifier is `GPL-3.0-or-later`, and Phase 1 is done (see "Status").
