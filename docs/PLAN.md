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
  RawFileReader is under Thermo's own license: hxex reads `.raw` files with it only through an
  optional extra (pythonnet) and assemblies the user supplies; it is never bundled, downloaded
  or required. A GPL section 7 permission (`LICENSE-EXCEPTION`, owner's decision of 2026-10-12)
  allows linking or combining hxex with it; Thermo's own terms are still open (item 43).
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
- `kint`: intrinsic amide exchange rates in H2O/D2O mixtures (forward and back), for
  theoretical references and protection factors (HISTORY item 70).
- `bootstrap`.
- `run`: chunking by device and orchestration.
- `report`: tables and plots.
- `cli`, plus a TOML or YAML configuration file. Decided: TOML only (`hxex.config`, read with
  the standard library's `tomllib`; item 45).

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
- **Figures are drawn from the bundle only (owner, 2026-10-20):** `hxex run` writes the bundle
  and tables and draws no figures; `hxex report` draws them, with the project file's
  `[figures]` table (`PlotConfig`, top level, not part of the run settings), so changing a
  figure setting never needs a new run. `[figures] k` (or `--k`) draws every series at one
  number of populations; it changes no table or decision (HISTORY item 123).
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

## Excluding spectra

Requirement (owner, 2026-10-19): a user can leave out spectra they know or suspect to be bad,
without deleting anything and without Python, both before fitting and after it. Design
(owner's choices, 2026-10-19; HISTORY item 116):

- **One rule format** for both levels: repeatable tables of the project file, `[[input.exclude]]`
  and `[[review.exclude]]` (named `[[analysis.exclude]]` until 2026-10-20; still accepted). A rule names any of `sample`, `peptide_range` (by residues, as the
  `peptide_ranges` selection), `charge`, `time` (a time point; with a unit), `role` (`undeut`,
  `fulldeut`), `rep`, `file` (the data file: a run's mzML or raw file, an HX-Express
  spreadsheet, a SpecExport spectrum file; by name) and `spectrum_id`, and a `note`. A spectrum
  is excluded when it matches every key of one rule; a rule must name at least one key; a rule
  that matches nothing is a warning (a typo excludes nothing silently otherwise).
- **Run level** (`[[input.exclude]]`): the matched spectra are read and stored with the other
  raw spectra, but not picked or fitted, and count for nothing (controls, noise references,
  envelopes). The bundle's manifest records the rules, the count per rule and the excluded ids
  (`excluded`); `hxex info` shows them. Removing a rule and rerunning (`--spectra`) fits them.
  Every format; it applies to stored spectra too.
- **After fitting** (`[[review.exclude]]`): the fits stay in the bundle; `hxex analyse`, and
  `hxex run` (its tables) and `hxex report` (tables and figures), treat the matched spectra as
  rejected. They are applied in memory as rejections added to the review layer (note
  "excluded by [[review.exclude]]: ..."), never written into the bundle, so every consumer of
  the review honours them and changing the rules needs only a rerun of the command.
  `hxex report` reads the project file for them (`hxex report hxex.toml`, the bundle from
  `[output] bundle`).
- **The user review table** (owner, 2026-10-20; HISTORY item 124): per-spectrum decisions made
  in a spreadsheet, a copy of `spectra.csv` named in `[review] table` (any name and
  folder; the guide suggests `user_review_table.csv` beside the project file). Only `spectrum_id`, `review_status`,
  `review_K` and `note` are read; applied in memory before the rules, like them. A chosen K
  needs a bootstrap at that K when the run has one (comparable errors): `hxex report` refuses
  until `hxex resume hxex.toml` bootstraps exactly those spectra (a partial fixed-K bootstrap);
  `hxex run` does it itself. Per spectrum only (owner). The table and the after-fitting rules
  are the top-level `[review]` table (owner: transparent, since run, report, and analyse all
  apply it).
- **The review layer stays**: per-spectrum decisions stored in the bundle (Python today). Both
  apply; a spectrum the review accepts but a rule excludes is excluded.
- `hxex check` lists the rules and counts what each matches among the spectra the read would
  make, without reading them (file names, HX-Express header rows, scan tables); a rule that
  matches nothing fails the check (owner, 2026-10-21, HISTORY 133). The run logs the counts
  again.

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
| 6. Extra input | Fragment-table generation, Thermo raw (mzML and Thermo raw are done, items 33 and 43); better peak picking | Each has golden tests |
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
- **Command line: done** (item 45): `hxex init`, `check`, `run`, `resume`, `report` and `info`,
  driven by a TOML project file (`hxex.config`).
- **Phase 5 (validation and release):** no work recorded yet. The noise model is decided ahead
  of it: the Gaussian loss (owner, 2026-10-13; HISTORY items 55 and 56), reopened for its level
  (2026-10-24): per-bin noise from the controls is built as an option (Next steps 13).
- **Phase 6 (extra input): partly done, ahead of order.** The mzML reader with scan summing
  (item 33), the composition ladder in the picker (item 34), ion-type mixtures for fragments
  (item 35), fragment targets and terminal groups (item 36), the m/z screen (item 37), precursor
  isolation windows (item 38), fragment comparisons with pyHXExpress and the closure test (items
  39 to 41), read records for every format (item 42) and Thermo `.raw` files read directly
  (item 43). Open: fragment-ion amide counts, the z-fragment offset, the shoulder rule.
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
- Controls are identified by a **role** (`undeut`, `fulldeut`) that the readers set from the
  formats' control labels (`Non`/`Full`, `undeut`/`TD`) or that `role` in `[[input.runs]]`
  gives a control file; their time is blank. pyHXExpress stamps them with the times 0 and
  `FullDeut_Time` (1e6) and finds them by those numbers, which collides with a real 0 s point
  in a back-exchange run. `undeut_at` and `fulldeut_at` in `[run.controls]` (and
  `[analysis.controls]`) make the spectra at a time serve as a control as well. Every time in
  a project file may be written with a unit (`"5 min"`, `"20 h"`); stored values stay in
  seconds (`docs/AUDIT_2026-10-17.md` design item 1; HISTORY 101).

Peak picking (`docs/PEAK_PICKING.md` "The intensity of a bin"):

- Each bin takes the tallest peak (a local maximum of the raw spectrum, its apex inside the
  window) instead of the tallest point in the window, so a neighbour's flank is never picked;
  `pick = "window_max"` picks as pyHXExpress, shoulder rule included (HISTORY 172).
- A bin takes the peak's area instead of its height when the run's undeuterated controls show
  weak peaks wider than tall ones (`[run.picker] measure = "auto"`, the default; Q-TOF exports
  do, an Orbitrap does not); `measure = "height"` picks as pyHXExpress (HISTORY 181, 185).

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
- Fragment amides are counted by position in the parent (HISTORY item 65): c, x and y ions hold
  one amide more than the reference's peptide rule (the cleavage-site N-H), and
  `nterm_subtract` applies to the parent's first residues, not to a C-terminal fragment's (the
  reference subtracts a z ion's first residue by its rule, which is right for z by
  coincidence, and wrong for y and x, and for z with `Nterm_subtract` other than 1).
- `nterm_subtract` applies to a free N-terminal amine only: after an acetylated N-terminus
  (`Ac-`) residue 1 keeps its amide and none is subtracted, so the amide count agrees with
  `hxex.kint`, which rates that amide (the reference subtracts `Nterm_subtract` residues
  whatever the terminus; `docs/AUDIT_2026-10-28.md` Low, owner 2026-10-29; HISTORY 195).
- Fragment lists stop at `n - 1` residues (a full-length c or z ion is not a fragment) and leave
  out c ions before a proline and z ions starting with one (not made by electron-based
  dissociation).

Readers (HISTORY "Phase 1 status: readers"):

- Readers do not pick peaks and have no `Test_Data` branch (reading the solutions file).
- An unknown or missing time unit raises `ReadError` (the reference silently scales such a label
  to 1/60 of its value); units and the `undeut`/`TD` prefixes are case-insensitive.
- SpecExport charges of 10 or more are read (the reference reads one digit); spectra are ordered
  by time, replicate and file name (the reference uses `os.listdir` order).
- Sample and peptide-range filters are exact matches (the reference's HX-Express peptide filter
  is a substring match); files and directories whose names do not parse are skipped with a
  message instead of stopping the scan; `.xls` is not read.
- A label's spectrum is read from the label's own column and the next, not from column `2 * i`
  (the same for the normal layout; a sheet with spacer columns is read by its labels, kept so
  by the owner, `docs/AUDIT_2026-10-28.md` question 27); with several FASTA records the last is
  used, as in the reference, but with a warning.
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
  `infer`) (item 35). Opt-in `share`: the conditions of one ion (scan filters, list label)
  share its envelope, checked pairwise; an envelope not measured well is flagged
  `poor_envelope` (item 87).
- The picking window is capped at half the isotope spacing, `1.00235 / (2 z)`: pyHXExpress's
  constant ppm window spans a whole spacing at high charge, and its shoulder rule then zeroes
  the envelope's flanks (a third of the bins and a 9-bin centre shift on a top-down example at
  charge 20; no peptide dataset reaches the cap) (items 99 and 100).
- Opt-in picking window from the instrument's resolving power (`resolving_power`, a value or
  `"recorded"` from Thermo raw files): the half-width is `window_fwhm` peak widths `m / R(m)`,
  floored at `mass_accuracy_ppm`; pyHXExpress has only the constant ppm window, which hxex
  keeps as the default (item 98).
- A peptide-ion series whose undeuterated controls put its envelope off the run's calibration
  (a Theil-Sen line in m/z per undeuterated injection; beyond `off_mz_sd` = 6 times the
  injection's robust scatter, at least 3 ppm) is flagged `off_mz`, and with `recentre` (on by
  default) its windows are moved onto its envelope in every spectrum of the series;
  pyHXExpress always centres the window on the prediction. Most such series share their peaks
  with a near-isobaric species; reading the merged peak's top instead of its flank brought an
  Orbitrap digest's flagged peptides from 73 % to 82 % of time points within 0.1 of the
  predicted uptake, and changes nothing with a ±50 ppm window (item 166).

Fitting (HISTORY "Evidence behind the ...", "Phase 2 plan" and "Phase 2 status"):

- A batched binomial-mixture engine fits a user-set range of K with BIC selection, instead of
  `run_hdx_fits` fitting one spectrum at a time with `curve_fit` or `differential_evolution`.
  Fixed-K bootstraps (`BootstrapConfig.ks`) cover pyHXExpress's separate fixed 1- and
  2-population runs (item 30).
- The real-valued binomial is masked at `b < n + 1` (exact gamma-function continuation), not
  `b <= n` as in pyHXExpress's `nCk_real` and the prototype: the pmf is continuous in `n`, and
  means at high `p` can move (Bimodal: median 0.005 D, 90th percentile 0.18 D). Every component
  is normalised to sum to 1 over its bins (HISTORY 78): the amplitudes are areas and the
  populations are shares of the molecules at any `n` and `p`.
- The percent noise sd of a series is a percentage of its undeuterated control's tallest peak
  (or the spectrum's own when there is none; kinetics runs: HISTORY 77), not of the average of
  the undeuterated and fully deuterated controls' maxima as in `run_hdx_fits` (`Y_ERR`).
  Confirmed by the owner (HISTORY 78).
- Bootstrap resamples add unit Gaussian noise (in noise-scaled units) to every bin, including
  bins that are zero; pyHXExpress's `fit_bootstrap` leaves zero bins unchanged and clips the
  resample to `[0, 2y]`. Confirmed by the owner (HISTORY 78).
- Back-exchange correction from the controls (control means from K = 1 fits, per-control
  fallbacks with flags); `hxex_updating.py` has no back-exchange correction (item 27). Uptake
  plots are corrected by default (item 31). The fully deuterated retention is combined over a
  peptide's charge states (`combine_charges`, item 140).
- The post-fitting analysis (`hxex_utils.py` and the owner's notebooks): the hybrid rule's
  `delta_crit` uses a robust typical replicate standard deviation (the median of the
  chi-square-scaled variances), not the plain pool of Hageman and Weis, which a few broken
  replicates dominate; spectra are left out by their disagreement with their own replicates
  and by an impossible K=1 mean, not by a fixed fD range (the owner's earlier practice,
  fD < 0.1 or > 1.1, removed mostly consistent low uptake and stand-in-control overshoots);
  series with a stand-in fully deuterated control are left out of the analysis by default
  (item 140).

Fitting, differences from the prototype (the engine's regression source; HISTORY "Phase 2 plan"
and "Phase 2 status"):

- Device and dtype are explicit engine settings, not module globals; the single-measurement code
  path and `mean_bounds` are dropped.
- float32 by default on the CPU as well as the GPU (the prototype used float64 on the CPU):
  1.5 times faster, and the selected K agrees with the GPU run for 99.75 % of the intact-peptide
  benchmark; `dtype = "float64"` remains an option (item 46).
- `n_max` is `n_max_factor` times the exchangeable sites (default 1.2).
- Gradient clipping per measurement, no L-BFGS (`lbfgs_steps=0`; a per-measurement
  Levenberg-Marquardt polish instead), so a spectrum's fit does not depend on the rest of the
  batch.
- Every random number of a measurement comes from a generator seeded by `seed` and what the
  measurement is (`Measurements.random_id`: sample, peptide, modification, range, charge,
  replicate, time, role, label; not the spectrum id, which carries the project's name), so runs
  are reproducible and independent of batch position and size, the project's name, and the file
  layout (item 182).
- A K with a nested start is also fitted from the k-means start (`FitConfig.kmeans_after_warm`),
  so a K in a run of several is not worse than the same K fitted alone (item 30).
- The k-means start clusters the bins weighted by their rounded intensities (the same clusters as
  the prototype's k-means on that many copies of each bin), batched over the measurements, with
  centroids starting at the weighted quantiles instead of random points (item 85).
- An optional exact screen (`FitConfig.screen`, off by default) skips the K that cannot be
  selected, from a lower bound on the loss of any mixture; the prototype fits every K (item 85).
- Bootstrap: a failed resample is dropped for that measurement only (the prototype discarded the
  iteration for all), a measurement whose own fit failed is not resampled, and resamples are
  seeded per iteration and measurement identity.
- An optional population test (`[run.population_test]`, off by default) decides each
  spectrum's number of populations by a simulated null instead of BIC or AIC, with a width
  check against populations narrower than deuterium allows, and the analysis decides each
  peptide set's and residue's from the same test (HISTORY 174; `docs/ANALYSIS.md` "The
  population test"); pyHXExpress chooses K per spectrum by its criterion or bootstrap
  p-values.

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
3. **Post-fitting analysis (designed 2026-10-13; phases 1 to 5 of its peptide branch built,
   including comparisons between runs): see `docs/ANALYSIS.md`.**
   Analyses that work on the protein rather than on one spectrum: coverage map, residue-level
   uptake per population, the one-or-two-population choice, comparisons between samples and
   between runs, region summaries, kinetics and export. A peptide (MS1) branch comes first and a
   fragment (MS2, MS3) branch after it; pyHXExpress's `hxex_utils.py` is the reference for the
   peptide analyses, rewritten for hxex without pyHDX or proplot. Phases, decisions and open
   questions are in `docs/ANALYSIS.md`.
4. Open design items: how K is capped by the number of bins; whether to add bounds on the means
   (per component, with decisions on K above the number of bounds) or a fixed-population
   (fractions only) mode for unfolding studies; fragment-ion amide counts (see "Risks and open decisions").
5. **Speed up reading scan files** (owner, 2026-10-16). DONE for the measured case (HISTORY
   94): 17 Thermo files of a targeted LC-MS dataset, about 23 min before, now 2.3 min. The time
   was not in pythonnet but in two Python loops over the scans, once per target. What is left
   (about 8 s per file: summing profile scans, the scan table through pythonnet) is spread
   out; reading files in parallel processes is the next step if it is needed.
6. **Audit of 2026-10-17: fixes, the overwrite contract, the readability pass and five design
   items** (control identity by label; the run as a variable in the analysis; the two-direction
   protection model; a picking window from the resolving power; a user guide). Findings,
   decisions and execution order are in `docs/AUDIT_2026-10-17.md`.
7. **User guide and tutorials** (owner, 2026-10-17). The documents in `docs/` are written for
   the developer; a complete pass to create user-friendly documentation and tutorials (inputs,
   a worked project from `hxex init` to `hxex analyse`, what every table and figure holds) is
   an open TODO. Begun 2026-10-19 (HISTORY item 107): `docs/guide/` with the inputs, the
   experiment types, the outputs and a generated settings reference. The worked tutorial on
   the published B5 data, with a notebook and a download, was added 2026-10-20 (HISTORY 127).
8. **Does retention depend on the charge state?** (owner, 2026-10-20; a future TODO, not part
   of the tutorial.) The fully deuterated retention is a property of the peptide (sequence,
   time on the column: HISTORY 126); check whether the charge states of one peptide also agree,
   which would let `fd_disagrees` compare across charges and the kint model treat them as one.
   First step built (HISTORY 129): a missing or rejected fully deuterated control takes the
   same peptide's other charge states in the same sample (`fd_other_charge`); in the tutorial
   data the measured charge states agree to within 0.006. In infusion kinetics, where all
   charge states are in the same scans, they agree within about 1.5 % of the deuterium
   present (HISTORY 130), so `fd_other_charge` now counts as a measured control for the
   back-exchange start of the protection fit. Since HISTORY 140 the retention is combined over
   charge states by default (`combine_charges`; charges that disagree are flagged
   `fd_charges_disagree`). Still open: the kint model treating charge states as one.
9. **Spectrum quality screens** (owner, 2026-10-23; HISTORY 149, 151, 152). Built: five
   fit-free scores for another species in the raw spectrum (`screen_*`, computed at picking,
   in the bundle and `spectra.csv`), and `[analysis.filters] spectrum_quality_filtering`
   (none / loose / strict, default loose) that removes (controls leave the correction too),
   keeps at K = 1 only, or warns; `accepted` in the review keeps a spectrum; fragments are not
   screened. Later the same setting holds the weak-spectrum rule (after the noise model), and
   fragment levels if fragment data are ever scored.
10. **Retention windows from the data** (owner, 2026-10-24; built 2026-10-24, HISTORY 165:
   `[input.scans] find_peak`, the peptide table's `run` column, `rt_start`/`rt_end` per spectrum;
   still open: the ambiguity flag, fragments). For LC-MS scan files,
   find each peptide's elution in the runs themselves, as well as using any range a peptide
   table or metadf-style file gives (`rt_start`, `rt_end`, kept as they are). Prompted by a
   public Orbitrap whole-cell digest whose tables give only one retention time per peptide:
   a fixed window guessed from the gradient took in co-eluting peptides (HISTORY 162). The
   reference practice (owner, 2026-10-24): HDExaminer starts from each peptide's retention time,
   usually from an MS/MS identification run, adjusts the LC peak it uses automatically for
   each spectrum (so it can move somewhat from the undeuterated control's), and lets the user
   move the peak's boundaries by hand. So the likely shape: a given time as the start, a
   per-spectrum (run and peptide) peak found near it, and a user override of the boundaries.
   Proposed home (2026-10-24): the peptide table, not the review table (windows act when the
   scans are summed, before spectra.csv exists): given `rt_start`/`rt_end` used as they are; the
   found windows written as a copy of the peptide table with a `run` column (one row per
   peptide and run, shared by its charge states: retention belongs to the peptide); the user
   edits that copy and points `[input] peptides` at it, its windows winning over finding (the
   pattern of the user review table). Needs a `run` column and charge-free rows for the
   retention columns in the peptide table. To design: the chromatogram from which isotope positions
   (a labelled envelope moves); how wide around the apex (a multiple of the peak's width); the
   search range around the given time; what to do when two species elute within it; where
   the found boundaries are recorded and shown; and what `hxex check` reports. Prototype
   (HISTORY 164): chromatogram summed over the predicted bins (+-10 ppm), smoothed over about
   2.5 s, nearest local maximum to the given time within +-0.5 min, window apex +- half the
   half-maximum width: a third fewer artefactual populations, uptake unchanged; the ambiguity
   rule needs rethinking (it flagged half the spectra of a crowded digest).
11. **The remaining interference: near-isobaric species** (owner, 2026-10-25; HISTORY 166).
   Built: the picker's m/z check (`[run.picker] off_mz_sd`, `recentre`; quality flag
   `off_mz`; `spectra.csv` `envelope_offset_ppm`, `envelope_offset_limit_ppm`,
   `window_shift_ppm`). Tried and not adopted: a per-bin apex rule (a bin whose window holds
   only a flank set to 0 or left out of the fit: uptake worse either way), a fixed ppm limit
   (removed 30 % of the known-answer series), leaving flagged series out by default (cost more
   good values than bad ones), recentring every series (nothing gained). Open: the series whose
   undeuterated envelopes are inconsistent (M, M+1, M+2 tops disagree: 32 % of their time
   points off on the Orbitrap digest, no worse than the rest on a Q-TOF); then the set test
   (HISTORY 162). The gap to HDExaminer (item 167): mostly series whose undeuterated control is
   another species; `stand_in_controls = "exclude"` now also leaves out `m0_high` series, and
   the programs then differ at 0.6 % of the kept peptide-times.
12. **The population test** (owner, 2026-10-25; design in `docs/ANALYSIS.md` "The population
   test"; evaluations HISTORY 169-173; built HISTORY 174). One calibrated test of the number
   of populations for each spectrum, each peptide set and each residue: opt-in
   `[run.population_test]` (simulated null per spectrum), the width check, the peptide-wide
   undeuterated cap, `[analysis] populations = "population_test"`, residue tiers up to three and their
   figure. Open: `hxex resume` adding the test to an existing bundle; stuck K = 1 refits in the
   null; fragment series.
13. **Per-bin noise from the controls** (owner, 2026-10-28; HISTORY 187-191). Opt-in
   `[run.fit.noise] estimate = "controls"`: each bin's variance `z b1 I + (m I)^2`, the counting
   scale `b1` measured once per bundle from the undeuterated controls that pass the loose
   screens and sit below the run's distortion onset (per-sample values reported, a warning when
   a sample differs), `model_error` m = 9 % fixed (calibrated on three Q-TOF datasets). The
   fit, the screen's loss floor, the bootstrap and the population test's null all use it.
   Agreed path: opt-in first, then the default once the checks pass, the Orbitrap excepted until
   its own look (owner: start from the instrument information in the .raw files). Checks
   (HISTORY 191): screen, regression and GPU parity pass; with the population test it lowers
   false splits of the controls and raises correct K on the known-answer set; with the criterion
   alone it is mixed. Owner (2026-10-28): not the default until it is more reliable. Not yet done: the
   lost-signal zone (bins below 1-2 % of the tallest peak) and outliers in one replicate
   (steps 2-3 of the noise plan).
14. **Audit of 2026-10-28** (seven reviewers; no Critical finding, nine High, 21 Medium, about
   40 Low, 27 questions). Findings, the owner's decisions of 2026-10-29 and the agreed order
   (Sessions A to D) are in `docs/AUDIT_2026-10-28.md`. Done: Session A (HISTORY 192), B
   (193), C (194) and D (195); the two deliberate differences decided there (the acetylated
   N-terminus keeps its residue-1 amide; HX-Express spectra read from the labelled column and
   the one to its right) are in "Deliberate differences from pyHXExpress" above. The BIC's
   sample size keeps every picked bin (tried without the end zero bins and reverted: they have
   meaning; HISTORY 195). Open: whether to recommend `measure = "height"` for unresolved
   envelopes (untested: the top-down example is resolved).
15. **Measure how much wider than a binomial real envelopes are** (owner, 2026-10-30; TODO,
   scratch first). A scratch simulation (stick level, Q-TOF parameters) found that envelopes
   wider than a binomial are read as two populations: with an intra-class correlation of 0.03
   (deuterium sd 6-25 % wider for 5-20 amides), 58-74 % of single-population peptides with 16
   or more amides were split, and the population test kept most of the splits (its null draws
   binomial envelopes). At 0.01 almost nothing changed. Whether real data sit near either value
   is unknown. The clean approach: fix n from the sequence (the exchangeable amides, without
   prolines and the N-terminal residue, as `chem.exchangeable_sites` with `nterm_subtract`
   gives) and fit p and rho of a beta-binomial (mean n p, variance n p (1 - p) (1 + (n - 1)
   rho); `scipy.stats.betabinom`, no new dependency), convolved with the natural envelope like
   the binomial. With n free, n and rho trade off (both widen the envelope), so fixing n is what
   makes rho measurable. Fit it to the fully deuterated controls (one population, amides nearly
   equal): uneven back-exchange between amides can only narrow an envelope, so a fitted rho > 0
   there is a lower bound on the spread between molecules that does not depend on the
   conformation (back-exchange spread across the elution peak, uneven labelling of the control).
   Spread from conformations (cooperative opening, a continuum of conformers) cannot show in a
   fully deuterated control, where every amide has exchanged; it exists only in labelled
   spectra, where it is confounded with populations. Report rho against length and charge on a
   Q-TOF and an Orbitrap dataset, beside a binomial with the same fixed n. Only then decide
   whether the model or the population test's null needs a width term.
   Measured 2026-10-30 (scratch): the method recovers rho 0, 0.01 and 0.03 on simulated
   controls. On four Q-TOF datasets the typical fully deuterated envelope is as wide as the
   binomial (median rho 0; upper quartile about 0.01), 5-27 % of controls (by dataset) are more
   than 10 % wider, and the width belongs to the peptide (it repeats across replicates and
   across charge states, and grows with charge). On an Orbitrap dataset the envelopes are
   narrower than the binomial (median 0.85 of its sd). Peptides with wider controls read as two
   populations more often in their labelled spectra. Open (owner): a flag, or carrying each
   series' measured width into its fits and the population test's null, or nothing. The
   wide controls of one protein turned out to be deamidation during the heated FD preparation
   (HISTORY 196; an opt-in correction from the peaks' m/z is built, HISTORY 197), and the FD
   stand-ins were improved
   (the same peptide in other samples, a retention map; built, HISTORY 196).

## Risks and open decisions

- **Back-exchange correction of fragment ions.** Fragment series are not corrected for now
  (owner, 2026-10-21, HISTORY 131): values in deuterons, as measured. The correction may
  belong at the parent level, applied to the fragments in a way still to be decided (owner,
  2026-10-15); one candidate is measuring each fragment against the parent's deuterium in
  the same scans (owner, 2026-10-21, HISTORY 130).
- **Modifications in sequence files.** FASTA cannot carry the modX labels; modifications come
  from the peptide list. A sequence format with modifications is a future consideration
  (owner, 2026-10-15; HISTORY 79).

- **Exchangeable amide count for fragment ions: by structure, awaiting the owner's
  confirmation (HISTORY item 65).** A fragment holds the backbone N-H of its residues, placed in
  the parent, plus at the cleavage site the next residue's for c ions and its own first
  residue's for x and y ions (not for a, b, z); these are the backbone nitrogens of each ion
  type's composition, checked for every ion type and length in `tests/test_fragments.py`.
  Prolines and the parent's first `nterm_subtract` residues do not count. Decided by the owner
  (HISTORY item 66): neutral-loss variants follow their base ion (a lost NH3 is mostly a side
  chain's or the N-terminal amine's), except `c-NH3`, counted as a b ion; in `c-1` the hydrogen
  given to the z partner is not an amide's (measured); b ions hold L - 1 (linear; an oxazolone
  would hold L - 2, `AmideRules({"b": False})`). Still UNVERIFIED: the neutral-loss rule itself
  (use those ions in residue differences only after checking) and the hydrogen received by
  `z+1` (no data). Collisional (CID, HCD) b and y ions are scrambled and are not for residue
  values. `AmideRules` sets the cleavage-site amide of any ion type.
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
  (`NoiseConfig(estimate='replicates')`, HISTORY "Phase 2 status"). Decided (owner, 2026-10-13):
  the Gaussian loss, the default; the replicate estimate fits with the Poisson loss, which
  splits K=2 badly on real data (HISTORY items 49 and 55).
- **Dependency footprint.** PyTorch is large. CPU-only wheels help, but an install story for
  Windows and Mac users needs testing.
  Update: CI installs and tests the CPU wheels on Windows and macOS (HISTORY "Phase 0 status").
- **Thermo raw** needs a vendor library or a third-party reader, with licensing to check.
  Update: `.raw` files are converted outside hxex (ThermoRawFileParser) and read as mzML; reading
  `.raw` directly stays a Phase 6 item (item 33).
  Update: `read_thermo` reads `.raw` files directly with Thermo's RawFileReader through
  pythonnet (optional extra `hxex[thermo]`; the assemblies are the user's), item 43. A GPL
  section 7 permission for linking with the RawFileReader is in `LICENSE-EXCEPTION`. Open:
  Thermo's clause 2.1 (no "reflection"; pythonnet calls .NET through reflection, as other
  open-source readers that use these assemblies do).
- **CPU time for large bootstrap runs** may need user guidance (smaller N, fewer restarts).
  Update: confirmed by the benchmark (item 32): the CPU is fine for fits but not for full
  bootstraps at that size; float32 and less single-threaded Python are possible speed-ups, not
  done.
  Update: the CPU fit of that benchmark went from 77.4 to 17.1 min (item 46: the integer
  convolution, float32, measured chunk memory). The command line takes a thread count (12 beat
  the default 24 by 7 to 15 % on a 24-core CPU). Not done: the k-means starts (single-threaded
  Python, about 10 % of a fit).
  Update: the model step is about twice as fast (one evaluation of the pmf, one convolution of
  the mixture), the k-means starts are batched, and the optional screen skips the K that
  cannot be selected (item 85).
- **Memory and plots at 10,000+ fits:** figures must be selective by default.
  Resolved: `PlotConfig` draws the `flagged` series by default and `write_figures` makes and
  releases one series at a time (item 31).

## Decisions needed

1. The remaining Phase 0 items above (initial commit, creating the GitHub repository, reserving
   the PyPI name, confirming the SPDX identifier), and what to start in Phase 1. (Name,
   repository, hosting, license and minimum Python version are decided.)
   Resolved except reserving the PyPI name: the repository exists and is public, the SPDX
   identifier is `GPL-3.0-or-later`, and Phase 1 is done (see "Status").
