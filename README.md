<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tuttlelm/hxex/main/docs/images/hxex_logo_dark.svg">
    <img src="https://raw.githubusercontent.com/tuttlelm/hxex/main/docs/images/hxex_logo.svg" alt="hxex logo: the h is a bimodal spectrum" width="150">
  </picture>
</p>

# hxex

Multimodal binomial-mixture analysis of HDX-MS spectra: from spectra (raw or exported) to
binned deuterium sticks to fitted populations with uncertainties and plots.

**Status: early development.** Reading, picking, fitting, bootstrapping, reports and the
command line work; nothing is released yet. The project plan is in
[`docs/PLAN.md`](docs/PLAN.md).

`hxex` is the planned successor to
[pyHXExpress](https://github.com/tuttlelm/pyHXExpress). It is standalone and does not depend
on it.

**Documentation:** the [user guide](docs/guide/README.md), starting with
[installing hxex](docs/guide/install.md) (no Python knowledge needed) and
[your first project](docs/guide/first-project.md). The rest of this page is for developers.

## Quick start

```bash
pip install -e ".[dev]"                      # from a clone; see "Development install" for PyTorch
hxex init --format specexport -o hxex.toml   # a project file with every default, commented
#   edit hxex.toml: the project name, [input] data_dir, [output] bundle
hxex check hxex.toml                         # what would be read, and any problem
hxex run hxex.toml                           # read, pick, fit, bootstrap; bundle and tables
hxex report results --figures                # figures from the bundle
hxex analyse hxex.toml                       # post-fitting analysis ([analysis] in the file)
```

`hxex run` writes a results bundle (`results`, a directory or `.zip` of Parquet tables, see
[`docs/BUNDLE.md`](docs/BUNDLE.md)) and CSV tables beside it (`results_tables`: one row per
series, per spectrum, per population and per replicate-averaged population). The project file's
comments explain every setting and its unit; `hxex check` reports type and value problems with
the table and key.

## Inputs

Five formats, chosen by `[input] format` (in detail in
[docs/guide/inputs.md](docs/guide/inputs.md)):

- `specexport`: an HDExaminer SpecExport folder, `SAMPLE/START-END-SEQUENCE/TIME-REP-zCHARGE.csv`
  (`TIME` is `Non`, `Full` or a number with `s`, `m` or `h`).
- `hxexpress`: HX-Express spreadsheets (`.xlsx` or `.csv`), one per sample, peptide and charge,
  with a column per time (`undeut`, `TD`, `60 s`, `5 min`).
- `pooled`: a SpecExport folder with peptides of several proteins, each matched to its protein
  by sequence.
- `mzml`: mzML files, the scans of each peptide (and fragment ion) chosen by m/z, MS level,
  activation and retention time.
- `thermo`: Thermo `.raw` files read directly (see below); the same settings as `mzml`.

The scan formats need a **peptide table** (CSV; a folder format may give one instead of
finding files by name). Required columns: `peptide`, `start_seq`, `end_seq`, `charge`.
Optional: `sample` (the peptide is read from the runs of that sample; empty or left out, from
every run), `file` (for reference), `peptide_range` (the range's label; made from `start_seq`
and `end_seq` when left out, a warning when it names other residues), `modification`, `rt` (retention time, minutes),
`rt_start` and `rt_end` (the peptide's retention window, minutes). A folder format's table
needs `file` and `sample` as well. Sequences are written in
modX: a modified residue is a lowercase prefix plus its letter (`acK`, `pS`, `pT`, `pY`, or any
label defined in `[run.registry]`), and a sequence is read residue by residue.

**Controls.** The undeuterated and fully deuterated controls are identified by a *role*, not
by a time. The folder formats take it from the file labels (`Non`/`undeut` and `Full`/`TD`);
for the scan formats each control file gets `role = "undeut"` or `role = "fulldeut"` in
`[[input.runs]]` (a file with a `time` as well is a time point that also serves as the
control). A control carries no time in the tables and figures. `[run.controls]` turns the
back-exchange correction on; `undeut_at` and `fulldeut_at` there make the spectra at a time
serve as a control too (a plateaued late point as the fully deuterated reference). Replicate
controls are averaged, and a series whose control is missing or fails the limits falls back to
a theoretical value and is flagged (`control_fallback` in the tables). `hxex check` counts the
control files per sample and names each scan-format run's role; `hxex info` counts the control
spectra of a bundle. `[input] time_scale = "order"` says the times only order the points
(dummy labels such as 1, 2, 3): the uptake is then drawn by index, and the kinetics and
protection analyses refuse.

**Units.** Times are in seconds everywhere, including `rt_window`, and may be written with a
unit in the project file (`time = "5 min"`, `times = ["1 min", "20 h"]`; `s`, `sec`, `m`,
`min`, `h`, `hr`, `ms` and their long forms); stored values are seconds. Chromatographic
retention times (`rt` columns, `retention_times` tables) are in minutes. `resolution_ppm` is the picker's
half-width around each predicted peak, not a resolving power; `retention_min` and
`retention_max` are deuterium retention (a fraction), not retention times. The bootstrap's `ci`
is a percentage (95.0), the analysis' `confidence` a fraction (0.5).

### From Python

```python
import hxex

project = hxex.load_project("hxex.toml")
spectra = hxex.read_input(project)                      # raw spectra, with their read record
result = hxex.run(spectra, project.run, out="results", project=project.project)

with hxex.open_bundle("results") as bundle:             # later, or in a notebook
    results = hxex.Results.from_bundle(bundle)
hxex.write_tables(results, "results_tables")
hxex.write_figures(results, "results_figures", hxex.PlotConfig(series="all"))
```

Plot functions return Matplotlib figures and never show them. Long operations take `progress`
and `cancel` hooks (`hxex.hooks`).

## Requirements

- Python 3.11 or newer
- Runs on CPU or GPU (PyTorch)

## Development install

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only PyTorch (smaller than the default Linux wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

For an NVIDIA GPU, install a CUDA build of PyTorch that matches your driver instead of the CPU
line. Tested: the CUDA 12.6 build on an RTX A4500 with driver 535 (CUDA 12 minor-version
compatibility):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

These wheels are several gigabytes; on a slow connection add `--default-timeout=600` to `pip`.

## Command line

A run is described by a TOML project file:

```bash
hxex init --format specexport -o hxex.toml   # a project file with every default; edit it
hxex check hxex.toml                         # what would be read, and any problem
hxex run hxex.toml                           # read, pick, fit, bootstrap; bundle and tables
hxex run hxex.toml --keep-previous           # again, keeping the earlier results as results.1
hxex run hxex.toml --spectra old_results     # the same from the raw spectra a bundle stored
hxex info results                            # what the results bundle holds
hxex report results --figures                # figures (and --tables) from the bundle
hxex resume results                          # finish a run stopped during its bootstrap
hxex analyse hxex.toml                       # post-fitting analysis: coverage, residues,
                                             # populations, comparisons ([analysis])
hxex analyse A=runA.zip B=runB.zip --reference A:WT   # compare separate runs (a run column;
                                                      # RUN:sample names one run's sample)
```

The output bundle (`[output] bundle`, say `results`) and the folders named after it
(`results_tables`, `results_figures`, `results_analysis`, and the generations `results.1`,
`results.1_tables`, ...)
are hxex's: a later `hxex run` refuses to write over them unless told to, and then either
replaces them (`--overwrite`; the review layer is carried over) or moves them to the next
generation first (`--keep-previous`). Files of your own inside them are kept, but one named
like an hxex file stops the run; keep your own files elsewhere. `hxex info` shows what a
bundle replaced, what changed and its earlier generations (see `docs/BUNDLE.md`).

Progress goes to stderr; Ctrl-C stops a run cleanly, keeping its fits if the bootstrap had
started. On the CPU, `--threads N` (or `threads = N` in the project file) sets the number of
threads; about half the physical cores was fastest on a 24-core machine.
With `screen = true` in `[run.fit]`, K = 2 and 3 are fitted only for the spectra that could
select them (an exact bound; the selected K does not change), which takes about a third of the
time, but the skipped fits are missing from fixed-K analyses (K = 2 descriptions, fixed-K
bootstraps). When speed matters, add `k1_adam_steps = 200` (fewer Adam steps for K = 1 only):
together about a quarter of the default time, with the same selected K for all but a few spectra
and undeuterated control means within a few hundredths of a deuteron.

## Thermo raw files (optional)

`hxex init --format thermo` writes a project file whose `[[input.runs]]` name `.raw` files;
`hxex run` then reads them directly (`hxex.io.read_thermo`), without converting them to mzML.
This uses Thermo's RawFileReader, which is not part of hxex and is under Thermo's own license.
It needs:

- `pip install "hxex[thermo]"` (pythonnet);
- a .NET runtime: Mono on Linux and macOS, the .NET Framework on Windows (built in), or .NET 8
  or newer with `PYTHONNET_RUNTIME=coreclr`;
- `ThermoFisher.CommonCore.Data.dll` and `ThermoFisher.CommonCore.RawFileReader.dll`, for
  example from a [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser) release,
  in a directory named by `HXEX_THERMO_DIR` (or `[input] thermo_dir` in the project file,
  `thermo_dir` in Python).

## License

GNU General Public License, version 3 or (at your option) any later version
(`GPL-3.0-or-later`; see [`LICENSE`](LICENSE) for the version 3 text). pyHXExpress's repository
declares GPL-3.0 without the "or later" option.

Additional permission under GPL version 3 section 7: hxex may be linked or combined with Thermo
Fisher Scientific's RawFileReader, which is not GPL, and the result conveyed (see
[`LICENSE-EXCEPTION`](LICENSE-EXCEPTION)). RawFileReader itself stays under Thermo's license.
