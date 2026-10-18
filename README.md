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
hxex analyse A=runA.zip B=runB.zip --reference A:WT   # compare separate runs
```

The output bundle (`[output] bundle`, say `results`) and the folders named after it
(`results_tables`, `results_figures` and the generations `results.1`, `results.1_tables`, ...)
are hxex's: a later `hxex run` refuses to write over them unless told to, and then either
replaces them (`--overwrite`; the review layer is carried over) or moves them to the next
generation first (`--keep-previous`). Files of your own inside them are kept, but one named
like an hxex file stops the run; keep your own files elsewhere. `hxex info` shows what a
bundle replaced, what changed and its earlier generations (see `docs/BUNDLE.md`).

Formats: `specexport`, `hxexpress`, `pooled`, `mzml` and `thermo`. Progress goes to stderr;
Ctrl-C stops a run cleanly, keeping its fits if the bootstrap had started. On the CPU,
`--threads N` (or `threads = N` in the project file) sets the number of threads; about half the
physical cores was fastest on a 24-core machine.
With `screen = true` in `[run.fit]`, K = 2 and 3 are fitted only for the spectra that could
select them (an exact bound; the selected K does not change), which takes about a third of the
time, but the skipped fits are missing from fixed-K analyses (K = 2 descriptions, fixed-K
bootstraps). When speed matters, add `k1_adam_steps = 200` (fewer Adam steps for K = 1 only):
together about a quarter of the default time, with the same selected K for all but a few spectra
and undeuterated control means within a few hundredths of a deuteron.

## Thermo raw files (optional)

`hxex.io.read_thermo` reads Thermo `.raw` files directly, without converting them to mzML. It
uses Thermo's RawFileReader, which is not part of hxex and is under Thermo's own license. It
needs:

- `pip install "hxex[thermo]"` (pythonnet);
- a .NET runtime: Mono on Linux and macOS, the .NET Framework on Windows (built in), or .NET 8
  or newer with `PYTHONNET_RUNTIME=coreclr`;
- `ThermoFisher.CommonCore.Data.dll` and `ThermoFisher.CommonCore.RawFileReader.dll`, for
  example from a [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser) release,
  in a directory named by `HXEX_THERMO_DIR` (or passed as `thermo_dir`).

## License

GNU General Public License, version 3 or (at your option) any later version
(`GPL-3.0-or-later`; see [`LICENSE`](LICENSE) for the version 3 text). pyHXExpress's repository
declares GPL-3.0 without the "or later" option.

Additional permission under GPL version 3 section 7: hxex may be linked or combined with Thermo
Fisher Scientific's RawFileReader, which is not GPL, and the result conveyed (see
[`LICENSE-EXCEPTION`](LICENSE-EXCEPTION)). RawFileReader itself stays under Thermo's license.
