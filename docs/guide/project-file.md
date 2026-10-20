# The project file and the commands

A run is described by one TOML file, usually `hxex.toml`. It says what to read (`[input]`),
how to process it (`[run]` and its sub-tables), where the results go (`[output]`) and, optionally,
what to analyse afterwards (`[analysis]`). The same file can be run again later and gives the
same results; the bundle records the settings it was made with.

## A session from start to finish

```bash
hxex init --format specexport -o hxex.toml   # a project file with every default, commented
#   edit hxex.toml: the project name, [input] data_dir, [output] bundle
hxex check hxex.toml                         # what would be read, and any problems
hxex run hxex.toml                           # read, pick, fit (and bootstrap); bundle and tables
hxex report results --figures                # figures from the bundle
hxex analyse hxex.toml                       # post-fitting analysis ([analysis] in the file)
```

1. **`hxex init`** writes a file for one input format with every setting at its default and a
   comment on each. Optional stages (bootstrap, controls, plots) are written commented out:
   uncomment a table to turn its stage on.
2. **`hxex check`** reads the file, checks every value's type and range, and lists what would be
   read: entries and control files per sample for the folder formats, each run's role, time and
   matching scans for the scan formats, and the resolving power recorded in Thermo files. It
   exits with an error if it finds a problem, and fits nothing.
3. **`hxex run`** reads the spectra, picks them, fits each number of populations from `k_min` to
   `k_max` (`[run.fit]`, default 1 to 3) to each and chooses the best, runs the bootstrap and the
   controls if they are on, and writes the bundle and the CSV tables. Progress
   goes to the terminal; Ctrl-C stops cleanly and keeps the fits if the bootstrap had started
   (`hxex resume` finishes it).
4. **`hxex report`** writes tables and figures again from a bundle, for example with other plot
   settings.
5. **`hxex analyse`** computes the post-fitting analysis from the bundle: coverage, residue maps,
   populations, comparisons, protection factors and fragment ladders.

See [Outputs](outputs.md) for what each file holds.

## How the file works

```toml
project = "my_project"            # prefixes every spectrum id
# threads = 12                    # CPU threads

[input]                           # what to read
format = "specexport"
data_dir = "SpecExport"

[run.fit]                         # change only what differs from the default
k_max = 3

[run.bootstrap]                   # a table present turns its stage on
n_boot = 200

[run.controls]
d2o_fraction = 0.9

[output]
bundle = "results"                # the folder (or .zip archive) hxex creates for the results

[analysis]
reference = "WT"
```

- **A key left out keeps its default.** The [settings reference](settings.md) lists every key with
  its default.
- **`false` means unset** where a setting can be unset (an optional stage, a limit, a scan
  condition): TOML has no null. A switch set to `false` is off. `bootstrap = false` in `[run]`
  turns the bootstrap off.
- **Times** are seconds, or text with a unit: `time = "5 min"`, `times = ["1 min", "20 h"]`.
  Retention times (`rt`, `rt_start`, `rt_end`) are minutes. The bootstrap's `ci` is a
  percentage (95.0); the analysis' `confidence` is a fraction (0.5).
- **Paths** are relative to the project file.
- **Unknown keys are refused**, with the table and key named, so a misspelt setting does not pass
  silently. Settings that were renamed or removed are refused with a message saying what replaces
  them.

## Commands

The main commands and options; every option is in the
[command-line reference](commands.md).

| Command | Does |
|---|---|
| `hxex init [--format F] [--project NAME] [-o FILE] [--force]` | write a project file with every default (`-o -` prints it) |
| `hxex check PROJECT [--spectra PATH]` | check the file and show what would be read |
| `hxex run PROJECT` | read, pick, fit, bootstrap; write the bundle, tables and figures |
| `hxex resume BUNDLE` | finish a run that stopped during its bootstrap |
| `hxex report BUNDLE [--tables [DIR]] [--figures [DIR]] [--series flagged\|all] [--format F]` | tables and figures from a bundle |
| `hxex analyse PROJECT` or `hxex analyse BUNDLE` or `hxex analyse A=runA B=runB` | the post-fitting analysis (`--out`, `--reference`, `--format`) |
| `hxex info BUNDLE [--verify]` | what a bundle holds: counts, K, bootstrap, controls, roles, times, review, what it replaced, earlier generations |

`hxex init` refuses to write over an existing project file, so your edits are not lost by
accident; `--force` replaces it with a fresh template.

Options of `hxex run`:

- `--bundle PATH` writes to another bundle than `[output] bundle`.
- `--overwrite` replaces an existing bundle, its tables and figures (the review is carried over).
- `--keep-previous` moves them to the next generation (`results.1`, ...) first.
- `--spectra PATH` fits the raw spectra stored in an earlier bundle instead of reading the input.
- `--device cpu|cuda|cuda:N` chooses where the fit runs.
- `--threads N` sets the CPU threads (about half the physical cores was fastest).

Every command takes `-q` (warnings only), `-v` (more messages) and `--progress-every SECONDS`.
Exit codes: 0 done, 1 error, 2 usage error, 130 stopped.

## Speed

The bootstrap takes most of a run's time. To save time:

- `[run.fit] screen = true` fits K=2 and K=3 only to spectra that could select them (the selected
  K does not change), in about a third of the time. The skipped fits are missing from fixed-K
  analyses.
- `[run.fit] k1_adam_steps = 200` as well: about a quarter of the default time, with the same
  selected K for all but a few spectra.
- A GPU (`[run.fit.backend] device = "cuda"`, or `--device cuda`) with a CUDA build of PyTorch.

## Reviewing results

The bundle has a **review layer**: per spectrum, a status (rejected or not), an optional K
override and a note. Rejected spectra are left out of the uptake, the reports and the analysis;
an override replaces the selected K. The review is kept when the bundle is overwritten by a new
run; entries the new run cannot hold are kept but marked stale. To leave spectra out without
Python, write rules in the project file ([Excluding spectra](excluding.md)); single review
decisions are entered from Python ([Using *hxex* from Python](python.md#reviewing-spectra)).

## From Python

Programmers can run every step from their own scripts: see
[Using *hxex* from Python](python.md).
