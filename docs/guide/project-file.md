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
hxex report hxex.toml --figures              # figures ([figures] in the file)
hxex analyse hxex.toml                       # post-fitting analysis ([analysis] in the file)
```

1. **`hxex init`** writes a file for one input format with every setting at its default and a
   comment on each. Optional stages (bootstrap, controls) are written commented out:
   uncomment a table to turn its stage on.
2. **`hxex check`** reads the file, checks every value's type and range, and lists what would be
   read: entries and control files per sample for the folder formats, each run's role, time and
   matching scans for the scan formats, and the resolving power recorded in Thermo files. It
   exits with an error if it finds a problem, and fits nothing.
3. **`hxex run`** reads the spectra, picks them, fits each number of populations from `k_min` to
   `k_max` (`[run.fit]`, default 1 to 3) to each and chooses the best, runs the bootstrap and the
   controls if they are on, and writes the bundle and the CSV tables. Progress
   goes to the terminal; Ctrl-C stops cleanly and keeps the fits if the bootstrap had started
   (`hxex resume` finishes it). It draws no figures.
4. **`hxex report`** draws the figures from a bundle, with the project file's `[figures]`
   settings when given the project file, and writes the tables again. Changing the figure
   settings needs only `hxex report`, never a new run.
5. **`hxex analyse`** computes the post-fitting analysis from the bundle: coverage, residue maps,
   populations, comparisons, protection factors and fragment ladders.

See [Outputs](outputs.md) for what each file holds.

## `run`, `report`, and `analyse`: what each one does

These three commands make the results, and each makes a different kind:

| | `hxex run` | `hxex report` | `hxex analyse` |
|---|---|---|---|
| What it does | reads your spectra and fits them | draws the fits, spectrum by spectrum | combines the fits: peptides into residues, samples against the reference |
| What it reads | your data files, with `[input]` and `[run]` | the results (the bundle), with `[figures]` | the results (the bundle), with `[analysis]` |
| What it writes | the results (`results`) and the tables (`results_tables`) | the figures (`results_figures`), and the tables again | the analysis (`results_analysis_WT`) |
| Fits anything | yes | no | no |
| How long it takes | minutes to hours: the fit, and the bootstrap most of all | seconds for the tables; the figures about a second per peptide drawn, so an hour or more for every peptide of a large run | seconds to minutes |
| Run it again after changing | `[input]` or `[run]` | `[figures]`, or the exclusion rules | `[analysis]` |

- **Only `hxex run` fits.** `hxex report` and `hxex analyse` only read the results it wrote,
  and they never change those results.
- **The figures can be slow too.** By default `hxex report` draws only the peptides that need a
  look (a quality flag, a rejected spectrum, a missing control), which keeps it quick.
  `series = "all"` draws every peptide: on a run of a few thousand peptides that takes an hour
  or more. Draw them all once you are happy with the fits, not after every change.
- **`hxex report` shows the fits as they are:** for each peptide, its spectra with the fitted
  populations, how the populations change over time, and its uptake. This is where you check
  whether the fits are good. It also writes the tables again (on its own, `hxex report
  hxex.toml` writes only the tables; add `--figures` for the figures), for example after you
  have added an exclusion rule.
- **`hxex analyse` goes further than the fits:** it combines replicates, charge states, and
  overlapping peptides into values per residue, decides residue by residue whether one or two
  populations describe the data, and compares each sample with the reference. These are the
  results you would usually report.
- The order matters only the first time: `hxex run` first, then `hxex report` and `hxex
  analyse` in either order. After that, rerun only the command whose settings you changed.

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
n_boot = 20

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
| `hxex run PROJECT` | read, pick, fit, bootstrap; write the bundle and tables, no figures. Slow: minutes to hours, most of it the bootstrap |
| `hxex resume BUNDLE` | finish a run that stopped during its bootstrap. Slow: the rest of the bootstrap |
| `hxex resume PROJECT` | add a bootstrap (the project's `[run.bootstrap]`) to its bundle, or replace the one there; the fits are kept. Slow: a whole bootstrap |
| `hxex report PROJECT` or `hxex report BUNDLE` `[--tables [DIR]] [--figures [DIR]] [--series flagged\|all] [--k K] [--format F]` | figures and tables from a bundle (given the project file: its bundle, `[figures]` settings and exclusion rules); fits nothing. The tables take seconds; the figures about a second per peptide drawn, so an hour or more with `--series all` on a large run |
| `hxex analyse PROJECT` or `hxex analyse BUNDLE` or `hxex analyse A=runA B=runB` | the post-fitting analysis (`--out`, `--reference`, `--format`); fits nothing. Seconds to minutes |
| `hxex info BUNDLE [--verify]` | what a bundle holds: counts, K, bootstrap, controls, roles, times, review, what it replaced, earlier generations |

`hxex init` refuses to write over an existing project file, so your edits are not lost by
accident; `--force` replaces it with a fresh template.

Options of `hxex run`:

- `--bundle PATH` writes to another bundle than `[output] bundle`.
- `--overwrite` replaces an existing bundle and its tables (the review is carried over).
- `--keep-previous` moves them to the next generation (`results.1`, ...) first.
- `--spectra PATH` fits the raw spectra stored in an earlier bundle instead of reading the input.
- `--device cpu|cuda|cuda:N` chooses where the fit runs.
- `--threads N` sets the CPU threads (about half the physical cores was fastest).

Every command takes `-q` (warnings only), `-v` (more messages) and `--progress-every SECONDS`.
Exit codes: 0 done, 1 error, 2 usage error, 130 stopped.

## Speed

The bootstrap takes most of a run's time. Each resample refits every spectrum once, at its
chosen number of populations and from the fitted values, so one resample costs about a sixth of
the fit. With the default `n_boot = 20`, the bootstrap takes about 3 times as long as the fit; a
fixed-K bootstrap at K = 1 and 2 (`ks = [1, 2]`, which the analysis' choice between one and two
populations needs) about 7 times. Twenty resamples give usable standard errors; raise `n_boot`
(to 100 or more) when the 95 % intervals themselves matter, for a published figure, say.

To save time:

- `[run.fit] screen = true` fits K=2 and K=3 only to spectra that could select them (the selected
  K does not change), in about a third of the time. The skipped fits are missing from fixed-K
  analyses.
- `[run.fit] k1_adam_steps = 200` as well: about a quarter of the default time, with the same
  selected K for all but a few spectra.
- A GPU (`[run.fit.backend] device = "cuda"`, or `--device cuda`) with a CUDA build of PyTorch.

## Reviewing results

After a run, you will want to look at the fits and decide which spectra to trust. *hxex* keeps
those decisions without changing the fits:

- **To leave spectra out**, write rules in the project file: before fitting with
  `[[input.exclude]]`, or after it with `[[analysis.exclude]]`. See
  [Excluding spectra](excluding.md).
- **To see every decision**, open `review.csv` in the tables folder (`results_tables`). It has
  one row per reviewed spectrum: its sample, peptide, charge, time, and replicate, its status
  (`rejected` or `accepted`), the number of populations chosen for it (`review_K`, empty when
  none was), the note, when it was made, and where it came from (`exclusion rule`, or
  `review` for a decision stored in the bundle). The file is there only when there is something
  to list.
- In `spectra.csv`, `review_status` says whether a spectrum is rejected, and `review_K` shows a
  number of populations chosen in the review: the `K` column then uses it instead of the fit's
  choice.

The bundle can also store decisions of its own, its *review layer*: a status, a chosen number
of populations, and a note per spectrum. These are kept when the run is repeated with
`--overwrite` (decisions the new run cannot apply are kept but marked stale). For now they are
entered from Python ([Using *hxex* from Python](python.md#reviewing-spectra)); choosing the
number of populations for a spectrum without Python is planned.

## From Python

Programmers can run every step from their own scripts: see
[Using *hxex* from Python](python.md).
