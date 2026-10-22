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
  percentage (95.0); the analysis' `confidence` is a fraction (0.5, or 0.95 with
  `populations = "bootstrap"`).
- **Paths**: see [File and folder locations](#file-and-folder-locations) below.
- **Unknown keys are refused**, with the table and key named, so a misspelt setting does not pass
  silently. Settings that were renamed or removed are refused with a message saying what replaces
  them.

## File and folder locations

Wherever a setting names a file or a folder (`data_dir`, `peptides`, `bundle`, `fasta`, the
review `table`, ...), you can write either its full location or a short form:

| Where it is written | The short form is counted from |
|---|---|
| any file or folder setting in `hxex.toml` | the folder that holds `hxex.toml` |
| `path` in `[[input.runs]]` | `data_root` (by default, the folder that holds `hxex.toml`) |
| the file names inside an HX-Express file table (`[input] peptides`) | `data_dir` |
| the command line (`--bundle`, `--spectra`, `--out`, a bundle named directly, ...) | the folder the terminal is in |

- **The short form** is a name such as `"SpecExport"` or `"reviews/run1.csv"` (a folder inside
  the one it is counted from), or one that starts with `../` for the folder above.
- **A full location** works everywhere, in `hxex.toml` and on the command line: for example
  `"C:/Users/me/Data/SpecExport"` on Windows or `"/Users/me/Data/SpecExport"` on macOS. `~`
  stands for your home folder (`"~/Data/SpecExport"`).
- **Windows backslashes:** a location copied from File Explorer has backslashes `\`. In
  `hxex.toml`, put it between single quotes (`'C:\Users\me\Data'`), or change each `\` to `/`;
  between double quotes, backslashes give an error. See
  [Writing folder names](first-project.md#step-4-edit-the-project-file).
- The folder that holds `hxex.toml` and the folder the terminal is in are usually the same,
  because you open the terminal in the project folder. They differ only when you name a
  project file in another folder (`hxex run ../other/hxex.toml`): its settings are still found
  from its own folder, but the command line's paths are found from where you are.

## Commands

The main commands and options; every option is in the
[command-line reference](commands.md).

| Command | Does |
|---|---|
| `hxex init [--format F] [--project NAME] [-o FILE] [--force]` | write a project file with every default (`-o -` prints it) |
| `hxex check PROJECT [--spectra PATH]` | check the file and show what would be read |
| `hxex run PROJECT` | read, pick, fit, bootstrap; write the bundle and tables, no figures. Slow: minutes to hours, most of it the bootstrap |
| `hxex resume BUNDLE` | finish a run that stopped during its bootstrap. Slow: the rest of the bootstrap |
| `hxex resume PROJECT` | complete the bootstrap of the project's bundle, the fits kept: add one (the project's `[run.bootstrap]`) when there is none, redo it when its settings changed, and bootstrap the spectra whose number of populations the review table chose (only those). Slow: a whole bootstrap; only the chosen spectra when that is all that is missing |
| `hxex report PROJECT` or `hxex report BUNDLE` `[--tables [DIR]] [--figures [DIR]] [--series flagged\|all] [--k K] [--format F]` | figures and tables from a bundle (given the project file: its bundle, `[figures]` settings and exclusion rules); fits nothing. The tables take seconds; the figures about a second per peptide drawn, so an hour or more with `--series all` on a large run |
| `hxex analyse PROJECT` or `hxex analyse BUNDLE` or `hxex analyse A=runA B=runB` | the post-fitting analysis (`--out`, `--reference`, `--format`); fits nothing. Seconds to minutes |
| `hxex info BUNDLE [--verify]` | what a bundle holds: counts, K, bootstrap, controls, roles, times, review, what it replaced, earlier generations |

`hxex init` refuses to write over an existing project file, so your edits are not lost by
accident; `--force` replaces it with a fresh template.

Options of `hxex run`:

- `--bundle PATH` writes to another bundle than `[output] bundle`.
- `--overwrite` replaces an existing bundle and its tables (the review is carried over).
- `--keep-previous` moves them to the next generation (`results.1`, ...) first.

`hxex report` and `hxex analyse` take `--overwrite` and `--keep-previous` too, for their own
folders (`results_tables`, `results_figures`, `results_analysis_...`): without one of them, a
folder that already holds results stops the command. `overwrite = true` or `keep_previous =
true` in `[output]` applies to every command. See [Running again](outputs.md#running-again).
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

After a run, you will want to look at the fits and decide which spectra to trust, and for some
spectra how many populations to use. *hxex* keeps those decisions without changing the fits,
and you can make them in a spreadsheet.

### The user review table

1. Copy `spectra.csv` from the tables folder (`results_tables`) to a place of your own, and give
   the copy a name of your own. We suggest `user_review_table.csv`, in the project folder beside
   `hxex.toml`; any name and folder work, as long as step 4 names the same file. Keep it out of
   the folders whose names start with `results`, and do not edit `spectra.csv` itself: *hxex*
   replaces those whenever it writes the results.
2. Open the copy in Excel (or any spreadsheet program). It has one row per spectrum. Fill in, for
   the spectra you have a view on:
   - `review_status`: `rejected` to leave the spectrum out, or `accepted`. `unreviewed` or an
     empty cell means no decision. An `accepted` spectrum is also kept when `hxex analyse`
     would leave it out (it lists those in `left_out_spectra.csv`).
   - `review_K`: the number of populations to use for that spectrum, instead of the one the fit
     chose (`K`). Leave it empty to keep the fit's choice.
   - `note`: why (optional, but it is kept with the decision and listed in `review.csv`).
3. Save it as CSV (in Excel, **File > Save As**, type **CSV UTF-8**, and keep the name).
4. Tell *hxex* about it, once, in the `[review]` part of `hxex.toml`, with the file's name:

   ```toml
   [review]
   table = "user_review_table.csv"
   ```

   A name on its own means a file in the folder holding `hxex.toml`. For a file elsewhere, give
   its full location, written as described in [Writing folder names](first-project.md#step-4-edit-the-project-file),
   for example `table = "C:/Users/me/reviews/run1_review.csv"`.

   `[review]` holds the decisions you make after looking at the results: this table and the
   exclusion rules (`[[review.exclude]]`). `hxex run` (for its tables), `hxex report`, and
   `hxex analyse` all apply them.

5. Run `hxex report hxex.toml --figures --overwrite`. The figures now use your decisions.

**Which columns to touch.** *hxex* reads only four columns: `spectrum_id`, `review_status`,
`review_K`, and `note`.

- Edit only `review_status`, `review_K`, and `note`.
- Do not change `spectrum_id`: it is how *hxex* finds each spectrum. If a cell in it is changed,
  that row no longer matches a spectrum and is skipped (with a warning).
- Every other column (sample, peptide, time, `K`, flags, the means, ...) is there for you to sort
  and filter by. *hxex* ignores it, so changing it has no effect. You can delete columns and rows
  you do not need, as long as `spectrum_id` and the columns you filled in stay.
- A copy made after a `[[review.exclude]]` rule has `rejected` in the rows that rule left out.
  Clear those cells if the rule should stay the only reason: otherwise the spectra stay rejected
  even when you delete the rule.

**What the decisions change.** A rejected spectrum is left out of everything that uses the
results: `uptake.csv`, the figures (where it is drawn hollow and grey), and `hxex analyse`. A
chosen number of populations changes the run's tables and figures (`spectra.csv`,
`components.csv`, `uptake.csv`, and the spectrum, population, and uptake figures). It does not
change `hxex analyse`, which works from fits with one and with two populations for every
spectrum and decides residue by residue
([What each one does](#run-report-and-analyse-what-each-one-does)).

**Error bars for a chosen number of populations.** When the run had a bootstrap, every
spectrum's error bars come from it. A spectrum given another number of populations needs a
bootstrap at that number too, so that its errors compare with the others'. `hxex report` says
when one is missing, and

```bash
hxex resume hxex.toml
```

bootstraps exactly those spectra (nothing else is redone), after which `hxex report` works.
When the table is in place before a run, `hxex run` does this itself.

**Checking it.** `hxex check hxex.toml` reads the table and says how many spectra it rejects,
accepts, and gives a number of populations, or which row has a value it cannot read. `review.csv`
in the tables folder lists every decision in effect, with `source` saying where it came from
(`review table`, `exclusion rule`, or `review`).

### Other ways to record decisions

- **Rules in the project file** leave out whole groups of spectra at once (every spectrum of a
  data file, a peptide in one sample): `[[input.exclude]]` before fitting, `[[review.exclude]]`
  after it. See [Excluding spectra](excluding-spectra.md). A spectrum a rule excludes stays excluded,
  whatever the review table says.
- **The bundle's own review layer** stores decisions inside the results, entered from Python
  ([Using *hxex* from Python](python.md#reviewing-spectra)). They are kept when the run is
  repeated with `--overwrite` (decisions the new run cannot apply are kept but marked stale).
  The review table's decisions take the place of the stored ones for the same spectrum.

## From Python

Programmers can run every step from their own scripts: see
[Using *hxex* from Python](python.md).
