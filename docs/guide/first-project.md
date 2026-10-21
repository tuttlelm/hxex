# Your first project

This page takes you from your exported spectra to results, one step at a time. It assumes *hxex*
is installed ([Installing *hxex*](install.md)) and that you have spectra exported from HDExaminer
(a SpecExport folder) or HX-Express. For raw instrument files, see [Input formats](inputs.md)
before step 3.

## Step 1: make a folder for the project

Make a new folder for this analysis, for example `Documents/hxex/my_protein`, in the usual way
(File Explorer on Windows, Finder on macOS). *hxex* will write its results into this folder. Your
data can stay where it is.

## Step 2: open a terminal in that folder

Commands act on the folder the terminal is "in". To open a terminal in your project folder:

- **Windows**: open the folder in File Explorer, click the address bar at the top, type
  `powershell` and press Enter.
- **macOS**: in Terminal, type `cd ` (with a space after it), drag the folder from Finder onto
  the Terminal window, and press Enter.
- **Linux**: right-click inside the folder and choose **Open in Terminal**, or type `cd ` and
  drag the folder in.

## Step 3: write a project file

The *project file* is a short text file that tells *hxex* what to read and what to do. *hxex* writes
one for you, with every setting at its usual value:

```bash
hxex init --format specexport
```

For HX-Express spreadsheets, write `--format hxexpress` instead. This makes a file called
`hxex.toml` in the folder.

## Step 4: edit the project file

Open `hxex.toml` in a plain text editor:

- **Windows**: right-click the file, choose **Open with**, then **Notepad**.
- **macOS**: right-click the file, choose **Open With**, then **TextEdit**. (If you make a new
  file in TextEdit, choose **Format > Make Plain Text** first.)
- A free editor such as [Notepad++](https://notepad-plus-plus.org/) (Windows) or
  [Visual Studio Code](https://code.visualstudio.com/) (all systems) shows the file in colour,
  which makes it easier to read.

Do not open it in Word: it must stay plain text.

The file is written in TOML, a simple format for settings: one setting per line, as
`name = value`. Text values go between quotes (`"WT"`), numbers do not (`0.9`), and switches are
`true` or `false`. Everything after a `#` on a line is a comment for you to read; *hxex* ignores
it. A line that starts with `#` is a setting that is switched off: delete the `#` to switch it
on.

Settings are grouped under headings. A heading's settings are the lines below it, up to the
next heading.

- **Single brackets**, such as `[input]`, are for something there is only one of: the project
  has one input, so `[input]` appears once.
- **Double brackets**, such as `[[analysis.regions]]`, are for something there can be several
  of. Each time the heading appears, it starts a new one, described by the lines below it.

For example, to mark two regions of the protein in the figures, write the heading twice, once
for each region:

```toml
[[analysis.regions]]
name = "N-terminal region"
start = 1
end = 63

[[analysis.regions]]
name = "core domain"
start = 64
end = 150
```

The same goes for `[[input.runs]]` (one per data file) and `[[analysis.sites]]` (one per site).
If there is only one, single brackets work too: `[analysis.regions]` is read as one region.

You only need to change three lines at first:

```toml
project = "my_protein"          # a short name for this analysis (no colon)

[input]
format = "specexport"
data_dir = "C:/Users/me/Data/SpecExport"   # where your exported spectra are

[output]
bundle = "results"              # name of the folder (or .zip archive) hxex creates for the results
```

**What a bundle is.** *hxex* calls its results folder a *bundle*: one folder that holds
everything a run produced (every fit, its uncertainties when the bootstrap is on, the settings
used, and a copy of the raw spectra) in *hxex*'s own file format. You do not open the files in
it yourself; the other commands (`hxex report`, `hxex analyse`, `hxex info`) read it, and the
spreadsheets and figures you look at are written beside it. Keep it: it is the record of the
analysis, and a later run can start again from the spectra stored in it.

**Writing folder names.** Always put a folder name between quotes. A name on its own, such as
`"SpecExport"`, means a folder inside the one that holds `hxex.toml`; anything else needs its
full location ([File and folder locations](project-file.md#file-and-folder-locations) has the
details).

- **Works on every system:** double quotes, with forward slashes `/` between the folders.
  - Windows: `"C:/Users/me/Data/SpecExport"`
  - macOS: `"/Users/me/Data/SpecExport"`
  - Linux: `"/home/me/Data/SpecExport"`
- **A folder inside the project folder:** just its name, `"SpecExport"`, on every system.
- **Windows only, backslashes:** Windows writes locations with backslashes `\`. A location
  copied with backslashes must go between *single* quotes: `'C:\Users\me\Data\SpecExport'`.
  Between double quotes, backslashes give an error.

To copy a folder's location:

- **Windows:** open the folder in File Explorer, click the address bar at the top, and copy
  the text (it has backslashes, so use single quotes, or change each `\` to `/`).
- **macOS:** select the folder in Finder and press Option + Cmd + C.
- **Linux:** in most file managers, select the folder, press Ctrl + C, and paste into the
  project file: the folder's location is pasted.

Save the file.

## Step 5: check it

```bash
hxex check hxex.toml
```

*hxex* reads the settings and looks at your data without analysing anything. It lists the samples
and peptides it found and the control files of each sample, and ends with `ok`. If something is
wrong, it says what and where, for example:

```
input.data_dir C:/Users/me/Data/SpecExprt: no such directory
1 problem(s) found
```

Here the folder name is misspelt (`SpecExprt` for `SpecExport`). The start of the message,
`input.data_dir`, says where to look: the `data_dir` line under `[input]` in `hxex.toml`. Correct
the folder name there, save the file, and run `hxex check hxex.toml` again. Repeat until it says
`ok`.

## Step 6: run it

```bash
hxex run hxex.toml
```

*hxex* reads every spectrum, fits it, and writes the results. It prints a line every half minute
to show how far it has got. A few hundred peptides take minutes to an hour, depending on the
computer and the settings. To stop it, press Ctrl + C. (Once the fits are done and the
bootstrap has begun, they are kept, and `hxex resume results` finishes the run later; stopped
earlier, the run starts again from the beginning.)

When it has finished, the project folder holds:

- `results`: the bundle, the results themselves in *hxex*'s own format (keep it; every later
  step reads it);
- `results_tables`: the results as spreadsheets (`.csv` files, which Excel opens).

## Step 7: look at the figures

```bash
hxex report hxex.toml --figures
```

This draws the figures into `results_figures`: an overview, and for each peptide that needs a
look (a poor fit, a missing control) its spectra and uptake. To draw every peptide, add
`--series all`, or set `series = "all"` in the `[figures]` part near the end of `hxex.toml`.
Open the `.pdf` files as usual. `hxex run` never draws figures, so changing how they look needs
only this command again, not a new run.

## Step 8: compare samples

To compare your samples with a reference sample, open `hxex.toml` again, find the commented-out
`[analysis]` group near the end, and switch on these two lines by deleting their `#`:

```toml
[analysis]
reference = "WT"                # the name of your reference sample
```

Then run:

```bash
hxex analyse hxex.toml
```

The comparisons, residue maps and their figures go to `results_analysis_WT`: the folder is
named after the reference. To compare with another reference as well, run
`hxex analyse hxex.toml --reference Mut`; it writes to `results_analysis_Mut`, beside the first,
without touching it. (Without a reference, the folder is `results_analysis`.)

After these steps, the project folder holds:

```text
my_protein/
├── hxex.toml               your project file
├── results/                the bundle: the results in hxex's own format (keep it)
├── results_tables/         the results as spreadsheets (.csv files)
├── results_figures/        the figures (step 7)
├── results_analysis_WT/    the comparisons with WT (step 8)
└── results_analysis_Mut/   the comparisons with Mut, if you ran that too
```

The folders whose names start with `results` belong to *hxex*: later runs replace them (see
below), so keep files of your own elsewhere. [Outputs](outputs.md) lists what is in each.

## Running it all again

Once `hxex.toml` holds everything (the input, the `[analysis]` settings, and any other changes),
the whole analysis takes three commands, one for each kind of result:

```bash
hxex run hxex.toml --overwrite
hxex report hxex.toml --figures
hxex analyse hxex.toml
```

- `hxex run` fits and writes the results and their tables (step 6), `hxex report` draws the
  figures (step 7), and `hxex analyse` makes the analysis (step 8). Each reads `hxex.toml`.
  [What each one does](project-file.md#run-report-and-analyse-what-each-one-does) sets the
  three side by side.
- After changing only the `[figures]` settings, run only `hxex report`; after changing only
  the `[analysis]` settings, only `hxex analyse`. Neither fits anything again.
- `--overwrite` replaces the earlier results. Without it, `hxex run` stops because `results`
  already exists; use `--keep-previous` instead to keep the earlier results as `results.1`.
- `hxex analyse` replaces the files of the earlier analysis with the same reference, in
  `results_analysis_WT` (files of your own there are kept). With `--keep-previous`, the earlier
  analyses move along with the earlier results, to `results.1_analysis_WT`.

Run `hxex check hxex.toml` first whenever you have changed the file.

## What next

- Better uncertainties: switch on `[run.bootstrap]` in the project file, then run
  `hxex resume hxex.toml`. It adds the bootstrap to your results without fitting again. It
  takes much longer than the fit, but gives the error bars the comparisons need. Run it again
  after changing the bootstrap settings (more resamples, say) to replace the earlier one; with
  the settings unchanged it keeps the bootstrap and adds only what is missing, such as the
  spectra you gave another number of populations in the
  [review table](project-file.md#the-user-review-table).
- Back-exchange correction from your controls: switch on `[run.controls]`. See
  [Back-exchange correction](back-exchange-correction.md).
- Leaving out spectra that look bad, without deleting them: [Excluding spectra](excluding-spectra.md).
- What every table and figure holds: [Outputs](outputs.md).
- Every setting: [Settings reference](settings.md).
