# The *hxex* window

*hxex* also has a window of its own, for those who prefer pointing and clicking to typing
commands. It works on the same files as the commands: the project file, your tables, and the
results. Anything done in the window can also be done in a terminal, and the other way round, so
you can switch between them at any point.

The window is being built in steps. What it does today:

- **Project**: makes a new project file or opens one, and changes its settings (below).
- **Results**: opens results as a tree of peptides, their charges, time points, and
  replicates, and shows the figures of what you select (below).
- **Run**: runs *hxex*'s commands on the project (check, run, resume, report, analyse, info),
  with their messages, a progress bar, and a Stop button (below).
- **Analysis**: shows the figures `hxex analyse` would write with the project's settings, and
  why each figure left out is left out (below).

## Installing it

The window needs one extra piece of software (Qt, about 100 MB). Install *hxex* with the
`gui` extra, which is step 3 of [Installing *hxex*](install.md) with `"hxex[gui] @ "` before the
address:

```bash
uv tool install "hxex[gui] @ https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip"
```

On Linux, add the same `--index ... --index-strategy unsafe-best-match` ending as in step 3 if
you used it there. Some Linux systems also need a few of Qt's system libraries (`libegl1`,
`libxkbcommon0`); if the window does not open and the message names a missing library, ask
whoever looks after your computer to install it.

## Opening it

```bash
hxex gui
```

or, to open a project file, its results, or a saved session straight away:

```bash
hxex gui hxex.toml
```

`hxex-gui` does the same; on Windows it opens without a terminal window. The window's icon is
the *hxex* logo, in its light or dark version to match your desktop. The window's **Messages**
pane at the bottom shows what *hxex* reports, as a terminal would.

**File > Open** takes a project file (`.toml`), a session (`.hxexgui`, below), or results.
Opening a project file also opens its results when it has some, with your review table and
exclusion rules applied, as `hxex report hxex.toml` uses them. **File > Open results** opens a
results folder (a bundle) on its own.

## Settings

The **Project** part of the window shows the project file's settings, one table at a time (the
list on the left), each with its description, the text of the [settings reference](settings.md).
**File > New project file** asks for the input format and a name and writes a project file with
every setting at its default, as `hxex init` does.

- A setting the file sets is in **bold**; an empty box shows the default in grey. Type a value
  and press Enter (or click elsewhere): it is written to the project file at once, as `hxex set`
  writes it, so the file keeps your comments and its layout. **Default** puts the setting back
  to its default (a `#` in front of its line). Text needs no quotes.
- A setting that holds a file or a folder has **Browse…**; a file inside the project's folder
  is written relative to it, so the folder can be moved or copied.
- **Find a setting** searches every table by name and description.
- Under `[input]`, only the settings of the chosen format are shown (tick **Show the input
  settings of every format** to see them all). Changing the format offers to switch off the
  settings the new format does not use.
- A stage that its heading alone turns on (the bootstrap, the population test, the controls,
  the analysis, the review) has a **Use** box. Setting any of its settings also switches it on.
- A value that cannot be right (a word where a number belongs) is refused, and the Messages
  pane says why. A file you are still filling in can lack something (a peptide table, a run):
  the red line at the top says what is still needed, as `hxex check` would.
- **File** shows the project file itself. A change made in a text editor shows in the window
  at once.

- A list of tables (`[[input.runs]]`, the exclusion rules `[[input.exclude]]` and
  `[[review.exclude]]`, regions and sites) is a table with one row per entry and one column per
  setting, each column's description below it. Type in a cell and press Enter to write it;
  empty a cell to switch that setting off. **Add an entry** and **Remove the selected entry**
  add and remove rows; **Browse for the selected cell** fills a cell with a file's name. Each
  entry is changed in its own lines, so comments elsewhere in the file stay.
- A table of names you choose (sample names under `[analysis.controls]`, run names under
  `[analysis.runs]`, ...) is a table of names and values: type a name and its value and click
  **Add**, or change a value in the table. Under `[analysis.colours]`, each colour is shown
  beside its name.
- `[analysis] samples` sets which samples the analysis uses and their order in every figure
  and table: it is a list, with **↑** and **↓** to move the selected sample, **Remove**, and
  **Add** with the samples of the results that are open (or type one). The reference goes
  first wherever it is in the list.

## Tables

**Tables**, beside Settings, holds the tables your project file names: the peptide table (mzML
and Thermo files) or the table of files (the folder formats), the retention-time table, and
the user review table. Choose one on the left to edit it as a grid.

- Type in a cell to change it; **Add a row**, **Remove the selected rows**, **Add a column**,
  and **Remove the selected columns** do what they say (click a column's header to select the
  whole column; removing one asks first). Columns of your own are kept (*hxex* ignores the
  ones it does not read), and the columns keep their order.
- Changes are kept until you press **Save**; **Revert** goes back to the file. Leaving a table,
  the project, or the window with unsaved changes asks first.
- **Save** checks the table as *hxex* reads it (a sample of the peptide table must be one of
  your runs', a review status must be `accepted` or `rejected`, ...). When it finds a problem
  it says so and lets you save anyway, for a table you are still filling in.
- The table is saved with commas (the only separator *hxex* reads) as UTF-8, keeping the
  byte-order mark a spreadsheet's "CSV UTF-8" puts first. A table saved with semicolons opens
  too, and is saved with commas.
- A table open in a spreadsheet: Excel locks the file it has open (on Windows), so *hxex* says
  so and keeps your changes; close it there and save again. A table changed by another
  program since you opened it here is not written over without asking.
- **New table** makes a table with the columns *hxex* reads for your format, where you choose,
  and names it in the project file.

## Results

**Results** shows a run's results as a tree: each peptide of each sample, under it its charges
(each a series), under a charge its time points (the undeuterated controls first, the fully
deuterated last), and under a time point its replicates, which are the spectra. Each row
counts its spectra, how many have a quality flag, how many you rejected, and how many have
more than one population; a time point lists the K of its replicates, and a replicate its K,
review status, and flags. Click a row, or move through the tree with the arrow keys, to see:

- **a peptide or a charge**: the series' pages, as `hxex report` draws them: the spectra, the
  uptake plot, and the population map (when a spectrum has more than one population);
- **a time point**: its replicates side by side, on the same m/z range as the whole series;
  its replicates fitted at each K, a row per replicate with the K chosen for it framed; and
  the series' uptake plot;
- **a replicate**: that spectrum on its own, its fits at each K side by side (the one chosen
  framed), and the series' uptake plot.

A time point with a single replicate shows that spectrum's figures, as its replicate does.

The uptake page has a list of its own in its bar, for how it spreads the times: the project
file's `[figures]` setting (named, with its value), or for the window only a log time axis, a
linear one, or the time points evenly spaced. The session remembers the choice.

Results opened through the project file carry its `[figures]` settings and its review: change
either (in the Project part, or in a text editor) and the results are shown again with the
change, at the same place in the tree. Other settings change what the next **Run** does, not
results already made. The commands read the project file when they start.

The list above the tree shows only some series (flagged, more than one population, rejected,
or not reviewed yet), and **Find** narrows them to a peptide, residues, or sample. Opening the
results through the project file applies your review table and exclusion rules, as `hxex
report hxex.toml` does.

## Reviewing spectra

The **Review** panel under the tree records your decision on what is selected: one spectrum,
or every spectrum of a time point or a charge.

- **◀** and **▶** (top right) go to the previous and next spectrum, time point, or charge
  (the list beside them), through everything the tree shows (with the list above the tree,
  only flagged series, say).
- **Status**: not reviewed, accepted, or rejected. In the tree, the keys **A**, **R**, and **U**
  do the same, so you can review with the arrow keys and those three letters.
- **K**: the fit's choice, or the number of populations to use (as far as the run fitted).
  When no bootstrap was made at the K you choose, **Add the bootstrap (Resume)** runs `hxex
  resume`, which adds it.
- **Note**: anything you want to keep with the decision.

Each decision is written at once to your review table, the same file you could edit in a
spreadsheet ([Excluding spectra](excluding-spectra.md)): its `review_status`, `review_K`, and
`note` columns, in the spectrum's row; your other columns stay as they are. When the project
has no review table yet, the first decision makes `user_review_table.csv` beside the project
file (a copy of the results' spectra table) and names it under `[review]`. The tree, the counts,
and the figures show the decision at once, and `hxex report` and `hxex analyse` read it as they
read a table you typed.

Decisions need the project file: open the results through it (**File > Open project file**).
A table open in a spreadsheet cannot be written (Excel locks it on Windows): the panel says so
and keeps the decision; close the table there and press **Try again**.

## Analysis

**Analysis** shows what `hxex analyse hxex.toml` would write, without writing anything: the
same figures, made by the same program from the same settings (`[analysis]` in the project
file), so what you see here is what the command writes.

- The list on the left has every kind of figure `hxex analyse` can draw. A kind with figures
  shows how many; open it to see them (one per sample, or per time point, named as the command
  names their files). **off** means a setting leaves it out: select it to see which, and
  **Change…** opens that setting. *none for these results* means the settings allow it, but
  these results have nothing to draw (no peptide with two populations, say). [Figures](figures.md)
  lists every kind, with what each shows and needs; `hxex figures hxex.toml` prints the same
  list in a terminal.
- Select a figure to see it; **Save figure…** suggests the name `hxex analyse` gives it.
- The analysis is computed when you first open **Analysis** with results: a few seconds for a
  small dataset, about half a minute for 25,000 spectra. Then every figure shows at once.
- When the results change (a new run, a review decision) or the `[analysis]` settings do, the
  line at the top says the figures are out of date and why; they stay until you press
  **Recompute**. Opening other results starts afresh.
- When `[analysis] reference` names several samples, or `"all"`, a list at the top chooses
  which reference's analysis is shown. Each is computed the first time you choose it and then
  kept, so switching back is instant; `hxex analyse` writes a folder for each.
- Results opened alone (a bundle, not through the project file) use the default settings.
  A project that compares separate runs (`[analysis.runs]`) is not shown in the window yet:
  `hxex analyse` writes it.

## Running the commands

**Run** has a button for each command: **Check**, **Run**, **Resume**, **Report**, **Analyse**,
and **Info**. Each runs that command on the project file, in the project's folder, exactly as
`hxex run hxex.toml` would in a terminal: the line above the messages shows the command as you
would type it. One command runs at a time.

- The messages appear as the command writes them, and the bar shows how far it is, with the
  step it is on and the time left when *hxex* can tell.
- **Stop** stops the command after the step it is on, as Ctrl-C does in a terminal: what is
  finished is kept, and a bootstrap stopped part way is finished later with **Resume**. Press
  it again (**Abort now**) to end the command at once; one that still does not end is ended by
  the window a few seconds later.
- When a command finds results it would replace (a second **Run**, say), it changes nothing
  and the window asks: replace them, keep them under the next free name (`results.1`, ...),
  or cancel. These are the command's `--overwrite` and `--keep-previous`.
- The list below the buttons chooses where **Run** and **Resume** fit. Its first entry is the
  project file's setting, `[run.fit.backend] device`, with its value (`auto`, the default, uses
  an NVIDIA graphics card when *hxex* can see one, else the CPU); **Change…** opens that
  setting. **cpu** or **cuda** use that device for this command only (`--device`). The box
  beside it makes **Report** draw the figures as well (`--figures`).
- The console keeps the last 20,000 lines and wraps long lines to its width (untick **Wrap
  lines** to keep each line whole and scroll sideways); **Clear** empties it. Every command's whole output
  also goes to a log file in a folder `hxex_logs` beside the project file, one per command,
  named by the command and when it started (`run_2026-11-03_112059.log`), with the command, the
  folder, the *hxex* version, and how it ended. The folder is the window's: delete old logs
  whenever you like.
- After **Run**, **Resume**, or **Report**, the Results part shows the new results.
- Closing the window, or opening another project, while a command runs asks first, and stops
  the command if you agree.

## Figures

Each figure has buttons above it, and its title beside them (the peptide, sample, and charge
of a series), which stays readable however far you zoom:

- **Fit page** shows the whole figure in the space there is (the figure follows the window's
  size); **−**, **+**, and **100 %** zoom out, zoom in, and show the figure at its own size.
- The **mouse wheel** zooms in and out around the pointer; **dragging** with the left button
  moves a zoomed figure (the scroll bars work too).
- **Double-click** a panel of a page (one spectrum of the spectra page, say) to magnify it to
  fill the space; double-click again for the whole page. The panel is the same drawing made
  bigger, sharp at any size.
- **Save figure…** saves the figure as SVG or PDF (sharp at any size, for papers and slides
  you edit further) or PNG (a picture).

The figures are those `hxex report` draws, drawn by the same program (matplotlib), so the
same peptide gives the same figure in the window and in the figures folder. A line under each
figure says how to move about it. (**View > Draw the figures with** can switch to Qt's SVG
renderer, which zooms without redrawing but draws past the panels' frames anything that runs
beyond the axis limits, such as a raw profile wider than the m/z range shown.) Changing a figure in the window (axis ranges, labels,
colours) will come later.

## Sessions

A session file remembers what the window showed: the project file, the results, the peptide you
were looking at, which part of the window was open, and the window's size. **File > Save
session** writes it (we suggest the project's name with `.hxexgui`, beside the project file),
and **File > Open recent** lists the last ones you opened.

A session holds no settings and no review decisions: those stay in the project file and your
tables, where the commands find them. Opening a session whose project or results have since
been moved or deleted opens what is left and says what is missing in the Messages pane.
Closing the window, starting a new session, opening another project file or session, or
making a new project file, after changing what the window shows, asks whether to save the
session first. Opening a project file starts a new session (not yet saved); opening results
stays in the session you are in.
