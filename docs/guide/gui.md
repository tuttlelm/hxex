# The *hxex* window

*hxex* also has a window of its own, for those who prefer pointing and clicking to typing
commands. It works on the same files as the commands: the project file, your tables, and the
results. Anything done in the window can also be done in a terminal, and the other way round, so
you can switch between them at any point.

The window is being built in steps. What it does today:

- **Project**: makes a new project file or opens one, and changes its settings (below).
- **Results**: opens results and lists every peptide of every sample at each charge, with its
  number of spectra, how many have a quality flag, how many you rejected, and how many have
  more than one population. Click a row to see its figures: the spectra, the uptake plot, and
  the population map (when a spectrum has more than one population).
- **Run** and **Analysis**: for now these show the commands to type in a terminal. Running
  stages from the window, reviewing spectra, and the analysis figures come later.

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
  **Add**, or change a value in the table.

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

The figures are those `hxex report` draws, so the same peptide gives the same figure in the
window and in the figures folder. Changing a figure in the window (axis ranges, labels,
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
