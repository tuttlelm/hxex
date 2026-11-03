# The *hxex* window

*hxex* also has a window of its own, for those who prefer pointing and clicking to typing
commands. It works on the same files as the commands: the project file, your tables, and the
results. Anything done in the window can also be done in a terminal, and the other way round, so
you can switch between them at any point.

The window is being built in steps. What it does today:

- **Project**: opens a project file (`hxex.toml`) and shows it. Changing settings in the window
  comes next; until then, edit the file in a text editor or with `hxex set` ([The project file
  and the commands](project-file.md)).
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

## Figures

Each figure has buttons above it:

- **Fit** shows the whole figure in the space there is (the figure follows the window's size);
  **−**, **+**, and **100 %** zoom out, zoom in, and show the figure at its own size. Holding
  Ctrl and turning the mouse wheel zooms too.
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
Closing the window after changing what it shows asks whether to save the session.
