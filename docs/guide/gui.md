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

In every table of the window (the data tabs, the HDX Summary Table, the Results list, the Tables
view, and the lists in the Project settings), hold Ctrl and turn the mouse wheel to make the
text larger or smaller, as in the Run tab's messages. The size applies to every table, and the
window remembers it; Ctrl+0 over a table goes back to the usual size.

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
- **Undo** and **Redo** (or Ctrl+Z and Ctrl+Shift+Z, while the Project part has the keyboard)
  take back or make again the changes made in the window, one at a time, as many as you like:
  the file is put back exactly as it was, comments included. The line beside them names the
  last change (`[run.controls] d2o_fraction: default → 0.85`), so a change made by accident is
  easy to spot. They cover this session's changes to this project file; when the file was
  changed by something else since (a text editor, or `hxex set`), Undo says so and leaves it.
- Turning the mouse wheel over a list does not change it (the form scrolls instead): choose by
  clicking.
- A setting that holds a file or a folder has **Browse…**; a file inside the project's folder
  is written relative to it, so the folder can be moved or copied.
- **Find a setting** searches every table by name and description.
- Under `[input]`, only the settings of the chosen format are shown (tick **Show the input
  settings of every format** to see them all). Changing the format offers to switch off the
  settings the new format does not use.
- A stage that its heading alone turns on (the bootstrap, the population test, the analysis,
  the review) has a **Use** box. Setting any of its settings also switches it on. The
  back-exchange correction (`[run.controls]`) is different: with its table off it runs by
  itself when the data hold undeuterated and fully deuterated controls, and its page says
  which state the file is in. **Always on** switches its table on; **Off, whatever the data
  hold** writes `controls = false` under `[run]`.
- A value that cannot be right (a word where a number belongs) is refused, and the Messages
  pane says why. A file you are still filling in can lack something (a peptide table, a run):
  the red line at the top says what is still needed, as `hxex check` would.
- **File** shows the project file itself. A change made in a text editor shows in the window
  at once.

- A list of tables (`[[input.runs]]`, the exclusion rules `[[input.exclude]]` and
  `[[review.exclude]]`, regions and sites) is a table with one row per entry and one column per
  setting, each column's description below it. Type in a cell and press Enter to write it;
  empty a cell to switch that setting off. **Add an entry** and **Remove the selected entry**
  add and remove rows, and **↑** and **↓** move the selected entry up or down: the outputs
  follow this order (the regions in `regions.csv` and the region figures, say). **Browse for
  the selected cell** fills a cell with a file's name. Each
  entry is changed in its own lines, so comments elsewhere in the file stay. Ctrl+C copies the
  selected cells (all of them, after Ctrl+A, with the column names first), and **Copy the
  table** (also on the right-click menu) copies every entry with the column names, ready to
  paste into a spreadsheet.
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
  and **Remove the selected columns** do what they say (select a cell in each column to remove;
  removing one asks first). Columns of your own are kept (*hxex* ignores the ones it does not
  read), and the columns keep their order.
- Select cells and press Ctrl+C, or right-click and choose **Copy**, to copy them as a block a
  spreadsheet takes (a cell not selected inside the block is left empty). With every cell
  selected (Ctrl+A), the column names come first.
- Click a column's header to sort the rows by it (numbers as numbers, empty cells last); click
  it again for the reverse order. Sorting changes only what you see: the file keeps its own
  order when saved, and each row keeps its number in the file (a spreadsheet numbers it one
  higher, as it counts the header).
- In the user review table, a row naming a spectrum your results do not have is shaded red,
  and a line above the grid gives its row number (the same rows `hxex check` and `hxex report`
  warn about: *hxex* skips them). **Show the next one** selects the next such row; hover over it
  to see why. Copy the right `spectrum_id` from `spectra.csv`, or remove the row. A row whose
  spectrum a later run summed from other scans (a moved retention window) is not shaded: its
  decision follows the spectrum, and hovering over the row names the new id.
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

**Sequence** shows the protein sequence of each sample as the analysis numbers it: from the
files `[analysis] fasta` names, or, for SpecExport data, the `SAMPLE.fasta` file beside each
sample's folder; a FASTA numbered by its header (`>NAME/START-END`) starts at START. The
residues run in blocks of ten, fifty to a line, with their numbers above; the list under the
view gives 30 to 100 residues a line instead (the window's session file keeps your choice).
Each distinct sequence
has a row, named by its samples (samples with the same sequence share one), the reference first;
a residue that differs from the reference's is highlighted, so a mutation stands out, and
prolines (which have no amide hydrogen to exchange) are in grey. The regions of
`[[analysis.regions]]` are coloured bars under their residues and the sites of
`[analysis.sites]` are underlined, with a key above. Click a residue to see its number, what it
is, how the other samples differ there, and its regions and site. When the project's results
are open, each sample's coverage runs under its sequence: a block under each residue, as high as
the number of peptides covering it, and a red dot where no peptide does. A peptide counts when
at least one of its spectra is not rejected, so the coverage follows your review. Clicking a
residue then also lists the peptides covering it. **Read again** reads the
FASTA files again after you change one. A run of SpecExport data keeps each sample's sequence in
its results, so the view still shows them when the FASTA files have moved (it says so), and it
says when a FASTA no longer matches the copy the run kept (the file is shown; run again to read
the peptides against it).

For pooled data, **Sequence** shows each record of `[input] pooled_fasta` instead, grouped by
class (`[input] pooled_classes`), with a line between the classes. A residue that differs from
the first record of its class is highlighted, and clicking a residue lists only the records of
its class that differ there. When the results are open, each record's coverage runs under it:
the peptides that record may hold, from every sample, in the record's own numbering (a record
whose numbering differs from the peptide's has its coverage moved to its own residues). A
residue covered by a peptide that only this record may hold is drawn in black, so you can see
where the data tell that record apart from the others.

## Results

**Results** shows a run's results as a tree: each peptide of each sample, under it its charges
(each a series), under a charge its time points (the undeuterated controls first, the fully
deuterated last), and under a time point its replicates, which are the spectra. Each row
counts its spectra, how many have a quality flag, how many are rejected, how many the analysis
leaves out, and how many have more than one population; a time point lists the K of its
replicates. A replicate's row says "yes" when it is flagged or rejected, why the analysis
leaves it out, its K, its review status, and its flags. Rest the pointer on a column's header
to see what it counts, and on a flag or a reason it is left out to see what it means (every one
is in [Flags and reasons](flags.md)). The same works in the List tab and in every table shown
in the window.

For results with fragment ions (read from mzML or Thermo files with fragment lists), a charge
row also names its fragment, its condition (the fragment list's label), and its MS level, for
example `z=1  c-1:3  ETD, MS2`; one fragment taken from two parent peptides adds the parent
(`from 2-12 z3`). The columns **ion**, **MS level**, **condition**, and **parent** show the
same, and two lists beside the filter show only one MS level or one ion type (**peptides**: the
peptides read at MS1). **Find** also finds a fragment (`c-1:3`), a condition, or a parent.

For pooled results with a pooled FASTA (`[input] pooled_fasta`), the **identity** column says
what each peptide may be: the one record whose sequence holds it, else the class all its
records fall in, else its classes (rest the pointer on it to see the level and every
candidate). Two lists beside the filter show only one identity level (**isoform**, **class**,
**ambiguous**, or **none**) or one class (`[input] pooled_classes`; an ambiguous peptide is in
each of its classes), and **Find** also finds a peptide by its identity or a candidate's name.
In **List**, the same columns come from `spectra.csv`, and **Find** finds them too.

- **rejected** counts the spectra the review rejected: your review table, your
  `[[review.exclude]]` rules, and the bundle's own review. Nothing else rejects a spectrum.
- **left out** gives the spectra `hxex analyse` leaves out with the project file's
  `[analysis]` settings (`[analysis.filters]`, `exclude_flagged`, and `exclude_times`), with
  the reason, as `left_out_spectra.csv` lists them: "off its m/z", "replicate outlier",
  "spectrum quality (...)", and so on. A quality flag on its own leaves nothing out; the ones
  that do are named here. "kept at K = 1" marks a spectrum the analysis uses only as one
  population; it is not counted. Accepting a spectrum in the review keeps it against
  `[analysis.filters]` (not against `exclude_flagged` or `exclude_times`). The column is
  worked out when the results open, and again only when you ask: after a review decision or a
  change to `[analysis]`, a line under the tree says it is from before them, with an
  **Update** button (a few seconds on a large run). Results opened as a bundle on their own
  use the default settings.

Click a row, or move through the tree with the arrow keys, to see:

- **a peptide or a charge**: the series' pages, as `hxex report` draws them: the spectra, the
  uptake plot, and the population map (when a spectrum has more than one population);
- **a time point**: its replicates side by side, on the same m/z range as the whole series;
  its replicates fitted at each K, a row per replicate with the K chosen for it framed; and
  the series' uptake plot;
- **a replicate**: that spectrum on its own, its fits at each K side by side (the one chosen
  framed), and the series' uptake plot.

A time point with a single replicate shows that spectrum's figures, as its replicate does.

A replicate read from mzML or Thermo raw files also has an **elution** tab: its elution
trace over retention time, to check that the scans summed sit on the peptide's own elution peak and not on a
neighbour's. *hxex* opens the spectrum's file again and draws, per scan, the intensity at the
ion's bins (the tallest point in the picking window of each bin, added up over the bins) over a
stretch of the run around the scans it summed (shaded). **Show elution traces for** chooses
what is drawn with it. Its choices are numbered as the **Save window** choices they go with
(below), so traces 2 show what you need to judge save choice 2, and so on. Hover over a choice
to see what it covers.

1. **this charge state only, in this file** (**this ion only** when you read fragments): this
   file's trace alone.
2. **every charge state of this peptide, in this file** (**this peptide, its charge states and
   fragment ions** when you read fragments): the peptide's other charge states and, when you
   read fragments, its fragment ions, one line per fragment channel (its ions added up), in
   colour, each scaled to the selected trace's height so that the peaks compare. They come
   from the same molecules and elute together; one whose peak sits apart from the others has
   interference at its own m/z and needs a window of its own (save choice 1).
3. **this charge state, in this file and its replicates** (the default): the same charge state
   in the sample's other files at the same time point (or the same control), in grey. A file
   that eluted earlier or later, or a window that caught another peak, stands out: that file
   needs a window of its own.
4. **this charge state, in every file of this sample**: the same, in every file of this sample
   that reads the peptide: its time points, replicates, and controls. Other samples' files are
   never drawn or saved to, since the peptide table goes by sample. With many files this is the
   slowest to draw.

For ion-mobility files the trace keeps to the spectrum's drift
range, and a second panel shows the drift profile of the scans summed, with that range shaded.
The selected run is drawn first, then the others (about half a second per file, longer the
first time a large file is opened). The files are read from the project's `data_root`, or,
for results opened on their own, from the folder that holds them; when a file is missing, the
tab says so.

To change a spectrum's window, switch on **Edit window** in the tab's bar. The window's start
and end appear as boxes (minutes), with the drift range (milliseconds) for ion-mobility files.
Type new times, or drag across the trace (or across the drift profile) to set them: while the
switch is on, dragging sets the window instead of moving the figure. The new window is hatched
on the trace, and under it the spectrum as the results hold it ("now") is drawn beside the
spectrum summed over the new window, each with its fit, its K, and its deuteration. The new
spectrum is made from the raw file (the results hold each spectrum only as summed over its own
window), so the file has to be where the project's `data_root` says; the first time a file is
opened this takes a few seconds, afterwards a fraction of one. The new spectrum shows as soon
as it is read; its fit follows within a second (every K is started both from the run's fit of
the spectrum and from the new spectrum itself, and keeps the better of the two, so it takes only
a few steps; a window moved away from another species therefore does not leave the fit on that
species' peaks). The line under the figure says which of the two is
under way. **Fits at each K** draws the new window's
spectrum at every K the run tried. This is a preview: it uses the run's settings and noise
curve, but not its bootstrap, population test, or controls, and nothing in the results
changes. Now and then the run's own fit, which tries more starting points, settles on another
K than the preview did. The line above the figure says it is a preview; hover over it to
see how this preview differs from a run (for example, K chosen without the population test).

While you edit with **Show elution traces for** on 3 or 4 (the replicates or every file of this
sample), the
other files are drawn on this file's clock instead of where they are: each is moved by the
same trace match a save at that level uses (below; within the drift range in the boxes) and
scaled to this trace's height, so the peaks line up under the new window when the match worked.
A file whose trace did not match is drawn dashed, where it is. **Each file's window** draws one
row per file instead, each on its own clock: the scans its spectrum sums now (shaded), the new
window shifted into it (hatched), and its shift; this file is the top row, and dragging across
that row sets the window. It shows, file by file, what saving at that level would write, and
whether a file's present window sits on the peak at all. The files are matched once for the
spectrum and the drift range (about half a second per file); moving the window does not match
them again, and a save at that level uses the same match. The traces drawn are always the ones
the run summed, each at its own drift range: a new drift range in the boxes is matched again
and may move the other files by their new shifts, but it does not change any trace, and the
files keep their places until the new match is ready.

**Save window** writes the window as a row of the project's elution-window table. The choice
beside it says what the row is for:

1. **this charge state only (z = 3), in this file** (**this ion only** when you read fragments):
   for interference at its own m/z, when this one needs a different window from the rest of its
   peptide (traces 2 show it). The row names the peptide as well, so it is for this peptide's
   ion only.
2. **every charge state of this peptide, in this file** (the default; **this peptide, its
   charge states and fragment ions** when you read fragments): every charge state of the
   peptide in this file takes the window and, when you read fragments, all its fragment ions
   (MS2, MS3). They are the same molecules and elute together, so this is usually right.
   If this charge state (or ion) had a window of its own, it now follows its peptide; the
   message names any other one of the peptide that keeps a window of its own.
3. **every charge state, in this file and its replicates (each shifted)**: as above, and in the
   sample's other files at the same time point (or the same control).
4. **every charge state, in every file of this sample (each shifted)**: as above, in every file
   of this sample that reads the peptide: its time points, replicates, and controls.

The two lists are independent: choosing how to save never changes the traces drawn, and
choosing the traces never changes how a window is saved. Before saving at a level, look at the
traces of the same number (for choice 1, at traces 2).

Retention moves a little between injections (a few seconds), so the last two choices do not
copy the minutes: *hxex* matches each file's trace of this charge state (within the drift
range in the boxes, on ion-mobility files, so that a brighter species at another drift time on
the same m/z does not decide it) against this file's trace, within 30 s either way, and shifts
the window by the difference. The message lists the files and their shifts (with more than four files,
their range; each file is in the log). A file whose match is unclear (the best match at the
edge of the 30 s, no signal, or traces that do not look alike) gets no window: the message
names it, so that you can set it by hand. If some of the files already have a window of this
peptide that you (or an earlier save) set, *hxex* asks first: **Replace them** with the
shifted window, **Keep them** as they are, or **Cancel**. Matching takes about half a second
per file (none when the tab has already matched them, above). Open any of the files' spectra afterwards to see the window it got.

When one file is read by several samples (controls shared by the samples of a state, say), the
window is saved for every one of them, since it is one injection;
the message after saving names the samples. On ion-mobility files a changed drift range is
always kept for this charge state (or ion) only, also when you save for the whole peptide
(drift times differ between charge states), and is not shifted in other files (drift times
hold between injections); the message says so.

When the project file names no table yet, the first save asks
where it goes, suggesting `rt_elution_windows.csv` beside the project file (with the project
file's name in front when it is not `hxex.toml`, such as `hxex_findpeak_rt_elution_windows.csv`,
so two project files in one folder keep their windows apart). If the file you choose exists
already, the window says so, and which other project file uses it, and adds to it only if you
say yes. The table is then named in the project file (`[input] rt_elution_windows`); saving
again for the same peptide or ion and file changes its row. Your peptide table is never changed. When you open a spectrum whose peptide
or ion already has a saved window in that file, the boxes show it and the choice is set to
match. **Undo** puts the table (and the project file, after the first save) back as it was
before the last save. Saving needs the results opened from their project file, and the table
opens in **Tables** like the others. The saved windows are used when you run the project again with overwrite enabled: **Run** on
the Run tab with `overwrite = true` in the project file's `[output]` part, or **Replace** when
it asks about the earlier results (on the command line, `hxex run --overwrite`). That run reads every file again but fits again only the series whose spectra
changed, so it takes a fraction of the first run's time, and your review decisions are kept; each spectrum stays in its series (see [A peptide's own window in one
file](inputs.md#a-peptides-own-window-in-one-file)). A spectrum's id names the scans it summed,
so a spectrum with a new window has a new id after the run; your review decisions for it follow
it to the new id (see [When a spectrum's id changes](project-file.md#the-user-review-table)).

Every level also has a **data** tab, named with its number of rows as in Analysis
(**data (12)**): the selection's rows of the tables `hxex report` writes.
A charge (series) shows its `uptake.csv` rows (each time, the centroid row and each K and
population, with errors and fD: the values of its uptake page), with its `series.csv` values in
a line above; the list beside **Save table…** switches to its `components.csv` rows (each
spectrum's populations: the values of its populations page) or its `spectra.csv` rows. A time
point shows its replicates' `spectra.csv` rows; a spectrum its own `spectra.csv` row, as names
and values. Sort, copy, and **Save table…** as in the Analysis data tab; double-click a row to
open its spectrum.

**List**, the tab beside **Tree**, shows every spectrum as one row with the columns of
`spectra.csv` (K and the populations at the K in use, the review, the flags, the screens'
scores, and why the analysis leaves it out). Click a column's header to sort by it, choose a
filter (flagged, K above 1, not reviewed, rejected, left out) or a sample, or type in **Find**.
Right-click a column's header to filter by that column: tick the values to keep when it has few
(a time by its label, such as `30 min`), give a range when it holds many numbers (a screen
score from `0.1`, say; a time with its units, `30 s` to `2 min`), or text the value contains.
Filters on several columns combine; a filtered column's header has a ▼ (its tooltip says how),
and **Clear column filters** shows every row again.
Choosing a row shows that spectrum, with the review panel, as the tree does, and **◀ ▶** step
through the list one row at a time in the order shown (the choice of spectrum, time point, or
charge beside them is greyed out on the List tab; Ctrl+C, or a right click, copies the row chosen or every
row shown, with the header): sort by a screen score, say, and work down from the
worst. The keys **A**, **R**, and **U** decide in the list too. A decision updates its row at
once; the row stays in view until you change the filter. The list is made in the background
when the results open (a few seconds for a large run).

The list at the left of its bar chooses which of `hxex report`'s tables the List shows:
`spectra.csv` (the spectra, as above), `series.csv`, `components.csv`, or `uptake.csv`, each
made the first time you choose it (`uptake.csv` takes a few seconds on a large run). Sorting,
the column filters, the sample, and **Find** work on each; the filter list is the spectra's
only. A row opens its spectrum, or its series for `series.csv` and `uptake.csv`. **Save
table…** writes the rows shown, in the order shown. After a review decision, `spectra.csv`
updates its rows at once; another table says it is from before your latest decisions until
you press **Update**.

Every level also has an **overview** tab, last, and it is what you see first when results
open: the run's summary pages, as `hxex report --figures` writes them to `overview.pdf`. A
list in its bar chooses the page: **K selection** (how many spectra chose each K, and the share
of each K at every time point; after review, how many use each), **controls** (the measured
back-exchange retention with its limits, the undeuterated controls, and each series' control
status), and **quality flags** (how many spectra carry each flag, where the flagged spectra
fall, and the share flagged per sample). The last two appear when the run has controls and
flags. The pages are drawn when the tab is in view and kept, so moving between peptides with
the overview open costs nothing; a review decision draws them again, since the K counts change.

A series whose spectra take several pages (`[figures] page_rows` and `page_columns`; a long
time course thinned to `max_points`, see [Outputs](outputs.md)) has a list in the spectra tab's
bar to choose the page; only the page shown is drawn. The populations tab goes on over the
same pages, with a list of its own that keeps in step: choosing a page in either turns both.
Stepping by series with **◀ ▶** goes through its pages before the next series (backwards, the
previous series' last page first), from the spectra or the populations tab.

Every such list in a figure's bar (the overview's pages, the spectra and populations pages, the
uptake page's time axis) has small **◀ ▶** buttons beside it: they step to the previous or next
entry, and go round from the last to the first.

The uptake page has a list of its own in its bar, for how it spreads the times: the project
file's `[figures]` setting (named, with its value), or for the window only a log time axis, a
linear one, or the time points evenly spaced. The session remembers the choice.

Results opened through the project file carry its `[figures]` settings and its review: change
either (in the Project part, or in a text editor) and the results are shown again with the
change, at the same place in the tree. Other settings change what the next **Run** does, not
results already made. The commands read the project file when they start.

The list above the tree shows only some series (flagged, more than one population, rejected,
left out, or not reviewed yet), and **Find** narrows them: every word must match, a number as a
residue the peptide covers (`20`), two numbers as its residues (`12-23`, however the range is
written), anything else as text in the sample, residues, sequence, or charge (`WT`, `SPFY`,
`z2`). So `WT 20` finds WT's peptides over residue 20. The List's **Find** works the same way. Opening the
results through the project file applies your review table and exclusion rules, as `hxex
report hxex.toml` does.

### Exporting a series

Select a peptide, a charge, a time point, or a replicate in the tree, then click **Export
data**, beside **Save figure…** above its figures: **Export for HX-Express** writes the series
(for a peptide, each of its charges) as an HX-Express spreadsheet, and **Export plot data**
writes the data behind its figures as CSV files (for a time point or a replicate, only that
part). A dialog asks for the folder, an
optional prefix and suffix for the names, and the options; it lists the names it will write
and asks before replacing anything. What each export holds is in
[Outputs](outputs.md#exporting-one-series).

## Reviewing spectra

The **Review** panel under the tree records your decision on what is selected: one spectrum,
or every spectrum of a time point, a charge, or a peptide (all the charges the tree shows under
it; its figures are those of its first charge).

- **◀** and **▶** (top right) go to the previous and next spectrum, time point, or charge
  (the list beside them), through everything the tree shows (with the list above the tree,
  only flagged series, say). On the List tab they go row by row.
- The line under the tree says how many series it shows of all the results' (under the list,
  how many spectra); the rows of choices above the tree and the list wrap onto more lines when
  you make that side of the window narrow.
- **Flags (N)**, at the end of the Status row: how many different quality flags the selected
  spectra carry (greyed when none). Rest the pointer on it (or click it) to see each flag with
  what it means (with several spectra, how many carry it). `hxex explain NAME` and
  [Flags and reasons](flags.md) say what *hxex* does about each and what to look at.
- When the results are open without their project file, the line under **Note** says so:
  decisions go in the project's review table, so open the project file to record them.
- **Status**: not reviewed, accepted, or rejected. In the tree, the keys **A**, **R**, and **U**
  do the same, so you can review with the arrow keys and those three letters.
- **K**: the fit's choice, or the number of populations to use (as far as the run fitted).
  When no bootstrap was made at the K you choose, **Add the bootstrap (Resume)** runs `hxex
  resume`, which adds it. When the run's quick screen (`[run.fit] screen`) skipped that K for
  the spectrum, the spectrum stays at the fit's K, and the panel says so, until **Resume** fits
  it (with its bootstrap). `hxex report` and `hxex analyse` wait for that too.
- **Note**: anything you want to keep with the spectrum. It is saved when you press Enter or
  leave the box. With a time point or a charge selected, the box shows the note its spectra
  share, and a note typed there replaces each spectrum's own. When their notes differ, the box
  is empty and says so; leaving it empty keeps each one's note. A note needs no decision: on a
  spectrum you leave not reviewed, it is kept and shown again, and it changes nothing in the
  fits or the analysis.

Each decision is written at once to your review table, the same file you could edit in a
spreadsheet ([Excluding spectra](excluding-spectra.md)): its `review_status`, `review_K`, and
`note` columns, in the spectrum's row; your other columns stay as they are. When the project
has no review table yet, the first decision asks where to put one, suggesting
`user_review_table.csv` beside the project file (with the project file's name in front when it
is not `hxex.toml`), makes it there (a copy of the results' spectra table) and names it under
`[review]`; a file that exists already is used as it is, after the window has said so and you
have said yes. The tree, the counts,
and the figures show the decision at once, and `hxex report` and `hxex analyse` read it as they
read a table you typed. When the review table is changed another way (saved in **Tables**, or
in a spreadsheet), the window reads the results again with it, at the same place in the tree.

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
- With two samples or more and no `[analysis] reference`, a red line above the list says
  *Reference needed for comparison plots*, with **choose a reference…** beside it: the sample
  chosen is written to the project file (**every sample in turn** writes `reference = "all"`).
- Select a figure to see it; **Save figure…** suggests the name `hxex analyse` gives it.
- The **data** tab beside the figure lists what the figure draws, one row per bar: on a coverage
  map each peptide's sample, time, residues, sequence, fD and its error, and how many spectra
  and charge states it comes from; on a Woods plot both samples' values, the difference and
  its error, and whether it is significant (with the reason when it was not tested). The
  residue figures list one row per residue: a profile or overlay each line's values and their
  errors, a choice figure the one- and two-population values with the choice and its reason,
  a significance map the residue comparison it colours, the tier map each residue's tier and
  test values, and redundancy and resolution each sample's coverage. This is the way to read
  their values, since a line names no peptide. The region bars and the region map list their
  regions. The columns are named as in the analysis tables (`species_uptake.csv`,
  `compare_peptides.csv`, `residues.csv`, `compare_residues.csv`, `regions.csv`); a row that a
  figure draws twice (the reference in every panel of an overlay) is listed once. The K=2 panel
  of a Woods plot, the butterfly plot, and the protection-factor figures list their peptides'
  rows of `compare_population_counts.csv`, `compare_peptides.csv`, and the protection tables.
  Click a column's header to sort (a crowded Woods plot is easier to read sorted by residue or
  by difference), press Ctrl+C to copy the rows you selected, and **Save table…** to write the
  rows, in the order shown, as a CSV file named after the figure. Choose a kind of figure
  rather than one figure (the coverage maps, say) and the data tab gathers the rows of all its
  figures, with a `figure` column naming each one's file. It does so when you open the data
  tab, since it draws every figure of the kind (minutes for a run with hundreds); choosing
  another figure meanwhile stops it.
- The **HDX Summary Table** tab beside **figure** and **data** shows the data summary of
  Masson et al. (2019) that `hxex analyse` writes as `summary.csv` and `summary.md`: one row per
  item and one column per sample, the reference first ([The HDX data summary for a
  paper](outputs.md#the-hdx-data-summary-for-a-paper)). A value that is the same for every
  sample is shown once, across their columns, and the notes are below the table (point at an
  item to see its note). It fills in when you press **Compute** and says when it is out of
  date. Your own items (the buffer, the quench, the
  protease) go in `[analysis.summary]` in the **Project** settings. Ctrl+C copies the rows
  you selected, and **Save table…** writes the summary as CSV or, with a name ending in
  `.md`, as the Markdown table.
- Right-click a peptide in a coverage map or a Woods plot for a menu of its spectra, and
  choose one to see them in **Results** (the peptide is selected in the tree, its charge
  states under it). Where bars overlap, the menu lists every peptide under the pointer, top
  to bottom; on a Woods plot each peptide is offered in both samples. A left click selects the
  peptide's row in the data tab. In the data tab, double-click a row, or right-click it, for
  the same.
- The analysis is computed when you press **Compute**, at the top beside the reference list:
  a few seconds for a small dataset, about half a minute for 25,000 spectra of peptides.
  With the ladder rate fits on (`[analysis.fragments] fit_rates = true`), every ladder step's
  decay is fitted first: about 2 minutes for 17,000 fragment spectra, and each ladder figure
  then takes a second or two to draw. For results with fragment ions, the **fit ladder rates**
  box beside **Compute** switches them on and off: it writes `fit_rates` to the project file,
  so `hxex analyse` does the same, and the figures shown are then out of date until you
  press **Recompute**. The line at the top says which to expect. Then every
  figure shows at once. Nothing is written to disk.
- When the results change (a new run, a review decision), the `[analysis]` settings change,
  or your review table changes (in **Tables**, or in a spreadsheet), the line at the top says
  the figures are out of date and why; they stay until you press **Recompute**, which draws the
  figure you are looking at again. When the review table changed, **Recompute** reads the
  results again with it first. Opening other results starts afresh.
- When `[analysis] reference` names several samples, or `"all"`, a list at the top chooses
  which reference's analysis is shown. Each is computed when you press **Compute** with it
  chosen, and then kept, so switching back is instant; `hxex analyse` writes a folder for
  each.
- The list has up to three parts. **Peptides (MS1): same for every reference** comes first:
  coverage, redundancy and resolution, the residue profiles and population choices, the
  population tiers, regions, and protection factors. These compare nothing, so they are drawn
  once, in the order and colours of `[analysis] samples`, from the latest analysis computed,
  and stay as they are when you choose another reference. A **Recompute** renews them too.
  **Peptides (MS1): compared with the reference** follows: the overlays, Woods and butterfly
  plots, the significance maps, and protection-factor differences, for the reference chosen.
  When the results hold fragment ions, **Fragment ions (MS2/MS3)** comes last: the ladders, the
  closure, and the residue rates, from the fragment scans (the closure is measured against the
  parent's MS1 spectrum where the run read it). Like the first part, they are drawn once
  whichever reference is chosen.
- Results opened alone (a bundle, not through the project file) use the default settings.
  A project that compares separate runs (`[analysis.runs]`) is not shown in the window yet:
  `hxex analyse` writes it.

## Running the commands

**Run** has a button for each command: **Check**, **Run**, **Resume**, **Report**, **Analyse**,
and **Info**. Each runs that command on the project file, in the project's folder, exactly as
`hxex run hxex.toml` would in a terminal: the line above the messages shows the command as you
would type it. One command runs at a time.

- The messages appear as the command writes them, and the bar shows how far it is, with the
  step it is on and the time left when *hxex* can tell. Beside the bar, **total** counts the
  command's run time, and keeps it when the command has ended.
- **Stop** stops the command after the step it is on, as Ctrl-C does in a terminal: what is
  finished is kept, and a bootstrap stopped part way is finished later with **Resume**. Press
  it again (**Abort now**) to end the command at once; one that still does not end is ended by
  the window a few seconds later.
- When a command finds results it would replace (a second **Run**, say), it changes nothing
  and the window asks: replace them, keep them under the next free name (`results.1`, ...),
  or cancel. These are the command's `--overwrite` and `--keep-previous`.
- Run the project again after changing what is read: the peptide table, the elution windows
  saved in the elution tab, or the picking or fit settings, with overwrite enabled:
  `overwrite = true` in the project file's `[output]` part (then **Run** replaces the earlier
  results without asking), or **Replace** when it asks. The run reads every file again but fits
  again only the series whose spectra changed, so it takes a fraction of the first run's time,
  and your review decisions are kept.
- The list below the buttons chooses where **Run** and **Resume** fit. Its first entry is the
  project file's setting, `[run.fit.backend] device`, with its value (`auto`, the default, uses
  an NVIDIA graphics card when *hxex* can see one, else the CPU); **Change…** opens that
  setting. **cpu** or **cuda** use that device for this command only (`--device`). The box
  beside it makes **Report** draw the figures as well (`--figures`).
- The console keeps the last 20,000 lines and wraps long lines to its width (untick **Wrap
  lines** to keep each line whole and scroll sideways); **Clear** empties it. Every command's whole output
  also goes to a log file in a folder `hxex_logs` beside the project file, one per command,
  named by the project file, the command, and when it started
  (`hxex_run_2026-11-03_112059.log`), with the command, the folder, the *hxex* version, and how
  it ended. The folder is the window's: delete old logs whenever you like.
- Opening a project shows the output of its last command, under a grey line saying which
  command it was, when it started, and how it ended (a long log shows its last 3,000 lines).
  **Earlier commands**, below the console, lists the project's other commands, newest first:
  choose one to see its output. Only the project file's own commands are listed, also when
  several project files share a folder. The next command you run clears the console.
- After **Run**, **Resume**, or **Report**, the Results part shows the new results.
- While a command runs, the other parts of the window show it too: the tab reads **Running**
  in the highlight colour, and the right end of the status bar, at the bottom of
  the window, shows the command, its step, and its run time (click it to open **Run**). When
  the command ends while you are elsewhere, the tab says how it ended (**Run (done)**,
  **(stopped)**, or **(failed)**) until you open it.
- Closing the window, or opening another project, while a command runs asks first, and stops
  the command if you agree.

## Figures

Each figure has buttons above it, and its title on a line of its own under them (the peptide,
sample, and charge of a series), which stays readable however far you zoom:

- **Fit page** shows the whole figure in the space there is (the figure follows the window's
  size); **−**, **+**, and **Actual size** zoom out, zoom in, and show the figure at its own size
  (100 %).
- The **mouse wheel** zooms in and out around the pointer; **dragging** with the left button
  moves a zoomed figure (the scroll bars work too).
- **Double-click** a panel of a page (one spectrum of the spectra page, say) to magnify it to
  fill the space; double-click again for the whole page. The panel is the same drawing made
  bigger, sharp at any size.
- **Mouse over** a peptide of a coverage, Woods, or protection-factor figure, on a bar
  of the regions figure, or on a peptide of the two-populations figure, and a note says what it
  is: the peptide's sequence and residues, its value with its standard error, and, on a
  comparison, both samples' values, the difference, and whether it is significant.
- **Save figure…** saves the figure as SVG or PDF (sharp at any size, for papers and slides
  you edit further) or PNG (a picture); the window's bottom bar says where it went.
- **Export data**, on the **Results** figures, writes the selection's data rather than the
  picture (see [Exporting a series](#exporting-a-series)).
- **?**, at the end of the buttons, lists these mouse actions: rest the pointer on it, or
  click it.

A line under each figure says what the panel is doing. While a figure is drawn, or its elution
trace read, a moving bar shows beside the words, and the figure shown until then is faded (it
is the one before). When there is no figure, the line says why ("No elution trace: …"); when a
figure could not be drawn, it says so in red (the messages pane says why). The line is empty
once the figure shows.

The figures are those `hxex report` draws, drawn by the same program (matplotlib), so the
same peptide gives the same figure in the window and in the figures folder. The window draws each figure as a picture at the size shown
(**matplotlib (raster, as PNG)** in the View menu); nothing is saved unless you press **Save
figure…**. (**View > Draw the figures with** can switch to Qt's SVG
renderer, which zooms without redrawing. Qt's renderer would draw past the panels' frames
anything that runs beyond the axis limits, such as a raw profile wider than the m/z range shown,
so the window cuts lines, bands, bars, and points at the frames before handing the figure to it;
images, colour maps, and text are not cut. A saved figure is not cut: matplotlib clips it.) Changing a figure in the window (axis ranges, labels,
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
session first. With a project file open in a session never saved, it asks even when nothing
changed, since **Open recent** lists sessions only: save one to find the project there next
time. Opening a project file starts a new session (not yet saved); opening results stays in
the session you are in.
