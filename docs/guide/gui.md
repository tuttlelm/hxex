# The *hxex* window

*hxex* also has a window of its own, for those who prefer pointing and clicking to typing
commands. It works on the same files as the commands: the project file, your tables, and the
results. Anything done in the window can also be done in a terminal, and the other way round, so
you can switch between them at any point.

The window is being built in steps. What it does today:

- **Project**: makes a new project file or opens one, and changes its settings (below).
- **Results**: opens results as a tree of peptides (or, with fragment ions, of precursors),
  their series (one per charge state, or per fragment ion), time points, and spectra (the
  replicates), and shows the figures of what you select (below).
- **Run**: runs *hxex*'s commands on the project (check, run, report, analyse, info),
  with their messages, a progress bar, and a Stop button (below).
- **Analysis**: shows the figures `hxex analyse` would write with the project's settings, and
  why each figure left out is left out (below).

In every table, list, and text box of the window (the data tabs, the HDX Summary Table, the
Results tree and list, the Analysis list, the Tables view, the lists in the Project settings,
the File tab, the sequence view, the Run tab's messages, and the messages pane), hold Ctrl and
turn the mouse wheel to make the text larger or smaller. The size applies to all of them, and
the window remembers it; Ctrl+0 over one of them goes back to the usual size. Figures are
enlarged with the mouse wheel alone, and labels and buttons keep their size.

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

**File > Open** takes a project file (`.toml`), a session (`.hxexgui`, below), or a results
bundle saved as a `.zip` file. Opening a project file also opens its results when it has some,
with your review table and exclusion rules applied, as `hxex report hxex.toml` uses them.
**File > Open results**, and the **Open results…** button in the Results part, open a results
folder (a bundle) on its own, with the default analysis settings. **File > New session** starts
afresh, **File > Save session as…** writes the session under a new name (below), and
**File > Quit** closes the window, asking about an unsaved session first. **Help > User guide**
opens this guide, and **Help > About hxex** names the version.

**View > Appearance** chooses the look of the window's widgets. It lists the widget styles
your Qt offers: **System default** is the one Qt chose when the window opened (the system's
own look on Windows and macOS), and **Fusion**, Qt's own style, is there on every system; the
others differ by system. The choice is remembered and used the next time the window opens; a
style remembered on another system that this one lacks is replaced by the system default,
with a line in the messages.

The window is light or dark as your desktop is, where the desktop tells Qt so (Windows and
macOS do; on Linux it depends on the desktop). For the other one, start the window with
`hxex gui --dark` or `hxex gui --light` (`hxex-gui` takes the same): *hxex* then uses its own
colours, which the Fusion style draws in full and a system style only partly. The figures are
the same pages in either case. There is no switch in the menu: switching the colour scheme of
a running window through Qt proved unreliable (going back did not restore the light look).

## Settings

The **Project** part of the window shows the project file's settings, one table at a time (the
list on the left), each with its description, the text of the [settings reference](settings.md).
**File > New project file** (or the **New project file…** button at the top of the Project
part, beside **Open project file…**) asks for the input format and a name and writes a project
file with every setting at its default, as `hxex init` does.

**File > Save project file as template** (or **Save as template…** at the top of the Project
part) starts another project from the one open: the same data with other settings, say. It asks
for the new file's name and a folder for its results, then writes a copy and opens it:

- Every setting and comment is kept.
- Paths the project reads (the data folder, the peptide table, the FASTA files, the elution
  windows) are rewritten so the copy reads the same files from its own folder.
- The copy gets its own results folder, which must not exist yet, so neither project replaces
  the other's results. `keep_previous` goes back to false, and a tables folder
  or an `[analysis] out` folder named in the original is left out.
- The review is not copied: `[review] table` and the `[[review.exclude]]` rules are left out,
  so the copy starts unreviewed. To use the original's review decisions, choose its review
  table with **Browse…** under `[review]` (both projects then write to that one file), or copy
  the table first and choose the copy.
- The elution windows table (`[input] rt_elution_windows`) stays the original's file, so a
  window edited in either project changes it for both.

- A setting the file sets is in **bold**; an empty box shows the default in grey. Type a value
  and press Enter (or click elsewhere): it is written to the project file at once, as `hxex set`
  writes it, so the file keeps your comments and its layout. **Default** puts the setting back
  to its default (a `#` in front of its line); the button reads **Preset** when a preset
  named in the file sets the value, and puts it back to the preset's. Text needs no quotes.
- **Undo** and **Redo** (or Ctrl+Z and Ctrl+Shift+Z, while the Settings form has the
  keyboard; the Tables grid has no Undo)
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
  settings and tables the new format does not use (`[input.scans]`, `[input.target]`, and the
  `[[input.runs]]` of the scan formats, say). A value the file holds that is none of the
  choices (a misspelt format) is shown as such, beside the problem line.
- A stage that its heading alone turns on (the bootstrap, the population test, the analysis,
  the review) has a **Use** box. Switching it off puts `#~ ` in front of its heading, its
  settings, and the tables under it, and switching it on again brings exactly those lines
  back (a plain `# ` marks a setting at its default, as the template writes it). Setting any
  of its settings also switches it on. A true/false setting whose default is neither (decided
  from the data) shows a half-ticked box; a click makes it true, another false, a third
  unset again. Settings the file keeps for one sample or run (`[analysis.labelling.D109H]`)
  are listed on their table's page; a text editor changes them. The
  back-exchange correction (`[run.controls]`) is different: with its table off it runs by
  itself when the data hold undeuterated and fully deuterated controls, and its page says
  which state the file is in. **Always on** switches its table on; **Off, whatever the data
  hold** writes `controls = false` under `[run]`.
- A value that cannot be right (a word where a number belongs) is refused, and the Messages
  pane says why. A file you are still filling in can lack something (a peptide table, a run):
  the red line at the top says what is still needed, as `hxex check` would.
- **File** shows the project file itself, to search and to edit. A change made in a text editor
  shows in the window at once. The text is coloured: table headings in bold blue, setting names
  purple, text green, numbers and true/false orange, comments grey. A setting switched off
  (a `#` in front) is grey too, so the settings in use stand out.
  - **Find in the file** (or Ctrl+F) marks every match; Enter goes to the next one,
    Shift+Enter to the one before, and the count beside the box says which one you are on.
    **Match case** tells capitals apart.
  - Type in the text to edit it. Nothing is written until **Save**; the tab then reads
    **File (unsaved)**, and **Discard changes** puts the file's text back. A save is one step of
    **Undo** on the Settings form.
  - Text that is not valid TOML is not saved: the line at the top says why, the cursor goes to
    the line with the mistake, and your text is kept. Text that is valid TOML but has a problem
    with a setting (a misspelt format) is saved, and the red line says what is wrong, as on the
    Settings form.
  - If the file changes while you have unsaved edits (on the Settings form, in a text editor,
    or from the review panel), your text is kept and a line says so. **Load the file** drops your
    edits; **Save** asks before writing over the other change.
  - Opening another project, or closing the window, with unsaved text asks whether to save it.

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
  **Add**, change a value in the table, or select a row and click **Remove the selected
  name**.
- `[analysis.colours]` lists every sample (the same samples as the list below), in
  `[analysis] samples`' order, each with a list of colours. The list starts with **default**
  and the colour the figures give the sample when the file names none (the first sample
  black, the others in turn, skipping colours chosen for other samples), then hxex's other
  colours, **Other…** (a colour picker), and **Type a colour…** (any colour matplotlib knows,
  `tab:red` or `#1f77b4`). Each entry shows its colour, so the list shows the colour the figures
  will use. Choosing a colour writes it; choosing **default** removes the sample's line from the
  file. **Add** gives a row to another name (`run:sample`, a modified state), written once it
  has a colour.
- `[analysis] samples` sets which samples the analysis uses and their order in every figure
  and table: it is a list, with **↑** and **↓** to move the selected sample, **Remove**, and
  **Add** with a list of samples to choose from (or type one). The reference goes first
  wherever it is in the list. `[analysis] reference` has **Samples ▾** beside its box, with the
  same list. The list holds the samples of the results that are open; before there are
  results, those of `[input] samples` when the project file sets it, else the samples the data
  hold (the SpecExport folders with their FASTA files beside them, the samples of the
  `[[input.runs]]`, ...). So a new project's samples, their order, and its reference can be set
  before anything is run. A name not in the list (a group, `run:sample`) is typed.
- Groups of samples ([Groups of samples](experiments.md#groups-of-samples)) have a page of
  their own, `[analysis.groups]`, listing the groups with their reference, samples, and whether
  their regions and sites are their own or the shared ones. **Add a group…** asks for its name,
  samples, and reference (when `[analysis]` names a reference or samples, they start the new
  group and are taken out of `[analysis]`, where groups refuse them); **Rename…** and
  **Remove** change the selected group, its own regions and sites with it (its analysis folder
  is left as it is). Each group then has a page under `[analysis.groups]` in the list:
  its **reference**, its **samples** (added from the same list of samples as `[analysis]
  samples`) with a **vial label** beside each (left as the sample's
  name, the samples are written as a plain list; **↑** and **↓** set their order), and its
  **Own regions** and **Own sites**, edited as any list of tables. A group without regions
  of its own takes the shared `[[analysis.regions]]`, and its page says how many.

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
- **New table** makes a table with the columns *hxex* reads for your format, where you choose
  (the name suggested carries the project file's name unless it is `hxex.toml`, so two project
  files in one folder keep their tables apart), and names it in the project file; a file of
  that name already there is replaced only if you say so, and the project file's change can be
  undone like the others.

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
the number of peptides covering it, and a red dot where no peptide does. Only the MS1 peptides
count, not fragment ions, and the line under the view says so. A peptide counts when
at least one of its spectra is not rejected, so the coverage follows your review. Clicking a
residue then also lists the peptides covering it. **Read again** reads the
FASTA files again after you change one. A run of SpecExport data keeps each sample's sequence in
its results, so the view still shows them when the FASTA files have moved (it says so), and it
says when a FASTA no longer matches the copy the run kept (the file is shown; run again to read
the peptides against it). If one sample's FASTA file is missing (or does not read), the view
shows the sequence the run stored for that sample, with a note naming the file; this holds for
a `SAMPLE.fasta` beside the sample folders and for a file named in `[analysis] fasta`. A
`SAMPLE.fasta` with no sample folder of exactly its name beside it is not read by *hxex*, so
the view leaves it out and says so.

For pooled data, **Sequence** shows each record of `[input] pooled_fasta` instead, grouped by
class (`[input] pooled_classes`), with a line between the classes. A residue that differs from
the first record of its class is highlighted, and clicking a residue lists only the records of
its class that differ there. When the results are open, each record's coverage runs under it:
the peptides that record may hold, from every sample, in the record's own numbering (a record
whose numbering differs from the peptide's has its coverage moved to its own residues). A
residue covered by a peptide that only this record may hold is drawn in black, so you can see
where the data tell that record apart from the others.

With groups of samples (`[analysis.groups]`), **Sequence** shows the groups in the file's
order, each group's reference first and then its samples, with a gap between the groups; a
sample in two groups is shown in each. A residue is highlighted where it differs from the
group's reference only, since another group may be another protein. Each group's regions
(its own, or the shared ones it takes) run under its rows, named by the group, and its sites
are underlined in its rows only; the key lists each group's regions on a line of its own.
Samples in no group are not analysed, so they are left out, and the line under the view says
which.

## Results

**Results** shows a run's results as a tree: each peptide of each sample, under it its series
(one per charge state, `z=2`, or per fragment ion), under a series its time points (the
undeuterated controls first, the fully deuterated last), and under a time point its spectra,
the replicates (`rep 1`, `rep 2`). Each row
counts its spectra, how many have a quality flag, how many are rejected, how many the analysis
leaves out, and how many have more than one population; a time point lists the K of its
spectra. A spectrum's row says "yes" when it is flagged or rejected, why the analysis
leaves it out, its K, its review status, and its flags. With the
[population test](population-test.md), a K written `K = 1 (self), K = 2 (set)` differs from the
number of populations of its set, all the spectra of its peptide in its sample at that time
(the analysis uses the set's), and the time point's row then ends with `(set 2)`: the spectrum
keeps its own K (hover the K
for the set's chance; [details](population-test.md#when-a-spectrum-and-its-peptide-disagree)).
The sets' numbers are worked out with the "left out" column, and its **Update** button brings
both up to date after review decisions. Rest the pointer on a column's header
to see what it counts, and on a flag or a reason it is left out to see what it means (every one
is in [Flags and reasons](flags.md)). The same works in the List tab and in every table shown
in the window.

For results with fragment ions (read from mzML or Thermo files with fragment lists), the tree
is grouped by precursor: each top row is a parent peptide at one precursor charge, for example
`2-12 ALKRINKELSD 3+`. Under it come the parent's own series at that charge (`MS1`), its
precursor ions (`M  z=1  ETD, MS2`), and then its fragments, by ion series (the N-terminal
ones, such as c, first), length, charge, MS level, and condition, so that one fragment's
conditions are next to each other: `c-1:3  z=1  ETD, MS2`, then `c-1:3  z=1  ETD_pre, MS3`. A
fragment read from two precursors (the 2+ and the 3+, say) is under each of them, as they are
separate measurements. A peptide read at MS1 at a charge with no fragments has a row of its own
at that charge. The columns **ion**, **MS level**, and **condition** show the same as the
names, and two lists beside the filter show only one MS level or one ion type (**peptides**:
the peptides read at MS1). **Find** also finds a fragment (`c-1:3`), a condition, or a parent;
a residue number finds the fragments that cover that residue (`20`, or `2-6` for the fragments
of exactly residues 2 to 6), under every precursor.

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
  use the default settings, and the line under the tree says so.

Click a row, or move through the tree with the arrow keys, to see:

- **a peptide or a series**: the series' pages, as `hxex report` draws them: the spectra, the
  uptake plot, and the population map (when a spectrum has more than one population); a
  peptide shows its first series' figures;
- **a time point**: its spectra side by side, on the same m/z range as the whole series;
  each fitted at each K, a row per spectrum with the K chosen for it framed; and the
  series' uptake plot;
- **a spectrum**: that spectrum on its own, its fits at each K side by side (the one chosen
  framed), and the series' uptake plot.

A time point with a single spectrum shows that spectrum's figures, as the spectrum does.
The tab in view is drawn when you select a row; the other tabs of the selection are drawn
when you open them (a selection no longer draws every tab, so moving through a large run is
quicker). On a spectra page, and on the fits at each K, click a spectrum's panel to select
that spectrum in the tree (the **?** by the buttons says so).

Each panel of the fits at each K says above its fit why that K was or was not chosen:
`selected by the population test (self)` or `selected (lowest ΔBIC)` on the chosen one; a smallest
population below the fit's minimum (`[run.fit] min_population`) or the test's
(`[run.population_test] min_population`), such as `population 8 % < test minimum 10 %`; and,
with the [population test](population-test.md), its step from one K fewer, such as
`population test (self) p-value 1.5e-21 < 0.01 not selected`: a step passes when its p-value is
below `significance` and its smallest population passes the test's minimum (otherwise the line
above names the minimum it failed); the K the test chose says `selected`, a passing step below it
`passed`. The p-value is the chance that the spectrum would gain this
much fit from a population it does not have; values far below 0.01 only mean far beyond what
noise alone produced (see [How it decides](population-test.md#how-it-decides)). A K
the fit's minimum rules out keeps its ΔBIC, marked `(ruled out)`, so you can see what the
criterion alone would have chosen. Where the spectrum's set has another number of populations
than the spectrum, the panel at the set's number adds "population test (set)
p-value 6.9e-04 < 0.01: K = 1 (self), K = 2 (set)". "(self)" is the test of this spectrum
alone, "(set)" the test of its set, every replicate and charge state of the peptide in its sample at that time,
which the analysis uses: the spectrum keeps its own K
(see [When a spectrum and its peptide disagree](population-test.md#when-a-spectrum-and-its-peptide-disagree)).

Tick **residuals** in the bar of the fits at each K to put a small panel under each fit with
its residuals (picked minus fit). The residual panels of a spectrum share one scale, so a
misfit that one more population removes stands out. The choice is the window's (it changes
no setting) and is kept in a saved session.

Results picked with the profile fit (`[run.picker] pick = "profile_fit"`) also have a
**picking check** tab on a replicate: the raw profile, the peptide's fitted peaks (dark), the
peaks the fit gave to other species (orange, dashed), and the bin values (triangles), with the
spectrum's purity and stray share in the title ([details](picking-window.md#fitting-the-profile-instead-opt-in)).
It is kept apart from the spectrum's own figure so that the populations' fit stays readable.
A spectrum the profile fit did not read (a fragment, another ion type, its own envelope, or
centroided data, which keep the pick) has no picking check, and the tab says so.

Results with fragment ions (MS2 or MS3) have a **fragments** tab at every level: every
fragment of the selection's parent at its precursor charge, deuterium against time, a panel
per ion type and fragment charge (rows) and condition, gentle or harsh say (columns). Each
fragment is a line, light (the shortest) to dark (the longest), with a pale band of plus and
minus one standard error; the fragment you selected is drawn thicker and its name is bold. A
spectrum you rejected, or one the analysis excludes, stays on the page as a faded dashed line
without a band, so you can see what you took out. It is the first look at which fragments a
ladder should use: click a fragment's name in a legend to go to its series (to review its
spectra), and the page follows your decisions as you make them. A peptide read at MS1, or a
parent's precursor ion, shows its own fragments. Time is on a linear axis for a kinetics run
and on a log axis otherwise, as in the
[analysis figure](fragment-residues.md), unless you choose another time axis in the list in the
tab's bar (below).

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
switch is on, dragging sets the window instead of moving the figure. A window is a set of
scans, so the boxes hold the first and last scan's own times: a time you type or drag to goes
to the nearest scan, the arrows step scan by scan, and an unchanged window is previewed and
saved over exactly the scans the results hold. A drift range you type widens to the file's
drift bins (never inwards). The new window is hatched
on the trace, and under it the spectrum as the results hold it ("now") is drawn beside the
spectrum summed over the new window, each with its fit, its K, and its deuteration. The new
spectrum is made from the raw file (the results hold each spectrum only as summed over its own
window), so the file has to be where the project's `data_root` says; the first time a file is
opened this takes a few seconds, afterwards a fraction of one. If the file there is not the
one the run read (another conversion, say), the figure says so in red. The new spectrum shows
as soon
as it is read; its fit follows within a second (every K is started both from the run's fit of
the spectrum and from the new spectrum itself, and keeps the better of the two, so it takes only
a few steps; a window moved away from another species therefore does not leave the fit on that
species' peaks). The line under the figure says which of the two is
under way. Both spectra are drawn at the K in effect for the spectrum now (your review K if
you set one, else the run's, which a rerun keeps); when the fit's own criterion would choose
another K for the new window, its title says so. For ion-mobility files the drift panel
shows the drift profile of the scans the new window sums. **Fits at each K** draws the new
window's spectrum at every K it was fitted at (K = 2 is not fitted for a spectrum with at most
one deuteron, nor through the per-scan cut). A fragment ion of a file the run fitted through
the instrument's per-scan cut is previewed through the cut as well. This is a preview: it
uses the run's settings and noise
curve, but not its bootstrap, population test, or controls, and nothing in the results
changes. Now and then the run's own fit, which tries more starting points, settles on another
K than the preview did. The line above the figure says it is a preview; hover over it to
see how this preview differs from a run (for example, K chosen by the fit's criterion, not by
the population test (self)).
A review decision on the spectrum (a status, a K, a note) keeps the window you are editing
and its preview; selecting another spectrum drops them. A window far from the trace (typed,
or saved earlier) is pointed to at the edge of the trace instead of shrinking the trace to
show it.

While you edit with **Show elution traces for** on 3 or 4 (the replicates or every file of this
sample), the
other files are drawn on this file's clock instead of where they are: each is moved by the
same trace match a save at that level uses (below; within the drift range in the boxes) and
scaled to this trace's height, so the peaks line up under the new window when the match worked.
A file whose trace did not match is drawn dashed, where it is. **Each file's window** draws one
row per file instead, each on its own clock: the scans its spectrum sums now (shaded), the new
window shifted into it (hatched), and its shift; a file whose trace could not be read (its
file missing, say) gets a row saying why; this file is the top row, and dragging across
that row sets the window. It shows, file by file, what saving at that level would write, and
whether a file's present window sits on the peak at all. The files are matched once for the
spectrum and the drift range (about half a second per file); moving the window does not match
them again, and a save at that level uses the same match. When the match finds no other file
of the ion, the switch is greyed out. The traces drawn are always the ones
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
   If this charge state (or ion) had a window of its own, it now follows its peptide; when
   others of the peptide in this file have windows of their own, *hxex* asks first whether
   they follow too (**Replace them**) or keep theirs (**Keep them**), as it asks for the other
   files below, and the message names any that keeps a window of its own.
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
hold between injections); the message says so. A drift range you did not change is not
written at any level, so a save that only moved the retention window leaves the drift range
as the run (or the table) had it. A save at the last two choices matches the other files in
the background; if you select another spectrum before it is done, it is dropped and the
status bar says so.

When the project file names no table yet, the first save asks
where it goes, suggesting `rt_elution_windows.csv` beside the project file (with the project
file's name in front when it is not `hxex.toml`, such as `hxex_findpeak_rt_elution_windows.csv`,
so two project files in one folder keep their windows apart). If the file you choose exists
already, the window says so, and which other project file uses it, and adds to it only if you
say yes. The table is then named in the project file (`[input] rt_elution_windows`); saving
again for the same peptide or ion and file changes its row. Your peptide table is never changed. When you open a spectrum whose peptide
or ion already has a saved window in that file, the boxes show it and the choice is set to
match; when the table cannot be read (a row edited by hand that the project refuses), the
status bar says why and the boxes show the window the results hold. **Undo** puts the table (and the project file, after the first save) back as it was
before the last save; it covers the saves made while these results are shown, and opening
other results leaves the earlier saves as they are. Saving needs the results opened from
their project file, and the table opens in **Tables** like the others. The saved windows are used when you bring the results up to date: **Run (update)** on
the Run tab (on the command line, `hxex run hxex.toml`). That run reads every file again but fits again only the series whose spectra
changed, so it takes a fraction of the first run's time, and your review decisions are kept; each spectrum stays in its series (see [A peptide's own window in one
file](inputs.md#a-peptides-own-window-in-one-file)). A spectrum's id names the scans it summed,
so a spectrum with a new window has a new id after the run; your review decisions for it follow
it to the new id (see [When a spectrum's id changes](project-file.md#the-user-review-table)).

Until that run, the results do not have the windows you saved. So that you can see which
spectra a run would change, a spectrum whose window in the table differs from the one the run
read it with is drawn in red in the Results tree and the List, and so are its time point,
series, and peptide rows, so that you can find it with the tree folded. A line under the tree
counts them ("3 spectra have windows saved since the run (red): rerun to apply"), and hovering
over a red spectrum shows both windows. A window saved again over the same scans the run summed
is not red. The red follows the table: after a save or an **Undo**, after the table is changed
in **Tables** or in another program, and after the run, which reads the new windows and so
clears it.

**Back to the run's window** (in the tab's bar, with or without **Edit window**) writes the
table so that the selected red spectrum is read as the run read it: the row a save added is
removed, or the rows the run had are put back as they were, for every sample that reads the
file. When the saved window is the peptide's and other charge states (or ions) of it in this
file take it too, *hxex* asks whether to put the peptide's window back for all of them (**The
whole peptide**) or to give this one alone the run's window (**This one only**). It reads the
run's windows from the results, so it works whatever was saved since, and **Undo** takes it
back as it takes back a save.

Every level also has a **data** tab, named with its number of rows as in Analysis
(**data (12)**): the selection's rows of the tables `hxex report` writes, as it writes them
(every K and population, corrected and uncorrected; the `[figures]` K and correction the
figures follow do not apply to the tables).
A charge (series) shows its `uptake.csv` rows (each time, the centroid row and each K and
population, with errors and fD), with its `series.csv` values in a line above; the list beside
**Save table…** switches to its `components.csv` rows (each spectrum's populations) or its
`spectra.csv` rows. A peptide shows the same tables for every charge state under it, while its
figures are its first charge state's. A time point shows its replicates' `spectra.csv` rows; a
spectrum its own `spectra.csv` row, as names and values. Sort, copy, and **Save table…** as in
the Analysis data tab (a copy keeps the digits the CSV files keep; the cells shown round
more); double-click a row to open its spectrum.

**List**, the tab beside **Tree**, shows every spectrum as one row with the columns of
`spectra.csv` (K and the populations at the K in use, the review, the flags, the screens'
scores, and why the analysis leaves it out). Click a column's header to sort by it, choose a
filter (flagged, K above 1, not reviewed, rejected, left out) or a sample, or type in **Find**.
Right-click a column's header to filter by that column: tick the values to keep when it has few
(a time by its label, such as `30 min`), give a range when it holds many numbers (a screen
score from `0.1`, say; a time with its units, `30 s` to `2 min`), or text the value contains;
in that dialog **All** ticks every value, **None** unticks them, and **Clear** removes the
column's filter. Filters on several columns combine; a filtered column's header has a ▼ (its
tooltip says how), and **Clear column filters** shows every row again.
Choosing a row shows that spectrum, with the review panel, as the tree does, and **◀ ▶** step
through the list one row at a time in the order shown (the choice of spectrum, time point, or
charge beside them is greyed out on the List tab; Ctrl+C, or a right click, copies the row chosen or every
row shown, with the header): sort by a screen score, say, and work down from the
worst. Sorting, filtering, and **Find** keep the row chosen, so the row highlighted is always
the spectrum shown. The keys **A**, **R**, and **U** decide in the list too. A decision updates
its row at once; the row stays in view until you change the filter. The list is made in the
background when the results open (a few seconds for a large run), and when the results are
read again (after a decision, a changed setting, or a run) it comes back on the row you were
on.

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
uptake and fragments pages' time axis) has small **◀ ▶** buttons beside it: they step to the previous or next
entry, and go round from the last to the first.

The uptake page has a list of its own in its bar, for how it spreads the times: the project
file's `[figures]` setting (named, with its value), or for the window only a log time axis, a
linear one, or the time points evenly spaced. The fragments tab has the same list, and the two
always show the same choice: change one and both pages follow. Its first entry is the fragment
page's own default (linear for a kinetics run, log otherwise). The session remembers the
choice.

Results opened through the project file carry its `[figures]` settings and its review: change
either (in the Project part, or in a text editor) and the results are shown again with the
change, at the same place in the tree, on the same figure tab, with the tree's filters and
**Find** as they were. A change made while the results are being read is applied as soon as
they arrive. Other settings change what the next **Run** does, not results already made. The
commands read the project file when they start.

The list above the tree shows only some series (flagged, more than one population, rejected,
left out, or not reviewed yet), and **Find** narrows them: every word must match, a number as a
residue the peptide covers (`20`), two numbers as its residues (`12-23`, however the range is
written), anything else as text in the sample, residues, sequence, or charge (`WT`, `SPFY`,
`z2`). So `WT 20` finds WT's peptides over residue 20. The List's **Find** works the same way. Opening the
results through the project file applies your review table and exclusion rules, as `hxex
report hxex.toml` does.

### Exporting a series

Select a peptide, a series, a time point, or a spectrum in the tree, then click **Export
data**, beside **Save figure…** above its figures: **Export for HX-Express** writes the series
(for a peptide, each of its series) as an HX-Express spreadsheet, and **Export plot data**
writes the data behind its figures as CSV files (for a time point or a spectrum, only that
part). A dialog asks for the folder (**Browse…**), an optional prefix and suffix for the names,
and the options (the **Spectra**, **Controls**, and **Fits** lists: which spectra, which
controls, and the fits at which K go into the files); it lists the names it will write
and asks before replacing anything. What each export holds is in
[Outputs](outputs.md#exporting-one-series).

## Reviewing spectra

The **Review** panel under the tree records your decision on what is selected: one spectrum,
or every spectrum of a time point, a series, or a peptide (all the series the tree shows under
it; its figures are those of its first series).

- **◀** and **▶** (top right) go to the previous and next spectrum, time point, or series
  (the list beside them), through everything the tree shows (with the list above the tree,
  only flagged series, say). On the List tab they go row by row.
- The line under the tree says how many series it shows of all the results' (under the list,
  how many spectra); the rows of choices above the tree and the list wrap onto more lines when
  you make that side of the window narrow.
- **?**, at the right of the panel's title line: rest the pointer on it (or click it) to see
  which file is your review table; the whole table is on the **Project** page's **Tables**
  tab, where you can see and edit every row.
- **Flags (N)**, at the end of the Status row: how many different quality flags the selected
  spectra carry (greyed when none). Rest the pointer on it (or click it) to see each flag with
  what it means (with several spectra, how many carry it). `hxex explain NAME` and
  [Flags and reasons](flags.md) say what *hxex* does about each and what to look at.
- When the results are open without their project file, the line under **Note** says so:
  decisions go in the project's review table, so open the project file to record them.
- **Status**: not reviewed, accepted, or rejected. In the tree, the keys **A**, **R**, and **U**
  do the same, so you can review with the arrow keys and those three letters.
- **K**: the fit's choice, or the number of populations to use (as far as the run fitted);
  with several spectra selected whose K differ, the list reads "(differs between the
  spectra)" until you choose one for them all.
  When no bootstrap was made at the K you choose, **Add the bootstrap (Run)…** runs `hxex
  run`, which brings the results up to date and so adds it. When the run's quick screen
  (`[run.fit] screen`) skipped that K for the spectrum, the spectrum stays at the fit's K, and
  the panel says so, until **Run** fits it (with its bootstrap); the window keeps opening the project's results meanwhile, while
  `hxex report` and `hxex analyse` (and **Compute** on the Analysis page) refuse until then.
- **Not reviewed** takes a decision back: it is written as `unreviewed`, which also clears a
  status the run's own review layer holds (one set from Python, or carried over by a rerun).
- **Note**: anything you want to keep with the spectrum. It is saved when you press Enter or
  leave the box, and a note you are still typing goes with the status or K you choose next,
  or with the spectrum when you move on to another. A note the run's own review layer holds
  (one set from Python, or carried over by a rerun) is shown, and the box says so when you
  rest the pointer on it; emptying the box clears it. With a time point or a series selected,
  the box shows the note its spectra
  share, and a note typed there replaces each spectrum's own. When their notes differ, the box
  is empty and says so; leaving it empty keeps each one's note. A note needs no decision: on a
  spectrum you leave not reviewed, it is kept and shown again, and it changes nothing in the
  fits or the analysis.

Each decision is written at once to your review table, the same file you could edit in a
spreadsheet ([Excluding spectra](excluding-spectra.md)): its `review_status`, `review_K`, and
`note` columns, in the spectrum's row; your other columns stay as they are. A row that says
something (a status, `unreviewed` included, or a K) carries its `note` cell as the note, an
empty cell included, while a row that is entirely empty leaves the run's own note alone (so
the window writes the note shown into the row with your decision, and `unreviewed` beside a
note you empty on a spectrum with no other decision). When the project
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
and keeps the decision; close the table there and press **Try again**. A decision still
waiting when you open other results is dropped, and the Messages pane says so.

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
- With groups of samples (`[analysis.groups]`) the list above is the **group**: each group's
  analysis is its own, against its own reference, and nothing is shared between groups. When
  two groups or more have a region of one name, **(across the groups)** follows the groups: its
  **Compute** makes every group's analysis not yet made (or out of date), then the Regions
  across the groups, one figure per time, as `hxex analyse` writes them in
  `results_analysis_groups`; the groups' analyses made on the way are shown when you choose
  them.
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
  you selected, and **Save table…** writes the summary in the format you choose in the save
  dialog, CSV or the Markdown table (with the Markdown filter, a name that does not end in
  `.md` gets `.md` added).
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
  figure shows at once. Nothing is written to disk. **Stop**, beside **Compute** while it
  computes, ends the computation at its next step (between the tables, the figures, and
  each parent's ladder rate fits); the figures of the earlier analysis stay. When your review
  chose a K the run's quick screen skipped for a spectrum, **Compute** refuses as `hxex
  analyse` does, and **Fit the chosen K (Run)…** beside it runs `hxex run`, which brings the
  results up to date and so fits it.
- When the results change (a new run, a review decision), the `[analysis]` settings change,
  or your review table changes (in **Tables**, or in a spreadsheet), the line at the top says
  the figures are out of date and why; they stay until you press **Recompute**, which draws the
  figure you are looking at again. When the review table changed, **Recompute** reads the
  results again with it first. A change that only redraws the Results part (a `[figures]`
  setting) does not count. The kinds you opened in the list stay open through all this.
  Opening other results starts afresh.
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

**Run** has a button for each command: **Check**, **Report**, **Analyse**, and **Info**, with
**Start again…** apart at the right end of that row and **Run** on the row below them. Each runs that command on the project file,
in the project's folder, exactly as `hxex run hxex.toml` would in a terminal: the line above the
messages shows the command as you would type it. One command runs at a time.

Under the buttons, a row shows where the project's results stand, one mark per stage of a run
(read, pick, noise, fit, population test, correction and flags, bootstrap) and, after a line,
one for what is made from the results (the tables, the figures when there are some, and the
analysis folders):

- **✓** done: up to date with the project file;
- **!** (bold, coloured) out of date: a setting, a data file, a review decision, or *hxex*
  itself changed since, so the next **Run** does it again;
- **○** not done: no results yet, or a run stopped part way through it;
- **–** off: the project does not ask for it (no population test, say).

Hold the mouse over a mark to see why it is out of date ("[run.controls] fulldeut_at: 27 h,
was FD") and what **Run** will do for it. A stage after one that is out of date is out of date
too ("after the fit"). The line beside **Run**, under the marks, says what it will do, in the
same words as the end of `hxex check` ("Run: keeps fits, runs bootstrap"). The row is worked out again whenever the project file, the review
table, or a review decision changes, when a command ends, and when you open the Run tab (a data
file changed outside the window, say); on a large project that takes a few seconds, and the
marks are greyed meanwhile. While the results are out of date, the status bar at the bottom of
the window says so from every tab (**results out of date (3 stages): Run tab**; click it to
open the Run tab). When the project file does not load, the row says why instead.

The **Run** button follows the row:

- **Run** when there are no results yet: it does every stage.
- **Run (update)** when the results are out of date: it brings them up to date, doing again
  only what your changes touch (`hxex run`), without asking first, since the row shows what it
  will do. When the results are up to date, **Run** is greyed out.

**Start again…** reads, picks, and fits everything anew (`hxex run --fresh`). You rarely need
it: it takes as long as the first run and does not keep the fits that are still up to date, so
it warns you first, saying what **Run (update)** would do instead. The results are replaced (or
kept beside the new ones as `NAME.N` when `[output] keep_previous` is on); your review table and
the files you added are kept.

- The messages appear as the command writes them, and the bar shows how far it is, with the
  step it is on and the time left when *hxex* can tell. Beside the bar, **total** counts the
  command's run time, and keeps it when the command has ended.
- **Stop** stops the command after the step it is on, as Ctrl-C does in a terminal: what is
  finished is kept, and a bootstrap stopped part way is finished by the next **Run**. Press
  it again (**Abort now**) to end the command at once; one that still does not end is ended by
  the window a few seconds later.
- Run the project again after changing anything in it: the peptide table, the elution windows
  saved in the elution tab, the picking or fit settings, the preset, the bootstrap, the
  population test, or the controls. **Run (update)** reads your data files again only
  when something that decides the read changed (otherwise it takes the spectra stored in the
  results: after a change of preset or fit settings, say), and fits again only what the change
  touches, so it usually takes a fraction of the first run's time, and your review decisions
  are kept ([Running again](outputs.md#running-again)).
- **Report** and **Analyse** write their folders again each time. When **Run**, **Report**, or
  **Analyse** finds a file it wrote before that was changed since (opened and saved in a
  spreadsheet, say), it changes nothing, and the window names the file and asks whether to keep
  that folder under the next free name (`results_tables.1`, ...; for **Run**, the results with
  their folders) and run the command again (its `--keep-previous`), or cancel. The row's
  marks for the tables and analysis folders name such files beforehand too.
- The list below the buttons chooses where **Run** fits. Its first entry is the
  project file's setting, `[run.fit.backend] device`, with its value (`auto`, the default, uses
  an NVIDIA graphics card when *hxex* can see one, else the CPU); **Change…** opens that
  setting. **cpu** or **cuda** use that device for this command only (`--device`). The box
  beside it makes **Report** draw the figures as well (`--figures`).
- The console keeps the last 20,000 lines and wraps long lines to its width (untick **Wrap
  lines** to keep each line whole and scroll sideways); **Clear** empties it. Every command's whole output
  also goes to a log file in a folder `hxex_logs` beside the project file, one per command,
  named by the project file, the command, and when it started
  (`hxex_run_2026-11-03_112059.log`), with the command, the folder, the *hxex* version, and how
  it ended. The folder is the window's: delete old logs whenever you like (a log that
  cannot be written any more during a command, or whose folder was removed meanwhile, is said
  in the console, which keeps the output).
- Opening a project shows the output of its last command, under a grey line saying which
  command it was, when it started, and how it ended (a long log shows its last 3,000 lines).
  **Earlier commands**, below the console, lists the project's other commands, newest first:
  choose one to see its output. Only the project file's own commands are listed, also when
  several project files share a folder. The next command you run clears the console.
- After **Run** or **Report**, the Results part shows the new results. While a **Run** of the
  project whose results are shown is running, the Results part
  and the status bar say that the results shown are updated when the run finishes; you can keep
  looking at them meanwhile.
- While a command runs, the other parts of the window show it too: the tab reads **Running**
  in the highlight colour, and the right end of the status bar, at the bottom of
  the window, shows the command, its step, and its run time (click it to open **Run**). When
  the command ends while you are elsewhere, the tab says how it ended (**Run (done)**,
  **(stopped)**, or **(failed)**) until you open it.
- Closing the window, or opening another project, while a command runs asks first (after the
  session question), and stops the command if you agree: the window waits for it to end,
  saying so on the Run tab; a step not over after 30 s is aborted, and the command is killed
  10 s after that. Opening the project file already open changes nothing and asks nothing.

## Figures

Each figure has buttons above it, and its title on a line of its own under them (the peptide,
sample, and charge of a series), which stays readable however far you zoom:

- **Fit page** shows the whole figure in the space there is (the figure follows the window's
  size); **−**, **+**, and **Actual size** zoom out, zoom in, and show the figure at its own size
  (100 %), keeping the part of the figure at the centre of the view where it is.
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
