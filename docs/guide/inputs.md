# Input formats

*hxex* reads five kinds of input, chosen by `[input] format` in the project file:

| Format | Kind | What it reads | Where the peptides come from | Where the times and controls come from |
|---|---|---|---|---|
| `hxexpress` | folder | HX-Express spreadsheets (`.xlsx`, `.csv`) | the file names, or [a peptide table (CSV)](#listing-the-files-in-a-table) | the column labels of each sheet |
| `specexport` | folder | an HDExaminer SpecExport folder, one sub-folder per sample | the folder names, cut from each sample's FASTA, or [a peptide table (CSV)](#listing-the-files-in-a-table) | the spectrum file names |
| `pooled` | folder | a SpecExport folder holding the peptides of several proteins, without samples | the folder names (the sequence is written in them), or [a peptide table (CSV)](#listing-the-files-in-a-table) | the spectrum file names |
| `mzml` | scan | mzML files from the instrument | [a peptide table (CSV)](#the-peptide-table) | `[[input.runs]]` in the project file, or each scan's own time |
| `thermo` | scan | Thermo `.raw` files, read directly | [a peptide table (CSV)](#the-peptide-table) | as `mzml` |

The first three are **folder formats**: the spectra have already been extracted (by HX-Express
or HDExaminer), one spectrum per file or per column pair. The last two are **scan formats**:
*hxex* finds each peptide's scans in the raw data itself, by m/z, MS level, activation and
retention time, and sums them into spectra.

Every format ends in the same place: a set of raw spectra, each with a sample, a peptide (its
sequence, residue range and charge), a time or a control role, and a replicate number. The rest
of *hxex* (picking, fitting, reports, analysis) does not depend on the format.

`hxex init --format FORMAT -o hxex.toml` writes a project file with the `[input]` table of that
format and every other setting at its default. Edit the paths, then run `hxex check hxex.toml`:
it lists what would be read and stops at the problems it can see without fitting anything. The
full list of `[input]` keys is in the [settings reference](settings.md#input).

## Things every format shares

**Times** are stored in seconds. The way they are written depends on where they come from, and
the formats differ (see each format below). In the project file, a time can be a number of
seconds or text with a unit: `60`, `"60 s"`, `"5 min"`, `"20 h"` (`s`, `sec`, `m`, `min`, `h`,
`hr`, `ms` and their long forms). A negative time is refused.

**Controls** are identified by a *role*, not by a time: `undeut` (the undeuterated control) or
`fulldeut` (the fully deuterated control). The folder formats take the role from a label
(`Non` and `Full` in SpecExport, `undeut` and `TD` in HX-Express); the scan formats take it from
`role` in `[[input.runs]]`. A control has no time in the tables and figures. Replicate controls
are averaged. See [Back-exchange correction](back-exchange-correction.md) for what the
controls are used for, and [Controls and times](controls.md) for how to use a time point as a
control.

**Sequences** are written in modX: a modified residue is a lowercase prefix followed by its
residue letter (`acK`, `pS`, `pT`, `pY`), and a sequence is read residue by residue. Other
modifications are defined in `[[run.registry.residues]]` (a label, its base residue and the
composition it adds). `hxex check` reports any peptide it cannot read, with the reason.

**Filters.** The folder formats can read only some of the data:

- `samples = ["WT", "mutant"]`: only these samples.
- `peptide_ranges = ["1-10", "45-60"]`: only these peptides, by their residues. `"1-10"` and
  `"0001-0010"` mean the same peptide, from residue 1 to residue 10.
- `peptide_region = [1, 50]`: only the peptides that lie wholly within residues 1 to 50.

With both `peptide_ranges` and `peptide_region`, a peptide that matches either is read. Sample
names are case-sensitive. A sample in `samples` that is not in the data is named in a warning,
and when nothing is left to read (the folder holds no data of its format, or the filters match
none of what was found) the read stops with an error that says which, listing the samples
found or the residues the peptides found cover.

**Tables (CSV).** The peptide table and the retention-time table are CSV files with a header
row. The columns are separated by commas: a table saved with semicolons (Excel's setting in
some countries) or tabs is refused, saying so; save it again as CSV with commas. Column names
are matched whatever their case or surrounding spaces (`Peptide`, `charge `), so Excel's
capitals do no harm; a missing column is named together with the columns found. Spaces at the
ends of a cell are ignored, rows of empty cells are skipped, and a row with fewer or more cells
than the header has columns, an empty number, or a charge written `2+` or `2,0` is refused
naming the line and the column. The file must be UTF-8 (Excel's **CSV UTF-8**) or Excel's
**Unicode text** (UTF-16); any other encoding is refused with that advice.

**Retention times** (minutes) can be given for any format in a separate table,
`[input] retention_times = "rt.csv"`: a column `rt` and any of `sample`, `start_seq` and `end_seq`
(or `peptide_range`), `peptide`, `charge`. An empty cell matches anything, and each spectrum takes
the row that names the most columns. The scan formats also have their own sources (see below).

**Recorded paths.** `record_paths` says how the files read are recorded in the bundle:
`"path"` (as given, the default), `"name"` (file names only, for results to be shared), or, for
the scan formats, `"relative"` (relative to `data_root`). `"name"` is refused when two runs read
files of the same name from different folders: a later read could not tell them apart.

**Rerunning from stored spectra.** A bundle keeps the raw spectra it was made from (unless
`[output] keep_spectra = false`). `hxex run hxex.toml --spectra old_results`, or
`[input] spectra = "old_results"`, fits them again without reading the files: useful after
changing the picker or fit settings, or when the original files are not at hand. The read is
skipped; the format comes from the stored record, and the read settings in the project file stay
as the description of the data (*hxex* warns when they differ from those the spectra were read
with). The new bundle's `run_settings_used.toml` records the read settings of the stored spectra.
`retention_times` and `time_scale` still apply. The stored spectra keep the control roles the
files gave them; `undeut_at` and `fulldeut_at` in `[run.controls]` (see
[Controls](controls.md)) act on every run, so a rerun follows the project file as it is now.

When nothing the picking depends on has changed (the same spectra, after `[[input.exclude]]`,
with the same control roles, the same `[run.picker]` and `[run.envelope]` settings, and the same
version of *hxex*), the rerun takes the picked spectra stored in the bundle instead of picking
again, which gives the same result. Otherwise everything is picked again, and the log says
what differed.

**Times that only order the points.** If the times are labels rather than labelling times (1, 2,
3 for three conditions), set `[input] time_scale = "order"`. The uptake is then drawn by index,
and the kinetic and protection analyses refuse to run.

---

## HX-Express spreadsheets (`hxexpress`)

```toml
[input]
format = "hxexpress"
data_dir = "data"           # the folder of spreadsheets
```

### Files

One spreadsheet per sample, peptide and charge, all directly in `data_dir` (sub-folders are not
searched). Files ending in `.xlsx` or `.csv` are read; other files and Excel's `~$` lock files are
ignored.

The file name gives the sample, the peptide and the charge:

```
SAMPLE_START-END-PEPTIDE-zCHARGE-anything.xlsx
WT_0001-0011-LKDVEHFKVSS-z2-run1.xlsx
```

- `SAMPLE` is everything before the last `_` of the first field. It may contain underscores but
  not hyphens.
- `START` and `END` are the residue numbers (leading zeros are kept in the range,
  `0001-0011`).
- `PEPTIDE` is the sequence, in modX if it carries modified residues (`PEPacKTIDE`). Terminal
  groups (`Ac-`, `-NH2`) cannot be written here, because `-` separates the fields: for a
  peptide with a terminal group, [list the files in a table](#listing-the-files-in-a-table)
  instead, whose `peptide` column takes the full sequence, `Ac-PEPacKTIDE-NH2`.
- `zCHARGE` is the charge, with its `z`.
- The last field is free text but cannot contain a hyphen.

A file whose name does not fit this pattern is skipped with a warning naming it. To use other
names, [list the files in a table](#listing-the-files-in-a-table).

### Contents

Only the first worksheet is read, with formulas read as their saved values (a workbook whose
first sheet is empty while another holds the data is refused, saying so: move the data to the
first sheet). Row 1 holds a label above each spectrum. The spectrum is the label's column
(m/z) and the next column (intensity), from row 2 down. Empty cells are skipped, and a note
written below the data (text with no numbers under it) ends the spectrum; text between the
numbers is refused.

| Label | Meaning |
|---|---|
| `undeut`, `Undeut_1`, `undeuterated` | undeuterated control (any label starting with `undeut`) |
| `TD`, `td` | fully deuterated control (any label starting with `TD`) |
| `60 s`, `5 min`, `2 h` | a time point: a number, a space, a unit |
| `1 min`, `1.2 min` | two replicates of 1 min (see below) |

Units are `ms`, `s`, `sec`, `m`, `min`, `h`, `hr` and their long forms. The space is needed:
`60s` is refused. `FD` and `Full` are not control labels here; a label that is neither a time
nor a control word is refused with the control words named.

**Replicates.** Spectra at the same time are numbered as replicates in column order. As in
HX-Express, digits after the decimal point of a time of 1 or more mark a replicate, not a
fraction: `1.2 min` is a replicate at 1 min, and `10.3 s` is 10 s. The digit itself is not read:
the replicates are counted from left to right. Below 1 the fraction is kept: `0.5 min` is 30 s.
Labels that give the same time are replicates of one point even when written differently
(`60 s` and `1 min`).

Two cases get a warning, because the label may not mean what *hxex* reads:

- A marked label with no plain label at its time (`1.5 min` without a `1 min` column). If you
  meant a fraction, write the time in a smaller unit: `90 s`.
- One time written in two ways in one file (`60 s` and `1 min`).

---

## HDExaminer SpecExport (`specexport`)

```toml
[input]
format = "specexport"
data_dir = "SpecExport"
# samples = ["WT", "mutant"]  # only these
```

### Files

```
SpecExport/
  WT.fasta
  WT/
    0001-0011-LKDVEHFKVSS/
      Non-D-1-z2.csv
      Non-D-2-z2.csv
      30s-1-z2.csv
      5m-1-z2.csv
      Full-D-1-z2.csv
  mutant.fasta
  mutant/
    ...
```

- **A sample** is a folder with a FASTA file of the same name beside it (`WT/` and `WT.fasta`).
  A folder without its FASTA, or a FASTA without its folder, is not read, with a warning that
  names it. The names are case-sensitive (`wt.fasta` does not pair with `WT/`), the FASTA sits
  beside its folder, not inside it, and `data_dir` is the folder that holds the samples, not a
  sample folder itself. The suffix is `.fasta` in any case (`WT.FASTA` too); `.fa` is not
  looked for. If the FASTA has several records, the last is used (with a warning). A sequence
  written wholly in lower case is read in capitals. The run keeps each sample's sequence in its
  results (`hxex info` lists them). The analysis does not take its sequence from these files
  unless you name them in `[analysis] fasta` (without it, the sequence comes from the
  peptides); `hxex check` and `hxex analyse` say so in one line, naming the files.
- **A peptide** is a folder named `START-END-...`. Its sequence is cut from the sample's FASTA by
  the range, and the sequence in the folder name is checked against the FASTA (the residues
  only: modified residues and terminal groups are not compared). Other folders (for example an
  `analysis` folder) are ignored.
- **When the folders and the FASTA disagree.** HDExaminer names a mutant's folders from the
  reference sequence, so a mutant's folders and its FASTA differ at the mutated residues: when
  they differ at a few residues only (a third of those the folders cover at most), the FASTA's
  residues are used and a warning names each residue (`D109H.fasta differs from the names of 5
  of its peptide folders at residue 109 (D in the folder names, H in the FASTA) ...`): check
  that these are the sample's substitutions. A difference that every sample shares is not a
  mutation (a mutation is in its own sample only): the FASTA files and the sequence HDExaminer
  searched against disagree there, and the warning says so, naming the residues, so that you
  can check which is right. `hxex check` shows these warnings too. When they differ more, or a range lies outside the
  FASTA, nothing is read: the error lists the folders, with the folder's sequence and the
  FASTA's at those residues. The usual cause is a FASTA that holds more than the protein
  HDExaminer numbered, such as an expression tag or a signal peptide, so that its numbering
  differs; the error then gives the header line that fixes it (below). Otherwise the FASTA is
  that of another construct: give each sample the FASTA its data were searched against.
- **Numbering the FASTA.** The FASTA's first residue is residue 1, unless its header line gives
  the numbers of its first and last residues, Pfam-style, after a `/` at the end of the name:

  ```
  >WT/-2-175 the construct with its tag
  GSHMDIAIHHPWIRR...
  ```

  Here the tag `GSH` is residues -2 to 0, the protein's first residue `M` is residue 1, and the
  sequence's last residue is 175. The two numbers must span as many residues as the sequence
  has (here 178), or the FASTA is refused. The results keep the folders' residue numbers.
- **Modifications.** A sequence cut from the FASTA has no modified residues or terminal groups.
  For modified peptides, [list the peptide folders in a table](#listing-the-files-in-a-table):
  its `peptide` column takes the full sequence, for example `Ac-PEPacKTIDE-NH2`, in place of
  the FASTA's.
- **A spectrum** is a file named `TIME-REP-zCHARGE.csv`. Extra fields between the time and the
  replicate are allowed (`Non-D-1-z3.csv`). Other files are ignored. The name is case-sensitive:
  `.CSV` or `Z2` is not read.

### Contents

Two columns, m/z and intensity, without a header row. Further columns are ignored.

### Times and controls

| Token | Meaning |
|---|---|
| `Non` | undeuterated control (`Non-D-1-z2.csv`; what follows `Non` up to the replicate is ignored) |
| `Full` | fully deuterated control (`Full-D-1-z2.csv`) |
| `30s`, `0.1s`, `5m`, `1.5m`, `4h` | a time point: a number and one letter, `s`, `m` or `h` |

The control words are case-sensitive (`non` is refused). The unit is a single letter: `5min`
and `250ms` are refused here, although the project file accepts them. Fractions are kept
(`1.5m` is 90 s). The replicate is the `REP` field; two files for the same time, replicate and
charge are refused, naming both.

---

## Pooled SpecExport (`pooled`)

For a SpecExport folder that holds the peptides of several proteins (a digest of a mixture,
for example), without the sample level.

```toml
[input]
format = "pooled"
data_dir = "pooled"
sample = "pool"             # the sample name given to every peptide
```

```
pooled/
  0012-0020-ACacKDEFG/
    Non-D-1-z2.csv
    60s-1-z2.csv
    ...
```

- There is no sample folder and no FASTA. Every peptide gets the sample name `sample`.
- The sequence is the one in the folder name, `START-END-SEQUENCE`. It may carry modX labels
  (`acK`, or any label of `[[run.registry.residues]]`) and terminal groups (`Ac-...`,
  `...-NH2`). A folder whose sequence cannot be read is
  skipped with a warning; one whose residue count differs from its range is kept, with a warning.
- The spectrum files, the time tokens and the controls are as in SpecExport.

*hxex* does not work out which protein each pooled peptide comes from: the command line has no
such step (programmers can use `match_candidates`, [Using *hxex* from Python](python.md)).

---

## Listing the files in a table

For the folder formats (HX-Express, SpecExport, and pooled SpecExport), *hxex* normally finds
the files by their names. A table of the files can replace that search. Use one when:

- the file names do not follow the pattern *hxex* expects;
- a sequence needs something the names or the FASTA cannot hold: a terminal group (`Ac-`,
  `-NH2`) for HX-Express, any modification for SpecExport.

Name the table in the project file:

```toml
[input]
format = "hxexpress"            # or "specexport", "pooled"
data_dir = "data"
peptides = "files.csv"          # the table of the files
```

The table is a CSV file, with one row per peptide and charge. Excel can save one (**File > Save
As**, then **CSV UTF-8**); the columns are separated by commas, as in the example below (a
table saved with semicolons, as Excel does in some countries, is refused saying so; see
[Tables](#things-every-format-shares) above for the other rules).

```
file,sample,peptide,start_seq,end_seq,charge
my_peptide.xlsx,WT,Ac-PEPacKTIDE-NH2,1,8,2
```

| Column | Meaning |
|---|---|
| `file` | HX-Express: the spreadsheet's name, in `data_dir`. SpecExport: the peptide folder's name, inside the sample's folder (`0001-0008-PEPKTIDE`). Pooled: the peptide folder's name, in `data_dir` |
| `sample` | the sample (pooled: used instead of `[input] sample`) |
| `peptide` | the full sequence: modified residues in modX (`acK`, `pS`) and terminal groups (`Ac-`, `-NH2`) |
| `start_seq`, `end_seq` | the first and last residue numbers |
| `charge` | the charge state |

These six columns are required; a table without one is refused, naming it. Three more are
optional:

| Column | Meaning |
|---|---|
| `peptide_range` | the range's label in tables and figures; left out, it is made from `start_seq` and `end_seq` (`0001-0008`). A label that names other residues gets a warning: the residues are always `start_seq` to `end_seq` |
| `rt` | the retention time, minutes, kept with the spectra |
| `modification` | extra composition, in pyHXExpress's syntax. The same peptide folder or file can be listed twice with different modifications: each row is read as a series of its own, and the ids of their spectra end with `#series` and the series number (the ids of every other row stay as they are) |

Other columns are kept but not used.

The table replaces the search of the folder: only the files it lists are read, so list every
peptide, modified or not. `samples` and `peptide_ranges` still select from it. `hxex check`
reports any sequence it cannot read. A row listed twice (the same file, sample, peptide,
residues, charge, and modification) is refused, naming both lines.

For SpecExport and pooled folders, the table is read as it says: where its `peptide` or its
`start_seq` and `end_seq` differ from the folder's name (`START-END-SEQUENCE`), the table's
values are used, with a warning that names the folder (modified residues and terminal groups
are not compared, so `Ac-PEPacKTIDE` agrees with a folder named `...-PEPKTIDE`; a mutant's
sequence agrees when it is the one the sample's FASTA gives there). `hxex check` counts each
such row as a problem, since a wrong sequence or a shifted range in the table changes the fit
or the residue numbers without a word in the results.

---

## mzML and Thermo raw files (`mzml`, `thermo`)

```toml
[input]
format = "mzml"             # or "thermo"
peptides = "peptides.csv"   # the peptide table
data_root = "."             # run paths are taken from here

[input.target]              # how the peptides' scans are chosen
ms_level = 1

[[input.runs]]
path = "UN_01.mzML"
sample = "WT"
role = "undeut"

[[input.runs]]
path = "D_60s_01.mzML"
sample = "WT"
time = "60 s"
```

*hxex* predicts the m/z of every peptide's isotope peaks, from undeuterated to fully deuterated,
and reads only those stretches of the matching scans. It needs two things: a **peptide table**
that says what to look for, and one **`[[input.runs]]`** table per file that says what the file
is.

An indexed mzML file is read through its index, which is fast. A file changed after it was
converted (line endings converted by an editor or by git, a text-mode transfer, an edit by
hand) has an index that no longer matches its contents: *hxex* then reads through the whole file
instead, with a warning naming it; converting the file again makes it fast again. A scan whose
start time or MS level is empty is refused naming the scan: the file is damaged, so convert it
again.

### The peptide table

A CSV file (UTF-8; the rules under [Tables](#things-every-format-shares) apply). Required
columns:

| Column | Meaning |
|---|---|
| `peptide` | the sequence, in modX |
| `start_seq`, `end_seq` | the first and last residue numbers |
| `charge` | the charge state |

Optional columns:

| Column | Meaning |
|---|---|
| `sample` | the runs the peptide is read from: those of this sample; empty (or no column), every run |
| `file` | for your reference (the file the peptide was identified in); not used |
| `peptide_range` | the range's label in tables and figures; left out, it is made from `start_seq` and `end_seq` (`0012-0020`) |
| `rt` | the retention time, minutes (stored with the spectra) |
| `rt_start`, `rt_end` | the peptide's retention window, minutes (both or neither): only scans inside it are summed |
| `drift_start`, `drift_end` | for files with ion mobility: the peptide's drift-time range, milliseconds (both or neither), in place of the one *hxex* finds (see [Ion mobility: each peptide's drift time](#ion-mobility-each-peptides-drift-time)) |
| `modification` | extra composition in pyHXExpress's syntax (`H:1 Hex:1 ion_type:c undeut_mz:523.1`); not allowed in a project that reads fragment ions (see [Fragment ions](#fragment-ions)) |
| `run` | the file name of one run (`D_60s_01.raw`): the row is read from that run only. A peptide can have one row without `run` and rows for single runs; a run with its own row uses it instead of the general one (to give one run its own `rt_start`, `rt_end`, or `drift_start`, `drift_end`, say; for a window corrected in one file, the elution-window table is the place, see [A peptide's own window in one file](#a-peptides-own-window-in-one-file)). The rows are still one peptide: the runs stay replicates of one series, with the same controls, also when every run has a row of its own |

Other columns are kept with the entry but not used.

**A peptide is read from the runs of its sample**, so a mutant's peptides are not looked for
in the wild type's files. Leave `sample` empty for a peptide every run has; its spectra then
take each run's sample. A sample that no `[[input.runs]]` has is refused (a misspelt name would
otherwise read nothing), as is a peptide listed twice for the same runs (twice for one sample,
or once for a sample and once for every run); the message names the line. `hxex check` says
when a run has no peptide of its sample at all.

### `[[input.runs]]`: one table per run

A *run* is one acquisition on the instrument, saved as one mzML or raw file. Each run gets its
own `[[input.runs]]` table, which names the file and says what was measured in it.

| Key | Meaning |
|---|---|
| `path` | the run's file (relative paths start at `data_root`) |
| `sample` | its sample |
| `rep` | replicate number. Left out, the runs of one sample at one time (or one control role, or the kinetics runs of a sample) are numbered 1, 2, ... in the order they are listed. The same `rep` given to two runs of one sample at one time or control role is refused; kinetics runs may share one (a time course split over files) |
| `time` | the labelling time of the whole file (`"60 s"`, `"5 min"`) |
| `role` | `"undeut"` or `"fulldeut"`: a control run |
| `time_offset` | for kinetics runs (below); refused on a run with a `time` or a `role` (it would be ignored there) |
| `usable_from` | leaves out the file's scans before this scan time, on any run (kinetics runs: below). On a run with a `time` or a `role` it also cuts every retention window that starts earlier, and a warning names those peptides |
| `activation`, `precursor_mz`, `precursor_charge`, `filter_contains`, `filter_excludes` | replace one scan condition of every target for this run only (`false` removes it) |

What a run holds depends on `time` and `role`:

- **A time point** (`time` given): an LC-MS run of one labelled sample. Each peptide's scans
  inside its retention window are summed into one spectrum at that time.
- **A control** (`role` given): the same, but the spectrum is a control, without a time. With
  `time` as well, the run is a time point that also serves as the control (a long time point
  that has reached full deuteration, for example).
- **A kinetics run** (neither given): every scan has its own time, the scan time. Used for
  continuous infusion, where exchange runs while the instrument acquires. See
  [Experiment types](experiments.md#kinetics).

Every scan must record its time (its scan start time) in the file, whichever kind of run it is:
a file with a scan without one is refused, naming the file. A kinetics run takes the scan times
as its exchange times, and retention windows choose scans by them, so such a scan would
otherwise be dropped or kept without notice. Converters such as ThermoRawFileParser and
msConvert always write the time, so this points to a damaged or unusual file.

The scan-condition keys are for runs acquired differently from the rest. For example, a control
run acquired with in-source activation throughout takes `filter_excludes = false` while the
other runs exclude `"sid"`, and a run whose scans record a wrong precursor charge takes
`precursor_charge = false`.

### `[input.target]`: which scans hold the peptides

| Key | Meaning |
|---|---|
| `ms_level` | MS level of the peptides' scans (default 1) |
| `rt_window` | `[start, end]` in seconds: the scans summed in time-point and control files (a peptide's `rt_start`, `rt_end` take its place) |
| `activation`, `preactivation` | the activation of the scans (MS2 and MS3 targets) |
| `precursor_mz`, `precursor_charge`, `parent_mz` | conditions on the scan's precursor |
| `filter_contains`, `filter_excludes` | text the scan's filter string must, or must not, contain (Thermo filters mark in-source activation with `sid=`) |

Activation names `ETD`, `ECD`, `CID`, `HCD`, `EThcD`, `ETciD` and `UVPD` match exactly (case is
ignored; `ETD` does not match EThcD); any other text matches part of the activation name.

Without a retention window, every matching scan in the file is summed. That suits direct
infusion, but in LC-MS runs it sums every co-eluting species with the same m/z, so give each
peptide its window (`rt_start`, `rt_end` in the table) or a common `rt_window`.

### `[input.scans]`: how scans become spectra

| Key | Default | Meaning |
|---|---|---|
| `group` | `"each"` | kinetics runs: how scans are grouped into spectra: `"each"` (every scan is a spectrum), `"every"` (see `every`), or `"windows"` (see `windows`) |
| `every` | 1 | with `group = "every"`: this many consecutive scans are summed into one spectrum (`every = 5`: scans 1 to 5, 6 to 10, ...). Summing raises the signal, at the cost of fewer time points |
| `windows` | | with `group = "windows"`: the stretches of scan time (s) whose scans are summed, one spectrum per stretch: `windows = [[0, 30], [30, 90], [90, 300]]` |
| `min_time` | | kinetics runs: leave out scans before this scan time (s) |
| `max_spectra` | | kinetics runs: at most this many spectra per peptide, spread over log time |
| `mz_margin` | 2.0 | m/z read on either side of a peptide's predicted peaks |
| `sum_ppm` | 10.0 | centroided peaks closer than this are merged when scans are summed |
| `gap_factor` | 3.0 | profile scans: gaps the instrument left out are not bridged when summing |
| `precursor_tol` | 1.0 | m/z tolerance when matching a scan's precursor |
| `find_peak` | false | time-point and control runs: find each peptide's elution peak near its `rt` and sum only that peak's scans (see [Finding each peptide's peak](#finding-each-peptides-peak)) |
| `find_fragment_peak` | false | time-point and control runs: find each fragment channel's elution peak in every run, from all its fragment ions together, and sum only that peak's scans (see [Finding each fragment channel's peak](#finding-each-fragment-channels-peak)) |
| `fragment_peak_width` | 1.0 | with `find_fragment_peak`: the window is the peak's apex plus and minus this many peak widths at half height |
| `peak_search` | 30 s | how far either side of `rt` the peak is looked for |
| `peak_smoothing` | (measured) | the chromatogram's smoothing, in seconds; left out, half the run's typical peak width |
| `peak_width` | 0.5 | the window: the peak's apex plus and minus this many peak widths at half height |
| `find_retention_time` | `"suggest"` | with `find_peak`: look for where a peptide's exact m/z elutes when its peptide-table time looks wrong (`"suggest"`, `"automatic"`, or `"off"`; see [When the peptide table's time is wrong](#when-the-peptide-tables-time-is-wrong)) |
| `rt_search_ppm` | (measured) | how far a peak top may sit from the peptide's m/z in that search |
| `rt_search_tolerance` | (measured) | how far apart, in seconds, the undeuterated injections may put the peptide's peak |
| `drift_gate` | `"auto"` | files with ion mobility: sum each peptide over its own drift-time range, found from its undeuterated controls (`"auto"`), or over every drift time (`false`); see [Ion mobility: each peptide's drift time](#ion-mobility-each-peptides-drift-time) |
| `mz_calibration` | `"auto"` | measure each file's m/z offset from its own peptide spectra and take it out before picking (`"auto"`), or use the m/z as the file gives it (`false`); see [Each file's m/z offset](#each-files-mz-offset) |

Profile and centroided scans are both read. Summed profile scans are put on a common m/z grid;
centroided scans are merged peak by peak. Profile data is recommended: when converting Thermo
files to mzML with ThermoRawFileParser, keep profile data (`-p`) and use zlib rather than
numpress compression (numpress is refused).

Waters `.raw` folders are converted with ProteoWizard's msConvert, for example:

```
msconvert RUN.raw --mzML --64 --zlib --ignoreCalibrationScans --combineIonMobilitySpectra --filter "msLevel 1"
```

`--ignoreCalibrationScans` leaves out the lockspray scans, and `msLevel 1` the high-energy scans
of MSe. Data acquired with ion mobility (HDMS, HDMSe) needs `--combineIonMobilitySpectra`, which
writes each scan with its drift-time bins together; *hxex* then keeps each peptide's own
drift-time range (see [Ion mobility: each peptide's drift time](#ion-mobility-each-peptides-drift-time)).
Files whose m/z the acquisition did not correct with the lockspray need no correction at
conversion: *hxex* measures each file's offset itself (see [Each file's m/z offset](#each-files-mz-offset)).
On a Q-TOF dataset, msConvert's `lockmassRefiner` filter left the files' offsets scattered
over about 18 ppm, so it is better left out.

### Fragment ions

Each `[[input.fragments]]` table adds fragment-ion targets made from every peptide of the table:
for example c and z ions from ETD scans of each precursor.

```toml
[[input.fragments]]
ions = "c-1:ALL z-dot:ALL"   # TYPE:LENGTHS:CHARGES, space-separated
label = "gentle"             # names the condition
ms_level = 2
activation = "ETD"
filter_excludes = "sid"
```

- `ions` lists the ion types with their lengths and charges, each `ALL` (the default) or numbers
  and ranges: `"c:2-6,9 z-dot:3-"`. The types are those of pyteomics: `a`, `b`, `c`, `x`, `y`,
  `z` with their variants (`c-1`, `c+1`, `z-dot`, `z+1`, `y-NH3`, `b-H2O`, ...), and `M`, the
  peptide itself as seen in the fragment scans.
- Each fragment is tied to its parent: by default only scans whose precursor isolation overlaps
  the parent's m/z range (`parent_mz = "parent"`) and that record the parent's charge
  (`precursor_charge = "parent"`) are used.
- With `ALL`, fragment charges run from 1 to the parent's charge minus 1 (`limit_charge = false`
  allows the parent's charge too); charges written out are used as given. Fragments as long as
  the parent are skipped, as are c ions before a proline and z ions that start with one.
- MS3: `ms_level = 3` with `preactivation`, the activation of the stage before the last
  (`"none"` for none). The earlier stages are read from the scan's precursor list, or from the
  Thermo filter string when the mzML gives only the last stage.
- Repeat the table for other conditions of the same ions (with and without in-source
  activation, say); `label` keeps them apart.
- `[input] parents = false` reads only the fragments.

**With fragment ions, the peptide table has no `modification` entries.** In a project with
`[[input.fragments]]`, every peptide of the table is fragmented, so this covers the whole table,
the peptides' own MS1 spectra included: a project reads a peptide and its fragments from the
same row. The same entries are fine in a project without fragments.

The reason is that *hxex* has to know which fragments carry each modification, and a
`modification` entry describes the peptide as a whole:

| In `modification` | Example | Why the fragments cannot have it |
|---|---|---|
| a composition change | `H:1`, `O:-1` | which residue carries it is not known, so neither is which fragments have it |
| extra exchangeable sites | `Hex:1` | the same |
| an explicit m/z | `undeut_mz:523.1` | it is the m/z of the peptide's own ion; no fragment's m/z follows from it |
| an ion type other than `M` | `ion_type:c` | a fragment is not fragmented again |

A table with any of these is refused and nothing is read: `hxex check` and `hxex run` name each
such peptide with its line in the table. An empty `modification`, or `ion_type:M`, is fine.
Write modified residues into the `peptide` sequence as modX labels instead (`PEPacKTIDE`,
`Ac-PEPpSIDE-NH2`; new ones in `[[run.registry.residues]]`): the sequence is read residue by
residue, so each fragment carries exactly the modifications of its own residues.

When a peptide's MS1 spectra need a `modification` (an `undeut_mz`, say), use two projects on
the same runs: one without `[[input.fragments]]`, whose table keeps the `modification`, for the
MS1 spectra; and one with `[[input.fragments]]` and `[input] parents = false`, whose table lists
the peptide without it, for the fragments.

**A fragment table made for pyHXExpress** (one row per fragment, with the ion type in
`modification`) is not read this way. List each parent peptide once, with its charge, and name
the ion types in `ions`: *hxex* makes the fragments itself, ties each one to its parent's scans,
and counts its amides by its position in the parent.

**Which ion type to name: the envelope.** In ETD and ECD spectra, one fragment often appears as
several ion types a hydrogen apart (c-1 and c, or z-dot and z+2), whose isotope peaks overlap.
By default (`[run.envelope] mode = "theory"`) *hxex* expects only the ion type you name. With
`mode = "mixed"`, it fits a mixture of that ion type and the next one a hydrogen heavier (the
number is `components`, default 2), learning their proportions from the undeuterated spectra.
The ion type you name is the *lightest* of the mixture, so name the lighter one: `c-1` with
`mode = "mixed"` covers c-1 and c, while `c` would cover c and c+1 and miss real peaks at c-1.
The same holds for `z-dot`, which covers z-dot and z+2 (in the names *hxex* takes from
pyteomics, `z+1` is the same ion as `z-dot`). See
[Experiment types](experiments.md#fragments) and the `[run.envelope]` settings in the
[settings reference](settings.md#runenvelope).

The fragments' own analysis is described in [Experiment types](experiments.md#fragments).

### Retention times

A spectrum's retention time (minutes) comes from, in order: the `[input] retention_times` table;
the peptide table's `rt` column; for a fragment, its parent's; in a time-point or control file,
the signal-weighted time of the summed scans. Kinetics spectra have none. With `find_peak`, a
found peak's own time takes the place of the `rt` column.

`spectra.csv` also says which scans each spectrum summed: `rt_start` and `rt_end` are the times,
in minutes, of its first and last scan (time-point and control files only).

### Finding each peptide's peak

In an LC-MS run each peptide elutes as a peak a few seconds wide, and other peptides elute
around it. Summing a fixed window takes in whatever else falls on the peptide's m/z values in
that time; in a crowded digest that is often another species. If your peptide table gives only
a retention time (`rt`) for each peptide, *hxex* can find the peptide's peak in every run
itself, as HDExaminer does:

```toml
[input.scans]
find_peak = true
```

For each run and peptide, *hxex* follows the intensity at the peptide's predicted isotope
positions, scan by scan, within 30 seconds either side of its `rt` (`peak_search`). It smooths
this trace, takes the peak nearest to `rt` (not the tallest one: another species often elutes
a few seconds away), and sums the scans from the peak's apex to half its width at half height
either side (`peak_width`), stopping where the trace turns up again. The charge states of a
peptide come out of the column together, so they share one peak. The smoothing follows the
run: half its typical peak width, measured on its strongest peptides, so it suits a short or
long gradient and a fast or slow scan rate alike; `peak_smoothing` sets it in seconds instead.
The run log reports each run's measured peak width and the smoothing used.

A peptide with its own `rt_start` and `rt_end` keeps them. To correct one file, give the
peptide its own window there in the elution-window table (see [A peptide's own window in one
file](#a-peptides-own-window-in-one-file); the window's elution tab writes it); the other files
keep finding the peak. Every peptide needs
either `rt` or `rt_start` and `rt_end`; `hxex check` names any that
has neither. Fragment ions keep their windows; `find_fragment_peak` (next section) finds theirs.

On a whole-cell digest measured on an Orbitrap (563 peptides, 45 runs), the peaks found were
a median of 0.8 seconds from those HDExaminer used, and the spectra that wrongly needed a second
population fell by about a third against a fixed window (HISTORY 164).

### Finding each fragment channel's peak

A fragment ion is read from the scans of its precursor, in the window its parent's row in the
peptide table gives. One table serves every file, but the retention time moves a little from
run to run: on a targeted dataset (18 precursors, 17 runs) a peptide eluted up to about 0.6
minutes earlier in some runs than in others. Where it moved out of the window, part of its
signal was left out, although the instrument had recorded it. To have each run find its own
peak:

```toml
[input.scans]
find_fragment_peak = true
```

For each run, *hxex* takes the fragments of one precursor that are read from the same scans
(one condition and MS level: a *channel*) and follows them together: in every scan, the
intensity at all their predicted peaks, added up. It then finds the peak as `find_peak` does
for peptides: smoothed over half the typical peak width of the run's fragment channels, the
peak nearest the parent's `rt` (or the middle of its window when the table gives no `rt`),
and the scans from its apex to one width at half height either side (`fragment_peak_width`),
stopping where the trace turns up again. It looks over the parent's window and `peak_search`
either side of `rt`, whichever reaches further. Every fragment of the channel then sums the
same scans.

The window is wider than a peptide's (`peak_width`, half a width) on purpose. In reduced profile
data the scans beside the peak add no noise, while a narrower window leaves signal out: on the
targeted dataset half widths kept a median 77 % of a fragment's signal and whole widths 97 %,
and with whole widths the fragments' uptake rose more cleanly with time.

All the fragments are followed together because a single ion can follow another species: on
that dataset the precursor's own ion peaked 0.6 to 0.8 minutes away from the peptide for one of
the 18 precursors, in every run, while the fragments together found the peptide within a second
of where its MS1 parent peaked (median 0.6 s, none beyond 15 s).

A fragment of a run's own row (the `run` column) keeps that row's window.

**When the instrument cut the peak.** A targeted method records each precursor only in a
scheduled window. When the peptide eluted early or late in a run, the peak can reach the first
or last scan the instrument recorded for it: the rest of the peak was never measured, and
*hxex* cannot recover it. Such spectra are flagged `elution_cut`. They are dimmer than their
elution, and in reduced profile data dim fragment spectra read less reliably (see [Reduced
profile data](#reduced-profile-data)). On the targeted dataset 8 of 612 channels were cut this
way: the four channels of one early-eluting peptide, in the two runs that eluted earliest.

### A peptide's own window in one file

The peptide table says what to read, in every file alike. When a peptide's window is wrong in
one file (its peak eluted early in that injection, say, or a neighbour's peak crept in),
correct it there only, in the elution-window table:

```toml
[input]
rt_elution_windows = "rt_elution_windows.csv"
```

| sample | file | peptide | ion | rt_start | rt_end | drift_start | drift_end |
|---|---|---|---|---|---|---|---|
| WT | D_60s_01.raw | 0012-0020-LVKDLEPAQ | | 5.62 | 5.81 | | |
| WT | D_60s_02.raw | 0002-0013-ASKELVDPLRKE | | 8.62 | 8.97 | | |
| WT | D_60s_02.raw | 0002-0013-ASKELVDPLRKE | 0002-0013-ASKELVDPLRKE-z3 | 8.70 | 8.90 | | |

Each row names a sample and a file (its name, as in `[[input.runs]]`), and then either a
peptide or a single ion:

- **A peptide** (its range and sequence, as the spectrum ids begin: `0012-0020-LVKDLEPAQ`, with
  its modification when it has one): every ion of that peptide in that file takes the window.
  That is each of its charge states and, when you read fragments, the peptide as a parent and
  all its fragment ions. They are the same molecules, so they come off the column together: on
  the data we measured, the charge states of one peptide peaked within half a second of each
  other, and fragment channels within half a second of their parent. One window for all of them
  is the physical choice, and it is what you will usually want.
- **One ion** (exactly as the spectrum ids name it: the part of a `spectrum_id` between the file
  and the scans; `spectra.csv` lists them): that ion alone, for the rare case of interference at
  that ion's own m/z (another species that elutes at a slightly different time, so that this
  one ion needs a narrower window). An ion's row wins over its peptide's row. Give the
  peptide on an ion's row as well: two peptides that start (or end) with the same residues
  give fragment ions of the same name, which elute with their own peptides, and the peptide
  says which one the row is for (a row without it is for the ion of every peptide that has
  one of that name). On ion-mobility data, drift times differ between charge states, so a drift range (`drift_start`, `drift_end`,
  milliseconds) is always given on an ion's row; such a row may leave `rt_start` and `rt_end`
  empty to keep the peptide's window.

`rt_start` and `rt_end` are minutes on the file's own clock. The spectra then sum the scans
inside the window, in that file only: `find_peak` and `find_fragment_peak` leave them alone,
the other files read as before, and every spectrum stays in its series with the same controls.

Cover the whole peak. Molecules that carry more deuterium leave the column very slightly
earlier, so the front of a deuterated peak reads heavier than its tail (by 0.1 to 0.2 D on the
data we measured), while the undeuterated control has no such difference. A window around only
half of the peak would shift the labelled spectra but not their reference.

You rarely type this table: the elution tab of the window's Results page writes it (**Edit
window**, then **Save window**, for all ions of this peptide or for this ion only; see
[Results](gui.md#results)), and the first save adds the setting to your project file. A table
written before peptide rows existed has an `ion` column only; its rows keep their meaning (one
ion each). `hxex check` says how many windows the table holds and names any row whose sample,
file, peptide, or ion the project does not read (a file renamed, a peptide taken out of the
table), and the run warns about such rows and leaves them unused. Edited windows only act on
files with a time (time points and controls), not on kinetics files. The table only acts when
the project is run (`hxex run --overwrite`). The new window gives the spectra a new
`spectrum_id` (it names the scans added up), and your review decisions follow them there (see
[When a spectrum's id changes](project-file.md#the-user-review-table)).

### When the peptide table's time is wrong

`find_peak` trusts the `rt` in your peptide table: it takes the peak nearest that time. If the
time is wrong, it finds whatever elutes there. On a whole-cell digest we checked against the
dataset's own identification runs, 50 of the 77 series in question were read at a time where
another peptide of almost the same mass elutes; the peptide itself came off the column 0.3 to 2
minutes away. Such a series usually shows up as [off its m/z](picking-window.md#a-series-whose-peaks-sit-off-its-mz).

With `find_peak` on, *hxex* therefore looks for each such series' exact m/z in the undeuterated
controls:

```toml
[input.scans]
find_peak = true
find_retention_time = "suggest"   # the default; or "automatic", or "off"
```

It looks at every series that sits off its m/z, that has no undeuterated envelope *hxex* could
place, or that has no undeuterated spectrum at all. In every scan of each undeuterated control
it notes the peaks whose tops sit at the peptide's predicted M-1, M, M+1, and M+2 (a top, not
just intensity nearby: the side of another species' peak does not count). A peak in time counts
as the peptide when:

- every undeuterated injection of the sample shows it at about the same time;
- its isotope peaks have roughly the heights the peptide's composition predicts, and there is no
  tall peak one isotope step below it (which would make it another species' second isotope);
- its m/z sits as close to the peptide's as the run's other peptides sit to theirs.

The verdict for each series is one of:

| Verdict | Meaning |
|---|---|
| `here` | the peptide elutes where *hxex* read it; something else shares its m/z |
| `elsewhere` | the peptide elutes clearly at another time: `proposed_rt` |
| `uncertain` | one peak elsewhere, but weaker than the signal at the peptide's m/z where it was read |
| `ambiguous` | several peaks fit, none at least three times taller than the others; all are listed |
| `not found` | nothing fits (often a peptide merged with another, or a modification that differs between your identification sample and the HDX samples) |

With `"suggest"`, nothing changes in the results: the verdicts go into
`retention_suggestions.csv` beside the other tables, and `hxex run` says how many series were
found elsewhere. Check those times (in your identification results, for example) and correct the
`rt` column of your peptide table. With `"automatic"`, *hxex* also reads every `elsewhere` series
at the time it found, in every run, and flags it `rt_found`; it does this only when at least two
undeuterated injections agree.

Two tolerances come from the run itself, so they suit any instrument and gradient:

- **m/z** (`rt_search_ppm`): a top counts within 1.25 times the limit of the
  [m/z check](picking-window.md#a-series-whose-peaks-sit-off-its-mz) for that injection, which
  follows how closely the run's peptides sit to their m/z. On an Orbitrap digest this was 5.1
  ppm. The run log reports it for each injection.
- **Time** (`rt_search_tolerance`): the injections must agree within the run's typical peak
  width at half height, or, if larger, within the spread that 99 % of the peptides show between
  the undeuterated injections. On the same digest this was 6.5 seconds.

Set either one to fix it instead. The search reads every MS1 scan of the undeuterated controls
once more, a few seconds per file.

On that digest (two undeuterated injections), 102 series were searched. Of the 40 whose true time
was known from the identification runs, 33 were found at that time, 3 were ambiguous (the right
time among the candidates), 3 were not found, and 1 was found on the same peak as the
identification but 0.2 minutes from its scan. Of all 52 proposals elsewhere, 44 matched the identification runs within
0.2 minutes, 6 had no identification to compare with, and 2 did not match: one 0.22 minutes off
against a weak identification, and one cysteine peptide, which the identification sample had
alkylated and the HDX samples had not. Only one Orbitrap dataset has been checked so far, so
`"automatic"` is not the default.

### Each file's m/z offset

An instrument's m/z scale can be off by some ppm, and the offset can drift from one
acquisition to the next. With `mz_calibration = "auto"` (the default), *hxex* measures each
file's offset from its own spectra before picking:

1. Every peptide spectrum of the file is picked once, and the picker notes how far its peaks
   sit from where the peptide's composition puts them (the intensity-weighted median, in ppm).
2. The file's offset is the median of those over its spectra with signal: undeuterated,
   labelled, and fully deuterated spectra alike.
3. The file's peptide spectra have their m/z divided by that offset, and are picked from there.

No reference compound or instrument setting is needed. The log gives the range of the files'
offsets, `hxex info` repeats it, and `spectra.csv` gives each spectrum's in
`mz_calibration_ppm`. A file with fewer than 10 peptide spectra with signal keeps its m/z as
read. Fragment ions keep their m/z. An offset is measured inside the picker's window, so the
log warns about files near the edge of the window (more than 45 ppm off with
`[run.picker] resolution_ppm` at its default of 50 ppm): widen the window to check them.

`mz_calibration = false` uses the m/z as the files give them, as earlier versions of *hxex* did.

On a Q-TOF dataset whose uncorrected files drifted from +16 to +40 ppm over the acquisition,
each file's measured offset agreed within 0.5 ppm with the correction HDExaminer had applied plus
what it had left. The fits then agreed with fits of HDExaminer's corrected files (the same number
of populations in 99.5 % of spectra), and replicates agreed slightly better. Without the
correction, about 5 % of the spectra in the files furthest off chose a different number of
populations.

### Ion mobility: each peptide's drift time

Files converted with their drift-time bins together (Waters HDMS data, msConvert's
`--combineIonMobilitySpectra`) hold every ion that reaches the detector in a scan, whatever its
drift time. Another ion at a nearby m/z that elutes with the peptide but drifts at another time
would then be added to the peptide's spectra. So, with `drift_gate = "auto"` (the default),
*hxex* keeps only each peptide's own drift time:

1. In the peptide's undeuterated controls (every sample's), it adds up the intensity at the
   peptide's natural isotope peaks at each drift time, over the scans its spectra take.
2. The peptide's range is the drift time where that intensity is highest, and the drift times
   either side of it down to a tenth of that height.
3. Every spectrum of the peptide, in every file, is summed over that range only. Deuterium
   does not change the drift time measurably, so one range serves every time point.

Each charge state of a peptide gets its own range. Drift times are used as the file records
them, in milliseconds; no instrument setting is needed. `spectra.csv` gives each spectrum's
range in `drift_start` and `drift_end`, and the log says for how many peptides a range was
found.

When the undeuterated drift profile has a second peak, at least half as high as the first,
outside its range, the peptide's own isotope spacing decides between the two. An ion of
another charge at the same m/z drifts at another time, and its isotope peaks fall between the
peptide's as well as on them. So *hxex* compares how much of each peak's intensity sits on the
peptide's isotope peaks rather than half-way between them, and takes the peak where clearly more
does (by a fifth of the intensity or more), even when it is the lower of the two.

A peptide without an undeuterated control with signal keeps every drift time, and its series
is flagged `drift_unclear`. So is a peptide whose two drift peaks the isotope spacing does not
tell apart: another ion of the same charge at that m/z, or the peptide in two shapes. Look at
its spectra; if you know its drift time, give it in the peptide table's `drift_start` and
`drift_end` columns (milliseconds), which take the place of the range found. Fragment ions keep
every drift time.

`drift_gate = false` sums every drift time, as earlier versions of *hxex* did. Files without
drift times are read the same either way.

On a Q-TOF dataset with five samples, the ranges *hxex* found were mostly four drift bins of
200, close to the five bins HDExaminer had chosen for the same peptides. The fits agreed with
fits of HDExaminer's drift-gated spectra: the same number of populations in 97 % of spectra,
deuterium within 0.01 D (median), and replicates that agreed as closely. Summed over every drift
time instead, a fifth of the spectra chose a different number of populations and the
replicates disagreed twice as much at the worst end.

### Thermo raw files

`format = "thermo"` reads `.raw` files directly, with the same settings as mzML. It needs:

- *hxex* installed with its Thermo extra: step 3 of [Installing *hxex*](install.md) with
  `"hxex[thermo] @ https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip"` in place of
  the plain address (the quotes are needed);
- a .NET runtime: Mono on Linux and macOS, the .NET Framework on Windows (built in), or .NET 8 or
  newer with `PYTHONNET_RUNTIME=coreclr`;
- Thermo's `ThermoFisher.CommonCore.Data.dll` and `ThermoFisher.CommonCore.RawFileReader.dll`
  (they come with [ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser)), in
  a folder named by `[input] thermo_dir` or the `HXEX_THERMO_DIR` environment variable.
  RawFileReader is under Thermo's own license.

The raw files also record each scan's resolving power and the instrument model. `hxex check`
prints them, and with `pick = "window_max"`, `[run.picker] resolving_power = "recorded"` sizes
the picking window from them (see [The picking window](picking-window.md)). mzML files carry no resolving power.

### Reduced profile data

Orbitrap instruments usually store their profile scans "reduced": to keep the files small, every
point below the instrument's noise threshold is set to 0. In each scan, a peak whose top is below
about twice the noise is deleted altogether; stronger peaks keep their full height. *hxex* sums
the scans of a peptide's retention-time window, so a weak isotope peak loses every scan in which
it sat under the threshold: it reads too low, or 0. The peaks hit are the small ones: the heavy
side of a dim envelope, and weak fragment ions. On the one Orbitrap digest measured, isotope
peaks of under a few percent of the envelope's tallest went missing in most dim spectra, and
the fits read the missing heavy side as a second population more often.

*hxex* notices reduced profile scans when it reads the files: about half their points are
exactly 0. `hxex check` and `hxex run` warn once for the dataset, and `hxex info` repeats it.
For Thermo raw files the warning also gives the weakest peaks kept, as a multiple of the noise
the files record. The warning ends with what *hxex* does about it: fragment ions are fitted
through the cut when the files record their scans' noise (next section); peptides read in MS1
scans are fitted without allowing for it. To reduce the lost signal when you acquire the data,
more ions per scan (a higher AGC target or a longer maximum injection time) lifts weak peaks
above the threshold; whether an instrument can store full profile scans is a question for
its maker. Only Orbitrap data are checked: Thermo raw files' Orbitrap scans, and mzML files
that name an Orbitrap (or another Fourier-transform) analyser. Other instruments leave zeros
too, such as a Q-TOF's empty channels where no ion arrived, but those are not this threshold. An
mzML file records no noise, so its warning gives no multiple.

#### Fitting fragment ions through the cut

For fragment ions, *hxex* allows for the lost signal. When a file's profile scans are reduced
and the file records each scan's noise, *hxex* keeps, for every fragment spectrum, the noise each of its scans recorded and how much of
the signal each scan holds (from the elution of the fragment's whole ladder, as
`find_fragment_peak` measures it). The fit then compares the data with what the instrument would
have kept of the fitted envelope: each predicted peak is split over the scans, every scan loses
its weak peaks as the instrument does (half of a peak's signal is lost at about twice the noise,
almost none above 15 times the noise), and the scans are added up again. A dim envelope's
missing heavy side is then expected, not read as a lower deuteration.

On one targeted Orbitrap dataset, this halved the amount by which a c ion and the matching z
ion fell short of their parent's deuteration (the closure): a median of 0.31 deuterons short
became 0.13, and 1.0 to 1.2 short in moderately dim pairs became 0.4. The brightest pairs, which
the cut hardly touches, moved from 0.14 to 0.07 short.

Things to know:

- Only fragment ions are fitted this way, and only with one population (K = 1): a fit through
  the cut cannot be compared fairly with a two-population fit that ignores it. Peptides read
  in MS1 scans are fitted as before.
- It needs the noise each scan recorded. Thermo raw files have it (`format = "thermo"`). An
  mzML file has it only when it was converted with ThermoRawFileParser's `-N` option, which
  writes each scan's noise. A reduced file without it is read as before, and the warning names
  how many such files there are: convert them again with `-N`, or read the raw files.
- The setting is `[input.scans] scan_cut`. `"auto"` (the default) does the above for the files
  whose scans are reduced and record their noise. `true` fits every file's fragments through
  the cut and stops at a file that records no noise. `false` never does:

  ```toml
  [input.scans]
  scan_cut = false
  ```

- The fitted curve drawn over a fragment spectrum is the expected reading, so it follows the
  data down where weak peaks were lost.
- The results take more room (the noise of every scan at every peak): on the dataset above,
  about twice as much.
- How much of a peak the instrument keeps was measured on one Orbitrap's undeuterated fragment
  ions. Below about 4 times the noise at the elution's top, no fit recovers the deuteration.
- Fragment spectra whose tallest peak, in the scan where it is largest, is below 4 times the
  noise that scan recorded are flagged `below_noise`, and `spectra.csv` gives that size as
  `signal_to_noise` (`[run.quality] min_signal_to_noise`, 4 by default). So many of such a
  spectrum's peaks were deleted scan by scan that even the fit through the cut cannot recover
  its deuteration. On the dataset above, 38 % of the labelled and fully deuterated fragment
  spectra were below 4 times the noise (23 % below the noise itself). Between 1 and 4 times,
  a c ion and its z ion still fell 0.37 deuterons short of their parent, against 0.11 between
  4 and 16 times; the two readings of one fragment (with and without the extra activation step)
  scattered by 0.50 deuterons, against 0.29. The flag only marks them; nothing is left out. To
  mark only the nearly empty spectra, set `min_signal_to_noise = 1`.

### What `hxex check` shows for scan formats

One line per run: its sample, replicate, time or role, the number of scans, and how many targets
have scans in it, followed by up to ten targets with none, the instrument (Thermo raw files),
and whether its profile scans are reduced. A run where no target has scans is a
problem; the usual causes are a wrong MS level, activation or filter in `[input.target]`, a
retention window that misses the peptides, or a charge state the file records differently.

---

## Choosing between them

- Spectra already extracted in HDExaminer: `specexport` (one protein or sample per folder) or
  `pooled` (many proteins, no designated samples).
- Spectra extracted in HX-Express: `hxexpress`.
- Raw data, or anything the extraction tools do not handle (fragment ions, kinetics runs where
  every scan is a time point, MS3): `mzml`, or `thermo` to skip the conversion to mzML and retain instrument resolution information.
