# Input formats

*hxex* reads five kinds of input, chosen by `[input] format` in the project file:

| Format | Kind | What it reads | Where the peptides come from | Where the times and controls come from |
|---|---|---|---|---|
| `hxexpress` | folder | HX-Express spreadsheets (`.xlsx`, `.csv`) | the file names | the column labels of each sheet |
| `specexport` | folder | an HDExaminer SpecExport folder, one sub-folder per sample | the folder names, cut from each sample's FASTA | the spectrum file names |
| `pooled` | folder | a SpecExport folder holding the peptides of several proteins, without samples | the folder names (the sequence is written in them) | the spectrum file names |
| `mzml` | scan | mzML files from the instrument | a peptide table (CSV) | `[[input.runs]]` in the project file, or each scan's own time |
| `thermo` | scan | Thermo `.raw` files, read directly | a peptide table (CSV) | as `mzml` |

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
`hr`, `ms` and their long forms).

**Controls** are identified by a *role*, not by a time: `undeut` (the undeuterated control) or
`fulldeut` (the fully deuterated control). The folder formats take the role from a label
(`Non` and `Full` in SpecExport, `undeut` and `TD` in HX-Express); the scan formats take it from
`role` in `[[input.runs]]`. A control has no time in the tables and figures. Replicate controls
are averaged. See [Controls and times](controls.md) for what the controls are used for and how
to use a time point as a control.

**Sequences** are written in modX: a modified residue is a lowercase prefix followed by its
residue letter (`acK`, `pS`, `pT`, `pY`), and a sequence is read residue by residue. Other
modifications are defined in `[[run.registry.residues]]` (a label, its base residue and the
composition it adds). `hxex check` reports any peptide it cannot read, with the reason.

**Filters.** The folder formats can read only some of the data:

- `samples = ["WT", "mutant"]`: only these samples.
- `peptide_ranges = ["1-10", "45-60"]`: only these peptides, by their residues. `"1-10"` and
  `"0001-0010"` mean the same peptide, from residue 1 to residue 10.
- `peptide_region = [1, 50]`: only the peptides that lie wholly within residues 1 to 50.

With both `peptide_ranges` and `peptide_region`, a peptide that matches either is read.

**Retention times** (minutes) can be given for any format in a separate table,
`[input] retention_times = "rt.csv"`: a column `rt` and any of `sample`, `start_seq` and `end_seq`
(or `peptide_range`), `peptide`, `charge`. An empty cell matches anything, and each spectrum takes
the row that names the most columns. The scan formats also have their own sources (see below).

**Recorded paths.** `record_paths` says how the files read are recorded in the bundle:
`"path"` (as given, the default), `"name"` (file names only, for results to be shared), or, for
the scan formats, `"relative"` (relative to `data_root`).

**Rerunning from stored spectra.** A bundle keeps the raw spectra it was made from (unless
`[output] keep_spectra = false`). `hxex run hxex.toml --spectra old_results`, or
`[input] spectra = "old_results"`, fits them again without reading the files: useful after
changing the picker or fit settings, or when the original files are not at hand. The format then
comes from the stored record. `retention_times` and `time_scale` still apply.

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

Only the first worksheet is read, with formulas read as their saved values. Row 1 holds a label
above each spectrum. The spectrum is the label's column (m/z) and the next column (intensity),
from row 2 down. Empty cells are skipped.

| Label | Meaning |
|---|---|
| `undeut`, `Undeut_1`, `undeuterated` | undeuterated control (any label starting with `undeut`) |
| `TD`, `td` | fully deuterated control (any label starting with `TD`) |
| `60 s`, `5 min`, `2 h` | a time point: a number, a space, a unit |
| `1 min`, `1.2 min` | two replicates of 1 min (see below) |

Units are `ms`, `s`, `sec`, `m`, `min`, `h`, `hr` and their long forms. The space is needed:
`60s` is refused. `FD` and `Full` are not control labels here.

**Replicates.** Spectra with the same label are numbered as replicates in column order. As in
HX-Express, digits after the decimal point of a time of 1 or more mark a replicate, not a
fraction: `1.2 min` is a second replicate at 1 min, and `10.3 s` is 10 s. Below 1 the fraction is
kept: `0.5 min` is 30 s.

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
      Non-1-z2.csv
      Non-2-z2.csv
      30s-1-z2.csv
      5m-1-z2.csv
      Full-1-z2.csv
  mutant.fasta
  mutant/
    ...
```

- **A sample** is a folder with a FASTA file of the same name beside it (`WT/` and `WT.fasta`).
  A folder without its FASTA, or a FASTA without its folder, is ignored. Only `.fasta` is looked
  for. If the FASTA has several records, the last is used (with a warning).
- **A peptide** is a folder named `START-END-...`. Its sequence is cut from the sample's FASTA by
  the range; the sequence in the folder name is not used. Other folders (for example an
  `analysis` folder) are ignored. A range outside the FASTA sequence is an error that names the
  folder.
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
| `Non` | undeuterated control |
| `Full` | fully deuterated control |
| `30s`, `5m`, `1.5m`, `4h` | a time point: a number and one letter, `s`, `m` or `h` |

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
    Non-1-z2.csv
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

For HX-Express and SpecExport data, *hxex* normally finds the files by their names. A table of
the files can replace that search. Use one when:

- the file names do not follow the pattern *hxex* expects;
- a sequence needs something the names or the FASTA cannot hold: a terminal group (`Ac-`,
  `-NH2`) for HX-Express, any modification for SpecExport.

Name the table in the project file:

```toml
[input]
format = "hxexpress"            # or "specexport"
data_dir = "data"
peptides = "files.csv"          # the table of the files
```

The table is a CSV file, with one row per peptide and charge. Excel can save one (**File > Save
As**, then **CSV**); check that the columns are separated by commas, as in the example below,
since in some countries Excel uses semicolons instead.

```
file,sample,peptide,start_seq,end_seq,charge
my_peptide.xlsx,WT,Ac-PEPacKTIDE-NH2,1,8,2
```

| Column | Meaning |
|---|---|
| `file` | HX-Express: the spreadsheet's name, in `data_dir`. SpecExport: the peptide folder's name, inside the sample's folder (`0001-0008-PEPKTIDE`) |
| `sample` | the sample |
| `peptide` | the full sequence: modified residues in modX (`acK`, `pS`) and terminal groups (`Ac-`, `-NH2`) |
| `start_seq`, `end_seq` | the first and last residue numbers |
| `charge` | the charge state |

These six columns are required; a table without one is refused, naming it. Three more are
optional:

| Column | Meaning |
|---|---|
| `peptide_range` | the range's label in tables and figures; left out, it is made from `start_seq` and `end_seq` (`0001-0008`). A label that names other residues gets a warning: the residues are always `start_seq` to `end_seq` |
| `rt` | the retention time, minutes, kept with the spectra |
| `modification` | extra composition, in pyHXExpress's syntax |

Other columns are kept but not used.

The table replaces the search of the folder: only the files it lists are read, so list every
peptide, modified or not. `samples` and `peptide_ranges` still select from it. `hxex check`
reports any sequence it cannot read.

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

### The peptide table

A CSV file (UTF-8). Required columns:

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
| `modification` | extra composition in pyHXExpress's syntax (`H:1 Hex:1 ion_type:c undeut_mz:523.1`) |

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
| `rep` | replicate number (default 1) |
| `time` | the labelling time of the whole file (`"60 s"`, `"5 min"`) |
| `role` | `"undeut"` or `"fulldeut"`: a control run |
| `time_offset`, `usable_from` | for kinetics runs (below) |
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

Profile and centroided scans are both read. Summed profile scans are put on a common m/z grid;
centroided scans are merged peak by peak. Profile data is recommended: when converting Thermo
files to mzML with ThermoRawFileParser, keep profile data (`-p`) and use zlib rather than
numpress compression (numpress is refused).

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
the signal-weighted time of the summed scans. Kinetics spectra have none.

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
prints them, and `[run.picker] resolving_power = "recorded"` sizes the picking window from them
(see [The picking window](picking-window.md)). mzML files carry no resolving power.

### What `hxex check` shows for scan formats

One line per run: its sample, replicate, time or role, the number of scans, and how many targets
have scans in it, followed by up to ten targets with none. A run where no target has scans is a
problem; the usual causes are a wrong MS level, activation or filter in `[input.target]`, a
retention window that misses the peptides, or a charge state the file records differently.

---

## Choosing between them

- Spectra already extracted in HDExaminer: `specexport` (one protein or sample per folder) or
  `pooled` (many proteins, no designated samples).
- Spectra extracted in HX-Express: `hxexpress`.
- Raw data, or anything the extraction tools do not handle (fragment ions, kinetics runs where
  every scan is a time point, MS3): `mzml`, or `thermo` to skip the conversion to mzML and retain instrument resolution information.
