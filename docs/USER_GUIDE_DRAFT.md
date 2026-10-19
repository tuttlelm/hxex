# User guide: draft material

Sections written for users of hxex as features are built, to be assembled into the user guide
(docs/PLAN.md, design item 5). Each section stands on its own. Developer notes stay in
`docs/HISTORY.md`; this file explains what a setting does and when to use it.

## The peak-picking window

hxex turns each spectrum into sticks: one intensity per deuterium bin, taken at the m/z
predicted for that bin from the peptide's composition and charge. The **picking window** is how
far from the predicted m/z hxex looks for the peak. Two settings in `[run.picker]` choose it.

### The default: a ppm tolerance

```toml
[run.picker]
resolution_ppm = 50.0   # half-width of the window, ppm of the predicted m/z
```

Each bin takes the tallest point within 50 ppm either side of its predicted m/z. This is a
mass-accuracy tolerance (the same as pyHXExpress's `Peak_Resolution`). It is simple and it is
the setting every earlier hxex result was made with, so it stays the default.

The window never reaches further than half the spacing of the isotope peaks, about 1 Da divided
by the charge. That matters only at high charge: at m/z 1000 a 50 ppm half-width is 50 mDa, a
tenth of the spacing for a 2+ peptide (peaks 500 mDa apart) but the whole spacing for a 20+
protein (peaks 50 mDa apart). Without the cap each window would hold both neighbouring peaks,
and hxex would discard the bins whose neighbour is taller. For peptides the cap never applies.

### A window from the instrument's resolving power

```toml
[run.picker]
resolving_power = 60000.0        # R, quoted at resolving_power_mz
resolving_power_mz = 200.0       # Thermo quotes R at m/z 200 (400 on the Elite/Velos generation)
resolving_power_exponent = 0.5   # how R falls with m/z: 0.5 Orbitrap, 0 TOF, 1 FT-ICR
window_fwhm = 1.0                # half-width in peak widths (full width at half maximum)
mass_accuracy_ppm = 10.0         # the half-width never falls below this
```

With `resolving_power` set, the window follows the peak width. hxex takes the resolving power
at each predicted m/z (`R(m) = R0 (m0 / m) ** exponent`, so an Orbitrap's R falls with the
square root of m/z), computes the peak's width at half maximum (`m / R(m)`), and uses
`window_fwhm` of those widths as the half-width. Two guards apply:

- the half-width never falls below `mass_accuracy_ppm`, so a very high resolving power does not
  shrink the window below the instrument's calibration error (series on a well-calibrated
  Orbitrap sit within a few ppm of their predicted m/z; a wrong sequence or modification moves
  them by tens of ppm, which the `mz_shift` quality flag reports);
- the half-width never exceeds half the isotope spacing (`1.00235 / (2 z)` Da), so the windows
  of neighbouring bins never overlap.

For R = 60,000 at m/z 200 that gives a half-width of 29 ppm at m/z 600 and 46 ppm at m/z 1500,
and at m/z 1000 with charge 20 the cap gives 25 mDa where the peak width would be 37 mDa.
`resolution_ppm` is not used while `resolving_power` is set (it remains the fallback, below).

### Taking the resolving power from the raw files

```toml
[run.picker]
resolving_power = "recorded"
```

Thermo `.raw` files record the resolving power of every scan (MS1 and MS2 scans often differ)
and the instrument model. With `"recorded"`, hxex uses each spectrum's own value, and the
model tells it whether the value is quoted at m/z 200 or 400. `hxex check` shows what a file
records:

```
run sample_01.raw: sample WT, rep 1, scan times, 885 scans; 12 of 12 targets have scans
  instrument Orbitrap Ascend; resolving power 120000 at MS1, 30000 at MS2 (quoted at m/z
  200); [run.picker] resolving_power = "recorded" would use it
```

mzML files (including those converted from `.raw` with ThermoRawFileParser) and the folder
formats (SpecExport, HX-Express) carry no resolving power. Spectra without a recorded value keep
the `resolution_ppm` window; `hxex run` logs how many. For those formats, set `resolving_power`
to a number instead.

### Seeing the window

The spectrum figures (`hxex report --figures`, `hxex.report.plot_spectrum`) draw every picked
bin as a shaded bar spanning the window hxex actually used for that spectrum, and the legend
says where the window came from: "±50 ppm window", "window from R 60000 at m/z 200" or
"window from the recorded R 120000".

### Which to choose

- Intact peptides at low charge on a calibrated instrument: the default is fine.
- High charge states (top-down, large fragments, intact proteins): set `resolving_power`, or
  `"recorded"` for Thermo raw files, so the windows stop overlapping.
- Centroided data or coarse profile data: either works; the shoulder rule (windows of ten or
  more points whose maximum sits at the edge are set to zero) does not apply to windows with
  fewer points.

The two modes do not give identical sticks on the same spectrum, so compare runs made with the
same setting. The results bundle records the picker settings, and `hxex info` shows them.

## Controls and times

Every series (a peptide at one charge) has two kinds of spectra: **time points**, measured
after a labelling time, and **controls**: the undeuterated spectrum and the fully deuterated
one, which set the scale of the back-exchange correction. hxex tells them apart by a **role**,
never by a number.

### Where the role comes from

- **HDExaminer SpecExport and HX-Express files.** The labels already say it: `Non` (SpecExport)
  and `undeut` (HX-Express) files are undeuterated controls, `Full` and `TD` files are fully
  deuterated controls. Nothing to set.
- **mzML and Thermo raw files.** Each control file gets a `role` in its `[[input.runs]]` table
  and no `time`:

  ```toml
  [[input.runs]]
  path = "UN_01.raw"
  sample = "WT"
  role = "undeut"

  [[input.runs]]
  path = "TD_01.raw"
  sample = "WT"
  role = "fulldeut"

  [[input.runs]]
  path = "D_5min_01.raw"
  sample = "WT"
  time = "5 min"
  ```

  A file with neither `time` nor `role` is a kinetics run, where every scan has its own time.

A control carries no time: its `time` is blank in every table and the figures show it as `UN`
or `FD`, first and last on an index axis, as a line on a time axis. That is why a
back-exchange experiment, where a real time point can be at 0 s next to the undeuterated
control, needs no trick to keep the two apart.

![Uptake of one peptide against time: the controls are the dotted limits at 0 and the number of amides](images/guide_uptake_time.svg)

![The same peptide by index: UN first, FD last, every point labelled with its time](images/guide_uptake_index.svg)

*The same series of the published B5 data drawn against time (`x = "time"`, log scale; the
controls set the dotted limits) and by index (`x = "index"`; the controls are the first and
last points).*

### Seeing the roles

`hxex check` counts the control files of every sample before anything is read (for HX-Express
sheets, whose labels are inside, it says so instead), and names each scan-format run's role:

```
entries        9 in SpecExport
  D109H: 3 entries, 9 undeuterated and 5 fully deuterated control files
  R120G: 3 entries, 6 undeuterated and 5 fully deuterated control files
  WT: 3 entries, 9 undeuterated and 9 fully deuterated control files
```

```
run UN_01.raw: sample WT, rep 1, undeuterated control, 412 scans; 12 of 12 targets have scans
run D_5min_01.raw: sample WT, rep 1, time 5 min, 398 scans; 12 of 12 targets have scans
```

Its `controls` line repeats any `undeut_at`, `fulldeut_at` and `*_marked = false` setting in
words. After the run, `hxex info` counts the control spectra of the bundle (and the times they
also have, when they are time points too) and says whether the times are real:

```
controls       yes
roles          24 undeuterated (9 series); 19 fully deuterated (9 series)
times          seconds
```

### A time point that is also a control

Some data have no separate fully deuterated file, but a late time point that has reached its
plateau. `fulldeut_at` in `[run.controls]` makes the spectra at that time serve as the fully
deuterated control as well; they keep their time, so they stay a point of the time course:

```toml
[run.controls]
fulldeut_at = "20 h"
```

`undeut_at` does the same for the undeuterated control, for example the equilibrium end point of
a back-exchange run. When the data also have a marked fully deuterated file that you do not
trust (one that reads low, say), add `fulldeut_marked = false`: the marked file then does not
serve as the control, only the spectra at `fulldeut_at` do, and its spectra stay fitted. The
same settings exist in `[analysis.controls]` for an analysis of an existing bundle. The time must be one at which spectra were measured; otherwise hxex lists the
times it has. For a scan-format file, giving the run both `time` and `role` in `[[input.runs]]`
does the same thing.

### Writing times

Every time in the project file is a number of seconds or text with a unit: `60`, `"60 s"`,
`"5 min"`, `"20 h"`, `"250 ms"`, `"1h21m"`. The units are `s`, `sec`, `m`, `min`, `h`, `hr`,
`ms` and their long forms, in any case. hxex converts them when it reads the file, so the tables
always hold seconds; figures label times in a readable unit (`20 h`, not `72000 s`). In an
HX-Express sheet the labels keep that format's own convention, where `1.2 min` means the second
spectrum at 1 min; in the project file `"1.5 min"` is 90 s.

### Times that only order the points

Some data carry no labelling times, only labels that order the points (1, 2, 3, or the
HX-Express test sets' dummy times). `[input] time_scale = "order"` says so:

```toml
[input]
time_scale = "order"
```

The numbers are still read and stored as given, but hxex never treats them as seconds: the
uptake is drawn by index whatever `[run.plots] x` says (with a warning when it says
`"time"`), figures and file names label the points by their number (`point 2`,
`profile_WT_2.png`), and the analyses that need real times, the fragment ladder kinetics and
the protection factors, refuse with a message that names the setting. The scale is recorded
with the spectra, so a rerun from a stored bundle keeps it unless the project file sets it
again, and `hxex check` and `hxex info` show it on their `times` line.

## Comparing separate runs

One `hxex run` makes one results bundle. To compare bundles, give `hxex analyse` several of
them, each under a name of your own, or list them in the project file:

```toml
[analysis]
reference = "WT"          # the sample the others are compared with

[analysis.runs]           # run name = bundle
A = "day1/results"
B = "day2/results"
```

```bash
hxex analyse A=day1/results B=day2/results --reference WT
```

The runs are analysed like one run with more samples. Every table then has a `run` column
before `sample`: a row of sample `WT` from run A reads `A,WT,...`. Filter on the two columns
in a spreadsheet or a script; the sample's name is the same in every run.

**Naming a sample.** `reference` and `samples` take a sample's name. If that name is in only
one of the runs (run A has `WT`, run B has `MUT`), the name is enough. If several runs have
it (both runs have `WT`), say which one as `run:sample`: `reference = "A:WT"`. A bare name
that is in several runs is an error that lists the choices.

**Figures.** Titles, legends and file names show a sample as `run:sample` only when the same
name is in several runs (`A:WT`, `B:WT`; the file `butterfly_B_WT.pdf`), and as the sample's
name alone otherwise. The tables never use that form: the run is a column there.

**Settings per sample, per run.** The tables keyed by sample (`[analysis.fasta]`,
`[analysis.labelling]`, `[analysis.fragments.exchange]`, `[analysis.fragments.labels]`) apply
to the sample in every run. For one run's own value, nest a table under the run's name:

```toml
[analysis.labelling.WT]      # WT in every run
ph_read = 7.0
temperature = 20.0
d2o_fraction = 0.9

[analysis.labelling.B.WT]    # WT of run B only (B is a run of [analysis.runs])
ph_read = 7.4
temperature = 20.0
d2o_fraction = 0.9
```

A nested table whose name is not one of your runs is refused, so a misspelt run name is
caught.

**Merging runs.** When the same sample was measured in two runs and you want them pooled
(replicates across days), give the two a common label with `[analysis.merge]`:

```toml
[analysis.merge]
"A:WT" = "WT"
"B:WT" = "WT"
```

The merged sample's rows have an empty `run` (they pool several runs; `runs` and
`n_spectra_by_run` in the species table say which and how many), and `run_offset` flags a
peptide whose runs disagree systematically. Fragment series (ladder kinetics) are never
pooled: a merged sample's runs are drawn side by side in its figures, each series named
`run:condition` where the same condition occurs in both runs, or by the run's own names from
`[analysis.fragments.labels.<run>]`.

## Protection factors from back-exchange kinetics

Protection factors (`[analysis] protection = true`) compare how fast each peptide exchanges
with how fast the same amides would exchange unprotected (their intrinsic rates). They work
for forward labelling (the protein goes into D2O and gains deuterium) and for back-exchange
kinetics: the protein is deuterated first, then put into a solvent with less deuterium, and
loses it over time.

You do not name the direction. It follows from the labelling conditions: for back-exchange,
add `initial_fraction`, the D fraction the amides started with, above the `d2o_fraction` of
the solvent the exchange happens in.

```toml
[analysis]
protection = true

[analysis.labelling]       # the solvent the back-exchange happens in
ph_read = 7.0
temperature = 20.0
d2o_fraction = 0.1         # 90 % D2O diluted ten-fold into H2O
initial_fraction = 0.9     # the amides began in 90 % D2O
```

Leave `initial_fraction` out for forward labelling. If the data go the other way from the
conditions (deuterium falls with time but no `initial_fraction` is set, or rises when it
is), hxex stops and says which setting to change.

**Where the exchange starts.** By default hxex fits each peptide's starting deuteration
(`fD0` in `protection.csv`) along with its protection factor, because the protein may not
have been fully deuterated when the exchange began. If the run has a measured fully
deuterated control (a run with `role = "fulldeut"`), each peptide is taken to start there
(`fD0 = 1`). Early points of a kinetics run are often unreliable: leave them out with
`[analysis] exclude_times`. hxex never drops them on its own.

**Without a fully deuterated control.** Back-exchange samples without one are fitted on the
absolute scale (deuterons per amide), relaxing to the solvent's equilibrium deuteration;
`protection.csv` says `corrected = false`. (Forward-labelling samples without controls get no
protection factors.)

**Reading the results.** `direction` says `back`, `equilibrium` is the deuteration each
peptide relaxes to, and `fD0` its fitted start with `fD0_se`. A peptide already at
equilibrium at the first time point gets only an upper limit (`bound = upper`); one that did
not exchange gets a lower limit. A fitted start is trustworthy when the loss of deuterium is
visible within your time points. When most of it happened before the first point, the start
is poorly known and its error is too small. Peptides with fewer than three time points get
no value when the start is fitted. Residue-level protection factors are not yet available for
back-exchange.

## Residue-level rates and protection factors from fragment ladders

With fragment ions (ETD, ECD and other low-scrambling activations), consecutive fragments of
a ladder differ by one residue, so the difference of their deuterium follows that residue
alone. `hxex analyse` fits an exponential decay (or rise) to every such single-residue step
and writes the rate to `ladder_rates.csv`. With `[analysis.fragments] kint = true` and the
kinetics' solvent in `[analysis.fragments.exchange]`, each step also gets the residue's
intrinsic rate and its protection factor with a standard error (`log10_protection`,
`log10_protection_se`).

**Which steps count.** A step is `precise` when its decay is visible in the time range and
its rate's relative error is at most `max_rate_error` (0.5 by default):

```toml
[analysis.fragments]
max_rate_error = 0.3      # stricter: rate_se / rate at most 30 %
```

**One value per residue.** `ladder_residues.csv` combines a residue's precise estimates from
the N-terminal ladder (c ions, say) and the C-terminal ladder (z ions), weighted by their
errors. It never combines different activation conditions (gentle and harsh), precursor
charges, fragment charges or runs, and never two ion types of the same ladder (c and c-1
come from the same peaks, so they are not independent): with both c and c-1 you get one row
for c with z and one for c-1 with z. `agreement` says whether the two ladders agree
(`agree`), disagree beyond their errors (`disagree`: look at both before using the mean), or
whether only one ladder measures the residue (`single`).

**Error ranges in the figures.** Every point of the ladder and closure figures carries a
light bar of plus and minus one standard error, when the points have one (they need a
bootstrap or replicates; a kinetics run fitted without a bootstrap has none). Every fitted
curve of the ladder figure has a shaded band of plus and minus one standard error of the
fit, whatever the points carry; with many points the band is narrow. In the residue-rates
figure, the rate, the amplitude and the protection factor of each step have error bars.
