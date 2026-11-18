# Experiment types

The same pipeline (read, pick, fit, report, analyse) serves several kinds of experiment. They
differ in what one spectrum is, which input formats can hold them, and which settings and
analyses apply. Fragment ions can be added to any of them that comes from raw scans.

| Experiment | One spectrum is | Formats | Analyses |
|---|---|---|---|
| [Bottom-up peptides](#bottom-up-peptides) | a peptide at one labelling time | all five | coverage, residues, populations, comparisons, peptide protection factors |
| [Top-down and intact proteins](#top-down-and-intact-proteins) | the whole protein (or a large fragment) at one time | `mzml`, `thermo` | as bottom-up, but no residue resolution from the protein alone |
| [Kinetics](#kinetics) | one scan, or a group of scans, of a run where exchange continues during acquisition | `mzml`, `thermo` | uptake or decay curves, protection factors, fragment ladders |
| [Fragments](#fragments) | a fragment ion of a peptide, in MS2 or MS3 scans | `mzml`, `thermo` | ladders, per-residue rates, closure |

Whatever the experiment, every spectrum is fitted with each number of binomial populations from
`k_min` to `k_max` (`[run.fit]`, default 1 to 3), and the number is chosen by BIC (or AIC, with
`criterion = "aic"`). Setting `k_min` and `k_max` to the same value fits that number only, with
no choice to make. A population can sit anywhere from 0 up to the spectrum's highest bin,
never beyond it: a population outside the bins would leave no trace in the spectrum. The controls are fitted over the same range, which shows up any problems
with them, and always at K = 1 as well: the back-exchange correction uses their one-population
fit, whatever `k_min` is. (The `screen` option needs `k_min = 1`.) When the data hold
undeuterated and fully deuterated controls, means are corrected for back-exchange from them;
otherwise, or with `controls = false` under `[run]`, they are reported in deuterons as measured
(see [Back-exchange correction](back-exchange-correction.md)).

---

## Bottom-up peptides

The usual HDX-MS experiment: a protein labelled for several times, quenched, digested and
measured by LC-MS, with undeuterated and fully deuterated controls.

**Input.** Spectra extracted in HDExaminer (`specexport`, or `pooled` for several proteins) or
HX-Express (`hxexpress`), or the LC-MS files themselves (`mzml`, `thermo`) with one
`[[input.runs]]` per file: `time` for a time point, `role` for a control. Give each peptide its
retention window (`rt_start`, `rt_end` in the peptide table) so that only its own elution is
summed. See [Input formats](inputs.md).

**Settings that matter most.**

- `[run.controls]`: the back-exchange correction. Without the table it runs by itself, with the
  default settings, when the data hold undeuterated and fully deuterated controls (`hxex check`
  and the run's log say whether it does); with the table it always runs; `controls = false`
  under `[run]` turns it off. `d2o_fraction` (default 0.9) is the labelling buffer's D
  fraction: set it when your buffer differs. Series whose controls are missing or out of
  limits fall back to the median retention of the run's accepted controls (or, with `quench`
  conditions, to a theoretical value) and are flagged (`control_fallback`).
- `[run.bootstrap]`: standard errors and intervals. The analysis' choice between one and two
  populations needs the fixed-K bootstraps, `ks = "all"` or `ks = [1, 2]`: without them nearly
  every residue passes the choice, and the analysis warns. `hxex info` says which were made: its
  `bootstrap` line names each fixed K and for how many spectra ("at K = 2 for every
  spectrum"), or "none at a fixed K".
- `[run.fit] screen = true` with `k1_adam_steps = 200`: about a quarter of the fitting time.
  The skipped K=2 and K=3 fits are then missing from fixed-K analyses.

**Analysis** (`hxex analyse`, the `[analysis]` table):

- Coverage, redundancy and resolution per residue; a residue map from the overlapping peptides
  (`fasta` gives the protein sequence; otherwise it is assembled from the peptides).
- Populations: where a K=2 description is trusted (`[analysis.filters]`) and separated enough
  (`[analysis.choice]`), a residue is described by two populations, otherwise by one. Three
  populations are shown but the choice is only between one and two.
- Comparisons with a `reference` sample: per peptide (Welch t-test together with a critical
  difference), per residue, per population, with Woods and butterfly plots. `peptide_match =
  "range"` pairs a mutant's peptide with the reference's over the same residues (the default,
  except for pooled data, which is compared by sequence).
- Regions and sites (`[[analysis.regions]]`, `[[analysis.sites]]`): regions are drawn, tabled,
  and tested; sites are only drawn (see [Regions and sites](#regions-and-sites)). *hxex* derives
  none itself.
- Several proteins, or several series, in one project: [Groups of samples](#groups-of-samples).
- Separate runs compared or pooled: see [Comparing separate runs](comparing-runs.md).
- Peptide protection factors (opt-in, `protection = true`, with the labelling conditions in
  `[analysis.labelling]`): one factor per peptide, from the uptake against the intrinsic rates.
  They need corrected data. Residue-level factors from overlapping peptides are deliberately
  not offered (they are unreliable); residue values come from fragment ladders.

**Practical notes.** `exclude_times` leaves unreliable time points out of the whole analysis
while keeping their fits. Peptides with a modification written in the sequence (such as `pS`
or `acK`) are states of their own (`"WT pS12"`) unless `modified_peptides = "combined"`. A
point mutation is not a modification: the mutant is a sample of its own, and `peptide_match`
decides whether its peptides are compared with the reference's over the same residues
(`"range"`) or only where the sequences are identical (`"sequence"`). The default, `"auto"`,
is `"range"`, except for pooled data, where it is `"sequence"`: there two sequences at one
range are different proteins.

### Regions and sites

Regions and sites are stretches and residues of the protein you name, so that the figures and
tables speak of them. *hxex* finds none by itself.

```toml
[[analysis.regions]]
name = "N-terminal region"
start = 1
end = 63

[[analysis.sites]]
residue = 19
label = "S19"
```

Repeat a table for every region or site; regions may overlap. Residues are numbered as the
protein's sequence numbers them (the FASTA, or the peptides' own numbers).

- **A region** is drawn as thin dashed lines at its boundaries on every by-residue figure (the
  residue overlays, the every-sample residue figure, the Woods plots, and redundancy and
  resolution): one line between two regions that follow each other, and none before residue
  1. It is also a unit of the analysis: `regions.csv` gives each region's fD per sample and
  time, the Regions figure (`regions_<time>`) draws them as bars, and with a reference
  `compare_regions.csv`, the region significance map, and the region composition maps say
  whether, and where in it, each sample differs from the reference (see
  [Outputs](outputs.md)).
- **A site** is only drawn: a dashed line at the residue, with its label above the Woods plots.
  It is not tabled or tested; to test a stretch around a site, name it as a region.

### Groups of samples

A project may hold samples that should not be compared with one reference: the proteins of a
heterooligomer, each measured in every vial it is in (exported once per protein per vial), or
a pH series of one protein next to a mutant's. Give each set of samples a group:

```toml
[analysis.groups.A]
reference = "A_alone"
samples = { alone = "A_alone", "A+B" = "A_in_AB", "A+C" = "A_in_AC" }

[analysis.groups.B]
reference = "B_alone"
samples = { alone = "B_alone", "A+B" = "B_in_AB", "B+C" = "B_in_BC" }

[[analysis.groups.B.regions]]
name = "N-terminal region"
start = 1
end = 64
```

- `samples` lists the group's samples (`["A_alone", "A_in_AB"]`), or names each with a label
  for the vial it was measured in (`"A+B" = "A_in_AB"`). A label names one vial as seen from
  every group that holds it: "A+B" in group A and in group B is the same vial, once with
  protein A measured and once with protein B.
- `reference` is the group's sample the others in the group are compared with.
- With groups, `[analysis] reference` and `samples` are refused: each group names its own.
  Every other `[analysis]` setting applies to every group.
- A group's own `[[analysis.groups.NAME.regions]]` replace the shared `[[analysis.regions]]`
  for that group, and its own `[[analysis.groups.NAME.sites]]` the shared sites, in the
  group's own residue numbers. A group that gives none of a kind takes the shared ones (right
  for a group of one protein); `regions = false` in a group gives it none. Own and shared are
  never merged.
- A sample may be in several groups; a sample in no group is not analysed (`hxex check` lists
  them).

**Peptides another protein of the vial also has.** In a vial that holds two of the groups'
proteins, a peptide whose sequence occurs in both proteins is measured as the sum of both
proteins' copies, so it is not like with like against the group's reference. *hxex* looks for
these in every vial two groups share: a label in both groups that names both, split at `+`
(`"A+B"` in groups A and B; `"alone"` names no group, so each group's sample labelled
`alone` is its own vial). A peptide counts as shared when its residues, modifications aside,
occur anywhere in the other protein's sequence, which comes from `[analysis.fasta]` or else
from the FASTA files the results were read with (SpecExport). Such peptides are always named in
the `shared_with` column of `species_uptake.csv` and `compare_peptides.csv`, and marked on the
sample's Woods and butterfly plots by a grey strip along the bottom of the panel under the
peptide ("shared peptide sequence" in the legend). With
`[analysis] shared_sequences = "keep"` (the default) they are compared like any other peptide;
with `"leave_out"` their spectra are left out of that sample's analysis (reason `sequence
shared within the vial` in `left_out_spectra.csv`), so the residues and comparisons agree.
`hxex check` lists the vials it found shared and the samples whose sequence it does not know
(their partners' peptides cannot be checked: give their FASTA in `[analysis.fasta]`).

`hxex analyse` then makes one analysis per group, each in its own folder named after the group
(`results_analysis_A`, `results_analysis_B`), with every table and figure of an ordinary
analysis of the group's samples against its reference. `hxex analyse hxex.toml --group B`
makes only group B's. `hxex check` shows each group: its reference, its samples by label, and
whether its regions and sites are its own or the shared ones.

**Across the groups.** When two groups or more have a region of the same name (each over its
own protein's residues: "ACD" may be residues 84-172 in one protein and 64-153 in another),
`hxex analyse` also writes a folder across the groups, `results_analysis_groups`, once every
group is analysed in the same command (not with `--group` for some):

- `cross_group_regions.csv`: one row per vial label, group, region, and time: the group's
  sample and reference, the region's residues in that group (`start`, `end`), `fD`, `fD_se`,
  `n_residues`, `coverage`, and the group's own test against its reference (`delta`,
  `delta_se`, `significant`; empty for the reference itself). A region a group does not have
  has no row.
- `cross_group_regions_<time>`: grouped bars of each region's fD, the region names along the
  bottom, the bars ordered by group (each group's reference first), one colour family per
  group with its reference darkest. The legend names each bar by its vial label with the
  measured protein underlined ("A+B" with A underlined is protein A in that vial; a sample
  labelled `alone` reads as its group's name); with matplotlib older than 3.9 it is in bold
  instead. Many regions go on further rows of panels.

A group cannot be named `groups` (the folder's name).

---

## Top-down and intact proteins

The whole protein, measured intact at high charge states (often with fragments, see
[Fragments](#fragments)).

**Input.** A scan format, with the protein as one row of the peptide table (`start_seq` 1,
`end_seq` the last residue, `peptide` the full sequence in modX, `charge` the charge state; one
row per charge state). In direct infusion there is usually no retention window; summing groups
of scans (`[input.scans] group = "every"`, `every = 20`) gives spectra with enough signal.

**The picking window.** At high charge the isotope peaks are close together, and a fixed ppm
window can be wider than half their spacing. *hxex* caps every window at half the spacing,
which matters only here: on a top-down example at charge 20, the uncapped window had zeroed
more than a quarter of the envelope's bins and moved its centre by about 9 bins. The capped
ppm window is all you need. See [The picking window](picking-window.md).

**What to expect.** One intact species gives its deuterium distribution (and populations) as a
whole: there is no residue resolution from the protein alone. Residue detail comes from its
fragments.

---

## Kinetics

Runs in which exchange continues while the instrument acquires, so that the scan time is the
exchange time: continuous-flow or infusion labelling, and back-exchange of a deuterated sample
diluted into H2O.

**Input.** A scan format, with `[[input.runs]]` that have neither `time` nor `role`: every scan
(or every group of scans) becomes a time point. The controls are separate files with a `role`.

```toml
[[input.runs]]
path = "kinetics_01.mzML"
sample = "WT"
time_offset = 12        # s: exchange began this long before the first scan
usable_from = 20        # s of scan time: leave out the scans before this

[input.scans]
group = "each"          # or "every" (every = 5) or "windows"
max_spectra = 60        # at most 60 spectra per peptide, spread over log time
```

**Two clocks.** The *acquisition clock* is the scan time the file records. The *exchange clock*
is the scan time plus `time_offset`, and it is the time of every spectrum. `usable_from` (per
file) and `[input.scans] min_time` (every file) are on the acquisition clock and leave scans
out; only `time_offset` moves the time axis. `[analysis.fragments] min_time` is on the exchange
clock: earlier points are drawn but not fitted.

**Noise.** Each kinetics series takes its noise reference from its own scans (the scan whose
tallest peak is the median of the series), not from a separate control file, which would make
the kinetics spectra look almost noiseless and favour two populations.

**Direction.** Forward labelling (the deuterium rises) and back-exchange (it falls) use the same
model and the same settings; nothing names the direction. A back-exchange is described by the
solvent conditions: `initial_fraction` (the D fraction the amides began with) above
`d2o_fraction` (the solvent's). If the data go the other way from the conditions given, *hxex*
refuses and says which setting to check. See
[Protection factors](protection-factors.md).

**Controls.** In a back-exchange the equilibrium end point is undeuterated-like, not a fully
deuterated control. `[run.controls] undeut_at` or `fulldeut_at` make the spectra at one time
serve as a control as well; `undeut_marked = false` or `fulldeut_marked = false` stop a marked
control file being used when it is known to be poor.

**Practical notes.** A kinetics run can produce hundreds of time points. Use `max_spectra`, or
`[analysis] times` to limit the per-time figures of the analysis. With `time_scale = "order"`
the kinetic and protection analyses refuse, because the times are not real.

---

## Fragments

Fragment ions of a peptide (or of an intact protein) in MS2 or MS3 scans: c and z ions from ETD
or ECD, a and x ions from UVPD, or others. A fragment is one more series, tied to its parent,
and is fitted like any other. Fragments can be added to a bottom-up, top-down or kinetics
experiment, as long as the input is a scan format.

**Input.** One `[[input.fragments]]` table per condition (see
[Input formats](inputs.md#fragment-ions)):

```toml
[[input.fragments]]
ions = "c:ALL z-dot:ALL"
label = "gentle"
activation = "ETD"
filter_excludes = "sid"

[[input.fragments]]
ions = "c:ALL z-dot:ALL"
label = "harsh"
activation = "ETD"
filter_contains = "sid"
```

Only low-scrambling activations (ETD, ECD, UVPD) give site-specific deuteration. CID and HCD b
and y ions are scrambled and are not suited to residue values.

**The envelope.** ETD spectra of one fragment can hold several ion types a hydrogen apart (c-1,
c, c+1). `[run.envelope] mode = "mixed"` fits their mixture to the undeuterated references;
`share = true` lets conditions that differ only in their scan filters or label (in-source
energy) share one envelope, after checking that they agree. Conditions whose undeuterated
spectra are the same measurement (one undeuterated file read for each, as when the in-source
energy was varied only in the labelled runs) count as one condition in that check: they cannot
confirm each other. A fragment whose envelope could not be measured well is flagged
`poor_envelope`: its deuteration can be off by up to a hydrogen.

**Amide counts.** A fragment's exchangeable amides are counted by its position in the parent.
Whether the amide at the cleavage site counts depends on the ion type (c, x and y keep it; a, b
and z do not); `[run.picker.amide_rules]` changes this per ion type.

**Analysis** (`[analysis.fragments]`, written whenever a run has fragments):

- `fragment_uptake.csv`: the K=1 deuteration of every fragment and of the precursor read with
  them.
- Flagged spectra left out: fragment spectra flagged `below_noise`, `poor_envelope`,
  `mz_shift`, or `overlapping_ion` are left out of every fragment output, and listed with their
  flags in `fragment_left_out.csv`. On a targeted dataset, and on infusion kinetics of three of
  four peptides (the fourth stayed poor whatever was left out), leaving them out made the c- and
  z-ladders agree, made the closures meet the precursor, and removed nearly all ladder steps
  outside what one amide can hold; a stricter signal threshold alone did not. On noisy data it
  costs a third to two thirds of the fragment spectra. The precursor's own ion is kept, and so is a spectrum you mark `accepted` in the
  review. `leave_out_flags` in `[analysis.fragments]` changes the list; `leave_out_flags = []`
  keeps every flagged spectrum.
- Ladders: consecutive fragments of one ion type differ by one residue (ladder figures). With
  `fit_rates = true` (opt-in, as the fits take minutes on a large run) their difference over
  time is fitted with `A exp(-kt) + C`, giving one rate per residue (`ladder_rates.csv`, and a
  fitted curve with a ±1 standard-error band on the ladder figures).
- Closure: an N-terminal fragment, plus the C-terminal fragment that completes it, minus the
  precursor should be zero; misses beyond `tolerance` are flagged (`closure.csv`). The
  precursor is the parent's MS1 spectrum in the same run, where the project reads the parents
  (`parents = true`); otherwise the precursor ion read in the fragment scans. The MS1 spectrum
  is the safer reference: the isolation window can let another species onto the precursor's
  m/z in the fragment scans (on a targeted dataset, 2+ precursors read 0.6 to 10 D above their
  MS1 spectra, while 3+ and 4+ ones agreed within 0.1 D). `closure.csv` says which was used
  (`reference`: `MS1` or `M`), and so does the closure figure's axis. A precursor ion that
  reads more than 1 D from its MS1 spectrum is flagged `precursor_off`.
- Overlapping ions: one fragment scan holds dozens of ions of the same peptide, and some fall
  almost on top of each other: a c ion and a z ion that lack different end residues can differ
  by close to a whole number of isotope steps (on a targeted dataset, c 13-29 and z 14-30 of
  one peptide lay 52 parts per million apart where the instrument's peaks were 64 wide), two
  charges of one ion share every other position, and a bright 3+ or 4+ precursor puts a peak
  every third or quarter m/z. *hxex* measures, for every fragment spectrum, how much of its
  picked intensity another ion read from the same scans puts there (from that ion's own fit,
  scaled by its peaks away from the fragment's strong peaks), and flags it `overlapping_ion` at
  20 % or more (`[run.quality] max_ion_overlap`); `spectra.csv` names the other ion
  (`overlapping_ion_with`). This includes another ion that lies further up the fragment's bins,
  where the fragment could move once labelled, and a fragment whose fit has settled on another
  ion's envelope instead of its own. On the dataset above, 12 % of fragment spectra were
  flagged (10 % of the undeuterated ones, 12 % of the labelled ones). An ion that falls on all
  of the fragment's strong peaks cannot be told apart from it and is not counted, nor are ions
  that are not read (a, b and y ions). Where two ions' peaks merge, *hxex* does not try to
  split the merged peak between them: on that dataset no way of doing so (the height at each
  ion's position, or fitting a peak at each position) read the true height better than the
  picked value.
- Per-residue rates from both ladders together (`ladder_residues.csv`, with `fit_rates =
  true`): an N-terminal and a C-terminal ion type are paired, and where they disagree the
  residue is marked. Conditions, precursor charges, fragment charges and runs are never pooled,
  nor are two ion types of one terminus (c and c-1 are the same peaks).
- With `kint = true` and the kinetics' solvent in `[analysis.fragments.exchange]`: theoretical
  curves on the ladder figures and, with `fit_rates = true`, intrinsic rates and protection
  factors per residue. See
  [Residue-level rates and protection factors from fragment ladders](fragment-residues.md).

Fragment series are kept out of the peptide analysis (coverage, residue maps); MS1 parents read
alongside them stay in it. Fragment ions are not corrected for back-exchange, even when the
correction runs: their values stay in deuterons, as measured. How to
correct them, probably through their parent peptide, is still to be designed.
