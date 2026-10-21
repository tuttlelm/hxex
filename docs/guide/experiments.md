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
no choice to make. The controls are fitted over the same range, which shows up any problems
with them, and always at K = 1 as well: the back-exchange correction uses their one-population
fit, whatever `k_min` is. (The `screen` option needs `k_min = 1`.) A run without `[run.controls]`
reports deuterons as measured; with it, means are corrected for back-exchange from the
undeuterated and fully deuterated controls (see [Controls and times](controls.md)).

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

- `[run.controls]`: turns on the back-exchange correction. `d2o_fraction` (default 0.9) is the
  labelling buffer's D fraction. Series whose controls are missing or out of limits fall back
  to the median retention of the run's accepted controls (or, with `quench` conditions, to a
  theoretical value) and are flagged (`control_fallback`).
- `[run.bootstrap]`: standard errors and intervals. The analysis' choice between one and two
  populations needs the fixed-K bootstraps, `ks = "all"` or `ks = [1, 2]`: without them nearly
  every residue passes the choice, and the analysis warns.
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
  "range"` pairs a mutant's peptide with the reference's over the same residues.
- Regions and sites (`[[analysis.regions]]`, `[[analysis.sites]]`) are drawn and summarised;
  *hxex* derives none itself.
- Separate runs compared or pooled: see [Comparing separate runs](comparing-runs.md).
- Peptide protection factors (opt-in, `protection = true`, with the labelling conditions in
  `[analysis.labelling]`): one factor per peptide, from the uptake against the intrinsic rates.
  They need corrected data. Residue-level factors from overlapping peptides are deliberately
  not offered (they are unreliable); residue values come from fragment ladders.

**Practical notes.** `exclude_times` leaves unreliable time points out of the whole analysis
while keeping their fits. Peptides with a modification written in the sequence (such as `pS`
or `acK`) are states of their own (`"WT pS59"`) unless `modified_peptides = "combined"`. A
point mutation is not a modification: the mutant is a sample of its own, and `peptide_match`
decides whether its peptides are compared with the reference's over the same residues
(`"range"`, the default) or only where the sequences are identical (`"sequence"`).

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
more than a quarter of the envelope's bins and moved its centre by about 9 bins. Better still, size the window from the instrument's resolving power:

```toml
[run.picker]
resolving_power = "recorded"     # Thermo raw files: the value each scan records
# resolving_power = 120000       # or a value quoted at resolving_power_mz (200 for Orbitraps)
```

See [The picking window](picking-window.md). mzML files do not carry the resolving power, so give
a number there.

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
[Protection factors from back-exchange kinetics](back-exchange-protection.md).

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
energy) share one envelope, after checking that they agree. A fragment whose envelope could not
be measured well is flagged `poor_envelope`: its deuteration can be off by up to a hydrogen.

**Amide counts.** A fragment's exchangeable amides are counted by its position in the parent.
Whether the amide at the cleavage site counts depends on the ion type (c, x and y keep it; a, b
and z do not); `[run.picker.amide_rules]` changes this per ion type.

**Analysis** (`[analysis.fragments]`, written whenever a run has fragments):

- `fragment_uptake.csv`: the K=1 deuteration of every fragment and of the precursor read with
  them.
- Ladders: consecutive fragments of one ion type differ by one residue. Their difference over
  time is fitted with `A exp(-kt) + C`, giving one rate per residue (`ladder_rates.csv`, ladder
  figures with a ±1 standard-error band).
- Closure: an N-terminal fragment, plus the C-terminal fragment that completes it, minus the
  precursor should be zero; misses beyond `tolerance` are flagged (`closure.csv`).
- Per-residue rates from both ladders together (`ladder_residues.csv`): an N-terminal and a
  C-terminal ion type are paired, and where they disagree the residue is marked. Conditions,
  precursor charges, fragment charges and runs are never pooled, nor are two ion types of one
  terminus (c and c-1 are the same peaks).
- With `kint = true` and the kinetics' solvent in `[analysis.fragments.exchange]`: intrinsic
  rates, protection factors per residue and theoretical curves. See
  [Residue-level rates and protection factors from fragment ladders](fragment-residues.md).

Fragment series are kept out of the peptide analysis (coverage, residue maps); MS1 parents read
alongside them stay in it. Fragment back-exchange is corrected from the fragments' own controls
or the fallback (the median retention of the run's accepted controls).
