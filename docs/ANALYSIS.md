# Post-fitting analysis: design

Analyses that start from fitted results and work on the protein rather than on one spectrum:
coverage, uptake per residue and per population, the choice between one and two populations,
comparisons between samples and between runs, region summaries, kinetics and export. This file
is the design; `docs/PLAN.md` ("Next steps") keeps the status and points here, and dated findings
and measurements go in `docs/HISTORY.md` as usual. Read this file before working on these
analyses.

Status (2026-10-13): phases 1 to 5 built: the species uptake table
(`hxex.analysis.species_uptake`), residue-level values (`residue_map`), coverage tables
(`coverage_table`), the coverage map and the redundancy/resolution plot
(`hxex.report.coverage`), the filters of K >= 2 descriptions (`SpeciesFilters`), residue
profiles (`hxex.report.residues`), the one-or-two-population choice per residue
(`choose_populations`, `plot_population_choice`; phase 2), and comparisons within a run
(`compare_peptides`, `compare_residues`, `region_summary`, `Annotations`; the residue overlay,
Woods and region bar plots in `hxex.report.compare`; phase 3), and the `[analysis]` section of
the project file (`AnalysisConfig`) with `hxex analyse`, which writes every table and figure
(`hxex.report.write_analysis`), and comparisons between separate runs (`RunSet`,
`compare_settings`, `check_identity`, `compare_choices`, the butterfly plot; phase 4), and
recomputing and combining (`analysis_controls`, `[analysis.controls]`, `RunSet(merge=...)`,
`[analysis.merge]`; phase 5). Next: the fragment branch (phase 6), with the owner's references.

## Ground rules

Decided with the owner (2026-10-12 and 2026-10-13):

- The tools are rewritten to fit hxex: its result types, NumPy arrays and typed indexes rather
  than pandas in the core, explicit serialisable configs, matplotlib plots that return figures.
  Neither pyHDX nor proplot becomes a dependency; what pyHXExpress does through pyHDX
  (residue-level averaging, the uptake-table conversion, the coverage plot) is rewritten.
- pyHXExpress's `hxex_utils.py` is the reference for what the peptide analyses do (see
  CLAUDE.local.md for its location); its bugs are not copied (see "Reference notes").
- Rejected spectra (review layer) are always left out and the reviewer's K overrides are
  honoured (`review_K = 1`: the spectrum's K >= 2 descriptions leave the species table and the
  set evidence, as the screens' one-population action does; audit 2026-10-28); uptake is back-exchange corrected by default (`hxex.controls`); populations are
  identified by rank, 1 being the lowest deuteration (see "Nomenclature" below).
- The controls (undeuterated and fully deuterated) take no part in any per-residue choice or
  comparison: their rows are dropped before the population choice, the residue and population
  comparisons and the region summary (`ResidueMap.time_points`). Their fits stay in the species
  and residue tables (audit 2026-10-28 Medium 10: before, a control's NaN time split the
  choice's and the regions' rows).
- Analyses run on one results bundle, or on several for the run comparisons, and are configured
  by an `[analysis]` section of the project file and a command-line command.
- Fragment data (MS2, MS3) is analysed differently from peptide data (MS1) and gets its own
  branch. Populations shared by a parent's fragments are a post-processing step, not a
  joint fit in the engine (owner, 2026-10-13).

## Structure

### Shared foundation: the species uptake table

One table for every kind of measured species (peptide, intact protein, fragment ion). One row
per sample, species, time and population rank, under a given description (the selected or
effective K, or a fixed K from the fixed-K bootstraps):

- identity: the sequence with modifications, kind (peptide, intact, fragment), ion type and
  fragment label, parent (for fragments: the ion isolated from the full scan, also for MS3
  scans that heat the precursor and fragment it again; only an MS3 fragment of an MS2 fragment,
  not built, would need the chain of parents), MS level, pre-activation (HCD before the last
  stage), activation, precursor charge;
- position on the protein: first and last residue;
- values: the absolute deuteration `mean` (deuterons) and the fractional deuteration `fD`
  (`mean / n_amides`), each with its standard error, the `population` (share) and its standard
  error, the number of replicates and charge states combined, flags.

Values mean the same for forward labelling (H to D) and reverse exchange (D to H, as in
back-exchange kinetics), so no setting says which (owner, 2026-10-13). Components are named by K
and rank (1 = lowest deuteration, the fit's own ranking), not slow and fast: which of them is the
protected one (the low one in forward labelling, the high one in reverse exchange) is for the
reader.

**The set's number of populations (`best_set_K`; owner, 2026-10-20; HISTORY item 125).** The run
chooses K spectrum by spectrum, so the replicates and charge states of one species at one time
can disagree, and the table above reports every fixed K without choosing. `best_set_K` is one
choice per species and time from the whole set: replicates and charge states are spectra of the
same molecules, so the evidence adds up, and the fits' criterion (BIC or AIC, `[run.fit]
criterion`; the raw criterion, not the run's selection score, which the run's population rule
sets to infinity and which would let one spectrum veto the set) is summed per K over the set's
spectra (those the tables use, fitted at every K analysed); the smallest sum wins. A K is
passed over when the table has no description at it (the filters left it out) or one of its
combined populations lies outside `[min_population, 1 - min_population]` (`[analysis.choice]`),
and the next best K is taken. Columns: `best_set_K`, `best_set` (the rows of the chosen K),
`best_set_reason`, `set_delta_criterion` (the summed criterion at K = 2 minus K = 1; negative
favours two). Per time point, never one K for the whole time course: populations appear and
merge with time (EX1, conformers that separate and converge). It needs no bootstrap. First
version: only these columns; whether the peptide comparisons (Woods, `compare_peptides.csv`,
K = 1 now) should use it is open. The per-residue choice (`populations.csv`) is separate and
can disagree (a bimodal peptide whose residues other peptides cover).

### Nomenclature

Owner, 2026-10-13, following published precedent; used throughout hxex (fit results, bootstrap,
tables, bundle, plots and these analyses):

| Quantity | hxex name | Plot label | Reference names |
|---|---|---|---|
| number of components fitted | `K` | K | ncurves, fixed1pop / fixed2pops |
| component order | `rank` (1 = lowest deuteration) | rank | pop1 / pop2 (slow / fast) |
| a component's share of the molecules | `population` (and `population_se`) | population | pop, p_slow, p_fast, frac |
| absolute deuteration (deuterons beyond natural abundance, 0 to about `n_amides`) | `mean` (`n * p`); back-exchange corrected when there are controls | deuterium | Dabs, centroid, D content |
| fractional deuteration (per exchangeable amide, 0 to 1), of a component or of any grouping (peptide, region, residue) | `fD` (`mean / n_amides`) | fD (fractional deuteration) | fD, rfu, D-uptake, %D / 100 |
| exchangeable amides | `n_amides` | exchangeable amides | max_namides, Nex (as a count) |
| binomial parameters of a component | `n`, `p` | n, p | Nex, mu |

"Population" names both a component ("two populations") and its share ("population 0.62"), as
in the published work; where it matters, a component is named by its rank.

### Peptide branch (MS1; first)

- **Residue level.** Per sample, time and rank: per-residue values from the overlapping peptides
  that cover each residue, with redundancy (peptides covering a residue), resolution (the
  length of the shortest block) and masks (prolines, the excluded N-terminal residues,
  uncovered residues). The pyHDX method, for reference: each peptide weighs
  `1 / (its exchanging residues)` (raised to a weight exponent, 1 by default), normalised per
  residue, and standard deviations add in quadrature with the same weights.
  As built: the same values as pyHDX 0.4.3 to machine precision, and the same redundancy and
  resolution (HISTORY item 48). One deliberate difference: pyHDX decides once for the whole
  protein whether a residue exchanges, so a residue that is among the first `nterm_subtract`
  residues of any peptide gets no value at all; hxex decides per peptide, so such a residue
  still gets a value from the peptides in which its amide counts (33 more residues of 173 on
  the intact-peptide benchmark, otherwise identical).
- **Coverage, redundancy and resolution** are judged on the peptides of the undeuterated control
  by default, or of time points the user chooses (owner, 2026-10-13); redundancy counts
  peptides, or spectra (replicates and charge states) on request, as the owner's earlier
  figures did. The coverage map draws one bar per peptide (charge states and replicates
  combined) in the owner's style (the CET-L20 "gouldian" colour scale pyHDX uses).
- **Why fixed K** (owner, 2026-10-13): the best-K fit judges each spectrum on its own (K=1 only
  says that spectrum does not justify more); since many spectra report on each residue, the
  fixed K=1 and K=2 fits are aggregated per rank, and whether the data support more than one
  population is judged per residue (or region). The K=1 fit stands for the average, which holds
  while the populations are not far apart.
- **Filters of K >= 2 descriptions** (`SpeciesFilters`; the owner's pyHXExpress practice,
  2026-10-13): per spectrum before combining, every population within 5 to 95 % and no
  component above fD 1.1 (`n * p` over the amides, as reported: beyond full deuteration is not
  physical; the fit's own bound on `n` stays the run setting `n_max_factor`); per peptide and
  time after combining, every rank's fD error below 0.25 and the K=1 value between the lowest
  and highest rank (`k1_between_ranks`, owner 2026-10-13: the same condition the population choice applies
  per residue, here also per peptide). K=1 values are not filtered, but whole spectra are left
  out first (owner, 2026-10-22, HISTORY 140; listed with the reason in `left_out_spectra.csv`,
  a spectrum accepted in the review kept): a replicate more than `replicate_outlier` (6)
  typical replicate standard deviations from the median of its series' replicates at that
  time (three or more; the typical sd is the run's own, robust), a K=1 mean above the amide
  count before correction (`above_amides`), and, with `stand_in_controls = "exclude"` (the
  default), every series whose fully deuterated control is a stand-in (median, kint or
  `d2o_fraction`; kept in a sample with no measured control at all). Before those, the
  spectrum quality screens (`spectrum_quality_filtering`, loose by default; HISTORY 149-152,
  `hxex.analysis.screening`) leave out spectra showing another species (a control also leaves
  the back-exchange correction, recomputed without it; a series whose measured fully
  deuterated controls all go is a `stand-in control (screened)`), keep spectra with a separate
  cluster between the isotope positions at K=1 only (their K >= 2 records dropped), and warn
  about intensity beyond the reachable range; fragment series are not screened, and screened
  spectra do not count in the replicate-outlier scatter. A fixed fD range (the
  owner's earlier practice, 0.1 to 1.1) was tried and not adopted: on the intact-peptide
  benchmark most values it removed agreed with their replicates. On the
  intact-peptide benchmark they bring the K=2 residue values in line with the owner's
  pyHXExpress analysis (HISTORY item 49). The K=2 split depends on the noise model: the Poisson
  loss of the replicate noise estimate puts a small component in the high-mass tail, least
  squares (the default, and pyHXExpress) a moderate split. Decided (owner, 2026-10-13): the
  Gaussian loss (HISTORY items 55 and 56).
- **Combining** replicates and charge states, in two steps (owner, 2026-10-20; HISTORY 128):
  the charge states of each replicate are averaged first (error `sqrt(sum se_i^2) / c`), since
  they share the replicate's sample, labelling, quench, and injection and are not independent
  replicates; the replicates are then combined: the mean, with the standard error
  `sqrt(max(mean(se_i^2), s^2) / n)` over the `n` replicates (the larger of the bootstrap
  variance and the observed scatter, so neither counts twice). The same per-replicate values
  feed the comparisons' Welch tests and pooled scatter. **Charge trend** flag: the lowest and highest charge state
  differ by more than 2 combined standard errors in the same direction at more than half of
  the times both were measured, and at two times at least.
- **One or two populations** (pyHXExpress `choose_fits`): per residue (or peptide) and time,
  the two-population description is kept when the populations are separated at a stated
  confidence (`(high - low) - z (se_low + se_high) >= 0`), both standard errors are below a
  limit (0.25), the K=1 value lies between them (`fD_low < fD_avg < fD_high`, with `fD_avg`
  from the K=1 fits; owner, 2026-10-13), and the rank-1 population from the lever rule,
  `(fD_high - fD_avg) / (fD_high - fD_low)`, is inside `[min_pop, 1 - min_pop]`; otherwise the
  K=1 value is used. Output: fD at K=2 rank 1 and rank 2 and at K=1, and the K=2 rank-1
  population, with standard errors (the error of the lever-rule population propagated
  correctly, unlike `choose_fits`). Built on the fixed-K values, the filters above and the review
  layer. As built (`ChoiceConfig`: confidence 0.5, error limit 0.25, population 5 to 95 %; with
  `populations = "bootstrap"` the defaults are 0.95 and 10 to 90 %, since the tests are then the
  rule itself, owner 2026-10-22, HISTORY 142), the
  reason a residue keeps K=1 is recorded (no K=2, not separated, error, K=1 outside,
  population). The tests need bootstrap standard errors at K=1 and K=2
  (`BootstrapConfig.ks`): without them only the scatter of replicates is left, which is far too
  small, and nearly every residue passes (`species_uptake` warns).
- **Comparisons within a run.** Differences between samples (a reference sample against each
  other) per peptide and per residue, with propagated standard errors and a significance rule;
  averages over user-defined regions of the sequence (pyHXExpress `group_fits`). As built:
  differences in fD at a fixed K and rank (K=1 by default); per peptide and time the hybrid rule
  (Hageman and Weis 2019; owner, 2026-10-13): Welch's t-test on the replicate spectra with
  `p < 0.05` and `|delta|` above `delta_crit = t(1 - alpha/2, 2n - 2) s sqrt(2/n)` (`s` the
  typical standard deviation of replicate spectra over every peptide, time and sample: since
  HISTORY 140 the median of the chi-square-scaled replicate variances, not their plain pool,
  which a few replicates reading a different envelope dominated (0.136 against 0.027 fD on the
  intact-peptide benchmark; the typical scatter is the same at every exposure time there, so
  one threshold serves all times); `n` the typical replicate count; replicates, their charge
  states averaged); per residue,
  `|delta| / se` above the normal quantile and `|delta|` above the peptides' `delta_crit`. A region's value is the mean of its residues and its error
  the mean of their errors (residues share peptides, so the error does not shrink with the
  region's length). Regions may overlap. Populations are compared too (owner, 2026-10-13), to
  see whether a sample shifts the balance between the populations rather than their
  deuteration, by the lever rule at both levels (owner, 2026-10-20; HISTORY 128): the low-D
  population is `(fD_high - fD_avg) / (fD_high - fD_low)`, the K=1 value standing for the
  average (it is far more precise than the K=2 fit's own share, whose scatter comes from the
  positions trading off against it). Per peptide (`compare_peptides(..., quantity='population')`)
  each replicate's K=1 value sits between the sample's K=2 positions averaged over its
  replicates; each sample keeps its own positions, so the K=2 fD comparisons say whether the
  states differ and the population comparison whether they are populated differently. A pair
  is compared only where, on both sides, the positions pass the choice's separation and error
  tests, every replicate's K=1 value lies between them, and the population is within the
  choice's limits; otherwise `reason` names the first failing test and side. The test is
  `|delta| / delta_se` against the normal distribution, `delta_se` from each side's whole
  error (the replicates and the K=2 positions; HISTORY 144: a Welch test on the replicates'
  populations, the positions held fixed, overstated it), and the threshold the K=1 `delta_crit` divided by the
  separation (the mean of both sides'): with fixed positions a population difference is the
  fD difference over the separation, so the K=1 standard carries over, where a pooled threshold
  was dominated by poorly separated peptides (0.49 on the tutorial data). Per residue
  (`compare_populations`) the same threshold with the residue's separation. The tables'
  `delta_crit` column is each row's threshold, and `compare_peptides` has the `separation`.
  The Woods plot draws a population scaled by its separation (`delta` and `delta_se` times
  `separation`, in fD units: the deuterium that moved between the states) against one dashed
  line, the K=1 `delta_crit` (owner, 2026-10-21, HISTORY 135;
  `PeptideComparison.scaled_by_separation`): the same test, one critical line, and the panel
  shares the fD panels' y range. The per-residue population figure is unchanged.
  **K=2 comparisons only where both sides have two populations** (owner, 2026-10-21, HISTORY
  137): a sample's K=2 fit of a peptide whose set is best described by one population
  (`best_set_K = 1`) is a small tail, not a second state, and comparing it with the
  reference's second state compared different things (the tutorial's D109H 113-134, whose
  "high state" shift was such a tail; a sign reversal in 24-34). Per peptide, the rank and
  population comparisons are made only where both sides have `best_set_K = 2`; elsewhere the
  difference is NaN and `reason` says `one population (sample)`, `(reference)` or `(both)`.
  Per residue (`residue_map(two_population_peptides=True)`) the K=2 values come from every
  covering peptide, as K=1 does (owner, 2026-10-22, HISTORY 141: from the two-population
  peptides alone they formed curves that did not bracket K=1), every covering peptide votes
  with the residue's weights, the residue keeps two populations only where the two-population peptides
  carry at least half of the weight (`two_population_weight`; reason
  `one-population peptides`; "any peptide" kept almost every residue, "all" almost none), and
  the residue comparisons count a K=2 difference as significant only where both sides keep
  two. Where the sides differ in the number of populations is a finding of its own:
  `compare_population_counts` (each side's `best_set_K` and summed-criterion evidence;
  `borderline` unless the two-population side's evidence is at least 10 below 0 and the other
  side's above 0), drawn on the first page of the Woods plot ("Two Populations Selected": the
  sample only, the reference only, both). The Woods plot is two figures, `woods_<sample>_<time>`
  (K=1 and the number of populations) and `woods_K2_<sample>_<time>` (the gated K=2 panels).
  On the tutorial data: rank comparisons 363 -> 92 (significant 23 -> 3, 20 -> 4), population
  245 -> 85 (120 -> 36). Since HISTORY 144 the K=2 ranks' `delta_crit` pools only the sets
  best described by two populations, and the population test includes the positions' error.
  The comparison tables carry
  their `quantity` (`value`, `value_reference` and their errors).
  Peptides are paired by residue range by default (owner, 2026-10-13: a peptide containing a
  mutation is compared with the reference's over the same residues; `same_sequence`,
  `peptide_reference` say so), or by identical sequence (`match='sequence'`, `[analysis]
  peptide_match`). Where a sample has several species over one range (a modified and an
  unmodified form), only identical sequences are paired. Modified residues are written in the
  modX sequence, so a modified peptide is a species of its own; at the residue level each
  modified form is by default a state of its own (`'WT pS59'`, `residue_map(modified=
  'separate')`, `[analysis] modified_peptides`), or is averaged with the sample's other peptides
  (`'combined'`; owner 2026-10-13: context dependent). Values corrected differently (one side
  with a fallback control) carry `control_fallback` flags and are marked `*` at the top of the
  comparison plots.

### Fragment branch (MS2, MS3; after the peptide branch)

How fragment data differs, and what this branch does about it:

- **Ladders, not overlaps.** N-terminal ions (a, b, c) grow from the parent's N-terminus and
  C-terminal ions (x, y, z) from its C-terminus; consecutive fragments differ by one residue, and
  `c_n + z_(N-n)` should equal the parent (the closure test, HISTORY item 41). Residue values come
  from differences, estimated as a least-squares fit of per-residue deuteration (between 0 and 1)
  to all fragments of a parent at once, with the closure as a check. Built (2026-10-14,
  `hxex.analysis.fragments`, HISTORY item 67): the K = 1 fragment table, ladder steps,
  scan-matched differences, decay fits, the bounded least squares and the closure test
  (`closure_table`: c + z, a + x, b + y against the precursor), the per-step rate table
  (`ladder_rates`), the closure and per-residue rate figures (no owner reference: designed
  in hxex, owner 2026-10-14), all written by `hxex analyse` for runs with fragment series
  (`[analysis.fragments]`; those series, the precursor M read with them included, are kept out
  of the peptide analysis by series, so a peptide also read from MS1 stays in it; the fully
  deuterated controls are not points of the kinetics, HISTORY 91), with the ladder kinetics
  figure (`hxex.report.ladder_kinetics_figure`). Owner (2026-10-14): these
  simple ladder values are wanted as a comparison for the owner's own method of deriving
  single-residue exchange rates and scrambling, which is unpublished and kept outside the
  repository until it is; hxex supplies its inputs (fragment and precursor deuteration per
  activation setting) but does not implement it.
- **Populations belong to the parent.** A fragment's populations are its parent's molecules, so
  the population is shared by all fragments of a parent; a fragment that does not
  span the region where the populations differ looks like one population. How to handle this is
  not yet known (owner): start with K=1 fits; once exchange is resolved to single amides,
  decide whether a single or a double exponential is needed.
- **Scrambling.** Only low-scrambling activation (ETD, ECD, UVPD) gives site-specific values;
  the activation is carried. The scrambling measure belongs to the owner's method above.
- **Amide counts per ion type** follow the structure (PLAN, "Risks and open decisions"; HISTORY
  items 65 and 66; neutral-loss ions and `z+1` unverified): per-residue values come from
  `hxex.fragments.amide_difference`, which names the residues a difference of two fragments
  measures (c_k minus c_(k-1) is residue k + 1; z_j minus z_(j-1) the residue after z_j's
  first) and refuses differences that are not a set of residues.
- **Controls** are per fragment and per fragmentation type (an undeuterated run per type and a
  final equilibrium run). Fragment series are not corrected (owner, 2026-10-21: flagged
  `uncorrected` in the control table, values as measured, out of the run's median and
  prediction; the correction may belong at the parent level, HISTORY 131). Back-exchange may
  not be fully correctable (in back-exchange
  kinetics the final run is back-exchanged to about 0 D), so theoretical references may be
  needed instead (owner); some kinetics runs have spurious early time points from the setup.
  Theoretical references and protection factors come from intrinsic rates in the kinetics'
  solvent (`hxex.kint`; opt-in, `[analysis.fragments] kint = true`, with the samples'
  conditions in `[analysis.fragments.exchange]`: pH_read, temperature, D2O fraction and the
  starting D fraction; HISTORY item 70): `ladder_rates` gives `kint` and `protection = kint /
  rate` per step, `expected_uptake` the deuterium of unprotected amides over time, drawn
  dashed in the ladder figure, and the residue-rates figure gains a protection-factor panel.
  Modified residues borrow their standard residue's factors and are marked approximate.
- **Residue-level protection factors from ladders (owner, 2026-10-19; HISTORY item 105).**
  The owner's choice for residue-level factors where the ladders allow it (the peptide-derived
  residue fit gave poor factors). `ladder_rates` adds `log10_protection` and its se (from
  `rate_se`; `kint` is exact) and `precise` (resolved, `rate_se / rate` at most
  `[analysis.fragments] max_rate_error`, default 0.5). `ladder_residues`
  (`ladder_residues.csv`) combines the precise steps of a residue: the N-terminal ladder's
  estimate with the C-terminal one's (weighted mean in log10 rate, se scaled by the misfit
  when above 1, `agreement` `agree`, `disagree` or `single`), within one run, sample, parent,
  precursor charge, condition and fragment charge. Owner: gentle and harsh are never pooled,
  fragment charges stay separate, and ion types of one terminus (c and c-1) are not combined
  because c-type ions are mixtures read from the same peaks: each (N ion, C ion) pair is its
  own row, alternatives rather than replicates. The fit is the free `A exp(-k t) + C` of
  `fit_decay`, so either direction works; fixing `C` at the solvent's equilibrium is a
  possible later option (opt-in; it would hide scrambling or a wrong solvent). Every
  fragment figure shows +-1 se error ranges (points of the ladder and closure figures; rate,
  amplitude, from `amplitude_se`, and protection in the residue-rates figure; a band around
  each resolved fitted curve of the ladder figure, from the fit's covariance); the combined
  values of `ladder_residues` are not drawn.
- **Variants of one fragment**: charge states are kept separate for now (owner: they may give
  different exchange rates), as is the precursor charge (fragment deuteration differed by
  precursor charge, HISTORY items 39 to 41); ion-type variants (c and c-1, z-dot and z+1) are
  still to be decided.
- **Kinetics.** For continuous-labelling or back-exchange kinetics with the scan time as the
  exchange time: an exponential decay (or rise) of each fragment's deuterium with a constant
  model as the alternative, and per-residue rates from consecutive fragments. Plots (owner):
  clean D content against time, with the traces of chosen fragment ions or of chosen
  differences (`amide_difference`). The reference is pyHXExpress's kinetics notebooks; the
  owner provides references when this branch starts.
- **Several runs** (owner, 2026-10-15; HISTORY item 80): activations and ladders are compared
  across runs with a `RunSet`, as peptides are. Each row keeps its `run`; `merge` gives
  runs one sample, whose runs are then drawn side by side in that sample's figures (one
  series per run and condition) and never pooled (each run has its own exchange clock). The
  tables start with a `run` column ("The run as a variable", below). The residue-rate figure
  holds every precursor charge of a parent (an ETD 3+ run beside an EThcD 2+ run); the ladder
  and closure figures stay per precursor charge.

### MS1 protection factors (owner, 2026-10-15; peptide level built, HISTORY item 81)

Opt-in, and only with the labelling conditions (owner). Owner's decisions: one protection
factor per peptide; K = 2 ranks get apparent values; differences between samples; figures in
the coverage map's layout (a residue-level fit from peptides was built and later removed:
below). The protection factor compares the
measured exchange with that of unprotected amides: an amide with intrinsic rate `kint`
(`hxex.kint`, `kforw + kback` in the labelling solvent) exchanges at `kint / protection`
(EX2). It is always named `protection` (columns) or "protection factor" (text, figures), never
`P` or `p`: `p` is already the binomial parameter of the fits and the comparisons' p-value
(owner). The fragment branch's `ladder_rates` uses the same name.

- **What is fitted.** The back-exchange-corrected fraction `fD` of a peptide (0 to 1; the
  plateau is 1 whatever the solvent's D fraction, since the correction scales to the fully
  deuterated control) against `mean_i(1 - exp(-kint_i t / protection))` over the amides the
  peptide's `n_amides` counts (the same `nterm_subtract`, prolines out). Uncorrected samples
  have no plateau of known height: no protection factors, with a warning.
- **Level: one protection factor per peptide (built).** Per-residue values from overlapping
  peptides need a regularised fit (pyHDX's ΔG fit): wanted by the owner, designed and built
  separately. Fitting single residues of the residue map is not proposed either: a
  residue's `fD` there is an average over whole peptides, whose amides have different `kint`.
- **Two estimates per peptide.** (1) The apparent protection factor at each time: the value
  that gives the measured `fD` (monotonic, one root; works with one time point). (2) One
  protection factor over all times by least squares on `fD` weighted by `fD_se`, with its
  standard error; a point whose error is unknown (no bootstrap) takes the median of the
  peptide's known errors, and with none known the fit is unweighted (an unknown error must
  never get the largest weight; audit of 2026-10-17). When the amides of one peptide are protected differently from each other
  (exposed and buried amides in one peptide; not conformational states, which the K = 2 fits
  describe), the exposed ones exchange first and the apparent value rises with time; one
  factor then summarises the peptide poorly. Reported as `protection_drift` (log10 of the last
  apparent value over the first) and `one_factor_fits` (whether one factor fits every time
  within the errors).
- **Bounds, not values, outside the window.** With `fD` above about 0.95 at the first time,
  the protection factor is only known to be below a limit; with `fD` below about 0.05 at the
  last time, above one. Reported as a bound (`bound = 'upper'` or `'lower'`), never as a value.
- **Populations.** K = 1 (the average) and each K = 2 rank (the ranks the species filters
  keep), in separate rows and panels. Two populations come from EX1 or mixed exchange, where a
  protection factor has no EX2 meaning: a rank's value is an apparent one.
- **Settings.** `[analysis] protection = true` and `[analysis.labelling]` (an
  `ExchangeConditions`: `ph_read`, `temperature`, `d2o_fraction`, `reference`; one table for
  every sample or one per sample, `[analysis.labelling.<sample>]`, and per run
  `[analysis.labelling.<run>.<sample>]`). Labelling conditions alone
  change nothing; `protection` without them is refused. Not the `[run.controls] quench`
  conditions, which are the quench and LC's.
- **Outputs.** `protection.csv`: sample, peptide, range, `protection`, `protection_se`,
  `log10_protection`, `n_times`, `bound`, `protection_drift`, `misfit`, `one_factor_fits`,
  `n_approximate` (amides whose `kint` borrows a factor: modified residues), `direction`
  (`forward` or `back`), `corrected`, `fD0` and `fD0_se` (NaN when fixed), `equilibrium`
  (the mean `e_i`; 1 forward);
  `protection_by_time.csv`: the apparent protection factor per time, with the exchange's
  `progress`. With a reference,
  `compare_protection.csv` (`delta` in log10 units, `delta_se`, `z`, `significant` at
  `alpha`, `note` where a bound leaves no value; peptides matched as `peptide_match` says).
  Figures (owner: the coverage map's layout): `protection_<sample>` (a panel per K and rank,
  bars coloured by `log10_protection`, the uptake map's colours reversed so protected is
  dark, bounds hatched at their limit) and `protection_difference_<sample>` (diverging
  colours, pale where not significant).
- **Intrinsic rates in the protein.** Labelling happens in the whole protein, so a peptide's
  ends are not termini (unless they are the protein's, or written `Ac-`, `-NH2`), and its first
  amide's rate depends on the residue before it, taken from the FASTA or else from the other
  peptides (`intrinsic_rates(..., free_termini=)`). Without a FASTA no peptide ends at a free
  carboxyl.
- **Unreliable times** are left out with `[analysis] exclude_times` (the whole analysis).
- **Direction (two-direction model, built 2026-10-19 at the peptide level, HISTORY item 104;
  audit design item 3).** Protection factors apply to back-exchange kinetics too (the same
  ratio under EX2). No setting names the direction (owner, 2026-10-13): it follows from the
  labelling conditions. `initial_fraction` (the D fraction the amides began with) above
  `d2o_fraction` (the solvent the exchange happens in) is a back-exchange; without it,
  forward labelling. One model covers both:
  `fD(t) = mean_i(e_i + (fD0 - e_i) exp(-kint_i t / protection))`, with `kint_i = kforw +
  kback` and `e_i = kforw / (kforw + kback)` from `hxex.kint` for that solvent.
  - Forward labelling: `fD0 = 0`, `e_i = 1` (the corrected plateau): the model above, unchanged.
  - Back-exchange, corrected: `e_i` is the amide's equilibrium over `initial_fraction` (the
    fully deuterated reference is taken to be made in the labelling solvent; owner,
    2026-10-19). `fD0` is fitted by default (owner, 2026-10-18), between the largest `e_i`
    and 2, from three starting factors; it is fixed at 1 only when the series has measured
    controls (corrected, no control fallback; a fully deuterated control from the same
    peptide's other charge states counts as measured, species column `fd_other_charge`:
    the charge state does not change the deuterium level, HISTORY 130). The first usable point stays
    the user's choice (`exclude_times`), never chosen automatically.
  - Back-exchange, not corrected (no fully deuterated control, as in infusion kinetics with
    an undeuterated control and a low-D end point): fitted on the absolute scale, `fD =
    mean / n_amides`, `e_i` the amide's equilibrium, `fD0` fitted (owner, 2026-10-19). A
    retention loss mostly goes into the fitted `fD0`, because `e` is small. Forward labelling
    without correction stays left out (its plateau height is unknown).
  - Apparent factors per time and the bounds use the progress `(fD0 - fD) / (fD0 - e)` (`e`
    the mean of `e_i`) in place of `fD`, so `upper` and `lower` mean the same in both
    directions. With `fD0` fitted, a peptide at equilibrium at every time (from the largest
    start the data can have, 1 or `initial_fraction`) is an `upper` bound with `fD0` NaN, and
    one with fewer than three times gets no value (logged).
  - A sample whose peptides mostly go against its conditions' direction is refused, with a
    message saying which setting to change (the falling-or-rising test of the audit).
  - Errors: from the Jacobian, as for one parameter. On synthetic data (200 noisy repeats,
    fD error 0.01) they match the scatter when the decay falls within the time range; when
    most of it is over before the first time, `fD0` is poorly determined and its standard
    error understates the scatter (0.09 against 0.15 at protection 100 with the first point
    at 10 s).
- **Peptides without data and modified forms.** A peptide with no finite `fD` at any time
  has `n_times = 0` in `protection.csv`; each modified form is a peptide of its own sample.
  Reference peptides are matched as in
  `compare_peptides` (one species per side over a range pairs by range; several pair by
  identical sequence), so two forms over one range never collapse to one.
- **Residue level from peptides: REMOVED (owner, 2026-10-19; HISTORY item 106).** A
  regularised fit of one factor per residue to all a state's peptides (pyHDX's
  `fit_gibbs_global` as the reference; built 2026-10-15, HISTORY item 82, commit a6cb28f) was
  removed. On B5 the factors carried little information for single residues (multimodal
  exchange), and the owner did not want an option users might take for reliable. Its
  settings (`protection_residues`, `protection_r1`, `protection_max_se`) are refused with the
  reason. Residue-level rates and protection factors come from fragment ladders (fragment
  branch, "Residue-level protection factors from ladders"); the residue-level fD map stays
  (deuteration, not rates). The design and code are in git history if ever wanted again.

### Common output and plots

Both branches produce residue-level tables of the same layout, so comparisons, region
summaries, difference plots and residue profiles work on either, and a protein measured both
ways can be shown together.

Plots (matplotlib, returning figures):

- coverage map in the owner's style: one thin bar per peptide spanning its residues, coloured
  by relative uptake (0 to 1, a dark blue to teal to yellow scale) at one time for one rank,
  ordered by first residue and stacked downwards, wrapping back to the top when the stack is
  full; colour bar, light vertical grid lines every 25 residues, no y axis; a grid of samples,
  times and ranks;
- residue profiles: the ranks' curves with uncertainty bands, the one-population value, the
  chosen description as markers sized by population, and a population panel below (the choice
  figure, `plot_population_choice`, one sample per figure: confirmed by the owner, 2026-10-13, as
  the way to judge the one-or-two choice; comparisons between samples belong in other plots);
- redundancy and resolution strips (uncovered residues and prolines marked);
- comparisons: overlays and stacks (a reference sample against each other), difference plots;
  the by-residue comparison of samples follows the owner's `2pop_vsWT_byresidue` figure
  (reference chosen by the owner, 2026-10-13: one panel per sample against the reference in
  black, points with capped standard-error bars, no lines), showing any of the K=2 low and high
  values (the default, as in the reference) and the K=1 average (open points), set by
  `[analysis] overlay` (`residue_overlay_figure`); by default the K=2 values only where the choice
  keeps two populations, as in the reference, under the user's control (`overlay_two_only`, owner
  2026-10-13); the plots of the other comparisons wait for the owner's current references;
- region bar charts (fD and populations per region and sample, with error bars);
- an optional annotation strip under the residue axis (domains, sites);
- annotations are protein-specific and user-defined, never derived: named regions (name, first
  and last residue) and sites (residue, label) in the `[analysis]` section of the project file,
  used consistently by every residue plot (dashed lines at region boundaries and sites, the
  annotation strip) and by the region bar charts (owner, 2026-10-13);
- for fragments: cleavage maps, uptake against fragment length (c ions, and the parent minus
  z ions), closure plots, comparisons between activations and precursor charges.

### The population test: one evaluation of K from spectrum to residue

Design: owner, 2026-10-25; decisions settled; built 2026-10-25 (HISTORY item 174):
`hxex.population_test` (the run), `hxex.analysis.set_test` (the sets), the residue tiers in
`hxex.analysis.residues`, the tier figure in `hxex.report.tiers`.

**Why.** Today each level decides the number of populations differently: the run per spectrum by
BIC or AIC; `best_set_K` per peptide set by the summed criterion (uncalibrated; minor share
5 %); the residues by a half-weight vote of the `best_set_K = 2` peptides, or in `"bootstrap"`
mode by the bootstrap errors; and the residue-level confidence of HISTORY item 148 by a
calibrated test with four tiers. The owner wants one consistent statistical evaluation of K at
the peptide level carried through to the residues. Evidence: HISTORY items 139, 148, 161, 162,
166-173.

**Picking first.** The test is only as good as the sticks: taking the window maximum picked a
neighbour's flank whenever a brighter peak sat just outside a window, and those flanks read as
extra populations. `[run.picker] pick = "peak"` (the default since HISTORY item 172) takes only
apexes; bundles picked before keep their sticks until run again.

**The test.** For a spectrum, the statistic of K against K + 1 is
`T = 2 (nll_K - nll_K+1)` from its own fits (negative values count as 0). Its null comes from
simulation: data drawn from the spectrum's K-population fit with the fit's noise, refitted at 1
to K + 1 as the run fits (nested starts), `simulations` times (20); each spectrum keeps the mean
and variance of its null draws. A set (sample, peptide, time or control role; charge states and
replicates pooled) adds up its spectra's statistics, and the means and variances of their nulls
(independent spectra); a gamma with those moments gives the set's p-value (per-set null; at 20
simulations it agrees with 100 on 97-99 % of sets on a Q-TOF and an Orbitrap dataset, HISTORY
161-162; a null pooled over the run was unstable on the Orbitrap and is not used). A set has at
least K + 1 populations when `p < significance` (0.01) and the added population's mean share
(the smallest share of the K + 1 fits, averaged over the set) is at least `min_population`
(0.10); the steps run from 1 to `k_max - 1` and stop at the first that does not reject.
Calibration on simulated single populations: about 1 % called two (HISTORY 139).

**The width check (per spectrum): a flag (owner, 2026-10-26: not yet justified as a rule;
HISTORY 175).** A deuterated population of mean mu D whose amides each hold at most the
labelling D2O fraction r has a variance of at least mu (1 - r). A K + 1 fit with a component of
at least 1 D and 10 % narrower than 0.3 sqrt(mu (1 - r)) may hold something else (a co-eluting
species on the isotope grid, a single-bin spike): the spectrum is flagged (`narrow_component`
in `spectra.csv`), and its test is unchanged. Measured as a rule (HISTORY 170-171, 173), it took
the undeuterated sets reading two from 7.0 to 2.8 % on a Q-TOF benchmark and from 15.8 to 3.2 %
on an Orbitrap digest, but nothing showed the labelled spectra it would stop were wrong. r is
`[run.controls] d2o_fraction` (its default, 0.9, without controls).

**The undeuterated reference (per peptide): a flag by default, a cap as a switch (owner,
2026-10-26; HISTORY 175).** An undeuterated set cannot hold two deuterium populations, so one
that reads two holds another species. Every set of such a peptide (peptide and modification, in
every sample of the run; charge states pooled) is flagged `undeuterated_two` in
`species_uptake.csv`, and its undeuterated rows show it in the tier figure. Whether the labelled
spectra share the species is not known: it can move out of the measured bins as the peptide
takes up deuterium, and the measured enrichment (labelled sets of such peptides read two 62
against 30 % on the Q-TOF, 33 against 9 % on the Orbitrap; HISTORY 171, 173) does not show
that their second populations are the species. So their tests are unchanged by default; the
switch `[analysis.filters] undeuterated_cap` keeps the peptide's labelled and fully deuterated
spectra at one population, in every sample and at every step (spectra the review accepted
excepted). Candidate for later: cap a labelled spectrum only where its extra component sits at
the species' position.

**Which spectra enter a set.** The spectra the analysis keeps: left-out spectra
(`left_out_spectra.csv`), spectra kept at one population (the spectrum quality screens, the
undeuterated cap when switched on), spectra with a reviewed `review_K = 1`, and, from the step
whose K + 1 fit it is, a spectrum with a population above full deuteration (corrected fD above
`[analysis.filters] max_fD`, 1.1; owner, 2026-10-26, HISTORY 178-179) leave the test (accepted
spectra excepted); a set made only of those has one population. The limit 1.1 is about the
99th to 99.5th percentile of genuine fully deuterated single populations on two datasets; with a
bootstrap the component's own error could widen it (`1 + sqrt(0.1^2 + (z se)^2)`; not built:
the hold-out works on fits alone).

**Each level from the same numbers.**
- Spectrum: with the population test on, each spectrum's own K (`spectra.csv`, the figures, a
  `"selected"` bootstrap) comes from the same steps, the spectrum tested as a set of one
  (owner: tie the spectrum's properties to the test when it is on; otherwise
  the fit's criterion, BIC or AIC, as now). The undeuterated cap is a peptide-level rule and acts
  in the analysis only. A single spectrum has less power than its set, so spectra stay at one
  population more often than their sets; the set pools the evidence. A reviewed `review_K` still
  wins.
- Peptide: the set's K fills `best_set_K` (and `best_set`, `best_set_reason`, with the set's
  p-values in new columns) and gates the peptide comparisons as today (HISTORY 137, 144). Where
  two samples differ in their number of populations, the difference is borderline unless the
  side with more has p below `significance / 10` and the other above `significance x 10` at
  the step where they differ (owner, 2026-10-26; replaces the summed-criterion rule of
  `borderline_evidence` when the test is on).
- Residue: from the covering sets' results at each step, agreement (the weighted share of
  covering sets with at least K + 1 populations) and strength (a weighted Stouffer combination of
  the sets' z values, `z = Phi^-1(1 - p)` clipped at +-3.09), and the owner's tiers (two,
  localised: strong and agreement >= 0.5; two: strong and agreement 0.33-0.5; undecided: one of
  the two; no sign; HISTORY 148), up to three for now: step two gives "three" and "three,
  localised" by the same rule, a set without a step-two test (it read one) entering step two
  with z = 0; a set never tested (no p-value at step one: every spectrum left out of the test)
  takes no part in either step (audit 2026-10-28). Risk, documented (HISTORY 169): two neighbouring two-state regions read as three at
  their junction, which the residue level cannot tell from three states. K = 2 values from every
  covering peptide (HISTORY 141). The residue choice (the `choice_*` figures, `populations.csv`,
  and every comparison built on it) follows the tiers (owner, 2026-10-26, HISTORY 177): two
  populations where the tier is "two" or above (`[analysis.choice] tier = "two"`, the default;
  `"localised"` for half the weight) and the residue has K = 2 values to draw; the choice's
  bootstrap tests (separation, precision, K = 1 between, population range) do not apply. A
  "three" tier is drawn as two (K = 3 residue values are not drawn yet).
- The bootstrap gives errors only (positions, populations), never the decision. Without one,
  the values are drawn without error bands, and the figures say so.

`significance` and `min_population` act in the run (each spectrum's K) and in the analysis
(sets), so one value serves both: set in the run's table, read by the analysis.

**The screen with the test.** The BIC screen does not combine with the test (it zeroes a fifth
of the two-population sets, HISTORY 169). With the test on, `[run.fit] screen = true` skips the
K + 1 fit only where the loss floor bounds the statistic below 1 (`T <= 2 (nll_K - floor) < 1`):
such a spectrum enters with T = 0 and a chi-square null with 3 degrees of freedom (mean 3,
variance 6) instead of its simulation (31 % of the fits and simulations saved, 99.1 % of the
labelled sets' verdicts unchanged on the benchmark). The quick look, `null = "chi_square"`, gives
every spectrum that null with every fit made (no simulation; 95.6 % of sets and 89 % of residue
cells as the simulated null on the benchmark, but drifting with peptide size on the Orbitrap,
HISTORY 162): for a first look, not for the results.

**Settings.**

```toml
[run.fit]
k_min = 1                 # must be 1 for the test
k_max = 2                 # 3: a second step (twice the simulations)
screen = false            # with the test: skip K + 1 where T is bounded below 1

[run.population_test]     # opt-in: the table turns the stage on, as [run.bootstrap] does
simulations = 20          # per spectrum and step
significance = 0.01
min_population = 0.10
null = "simulated"        # "chi_square": the quick look, no simulation

[analysis]
populations = "population_test"      # the default when the bundle has the test; "best_set" and
                          # "bootstrap" kept
[analysis.filters]
undeuterated_cap = false   # true: keep a flagged peptide's other spectra at one population
```

Without `[run.population_test]` the analysis uses `best_set_K` as now; `populations = "population_test"` on
a bundle without the test is refused with the fix (add the table, run again). The rule travels
with the tables (audit 2026-10-28, question 17): `species_uptake` and `compare_peptides` take the
bundle's rule when none is given (as `hxex analyse` does), the species table records the rule it
was made with (`SpeciesUptake.populations`), `residue_map` follows it, and `choose_populations`
uses the tiers whenever the residue table has them; each can still be set explicitly. Refused: `k_min >
1` with the test on, the Poisson loss (the null draws the Gaussian noise of the fit).

**Workflow.** `hxex run` fits K from `k_min` to `k_max`, runs the test, sets each spectrum's K,
then bootstraps if asked; `hxex analyse` forms the sets, their K and the residue tiers in
seconds (changing filters, review or `significance` needs no new simulation); the bootstrap can
come later (`hxex resume`, `ks = [1, 2]`) for error bars, without changing the tiers. Later:
`hxex resume` adding the test to an existing bundle.

**Cost (measured, HISTORY 168-169, 174).** 1.4-1.8 ms per spectrum and simulation on the GPU
(RTX A4500: 26,164 spectra, one step, 12 min); on the CPU 13-21 ms on 2-8 cores (2.1 hours on
four cores for 26,000 spectra, one step), and 10 ms for step one, 17 ms for step two with the
default threads of a 48-core workstation (the tutorial's 3,874 spectra: fits 73 s, step one 13
min, step two 22 min: 11 and 18 times the fits). Opt-in, and the guide states the cost.

**Build order.** (1) The run stage: `hxex.population_test` (settings, the null, the test screen,
the width check flag, each spectrum's K), stored in the bundle (`population_test.parquet`) and
`spectra.csv`. (2) The analysis: the sets, the undeuterated cap, `populations = "population_test"` filling
the set columns. (3) The residue tiers (table, then the tier figure, the owner's reference being
the HISTORY 148 tier figure). (4) The guide, the settings reference, `hxex resume`.

**Evaluated (HISTORY items 169-173).** 1: the BIC screen does not combine with the test; the test
screen and the chi-square quick look above. 2: step two works where three states are resolvable;
the junction risk above. 3: the undeuterated rule as first designed (one sample's series) was
replaced by the width check and the peptide-wide cap, both flags since HISTORY 175 (the cap a
switch, off by default). 4: the CPU cost above.

## Later parts

- **Comparing separate runs** (a common workflow; owner, 2026-10-12): the same comparisons
  across bundles as across samples. Series are matched by identity (sequence with
  modifications, charge; never by `data_id`, a position within its run) with a check that they
  agree; overlays with one colour per run; differences per peptide and time with propagated
  uncertainty, difference or butterfly plots; a comparison of the runs' settings (manifests)
  that warns where they differ in ways that matter (picker, noise model, control times and
  limits); and a rule for a peptide and time fitted with one population in one run and two in
  another.
  As built (phase 4; HISTORY item 53): a `RunSet` (run name -> results) is analysed like one
  run whose samples carry their run in a `run` column (since HISTORY item 103; before, the
  sample was labelled `run:sample`), so every table, comparison and figure of phases 1 to 3
  works across runs (`hxex analyse A=a.zip B=b.zip --reference A:WT`, or `[analysis.runs]`).
  Peptides are matched by residue range by default (see "Comparisons within a run"; charge
  states are combined); `check_identity` lists species whose position or amide count differs
  between runs. `delta_crit` is pooled over every run. Runs that subtract different numbers
  of N-terminal residues are refused; different undeuterated times or back-exchange correction
  are warned about. `compare_settings` compares the recorded run settings with defaults filled
  in, and warns where they differ in what matters (`MATTERS`: picker, noise model, K values,
  criterion, bootstrap, controls and limits, envelope, `n_max_factor`, the modification
  registry), listing the rest (optimiser settings, seeds, quality limits, package versions)
  without a warning. One population in one run and two in another: every spectrum is fitted at
  every K, so the K=1 values are always compared; `compare_choices` reports per residue where
  the one-or-two choice differs, and neither side's description is imposed on the other (the
  same rule as between samples of a run). Whether one side supports two populations and the
  other one is worth knowing (owner, 2026-10-15; HISTORY item 75): residue comparisons are not
  filtered by the choice, and each row carries both sides' choices (`two_reference`, `two`,
  `choices_differ`) when a choice is given (`compare_residues(choice=...)`, always in
  `compare_populations` and in `hxex analyse`). The butterfly plot (reference up,
  other sample down, one line per time, differences below) is drawn for every comparison.
- **Recomputing and combining**: uptake without the spectra the review rejected
  (`recalc_uptake`), and combining runs (`combine_batch_data`).
  As built (phase 5; HISTORY item 63): rejected spectra were already left out of the uptake;
  the controls are now recomputed from the run's K=1 fits when the review rejected a control
  spectrum, or when the analysis changes control settings (`analysis_controls`,
  `[analysis.controls]`, e.g. `d2o_fraction`), which also applies new control defaults to an
  existing bundle without refitting. Runs are combined by `RunSet(merge=...)`
  (`[analysis.merge]`): samples given the same label are one sample, their replicate spectra
  pooled; the identity check still compares the runs unmerged. Each spectrum keeps its run:
  the species table lists a merged species' `runs` and `n_spectra_by_run`, and `run_offset`
  flags one whose runs differ systematically (the charge-trend rule, each run against the
  first), a batch effect pooling would otherwise hide (owner, 2026-10-13).
- **The run as a variable** (audit design item 2, docs/AUDIT_2026-10-17.md; built
  2026-10-18, HISTORY item 103). The run is a column, not part of a sample's name:
  - Every analysis table has a `run` column before `sample`. `run` is the run the row's
    values come from: blank for an analysis of one bundle, the run's name in a `RunSet`, and
    blank for a merged sample (its spectra pool several runs; `runs` and `n_spectra_by_run`
    list them). Fragment rows always keep their run, because fragment series are never pooled:
    a merged sample's runs are drawn side by side in its figures (one series per run and
    condition; `condition` stays as recorded, no `run:` prefix). A species is identified by
    `(run, sample, ...)` everywhere (`where(run=..., sample=...)`); the comparison tables carry
    `reference_run` beside `reference`.
  - `run:sample` is a display label only: figure titles, legends and file stems use it where
    the same sample name occurs in several runs of the table, and the plain sample name
    otherwise (`hxex.analysis.display_labels`). A merged sample is one sample and is never
    prefixed. Conditions of a merged fragment sample are labelled `run:condition` in the figures
    where a condition occurs in several of its runs, by the same rule.
  - Settings name samples, with the run as a qualifier: `reference`, `samples` (and the CLI
    `--reference`) take a sample name, which must be in one run of the set (an error lists
    the `run:sample` choices when it is in several), or `run:sample` to name one run's.
    Per-sample tables (`[analysis.fasta]`, `[analysis.labelling]`,
    `[analysis.fragments.exchange]`) are keyed by sample for every run, with a nested table
    under a run's name for that run alone: `[analysis.labelling.WT]` for WT in every run,
    `[analysis.labelling.A.WT]` for WT of run A (the run-specific entry wins; a nested table
    whose name is not a run of `[analysis.runs]` is refused). `[analysis.fragments.labels]`
    takes a run's own condition names the same way (`[analysis.fragments.labels.etd]`), which
    replaces the former `"etd:gentle"` keys. `merge` is unchanged: its keys are `run:sample`,
    which names one run's sample.
- **Export** to the DynamX-like format pyHDX reads (`hdexa_to_pyhdx`): not needed (owner,
  2026-10-13).

## Phases

1. The species uptake table (peptides), the residue level, the coverage map, the redundancy and
   resolution strips, and residue profiles.
2. The one-or-two-population choice and its figures (residue profiles with the chosen
   description and populations; population maps per sample).
3. Comparisons within a run, region bar charts, difference plots.
4. Comparing separate runs (built).
5. Recomputing and combining (built; the export was dropped).
6. The fragment branch, kinetics included (its own detailed design, with the owner's
   references).

## Validation

- Synthetic data with known answers for every layer (known residue deuteration, known
  components and populations).
- Residue values against pyHDX 0.4.3 on the same peptides (installed outside hxex; see
  CLAUDE.local.md), as a test oracle only.
- A visual check against the owner's existing figures, recomputed from the intact-peptide
  benchmark bundle.

## Decisions

Owner, 2026-10-13:

1. **Residue weighting**: pyHDX's (`1 / exchanging residues` per peptide, normalised per
   residue) as the starting point, with the weight exponent as a setting.
2. **Charge states and replicates** are combined into one value per peptide and time (redundancy
   counts peptides), with a flag for an obvious charge trend: a peptide whose charge states
   differ by more than their combined uncertainty, in the same direction, at most of its time
   points (a systematic effect such as an overlapping species, rather than noise; the exact
   rule is settled when it is built).
3. **Uncertainty**: the bootstrap standard error and the spread between replicates combined.
4. **Protein sequence**: a FASTA file (per sample, as variants differ). A "stitched" sequence
   option, assembled from the peptides where no FASTA is given, is under consideration by the
   owner (not decided).

## Open decisions

Recommendation first:

1. Spectra with quality flags: kept by default, with an option to leave them out.
2. Whether the undeuterated and fully deuterated controls appear as panels in the residue
   figures (the owner's figures show them).
3. Three populations: ranks 1 to 3 are shown, but the choice is between one and two only, for now
   (the population test, above, is designed to step beyond two).
4. The stitched-sequence option (decision 4 above).

## Reference notes (pyHXExpress `hxex_utils.py`)

What it does and what not to copy:

- It converts pyHXExpress output to pyHDX tables, builds pyHDX `HDXMeasurement` objects, and
  uses pyHDX's per-residue averaging, coverage plot and coverage matrix; populations become
  separate pyHDX states (`sample_pop1`, `sample_pop2`, and a fixed-one-population run for the
  average). It has no significance testing, no kinetics and no protection factors.
- `choose_fits` propagates the error of the population with an unsquared derivative in
  one term (`((avg - p1) / (p2 - p1)**2) * sd2**2`); `group_fits` repeats it and averages
  standard errors instead of propagating them.
- `plot_resred` takes the residue numbers of the first state for every state, and overwrites
  its sequence argument on the first state.
- `hdexa_to_pyhdx` parses time units by the last character (other units scale wrongly), puts
  standard deviations in deuterons where fractions are expected, and sets the fully deuterated
  uptake without a state filter (pop1 and pop2 overwrite each other).
- `convert_to_uptakedf` fills a one-population peptide's second population with its first
  population's value, and drops rows with a correction factor of 1.05 or more (a fixed limit).
- The first two residues are excluded and pyHDX's default of 85 % deuterium is assumed in the
  conversion; hxex has its own N-terminal subtraction and correction.
