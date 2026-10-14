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
  honoured; uptake is back-exchange corrected by default (`hxex.controls`); populations are
  identified by rank, 1 being the lowest deuteration (see "Nomenclature" below).
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
  per residue, here also per peptide). K=1 is not filtered. On the
  intact-peptide benchmark they bring the K=2 residue values in line with the owner's
  pyHXExpress analysis (HISTORY item 49). The K=2 split depends on the noise model: the Poisson
  loss of the replicate noise estimate puts a small component in the high-mass tail, least
  squares (the default, and pyHXExpress) a moderate split. Decided (owner, 2026-10-13): the
  Gaussian loss (HISTORY items 55 and 56).
- **Combining** replicates and charge states: the mean, with the standard error
  `sqrt(max(mean(se_i^2), s^2) / n)` (the larger of the bootstrap variance and the observed
  scatter, so neither counts twice). **Charge trend** flag: the lowest and highest charge state
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
  layer. As built (`ChoiceConfig`: confidence 0.5, error limit 0.25, population 5 to 95 %), the
  reason a residue keeps K=1 is recorded (no K=2, not separated, error, K=1 outside,
  population). The tests need bootstrap standard errors at K=1 and K=2
  (`BootstrapConfig.ks`): without them only the scatter of replicates is left, which is far too
  small, and nearly every residue passes (`species_uptake` warns).
- **Comparisons within a run.** Differences between samples (a reference sample against each
  other) per peptide and per residue, with propagated standard errors and a significance rule;
  averages over user-defined regions of the sequence (pyHXExpress `group_fits`). As built:
  differences in fD at a fixed K and rank (K=1 by default); per peptide and time the hybrid rule
  (Hageman and Weis 2019; owner, 2026-10-13): Welch's t-test on the replicate spectra with
  `p < 0.05` and `|delta|` above `delta_crit = t(1 - alpha/2, 2n - 2) s_pooled sqrt(2/n)` (the
  pooled standard deviation of replicate spectra over every peptide, time and sample, `n` the
  typical replicate count); per residue, `|delta| / se` above the normal quantile and `|delta|`
  above the peptides' `delta_crit`. A region's value is the mean of its residues and its error
  the mean of their errors (residues share peptides, so the error does not shrink with the
  region's length). Regions may overlap. Populations are compared too (owner, 2026-10-13): a
  K=2 rank's population per peptide (`compare_peptides(..., quantity='population')`, the same
  hybrid rule) and the lever-rule population per residue (`compare_populations`), to see whether
  a sample shifts the balance between the populations rather than their deuteration; the
  comparison tables carry their `quantity` (`value`, `value_reference` and their errors).
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
  (`closure_table`: c + z, a + x, b + y against the precursor), with the ladder kinetics
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
  final equilibrium run). Back-exchange may not be fully correctable (in back-exchange
  kinetics the final run is back-exchanged to about 0 D), so theoretical references may be
  needed instead (owner); some kinetics runs have spurious early time points from the setup.
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
  run whose samples are `run:sample`, so every table, comparison and figure of phases 1 to 3
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
  every K, so the K=1 values are always compared; a K=2 rank or population only where both sides
  keep a K=2 description; `compare_choices` reports per residue where the one-or-two choice
  differs, and neither side's description is imposed on the other (the same rule as between
  samples of a run; recommended, to confirm with the owner). The butterfly plot (reference up,
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
3. Three populations: ranks 1 to 3 are shown, but the choice is between one and two only, for now.
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
