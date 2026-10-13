# Post-fitting analysis: design

Analyses that start from fitted results and work on the protein rather than on one spectrum:
coverage, uptake per residue and per population, the choice between one and two populations,
comparisons between samples and between runs, region summaries, kinetics and export. This file
is the design; `docs/PLAN.md` ("Next steps") keeps the status and points here, and dated findings
and measurements go in `docs/HISTORY.md` as usual. Read this file before working on these
analyses.

Status: designed (2026-10-13); nothing built yet. The peptide (MS1) branch comes first.

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
  identified by rank, 1 being the lowest uptake (the slow, protected population).
- Analyses run on one results bundle, or on several for the run comparisons, and are configured
  by an `[analysis]` section of the project file and a command-line command.
- Fragment data (MS2, MS3) is analysed differently from peptide data (MS1) and gets its own
  branch. Population fractions shared by a parent's fragments are a post-processing step, not a
  joint fit in the engine (owner, 2026-10-13).

## Structure

### Shared foundation: the species uptake table

One table for every kind of measured species (peptide, intact protein, fragment ion). One row
per sample, species, time and population rank, under a given description (the selected or
effective K, or a fixed K from the fixed-K bootstraps):

- identity: the sequence with modifications, kind (peptide, intact, fragment), ion type and
  fragment label, parent (for fragments; an MS3 fragment's parent is an MS2 fragment, so
  positions on the protein follow the chain of parents), activation, precursor charge;
- position on the protein: first and last residue;
- values: corrected relative uptake (fraction of the exchangeable amides) and its standard
  error, population fraction and its standard error, the number of replicates and charge
  states combined, flags.

### Peptide branch (MS1; first)

- **Residue level.** Per sample, time and rank: per-residue values from the overlapping peptides
  that cover each residue, with redundancy (peptides covering a residue), resolution (the
  length of the shortest block) and masks (prolines, the excluded N-terminal residues,
  uncovered residues). The pyHDX method, for reference: each peptide weighs
  `1 / (its exchanging residues)` (raised to a weight exponent, 1 by default), normalised per
  residue, and standard deviations add in quadrature with the same weights.
- **One or two populations** (pyHXExpress `choose_fits`): per residue (or peptide) and time,
  the two-population description is kept when the populations are separated at a stated
  confidence (`(fast - slow) - z (se_slow + se_fast) >= 0`), both standard errors are below a
  limit and the slow fraction (from the lever rule with the one-population value) is inside
  `[min_pop, 1 - min_pop]`; otherwise the one-population value is used. Output: fD_slow, fD_fast,
  fD_avg and p_slow, with standard errors. Built on the selected K, the fixed-K bootstraps and
  the review layer.
- **Comparisons within a run.** Differences between samples (a reference sample against each
  other) per peptide and per residue, with propagated standard errors and a significance rule;
  averages over user-defined regions of the sequence (pyHXExpress `group_fits`).

### Fragment branch (MS2, MS3; after the peptide branch)

How fragment data differs, and what this branch does about it:

- **Ladders, not overlaps.** N-terminal ions (a, b, c) grow from the parent's N-terminus and
  C-terminal ions (x, y, z) from its C-terminus; consecutive fragments differ by one residue, and
  `c_n + z_(N-n)` should equal the parent (the closure test, HISTORY item 41). Residue values come
  from differences, estimated as a least-squares fit of per-residue deuteration (between 0 and 1)
  to all fragments of a parent at once, with the closure as a check.
- **Populations belong to the parent.** A fragment's populations are its parent's molecules, so
  the population fraction is shared by all fragments of a parent; a fragment that does not
  span the region where the populations differ looks like one population. The one-or-two
  choice is therefore made with fractions from the parent (or shared across its fragments) in
  post-processing, not fragment by fragment.
- **Scrambling.** Only low-scrambling activation (ETD, ECD, UVPD) gives site-specific values;
  the activation is carried and a scrambling check is part of the branch.
- **Amide counts per ion type** are unverified (PLAN, "Risks and open decisions") and must be
  settled before per-residue values from fragments are trusted.
- **Controls** are per fragment and per fragmentation type (an undeuterated run per type and a
  final equilibrium run); per-residue back-exchange can come from fully deuterated fragments.
- **Variants of one fragment**: charge states and ion-type variants (c and c-1, z-dot and z+1)
  are combined; precursor charge is kept as a separate condition by default (fragment
  deuteration differed by precursor charge, HISTORY items 39 to 41).
- **Kinetics.** For continuous-labelling or back-exchange kinetics with the scan time as the
  exchange time: an exponential decay (or rise) of each fragment's deuterium with a constant
  model as the alternative, and per-residue rates from consecutive fragments. The reference is
  pyHXExpress's kinetics notebooks; the owner provides references when this branch starts.

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
  chosen description as markers sized by fraction, and a fraction panel below;
- redundancy and resolution strips (uncovered residues and prolines marked);
- comparisons: overlays and stacks (a reference sample against each other), difference plots;
- region bar charts (uptake and fractions per region and sample, with error bars);
- an optional annotation strip under the residue axis (domains, sites);
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
- **Recomputing and combining**: uptake without the spectra the review rejected
  (`recalc_uptake`), and combining runs (`combine_batch_data`).
- **Export**: the uptake table in the DynamX-like format pyHDX reads (`hdexa_to_pyhdx`).

## Phases

1. The species uptake table (peptides), the residue level, the coverage map, the redundancy and
   resolution strips, and residue profiles.
2. The one-or-two-population choice and its figures (residue profiles with the chosen
   description and fractions; fraction maps per sample).
3. Comparisons within a run, region bar charts, difference plots.
4. Comparing separate runs.
5. Export, recomputing and combining.
6. The fragment branch, kinetics included (its own detailed design, with the owner's
   references).

## Validation

- Synthetic data with known answers for every layer (known residue deuteration, known
  populations and fractions).
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
2. Significance for comparisons: a hybrid rule (a difference threshold and a Welch test across
   replicates); or a threshold on the difference over its standard error.
3. Whether the undeuterated and fully deuterated controls appear as panels in the residue
   figures (the owner's figures show them).
4. Three populations: ranks 1 to 3 are shown, but the choice is between one and two only, for now.
5. The stitched-sequence option (decision 4 above).

## Reference notes (pyHXExpress `hxex_utils.py`)

What it does and what not to copy:

- It converts pyHXExpress output to pyHDX tables, builds pyHDX `HDXMeasurement` objects, and
  uses pyHDX's per-residue averaging, coverage plot and coverage matrix; populations become
  separate pyHDX states (`sample_pop1`, `sample_pop2`, and a fixed-one-population run for the
  average). It has no significance testing, no kinetics and no protection factors.
- `choose_fits` propagates the error of the population fraction with an unsquared derivative in
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
