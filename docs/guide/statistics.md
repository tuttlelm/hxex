# Statistics: fits, bootstrap, replicates, and controls

*hxex* learns from four sources: the **fits** of every spectrum, the **bootstrap** that gives
them error bars, the **replicates** of each time point, and the **controls** of each series.
This page says what each one contributes, so you can choose how much of the work to do:
a quick look at a dataset needs only the fits, while error bars and some of the comparisons
need the bootstrap as well.

## What each source is

- **The fits.** `hxex run` fits every spectrum with one, two, and three populations (K = 1, 2,
  and 3) and chooses the number the spectrum supports best. This is fast: a minute or two for
  the [tutorial](tutorial.md)'s 3,874 spectra on a laptop.
- **The bootstrap.** Every spectrum is fitted again many times, each time with new random noise
  of the size the fit found, and the spread of these refits is the spectrum's standard error.
  It is the slow part, which is why it is switched on separately (`[run.bootstrap]`).
- **The replicates.** Separately labelled samples measured at the same time point (three or
  four per time in the tutorial data). They show how much a value varies from one sample to
  the next, which no single spectrum can tell you.
- **The controls.** The undeuterated (UN) control of each series sets its zero, and the fully
  deuterated (FD, or TD) control sets its scale, for the
  [back-exchange correction](back-exchange-correction.md).

## What the fits alone tell you

Without any bootstrap, *hxex* gives you:

- **The number of populations of every spectrum**, and of every peptide at every time. The
  second is `best_set_K` in `species_uptake.csv`: all the spectra of one peptide at one time in
  one sample, replicates and charge states together, are asked whether two populations describe
  them better than one. This is the decision the residue plots and the population comparisons
  are built on. The optional [population test](population-test.md) makes this decision by a
  calibrated test instead, from spectrum to residue.
- **The deuterium uptake** of every population of every spectrum, and the corrected values
  when the back-exchange correction runs (by itself when the data hold undeuterated and fully
  deuterated controls). The correction uses the controls' fits only.
- **Which peptides differ between samples**, the coloured bars of the Woods plot. This test
  uses the replicates, not the bootstrap (see [Replicates](#replicates) below).
- **The quality flags and the spectrum quality screens** ([Outputs](outputs.md#quality-flags)).

This makes a good first look at a new dataset: which peptides are multimodal, at which times,
and where the samples differ, in a tenth of the time of a full analysis. On the tutorial data,
the analysis without the bootstrap chose the same number of populations as the full analysis
for 747 of 756 peptides and times (98.8 %), and the same peptide differences, all 326 of them.
On a larger dataset of 26,000 spectra the numbers of populations agreed for 99.5 %.

What it lacks:

- **Error bars.** Without the bootstrap, the only errors are the scatter of the replicates,
  which here were 0.4 to 0.8 times the full errors: too small, most of all for the two
  positions of a two-population peptide. `hxex analyse` warns about this.
- **The precision check on two populations.** A two-population description whose populations
  are not measured precisely (an fD error of 0.25 or more, `max_fD_se`) is set aside, but only
  once its error is known. Adding the bootstrap therefore moves a few peptides and times from
  two populations to one: 9 of 756 in the tutorial data.
- **The residue comparisons** and **the population comparison**, described next.

## What the bootstrap adds

- **Error bars** on every value: the uptake plots, the Woods plots, and the residue plots.
- **The residue comparisons** (`compare_residues.csv` and `residue_significance`): a residue's difference
  is significant when it is large compared with its error, so the error must be right. Without
  the bootstrap, 97.5 % of the tutorial's residue calls agreed with the full analysis, but only
  89 % on the larger dataset.
- **The population comparison** (the lower panel of the second Woods figure): whether a sample
  holds more or less of the low-deuterium population than the reference. Where the two
  populations sit at the same places in both samples, it is judged as the K = 1 difference is
  (see [Same states, different shares](#same-states-different-shares)); where they do not, its
  test uses the errors of both population positions, so without the bootstrap it cannot be made
  for most of those peptides.
- **The residue-level comparisons at K = 2** (`compare_populations.csv`, `compare_residues_K2.csv`,
  and the coloured parts of `residue_significance_K2_<sample>`): whether a sample differs from the
  reference in the share of the low-deuterium population or in the positions of the two
  populations. Without a bootstrap at K = 2 these tests have no error for half or more of the
  residues (a residue whose peptides have one two-population replicate has no scatter to use),
  and the few differences they call are not the ones the bootstrapped analysis calls (0 to 63 %
  agree, against 59 to 96 % at K = 1), so *hxex* does not test these differences at all: the `significant`
  column of `compare_populations.csv` is left empty, `compare_residues_K2.csv` is not written,
  and `residue_significance_K2_<sample>` shows only where the number of populations differs. Bootstrap
  both numbers of populations (below) to get them.
- **The precision check** above, and the other way of choosing one or two populations,
  `populations = "bootstrap"` in `[analysis]`, which decides from the population positions and
  their errors.

## Two populations far apart

Every deuteration value at K = 1 (the uptake plots, the Woods plots, the residue values)
assumes that a fit with one population lands on the envelope's average. That holds for one
population, and for two that lie close together. It fails when two populations are far apart:
the one-population fit then settles on one of them, and which one can change from replicate to
replicate. On simulated data with two equal populations a whole amide count apart, the K = 1
value missed the true average by a third to two thirds of a deuteron.

So where a peptide's spectra at a time are best described by two populations or more (`best_set_K` 2 or more,
or the population test's decision) and the two lie at least 2 deuterons apart, *hxex* takes that
peptide's average from the two populations instead: the share-weighted mean of their positions,
for each spectrum. Those rows are marked `average_from_k2` in `species_uptake.csv`, and the
values feed everything built on the K = 1 values, including the comparisons and the residue
values. The populations' shares are reliable in exactly these cases, because the two are far
enough apart to be fitted well. On simulated data this brought the averages of such peptides
five times closer to the truth; on two real datasets it switched a fifth to a quarter of the peptides and
times, changing their values by 0.02 fD in the median.

The error of that average comes from the bootstrap, which measures how much the average itself
varies over its resamples: within one fit the positions and the shares trade off, and the
average is the part the spectrum pins down best. Without a bootstrap the error is the scatter
of the replicates, as for any value. A bundle whose bootstrap was made before this (before
*hxex* stored that error) gets an error built from the two positions' errors as if they were
independent, which is too large: small real differences in such peptides are then missed more
often (on simulated data 7 to 16 percentage points fewer); run its bootstrap again (`hxex
resume`) to use the right error. On simulated data with the right error, these peptides' small
differences were found as often as before the switch, with no more false ones.

## Same states, different shares

When both samples have two populations at the same places (no difference in either position
is found), a difference in the K = 1 average can only come from a different share: the share
difference is the K = 1 difference divided by the distance between the two populations. *hxex*
then compares the shares on positions shared by both samples (each position averaged over the
two), so that the positions' own errors cancel. At a peptide the share difference then takes
the K = 1 comparison's verdict, and at a residue the residue test is made the same way with the
K = 1 errors. Where a position does differ, each sample keeps its own positions and the test
includes their errors, as before. The column `positions` (`shared` or `own`) says which, in
`compare_peptides.csv` and `compare_populations.csv`. Each sample's own share is still the one
shown; only the difference and its test use the shared positions.

Two things follow. First, where the positions are shared, the K = 1 difference and the share
difference of a peptide are one measurement written two ways: they do not confirm each other.
Second, a clear K = 1 difference between two two-population samples now shows as a difference in
the share or in a position, as expected, instead of disappearing at K = 2 because the share's
error carried the positions' errors (about four times larger). On simulated data analysed with
the default noise model, no residue was wrongly called different (none, with or without this),
more of the small true share changes were found (59 to 69 %, from 56 %), and at the peptide
level the share was wrongly called different in 2 to 3 % of unchanged peptides, which are
exactly the K = 1 comparison's own wrong calls (HISTORY 281).

**Which numbers of populations to bootstrap.** The analysis uses each peptide's K = 1 and K = 2
values, whatever number its spectra chose, so bootstrap both for every spectrum:

```toml
[run.bootstrap]
n_boot = 20
ks = [1, 2]
```

The default, `ks = "selected"`, bootstraps each spectrum only at its own number, which leaves a
one-population spectrum without K = 2 errors and a two-population spectrum without K = 1 errors.
K = 3 is never needed for the analysis. Twenty resamples give usable standard errors; use 100 or
more when the 95 % intervals themselves matter, for a published figure, say.

**How long it takes.** On the tutorial data, with `ks = [1, 2]`:

| Step | Time |
|---|---|
| the fits | about 1.3 minutes on a CPU |
| the bootstrap, on a CPU | about 13.5 minutes (a laptop: 20 to 30 minutes) |
| the bootstrap, on a GPU | about 6 minutes (an RTX A4500) |

You do not have to decide at the start. Run without the bootstrap, look at the results, and
add it later with `hxex resume hxex.toml` after adding the `[run.bootstrap]` table: the fits
are kept, and only the bootstrap is computed
([The project file and the commands](project-file.md#run-report-and-analyse-what-each-one-does)).

## Replicates

Replicates are what a difference between samples is judged against. *hxex* uses them in four
ways:

- **The peptide comparison** (the Woods plot) uses the hybrid rule of Hageman and Weis
  ([*Anal. Chem.* 2019](https://doi.org/10.1021/acs.analchem.9b01325)): a difference is
  significant when Welch's t-test on the two samples' replicates gives p below 0.05 *and* the
  difference is larger than a threshold set by the typical scatter of replicates over the whole
  dataset (`delta_crit`; every sample in the results counts towards it, also one that
  `[analysis] samples` leaves out of the comparisons, so the threshold is the run's and does
  not change with the samples you choose to compare). The test needs at least **two
  replicates in each sample** at that
  time; with one, the peptide is not tested. Three or more are recommended, as in the
  community's recommendations for HDX-MS (Masson et al.,
  [*Nat. Methods* 2019](https://doi.org/10.1038/s41592-019-0459-y)).
- **The number of populations of a peptide** (`best_set_K`) adds up the evidence of all its
  replicates, so more replicates make the decision firmer.
- **The error of a combined value** is the larger of the bootstrap error and the scatter of the
  replicates. The two measure different things: the bootstrap sees the noise of each spectrum,
  and the replicates see the variation from one labelled sample to the next, which no bootstrap
  can see.
- **Odd replicates** are found by comparing them with the others: a spectrum far from its
  replicates (`replicate_outlier` in `[analysis.filters]`, with three or more) is left out of the
  analysis, and one much weaker than its replicates is flagged `low_intensity`.

The charge states of a peptide are **not** replicates: they come from the same sample, so
*hxex* averages them within each replicate before it combines the replicates. The averaged
value's error comes from the charge states that have one: where only some were bootstrapped
(a review that chose a K the bootstrap had not covered for every charge state), the error of
the average is that of the bootstrapped charge states alone, which is a little larger than the
error of all of them together would be.

## Controls

The controls turn deuterons into the fraction of each peptide's amides that exchanged, and a
bad control shifts or stretches every value of its series. They are fitted with one population,
always, so the correction needs only the fits.

- **The undeuterated control** sets the zero. *hxex* rejects one whose deuterium is more than
  0.5 above natural abundance, notes one whose replicates were fitted with two populations
  (`m0_multimodal`), and compares its isotope pattern with the one the peptide's composition
  predicts. A `saturated` control reads 0.07 to 0.17 deuterons too high, so the corrected uptake
  of its series reads low by about as much
  ([Very bright spectra](outputs.md#very-bright-spectra-saturated-tails_low-and-bright)).
- **The fully deuterated control** sets the scale, through its *retention*, the share of the
  label it still holds when measured. *hxex* judges each one against the rest of the run, against
  the series' own time points, and against the same peptide in the other samples
  ([How *hxex* judges the controls](back-exchange-correction.md#how-hxex-judges-the-controls)).
- **A control that fails is replaced by a stand-in**, such as the run's median retention, and
  `hxex analyse` leaves such series out of its comparisons by default
  (`stand_in_controls = "exclude"`). On a dataset with many missing or poor fully deuterated
  controls this can leave out a large share of the peptides, which is why good controls matter
  as much as good time points.
- **Replicate controls** are averaged, and one far from the others is left out first, as for the
  time points. A single control replicate gives the correction no error of its own (flagged
  `m0_unreplicated` or `fd_unreplicated`).

## Seeing what was left out

Every analysis folder has `left_out_summary.csv`: one row per step from the spectra read to
the comparisons, with how many spectra, series, or peptides each step left out, and the
setting that controls it. It is the quickest way to see where a dataset loses its data: on
a dataset with many missing fully deuterated controls, for example, the row `left out:
stand-in control` can be a tenth of all spectra. `left_out_spectra.csv` lists the spectra
themselves ([Outputs](outputs.md#tables-from-hxex-analyse)).

## Choosing a route

| You want to know | The fits | Add the bootstrap | Needs replicates | Needs controls |
|---|---|---|---|---|
| how many populations each spectrum, and each peptide at each time, has | yes | refines it (a few move from two to one) | more make it firmer | no |
| the deuterium uptake, in deuterons or corrected | yes | for its error bars | for the error of averages | for the correction |
| which peptides differ between samples (Woods plot colours) | yes | for its error bars | yes, two or more per sample | for corrected values |
| which residues differ | not reliably | yes | yes | for corrected values |
| whether the balance between two populations shifted | no | yes, at K = 1 and 2 | yes | for corrected values |
| which spectra look suspect (flags, screens) | yes | no | for `low_intensity` and replicate outliers | for the control checks |

A typical project:

1. **A first look:** `hxex run` without `[run.bootstrap]`, then `hxex report --figures` and
   `hxex analyse`. Use it to check the controls, review suspect spectra, and see which peptides
   have two populations and where the samples differ. Treat its error bars, residue
   comparisons, and population comparisons as provisional.
2. **The full analysis:** add `[run.bootstrap]` with `ks = [1, 2]`, run `hxex resume hxex.toml`,
   and run `hxex report --figures --overwrite` and `hxex analyse --overwrite` again.
3. **For publication:** raise `n_boot` to 100 or more, and run `hxex resume hxex.toml` again.

[Analysis methods](../ANALYSIS.md) gives the formulas behind each step.
