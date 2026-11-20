# The population test: how many populations, from spectrum to residue

A peptide whose molecules exchange in two (or more) different ways gives a spectrum with two
overlapping envelopes. *hxex* fits every spectrum with one, two, and three populations. The
**population test** decides how many of them the data really support, with one rule that is
used for each spectrum, for each peptide at each time, and for each residue. It is optional:
switch it on with a `[run.population_test]` table in the project file.

The test is made twice, and the pages of this guide name the two apart: the **population test
(self)** tests each spectrum alone, when the spectra are fitted (`hxex run`), and the
**population test (set)** tests every replicate and charge state of a peptide in one sample at
one time together, its *set*, when the results are analysed (`hxex analyse`). The residue tiers
are built from the sets' results.

Without the test, *hxex* chooses the number of populations of a spectrum by a score of each fit
(BIC, `[run.fit] criterion`) and that of a peptide by adding up its spectra's scores
(`best_set_K`). That works, but nothing says how often it calls two populations where there is
one. The test does say: it is set so that a single population is called two about once in a
hundred times.

## How it decides

1. **Each spectrum: the population test (self).** The fit with two populations always matches a
   spectrum at least as well as the fit with one; the question is whether it matches *better
   than noise alone would allow*. *hxex* finds out by simulation: it makes 20 copies of the
   spectrum from its one-population fit, each with new noise of the size the fit found, and fits
   each copy with one and two populations again. How much two populations improve the fit of
   these copies, which have only one, is the yardstick for the real spectrum. The spectrum's own
   number of populations follows the same two limits as the peptide's below (the chance, and the
   share of the second population); the [fits at each K](gui.md) in the window say which one
   decided.
2. **Each peptide at each time: the population test (set).** All the spectra of a peptide at one
   time in one sample, its replicates and charge states, are tested together: their improvements
   and their yardsticks are added up. A peptide has two populations at that time when the chance
   of seeing its improvement with only one population is below 1 in 100 (`significance = 0.01`),
   and the second population holds at least 10 % of the molecules (`min_population = 0.10`).
   That limit was checked on simulated data: up to 10 % it removes about half of the false
   second populations and loses almost no real ones (under 1 percentage point); a higher limit
   starts to discard real minor populations of 15 to 30 %.
3. **Each residue.** Every residue is covered by several peptides. *hxex* combines the results
   of their sets into two numbers: how *strong* the evidence of the covering peptides is
   together, and how much of their *weight agrees* that there are two populations. Together they
   place the residue in a tier (below).

With `[run.fit] k_max = 3` the same steps are taken once more: two populations against three.

### When a spectrum and its peptide disagree

A spectrum and its set (all the spectra of its peptide in its sample at that time) can come to
different numbers. A replicate with a small second population can stay at one when its share
is below the 10 % limit on its own, while the set, with the evidence of every replicate added
up, clearly has two: on its own the spectrum cannot quite support two populations, but the set
adds the confidence to call it two. *hxex* keeps both, so nothing is lost: the spectrum keeps
its own `K` (its values do not change), and the peptide at that time, in every result of the
analysis, takes the set's. Where they differ, *hxex* writes both:

- `spectra.csv` has `set_K`, the set's number of populations, next to `K`;
- the window's results tree writes the spectrum's K as `K = 1 (self), K = 2 (set)` (hover it for the
  set's chance), and the time point's row ends with `(set 2)` when one of its replicates
  differs;
- the [fits at each K](gui.md) panel at the set's number adds the line "population test (set)
  p-value 6.9e-04 < 0.01: K = 1 (self), K = 2 (set)", beside the spectrum's own "population
  test (self)" line.

The sets are the ones `hxex analyse` tests, so they follow the analysis settings: the spectra
the review rejected, the spectra `[analysis.filters]` leave out, and the times
`[analysis] exclude_times` leaves out belong to no set, and their `set_K` is empty. `hxex run`,
`hxex resume`, and `hxex report hxex.toml` take the project file's `[analysis]` settings for
it, as does the window when the results are opened through their project file; a results
folder opened or reported on its own uses the default analysis settings. In the window the
set's numbers are worked out together with the "left out" column, so after a review decision
they are brought up to date with its **Update** button. `set_K` is empty without the
population test, and with `[analysis] populations` set to another rule.

## Two flags for other species

An extra population is not always a second way of exchanging. Another peptide with nearly the
same mass can sit under the envelope, and the fit then reads it as a second population. *hxex*
flags two signs of this; neither changes any number of populations:

- **A population narrower than deuterium allows** (`narrow_component` in `spectra.csv`, from the
  population test (self)). A population of deuterated molecules cannot be narrower than a
  certain width, set by how many amides it has and the D2O fraction of the labelling buffer
  (`[run.controls] d2o_fraction`). A "population" much narrower than that may be another
  species, or a single spike.
- **Two populations in the undeuterated spectra** (`undeuterated_two` in `species_uptake.csv`).
  An undeuterated sample has one population by definition, so when the population test (set) of
  a peptide's undeuterated spectra reads two, something else is there. Every row of that
  peptide, in every sample, carries the flag, and the undeuterated rows of the tier figure show
  it. Whether the labelled spectra share the other species is not known: it can move out of the
  measured range as the peptide takes up deuterium. If you are sure it does not, set
  `undeuterated_cap = true` in `[analysis.filters]`: the peptide's labelled and fully deuterated
  spectra then take no part in the population test (set), in every sample, and stay at one
  population (spectra you accepted in the review excepted).

Spectra the [spectrum quality screens](outputs.md#tables-from-hxex-analyse) keep at one
population, and spectra you set to one population in the review (`review_K = 1`), take no part
in the population test (set). Neither does a spectrum whose two-population fit puts a population
above full deuteration (above `max_fD`, 1.1, in `[analysis.filters]`): no deuterated form of the
peptide can sit there, so such a fit says the spectrum is not one clean envelope (a heavy tail,
another species), not that it has two populations. It stays at one population (unless you
accepted it in the review), and `left_out_summary.csv` counts these spectra. Genuine fully
deuterated measurements read above 1.1 about once in 100 to 200 on the data measured so far.

### Peptides and times not tested

A peptide and time whose spectra all take no part is not tested: when they were all kept at one
population (by the screens, the review, or `undeuterated_cap`) it reads one population, as you
or the screens decided, but when the test held them out itself (a population above `max_fD`, or
spectra the run's test did not test) its number of populations is not known. It then reads
`set_test_K` 0 in `species_uptake.csv` (and `set_K` 0 in `spectra.csv`), `best_set_reason` says
"not tested" and why, it counts in no tier, and its values come from the one-population fits.

## The residue tiers

| Tier | What it means |
|---|---|
| two, localised | strong evidence, and the peptides carrying at least half of the residue's weight have two populations |
| two | strong evidence, and a third to a half of the weight has two |
| undecided | only one of the two: strong evidence without agreement, or agreement without strong evidence |
| no sign | neither |
| three, localised; three | the same rule one step further, where the residue already has two |

*Strong* means the covering peptides' evidence together has a chance below 1 in 100 of
appearing with fewer populations. A residue between two neighbouring regions that each have two
populations can read as three, because most peptides there span both regions: overlapping
peptides that end between the regions tell the two cases apart, the residue level cannot.

`hxex analyse` draws the tiers in `population_tiers.pdf`: one row per sample and time point,
with the undeuterated and fully deuterated controls first, one column per residue. The
undeuterated rows should show no tier above *no sign*: where they do, look at those peptides.
Each row outlines the residue blocks of that sample's peptides at that point: the stretches
between neighbouring peptide ends, which the peptides cannot divide. Every residue of a block
has the same covering peptides, so the same tier: a long stretch of one colour can be several
blocks, and "localised" means localised to a block, not to a single residue.
A small resistor symbol marks each border of no information in that sample's peptides: a
junction no peptide spans, or the edges of a stretch no peptide covers (as on the significance
maps, see [Outputs](outputs.md)).

The tiers also decide what the residue plots show (`choice_<sample>_<time>`,
`populations.csv`, and the comparisons between samples): two populations wherever the tier is
"two" or above, one elsewhere. For a stricter reading, set `tier = "localised"` in
`[analysis.choice]`: only residues where half of the weight agrees then show two. A "three"
tier is drawn with its two-population values for now.

**Error bars need a bootstrap.** The test decides how many populations there are; the
bootstrap (`[run.bootstrap]` with `ks = [1, 2]`) gives the error bars on their values. Without
it, the residue plots draw the values without error bands and say so on the figure, and the
comparisons between samples, which judge differences against those errors, cannot be made
reliably.

## Turning it on

```toml
[run.fit]
k_min = 1                    # the test starts from one population
k_max = 2                    # or 3, to test for three as well (twice the time)

[run.population_test]        # the table turns the test on; every setting has a default
```

Then `hxex run hxex.toml`. With the test in the results, `hxex analyse` uses it by default
(`[analysis] populations = "population_test"`), and its tables carry the test's numbers:

- `spectra.csv`: `K` is now the number of the population test (self) for each spectrum;
  `criterion_K` is the number the fit's score chose, `test_p_1` the chance of the spectrum's
  improvement with one population (and `test_p_2` with two, for `k_max = 3`), and
  `narrow_component` is the flag above, for the fit at the spectrum's `K` (never set at one
  population). `set_K`, beside `K`, is the number of populations of the spectrum's set (see
  "When a spectrum and its peptide disagree" above).
- `species_uptake.csv`: `best_set_K` is each peptide's number at each time from the population
  test (set), with `set_p_1` (and `set_p_2`), the reason in `best_set_reason`, `set_test_K` (the
  number the population test (set) gave, which `best_set_K` follows only up to the largest
  number analysed) and the flag `undeuterated_two`.
- `compare_population_counts.csv` (with a reference sample): where a sample and the reference
  differ in their number of populations, `p` and `p_reference` are the two sides' chances at the
  step where they differ, from the population test (set). The difference counts as clear only
  when the side with more populations is below a tenth of `significance` (0.001) and the other
  side above ten times it (0.1); otherwise it is `borderline` (the outlined bars of the Woods
  plot).
- `residues.csv`: `population_tier`, with `test_strength_2` and `test_agreement_2` (and the
  same for three), from the population test (set) of the covering peptides.

A single spectrum has less evidence than all the spectra of its peptide together, so the
population test (self) leaves spectra at one population more often than the population test
(set) leaves their peptides.

**Adding the test to finished results.** Results made without the test do not need a new run:
add the `[run.population_test]` table to the project file and type `hxex resume hxex.toml`.
*hxex* keeps the spectra, the picking, and the fits, runs the test, and gives each spectrum the
number of populations of the population test (self). The spectra whose number changed get their
error bars at the new number (taken from the bootstrap at fixed numbers, `ks = [1, 2]`, when the
results have one). The results are then the same as those of a new run with the test. The test
uses the results' own noise model, so results fitted with an earlier *hxex* stay consistent with
themselves. Run `hxex resume hxex.toml` again after changing the test's settings, and the test
is made again with them.

A different `k_max` needs no new run either: `resume` fits the numbers of populations the
results lack (results fitted with `k_max = 1` get their two- and three-population fits) and
leaves out those above, and keeps every other fit as it is. Simulations the earlier test made
with the same settings are kept too, so only the new steps are simulated.

Two things need a new run (`hxex run hxex.toml --overwrite`) instead:

- results fitted with `[run.fit]` settings other than the project file's, apart from
  `k_max` (for example other `adam_steps`): the test is made from the fits, so they must be
  the ones you want;
- results fitted with `model = "poisson"` in `[run.fit.noise]`.

Any other change of the project file since the results were made stops `hxex resume`, which
names each one: a `[run]` setting such as the picking, or an `[input]` setting such as the runs,
the peptide table, the elution windows, or `[[input.exclude]]`. `resume` never reads or picks
the spectra again, so these need `hxex run hxex.toml --overwrite`, which fits again only the
series whose spectra changed. Set them back to resume instead. A project file without `[run.population_test]` leaves a test already in the results in place.
`hxex info` shows when the test was added. The results' `run_settings_used.toml` then holds
the project file's `[input]` and `[output]` settings, with the `[run]` settings the results
were made with, the test included.

**How long it takes.** The simulations take ten to twenty times as long as the fits. On the
[tutorial](tutorial.md)'s 3,874 spectra, on a workstation's CPU: the fits 1.2 minutes, the
test of one against two populations 13 minutes, and of two against three another 22 minutes
(`k_max = 3`). On a GPU, 26,000 spectra took 12 minutes for one against two. A laptop takes
longer; start with `k_max = 2`, or with the quick look below.

**A quicker look.** `null = "chi_square"` in `[run.population_test]` skips the simulations and
uses a standard yardstick instead. It agrees with the full test for about 96 % of peptides and
times, but less well for some peptide sizes: use it to look, not for results. With
`[run.fit] screen = true` the run also skips the two-population fit wherever the spectrum's
shape shows that two populations cannot improve it by enough to count (about a third of the
spectra), with hardly any effect on the peptides' results.

**Two populations decided, a third flagged.** The step from two populations to three takes
about 40 % of a complete run. With `decide_up_to = 2` in `[run.population_test]` (and
`k_max = 3`), the test decides one or two populations as usual, and the step to three uses the
quick yardstick above instead of the simulations. It decides nothing: where it would add a
population, the spectrum or the peptide at that time is flagged `more_populations_possible`
(in `spectra.csv` and `species_uptake.csv`), and keeps its number of populations. A set the
flag marks may hold a third population; a run that decides three (the setting left out, as
by default) tells. On the tutorial and on a larger dataset, the quick step flagged 4 of the 4 and
14 of the 16 peptides and times the full test put at three, and 5 others. With the setting,
`set_p_2` is left empty (that step decides nothing), and the residue tiers stop at two.

## The numbers behind it

This section gives the arithmetic for readers who want it; nothing above depends on it.

**How well a fit matches a spectrum: its loss.** Each bin of the spectrum is compared with the
fit, and the difference is weighed by how noisy that bin is expected to be (its variance *v*).
For a spectrum with bins *b*:

&nbsp;&nbsp;&nbsp;&nbsp;loss = ½ Σ<sub>b</sub> [ (predicted<sub>b</sub> − observed<sub>b</sub>)² / *v*<sub>b</sub> + ln *v*<sub>b</sub> ]

With the [per-bin noise model](noise-model.md) (the default), each bin's variance is

&nbsp;&nbsp;&nbsp;&nbsp;*v*<sub>b</sub> = max( *z* · *b*<sub>1</sub> · *I*<sub>b</sub> + (*m* · predicted<sub>b</sub>)², (0.005 · *I*<sub>max</sub>)² )

where *z* is the ion's charge, *b*<sub>1</sub> the counting scale measured from the run's
undeuterated controls, *I*<sub>b</sub> the observed intensity (smoothed over three neighbouring
bins), *m* the model error (`model_error`, 9 %), and the last term a lower limit, 0.5 % of the
spectrum's tallest bin, so that empty bins keep a finite weight. The ln *v* term is there
because the model error follows the prediction, so fits with different numbers of populations
have different variances; without it a fit could lower its loss by predicting more noise. With
one noise level per spectrum (`estimate = false`), *v* is 1 in units of that level, and the
loss is half the sum of the squared differences.

**How much a population more improves the fit: the statistic.** For the step from K to K + 1
populations:

&nbsp;&nbsp;&nbsp;&nbsp;*T* = 2 × (loss with K populations − loss with K + 1 populations)

(0 when the difference is negative or smaller than the precision of the losses).

**How much it would improve by chance: the null.** The 20 copies of the spectrum, made from
its K-population fit with new noise and fitted again, give 20 values of *T* for a spectrum that
has only K populations. Their mean *μ* and variance *σ*² describe what chance alone gives.

**The chance: the p-value.** *T* is compared with a gamma distribution of that mean and
variance (shape *μ*²/*σ*², scale *σ*²/*μ*): *p* is the share of that distribution at or above
*T*. A spectrum the screen skipped, which has no simulated copies, takes *μ* = 3 and *σ*² = 6
(a chi-square distribution with three degrees of freedom, a gamma too).

**The population test (self)** uses one spectrum's *T*, *μ*, and *σ*². The step adds a
population when *p* < `significance` (0.01) and the population it adds holds at least
`min_population` (10 %) of that spectrum.

**The population test (set)** adds up the three numbers of every spectrum in the set:

&nbsp;&nbsp;&nbsp;&nbsp;*T*<sub>set</sub> = Σ *T*<sub>i</sub>, &nbsp; *μ*<sub>set</sub> = Σ *μ*<sub>i</sub>, &nbsp; *σ*²<sub>set</sub> = Σ *σ*²<sub>i</sub>

and reads its *p* from the gamma with *μ*<sub>set</sub> and *σ*²<sub>set</sub> at
*T*<sub>set</sub>, as for one spectrum. Adding the variances treats the spectra as independent.
The step adds a population when that *p* < 0.01 and the *mean* of the spectra's added
populations is at least 10 %. The set's p-value does not come from the spectra's p-values; it
comes from their statistics and their nulls added up.

Each spectrum keeps its own fits throughout: the spectra are never fitted together, with
shared positions or shares (a "global fit", as some other tools do). Only the evidence is
pooled. Two things follow:

- A spectrum whose extra population is no better than chance adds about as much to
  *T*<sub>set</sub> as to *μ*<sub>set</sub>: it weakens the set's evidence, but does not veto it.
  For example, with a typical null (*μ* = 3, *σ*² = 6 for each spectrum), three spectra each
  with *p* ≈ 0.002 and a fourth with *p* ≈ 0.8 give the set *p* ≈ 7 × 10⁻⁶.
- One very strong spectrum can carry a set on its own, when the others show nothing (one with
  *p* ≈ 10⁻⁸ and three with *p* ≈ 0.8 give the set *p* ≈ 2 × 10⁻⁵); the 10 % mean share, which
  counts every spectrum, is then the guard. When the set passes, the other spectra show
  `K = 1 (self), K = 2 (set)`; when the share stops it, the strong one shows
  `K = 2 (self), K = 1 (set)` (see
  [When a spectrum and its peptide disagree](#when-a-spectrum-and-its-peptide-disagree)).

[Analysis methods](../ANALYSIS.md#the-population-test-one-evaluation-of-k-from-spectrum-to-residue)
gives the formulas and the evidence behind each choice.
