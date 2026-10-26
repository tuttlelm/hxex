# The population test: how many populations, from spectrum to residue

A peptide whose molecules exchange in two (or more) different ways gives a spectrum with two
overlapping envelopes. *hxex* fits every spectrum with one, two, and three populations. The
**population test** decides how many of them the data really support, with one rule that is
used for each spectrum, for each peptide at each time, and for each residue. It is optional:
switch it on with a `[run.population_test]` table in the project file.

Without the test, *hxex* chooses the number of populations of a spectrum by a score of each fit
(BIC, `[run.fit] criterion`) and that of a peptide by adding up its spectra's scores
(`best_set_K`). That works, but nothing says how often it calls two populations where there is
one. The test does say: it is set so that a single population is called two about once in a
hundred times.

## How it decides

1. **Each spectrum.** The fit with two populations always matches a spectrum at least as well
   as the fit with one; the question is whether it matches *better than noise alone would
   allow*. *hxex* finds out by simulation: it makes 20 copies of the spectrum from its
   one-population fit, each with new noise of the size the fit found, and fits each copy with
   one and two populations again. How much two populations improve the fit of these copies,
   which have only one, is the yardstick for the real spectrum.
2. **Each peptide at each time.** All the spectra of a peptide at one time in one sample, its
   replicates and charge states, are tested together: their improvements and their yardsticks
   are added up. A peptide has two populations at that time when the chance of seeing its
   improvement with only one population is below 1 in 100 (`significance = 0.01`), and the
   second population holds at least 10 % of the molecules (`min_population = 0.10`).
3. **Each residue.** Every residue is covered by several peptides. *hxex* combines their
   results into two numbers: how *strong* the evidence of the covering peptides is together,
   and how much of their *weight agrees* that there are two populations. Together they place
   the residue in a tier (below).

With `[run.fit] k_max = 3` the same steps are taken once more: two populations against three.

## Two flags for other species

An extra population is not always a second way of exchanging. Another peptide with nearly the
same mass can sit under the envelope, and the fit then reads it as a second population. *hxex*
flags two signs of this; neither changes any number of populations:

- **A population narrower than deuterium allows** (`narrow_component` in `spectra.csv`). A
  population of deuterated molecules cannot be narrower than a certain width, set by how many
  amides it has and the D2O fraction of the labelling buffer (`[run.controls] d2o_fraction`).
  A "population" much narrower than that may be another species, or a single spike.
- **Two populations in the undeuterated spectra** (`undeuterated_two` in
  `species_uptake.csv`). An undeuterated sample has one population by definition, so when a
  peptide's undeuterated spectra read two, something else is there. Every row of that peptide,
  in every sample, carries the flag, and the undeuterated rows of the tier figure show it.
  Whether the labelled spectra share the other species is not known: it can move out of the
  measured range as the peptide takes up deuterium. If you are sure it does not, set
  `undeuterated_cap = true` in `[analysis.filters]`: the peptide's labelled and fully
  deuterated spectra then take no part in the test, in every sample, and stay at one
  population (spectra you accepted in the review excepted).

Spectra the [spectrum quality screens](outputs.md#tables-from-hxex-analyse) keep at one
population, and spectra you set to one population in the review (`review_K = 1`), take no part
in the test. Neither does a spectrum whose two-population fit puts a population above full
deuteration (above `max_fD`, 1.1, in `[analysis.filters]`): no deuterated form of the peptide
can sit there, so such a fit says the spectrum is not one clean envelope (a heavy tail, another
species), not that it has two populations. It stays at one population (unless you accepted it
in the review), and `left_out_summary.csv` counts these spectra. Genuine fully deuterated
measurements read above 1.1 about once in 100 to 200 on the data measured so far.

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

- `spectra.csv`: `K` is now the test's number for each spectrum; `criterion_K` is the number
  the fit's score chose, `test_p_1` the chance of the spectrum's improvement with one
  population (and `test_p_2` with two, for `k_max = 3`), and `narrow_component` is the flag
  above.
- `species_uptake.csv`: `best_set_K` is each peptide's number at each time from the test, with
  `set_p_1` (and `set_p_2`), the reason in `best_set_reason`, `set_test_K` (the test's own
  number, which `best_set_K` follows only up to the largest number analysed) and the flag
  `undeuterated_two`.
- `compare_population_counts.csv` (with a reference sample): where a sample and the reference
  differ in their number of populations, `p` and `p_reference` are the two sides' chances at
  the step where they differ. The difference counts as clear only when the side with more
  populations is below a tenth of `significance` (0.001) and the other side above ten times it
  (0.1); otherwise it is `borderline` (the outlined bars of the Woods plot).
- `residues.csv`: `population_tier`, with `test_strength_2` and `test_agreement_2` (and the
  same for three).

A single spectrum has less evidence than all the spectra of its peptide together, so spectra
stay at one population more often than their peptides.

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

[Analysis methods](../ANALYSIS.md#the-population-test-one-evaluation-of-k-from-spectrum-to-residue)
gives the formulas and the evidence behind each choice.
