# The noise model

Every fit needs to know how much of a spectrum is noise: a spectrum that departs from a
one-population envelope by more than its noise is fitted with two or three populations, and the
error bars of every value grow with the noise. This page describes the two noise models *hxex*
offers, how to switch back to the older one, and what the default's one setting does.

## The two models

- **Noise per bin, measured from your controls (the default).** Each bin gets its own noise,
  from two parts:
  - **Counting noise**, which grows with the bin's intensity and with the ion's charge. *hxex*
    measures its size once for the whole run from the undeuterated controls (`b1`, the
    instrument's counting scale), so it needs no information about the instrument.
  - **Model error**, a fixed share of the fitted intensity at each bin (`model_error`, 9 % by
    default): an allowance for the ways real envelopes depart from the model's binomial
    populations, which repeat from one replicate to the next and so are not noise in the usual
    sense. It is a share of what the fit predicts, not of what was measured: a fit that misses
    a tall peak altogether then pays for it in proportion to the peak, so it explains the
    intensity first. (Until 5 November 2026 it was a share of the measured intensity, so a
    missed peak cost about the same however tall it was, and a fit could prefer a small
    envelope spread over more bins to a tall one. Results made before keep that rule when
    they are resumed.)
- **One noise level per spectrum (the older default).** Every bin of a spectrum gets the same
  noise, 3 % of the tallest peak of its series' undeuterated control (`[run.fit.noise]
  percent`). This is what *hxex* used until 31 October 2026, and what pyHXExpress uses. It is
  simple, but the level was never measured: it gives weak bins far more noise than they have,
  and bright bins of a strong spectrum less.

On three Q-TOF datasets the per-bin model gave error bars that matched the scatter between
replicates (within about 10 %; with one level per spectrum they were up to 1.7 times too
large). It works best together with the [population test](population-test.md): there it fitted
undeuterated and fully deuterated controls, which should hold one population, with two less
often than one level per spectrum on every dataset tried but one (that dataset's fully deuterated controls: 4.3 against
3.4 %), and chose the right number of populations more often on a test set with known answers
(85 % against 78 %). Without the population test the picture is mixed: on two datasets more
undeuterated controls were split (for example 10.6 against 8.3 %), and spectra in general were
split slightly more often. Neither model fully covers the differences between charge states of
one injection. On the one Orbitrap dataset tried, the counting scale measured from the
controls came out about four times too large, so its error bars were about 1.5 times larger than
the replicates' scatter whatever `model_error` was, and it chose two populations more often than
one level per spectrum; use it there with care.

**Comparing samples.** Because the per-bin model fits the replicates of a sample more
consistently, the scatter between replicates is smaller, and so is the smallest difference that
counts (`delta_crit`, [Statistics](statistics.md)). More differences between samples are then
found: on the tutorial data and on a larger Q-TOF dataset, about 20 % and 70 % more residues
differed from the reference. A check that compares two replicates of a sample with its other two,
where every difference found is false, found false differences in about 1 % of peptides and
residues with either model, so the extra differences are real gains in sensitivity, not noise.

## What `hxex info` shows

`hxex info` shows what was measured, for example:

```text
noise     per bin, z x b1 x I + (9 % of the fit)^2 with b1 = 8.16 +- 0.49 (from 2683 undeuterated
          controls of 1172 series, controls with tops of 400,000 or more left out); b1 by
          sample: A 9.32 +- 1.7, B 8.93 +- 3.3, C 7.75 +- 1.1, D 8.02 +- 5.8, E 7.92 +- 2.4
```

Which controls are used: the undeuterated controls that every spectrum quality screen passes
(at the `loose` level, [Outputs](outputs.md)), and only those below the brightness where the
run's controls stop matching their natural isotope envelope (detector compression or a similar
distortion; *hxex* finds this brightness from the controls themselves and reports it). At least
five series need such controls; with fewer, *hxex* says so and uses one noise level per spectrum
instead.

**Samples measured differently.** *hxex* also measures `b1` from each sample's controls alone.
When a sample's own value differs from the run's by more than three standard errors, `hxex run`
warns: its spectra may come from another instrument or another setting, and a run that mixes
them gets one counting scale that suits neither. On the datasets tried so far every sample
agreed with its run. A sample whose own controls are too few to measure it (fewer than five
series), or whose controls show no counting noise of their own, is listed as `not measured`;
its spectra are fitted with the run's value like every other sample's.

## What `model_error` does

`model_error` sets how far a spectrum may depart from the model's envelopes before *hxex* reads
the difference as information. It is a choice between sensitivity and caution, not a correction
for a noisy dataset: the counting noise is measured, and `model_error` covers what the model
cannot describe.

- **Lowering it** (for example to 7) makes *hxex* more sensitive: more spectra are fitted with
  two or three populations, and the error bars become smaller. At 7 % the replicates on the three
  Q-TOF datasets scattered 1.1 to 1.3 times more than their error bars said, and more of the
  fully deuterated controls, which should hold one population, were fitted with two (for
  example 8.3 against 4.7 % at 9 % on the tutorial data). On the test set with known answers,
  7 to 9 % chose the right number of populations equally often (83 %).
- **Raising it** (for example to 12) makes *hxex* more cautious: fewer spectra are fitted with
  more than one population, and the error bars become larger. At 12 % the error bars were
  larger than the replicates' scatter on all three Q-TOF datasets (by up to a factor of 1.2),
  and on the test set more spectra with two real populations were fitted with one (82.5 %
  right).
- **Setting it to 0** leaves the counting noise alone. Then nearly every small mismatch between a
  spectrum and the model counts, and many spectra are split: about a third of the fully
  deuterated controls were fitted with two populations, and the error bars were two to three
  times too small.

At 9 % the error bars matched the replicates' scatter on one of the three Q-TOF datasets, and on
the other two the replicates scattered up to 1.12 times more than the error bars said; about 11 %
would match those two, and leave the first's error bars about 10 % too large. That scatter
includes small differences in labelling time between replicates, so the value that matches it is
at the upper end of what the model's own error needs, and 9 % was kept. Use the same value for
every run you intend to compare.

## Switching back to one level per spectrum

In the project file:

```toml
[run.fit.noise]
estimate = false
```

This is also needed with `model = "poisson"`, which has no per-bin form. Results made before 31
October 2026 were fitted with one level per spectrum; `hxex info` says which noise a run used.
Run them again (with `--overwrite` or `--keep-previous`: every fit changes) to use the per-bin
noise, and compare only runs made with the same noise model.
