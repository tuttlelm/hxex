# The noise model

Every fit needs to know how much of a spectrum is noise: a spectrum that departs from a
one-population envelope by more than its noise is fitted with two or three populations, and the
error bars of every value grow with the noise. This page describes the two noise models *hxex*
offers, how to switch to the newer one, and what its one setting does.

## The two models

- **One noise level per spectrum (the default).** Every bin of a spectrum gets the same noise,
  3 % of the tallest peak of its series' undeuterated control (`[run.fit.noise] percent`). This
  is the setting *hxex* has always used. It is simple, but the level was never measured: it gives
  weak bins far more noise than they have, and bright bins of a strong spectrum less.
- **Noise per bin, measured from your controls (opt-in).** Each bin gets its own noise, from two
  parts:
  - **Counting noise**, which grows with the bin's intensity and with the ion's charge. *hxex*
    measures its size once for the whole run from the undeuterated controls (`b1`, the
    instrument's counting scale), so it needs no information about the instrument.
  - **Model error**, a fixed share of each bin's intensity (`model_error`, 9 % by default): an
    allowance for the ways real envelopes depart from the model's binomial populations, which
    repeat from one replicate to the next and so are not noise in the usual sense.

On three Q-TOF datasets the per-bin model gave error bars that matched the scatter between
replicates (within about 10 %; with the default they were up to 1.7 times too large). It works
best together with the [population test](population-test.md): there it fitted undeuterated and
fully deuterated controls, which should hold one population, with two less often than the
default on every dataset tried but one (that dataset's fully deuterated controls: 4.3 against
3.4 %), and chose the right number of populations more often on a test set with known answers
(85 % against 78 %). Without the population test the picture is mixed: on two datasets more
undeuterated controls were split (for example 10.6 against 8.3 %), and spectra in general were
split slightly more often. Neither model fully covers the differences between charge states of
one injection. It has not yet been checked on Orbitrap data, where on the one dataset tried it
chose two populations more often than the default; use it there with care.

## Switching it on

In the project file:

```toml
[run.fit.noise]
estimate = "controls"
model_error = 9          # percent; the default
```

Then run `hxex run` as usual (with `--overwrite` or `--keep-previous` if the run's results
already exist: every fit changes). `hxex info` shows what was measured, for example:

```text
noise     per bin, z x b1 x I + (9 % of I)^2 with b1 = 8.16 +- 0.49 (from 2683 undeuterated
          controls of 1172 series, controls with tops of 400,000 or more left out); b1 by
          sample: A 9.32 +- 1.7, B 8.93 +- 3.3, C 7.75 +- 1.1, D 8.02 +- 5.8, E 7.92 +- 2.4
```

Which controls are used: the undeuterated controls that every spectrum quality screen passes
(at the `loose` level, [Outputs](outputs.md)), and only those below the brightness where the
run's controls stop matching their natural isotope envelope (detector compression or a similar
distortion; *hxex* finds this brightness from the controls themselves and reports it). At least
five series need such controls; with fewer, *hxex* says so and uses the default noise instead.

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

- **Lowering it** (for example to 5 or 7) makes *hxex* more sensitive: more spectra are fitted
  with two or three populations, and the error bars become smaller. Below about 9 % the error
  bars on the Q-TOF datasets were smaller than the scatter between replicates (at 5 %, the
  replicates scattered 1.2 to 1.5 times more than their error bars said), and more of the fully
  deuterated controls, which should hold one population, were fitted with two. On the test set
  with known answers, 5 to 7 % chose the right number of populations most often (83 %), because
  most of its spectra truly hold two or three populations.
- **Raising it** (for example to 12 or 15) makes *hxex* more cautious: fewer spectra are fitted
  with more than one population, and the error bars become larger. At 15 % the error bars were
  larger than the replicates' scatter (by a factor of about 1.4), and on the test set some
  spectra with two real populations were fitted with one (79 % right).
- **Setting it to 0** leaves the counting noise alone. Then nearly every small mismatch between a
  spectrum and the model counts, and many spectra are split: about a third of the fully
  deuterated controls were fitted with two populations, and the error bars were two to three
  times too small.

9 % is the value at which the error bars matched the replicates' scatter on all three Q-TOF
datasets. That scatter includes small differences in labelling time between replicates, so 9 % is
at the upper end of what the model's own error needs; it was chosen over the known-answer set's
5 to 7 % because the known-answer set holds mostly multi-population spectra. Use the same value
for every run you intend to compare.
