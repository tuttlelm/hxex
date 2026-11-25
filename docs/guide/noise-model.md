# The noise model

Every fit needs to know how much of a spectrum is noise: a spectrum that departs from a
one-population envelope by more than its noise is fitted with two or three populations, and the
error bars of every value grow with the noise. This page describes the two noise models *hxex*
offers, how to switch back to the older one, and what the default's one setting does.

## The two models

- **Noise per bin, measured from your data (the default).** Each bin gets its own noise,
  from two parts:
  - **Counting noise**, which grows with the bin's intensity and with the ion's charge. For
    spectra *hxex* sums from scans (mzML and Thermo raw files) it measures the noise curve of
    every scan file from that file's own scans (see [Where the counting noise comes
    from](#where-the-counting-noise-comes-from)); for pre-summed formats (SpecExport,
    HX-Express, pooled folders) it measures one counting scale for the whole run from the
    undeuterated controls (`b1`). Either way it needs no information about the instrument.
  - **Model error**, a fixed share of the fitted intensity at each bin (`model_error`, 9 % by
    default): an allowance for the ways real envelopes depart from the model's binomial
    populations, which repeat from one replicate to the next and so are not noise in the usual
    sense. It is a share of what the fit predicts, not of what was measured: a fit that misses
    a tall peak altogether then pays for it in proportion to the peak, so it explains the
    intensity first. (Until 5 November 2026 it was a share of the measured intensity, so a
    missed peak cost about the same however tall it was, and a fit could prefer a small
    envelope spread over more bins to a tall one. Results made before keep that rule until
    they are fitted again, which `hxex run` does, since other fitting code made them.)
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
one injection. (Until 24 November 2026 the counting scale came from the undeuterated controls'
residuals for every format; on an Orbitrap dataset that read it far too large, so the error bars
were 1.5 times the replicates' scatter. Scan files now take their own curves, below.)

**Comparing samples.** Because the per-bin model fits the replicates of a sample more
consistently, the scatter between replicates is smaller, and so is the smallest difference that
counts (`delta_crit`, [Statistics](statistics.md)). More differences between samples are then
found: on the tutorial data and on a larger Q-TOF dataset, about 20 % and 70 % more residues
differed from the reference. A check that compares two replicates of a sample with its other two,
where every difference found is false, found false differences in about 1 % of peptides and
residues with either model, so the extra differences are real gains in sensitivity, not noise.

## Where the counting noise comes from

**Scan files (mzML, Thermo raw).** When *hxex* sums a spectrum from several scans it also sums
the odd scans and the even scans separately. The two halves see the same peptide, the same
co-eluting species, and the same elution; only their noise differs. From the halves of every
spectrum of a scan file, labelled, undeuterated, and fully deuterated alike, *hxex* fits that
file's own noise curve: a floor, a counting term that grows with the intensity and the charge,
and a proportional term (`a2 + z b1 I + c2 I^2`). Every spectrum of the file is then fitted with
its file's curve. The curve needs no replicate injection and no control: a labelled file that has
no control of its own still gets its own curve, and a file that was recorded with other settings
shows it. On two Q-TOF datasets the halves give the same counting scale as the undeuterated
controls' residuals and as replicate injections; on an Orbitrap dataset the controls' residuals
had read the counting scale fifteen times too large (the instrument's per-scan cut was read as
noise), and the halves give the scale the replicate injections give, with error bars that match
the scatter between replicates.

- **Kinetics runs** (`[input.scans] group = "each"`, every scan its own spectrum) have nothing
  to halve: their curve comes from each scan against its two neighbours (the second difference
  of consecutive scans of one target, which removes the exchange between them), from
  `usable_from` on.
- **Reduced profile files** (Orbitrap scans stored with their weakest peaks deleted,
  [Inputs](inputs.md)) hide the floor of the curve, so their floor is the noise the instrument
  recorded for the scans, summed over the spectrum's scans, where that is larger than the
  fitted one.
- **A file with too few spectra** for a curve (fewer than about sixty bins with signal in both
  halves, or no counting term found) is named in the log and in `hxex info`, and its spectra
  take the run's median curve.
- The review does not change these curves: a rejected spectrum's halves still differ by its
  noise alone, and the robust fit absorbs the odd spectrum. So rejecting a spectrum, or editing
  one elution window, changes one file's curve by at most one spectrum among its hundreds, and
  the fits of the other spectra are kept ([Running again](outputs.md#running-again)).

**Pre-summed formats.** SpecExport, HX-Express, and pooled folders hold spectra already summed,
so the halves are not available. There *hxex* measures one counting scale for the run (`b1`)
from the undeuterated controls' residuals, as described under [What `hxex info`
shows](#what-hxex-info-shows). A project that mixes both kinds of input gets a curve per scan
file and the controls' scale for the rest.

`hxex check` says which route each scan file will take, and `hxex info` shows the curves:

```text
noise     per bin, a2 + z x b1 x I + c2 x I^2 + (9 % of the fit)^2, one curve per scan file (45
          files: 45 from the odd and even halves of each spectrum's scans); b1 median 783 +- 21
          (447 to 1.1e+03), a median 2.9e+04 (...; 45 from the recorded scan noise), c median
          0.2 % (0 to 0.4 %)
```

The files of one run differ by 10 to 20 % in their counting scale. A file whose own differs
from the run's median by more than three standard errors and by more than 10 % is named with a
warning (`differs`): it may come from another instrument or setting, or be worth a look.

## What `hxex info` shows

For pre-summed formats `hxex info` shows what was measured from the controls, for example:

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

An undeuterated control you reject in the review (`review_status = rejected` in your review
table, or a `[[review.exclude]]` rule) is left out of the noise too, and of that brightness:
a rejected control is often another species, whose misfit would read as noise. `hxex info`
counts the controls left out this way. Rejecting a control after the run makes the noise out of
date (`hxex check` says so), and the next `hxex run` measures it again. One control changes
`b1` very little, so the fits are usually kept (see [Running again](outputs.md#running-again)).

**Controls shared by samples.** When several samples read the same undeuterated control file (a
pooled project often does, and so may a copy or a link), *hxex* recognises the copies by their
content and counts each measurement once: in `b1` and its error, in the brightness where the
controls stop matching their natural isotope envelope, and in the counts `hxex info` shows. Each
sample's own `b1` still uses every control read for that sample. Counted five times over, one
file made the error of `b1` look about half its real size, and a handful of bright controls
could set that brightness on their own.

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
October 2026 were fitted with one level per spectrum, and results from scan files made before 24
November 2026 with the controls' counting scale; `hxex info` says which noise a run used. Run
them again (`hxex run`, which reads the files and fits every spectrum again here; add
`--keep-previous` to keep the earlier results beside them) to use the per-bin noise, and compare
only runs made with the same noise model.
