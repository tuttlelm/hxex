# Presets: how much work a run does, in one line

A run can be quick, so that you can look at your data first, or thorough, so that the results
can be reported. The difference lies in a handful of settings: how many populations each
spectrum is fitted with, how the population test decides, and how much bootstrapping is done.
A **preset** sets them all at once, with one line at the top of the project file:

```toml
project = "my_project"
preset = "bimodal"
```

Or, without a text editor: `hxex set hxex.toml preset bimodal`.

Without a preset, every setting keeps its default, as described in the
[settings reference](settings.md).

## The four presets

| Preset | Populations tried | Population test | Bootstrap | Use it for |
|---|---|---|---|---|
| `quick` | 1 and 2 | without simulations (the chi-square reference) | none | a first look; the results are provisional |
| `bimodal` | 1 to 3 | simulated, deciding between one and two; three is only flagged | at K = 2 for every spectrum, and at each spectrum's and set's own K | proteins with up to two populations: comparing them between samples |
| `multimodal` | 1 to `k_max` (3 unless you set it) | simulated, at every step | at each spectrum's and set's own K | proteins that may have three or more populations |
| `maximum` | as `multimodal` | as `multimodal` | as `multimodal`, with 100 resamples | the 95 % intervals of a publication; figures of every peptide |

What each one means:

- **`quick`** fits one and two populations and decides between them with the
  [population test](population-test.md), using its quick reference, which needs no simulations.
  There is no bootstrap, so there are no error bars from it. Use it to check the controls, look
  for spectra to review, see which peptides have two populations and where the samples differ,
  and set retention windows. *hxex* marks its results as provisional: `hxex info` says so, the
  analysis writes a warning in its log, and every figure carries a small note at its bottom
  left.
- **`bimodal`** is the complete analysis of up to two populations. The test decides between
  one and two populations with simulations; where a third population might be present, the
  spectrum is flagged `more_populations_possible`, but it is not described with three. Every
  spectrum is bootstrapped at K = 2, which the comparison of two populations between samples
  needs, and at the number of populations its peptide's set of replicates has (`set_k`), which
  gives every average its error.
- **`multimodal`** decides every step up to `k_max` with simulations, so a peptide can have
  three or more populations. Each spectrum is bootstrapped at its own number of populations and
  at its set's. Above two populations *hxex* gives each spectrum's populations with their
  errors, each set's average, and the residue tiers up to "three or more", but no
  residue-by-residue values or comparisons (see [Statistics](statistics.md)). To try more than
  three populations, set `[run.fit] k_max` as well.
- **`maximum`** is `multimodal` with 100 bootstrap resamples instead of 20, for 95 % intervals
  you can rely on, and the figures of every peptide (`[figures] series = "all"`) instead of the
  flagged ones only. It is slow: use a computer with a GPU if you can.

With an `[analysis]` table, `bimodal`, `multimodal` and `maximum` also fit the rate of each
ladder step of fragment ions (`[analysis.fragments] fit_rates`). A preset never switches the
analysis on by itself.

## How long each one takes

Measured with `hxex run` on the [tutorial](tutorial.md) data (3,874 spectra, on a shared
computer without a GPU) and on a dataset of about 26,000 spectra (on a GPU), before the presets
had their names (the complete run then did a little more than `bimodal` does now):

| Preset | Tutorial, no GPU | 26,000 spectra, GPU |
|---|---|---|
| `quick` | about 1 minute | about 2 minutes |
| `bimodal`, `multimodal` | under an hour | under an hour |
| `maximum` | 1.5 to 2 hours (estimated) | about 1.5 hours |

On a large dataset without a GPU, a complete preset can take several hours: about 6 hours for
the 26,000 spectra, and 8 to 10 hours with `maximum`. `hxex analyse` takes under a minute or
two with every preset.

## Your own settings win

A preset only fills in the settings the project file does not set itself. If the file says
`k_max = 4` under `[run.fit]`, the run tries up to four populations whatever the preset says.
This is how you adjust a preset: choose the one nearest to what you want, and write only the
settings you want different.

It also means that an old line can quietly override the preset. Project files written by
earlier versions of *hxex* have `k_max = 3` under `[run.fit]` and `series = "flagged"` under
`[figures]` switched on; with `preset = "quick"`, that `k_max = 3` wins over the preset's 2.
*hxex* tells you:

- `hxex check` prints a `preset` line: what the preset sets, and every setting the file sets
  otherwise, with both values.
- `hxex run`, `hxex resume` and `hxex analyse` write a warning naming those settings.
- `hxex set` says what a setting you switch off falls back to: `run.fit.k_max: 4 -> not set
  (preset "bimodal": 3)`.
- In the [*hxex* window](gui.md), a setting the preset fills in shows the preset's value, and its
  button reads **Preset** instead of **Default**.

To take the preset's value, delete the line, or put a `#` in front of it (or use
`hxex set hxex.toml run.fit.k_max --default`).

## What a preset never changes

A preset changes how much work is done, never how the results are judged. The same settings
apply at every level:

- what describes the experiment: the controls, `d2o_fraction`, the samples, the reference, the
  FASTA files, the labelling and quench conditions;
- the review: your review table and exclusion rules;
- the statistical rules: `alpha`, `[analysis.filters]`, `[analysis.choice]`, and the limits of
  the control checks;
- the instrument settings: the picking window, heights or areas, the noise model.

So `quick` and `bimodal` differ only in the evidence behind the results: the population test's
simulations and the bootstrap's errors.

## What the results record

The bundle records the preset and what it set. The `run_settings_used.toml` beside the results
writes out every setting the run used, with a comment naming the preset and the settings your
file set otherwise, so the record still tells what was done if a later version of *hxex*
changes what a preset means. `hxex info` shows the preset on its `preset` line.

## Changing the preset of a finished run

After a `quick` run, set the preset you want (`hxex set hxex.toml preset bimodal`) and run
again with `hxex run hxex.toml --overwrite`. `hxex resume hxex.toml` adds a bootstrap or the
population test to a finished run, but it does not fit new numbers of populations: from `quick`
(one and two) to a complete preset (up to three), it says so and changes nothing.
