# Analysing part of a dataset

You may want to fit only some of your data: a few peptides to try the settings, one sample
before the others are measured, or a dataset split into smaller projects. You may also add data
to a project you have already run. In either case some results can change even for spectra you
did not touch. This page says which results can change, why, and how to keep them steady.

The reason is simple: a few values are worked out from many spectra together, and then used for
each of them. When the set of spectra changes, those values change, and so do the results that
use them.

## What does not change

The fit of a spectrum depends on that spectrum and on its own series' controls, not on the other
peptides in the project. The same spectrum, with the same controls and the same settings, gives
the same fit, the same number of populations, and the same bootstrap whatever else is in the
project.

The fits, the bootstrap, and the [population test](population-test.md) use random numbers.
Each spectrum's random numbers come from the stage's seed (`seed` in `[run.fit]`,
`[run.bootstrap]`, and `[run.population_test]`) and from what the spectrum is: its sample,
peptide, charge, replicate, time, and role. They do not depend on the project's name, on how the
files are laid out, or on the other spectra, so the same spectrum gets the same random numbers in
any project with the same seeds. The seeds used are written in the bundle's
`run_settings_used.toml` ([The settings used](outputs.md#the-settings-used)).

## What depends on the spectrum's own series

These are expected: keep each series together with its controls.

- **The undeuterated control sets the noise level of its series.** *hxex* judges how well a
  spectrum is fitted against the noise of its series, which it takes from that series'
  undeuterated control. If the control is missing from the project, each spectrum is judged
  against itself instead. A labelled spectrum is usually dimmer than its undeuterated control,
  so its noise is then set lower, and the fit chooses two or three populations more often than
  it should. `hxex run` warns when this happens and names the series, the `noise_reference`
  column of `series.csv` says for every series where its noise level came from (`undeuterated
  control` or `own spectrum`), and `hxex info` counts them. Keep every peptide and charge state
  you fit together with its undeuterated control in the same project.
- **The fully deuterated control sets the scale of the correction** (see
  [Back-exchange correction](back-exchange-correction.md)).
- **The charge states of a peptide share their fully deuterated controls.** With
  `combine_charges` (on by default), every charge state of a peptide takes the mean of its
  charge states' controls, so leaving out one charge state changes the others' correction.

## What depends on the rest of the project

These values come from every peptide and every sample in the project, so fitting part of the
dataset, or adding more, changes them:

- **Stand-in controls.** A series without a good fully deuterated control takes the median of
  all the project's accepted controls (`fd_fallback = "median"`), or a prediction fitted to all
  of them (`"kint"`, which needs at least 10 usable controls and otherwise uses the median). A
  different set of peptides gives a different stand-in.
- **Which fully deuterated controls are rejected.** A control far below the others is rejected
  (`outlier_sd`), and "the others" are the project's controls: the check is made only when there
  are at least 5 of them. In a small project a poor control can be kept that the full project
  would reject, or the reverse.
- **Replicates left out as outliers.** Both the controls' replicates (`[run.controls]
  replicate_outlier`) and the analysis's spectra (`[analysis.filters] replicate_outlier`) are
  judged against the typical replicate scatter of the whole project.
- **Stand-in series in the analysis.** With `stand_in_controls = "exclude"` (the default), a
  sample keeps the series corrected with a stand-in only when none of its series has a measured
  control. A project holding only some of a sample's peptides can therefore keep series that the
  whole dataset would leave out.
- **The critical difference of the comparisons** (`delta_crit` in `compare_peptides.csv` and
  `compare_populations.csv`). It is worked out from the replicate scatter of every peptide, time,
  and sample in the analysis, including samples you are not comparing. The same peptide can be
  significant in one analysis and not in another.
- **The m/z check and the picking window.** *hxex* checks each series' undeuterated envelope
  against the m/z calibration of its injection, worked out from all the peptides in it (or from
  the whole project when an injection has fewer than 10). With `recentre` (on by default), a
  series flagged `off_mz` has its windows moved, which changes its picked spectra.
- **Some quality flags.** `saturated` and `tails_low` are judged only on the brighter half of the
  project's undeuterated controls, and `mz_shift` compares a series with the project's typical
  offset.

## How to keep results steady

- **Fit the whole dataset in one project** when you can. To look at a few peptides, fit
  everything and read only the rows you need. Spectra you set aside with
  [Excluding spectra](excluding-spectra.md) also leave these values, like spectra that were never
  read, but the rules stay in the project file, so the change is recorded and easy to undo.
- **Keep every series with its controls**: the undeuterated and fully deuterated controls of
  every charge state, in the same project as the labelled spectra.
- **When you add data or rerun part of it, expect the values above to move.** Compare the
  `controls` and `control_fallback` columns of `series.csv` and the `delta_crit` column of the
  comparison tables between the two runs: they show whether a stand-in, a rejected control, or a
  different threshold explains a change.
- **Keep the same settings, seeds included,** when you want two runs to be comparable.
