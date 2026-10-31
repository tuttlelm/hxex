# hxex user guide

*hxex* fits HDX-MS spectra with mixtures of binomial populations: it reads spectra (exported or
raw), peak picks the data, fits each spectrum to several populations with uncertainties, and
reports the populations per peptide, per residue, and allows comparison between samples.

This guide is for people running *hxex* on their data. The documents one level up
(`docs/PLAN.md`, `docs/HISTORY.md`, ...) are written for its developers.

No programming is needed: *hxex* is run by typing a few commands, and its settings are written in
a short text file.

## Start here

1. [Installing *hxex*](install.md): step by step, on Windows, macOS or Linux.
2. [Tutorial](tutorial.md): a complete analysis of published data (a protein and two disease
   variants), from the exported spectra to the comparison, with a notebook of the same steps.
3. [Your first project](first-project.md): from your exported spectra to tables and figures.
4. [The project file and the commands](project-file.md): how the file works, the commands and
   their options.
5. [Input formats](inputs.md): HX-Express, SpecExport (separate or pooled), mzML and Thermo raw
   files; what each expects on disk and how samples, peptides, times and controls are read.
6. [Experiment types](experiments.md): bottom-up peptides, top-down and intact proteins, kinetics
   where the scan time is the exchange time, and fragment ions added to any of them.
7. [Outputs](outputs.md): every table and figure, and the quality flags.

## Topics

- [Statistics: fits, bootstrap, replicates, and controls](statistics.md): what the fits
  alone tell you, what the bootstrap adds, and the role of replicates and controls.
- [The noise model](noise-model.md): one noise level per spectrum, or noise per bin measured
  from your controls (opt-in), and what its `model_error` setting does.
- [The population test](population-test.md): how many populations each spectrum, peptide,
  and residue has, decided by one calibrated test, and the checks that keep other species out.
- [Controls and times](controls.md): undeuterated and fully deuterated controls, a time point as
  a control, writing times, times that only order the points.
- [Back-exchange correction](back-exchange-correction.md): how the controls are judged, and
  what replaces a missing or rejected one, step by step.
- [Excluding spectra](excluding-spectra.md): leaving out bad spectra, before or after fitting, without
  deleting anything.
- [The peak-picking window](picking-window.md): the ppm window (or, with `pick = "window_max"`,
  one from the instrument's resolving power), and whether a bin takes a peak's height or its
  area.
- [Comparing separate runs](comparing-runs.md): several bundles in one analysis, pooled or side
  by side.
- [Analysing part of a dataset](part-of-a-dataset.md): which results can change when you fit
  only some of your data, or add more later, and how to keep them steady.
- [Protection factors](protection-factors.md).
- [Residue-level rates and protection factors from fragment ladders](fragment-residues.md).

## Reference

- [Settings reference](settings.md): every key of the project file, with its type, default and
  meaning (generated from the code).
- [Command-line reference](commands.md): every command and option (generated from the code).
- [The results bundle](../BUNDLE.md): the bundle's files and their columns.
- [Analysis methods](../ANALYSIS.md): how the post-fitting analysis computes its values.
- [Using *hxex* from Python](python.md): for programmers.
