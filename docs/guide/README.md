# hxex user guide

hxex fits HDX-MS spectra with mixtures of binomial populations: it reads spectra (exported or
raw), turns each into deuterium sticks, fits one to several populations with uncertainties, and
reports the populations per peptide, per residue and between samples.

This guide is for people running hxex on their data. The documents one level up
(`docs/PLAN.md`, `docs/HISTORY.md`, ...) are written for its developers.

## Start here

1. [The project file and the commands](project-file.md): a session from `hxex init` to
   `hxex analyse`, how the file works, the commands and their options.
2. [Input formats](inputs.md): HX-Express, SpecExport (separate or pooled), mzML and Thermo raw
   files; what each expects on disk and how samples, peptides, times and controls are read.
3. [Experiment types](experiments.md): bottom-up peptides, top-down and intact proteins, kinetics
   where the scan time is the exchange time, and fragment ions added to any of them.
4. [Outputs](outputs.md): every table and figure, and the quality flags.

## Topics

- [Controls and times](controls.md): undeuterated and fully deuterated controls, a time point as
  a control, writing times, times that only order the points.
- [The peak-picking window](picking-window.md): the ppm window, or one from the instrument's
  resolving power.
- [Comparing separate runs](comparing-runs.md): several bundles in one analysis, pooled or side
  by side.
- [Protection factors from back-exchange kinetics](back-exchange-protection.md).
- [Residue-level rates and protection factors from fragment ladders](fragment-residues.md).

## Reference

- [Settings reference](settings.md): every key of the project file, with its type, default and
  meaning (generated from the code).
- [Command-line reference](commands.md): every command and option (generated from the code).
- [The results bundle](../BUNDLE.md): the bundle's files and their columns.
- [Analysis methods](../ANALYSIS.md): how the post-fitting analysis computes its values.
