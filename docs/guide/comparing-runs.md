# Comparing separate runs

One `hxex run` makes one results bundle. To compare bundles, give `hxex analyse` several of
them, each under a name of your own, or list them in the project file:

```toml
[analysis]
reference = "WT"          # the sample the others are compared with

[analysis.runs]           # run name = bundle
A = "day1/results"
B = "day2/results"
```

```bash
hxex analyse A=day1/results B=day2/results --reference WT
```

The runs are analysed like one run with more samples. Every table then has a `run` column
before `sample`: a row of sample `WT` from run A reads `A,WT,...`. Filter on the two columns
in a spreadsheet or a script; the sample's name is the same in every run.

**Naming a sample.** `reference` and `samples` take a sample's name. If that name is in only
one of the runs (run A has `WT`, run B has `MUT`), the name is enough. If several runs have
it (both runs have `WT`), say which one as `run:sample`: `reference = "A:WT"`. A bare name
that is in several runs is an error that lists the choices.

**Figures.** Titles, legends and file names show a sample as `run:sample` only when the same
name is in several runs (`A:WT`, `B:WT`; the file `butterfly_B_WT.pdf`), and as the sample's
name alone otherwise. The tables never use that form: the run is a column there.

**Settings per sample, per run.** The tables keyed by sample (`[analysis.fasta]`,
`[analysis.labelling]`, `[analysis.fragments.exchange]`, `[analysis.fragments.labels]`) apply
to the sample in every run. For one run's own value, nest a table under the run's name:

```toml
[analysis.labelling.WT]      # WT in every run
ph_read = 7.0
temperature = 20.0
d2o_fraction = 0.9

[analysis.labelling.B.WT]    # WT of run B only (B is a run of [analysis.runs])
ph_read = 7.4
temperature = 20.0
d2o_fraction = 0.9
```

A nested table whose name is not one of your runs is refused, so a misspelt run name is
caught.

**Merging runs.** When the same sample was measured in two runs and you want them pooled
(replicates across days), give the two a common label with `[analysis.merge]`:

```toml
[analysis.merge]
"A:WT" = "WT"
"B:WT" = "WT"
```

The merged sample's rows have an empty `run` (they pool several runs; `runs` and
`n_spectra_by_run` in the species table say which and how many), and `run_offset` flags a
peptide whose runs disagree systematically. Fragment series (ladder kinetics) are never
pooled: a merged sample's runs are drawn side by side in its figures, each series named
`run:condition` where the same condition occurs in both runs, or by the run's own names from
`[analysis.fragments.labels.<run>]`.

