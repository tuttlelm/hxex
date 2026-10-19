# Excluding spectra

Some spectra are bad: a failed injection, a co-eluting species, a replicate that clearly went
wrong. *hxex* lets you leave them out without deleting anything. You write a short rule in the
project file that says which spectra to leave out, and why. To bring them back, delete the rule.

There are two places to put a rule, depending on when you know the spectra are bad:

| | `[[input.exclude]]` | `[[analysis.exclude]]` |
|---|---|---|
| When to use it | you know before fitting (a failed run, a known bad file) | you found out after looking at the results |
| What happens | the spectra are not fitted at all | the spectra are fitted, but left out of the results |
| What is kept | the raw spectra, in the bundle | the raw spectra and their fits, in the bundle |
| To undo it | delete the rule and run again (`hxex run --overwrite`) | delete the rule and rerun `hxex analyse` or `hxex report` |
| Refitting needed | yes, to bring them back | no |

## Writing a rule

A rule is a table in the project file. It names one or more details of the spectra to leave
out, and a spectrum is left out when it matches **every** detail the rule names. For example:

```toml
[[analysis.exclude]]
sample = "WT"
peptide_range = "45-60"
charge = 3
note = "co-eluting species"
```

leaves out peptide 45-60 at charge 3 in sample WT: every time point and every replicate of it,
but nothing else. To leave out more than one thing, write one rule for each, each under its own
heading:

```toml
[[analysis.exclude]]
sample = "WT"
time = "30 s"
rep = 2
note = "bad injection"

[[analysis.exclude]]
peptide_range = "112-120"
note = "not this protein"
```

The second rule names only the peptide, so it leaves that peptide out of every sample.

| Detail | What it matches |
|---|---|
| `sample` | the sample (`"WT"`) |
| `peptide_range` | the peptide, by its residues (`"45-60"`; `"0045-0060"` is the same) |
| `charge` | the charge state (`3`) |
| `time` | one time point (`"30 s"`, `"5 min"`, or a number of seconds) |
| `role` | the controls: `"undeut"` (undeuterated) or `"fulldeut"` (fully deuterated) |
| `rep` | the replicate (`2`) |
| `file` | the data file's name: an mzML or raw file, an HX-Express spreadsheet, or a SpecExport spectrum file (`"30s-2-z3.csv"`) |
| `spectrum_id` | one spectrum exactly, by its id |
| `note` | why (optional, but recommended: it is kept with the results) |

The values to use are in the tables of an earlier run: `spectra.csv` in `results_tables` has a
row for every spectrum, with its sample, peptide range, charge, time, replicate, and
`spectrum_id`. Copy them from there.

## Before fitting: `[[input.exclude]]`

Use this when you know a spectrum is bad before you fit, for example a run that failed.

- *hxex* reads the spectra, sets the excluded ones aside, and fits the rest. Excluded spectra
  take no part at all: an excluded control does not enter the back-exchange correction.
- The excluded spectra are still stored in the bundle with the others, so nothing is lost. To
  fit them after all, delete the rule and run again from the stored spectra:
  `hxex run hxex.toml --spectra results --bundle results_all`.
- `hxex info results` shows how many spectra each rule left out, with its note.
- Fitting fewer spectra also saves time.

## After fitting: `[[analysis.exclude]]`

Use this when the figures or tables show you a spectrum is bad. The rule goes under the
`[analysis]` part of the project file.

- The fits stay in the bundle; nothing is refitted.
- `hxex analyse hxex.toml` leaves the spectra out of the analysis: the uptake, residue maps,
  comparisons, protection factors, and fragment ladders.
- For the run's own tables and figures, give `hxex report` the project file instead of the
  bundle: `hxex report hxex.toml --tables --figures` reads the bundle named in `[output]`,
  leaves the spectra out of `uptake.csv`, and marks them `rejected` in `spectra.csv` (its
  `review_status` column) and on their spectrum figures. `hxex run` does the same for the
  tables and figures it writes.
- The figures still show an excluded spectrum, so you can see what was left out, but marked:
  its spectrum and population panels are labelled `rejected`, and the uptake figure draws its
  point hollow and grey ("rejected / excluded" in the legend). Spectra rejected in the review
  are drawn the same way.
- Changing your mind costs only a rerun of `hxex analyse` or `hxex report`.

## Checking your rules

`hxex check hxex.toml` lists the rules, so you can see that they read as you meant. When the
run or the analysis applies them, *hxex* reports how many spectra each rule left out. A rule
that matches no spectrum at all gets a warning: usually a misspelt sample name, or a time or
range written differently from the data.

## Reviewing single spectra from Python

The bundle also has a *review layer*, where single spectra can be marked as rejected, or given
a chosen number of populations, from Python (see [Using *hxex* from Python](python.md)). Both
apply: a spectrum is left out if the review rejects it or a rule excludes it.
