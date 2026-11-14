# Using hxex from Python (for programmers)

Everything in the rest of this guide is done with the `hxex` command and the project file; no
programming is needed. This page is for people who want to use *hxex* inside their own Python
scripts or notebooks. Install it into your own environment with
`pip install "git+https://github.com/tuttlelm/hxex"` (PyTorch first, as the README describes, if
you want a particular build).

## A run

```python
import hxex

project = hxex.load_project("hxex.toml")
spectra = hxex.read_input(project)                      # raw spectra, with their read record
result = hxex.run(spectra, project.run, out="results", project=project.project)

with hxex.open_bundle("results") as bundle:             # later, or in a notebook
    results = hxex.Results.from_bundle(bundle)
hxex.write_tables(results, "results_tables")
hxex.write_figures(results, "results_figures", hxex.PlotConfig(series="all"))
```

Plot functions return Matplotlib figures and never show them. Long operations take `progress`
and `cancel` hooks (`hxex.hooks`). Settings are explicit config objects (`hxex.pipeline.RunConfig`
and its parts); nothing is set globally.

Unlike the commands, `write_tables` and `write_figures` replace the files of an existing folder
without asking. From Python, *hxex* prints only warnings; to see the progress and the lines the
commands print, switch on logging before the run:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
```

## Looking at results

`hxex.load_results` loads finished results as `hxex report` sees them: from the project file
(its bundle, with its `[review]` applied and its `[figures]` settings) or from a bundle folder.
Typing the object's name at the prompt prints a one-line summary:

```python
import hxex

results = hxex.load_results("hxex.toml")
results              # Results(hspb5: 3874 spectra in 195 series; samples D109H, R120G, WT; ...)
results.samples()    # ['D109H', 'R120G', 'WT']
results.peptides("WT")   # [('0001-0009', 'MDIAIHHPW'), ('0001-0011', 'MDIAIHHPWIR'), ...]
results.times()      # [4.0, 60.0, 1800.0, 72000.0]  (seconds; controls left out)

hxex.show_series(results, "WT", "87-96", 2)                  # every page of one peptide
figures = hxex.show_series(results, "WT", "87-96", kinds=["uptake"])  # outside a notebook
figures[0].savefig("WT_87-96_uptake.png")
```

In a notebook `show_series` shows the figures; elsewhere it returns them. A peptide is named by
its residues (`"87-96"` or `"0087-0096"`); when nothing matches, the error says which samples,
charges, or nearby peptides there are.

The tables of `hxex report` are available as columns (a dictionary of NumPy arrays, one entry
per column, as in the CSV files):

```python
from hxex.report import find_series, format_table, series_table, uptake_table

series = series_table(results)                    # series.csv
print(format_table(series, limit=10))             # the first rows as text
(d,) = find_series(results, samples=["WT"], peptide_ranges=["87-96"], charges=[2])
uptake = uptake_table(results)                    # uptake.csv
rows = ((uptake["data_id"] == d) & (uptake["role"] == "")    # labelled times only,
        & (uptake["description"] == "centroid"))            # every replicate averaged
print(uptake["time"][rows], uptake["mean"][rows], uptake["mean_se"][rows])
```

`uptake_table` has, for each time, a `centroid` row (every replicate averaged, whatever its
number of populations) and then one row per number of populations (`K`) and population
(`rank`), described as `K=1`, `K=2 low-D`, `K=2 high-D` in the `description` column.
The undeuterated and fully deuterated controls have their own rows, with an empty time (NaN)
and the `role` column saying which control it is (`undeut` or `fulldeut`).
`find_series` returns the `data_id` of every series that matches (an empty list when none
does). `hxex.write_analysis(results, hxex.AnalysisConfig(reference="WT"), "analysis")` writes
what `hxex analyse` writes.

One series can be exported as an HX-Express spreadsheet, or as the CSV files behind its
figures ([Outputs](outputs.md#exporting-one-series) says what they hold):

```python
from hxex.report import export_hxexpress, plot_data, write_plot_data

export_hxexpress(results, d, "exports")                     # raw spectra, every replicate
export_hxexpress(results, d, "exports", spectra="picked", controls="first", suffix="sticks")
write_plot_data(results, "exports", data_id=d, ks="all")  # exports/STEM_bins.csv, ...
tables = plot_data(results, data_id=d)       # the same tables in memory: tables["bins"], ...
```

Both refuse to replace what is there unless given `overwrite=True`. `write_plot_data` and
`plot_data` also take `spectrum_ids=[...]` instead of `data_id`, for a time point or one
spectrum; `write_plot_data(..., name="...")` sets the start of the file names (by default the
series' name, as `plot_data_name(results, d)` gives it).

What a flag, control note, screen, or left-out reason means is in `hxex.flags`, the same text
as [Flags and reasons](flags.md) and `hxex explain`:

```python
from hxex import flags

e = flags.entry("off_mz")                    # also "fd_multimodal (retention agrees)", ...
print(flags.describe(e)["meaning"])          # Markdown, thresholds at the defaults
values = flags.setting_values(results.settings.run)   # the thresholds these results used
print(flags.plain(flags.describe(e, values)["action"]))
```

## Reviewing spectra

A bundle's review layer records, per spectrum, a status (`accepted` or `rejected`), a chosen K
(`k_override`, 0 for none) and a note. Rejected spectra are left out of the uptake, the reports
and the analysis; a chosen K replaces the selected one. The commands take the review from the
user review table instead ([The user review table](project-file.md#the-user-review-table)),
which leaves the bundle as it is; from Python the bundle's own layer is written like this:

```python
import hxex

with hxex.open_bundle("results") as bundle:        # a bundle folder (a .zip is read only)
    review = bundle.review()
    review = review.update("my_protein:WT/0001-0011-LKDVEHFKVSS/30s-1-z2.csv",
                           status="rejected", note="co-eluting species")
    review = review.update("my_protein:WT/0012-0020-.../5m-1-z2.csv", k_override=1)
    bundle.set_review(review)
```

The spectrum ids are in the `spectrum_id` column of `spectra.csv`. A later `hxex run
--overwrite` carries the review over; entries the new run cannot hold are kept but marked stale.

## Pooled peptides and their proteins

A run does this itself when the project file names `pooled_fasta` ([What each peptide may
be](inputs.md#what-each-peptide-may-be)): the tables carry each peptide's identity, and
`hxex.report.tables.series_identities(results)` gives them by series. Without a run,
`hxex.io.match_candidates(entries, "proteins.fasta")` lists the FASTA records each peptide
could come from (every match, so ambiguous peptides show up), and
`hxex.io.pooled.identify(peptide, start, record)` gives one peptide's identity:

```python
from hxex.io import match_candidates, scan

entries = scan("pooled", "pooled", sample="pool")
for m in match_candidates(entries, "proteins.fasta"):
    print(m.entry.peptide, m.matches)
```

## More

The modules' docstrings describe every function. The developer documents are one level up:
[PLAN.md](../PLAN.md) (design and decisions), [BUNDLE.md](../BUNDLE.md) (the bundle's files),
[ANALYSIS.md](../ANALYSIS.md) (the analysis methods).
