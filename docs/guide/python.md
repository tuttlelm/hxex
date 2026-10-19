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
and `cancel` hooks (`hxex.hooks`). Settings are explicit config objects (`hxex.run.RunConfig`
and its parts); nothing is set globally.

## Reviewing spectra

A bundle's review layer records, per spectrum, a status (`accepted` or `rejected`), a chosen K
(`k_override`, 0 for none) and a note. Rejected spectra are left out of the uptake, the reports
and the analysis; a chosen K replaces the selected one. There is no command for this yet:

```python
from hxex.io import open_bundle

with open_bundle("results") as bundle:        # a bundle folder (a .zip is read only)
    review = bundle.review()
    review = review.update("my_protein:WT/0001-0011-LKDVEHFKVSS/30s-1-z2.csv",
                           status="rejected", note="co-eluting species")
    review = review.update("my_protein:WT/0012-0020-.../5m-1-z2.csv", k_override=1)
    bundle.set_review(review)
```

The spectrum ids are in the `spectrum_id` column of `spectra.csv`. A later `hxex run
--overwrite` carries the review over; entries the new run cannot hold are kept but marked stale.

## Pooled peptides and their proteins

For a pooled SpecExport folder, `hxex.io.match_candidates(entries, "proteins.fasta")` lists the
FASTA records each peptide could come from (every match, so ambiguous peptides show up):

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
