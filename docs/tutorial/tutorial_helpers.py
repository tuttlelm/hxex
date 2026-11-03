# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers of the hxex tutorial notebooks: they change the project file and fill in the review
table, as the tutorial's pages describe doing in a text editor and a spreadsheet. Each prints
the lines it changes; running a cell again finds its change already made."""

import csv
from pathlib import Path


def change(old, new, section=None):
    """Replace the start of the first line of hxex.toml that starts with `old` by `new`
    (after the line [section], or # [section], when given), and print the changed line. A
    line that already starts with `new` (the cell was run before) is left as it is."""
    path = Path("hxex.toml")
    lines = path.read_text().splitlines(keepends=True)
    start = 0
    if section is not None:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().lstrip("#").strip() == section)
    for i in range(start, len(lines)):
        if lines[i].startswith(new):  # already changed
            print(lines[i].rstrip())
            return
        if lines[i].startswith(old):
            lines[i] = new + lines[i][len(old):]
            path.write_text("".join(lines))
            print(lines[i].rstrip())
            return
    raise ValueError(f"no line starting with {old!r}")


def add_before(line, text):
    """Add `text` to hxex.toml just above the first line that starts with `line`, and print it
    (unless it is there already: the cell was run before)."""
    path = Path("hxex.toml")
    if text.strip() in path.read_text():
        print(text.strip())
        return
    lines = path.read_text().splitlines(keepends=True)
    i = next(i for i, x in enumerate(lines) if x.startswith(line))
    lines.insert(i, text.strip() + "\n\n")
    path.write_text("".join(lines))
    print(text.strip())


def review(sample, peptide_range, time, rep, review_K="", review_status="", note=""):
    """Fill in the review of one spectrum in user_review_table.csv (copied from
    results_tables/spectra.csv the first time)."""
    path = Path("user_review_table.csv")
    source = path if path.exists() else Path("results_tables/spectra.csv")
    with open(source, newline="") as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows if r["sample"] == sample and r["peptide_range"] == peptide_range
            and float(r["time"] or "nan") == time and r["rep"] == str(rep)]
    for r in hits:
        r.update(review_K=str(review_K), review_status=review_status, note=note)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(hits)} spectrum(s) of {sample} {peptide_range}, {time} s, replicate {rep}: "
          f"review_K {review_K or '-'}, review_status {review_status or '-'}, note {note!r}")


def show_table(path, columns=None, **match):
    """Print the rows of a CSV table whose columns equal the values given (only the `columns`
    named, when given), for example
    show_table("results_tables/series.csv", ["data_id", "charge"], sample="WT")."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if all(r.get(k) == str(v) for k, v in match.items())]
    if not rows:
        print("no rows match")
        return
    names = list(columns or rows[0])
    widths = [max(len(n), *(len(r[n][:24]) for r in rows)) for n in names]
    print("  ".join(n.ljust(w) for n, w in zip(names, widths, strict=True)))
    for r in rows:
        print("  ".join(r[n][:24].ljust(w) for n, w in zip(names, widths, strict=True)))
