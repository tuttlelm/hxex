# SPDX-License-Identifier: GPL-3.0-or-later
"""The HDX data summary table (Masson et al. 2019) of ``hxex analyse``: its rows and columns,
the user's items from ``[analysis.summary]``, the values taken from the analysis, and the
Markdown copy."""
import csv
import dataclasses

import numpy as np
import pytest
from scipy import stats

from hxex.analysis import AnalysisConfig
from hxex.analysis.compare import _delta_crit
from hxex.io import open_bundle
from hxex.pipeline import run
from hxex.report import Results, write_analysis
from hxex.report.summary import (
    BACK_EXCHANGE,
    CONTROLS,
    MASSON_ROWS,
    PEPTIDES,
    REACTION,
    REPEATABILITY,
    REPLICATES,
    SIGNIFICANT,
    SOFTWARE,
    TIME_COURSE,
    _row_of,
    _typical_sd,
    base_item,
    note_number,
    split_notes,
    summary_markdown,
)
from test_run import config, raw_spectra  # noqa: I001


@pytest.fixture(scope="module")
def two_samples(tmp_path_factory):
    """A bundle of two samples (the synthetic run's series 1 renamed V)."""
    raw = raw_spectra(n_series=2)
    raw = dataclasses.replace(raw, sample=np.where(raw.data_id == 1, "V", "S"))
    out = tmp_path_factory.mktemp("summary") / "bundle"
    run(raw, config(), out=out)
    return out


def _read(path):
    """summary.csv as {row: {column: value}} keyed by the row each item stands for, and the
    notes."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = list(rows[0])
    columns = {n: np.array([r[n] for r in rows], dtype=object) for n in names}
    table, notes = split_notes(columns)
    by = {base_item(item): {n: table[n][i] for n in names}
          for i, item in enumerate(table["item"])}
    return names, list(table["item"]), by, notes


def test_the_summary_has_masson_rows_a_column_per_sample_and_the_users_items(
        two_samples, tmp_path):
    items = {"Protease": "pepsin", "HDX reaction details": "PBS, pD 7.4",
             "replicates": "technical", "Quench": {"V": "only V"}}
    cfg = AnalysisConfig(reference="S", format="png", summary=items)
    with open_bundle(two_samples) as b:
        names = {p.name for p in write_analysis(Results.from_bundle(b), cfg, tmp_path / "a")}
    assert {"summary.csv", "summary.md"} <= names
    columns, items_shown, by, notes = _read(tmp_path / "a" / "summary.csv")
    assert columns == ["item", "S", "V"]  # the reference first
    order = [base_item(i) for i in items_shown]
    # the user's own items first, then Masson's rows in the paper's order, then hxex's
    assert order[:2] == ["Protease", "Quench"]
    assert [i for i in order if i in MASSON_ROWS] == list(MASSON_ROWS)
    assert order.index(MASSON_ROWS[-1]) < order.index(SOFTWARE) == len(order) - 1
    assert by["Quench"]["V"] == "only V" and by["Quench"]["S"] == ""
    # an item named like a Masson row (any case) is added to that row
    assert by[REACTION]["S"] == "PBS, pD 7.4"
    assert by[REPLICATES]["V"].endswith("; technical")
    # short values; how they were made is said once, in the notes the items point to
    assert by[PEPTIDES]["S"] == "1 (1)"
    assert by[TIME_COURSE]["S"] == "1 min"
    assert by[SIGNIFICANT]["S"] == "reference"
    assert by[SIGNIFICANT]["V"].startswith("> ") and by[SIGNIFICANT]["V"].endswith(" D)")
    assert by[REPEATABILITY]["V"].endswith(" D)") and " fD (" in by[REPEATABILITY]["V"]
    assert by[SOFTWARE]["S"].startswith("hxex ")
    numbers = [note_number(i) for i in items_shown]
    assert [n for n in numbers if n] == list(range(1, len(notes) + 1))
    assert note_number(next(i for i in items_shown if base_item(i) == SOFTWARE)) is None
    assert any("Hageman and Weis" in n for n in notes)
    md = (tmp_path / "a" / "summary.md").read_text(encoding="utf-8")
    assert "| Protease | pepsin | pepsin |" in md and "Masson et al. (2019)" in md
    assert f"\n{len(notes)}. " in md  # the notes below the table


def test_the_summary_says_what_it_cannot_fill(two_samples, tmp_path):
    """No reference: no threshold; no labelling conditions or text: reaction details not
    given; no protein sequence: the coverage's note says what was assumed."""
    with open_bundle(two_samples) as b:
        write_analysis(Results.from_bundle(b), AnalysisConfig(format="png"), tmp_path / "a")
    _, _, by, notes = _read(tmp_path / "a" / "summary.csv")
    assert by[SIGNIFICANT]["S"] == "not tested"
    assert by[REACTION]["V"] == "not given"
    assert any(n.startswith("Not given: write it in [analysis.summary]") for n in notes)
    assert any("No protein sequence ([analysis] fasta) for every sample" in n for n in notes)
    # the back-exchange of the series' own fully deuterated controls (each sample one here)
    with open_bundle(two_samples) as b:
        ctrl = Results.from_bundle(b).controls
    for sample, data_id in (("S", 0), ("V", 1)):
        loss = 100 * (1 - ctrl.retention[list(ctrl.data_id).index(data_id)] / 0.9)
        assert by[BACK_EXCHANGE][sample] == f"{loss:.0f} % ({loss:.0f}-{loss:.0f} %; 1)"
    assert by[CONTROLS]["S"] == "UN 1, FD 1 of 1"


def test_without_controls_the_summary_says_so(tmp_path):
    raw = raw_spectra(n_series=1)
    run(raw, config(controls=None), out=tmp_path / "bundle")
    with open_bundle(tmp_path / "bundle") as b:
        write_analysis(Results.from_bundle(b), AnalysisConfig(format="png"), tmp_path / "a")
    names, _, by, _ = _read(tmp_path / "a" / "summary.csv")
    sample = names[1]
    assert by[CONTROLS][sample].endswith("; not corrected")
    assert by[BACK_EXCHANGE][sample] == "not measured"


def test_the_repeatability_is_the_scatter_delta_crit_is_made_from():
    rng = np.random.default_rng(1)
    arrays = [rng.normal(0.4, 0.01, size=n) for n in (3, 3, 4, 3, 2, 1)]
    sd, n = _typical_sd(arrays)
    assert n == 5  # the single replicate does not count
    typical = 3.0  # the median replicate count of the five
    t = stats.t.ppf(1 - 0.05 / 2, 2 * typical - 2)
    assert _delta_crit(arrays, 0.05) == pytest.approx(t * sd * np.sqrt(2 / typical))


def test_user_items_are_text_and_markdown_cells_are_escaped():
    assert _row_of("Average peptide length/redundancy") == MASSON_ROWS[6]
    assert _row_of("hdx  CONTROLS") == CONTROLS
    assert _row_of("Protease") is None
    c = AnalysisConfig(summary={"Temperature": 20, "Quench": {"S": "x"}})
    assert c.summary == {"Temperature": "20", "Quench": {"S": "x"}}
    assert AnalysisConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="in quotes"):
        AnalysisConfig(summary={"Quench": True})
    with pytest.raises(ValueError, match="not a run"):
        AnalysisConfig(summary={"Quench": {"S": {"x": "y"}}})
    md = summary_markdown({"item": np.array(["a|b ¹", "", "Note 1"]),
                           "S": np.array(["1\n2", "", "x | y"])})
    assert "| a\\|b ¹ | 1 2 |" in md and md.endswith("\n1. x | y\n")
    assert note_number("Back-exchange, mean (IQR; n) ¹²") == 12 and note_number("a") is None
    assert base_item("Back-exchange, mean (IQR; n) ³") == BACK_EXCHANGE


def test_the_project_file_takes_an_analysis_summary_table(tmp_path):
    from hxex.config import load_project

    (tmp_path / "SpecExport").mkdir()
    (tmp_path / "hxex.toml").write_text(
        'project = "t"\n[input]\nformat = "specexport"\ndata_dir = "SpecExport"\n'
        '[analysis.summary]\n"HDX reaction details" = "PBS"\n'
        '[analysis.summary.Quench]\nWT = "0 C"\n', encoding="utf-8")
    project = load_project(tmp_path / "hxex.toml")
    assert project.analysis.summary == {"HDX reaction details": "PBS", "Quench": {"WT": "0 C"}}
