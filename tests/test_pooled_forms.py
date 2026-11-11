# SPDX-License-Identifier: GPL-3.0-or-later
"""The forms of a pooled sample (hxex.analysis.pooled): the minimum set of candidates that
explains every peptide, modified forms on their base form, each form's own peptides, its
residue values, its comparison with the same form of the reference, and its figure page."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hxex.analysis import AnalysisConfig, PeptideComparison, SpeciesUptake
from hxex.analysis.pooled import (
    form_class,
    form_comparison,
    form_sequences,
    form_uptake,
    minimum_set,
    pooled_forms,
)
from hxex.chem import DEFAULT_MODIFICATIONS
from hxex.io.pooled import candidates_record

POOL = (">X1\nMAKLPGSTVEDRWQAFKLT\n>X2\nMAKLPGSTVEDRWQAYKLT\n"
        ">Y1\nMSSTTEEDDRRWWQQ\n")
# (peptide, start, end): shared by X1 and X2; X1's own; X2's own; X1 acetylated at K17
SHARED, ONLY_X1, ONLY_X2, ACETYL = (("AKLPGSTV", 2, 9), ("WQAFKLT", 13, 19),
                                    ("WQAYKLT", 13, 19), ("WQAFacKLT", 13, 19))


@pytest.fixture
def record(tmp_path):
    (tmp_path / "pool.fasta").write_text(POOL)
    return candidates_record(tmp_path / "pool.fasta", ["X", "Y"])


def keys(*peptides):
    return [(p, s, e, "") for p, s, e in peptides]


def uptake(rows) -> SpeciesUptake:
    """A species table (one run) from (sample, (peptide, start, end), time, fD)."""
    cols = {name: [] for name in ("sample", "peptide", "modification", "kind", "start", "end",
                                  "time", "role", "K", "rank", "mean", "mean_se", "fD", "fD_se",
                                  "population", "population_se", "n_amides", "n_spectra",
                                  "n_charges", "corrected", "charge_trend")}
    for sample, (peptide, start, end), time, fd in rows:
        values = {"sample": sample, "peptide": peptide, "modification": "", "kind": "peptide",
                  "start": start, "end": end, "time": time, "role": "", "K": 1, "rank": 1,
                  "mean": fd * 5, "mean_se": 0.05, "fD": fd, "fD_se": 0.01,
                  "population": 1.0, "population_se": 0.0, "n_amides": 5.0, "n_spectra": 3,
                  "n_charges": 1, "corrected": True, "charge_trend": False}
        for name, value in values.items():
            cols[name].append(value)
    return SpeciesUptake(**{k: np.array(v) for k, v in cols.items()}, nterm_subtract=1)


def test_the_minimum_set_keeps_required_candidates_and_merges_those_no_peptide_tells_apart():
    groups, alternatives, exact = minimum_set({"p1": ["A"], "p2": ["A", "B"], "p3": ["C", "D"]},
                                              ["A", "B", "C", "D"])
    # A alone holds p1; C and D hold the same peptides, so the data cannot tell them apart
    assert groups == [("A",), ("C", "D")] and alternatives == [] and exact
    # three candidates, each pair holding everything: three minimum sets of two
    triangle = {"p1": ["A", "B"], "p2": ["B", "C"], "p3": ["A", "C"]}
    groups, alternatives, exact = minimum_set(triangle, ["A", "B", "C"])
    assert groups == [("A",), ("B",)] and alternatives == [[("A",), ("C",)], [("B",), ("C",)]]
    assert minimum_set({}, ()) == ([], [], True)


def test_too_many_candidates_are_searched_one_group_at_a_time(monkeypatch):
    from hxex.analysis import pooled

    monkeypatch.setattr(pooled, "EXACT_LIMIT", 1)
    groups, alternatives, exact = minimum_set(
        {"p1": ["A", "B"], "p2": ["B", "C"], "p3": ["A", "C"]}, ["A", "B", "C"])
    assert not exact and alternatives == [] and len(groups) == 2


def test_the_forms_of_a_pool_with_a_modified_form_on_its_base(record):
    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, ACETYL), record)
    assert [f.name for f in forms.forms] == ["X1", "X2", "X1 acK17"]
    x1, x2, acetyl = forms.forms
    assert x1.peptides == (SHARED, ONLY_X1) and x1.own == (ONLY_X1,)
    assert x2.peptides == (SHARED, ONLY_X2) and x2.own == (ONLY_X2,)
    # the acetylated form: the base form's peptides that do not cover K17, and its own
    assert acetyl.base == "X1" and acetyl.sites == "acK17" and acetyl.members == ("X1",)
    assert acetyl.peptides == (SHARED, ACETYL) and acetyl.own == (ACETYL,)
    assert acetyl.candidates == ("X1",)
    assert [form_class(f, record) for f in forms.forms] == ["X", "X", "X"]
    cols = forms.columns()
    assert list(cols["form"]) == ["X1", "X2", "X1 acK17"]
    assert list(cols["n_own"]) == [1, 1, 1] and list(cols["minimum_set_unique"]) == [True] * 3
    assert cols["own_peptides"][0] == "13-19 WQAFKLT"
    assert cols["shared_peptides"][0] == "2-9 AKLPGSTV"


def test_a_peptide_no_candidate_holds_is_left_out(record):
    forms = pooled_forms(keys(SHARED, ("GGGGGG", 30, 35)), record)
    # the shared peptide alone: X1 and X2 cannot be told apart, so they are one form
    assert [f.name for f in forms.forms] == ["X1 or X2"]
    assert forms.forms[0].peptides == (SHARED,)
    assert form_class(forms.forms[0], record) == "X"


def test_each_form_gets_its_peptides_rows_and_its_records_sequence(record):
    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, ACETYL), record)
    up = uptake([("R", SHARED, 60.0, 0.3), ("R", ONLY_X1, 60.0, 0.4),
                 ("R", ONLY_X2, 60.0, 0.5), ("R", ACETYL, 60.0, 0.6)])
    fu, base = form_uptake(up, {"": forms})
    pairs = sorted(zip(fu.sample, fu.peptide, strict=True))
    assert pairs == [("R X1", "AKLPGSTV"), ("R X1", "WQAFKLT"), ("R X1 acK17", "AKLPGSTV"),
                     ("R X1 acK17", "WQAFacKLT"), ("R X2", "AKLPGSTV"), ("R X2", "WQAYKLT")]
    assert base[("", "R X1 acK17")] == "R"
    seqs = form_sequences(forms, record, base)
    assert seqs == {"R X1": "MAKLPGSTVEDRWQAFKLT", "R X2": "MAKLPGSTVEDRWQAYKLT",
                    "R X1 acK17": "MAKLPGSTVEDRWQAFKLT"}


def _comparison(rows):
    """A K=1 peptide comparison of V against R from (peptide key, delta, significant)."""
    n = len(rows)
    return PeptideComparison(
        sample=np.array(["V"] * n), peptide=np.array([k[0] for k, _, _ in rows]),
        modification=np.array([""] * n), start=np.array([k[1] for k, _, _ in rows]),
        end=np.array([k[2] for k, _, _ in rows]), time=np.full(n, 60.0),
        value_reference=np.full(n, 0.3), value_reference_se=np.full(n, 0.01),
        value=np.array([0.3 + d for _, d, _ in rows]), value_se=np.full(n, 0.01),
        delta=np.array([d for _, d, _ in rows]), delta_se=np.full(n, 0.01),
        t=np.full(n, 5.0), p=np.full(n, 0.001), n_reference=np.full(n, 3), n=np.full(n, 3),
        significant=np.array([s for _, _, s in rows]), reference="R", delta_crit=0.05)


def test_the_peptide_comparison_follows_the_forms(record):
    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, ACETYL), record)
    pc = form_comparison(_comparison([(SHARED, 0.0, False), (ONLY_X1, 0.2, True)]),
                         {"": forms})
    assert sorted(zip(pc.sample, pc.peptide, strict=True)) == [
        ("V X1", "AKLPGSTV"), ("V X1", "WQAFKLT"), ("V X1 acK17", "AKLPGSTV"),
        ("V X2", "AKLPGSTV")]
    assert pc.delta_crit == 0.05


def test_a_value_with_a_large_error_does_not_stretch_the_shared_axis():
    from types import SimpleNamespace

    from hxex.report.analysis import _highest

    table = SimpleNamespace(K=np.array([1, 1, 1, 2]), rank=np.ones(4, dtype=int),
                            time=np.full(4, 60.0), fD=np.array([1.2, 0.9, 0.5, 2.0]),
                            fD_se=np.array([1.0, 0.1, np.nan, 0.01]))
    # 1.2 +- 1.0 counts as 1.2 (its point stays on the axis); 0.9 +- 0.1 keeps its error
    assert _highest(table, [60.0], 0.25) == pytest.approx(1.2)
    table.fD[0] = 0.7
    assert _highest(table, [60.0], 0.25) == pytest.approx(1.0)
    assert _highest(table, [30.0], 0.25) is None


def test_pooled_outputs_compare_each_form_with_the_same_form_of_the_reference(record):
    from hxex.report.analysis import _pooled_form_outputs
    from hxex.report.coverage import sample_styles

    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, ACETYL), record)
    up = uptake([(s, k, 60.0, fd + (0.2 if s == "V" and k == ONLY_X1 else 0.0))
                 for s in ("R", "V")
                 for k, fd in ((SHARED, 0.3), (ONLY_X1, 0.4), (ONLY_X2, 0.5), (ACETYL, 0.6))])
    k1 = _comparison([(SHARED, 0.0, False), (ONLY_X1, 0.2, True), (ONLY_X2, 0.0, False),
                      (ACETYL, 0.0, False)])
    tables: dict = {}
    jobs, top = _pooled_form_outputs(
        {"": (forms, record, DEFAULT_MODIFICATIONS)}, up, {("", "R"), ("", "V")}, tables,
        AnalysisConfig(), labels={("", "R"): "R", ("", "V"): "V"},
        styles=sample_styles(["R", "V"], "R"), order=["R", "V"], ref_state=("", "R"),
        k1=k1, times=[60.0], scale="seconds")
    assert list(tables["pooled_forms"]["form"]) == ["X1", "X2", "X1 acK17"]
    assert set(tables["residues"]["sample"]) == {"R X1", "R X2", "R X1 acK17", "V X1", "V X2",
                                                 "V X1 acK17"}
    cmp = tables["compare_residues"]
    assert set(cmp["sample"]) == {"V X1", "V X2", "V X1 acK17"}  # each against the same form
    called = {(s, int(r)) for s, r, sig in zip(cmp["sample"], cmp["residue"],
                                                cmp["significant"], strict=True) if sig}
    # X1's own peptide (14-19 exchanging) moved and its comparison is significant; the shared
    # peptide and the other forms did not
    assert {r for s, r in called if s == "V X1"} == {14, 15, 16, 17, 18, 19}
    assert not {s for s, _ in called} - {"V X1"}
    assert [stem for stem, _ in jobs if stem.startswith("pooled_")] == ["pooled_forms_X_1min"]
    # and each form's population choice per sample, provisional (owner, 2026-11-10)
    assert {stem for stem, _ in jobs if stem.startswith("choice_")} == {
        f"choice_{s} {f}_1min" for s in ("R", "V") for f in ("X1", "X2", "X1 acK17")}
    assert top == pytest.approx(0.6 + 0.01)  # the highest residue value with its error
    fig = jobs[0][1]()
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == ["X1: 1 of its 2 peptides are its own",
                      "X2: 1 of its 2 peptides are its own",
                      "X1 acK17: 1 of its 2 peptides are its own"]
    strips = [p for p in fig.axes[0].patches if p.get_facecolor()[:3] == pytest.approx(
        (0.91, 0.91, 0.91))]
    assert len(strips) == 1  # X1's own residues, 14 to 19
    box = strips[0].get_path().transformed(strips[0].get_patch_transform()).get_extents()
    assert (box.x0, box.x1) == pytest.approx((13.5, 19.5))
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == ["R", "V", "measured by this form's own peptides"]


def test_a_modified_form_whose_peptides_fit_other_records_names_them(tmp_path):
    from hxex.analysis import residue_map
    from hxex.report.pooled import plot_pooled_forms

    (tmp_path / "pool.fasta").write_text(POOL + ">X3\nMAKLPGSTVEDRWQAFKLA\n")
    record = candidates_record(tmp_path / "pool.fasta", ["X", "Y"])
    # WQAFacKL (13-18) fits X1 and X3; X1 is in the set (its own WQAFKLT), X3 is not needed
    acetyl = ("WQAFacKL", 13, 18)
    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, acetyl), record)
    form = forms.form("X1 acK17")
    assert form.candidates == ("X1", "X3") and form.members == ("X1",)
    up = uptake([("R", k, 60.0, 0.3) for k in (SHARED, ONLY_X1, ONLY_X2, acetyl)])
    fu, _ = form_uptake(up, {"": forms})
    fig = plot_pooled_forms(residue_map(fu, modified="combined"), [form], ["R"], 60.0)
    assert fig.axes[0].get_title(loc="left").endswith("(the modified peptides also fit X3)")


def test_a_forms_class_follows_the_classes_named_by_their_records(tmp_path):
    (tmp_path / "pool.fasta").write_text(POOL)
    record = candidates_record(tmp_path / "pool.fasta", {"first": ["X1"], "second": ["X2", "Y1"]})
    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2), record)
    assert [(f.name, form_class(f, record)) for f in forms.forms] == [("X1", "first"),
                                                                       ("X2", "second")]
    one = pooled_forms(keys(SHARED), record)  # X1 or X2: one form across two classes
    assert [(f.name, form_class(f, record)) for f in one.forms] == [("X1 or X2", "")]


def test_head_to_head_panels_put_one_class_sequences_of_a_range_side_by_side(tmp_path):
    from hxex.analysis.pooled import head_to_head_panels, sequence_difference

    (tmp_path / "pool.fasta").write_text(POOL + ">Y2\nMAKLPGSTVEDRWQAFKLT\n")
    # Y2 has X1's sequence but is of class Y: WQAFKLT then fits two classes and is left out
    record = candidates_record(tmp_path / "pool.fasta", {"X": ["X1", "X2"], "Y": ["Y1", "Y2"]})
    assert sequence_difference("WQAFKLT", "WQAYKLT", 13) == "F16Y"
    assert sequence_difference("WQAFKLT", "WQAFacKLT", 13) == "acK17"
    assert sequence_difference("WQAFKLT", "WQAFKLT", 13) == ""
    record_x = candidates_record(tmp_path / "pool.fasta", {"X": ["X1", "X2", "Y2"]})
    panels = head_to_head_panels([SHARED, ONLY_X1, ONLY_X2, ACETYL], record_x)
    # the X sequences at 13-19: the unmodified class sequence first, then X2's, then acK17
    assert [(p.name, p.group, p.sequences) for p in panels] == [
        ("13-19", "X", ("WQAFKLT", "WQAYKLT", "WQAFacKLT"))]
    assert panels[0].labels == ("X", "X2: F16Y", "X: acK17")
    # with Y2 in another class, X1's sequence (also Y2's) fits two classes and leaves; X2's
    # sequence is then alone at its range, an isoform: a panel of its own
    panels = head_to_head_panels([SHARED, ONLY_X1, ONLY_X2], record)
    assert [(p.name, p.sequences) for p in panels] == [("13-19", ("WQAYKLT",))]
    # a list of ranges: those ranges' groups, whatever their size; false: none
    listed = head_to_head_panels([SHARED, ONLY_X1, ONLY_X2], record_x, ["2-9"])
    assert [(p.name, p.sequences) for p in listed] == [("2-9", ("AKLPGSTV",))]
    assert head_to_head_panels([SHARED], record_x, False) == []


def test_the_head_to_head_setting_takes_auto_false_or_ranges():
    assert AnalysisConfig().head_to_head == "auto"
    assert AnalysisConfig(head_to_head=True).head_to_head == "auto"
    assert AnalysisConfig(head_to_head=["27-52", "1-9"]).head_to_head == ("27-52", "1-9")
    assert AnalysisConfig(head_to_head="27-52").head_to_head == ("27-52",)
    with pytest.raises(ValueError, match='not a range such as "27-52"'):
        AnalysisConfig(head_to_head=["27"])
    with pytest.raises(ValueError, match="ends before it starts"):
        AnalysisConfig(head_to_head=["52-27"])


def test_the_head_to_head_figure_shares_its_axis_and_marks_the_first_sequence(record):
    from hxex.analysis.pooled import head_to_head_panels
    from hxex.report.coverage import sample_styles
    from hxex.report.hover import row_of
    from hxex.report.pooled import STAND_IN_ALPHA, plot_head_to_head

    panels = head_to_head_panels([SHARED, ONLY_X1, ONLY_X2, ACETYL], record)
    up = uptake([(s, k, 60.0, fd) for s in ("R", "V")
                 for k, fd in ((ONLY_X1, 0.4), (ONLY_X2, 0.5), (ACETYL, 0.6))])
    fig = plot_head_to_head(up, panels, ["R", "V"], 60.0, styles=sample_styles(["R", "V"], "R"),
                            top=0.9)
    ax = fig.axes[0]
    assert ax.get_ylim() == pytest.approx((0.0, 0.9))
    assert ax.get_title(loc="left") == "13-19 (X)"
    # X1's sequence first (FASTA order among isoforms, unmodified before modified)
    labels = ["X1", "X2: F16Y", "X1: acK17"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == labels
    rows = [row_of(line) for line in ax.lines if row_of(line)]
    assert len(rows) == 6 and {r["sequence"] for r in rows} == set(labels)
    first = [line for line in ax.lines if (row_of(line) or {}).get("sequence") == "X1"]
    others = [line for line in ax.lines if (row_of(line) or {}).get("sequence") == "X2: F16Y"]
    assert first[0].get_markerfacecolor() != "white" and others[0].get_markerfacecolor() == "white"
    # a sequence corrected by a stand-in control: drawn pale, said in the legend and hover
    panel = panels[0]
    pale = {("V", panel.sequences[1], panel.start, panel.end)}
    fig = plot_head_to_head(up, panels, ["R", "V"], 60.0, styles=sample_styles(["R", "V"], "R"),
                            top=0.9, stand_ins=pale)
    ax = fig.axes[0]
    drawn = {(r["sample"], r["peptide"]): line for line in ax.lines if (r := row_of(line))}
    assert drawn[("V", panel.sequences[1])].get_alpha() == pytest.approx(STAND_IN_ALPHA)
    assert row_of(drawn[("V", panel.sequences[1])])["controls"] == "stand-in: not compared"
    assert drawn[("R", panel.sequences[1])].get_alpha() in (None, 1.0)
    assert ax.get_legend().get_texts()[-1].get_text() == "pale: stand-in control, not compared"


def test_each_form_has_a_coverage_map_with_its_own_peptides_outlined(record):
    from hxex.report.analysis import _pooled_coverage_jobs
    from hxex.report.coverage import plot_coverage

    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2, ACETYL), record)
    up = uptake([(s, k, 60.0, 0.4) for s in ("R", "V")
                 for k in (SHARED, ONLY_X1, ONLY_X2, ACETYL)])
    jobs = _pooled_coverage_jobs({"": (forms, record, DEFAULT_MODIFICATIONS)}, up,
                                 {("", "R"), ("", "V")}, AnalysisConfig(),
                                 labels={("", "R"): "R", ("", "V"): "V"}, order=["R", "V"],
                                 times=[60.0], scale="seconds")
    assert [stem for stem, _ in jobs] == ["pooled_coverage_X1", "pooled_coverage_X2",
                                          "pooled_coverage_X1_acK17"]
    fig = jobs[0][1]()
    assert fig.get_suptitle() == "X1: its own peptides outlined in black and white"
    bars = {p.get_gid(): p for ax in fig.axes for p in ax.patches if p.get_gid()}
    assert set(bars) == {"peptide:2-9:AKLPGSTV", "peptide:13-19:WQAFKLT"}
    assert bars["peptide:13-19:WQAFKLT"].get_linewidth() > bars[
        "peptide:2-9:AKLPGSTV"].get_linewidth()  # X1's own peptide, outlined
    # two sequences over one range in one sample: two bars, each named by its sequence
    both = plot_coverage(up, "R", 60.0)
    gids = sorted(p.get_gid() for p in both.axes[0].patches if p.get_gid())
    assert gids == ["peptide:13-19:WQAFKLT", "peptide:13-19:WQAFacKLT",
                    "peptide:13-19:WQAYKLT", "peptide:2-9:AKLPGSTV"]


def test_each_form_has_its_population_choice_and_k2_comparison_marked_provisional(record):
    """The provisional bimodal analysis of the forms (owner, 2026-11-10): each form's
    population choice, its comparison with the same form of the reference at K = 2 (the
    choices and the share), the profile, overlay and K = 2 map pages, every page marked
    provisional, the profiles with the strip of the form's own residues."""
    from hxex.report.analysis import _pooled_form_outputs
    from hxex.report.coverage import sample_styles
    from hxex.report.pooled import OWN_FILL

    forms = pooled_forms(keys(SHARED, ONLY_X1, ONLY_X2), record)
    base = uptake([(s, k, 60.0, 0.5) for s in ("R", "V") for k in (SHARED, ONLY_X1, ONLY_X2)])
    # every peptide also at K = 2: two populations at 0.2 and 0.8, V's high-D share larger
    k2 = []
    for i in range(len(base)):
        for rank, fd in ((1, 0.2), (2, 0.8)):
            share = 0.5 + (0.2 if base.sample[i] == "V" else 0.0) * (1 if rank == 2 else -1)
            k2.append((i, rank, fd, share))
    cols = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)
            if isinstance(getattr(base, f.name), np.ndarray)}
    idx = np.concatenate([np.arange(len(base)), [i for i, *_ in k2]])
    cols = {name: values[idx] for name, values in cols.items()}
    n1 = len(base)
    cols["K"][n1:], cols["rank"][n1:] = 2, [r for _, r, _, _ in k2]
    cols["fD"] = cols["fD"].astype(float)
    cols["fD"][n1:] = [fd for _, _, fd, _ in k2]
    cols["mean"] = cols["fD"] * 5
    cols["population"] = cols["population"].astype(float)
    cols["population"][n1:] = [p for *_, p in k2]
    cols["population_se"] = np.full(len(idx), 0.02)
    up = dataclasses.replace(base, **cols)
    k1 = _comparison([(k, 0.0, False) for k in (SHARED, ONLY_X1, ONLY_X2)])
    comparisons = {"K=1 fD": k1, "K=2 rank 1 (low D) fD": k1, "K=2 rank 2 (high D) fD": k1,
                   "K=2 rank 1 (low D) population": k1}
    tables: dict = {}
    jobs, _ = _pooled_form_outputs(
        {"": (forms, record, DEFAULT_MODIFICATIONS)}, up, {("", "R"), ("", "V")}, tables,
        AnalysisConfig(populations="bootstrap"), labels={("", "R"): "R", ("", "V"): "V"},
        styles=sample_styles(["R", "V"], "R"), order=["R", "V"], ref_state=("", "R"), k1=k1,
        times=[60.0], scale="seconds", comparisons=comparisons)
    assert set(tables["populations"]["sample"]) == {"R X1", "R X2", "V X1", "V X2"}
    assert set(tables["compare_choices"]["sample"]) == {"V X1", "V X2"}
    assert set(tables["compare_populations"]["sample"]) == {"V X1", "V X2"}
    # no bootstrap of every spectrum at K = 2: the share's test is left empty (HISTORY 204)
    assert np.isnan(np.asarray(tables["compare_populations"]["z"], dtype=float)).all()
    stems = {stem: job for stem, job in jobs}
    assert {"choice_R X1_1min", "choice_V X2_1min", "overlay_populations_X1_1min",
            "residue_significance_K2_V X1"} <= set(stems)
    for stem in ("choice_V X1_1min", "overlay_populations_X1_1min",
                 "residue_significance_K2_V X1"):
        fig = stems[stem]()
        notes = [t.get_text() for t in fig.texts if t.get_text().startswith("Provisional")]
        assert len(notes) == 1, stem
        strips = [p for ax in fig.axes for p in ax.patches
                  if p.get_facecolor()[:3] == pytest.approx((0.91, 0.91, 0.91))]
        if stem.startswith("residue_significance"):  # a map: the residues in the note
            assert not strips and "outside residues 14-19" in notes[0].replace("\n", " ")
        else:
            assert strips and "grey strip" in notes[0], stem
    assert OWN_FILL == "0.91"


def test_a_pools_candidates_go_with_its_instrument_files(tmp_path):
    """[input] pooled_fasta with the scan formats (owner, 2026-11-10): the instrument files
    behind a pooled export carry the pool's candidates, so the identities, the forms and the
    matching by sequence follow; the folder formats other than pooled still refuse it."""
    from types import SimpleNamespace

    from hxex.analysis.settings import resolve_match
    from hxex.config import ProjectError, load_project, read_input
    from hxex.io.pooled import CANDIDATES_RECORD
    from test_elution_windows import PEPTIDE, _project

    path = _project(tmp_path)
    (tmp_path / "pool.fasta").write_text(f">A1\n{PEPTIDE}GG\n>A2\n{PEPTIDE}SS\n>B1\nMMMM\n")
    text = path.read_text().replace('peptides = "peptides.csv"\n', 'peptides = "peptides.csv"\n'
                                    'pooled_fasta = "pool.fasta"\npooled_classes = ["A", "B"]\n')
    path.write_text(text)
    spectra = read_input(load_project(path))
    record = spectra.source[CANDIDATES_RECORD]
    assert [name for name, _ in record["records"]] == ["A1", "A2", "B1"]
    run = SimpleNamespace(settings=SimpleNamespace(spectra_source=spectra.source))
    assert resolve_match("auto", {"": run}) == "sequence"
    plain = SimpleNamespace(settings=SimpleNamespace(spectra_source={"reader": "mzml"}))
    assert resolve_match("auto", {"": plain}) == "range"
    folder = tmp_path / "folder.toml"
    folder.write_text('project = "p"\n\n[input]\nformat = "specexport"\ndata_dir = "."\n'
                      'pooled_fasta = "pool.fasta"\n')
    with pytest.raises(ProjectError, match="for pooled data"):
        load_project(folder)
