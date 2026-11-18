# SPDX-License-Identifier: GPL-3.0-or-later
"""Groups of samples in the analysis (``[analysis.groups]``, docs/ANALYSIS.md "Groups of
samples"): the settings, one analysis folder per group with the group's own regions and sites,
and what ``hxex check`` and ``hxex analyse`` say."""

import csv
import tomllib

import pytest

from hxex import cli
from hxex.analysis import AnalysisConfig, AnalysisGroup
from hxex.config import ProjectError, load_project, project_from
from test_cli import FAST_RUN  # noqa: I001
from test_mzml import PEPTIDE, time_course, write_mzml

INPUT = 'project = "p"\n[input]\nformat = "specexport"\ndata_dir = "d"\n'


def analysis(text: str, tmp_path):
    return project_from(tomllib.loads(INPUT + text), tmp_path).analysis


def test_groups_are_read_with_their_labels_regions_and_sites(tmp_path):
    a = analysis("""
[analysis]
ks = [1]
overlay = ["avg"]
[[analysis.regions]]
name = "shared"
start = 1
end = 5
[[analysis.sites]]
residue = 3
label = "S3"
[analysis.groups.A]
reference = "A1"
samples = ["A1", "A2"]
[analysis.groups.B]
reference = "B_alone"
samples = { alone = "B_alone", "A+B" = "B_in_AB" }
sites = false
[[analysis.groups.B.regions]]
name = "N-terminal region"
start = 2
end = 8
""", tmp_path)
    assert a.by_group and not a.single_reference and a.analysis_names == ("A", "B")
    assert a.groups["B"].labels == ("alone", "A+B") and a.groups["A"].labels == ("A1", "A2")
    assert a.groups["B"].label_of("B_in_AB") == "A+B"
    one = a.one("B")
    assert one.single_reference and one.group == "B"
    assert (one.reference, one.samples) == ("B_alone", ("B_alone", "B_in_AB"))
    # its own regions replace the shared ones, sites = false gives none; never merged
    assert [r.name for r in one.annotations.regions] == ["N-terminal region"]
    assert one.annotations.sites == ()
    # a group without its own takes the shared ones
    assert [r.name for r in a.one("A").annotations.regions] == ["shared"]
    assert [s.label for s in a.one("A").annotations.sites] == ["S3"]
    # to_dict and back
    again = AnalysisConfig.from_dict(a.to_dict())
    assert again.groups == a.groups and again.one("B").annotations == one.annotations


@pytest.mark.parametrize("text, message", [
    ('[analysis]\nreference = "A1"\n[analysis.groups.A]\nreference = "A1"\nsamples = ["A1"]\n',
     r"reference and samples go in each group"),
    ('[analysis]\nsamples = ["A1"]\n[analysis.groups.A]\nreference = "A1"\nsamples = ["A1"]\n',
     r"reference and samples go in each group"),
    ('[analysis.groups.A]\nreference = "Z"\nsamples = ["A1"]\n',
     r"\[analysis.groups.A\] the reference 'Z' is not among the group's samples"),
    ('[analysis.groups.A]\nsamples = ["A1"]\n', r"\[analysis.groups.A\] needs reference"),
    ('[analysis.groups.A]\nreference = "A1"\nsamples = ["A1"]\ncolour = "red"\n',
     r"\[analysis.groups.A\].*colour"),
    ('[analysis.groups.A]\nreference = "A1"\nsamples = "A1"\n',
     r"samples: a list of sample names"),
    ('[analysis.groups.A]\nreference = "A1"\nsamples = ["A1", "A1"]\n',
     r"the sample 'A1' is named twice"),
    ('[analysis.groups.A]\nreference = "A1"\nsamples = ["A1"]\n'
     '[[analysis.groups.A.regions]]\nname = "x"\nstart = 1\n',
     r"\[\[analysis.groups.A.regions\]\] #1 needs end"),
    ("[analysis]\ngroups = 3\n", r"\[analysis.groups\] holds one table per group"),
])
def test_groups_are_refused_with_the_reason(tmp_path, text, message):
    with pytest.raises(ProjectError, match=message):
        analysis(text, tmp_path)


def test_a_group_from_python():
    g = AnalysisGroup("WT", ("WT", "Mut"))
    assert g.labels == g.samples and g.regions is None
    with pytest.raises(ValueError, match="one label per sample"):
        AnalysisGroup("WT", ("WT", "Mut"), labels=("a",))
    with pytest.raises(ValueError, match="no group 'C'"):
        AnalysisConfig(groups={"A": g}).with_group("C")
    with pytest.raises(ValueError, match="there are no groups"):
        AnalysisConfig(group="A")


def groups_project(tmp_path, analysis_text: str):
    """Three samples, one file each (one peptide over residues 1-10), and ``analysis_text``."""
    runs = ""
    for sample in ("A1", "A2", "B1"):
        write_mzml(tmp_path / f"{sample}.mzML", time_course(n=6))
        runs += f'\n[[input.runs]]\npath = "{sample}.mzML"\nsample = "{sample}"\n'
    (tmp_path / "peptides.csv").write_text(
        "file,sample,peptide,peptide_range,start_seq,end_seq,charge\n"
        f"x,,{PEPTIDE},0001-0010,1,10,2\n")
    path = tmp_path / "hxex.toml"
    path.write_text(f'project = "groups"\n[input]\nformat = "mzml"\npeptides = "peptides.csv"\n'
                    f'{runs}\n[output]\nbundle = "out"\n{FAST_RUN}\n{analysis_text}')
    return path


GROUPS = """
[analysis]
ks = [1]
overlay = ["avg"]
residue_level = true
[[analysis.regions]]
name = "shared"
start = 1
end = 5
[analysis.groups.A]
reference = "A1"
samples = { alone = "A1", "A+B" = "A2" }
[[analysis.groups.A.regions]]
name = "own"
start = 2
end = 8
[analysis.groups.B]
reference = "B1"
samples = ["B1"]
"""


def test_each_group_is_an_analysis_of_its_own(tmp_path, capsys):
    """hxex analyse writes one folder per group (results_analysis_GROUP), each an ordinary
    analysis of its samples against its reference with its own regions (or the shared ones);
    the settings file there repeats it as an analysis of its own; --group chooses groups;
    --reference and a single --out for several groups are refused; hxex check shows the
    groups and the samples in none."""
    path = groups_project(tmp_path, GROUPS)
    assert cli.main(["-q", "run", str(path)]) == 0
    capsys.readouterr()
    assert cli.main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "groups         A: reference A1; samples alone = A1, A+B = A2; 1 regions (its own)" \
        in out
    assert "B: reference B1; samples B1; 1 regions (shared), 0 sites" in out
    assert "in no group" not in out

    assert cli.main(["analyse", str(path), "--reference", "A1"]) == 1
    assert "--group" in capsys.readouterr().err
    assert cli.main(["analyse", str(path), "--out", str(tmp_path / "x")]) == 1
    assert "make a folder each" in capsys.readouterr().err
    assert cli.main(["analyse", str(path), "--group", "C"]) == 1
    assert "no such group" in capsys.readouterr().err

    assert cli.main(["-q", "analyse", str(path)]) == 0
    a, b = tmp_path / "out_analysis_A", tmp_path / "out_analysis_B"

    def regions(folder):
        with open(folder / "regions.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        return {r["region"] for r in rows}, {r["sample"] for r in rows}

    assert regions(a) == ({"own"}, {"A1", "A2"})
    assert regions(b) == ({"shared"}, {"B1"})
    assert (a / "compare_peptides.csv").is_file()
    # the settings used repeat the group's analysis as one of its own, and read back
    text = (a / "analysis_settings_used.toml").read_text()
    assert "The analysis of group A" in text and "alone = A1, A+B = A2" in text
    again = load_project(a / "analysis_settings_used.toml", analysis_only=True).analysis
    assert (again.reference, again.samples, again.groups) == ("A1", ("A1", "A2"), None)
    assert [r.name for r in again.annotations.regions] == ["own"]

    # one group alone; a sample in no group is named by check and analyse
    path.write_text(path.read_text().replace('[analysis.groups.A]\nreference = "A1"\n'
                             'samples = { alone = "A1", "A+B" = "A2" }',
                             '[analysis.groups.A]\nreference = "A1"\nsamples = ["A1"]'))
    assert cli.main(["-q", "analyse", str(path), "--group", "A", "--overwrite"]) == 0
    assert regions(a) == ({"own"}, {"A1"})
    capsys.readouterr()
    assert cli.main(["analyse", str(path), "--group", "A", "--overwrite"]) == 0
    assert "in no group of [analysis.groups], so not analysed: A2" in capsys.readouterr().err
    assert cli.main(["check", str(path)]) == 0
    assert "in no group, so not analysed: A2" in capsys.readouterr().out


def test_check_finds_a_group_sample_that_is_not_read(tmp_path, capsys):
    path = groups_project(tmp_path, GROUPS.replace('samples = ["B1"]', 'samples = ["B1", "B2"]'))
    assert cli.main(["check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "[analysis.groups.B] samples: no sample 'B2' among those read (A1, A2, B1)" in err
