# SPDX-License-Identifier: GPL-3.0-or-later
"""Where a project's results stand, stage by stage (hxex.stages; docs/PLAN.md Next steps 40):
the manifest's stages block, each cause of a stage out of date, the chain after it, and what
hxex run would do, as hxex check and hxex info print them."""

import dataclasses
import json

from hxex import cli
from hxex.io import open_bundle
from hxex.pipeline import RunConfig
from hxex.stages import (
    DONE,
    NOT_DONE,
    NOT_RESULTS,
    OFF,
    OUT_OF_DATE,
    STAGES,
    _stage_of,
    current_codes,
    record,
    standing,
)
from test_cli import mzml_project  # noqa: I001


def _project(path):
    from hxex.config import load_project

    return load_project(path, analysis_only=True)


def _made(tmp_path, output="[run.controls]\nd2o_fraction = 0.9"):
    path = mzml_project(tmp_path, output=output)
    assert cli.main(["-q", "run", str(path)]) == 0
    return path


def _edit_manifest(bundle, change):
    manifest = bundle / "manifest.json"
    data = json.loads(manifest.read_text())
    change(data)
    manifest.write_text(json.dumps(data))


def _stage_lines(out: str) -> list[str]:
    lines = out.splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith("stages"))
    return [x for x in lines[start:] if not x.startswith(("ok", "not ok"))]


def test_the_manifest_records_each_stage_and_a_fresh_run_is_done(tmp_path, capsys):
    path = _made(tmp_path)
    with open_bundle(tmp_path / "out") as b:
        block = b.manifest["stages"]
        assert block == record(b.manifest)  # worked out from the manifest, the same
    now = current_codes()
    assert list(block) == list(STAGES)
    assert block["pick"]["code"] == now["pick"] and block["correction"]["code"] == now[
        "correction"]
    assert block["fit"]["code"] == now["fit"] and block["read"]["code"] == now["read"]
    assert block["population_test"]["state"] == "off" and block["bootstrap"]["state"] == "done"
    st = standing(_project(path), None)
    assert {s.name: s.state for s in st.stages} == {
        "read": DONE, "pick": DONE, "noise": OFF, "fit": DONE, "population_test": OFF,
        "correction": DONE, "bootstrap": DONE}
    assert st.up_to_date
    assert st.run.startswith("reads spectra stored in out, keeps peak picking, keeps fits")
    capsys.readouterr()
    cli.main(["check", str(path)])  # a problem of its own: the controls marked nowhere
    lines = _stage_lines(capsys.readouterr().out)
    assert lines[0] == "stages         read: done"
    assert "hxex run       nothing to do: the results are up to date" in lines


def test_each_cause_is_named_and_what_comes_after_it_follows(tmp_path):
    path = _made(tmp_path)
    plain = path.read_text()

    def stand(text):
        path.write_text(text)
        return standing(_project(path), None)

    st = stand(plain.replace("k_max = 1", "k_max = 2"))
    assert st["fit"].state == OUT_OF_DATE and st["fit"].causes == ["[run.fit] k_max: 2, was 1"]
    assert st["bootstrap"].causes == ["after the fit"]
    assert st["correction"].causes == ["after the fit"] and st["pick"].state == DONE
    assert "keeps fits, fits missing K (k_max 2), runs bootstrap for changed spectra" in st.run
    st = stand(plain.replace("adam_steps = 60", "adam_steps = 70"))
    assert "fits all spectra, runs bootstrap" in st.run
    st = stand(plain.replace("n_boot = 3", "n_boot = 4"))
    assert st["bootstrap"].causes == ["[run.bootstrap] n_boot: 4, was 3"]
    assert st["fit"].state == DONE and "keeps fits, runs bootstrap" in st.run
    # a role: the fit and the correction, and run picks again (the roles of the spectra differ)
    st = stand(plain.replace("d2o_fraction = 0.9", "d2o_fraction = 0.9\nfulldeut_at = 60"))
    words = "[run.controls] fulldeut_at: 1 min, was unset (which spectra are controls)"
    assert st["fit"].causes == [words] and st["correction"].causes == [words]
    assert st["pick"].state == DONE and "peak picks" in st.run
    st = stand(plain.replace("d2o_fraction = 0.9", "d2o_fraction = 0.8"))
    assert st["correction"].causes == ["[run.controls] d2o_fraction: 0.8, was 0.9"]
    assert st["fit"].state == DONE and "keeps fits" in st.run
    st = stand(plain.replace("[[input.runs]]", "[[input.exclude]]\ntime = 60\n\n[[input.runs]]"))
    assert st["fit"].causes == ["[[input.exclude]]: 1 rule(s), was 0"]
    st = stand(plain.replace("[run.controls]\nd2o_fraction = 0.9", "[run]\ncontrols = false"))
    assert st["correction"].causes == ["[run] controls: false, was a back-exchange correction"]
    # the bootstrap no longer asked for, and the device (where it runs) not a change
    st = stand(plain.replace('device = "cpu", dtype', 'device = "auto", dtype'))
    assert all(s.state in (DONE, OFF) for s in st.stages)
    # a data file changed: the read, and every stage after it
    path.write_text(plain)
    run = tmp_path / "run1.mzML"
    run.write_bytes(run.read_bytes() + b"\n")
    st = standing(_project(path), None)
    assert st["read"].causes == ["1 data file changed: run1.mzML"]
    assert st["pick"].causes == ["after the read"] and st["fit"].causes == ["after the pick"]
    assert st.run.startswith("reads the data files, peak picks")


def test_a_stopped_bootstrap_is_not_done_and_nothing_follows_it(tmp_path):
    path = _made(tmp_path)
    _edit_manifest(tmp_path / "out", lambda m: m.update(pending=["bootstrap"]))
    st = standing(_project(path), None)
    assert st["bootstrap"].state == NOT_DONE
    assert st["bootstrap"].causes == ["stopped partway: hxex run continues it"]
    assert st["fit"].state == DONE and st["correction"].state == DONE
    assert "keeps fits, finishes bootstrap (it stopped partway)" in st.run


def test_no_results_yet(tmp_path, capsys):
    path = mzml_project(tmp_path)
    st = standing(_project(path), None)
    assert all(s.state in (NOT_DONE, OFF) for s in st.stages) and st.run == "does every stage"
    assert st["read"].causes == ["no results yet"]
    assert cli.main(["check", str(path)]) == 0
    lines = _stage_lines(capsys.readouterr().out)
    assert "hxex run       does every stage" in lines


def test_the_code_of_a_stage(tmp_path, capsys):
    """Another hxex's picking or correction code; a bundle written before the code was recorded
    (its stages judged on their settings, said once); and the run, which keeps stored picking
    only when picked by this hxex's picking code (the version number alone stays the same
    between commits)."""
    path = _made(tmp_path)
    bundle = tmp_path / "out"

    def other_code(m):
        m["stages"]["pick"]["code"] = "0" * 16
        m["stages"]["correction"]["code"] = "1" * 16

    _edit_manifest(bundle, other_code)
    st = standing(_project(path), None)
    assert st["pick"].causes == ["another version of hxex's picking code made it"]
    assert st["correction"].causes == ["another version of hxex's correction code made it"]
    assert st["fit"].causes == ["after the pick"] and "peak picks" in st.run
    _edit_manifest(bundle, lambda m: m.pop("stages"))
    st = standing(_project(path), None)
    assert st["pick"].causes == ["made before hxex recorded its picking code"]
    assert st["correction"].causes == ["made before hxex recorded its correction code"]
    assert "peak picks" in st.run and not st.up_to_date
    capsys.readouterr()
    assert cli.main(["run", str(path)]) == 0
    assert "the picking code (not recorded by the hxex that picked them)" in capsys.readouterr(
    ).err
    assert cli.main(["run", str(path)]) == 0  # recorded now: nothing to do
    assert "the results are up to date" in capsys.readouterr().out
    # the fitting code: every fitting stage from the first one on
    _edit_manifest(bundle, lambda m: m["fitted_with"].update(engine="0" * 16))
    st = standing(_project(path), None)
    assert st["fit"].state == OUT_OF_DATE and st["fit"].causes[0].startswith(
        "hxex's fitting code differs from the one that made it")
    assert "fits all spectra" in st.run


def test_info_and_an_analysis_only_project_file(tmp_path, capsys):
    _made(tmp_path)
    capsys.readouterr()
    assert cli.main(["info", str(tmp_path / "out")]) == 0
    lines = _stage_lines(capsys.readouterr().out)
    assert "               bootstrap: done" in lines
    assert not any(x.startswith("hxex run") for x in lines)  # the bundle alone: no project
    # a project file for the analysis is compared only in the [run] tables it writes
    other = tmp_path / "analysis.toml"
    other.write_text('[output]\nbundle = "out"\n[analysis]\nreference = "WT"\n'
                     "[run.bootstrap]\nn_boot = 4\nn_restarts = 1\nadam_steps = 30\n"
                     "polish_steps = 0\n")
    st = standing(_project(other), None)
    assert st["bootstrap"].causes == ["[run.bootstrap] n_boot: 4, was 3"]
    assert all(s.state in (DONE, OFF) for s in st.stages if s.name != "bootstrap")
    assert st.run == ""  # no [input]: hxex run cannot run it


def test_every_run_setting_belongs_to_a_stage():
    """A [run] setting added later is placed in a stage here (else it counts as the fit's)."""
    placed = {"fit", "n_max_factor", "picker", "envelope_fallback_time", "envelope", "registry",
              "drop_no_signal", "population_test", "quality", "bootstrap", "controls",
              *NOT_RESULTS}
    for f in dataclasses.fields(RunConfig):
        assert f.name in placed, f.name
    assert _stage_of("fit.noise.estimate") == "noise" and _stage_of("fit.k_max") == "fit"
    assert _stage_of("envelope_fallback_time") == "pick" and _stage_of("envelope.mode") == "pick"


def test_the_plan_by_stage_and_the_quick_pass(tmp_path):
    """What hxex run does, part by part (the window's tooltips), joins to the line hxex check
    prints; the quick pass (slow=False) agrees with the whole on every stage."""
    path = _made(tmp_path)
    path.write_text(path.read_text().replace("k_max = 1", "k_max = 2"))
    whole = standing(_project(path), None)
    quick = standing(_project(path), None, slow=False)
    assert whole.complete and not quick.complete
    assert ", ".join(text for _, text in whole.plan) == whole.run
    assert {stage for stage, _ in whole.plan} <= set(STAGES)
    assert whole.plan_of("fit") == ["keeps fits", "fits missing K (k_max 2)"]
    assert quick.run == whole.run
    assert [(s.name, s.state, s.causes) for s in quick.stages] == \
        [(s.name, s.state, s.causes) for s in whole.stages]


def test_the_outputs_made_from_the_results(tmp_path):
    """The tables and analysis folders, each made from the results now or not (the window's
    marks after the stages), as hxex check words them."""
    from hxex.stages import output_marks, output_states

    path = mzml_project(tmp_path, output="[analysis]\nreference = \"A\"")
    project = _project(path)
    marks = {m.name: m for m in output_marks(output_states(project))}
    assert marks["tables"].state == NOT_DONE and marks["analysis"].state == NOT_DONE
    assert cli.output_lines(project) == ["none yet: hxex run writes out"]
    assert cli.main(["-q", "run", str(path)]) == 0
    marks = {m.name: m for m in output_marks(output_states(project))}
    assert marks["tables"].state == DONE and "figures" not in marks
    assert marks["analysis"].causes == ["out_analysis_A: none yet (hxex analyse writes it)"]
    lines = cli.output_lines(project)
    assert lines[:2] == ["out exists (hxex run writes it)",
                         "out_tables: up to date (hxex run and hxex report write it)"]
    # a table edited by hand: said in the mark, and in hxex check's words
    uptake = tmp_path / "out_tables" / "uptake.csv"
    uptake.write_text(uptake.read_text() + "a note\n")
    marks = {m.name: m for m in output_marks(output_states(project))}
    assert "1 file(s) changed since hxex wrote them" in marks["tables"].causes[0]
    assert "changed since hxex wrote it (uptake.csv)" in cli.output_lines(project)[1]


def test_kept_checksums_follow_a_changed_file(tmp_path):
    from hxex.io import identity

    data = tmp_path / "data.bin"
    data.write_bytes(b"a" * 100)
    identity.keep_checksums(True)
    try:
        first = identity.file_identity(data)["checksum"]
        assert identity.file_identity(data)["checksum"] == first
        data.write_bytes(b"b" * 100)  # the same size: the modification time tells
        import os

        os.utime(data, ns=(1, 2))
        assert identity.file_identity(data)["checksum"] != first
    finally:
        identity.keep_checksums(False)
    assert identity._KEPT is None
