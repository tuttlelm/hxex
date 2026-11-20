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
    assert st.run.startswith("takes the stored spectra, keeps the picking")
    assert "keeps the fits and bootstrap of the spectra whose input is unchanged" in st.run
    capsys.readouterr()
    cli.main(["check", str(path)])  # a problem of its own: the controls marked nowhere
    lines = _stage_lines(capsys.readouterr().out)
    assert lines[0] == "stages         read: done"
    assert any(x.startswith("hxex run       with --overwrite or --keep-previous: takes the "
                            "stored spectra")
               for x in lines)


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
    assert "fits every spectrum again ([run.fit] changed)" in st.run
    st = stand(plain.replace("n_boot = 3", "n_boot = 4"))
    assert st["bootstrap"].causes == ["[run.bootstrap] n_boot: 4, was 3"]
    assert st["fit"].state == DONE and "([run.bootstrap] changed)" in st.run
    # a role: the fit and the correction, and run picks again (the roles of the spectra differ)
    st = stand(plain.replace("d2o_fraction = 0.9", "d2o_fraction = 0.9\nfulldeut_at = 60"))
    words = "[run.controls] fulldeut_at: 1 min, was unset (which spectra are controls)"
    assert st["fit"].causes == [words] and st["correction"].causes == [words]
    assert st["pick"].state == DONE and "picks again" in st.run
    st = stand(plain.replace("d2o_fraction = 0.9", "d2o_fraction = 0.8"))
    assert st["correction"].causes == ["[run.controls] d2o_fraction: 0.8, was 0.9"]
    assert st["fit"].state == DONE and "keeps the fits" in st.run
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
    assert st.run.startswith("reads the input again, picks again")


def test_a_stopped_bootstrap_is_not_done_and_nothing_follows_it(tmp_path):
    path = _made(tmp_path)
    _edit_manifest(tmp_path / "out", lambda m: m.update(pending=["bootstrap"]))
    st = standing(_project(path), None)
    assert st["bootstrap"].state == NOT_DONE
    assert st["bootstrap"].causes == ["stopped partway: hxex resume continues it"]
    assert st["fit"].state == DONE and st["correction"].state == DONE
    assert "fits every spectrum again (a stage stopped partway)" in st.run


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
    assert st["fit"].causes == ["after the pick"] and "picks again" in st.run
    _edit_manifest(bundle, lambda m: m.pop("stages"))
    st = standing(_project(path), None)
    assert st["pick"].state == DONE and st["correction"].state == DONE
    assert st.notes == ["the picking and correction code is not recorded (results made before "
                        "hxex recorded it): those stages are judged on their settings alone"]
    assert "picks again" in st.run
    capsys.readouterr()
    assert cli.main(["run", "--overwrite", str(path)]) == 0
    assert "the picking code (not recorded by the hxex that picked them)" in capsys.readouterr(
    ).err
    assert cli.main(["run", "--overwrite", str(path)]) == 0
    assert "the picking stored in" in capsys.readouterr().err  # recorded now: kept
    # the fitting code: every fitting stage from the first one on
    _edit_manifest(bundle, lambda m: m["fitted_with"].update(engine="0" * 16))
    st = standing(_project(path), None)
    assert st["fit"].state == OUT_OF_DATE and st["fit"].causes[0].startswith(
        "hxex's fitting code differs from the one that made it")
    assert "fits every spectrum again (other fitting code)" in st.run


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
