# SPDX-License-Identifier: GPL-3.0-or-later
"""The GUI's parts without Qt: the session file and saving figures (hxex.gui.session,
hxex.gui.figures)."""
import logging

import pytest
from matplotlib.figure import Figure

from hxex.gui.figures import figure_ids, figure_svg, save_figure
from hxex.gui.session import (
    VERSION,
    FigureState,
    GuiSession,
    SessionError,
    load_session,
    save_session,
    session_text,
)


def test_a_session_round_trips_with_paths_relative_to_its_file(tmp_path):
    folder = tmp_path / "project"
    (folder / "results").mkdir(parents=True)
    (folder / "hxex.toml").write_text("")
    elsewhere = tmp_path / "other" / "bundle.zip"
    session = GuiSession(
        project=folder / "hxex.toml", results=elsewhere, workspace="results",
        selection={"sample": "WT", "peptide_range": "1-23", "charge": 3},
        figures=[FigureState("uptake", {"sample": "W\"T", "times": [30.0, 600.0]},
                             [{"change": "title", "text": "two\nlines é"},
                              {"change": "xlim", "range": [0, 1e5], "log": False}])],
        layout={"geometry": "AdnQywAD"})
    path = save_session(session, folder / "view.hxexgui")
    text = path.read_text()
    assert 'project = "hxex.toml"' in text  # inside the session's folder: relative
    assert elsewhere.as_posix() in text  # outside it: as given
    assert "[review" not in text and "[run" not in text  # the view only
    back = load_session(path)
    assert back == session
    # the folder moved with its session: the paths follow
    moved = tmp_path / "moved"
    folder.rename(moved)
    assert load_session(moved / "view.hxexgui").project == moved / "hxex.toml"


def test_a_session_says_what_no_longer_exists(tmp_path):
    session = GuiSession(project=tmp_path / "gone.toml", results=tmp_path)
    assert session.missing() == [f"the project file {tmp_path / 'gone.toml'} no longer exists"]
    with pytest.raises(ValueError, match="workspace"):
        GuiSession(workspace="review")


def test_a_newer_or_broken_session_is_refused_and_unknown_keys_left_out(tmp_path, caplog):
    path = tmp_path / "s.hxexgui"
    path.write_text(f"version = {VERSION + 1}\n")
    with pytest.raises(SessionError, match="newer hxex"):
        load_session(path)
    path.write_text("version = \n")
    with pytest.raises(SessionError, match="not a session file"):
        load_session(path)
    path.write_text('workspace = "results"\n')
    with pytest.raises(SessionError, match="no version"):
        load_session(path)
    path.write_text(f'version = {VERSION}\nworkspace = "elsewhere"\ncolour = "red"\n'
                    '[[figures]]\nselection = {}\n')
    with caplog.at_level(logging.WARNING, logger="hxex"):
        session = load_session(path)
    assert session.workspace == "project" and session.figures == []
    for word in ("colour", "elsewhere", "without a kind"):
        assert word in caplog.text


def test_session_text_writes_toml_for_awkward_values(tmp_path):
    import tomllib

    session = GuiSession(selection={"odd key": float("nan"), "flag": True, "n": [1, 2.5]})
    data = tomllib.loads(session_text(session, tmp_path))
    assert data["selection"]["flag"] is True and data["selection"]["n"] == [1, 2.5]
    assert data["selection"]["odd key"] != data["selection"]["odd key"]  # nan


def _figure():
    fig = Figure(figsize=(3, 2))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], gid="line")
    ax.bar([0.5], [0.5], gid="bar", hatch="//")
    return fig


def test_figures_save_as_svg_pdf_and_png(tmp_path):
    fig = _figure()
    assert figure_svg(fig) == figure_svg(fig)  # no date: the same figure, the same bytes
    assert set(figure_ids(fig)) == {"line", "bar"}
    for suffix, start in ((".svg", b"<?xml"), (".pdf", b"%PDF"), (".PNG", b"\x89PNG")):
        path = save_figure(fig, tmp_path / f"f{suffix}")
        assert path.read_bytes().startswith(start)
    with pytest.raises(ValueError, match=r"\.svg, \.pdf, \.png"):
        save_figure(fig, tmp_path / "f.jpg")


def test_the_window_icons_are_the_logo_files():
    """The package's copies of the logo (the window's icon) are docs/images' (make_logo.py
    writes both)."""
    from pathlib import Path

    from hxex.gui import icon  # noqa: F401 - imports without a display

    root = Path(__file__).resolve().parents[1]
    for name in ("hxex_logo.svg", "hxex_logo_dark.svg"):
        assert ((root / "src/hxex/gui/icons" / name).read_bytes()
                == (root / "docs/images" / name).read_bytes()), name
