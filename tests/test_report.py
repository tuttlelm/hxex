# SPDX-License-Identifier: GPL-3.0-or-later
"""Report: the NumPy curves equal the PyTorch model, plot layers and component selection,
and a loaded bundle plots exactly like the run that wrote it."""

import csv
import logging

import numpy as np
import pytest
from matplotlib.figure import Figure

from hxex import chem
from hxex.io import ReviewTable, open_bundle, write_bundle
from hxex.model import Backend
from hxex.model.mixture import MixtureBatch, PaddedBatch, shape_weights
from hxex.report import (
    PlotConfig,
    Reference,
    Results,
    Settings,
    component_curves,
    components_table,
    load_reference,
    plot_populations,
    plot_series_spectra,
    plot_spectrum,
    plot_uptake,
    population_data,
    select_series,
    series_info,
    spectra_table,
    spectrum_data,
    uptake_data,
    uptake_table,
    write_figures,
    write_tables,
)
from hxex.run import RunConfig, run
from test_run import boot_config, config, raw_spectra


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    raw = raw_spectra(n_series=2)
    path = tmp_path_factory.mktemp("rep") / "b"
    res = run(raw, config(bootstrap=boot_config(ks="all", keep_draws=False)), out=path)
    return raw, res, path


@pytest.fixture(scope="module")
def results(written):
    raw, res, _ = written
    return Results.from_run(res, spectra=raw)


def gids(fig):
    return {a.get_gid() for ax in fig.axes for a in ax.get_children() if a.get_gid()}


def test_numpy_curves_equal_the_torch_model(results):
    be = Backend("cpu", "float64")
    m, fit = results.measurements, results.fit
    for k in fit.k_values:
        idx = np.flatnonzero(fit.selected_K == k)
        if not len(idx):
            continue
        kf = fit.fits[k]
        batch = PaddedBatch.from_arrays(
            [m.bins[i] for i in idx], [np.zeros(len(m.bins[i])) for i in idx], m.n_max[idx], be)
        model = MixtureBatch(batch, shape_weights([m.shapes[i] for i in idx], k, be),
                             be.tensor(kf.log_A[idx]), be.tensor(kf.n_raw[idx]),
                             be.tensor(kf.logit_p[idx]), "gaussian")
        pred = model.predict().detach().numpy() * fit.obs_scale[idx][None, :]
        for j, i in enumerate(idx):
            ours = component_curves(m.bins[i], kf.A[i], kf.n[i], kf.p[i], m.shapes[i]).sum(axis=0)
            np.testing.assert_allclose(ours, pred[: len(ours), j], rtol=1e-6,
                                       atol=1e-6 * ours.max())


def test_every_subset_of_layers_can_be_drawn(results, written):
    raw, res, _ = written
    sid = str(results.fit.spectrum_id[3])
    for show in (("raw",), ("picked",), ("fit",), ("raw", "picked"), ("picked", "fit"),
                 ("raw", "picked", "fit")):
        fig = plot_spectrum(sid, results, show=show)
        assert isinstance(fig, Figure) and gids(fig) & {"raw", "picked", "fit"} == set(show)
    # without any results: raw spectra alone, or picked peaks alone
    assert gids(plot_spectrum(sid, spectra=raw, show=("raw",))) == {"raw"}
    assert gids(plot_spectrum(sid, binned=res.binned, show=("picked",))) == {"picked"}


def test_a_missing_layer_is_left_out_and_nothing_to_draw_is_an_error(results, written):
    raw, res, _ = written
    sid = str(results.fit.spectrum_id[0])
    d = spectrum_data(sid, binned=res.binned, show=("raw", "picked", "fit"))
    assert set(d["missing"]) == {"raw", "fit"} and "picked" in d
    with pytest.raises(ValueError, match="nothing to draw"):
        plot_spectrum(sid, show=("fit",))
    with pytest.raises(ValueError, match="subset"):
        plot_spectrum(sid, results, show=("raw", "nonsense"))
    with pytest.raises(ValueError, match="nothing to draw"):
        plot_spectrum("proj:nope#1", results)


def test_axes_rules(results):
    sid = str(results.fit.spectrum_id[3])
    nodeut = Results(measurements=results.measurements, fit=results.fit)  # no picked table
    fig = plot_spectrum(sid, nodeut, show=("picked", "fit"))
    assert fig.axes[0].get_xlabel() == "deuterium level (bin)"  # no predicted m/z available
    with pytest.raises(ValueError, match="m/z axis"):
        plot_spectrum(sid, results, xaxis="n_deut")  # raw is drawn too
    drawn = plot_spectrum(sid, results, show=("picked", "fit"), xaxis="n_deut")
    assert drawn.axes[0].get_xlabel() == "deuterium level (bin)"
    with pytest.raises(ValueError, match="needs the picked table"):
        plot_spectrum(sid, nodeut, show=("fit",), xaxis="mz")
    with pytest.raises(ValueError, match="residuals need"):
        plot_spectrum(sid, results, show=("fit",), residual=True)
    with pytest.raises(ValueError, match="xaxis"):
        plot_spectrum(sid, results, xaxis="time")


def test_component_selection_and_k(results):
    sid = str(results.fit.spectrum_id[8])  # a 60 s spectrum; fitted at K=1 and 2
    full = plot_spectrum(sid, results, k=2, show=("picked", "fit"))
    assert {"component:1", "component:2", "fit", "band"} <= gids(full)
    only2 = plot_spectrum(sid, results, k=2, show=("picked", "fit"), components=[2])
    assert "showing pop 2" in only2.axes[0].get_title()
    assert "showing" not in full.axes[0].get_title()
    g = gids(only2)
    assert "component:2" in g and "component:1" not in g and "fit" in g and "band" in g
    d = spectrum_data(sid, results, show=("fit",), k=2, components=[2])["fit"]
    assert d["shown"] == [2] and d["components"].shape[0] == 2  # data keeps every population
    np.testing.assert_allclose(d["components"].sum(axis=0), d["total"])
    assert np.all(np.diff(d["mean"]) >= 0)  # ranked by mean
    with pytest.raises(ValueError, match="ranks"):
        plot_spectrum(sid, results, k=1, components=[2], show=("fit",))
    with pytest.raises(ValueError, match="not fitted"):
        plot_spectrum(sid, results, k=3, show=("fit",))


def test_the_fit_line_has_no_markers_and_the_picked_bins_have_circles(results):
    sid = str(results.fit.spectrum_id[3])
    fig = plot_spectrum(sid, results, show=("picked", "fit"))
    fit_line = [a for a in fig.axes[0].get_lines() if a.get_gid() == "fit"][0]
    assert fit_line.get_marker() in ("None", None, "")
    d = spectrum_data(sid, results, show=("picked",))["picked"]
    circles = [a for a in fig.axes[0].collections if a.get_gid() == "picked"
               and len(a.get_offsets()) == len(d["mz"])]
    assert circles  # one circle on top of each picked bin


def test_residuals_normalise_and_corrected_labels(results):
    sid = str(results.fit.spectrum_id[3])
    fig = plot_spectrum(sid, results, show=("picked", "fit"), residual=True, normalize=True)
    assert len(fig.axes) == 2 and "residual" in gids(fig)
    assert max(ax.get_ylim()[1] for ax in fig.axes[:1]) <= 1.6
    text = " ".join(t.get_text() for t in plot_spectrum(sid, results, show=("fit",)).axes[0]
                    .get_legend().get_texts())
    assert "corr." in text and "±" in text
    plain = plot_spectrum(sid, results, show=("fit",), corrected=False).axes[0].get_legend()
    assert "corr." not in " ".join(t.get_text() for t in plain.get_texts())


def test_a_loaded_bundle_plots_the_same_data(written, results):
    raw, res, path = written
    with open_bundle(path) as b:
        loaded = Results.from_bundle(b)
        for sid in map(str, results.fit.spectrum_id[[0, 3, 8, 12]]):
            a = spectrum_data(sid, results)
            c = spectrum_data(sid, loaded)  # the raw spectrum comes from the open bundle
            assert a.keys() == c.keys() and not c["missing"]
            for layer in ("raw", "picked", "fit"):
                for key, value in a[layer].items():
                    other = c[layer][key]
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, other)
                    elif value is not None and not isinstance(value, (list, int)):
                        assert value == other
        fig = plot_spectrum(str(results.fit.spectrum_id[3]), loaded)
        assert {"raw", "picked", "fit"} <= gids(fig)


def test_the_review_layer_changes_the_k_without_touching_the_fit(results):
    sid = str(results.fit.spectrum_id[3])
    base = spectrum_data(sid, results, show=("fit",))["fit"]
    other_k = 2 if base["k"] == 1 else 1
    review = ReviewTable.empty().update(sid, status="rejected", k_override=other_k)
    changed = results.with_review(review)
    assert changed.fit is results.fit and changed.review is review and results.review is None
    assert spectrum_data(sid, changed, show=("fit",))["fit"]["k"] != base["k"]
    assert "rejected" in plot_spectrum(sid, changed, show=("fit",)).axes[0].get_title()


# --- series plots ---------------------------------------------------------------------------

def test_series_info_orders_by_time_then_replicate(results):
    info = series_info(results, 1)
    assert list(info["times"]) == [0, 0, 0, 60, 60, 60, 1e6, 1e6]
    assert list(info["reps"]) == [1, 2, 3, 1, 2, 3, 1, 2]
    assert list(info["time_index"]) == [0, 0, 0, 1, 1, 1, 2, 2]
    assert info["n_amides"] == results.controls.n_amides[1] and "RDKVQKEYAL" in info["label"]
    assert (results.fit.data_id[info["rows"]] == 1).all()
    with pytest.raises(KeyError):
        series_info(results, 9)


def test_uptake_data_corrected_and_raw(results):
    raw = uptake_data(results, 0, corrected=False)
    cor = uptake_data(results, 0)
    assert cor["corrected"] and not raw["corrected"] and len(raw["mean"]) == len(cor["mean"])
    m0, mfd, n = cor["m0"], cor["mfd"], cor["n_amides"]
    np.testing.assert_allclose(cor["mean"], np.maximum((raw["mean"] - m0) / (mfd - m0) * n, 0))
    assert not cor["control_fallback"] and cor["fd_time"] == 1e6
    nocontrols = Results(measurements=results.measurements, fit=results.fit,
                         bootstrap=results.bootstrap, bootstrap_by_k=results.bootstrap_by_k)
    d = uptake_data(nocontrols, 0)
    assert not d["corrected"] and d["m0"] is None and d["fd_time"] is None
    np.testing.assert_allclose(d["mean"], raw["mean"])


def test_uptake_components_and_k(results):
    both = uptake_data(results, 0, k=2)
    assert set(both["rank"]) == {1, 2} and len(both["rank"]) == 2 * 8
    assert set(uptake_data(results, 0, k=2, components=[2])["rank"]) == {2}
    assert np.isfinite(both["mean_se"]).all()  # fixed-K bootstrap covers every K


def test_plot_uptake_axes(results):
    by_index = plot_uptake(results, 0, x="index")
    ax = by_index.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "60", "FD"]
    assert {"rep:1", "rep:2", "rep:3", "limits"} <= gids(by_index)
    assert ax.get_ylabel().endswith("corrected)")
    by_time = plot_uptake(results, 0, x="time", corrected=False)
    pts = [a for a in by_time.axes[0].get_children() if a.get_gid() == "rep:1"][0]
    assert pts.get_offsets()[:, 0].max() == 60  # fully deuterated: a line, not a point
    logged = plot_uptake(results, 0, x="time", scale="log")
    pts = [a for a in logged.axes[0].get_children() if a.get_gid() == "rep:1"][0]
    assert (pts.get_offsets()[:, 0] > 0).all() and logged.axes[0].get_xscale() == "log"
    for bad in ({"x": "date"}, {"scale": "sqrt"}):
        with pytest.raises(ValueError):
            plot_uptake(results, 0, **bad)


def test_reference_overlay_and_loader(results, tmp_path):
    path = tmp_path / "sol.tsv"
    path.write_text("time\tfd1\tfd2\tp1\tp2\n0\t0\t#N/A\t1\t0\n60\t25\t100\t0.5\t0.5\n"
                    "1.00E+06\t100\t#N/A\t1\t0\n")
    ref = load_reference(path)
    means, fractions = ref.at(60, 8.0)
    np.testing.assert_allclose(means, [2.0, 8.0])
    np.testing.assert_allclose(fractions, [0.5, 0.5])
    assert ref.at(1e6, 8.0)[0].tolist() == [8.0] and ref.at(5, 8.0) is None
    assert "reference" in gids(plot_uptake(results, 0, x="index", reference=ref))
    assert "reference" in gids(plot_populations(results, 0, reference=ref))
    assert isinstance(ref, Reference)


def test_population_plots_and_components(results):
    sid = str(results.fit.spectrum_id[8])
    d = population_data(results, sid, k=2)
    assert d["draws"] is None  # this run did not keep the draws
    assert d["corrected"] and np.isclose(d["centroid"], (d["mean"] * d["fraction"]).sum())
    fig = plot_populations(results, 0, k=2, components=[2])
    assert {"pop:2", "centroid", "limits"} <= gids(fig) and "pop:1" not in gids(fig)
    n_cells = sum(ax.get_visible() for ax in fig.axes)
    assert n_cells == 8 + 3  # eight spectra and an overlay for each of the three time points
    assert sum(ax.get_visible() for ax in plot_populations(results, 0, overlay=False).axes) == 8
    # the replicate overlay has the same dashed guides at 0 and at n_amides as the single cells
    overlays = [ax for ax in fig.axes if ax.get_visible() and "replicates" in ax.get_title()]
    assert len(overlays) == 3
    for ax in fig.axes:
        if ax.get_visible():
            guides = [c for c in ax.collections if c.get_gid() == "limits"]
            assert len(guides) == 1 and ax.get_xlim() == (-1, results.controls.n_amides[0] + 1)


def test_series_spectra_grid_and_blank_cells(results):
    fig = plot_series_spectra(results, 0, show=("picked", "fit"), components=[1])
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert len(shown) == 8 and "component:1" in gids(fig) and "component:2" not in gids(fig)
    nobinned = Results(measurements=results.measurements, fit=results.fit)
    text = plot_series_spectra(nobinned, 0, show=("raw",))  # nothing can be drawn: blank cells
    assert all(not ax.get_lines() and ax.texts for ax in text.axes if ax.get_visible())


def test_uptake_and_populations_match_after_loading(written, results):
    raw, res, path = written
    with open_bundle(path) as b:
        loaded = Results.from_bundle(b)
    for kw in ({}, {"corrected": False}, {"k": 1}):
        a, c = uptake_data(results, 1, **kw), uptake_data(loaded, 1, **kw)
        for key, value in a.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, c[key])
            else:
                assert value == c[key]


# --- tables, figure files and the run option ------------------------------------------------



def test_tables_have_the_documented_columns_and_values(results):
    comp, spec, up = components_table(results), spectra_table(results), uptake_table(results)
    ks = results.effective_K
    assert len(comp["rank"]) == ks.sum() and len(spec["K"]) == len(results)
    assert {"mean", "mean_corrected", "mean_corrected_se", "fraction", "status", "flags",
            "peptide", "rep"} <= set(comp)
    assert (comp["peptide"] == "RDKVQKEYAL").all() and set(comp["rep"]) == {1, 2, 3}
    k_max = int(ks.max())
    assert f"mean_{k_max}" in spec and f"fraction_se_{k_max}" in spec
    i = 8
    sel = comp["spectrum_id"] == spec["spectrum_id"][i]
    assert np.isclose(spec["centroid"][i], (comp["mean"][sel] * comp["fraction"][sel]).sum())
    assert np.isclose(spec["mean_corrected_1"][i], comp["mean_corrected"][sel][0])
    # the replicate average: three replicates at 60 s
    row = (up["data_id"] == 0) & (up["time"] == 60) & (up["rank"] == 1)
    assert up["n"][row][0] == 3 and np.isfinite(up["mean_sd"][row][0])
    mine = (comp["data_id"] == 0) & (comp["time"] == 60) & (comp["rank"] == 1)
    assert np.isclose(up["mean"][row][0], comp["mean_corrected"][mine].mean())


def test_a_rejected_spectrum_leaves_the_uptake_average(results):
    rows = np.flatnonzero((results.fit.data_id == 0) & (results.fit.time == 60))
    sid = str(results.fit.spectrum_id[rows[0]])
    rejected = results.with_review(ReviewTable.empty().update(sid, status="rejected"))
    up = uptake_table(rejected)
    row = (up["data_id"] == 0) & (up["time"] == 60) & (up["rank"] == 1)
    assert up["n"][row][0] == 2
    assert components_table(rejected)["status"].tolist().count("rejected") >= 1


def test_csv_files(results, tmp_path):
    files = write_tables(results, tmp_path / "t")
    assert [f.name for f in files] == ["spectra.csv", "components.csv", "uptake.csv"]
    with files[1].open() as fh:
        rows = list(csv.DictReader(fh))
    comp = components_table(results)
    assert len(rows) == len(comp["rank"]) and rows[0]["spectrum_id"] == comp["spectrum_id"][0]
    assert float(rows[0]["mean"]) == pytest.approx(comp["mean"][0])
    nocontrols = Results(measurements=results.measurements, fit=results.fit)
    with write_tables(nocontrols, tmp_path / "n")[1].open() as fh:
        assert next(csv.DictReader(fh))["mean_corrected"] == ""  # NaN is an empty cell


def test_select_series(results):
    assert select_series(results, "all") == [0, 1]
    assert select_series(results, [1]) == [1]
    assert select_series(results) == []  # nothing is flagged in this clean run
    with pytest.raises(KeyError):
        select_series(results, [7])
    with pytest.raises(ValueError):
        select_series(results, "some")
    rejected = results.with_review(
        ReviewTable.empty().update(str(results.fit.spectrum_id[12]), status="rejected"))
    assert select_series(rejected) == [1]


def test_figure_files(results, tmp_path):
    files = write_figures(results, tmp_path / "pdf", PlotConfig(series="all"))
    assert len(files) == 2 and all(f.suffix == ".pdf" and f.stat().st_size > 5000 for f in files)
    assert b"/Count 3" in files[0].read_bytes()  # one page for each kind
    png = write_figures(results, tmp_path / "png",
                        PlotConfig(series=(1,), kinds=("uptake",), format="png", dpi=50))
    assert [f.name.endswith("_uptake.png") for f in png] == [True]
    assert write_figures(results, tmp_path / "none", PlotConfig()) == []  # nothing flagged
    seen = []
    write_figures(results, tmp_path / "p", PlotConfig(series="all", kinds=("uptake",)),
                  progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


def test_plot_config_checks_and_round_trips():
    c = PlotConfig(series=[3, 4], kinds=["uptake"], show=["fit"], x="index", format="svg")
    assert c.series == (3, 4) and PlotConfig.from_dict(c.to_dict()) == c
    for bad in ({"series": "some"}, {"kinds": ()}, {"kinds": ("pie",)}, {"show": ("x",)},
                {"x": "date"}, {"scale": "sqrt"}, {"format": "gif"}, {"dpi": 0}):
        with pytest.raises(ValueError):
            PlotConfig(**bad)
    with pytest.raises(ValueError, match="unknown"):
        PlotConfig.from_dict({"bogus": 1})


def test_a_run_writes_figures_when_asked(tmp_path):
    raw = raw_spectra(n_series=2)
    plots = PlotConfig(series="all", kinds=("uptake", "populations"))
    cfg = config(bootstrap=None, plots=plots)
    run(raw, cfg, out=tmp_path / "bundle")
    assert len(list((tmp_path / "bundle_figures").glob("*.pdf"))) == 2
    run(raw, cfg, figures_dir=tmp_path / "elsewhere")
    assert len(list((tmp_path / "elsewhere").glob("*.pdf"))) == 2
    with pytest.raises(ValueError, match="figures_dir or out"):
        run(raw, cfg)  # nowhere to write: refused before any work
    none = run(raw, config(bootstrap=None), out=tmp_path / "plain")
    assert not (tmp_path / "plain_figures").exists() and none.config.plots is None
    from hxex.run import RunConfig
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    with open_bundle(tmp_path / "bundle") as b:
        assert b.manifest["configs"]["run"]["plots"]["kinds"] == ["uptake", "populations"]


def test_grid_layout_wraps_a_single_replicate_time_course():
    from hxex.report.series import _layout

    def info(n_times, n_reps):
        times = np.repeat(np.arange(n_times, dtype=float), n_reps)
        reps = np.tile(np.arange(1, n_reps + 1), n_times)
        return {"rows": np.arange(len(times)), "times": times, "reps": reps}

    nrows, ncols, pos, by_rep = _layout(info(24, 1), None)
    assert (nrows, ncols, by_rep) == (6, 4, False)
    assert pos[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    assert len(set(pos)) == 24 and pos[-1] == (5, 3)
    nrows, ncols, pos, by_rep = _layout(info(3, 2), None)  # replicates: a row per time
    assert (nrows, ncols, by_rep) == (3, 2, True)
    assert pos == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert _layout(info(3, 2), 3)[:2] == (2, 3) and not _layout(info(3, 2), 3)[3]  # forced wrap
    assert _layout(info(2, 1), None)[:2] == (1, 2)  # few spectra: no empty columns
    with pytest.raises(ValueError, match="columns"):
        _layout(info(3, 1), 0)


def test_columns_option_on_the_series_plots(results):
    fig = plot_series_spectra(results, 0, show=("picked", "fit"), columns=3)
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert len(shown) == 8 and len(fig.axes) == 9  # three rows of three, one cell unused
    pops = plot_populations(results, 0, columns=2)
    assert sum(ax.get_visible() for ax in pops.axes) == 8 and len(pops.axes) == 8  # no overlay


def test_the_raw_line_is_not_drawn_across_gaps(results, written):
    from hxex.report.spectrum import _break_gaps

    mz = np.array([1.0, 1.001, 1.002, 5.0, 5.001, 5.002, 9.0])
    y = np.arange(7, dtype=float)
    bx, by = _break_gaps(mz, y)
    assert len(bx) == 9 and np.isnan(bx).sum() == 2 and np.isnan(by).sum() == 2
    assert list(bx[:3]) == [1.0, 1.001, 1.002] and np.isnan(bx[3]) and bx[4] == 5.0
    dense = np.linspace(0, 1, 100)  # evenly spaced profile data is untouched
    assert len(_break_gaps(dense, dense)[0]) == 100
    assert _break_gaps(mz[:2], y[:2])[0].tolist() == [1.0, 1.001]
    raw, res, _ = written
    sid = str(results.fit.spectrum_id[3])
    fig = plot_spectrum(sid, results, show=("raw",))
    line = [a for a in fig.axes[0].get_children() if a.get_gid() == "raw"][0]
    assert np.isnan(line.get_xdata()).any()  # the synthetic spectrum has three points per bin
    assert spectrum_data(sid, results, show=("raw",))["raw"]["mz"].size == len(
        raw.spectrum(raw.find(sid))[0])  # the data itself is not altered


# --- settings stored with the results and the picking window --------------------------------

def test_the_modification_registry_round_trips_and_compares_by_mass():
    d = chem.DEFAULT_MODIFICATIONS
    assert chem.ModificationRegistry.from_dict(d.to_dict()) == d
    custom = d.with_modification("meK", "K", "CH2")
    assert custom != d and chem.ModificationRegistry.from_dict(custom.to_dict()) == custom
    other = d.with_modification("meK", "K", "C2H4")
    assert other != custom  # the same label with another mass is another registry
    assert hash(custom) == hash(chem.ModificationRegistry.from_dict(custom.to_dict()))
    with pytest.raises(ValueError, match="unknown"):
        chem.ModificationRegistry.from_dict({"bogus": 1})


def test_a_run_stores_its_settings_and_results_read_them_back(tmp_path):
    raw = raw_spectra(n_series=1)
    custom = chem.DEFAULT_MODIFICATIONS.with_modification("meK", "K", "CH2")
    cfg = config(bootstrap=None, registry=custom)
    res = run(raw, cfg, out=tmp_path / "b", project="proj", notes="a note")
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    fresh = Results.from_run(res, spectra=raw)
    assert fresh.settings.run == cfg and fresh.settings.registry == custom
    assert fresh.settings.picker == cfg.picker and fresh.settings.versions["hxex"]
    with open_bundle(tmp_path / "b") as b:
        loaded = Results.from_bundle(b)
    s = loaded.settings
    assert (s.project, s.notes) == ("proj", "a note") and s.created and s.versions["numpy"]
    assert s.run == cfg and s.picker == cfg.picker and s.registry == custom
    assert s.configs["fit"]["k_max"] == cfg.fit.k_max and s.configs["quality"] is not None
    assert Settings().picker is None and Settings().run is None
    assert Settings().registry == chem.DEFAULT_MODIFICATIONS
    only_run = Settings(configs={"run": cfg.to_dict()})  # picker found in the run settings
    assert only_run.picker == cfg.picker


def test_a_bundle_written_without_settings_says_so(tmp_path, results, caplog):
    with caplog.at_level(logging.WARNING, logger="hxex.io.bundle"):
        write_bundle(tmp_path / "x", measurements=results.measurements, fit=results.fit)
    assert "no picker settings" in caplog.text and "no run settings" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hxex.io.bundle"):
        settings = results.settings
        write_bundle(tmp_path / "y", measurements=results.measurements, fit=results.fit,
                     picker_config=settings.picker, run_config=settings.run.to_dict())
    assert caplog.text == ""


def test_picked_bins_are_drawn_as_the_pickers_window(results, written):
    from matplotlib.patches import Rectangle

    sid = str(results.fit.spectrum_id[3])
    ppm = results.settings.picker.resolution_ppm
    d = spectrum_data(sid, results, show=("picked",))["picked"]
    np.testing.assert_allclose(d["window"], d["mz"] * ppm / 1e6)
    assert d["resolution_ppm"] == ppm
    fig = plot_spectrum(sid, results, show=("picked",))
    bars = [a for a in fig.axes[0].patches if a.get_gid() == "picked"]
    assert len(bars) == len(d["mz"]) and all(isinstance(b, Rectangle) for b in bars)
    np.testing.assert_allclose([b.get_width() for b in bars], 2 * d["window"])
    np.testing.assert_allclose([b.get_x() + b.get_width() / 2 for b in bars], d["mz"])
    assert "50 ppm window" in " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert all(b.get_linewidth() >= 1.0 for b in bars)  # a window is never thinner than this
    assert all(b.get_edgecolor() == b.get_facecolor() for b in bars)  # shaded, no outline
    circles = [a for a in fig.axes[0].collections if a.get_gid() == "picked"]
    assert len(circles) == 1 and len(circles[0].get_offsets()) == len(d["mz"])
    np.testing.assert_allclose(circles[0].get_offsets()[:, 0], d["mz"])
    np.testing.assert_allclose(circles[0].get_offsets()[:, 1], d["intensity"])
    stick = plot_spectrum(sid, results, show=("picked",), picked_style="stick")
    assert any(len(a.get_offsets()) == len(d["mz"]) for a in stick.axes[0].collections
               if a.get_gid() == "picked")  # sticks have the circles too
    assert not stick.axes[0].patches and "picked" in gids(stick)
    deut = plot_spectrum(sid, results, show=("picked", "fit"), xaxis="n_deut")
    assert not deut.axes[0].patches  # no window on the deuterium axis: sticks
    with pytest.raises(ValueError, match="needs the m/z axis"):
        plot_spectrum(sid, results, show=("picked", "fit"), xaxis="n_deut", picked_style="window")
    with pytest.raises(ValueError, match="picked_style"):
        plot_spectrum(sid, results, picked_style="fat")


def test_the_window_needs_the_picker_setting_or_an_argument(written):
    raw, res, _ = written
    sid = str(res.fit.spectrum_id[3])
    bare = plot_spectrum(sid, binned=res.binned, show=("picked",))  # no results: sticks
    assert not bare.axes[0].patches
    given = plot_spectrum(sid, binned=res.binned, show=("picked",), resolution_ppm=100.0)
    assert len(given.axes[0].patches) > 0
    with pytest.raises(ValueError, match="resolution_ppm"):
        plot_spectrum(sid, binned=res.binned, show=("picked",), picked_style="window")


# --- bands of the populations, and the colours -----------------------------------------------

def test_the_bootstrap_holds_a_band_for_every_population(results):
    boot = results.bootstrap_by_k[2]
    for i in (0, 5, 12):
        comp = boot.curve_components(i)
        n_bins = len(results.measurements.bins[i])
        assert comp is not None and comp[0].shape == (2, n_bins) == comp[1].shape
        assert np.all(comp[0] <= comp[1] + 1e-9) and np.all(comp[0] >= -1e-9)
        lo, hi = boot.curve_band(i)  # the total band is wider than any one population's band
        assert np.all(comp[1] <= hi + 1e-6)
    sub = boot.select([3, 1])
    assert np.array_equal(sub.curve_components(0)[0], boot.curve_components(3)[0])
    assert np.array_equal(sub.curve_components(1)[1], boot.curve_components(1)[1])
    one = results.bootstrap_by_k[1].curve_components(0)
    assert one[0].shape[0] == 1


def test_results_without_population_bands_still_plot(results):
    import dataclasses

    old = dataclasses.replace(results.bootstrap_by_k[2], component_lo=None, component_hi=None,
                              component_offsets=None)
    assert old.curve_components(0) is None
    legacy = dataclasses.replace(results, bootstrap_by_k={2: old}, bootstrap=None)
    sid = str(results.fit.spectrum_id[8])
    fig = plot_spectrum(sid, legacy, k=2, show=("picked", "fit"))
    assert "band" in gids(fig) and not any(g.startswith("component_band") for g in gids(fig))
    assert spectrum_data(sid, legacy, k=2, show=("fit",))["fit"]["component_band"] is None


def test_the_bundle_keeps_the_population_bands(written, results):
    raw, res, path = written
    with open_bundle(path) as b:
        for k in (1, 2):
            got, want = b.bootstrap(k), results.bootstrap_by_k[k]
            for name in ("component_lo", "component_hi", "component_offsets"):
                np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
        sel = b.bootstrap()
        np.testing.assert_array_equal(sel.component_lo, results.bootstrap.component_lo)


def test_the_fit_has_its_own_colour_and_populations_keep_theirs(results):
    import matplotlib.colors as mc

    from hxex.report.spectrum import COLORS, FIT_COLOR

    assert FIT_COLOR not in COLORS and len(set(COLORS)) == len(COLORS)
    sid = str(results.fit.spectrum_id[8])
    fig = plot_spectrum(sid, results, k=2, show=("picked", "fit"))
    ax = fig.axes[0]
    fit_line = [a for a in ax.get_lines() if a.get_gid() == "fit"][0]
    assert mc.same_color(fit_line.get_color(), FIT_COLOR)
    for rank in (1, 2):
        line = [a for a in ax.get_lines() if a.get_gid() == f"component:{rank}"][0]
        band = [a for a in ax.collections if a.get_gid() == f"component_band:{rank}"][0]
        assert mc.same_color(line.get_color(), COLORS[rank - 1])
        assert mc.same_color(band.get_facecolor()[0][:3], COLORS[rank - 1])
    total = [a for a in ax.collections if a.get_gid() == "band"][0]
    assert mc.same_color(total.get_facecolor()[0][:3], FIT_COLOR)  # the total band follows the fit
    only = plot_spectrum(sid, results, k=2, show=("picked", "fit"), component_band=False)
    assert "band" in gids(only) and not any(g.startswith("component_band") for g in gids(only))
    one = plot_spectrum(sid, results, k=2, show=("picked", "fit"), components=[2])
    assert "component_band:2" in gids(one) and "component_band:1" not in gids(one)
    d = spectrum_data(sid, results, k=2, show=("fit",))["fit"]["component_band"]
    assert d.shape[:2] == (2, 2) and np.all(d[:, 0] <= d[:, 1] + 1e-9)


def test_the_replicate_overlay_keeps_the_population_colours(results):
    import matplotlib.colors as mc

    from hxex.report.spectrum import COLORS

    fig = plot_populations(results, 0, k=2)
    overlay = [ax for ax in fig.axes if ax.get_visible() and "replicates" in ax.get_title()][1]
    points = [a for a in overlay.get_lines() if (a.get_gid() or "").startswith("overlay:")
              and a.get_marker() != "None"]
    assert {a.get_marker() for a in points} == {"o", "s", "^"}  # one shape per replicate
    assert {mc.to_hex(a.get_color()) for a in points} == {mc.to_hex(c) for c in COLORS[:2]}


def test_populations_beyond_the_palette_change_line_style(results, monkeypatch):
    import matplotlib.colors as mc

    from hxex.report import spectrum as sp

    n = len(sp.COLORS)
    first, again, third = sp.population_style(1), sp.population_style(n + 1), sp.population_style(
        2 * n + 1)
    assert first["color"] == again["color"] == third["color"] == sp.COLORS[0]
    assert len({first["linestyle"], again["linestyle"], third["linestyle"]}) == 3
    assert (first["fillstyle"], again["fillstyle"]) == ("full", "none")
    styles = {(sp.population_style(r)["color"], str(sp.population_style(r)["linestyle"]))
              for r in range(1, 4 * n + 1)}
    assert len(styles) == 4 * n  # every population up to K = 24 looks different
    # with a palette of one colour, K = 2 already needs the fallback
    monkeypatch.setattr(sp, "COLORS", ("#000000",))
    sid = str(results.fit.spectrum_id[8])
    ax = plot_spectrum(sid, results, k=2, show=("fit",)).axes[0]
    one, two = (next(a for a in ax.get_lines() if a.get_gid() == f"component:{r}") for r in (1, 2))
    assert mc.same_color(one.get_color(), two.get_color())
    assert one.get_linestyle() != two.get_linestyle()
    cells = plot_populations(results, 0, k=2, overlay=False).axes
    hollow = [a for ax in cells for a in ax.get_lines() if a.get_gid() == "pop:2"]
    assert hollow and all(a.get_fillstyle() == "none" for a in hollow)
