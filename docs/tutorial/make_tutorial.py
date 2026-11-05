# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the tutorial download and work through the tutorial (docs/guide/tutorial.md).

    python docs/tutorial/make_tutorial.py SPECEXPORT_DIR OUT_DIR          # the zip only
    python docs/tutorial/make_tutorial.py SPECEXPORT_DIR OUT_DIR --run    # and work through it

SPECEXPORT_DIR holds the published B5 SpecExport data: the WT, D109H, and R120G folders and
their .fasta files. OUT_DIR gets ``hxex_b5_tutorial.zip``, which is uploaded to a GitHub
release tagged ``tutorial-data-N`` (docs/RELEASING.md). With ``--run`` the zip is unpacked into
``OUT_DIR/work`` and the two notebooks, which hold the commands and changes of the tutorial's
two pages as the pages give them (tests/test_tutorial.py checks that they agree), are executed
there in turn (needs the ``notebook`` extra; about an hour and a half on a CPU, half an hour
with a GPU); the executed copies are saved in that folder, and each page's figures are drawn
into ``docs/images`` (``tutorial_*.png``) from the results its notebook left.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "hxex_b5_tutorial"
SAMPLES = ("WT", "D109H", "R120G")
# the notebooks of the two pages, in the order they run
NOTEBOOKS = (HERE / f"{NAME}_1_basics.ipynb", HERE / f"{NAME}_2_fits.ipynb")
HELPERS = HERE / "tutorial_helpers.py"


def build_zip(specexport: Path, out: Path) -> Path:
    """``out/hxex_b5_tutorial.zip``: the samples' SpecExport folders and sequences under
    ``hxex_b5_tutorial/SpecExport``, with the notebooks and their helpers, the README, and the
    licence."""
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{NAME}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for sample in SAMPLES:
            fasta = specexport / f"{sample}.fasta"
            z.write(fasta, f"{NAME}/SpecExport/{fasta.name}")
            for path in sorted((specexport / sample).rglob("*.csv")):
                z.write(path, f"{NAME}/SpecExport/{path.relative_to(specexport).as_posix()}")
        for extra in (*NOTEBOOKS, HELPERS, HERE / "README.txt", HERE / "LICENSE.txt"):
            z.write(extra, f"{NAME}/{extra.name}")
    print(f"{target}: {target.stat().st_size / 1e6:.1f} MB")
    return target


def unpack(archive: Path, work: Path) -> Path:
    """Unpack the zip into ``work`` (replaced) and return the project folder."""
    if work.exists():
        shutil.rmtree(work)
    with zipfile.ZipFile(archive) as z:
        z.extractall(work)
    return work / NAME


def execute_notebook(folder: Path, notebook: Path) -> Path:
    """Execute ``notebook`` in ``folder`` (its ``!hxex`` commands use the ``hxex`` of this
    Python's environment) and save the executed copy there."""
    import nbformat
    from nbclient import NotebookClient

    path = folder / notebook.name
    nb = nbformat.read(path, as_version=4)
    env_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(folder)}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path}")
    return path


# the pages' figures (part 1: docs/guide/tutorial.md, part 2: tutorial-fits.md). The spectra are
# drawn after part 1's notebook, as both pages first show them (part 2's review and population
# test come later): image name -> (sample, residues, charge, kinds)
SERIES_IMAGES = {
    "tutorial_one_population": ("WT", "87-96", 2, ["spectra"]),
    "tutorial_two_populations": ("WT", "42-54", 2, ["spectra"]),
    "tutorial_uptake": ("WT", "42-54", 2, ["uptake"]),
    "tutorial_bright": ("WT", "68-75", 2, ["spectra"]),
    "tutorial_second_species": ("R120G", "114-137", 3, ["spectra"]),
}
# the analysis figures each part's notebook leaves: image name -> the figure it copies
ANALYSIS_IMAGES = (
    {"tutorial_butterfly": "butterfly_R120G.png",
     "tutorial_woods": "woods_R120G_1min.png",
     "tutorial_residues": "residue_significance.png"},
    {"tutorial_woods_k2": "woods_K2_R120G_1min.png",
     "tutorial_residues_k2": "residue_significance_K2_R120G.png",
     "tutorial_tiers": "population_tiers.png"},
)


def write_images(folder: Path, images: Path, part: int) -> None:
    """Draw the figures of page ``part`` (1 or 2) from the tutorial's results in ``folder``, as
    its notebook left them: after part 1 the spectra of both pages and part 1's analysis, after
    part 2 its analysis."""
    import dataclasses

    import hxex
    from hxex.report import PlotConfig, find_series, plot_controls_overview, series_figures

    images.mkdir(parents=True, exist_ok=True)
    results = hxex.load_results(folder / "hxex.toml")
    log_scale = PlotConfig(scale="log")
    for name, (sample, rng, charge, kinds) in (SERIES_IMAGES if part == 1 else {}).items():
        (d,) = find_series(results, samples=[sample], peptide_ranges=[rng], charges=[charge])
        fig = series_figures(results, d, dataclasses.replace(log_scale, kinds=tuple(kinds)))
        fig[kinds[0]].savefig(images / f"{name}.png", dpi=60)
    if part == 1:
        plot_controls_overview(results).savefig(images / "tutorial_controls.png", dpi=80)
    for name, source in ANALYSIS_IMAGES[part - 1].items():
        shutil.copyfile(folder / "results_analysis_WT" / source, images / f"{name}.png")
    print(f"figures in {images}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specexport", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--run", action="store_true",
                        help="execute the notebooks and draw the pages' figures")
    parser.add_argument("--images", type=Path, default=HERE.parent / "images")
    args = parser.parse_args(argv)
    archive = build_zip(args.specexport, args.out)
    if args.run:
        folder = unpack(archive, args.out / "work")
        for part, notebook in enumerate(NOTEBOOKS, start=1):
            execute_notebook(folder, notebook)
            write_images(folder, args.images, part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
