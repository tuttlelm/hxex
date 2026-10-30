# SPDX-License-Identifier: GPL-3.0-or-later
"""Deamidated fully deuterated controls: Asn-Gly candidates, the share from the picked peaks'
m/z, and the correction of the controls (docs/ANALYSIS.md, HISTORY 197)."""

from types import SimpleNamespace

import numpy as np
import pytest

from hxex.controls import ControlConfig, charge_groups, fit_controls, series_residues
from hxex.deamidation import (
    LIGHTER_DA,
    Deamidation,
    SampleDeamidation,
    asn_gly_series,
    assess_deamidation,
    deamidation_from_picked,
)
from test_controls import fake_fit

ON = ControlConfig(deamidation=True)


def _groups(peptides):
    """peptides: data_id -> (sample, sequence, first position)."""
    ids = list(peptides)
    charges = charge_groups([peptides[d][0] for d in ids], [peptides[d][1] for d in ids],
                            [""] * len(ids), [2] * len(ids), ids)
    residues = series_residues([peptides[d][1] for d in ids], [""] * len(ids),
                               [peptides[d][2] for d in ids], ids)
    return charges, residues


def test_asn_gly_candidates_from_the_sequence_and_overlapping_peptides():
    """An Asn followed by Gly (base residues of the modX labels), or a C-terminal Asn whose
    next residue, read from an overlapping peptide of the same sample, is Gly."""
    peptides = {0: ("A", "AKNGLE", 1),  # internal Asn-Gly
                1: ("A", "LpSNGK", 20),  # modified neighbour: still Asn-Gly
                2: ("A", "AKLEN", 30),  # ends in Asn at 34; residue 35 is G (peptide 3)
                3: ("A", "NGQRT", 34),
                4: ("B", "AKLEN", 30),  # the same peptide in a sample without that overlap
                5: ("A", "ANDGK", 50),  # Asn-Asp, Asp-Gly: not a candidate
                6: ("A", "QWERN", 60)}  # C-terminal Asn, next residue unknown
    charges, residues = _groups(peptides)
    assert asn_gly_series(charges, residues) == {0: "A", 1: "A", 2: "A", 3: "A"}
    assert asn_gly_series(charges, None) == {0: "A", 1: "A", 3: "A"}


def _spectra(shares, mass=2000.0, scatter=0.3, seed=0, n_other=6):
    """Per-spectrum arrays of one run: per sample, Asn-Gly series (``shares[sample]`` their
    deamidated share, None for no candidate) and ``n_other`` other series, each with two
    undeuterated and two fully deuterated controls (replicates 1 and 2)."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("data_id", "role", "sample", "rep", "mass", "offset")}
    peptides = {}
    d = 0
    for sample, (n_ng, share) in shares.items():
        for j in range(n_ng + n_other):
            ng = j < n_ng
            peptides[d] = (sample, ("AKNGLE" if ng else "AKLLE") + "W" * j, 1)
            for role in ("undeut", "fulldeut"):
                for rep in (1, 2):
                    shift = -share * LIGHTER_DA / mass * 1e6 if ng and role == "fulldeut" else 0
                    for name, value in (("data_id", d), ("role", role), ("sample", sample),
                                        ("rep", rep), ("mass", mass),
                                        ("offset", shift + rng.normal(0, scatter))):
                        cols[name].append(value)
            d += 1
    return {k: np.array(v) for k, v in cols.items()}, peptides


def test_the_share_comes_from_the_mass_offset_per_sample():
    """The deamidated copy is 0.022 Da lighter than a deuteron: the Asn-Gly series' fully
    deuterated controls read lower in m/z by share x 0.022 Da / M against their undeuterated
    ones, relative to the other peptides of their injection; a sample's share is the median
    over its candidates less that over its other series."""
    cols, peptides = _spectra({"heated": (4, 0.5), "clean": (4, 0.0), "few": (1, 0.5)})
    charges, residues = _groups(peptides)
    m = len(cols["role"])

    def assess(offset=cols["offset"], min_share=0.25):
        return assess_deamidation(cols["data_id"], cols["role"], np.full(m, np.nan),
                                  np.ones(m, bool), charges, residues, sample=cols["sample"],
                                  rep=cols["rep"], mass=cols["mass"], mz_offset_ppm=offset,
                                  min_share=min_share)

    found = assess()
    by = {s.sample: s for s in found.samples}
    assert by["heated"].decision == "deamidated"
    assert by["heated"].share == pytest.approx(0.5, abs=0.1)
    assert by["heated"].n_candidates == 4 and by["heated"].n_others == 6
    assert by["clean"].decision == "not_found" and abs(by["clean"].share) < 0.1
    assert by["few"].decision == "not_assessed" and np.isnan(by["few"].share)
    first = next(d for d, s in found.candidates.items() if s == "heated")
    assert found.shift(first) == by["heated"].share
    assert found.shift(next(d for d, s in found.candidates.items() if s == "clean")) == 0.0
    assert found.shift(10_000) == 0.0 and found.of(10_000) is None
    assert assess(min_share=0.9).samples[0].decision == "not_found"
    # tables picked elsewhere carry no offsets: nothing is assessed
    blind = assess(offset=np.full(m, np.nan))
    assert {s.decision for s in blind.samples} == {"not_assessed"}
    # read back from a bundle's JSON (NaN written as null)
    import json

    from hxex.io._json import dumps

    back = Deamidation.from_dict(json.loads(dumps(found.to_dict())))
    assert back.candidates == found.candidates
    assert back.samples[0] == found.samples[0]
    assert np.isnan(next(s for s in back.samples if s.sample == "few").share)


def test_from_a_picked_table_in_the_fit_order():
    """deamidation_from_picked takes the ion mass from the first bin's m/z and the charge, and
    the picked table's rows by spectrum id (it may hold more spectra than the fit)."""
    cols, peptides = _spectra({"heated": (3, 0.6)})
    m = len(cols["role"])
    ids = np.array([f"p:{i}" for i in range(m)])
    order = np.arange(m)[::-1]  # the fit lists them in another order
    picked = SimpleNamespace(
        spectrum_id=np.concatenate([ids, ["extra"]]), offsets=np.arange(m + 2) * 2,
        mz=np.repeat(np.concatenate([cols["mass"] / 2, [500.0]]), 2),
        data_id=np.concatenate([cols["data_id"], [99]]), time=np.full(m + 1, np.nan),
        sample=np.concatenate([cols["sample"], ["heated"]]),
        rep=np.concatenate([cols["rep"], [1]]), charge=np.full(m + 1, 2),
        mz_offset_ppm=np.concatenate([cols["offset"], [0.0]]))
    charges, residues = _groups(peptides)
    found = deamidation_from_picked(picked, ids[order], cols["role"][order], np.ones(m, bool),
                                    charges, residues, min_share=0.25)
    assert found.samples[0].decision == "deamidated"
    assert found.samples[0].share == pytest.approx(0.6, abs=0.1)


def test_a_deamidated_control_is_lowered_by_the_share_before_the_checks():
    """With deamidation on, an Asn-Gly series of a deamidated sample keeps its own fully
    deuterated control, its mean lowered by the share (fd_deamidated, measured); the lowered
    mean is what the checks judge, its replicates' included. Off, or a sample not found
    deamidated: as measured."""
    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 1e6, 7.0), (0, 1e6, 7.0),  # Asn-Gly, retention 0.70
            (1, 0, 0.0), (1, 0, 0.0), (1, 1e6, 10.3), (1, 1e6, 10.3),  # Asn-Gly, above its amides
            (2, 0, 0.0), (2, 0, 0.0), (2, 1e6, 6.5), (2, 1e6, 6.5)]  # other peptide
    found = Deamidation(samples=(SampleDeamidation("S", 0.5, 0.05, 4, 6, "deamidated"),),
                        candidates={0: "S", 1: "S"})
    n = np.full(len(rows), 10.0)
    c = fit_controls(fake_fit(rows), n, ON, deamidation=found)
    assert c.flags[0] == "fd_deamidated" and c.measured[0] and not c.fallback[0]
    assert c.mfd_measured[0] == pytest.approx(6.5) and c.retention[0] == pytest.approx(0.65)
    assert c.flags[1] == "fd_deamidated" and c.retention[1] == pytest.approx(0.98)
    assert c.flags[2] == "" and c.retention[2] == pytest.approx(0.65)
    assert c.deamidation is found
    off = fit_controls(fake_fit(rows), n, ControlConfig(), deamidation=found)
    assert off.retention[0] == pytest.approx(0.70) and off.deamidation is None
    assert off.flags[1] == "fd_replicate_left_out;fd_missing;fd_median"  # above its amides
    clean = Deamidation(samples=(SampleDeamidation("S", 0.1, 0.05, 4, 6, "not_found"),),
                        candidates={0: "S", 1: "S"})
    assert fit_controls(fake_fit(rows), n, ON, deamidation=clean).retention[0] == \
        pytest.approx(0.70)


def test_the_settings():
    assert not ControlConfig().deamidation and ControlConfig().deamidation_min_share == 0.25
    for bad in ({"deamidation": "yes"}, {"deamidation_min_share": 0.0},
                {"deamidation_min_share": 1.5}, {"deamidation_min_share": True}):
        with pytest.raises(ValueError, match="deamidation"):
            ControlConfig(**bad)


def test_a_run_records_the_assessment_and_the_analysis_recomputes_it(tmp_path):
    """[run.controls] deamidation: the run assesses its controls (no Asn-Gly peptide here: an
    empty assessment), the bundle keeps it, series.csv carries deamidated_share, and
    [analysis.controls] deamidation recomputes it from the picked table."""
    import hxex
    from hxex.analysis import analysis_controls
    from hxex.report.tables import series_table
    from hxex.run import run
    from test_run import config, raw_spectra

    res = run(raw_spectra(n_series=2), config(controls=ON, bootstrap=None), out=tmp_path / "b")
    assert res.controls.deamidation == Deamidation(samples=(), candidates={})
    results = hxex.load_results(tmp_path / "b")
    assert results.controls.deamidation == res.controls.deamidation
    assert np.isnan(series_table(results)["deamidated_share"]).all()
    plain = run(raw_spectra(n_series=2), config(bootstrap=None))
    assert plain.controls.deamidation is None
    again = analysis_controls(results, {"deamidation": False}, quiet=True)
    assert again.deamidation is None
    assert analysis_controls(results, {"deamidation_min_share": 0.5},
                             quiet=True).deamidation is not None
