# SPDX-License-Identifier: GPL-3.0-or-later
"""Claims of each undeuterated cluster by the listed ions (hxex.picking.claims; PLAN 34) and the
m/z check's term per charge (hxex.picking.mz_check).

Synthetic profile spectra: a sample of ten ordinary peptides at 2+ (three undeuterated
replicates each, scattering by about 0.7 ppm) plus the tutorial's pair 13-24 PFFPFHSPSRLF and
55-67 LRAPSWFDTGLSE at 2+ (20.6 ppm apart), whose 13-24 windows hold 55-67's envelope, and two
isomers (SEMRLEKDRF and EMRLEKDRFS, one composition)."""

import dataclasses

import numpy as np
import pytest

from hxex import load_results
from hxex.analysis import SpeciesFilters, left_out_spectra
from hxex.analysis.screening import OTHER_ION_REASON
from hxex.config import run_config_from
from hxex.fit import FitConfig
from hxex.flags import entry
from hxex.io.bundle import ReviewTable
from hxex.model import Backend
from hxex.noise import NoiseConfig
from hxex.picking import PickerConfig, pick_spectra, plan_bins
from hxex.picking.claims import claims, ion_label
from hxex.picking.mz_check import series_offsets
from hxex.pipeline import RunConfig, run
from hxex.report.tables import series_table, spectra_table
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet

CFG = PickerConfig()
BACKGROUND = ("MDIAIHHPWIR", "GEHLLESDLF", "EELKVKVLGD", "FPTSTSLSPFYLRPPS", "VQKEYALFYG",
              "RDKVQKEYAL", "ITREEKPAVTAAPKK", "VIEVHGKHEERQDEHG", "FDQFFGEHLLE", "TVNGPRKQVSGPER")
LOST, CLAIMANT = "PFFPFHSPSRLF", "LRAPSWFDTGLSE"
ISOMERS = ("SEMRLEKDRF", "EMRLEKDRFS")
IONS = (*BACKGROUND, LOST, CLAIMANT, *ISOMERS)


def envelope(peptide, z=2, shift_ppm=0.0, height=1e4):
    p = plan_bins(peptide, z, config=CFG)
    shape, offs = p.clusters
    pos = (p.undeut_mz + np.asarray(offs) / z) * (1 + shift_ppm * 1e-6)
    return pos, height * np.asarray(shape) / max(shape)


def profile(peptide, z, env, rp=20000.0):
    """A profile over the ion's bins: Gaussian peaks (FWHM m/z / rp), a point every 4 ppm."""
    p = plan_bins(peptide, z, config=CFG)
    lo, hi = p.undeut_mz - 3 / z, p.pred_mz[-1] + 1 / z
    mz = np.arange(lo, hi, lo * 4e-6)
    sigma = env[0] / rp / 2.3548
    return mz, 1.0 + (env[1] * np.exp(-0.5 * ((mz[:, None] - env[0]) / sigma) ** 2)).sum(axis=1)


def spectra_set(*, windows=None, labelled=False, charge_tilt=0.0, other_peaks=None):
    """Three undeuterated replicates (and with ``labelled`` a labelled spectrum each) of every
    ion of :data:`IONS` in sample S; 13-24's windows hold ``other_peaks`` (55-67, by default).
    ``windows``: (start, end) minutes of each ion's retention window, by peptide (default
    none: exported spectra). ``charge_tilt``: ppm added to the 3+ ions (see the tilt test)."""
    rng = np.random.default_rng(0)
    other_peaks = CLAIMANT if other_peaks is None else other_peaks
    out = []
    for d, pep in enumerate(IONS):
        entry_ = PeptideEntry("f", "S", pep, f"{d + 1:04d}-{d + len(pep):04d}", d + 1,
                              d + len(pep), 2)
        rt = (windows or {}).get(pep, (np.nan, np.nan))
        for rep in (1, 2, 3):
            shift = 1.0 + rng.normal(0, 0.7)
            source = other_peaks if pep == LOST else pep
            height = 10 ** rng.uniform(3.5, 4.5)
            mz, y = profile(pep, 2, envelope(source, shift_ppm=shift, height=height))
            out.append(Spectrum(mz, y, entry_, np.nan, rep, data_id=d,
                                spectrum_id=f"{d}:{rep}", role="undeut",
                                rt_start=rt[0], rt_end=rt[1]))
            if labelled:  # the same peaks, brighter (a copy would be refused)
                out.append(Spectrum(mz, 1.3 * y, entry_, 60.0, rep, data_id=d,
                                    spectrum_id=f"{d}:{rep}:60", rt_start=rt[0], rt_end=rt[1]))
    return SpectrumSet.from_spectra(out)


def verdicts(b):
    first = {}
    for i, d in enumerate(np.asarray(b.data_id)):
        first.setdefault(int(d), i)
    return {IONS[d]: (b.claim[i], b.claim_other[i], b.claim_offset_ppm[i],
                      b.claim_other_offset_ppm[i], bool(b.claim_by_position[i]))
            for d, i in first.items()}


def test_an_ion_on_another_listed_ions_peaks_is_lost_and_names_it():
    v = verdicts(pick_spectra(spectra_set(), CFG))
    claim, other, own_ppm, other_ppm, by_position = v[LOST]
    assert claim == "lost"
    assert other == ion_label(f"{IONS.index(CLAIMANT) + 1:04d}-"
                              f"{IONS.index(CLAIMANT) + len(CLAIMANT):04d}", CLAIMANT, 2)
    assert own_ppm == pytest.approx(-20.6, abs=1.5) and abs(other_ppm) < 1.5
    assert by_position  # exported spectra: no retention windows to compare
    assert v[CLAIMANT][0] == "own"  # 20 ppm is far beyond its own grid's scatter
    assert all(v[p][0] == "own" for p in BACKGROUND)


def test_isomers_name_each_other():
    v = verdicts(pick_spectra(spectra_set(), CFG))
    for a, b in (ISOMERS, ISOMERS[::-1]):
        claim, other, *_ = v[a]
        assert claim == "isomer" and f" {b} 2+" in other


def test_windows_apart_rule_a_claim_out():
    """The two ions' retention windows apart in the injection: 55-67 cannot be what 13-24's
    window read, so nobody listed claims 13-24's cluster (and nothing is flagged)."""
    apart = {LOST: (5.0, 5.5), CLAIMANT: (8.0, 8.5)}
    assert verdicts(pick_spectra(spectra_set(windows=apart), CFG))[LOST][0] == "unclaimed"
    together = {LOST: (5.0, 5.5), CLAIMANT: (5.2, 5.6)}
    claim, *_, by_position = verdicts(pick_spectra(spectra_set(windows=together), CFG))[LOST]
    assert claim == "lost" and not by_position


def test_a_cluster_off_every_grid_is_unclaimed():
    """13-24's windows hold 55-67's envelope 10 ppm further: on no listed ion's grid, so
    nobody claims it; no flag."""
    b = pick_spectra(_shifted_lost(np.random.default_rng(3)), CFG)
    assert verdicts(b)[LOST][0] == "unclaimed"
    flagged = set(np.asarray(b.data_id)[_flags(b)].tolist())
    assert flagged == {IONS.index(p) for p in ISOMERS}  # the isomers only


def _shifted_lost(rng):
    out = []
    for d, pep in enumerate(IONS):
        entry_ = PeptideEntry("f", "S", pep, f"{d + 1:04d}-{d + len(pep):04d}", d + 1,
                              d + len(pep), 2)
        for rep in (1, 2, 3):
            shift = 1.0 + rng.normal(0, 0.7) + (-10.0 if pep == LOST else 0.0)
            source = CLAIMANT if pep == LOST else pep
            mz, y = profile(pep, 2, envelope(source, shift_ppm=shift,
                                             height=10 ** rng.uniform(3.5, 4.5)))
            out.append(Spectrum(mz, y, entry_, np.nan, rep, data_id=d,
                                spectrum_id=f"{d}:{rep}", role="undeut"))
    return SpectrumSet.from_spectra(out)


def _flags(b):
    return np.isin(np.asarray(b.claim, dtype=object), ["lost", "isomer", "shared"])


def test_labelled_spectra_share_their_series_verdict():
    b = pick_spectra(spectra_set(labelled=True), CFG)
    lost = np.flatnonzero(np.asarray(b.data_id) == IONS.index(LOST))
    assert len(lost) == 6 and set(b.claim[lost]) == {"lost"}
    assert len(set(b.claim_other[lost])) == 1


def test_no_claims_without_the_mz_check_or_for_fragments():
    raw = spectra_set()
    b = pick_spectra(raw, dataclasses.replace(CFG, off_mz_sd=None))
    assert set(b.claim) == {""} and np.isnan(b.claim_offset_ppm).all()
    checked = series_offsets(raw, CFG)
    skipped = claims(raw, CFG, checked, skip=[IONS.index(LOST)])
    lost = np.asarray(raw.data_id) == IONS.index(LOST)
    assert set(skipped.claim[lost]) == {""}


def test_too_few_clusters_assess_nothing():
    """Four own clusters on their grid (fewer than MIN_CLUSTERS): no scatter, no claims."""
    raw = spectra_set()
    ids = [IONS.index(p) for p in (BACKGROUND[0], BACKGROUND[1])]
    keep = np.isin(np.asarray(raw.data_id), ids) & np.isin(np.asarray(raw.rep), [1, 2])
    assert set(pick_spectra(raw.select(keep), CFG).claim) == {""}


def test_the_mz_check_takes_out_a_tilt_with_charge():
    """Ions at 3+ read 6 ppm high and at 2+ not: one line in m/z cannot take that out, a term
    per charge does, so an off series at 3+ is judged against its own charge."""
    rng = np.random.default_rng(5)
    out = []
    peps = BACKGROUND
    for d, pep in enumerate(peps):
        for z in (2, 3):
            entry_ = PeptideEntry("f", "S", pep, f"{d + 1:04d}-{d + len(pep):04d}", d + 1,
                                  d + len(pep), z)
            for rep in (1, 2, 3):
                shift = rng.normal(0, 0.5) + (6.0 if z == 3 else 0.0)
                if pep == BACKGROUND[0] and z == 3:
                    shift += 8.0  # off its m/z by 8 ppm against its own charge
                mz, y = profile(pep, z, envelope(pep, z, shift_ppm=shift))
                out.append(Spectrum(mz, y, entry_, np.nan, rep, data_id=2 * d + (z - 2),
                                    spectrum_id=f"{d}:{z}:{rep}", role="undeut"))
    raw = SpectrumSet.from_spectra(out)
    checked = series_offsets(raw, CFG)
    # the 3+ ions sit at lower m/z than the 2+ ones, so the runs' line takes a share of the
    # tilt and the terms the rest: together they take it out
    assert checked.charge_terms[3] - checked.charge_terms[2] > 3.0
    off = dict(zip(checked.data_id.tolist(), checked.off.tolist(), strict=True))
    assert off[1]  # the 8 ppm series at 3+
    assert sum(off.values()) == 1
    rel = dict(zip(checked.data_id.tolist(), checked.relative_ppm.tolist(), strict=True))
    assert rel[1] == pytest.approx(8.0, abs=1.5)
    others = [abs(v) for d, v in rel.items() if d != 1]
    assert max(others) < 2.5  # the tilt taken out: every other series near 0


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A run of the set with labelled spectra (fits at K = 1, no bootstrap), loaded back."""
    cpu = Backend("cpu", "float64")
    fit = FitConfig(k_max=1, adam_steps=60, n_restarts=1, backend=cpu,
                    noise=NoiseConfig("gaussian", percent=3.0))
    out = tmp_path_factory.mktemp("claims") / "b"
    run(spectra_set(labelled=True), RunConfig(picker=CFG, fit=fit), out=out)
    return load_results(out)


def test_the_flags_follow_the_claims(results):
    q = results.quality
    data_id = np.asarray(results.fit.data_id)
    lost = data_id == IONS.index(LOST)
    iso = np.isin(data_id, [IONS.index(p) for p in ISOMERS])
    assert q.other_ion[lost].all() and not q.other_ion[~lost].any()
    assert q.isomer_overlap[iso].all() and not q.isomer_overlap[~iso].any()
    assert not q.shared_cluster.any()
    spec = spectra_table(results)
    assert "other_ion" in spec["flags"][lost][0].split(";")
    assert set(spec["claim"][lost]) == {"lost"} and spec["claim_by_position"][lost].all()
    t = series_table(results)
    row = list(t["peptide"]).index(LOST)
    assert t["claim"][row] == "lost" and CLAIMANT in t["claim_other"][row]


def test_the_analysis_leaves_a_lost_series_out_with_the_other_ion_named(results):
    left = left_out_spectra(results, SpeciesFilters(replicate_outlier=None))
    reasons = [str(r) for r in left["reason"]]
    other = [r for r in reasons if r.startswith(OTHER_ION_REASON)]
    assert len(other) == 6  # three controls and three labelled spectra
    assert CLAIMANT in other[0] and "ppm" in other[0] and "retention not known" in other[0]
    assert entry(other[0]) is not None  # the dictionary knows it, the note in brackets aside
    kept = left_out_spectra(results, SpeciesFilters(replicate_outlier=None, other_ion=False))
    assert not any(str(r).startswith(OTHER_ION_REASON) for r in kept["reason"])


def test_accepted_keeps_a_lost_series(results):
    lost = np.asarray(results.fit.spectrum_id)[np.asarray(results.fit.data_id)
                                               == IONS.index(LOST)]
    n = len(lost)
    accepted = results.with_review(ReviewTable(
        spectrum_id=np.array(lost, dtype=object), status=np.full(n, "accepted", dtype=object),
        k_override=np.zeros(n, dtype=np.int64), note=np.full(n, "", dtype=object),
        updated=np.full(n, "", dtype=object)))
    left = left_out_spectra(accepted, SpeciesFilters(replicate_outlier=None))
    assert not any(str(r).startswith(OTHER_ION_REASON) for r in left["reason"])


def test_the_project_file_turns_the_filter_off(tmp_path):
    from hxex.config import load_project

    path = tmp_path / "hxex.toml"
    path.write_text('project = "p"\n[analysis.filters]\nother_ion = false\n')
    assert load_project(path, analysis_only=True).analysis.filters.other_ion is False
    assert run_config_from({}).picker.off_mz_sd is not None
