# SPDX-License-Identifier: GPL-3.0-or-later
"""Retention times: kept with the spectra (peptide table column, table of their own, scans),
through picking and the bundle to the species table, and used by the theoretical
back-exchange fallback when every control has one."""

import dataclasses
import logging

import numpy as np
import pytest

from hxex.io import ReadError, apply_retention_times, read_peptide_table, read_retention_times
from hxex.spectra import PeptideEntry, Spectrum, SpectrumSet, entry_rt


def spectra(rts=(None, None, None)):
    out = []
    for i, (sample, peptide, start, rt) in enumerate(zip(
            ("WT", "WT", "M"), ("AKLV", "GSTVED", "AKLV"), (2, 6, 2), rts, strict=True)):
        extra = {} if rt is None else {"rt": str(rt)}
        entry = PeptideEntry("f", sample, peptide, f"{start}-{start + len(peptide) - 1}", start,
                             start + len(peptide) - 1, 2, extra=extra)
        out.append(Spectrum(np.array([1.0, 2.0]), np.array([3.0, 4.0]), entry, 0.0, 1,
                            data_id=i, spectrum_id=f"p:{i}"))
    return SpectrumSet.from_spectra(out)


def test_spectra_keep_the_peptide_tables_rt(tmp_path):
    from hxex.io import load_spectra, save_spectra

    s = spectra((12.5, None, 13.0))
    assert s.rt[0] == 12.5 and np.isnan(s.rt[1]) and s.select(np.array([0, 0, 1], bool)).rt[0] \
        == 13.0
    # a set built without retention times has none
    bare = SpectrumSet(**{f.name: getattr(s, f.name) for f in dataclasses.fields(s)
                          if f.name not in ("rt", "source")})
    assert np.isnan(bare.rt).all()
    back = load_spectra(save_spectra(s, tmp_path / "s.parquet"))
    assert np.array_equal(back.rt, s.rt, equal_nan=True)
    # a peptide table column, checked when read
    path = tmp_path / "metadf.csv"
    path.write_text("file,sample,peptide,peptide_range,start_seq,end_seq,charge,rt\n"
                    "f,WT,AKLV,2-5,2,5,2,12.5\nf,WT,GSTVED,6-11,6,11,2,\n")
    entries = read_peptide_table(path)
    assert entry_rt(entries[0]) == 12.5 and np.isnan(entry_rt(entries[1]))
    path.write_text("file,sample,peptide,peptide_range,start_seq,end_seq,charge,rt\n"
                    "f,WT,AKLV,2-5,2,5,2,early\n")
    with pytest.raises(ReadError, match="line 2"):
        read_peptide_table(path)


def test_a_table_of_retention_times(tmp_path, caplog):
    path = tmp_path / "rt.csv"
    path.write_text("peptide_range,peptide,sample,charge,rt\n"
                    "2-5,AKLV,,,10.0\n"      # every sample
                    "2-5,AKLV,M,,11.0\n"     # but M (more specific)
                    "6-11,GSTVED,WT,3,9.0\n"  # charge 3: matches nothing
                    "50-60,,,,1.0\n")         # no such peptide
    rows = read_retention_times(path)
    assert rows[1] == {"rt": 11.0, "sample": "M", "start_seq": 2, "end_seq": 5,
                       "peptide": "AKLV"}
    with caplog.at_level(logging.WARNING):
        out = apply_retention_times(spectra((99.0, 7.0, None)), rows)
    assert out.rt.tolist()[0] == 10.0 and out.rt[2] == 11.0 and out.rt[1] == 7.0  # kept
    assert "2 row(s) match no spectrum" in caplog.text
    path.write_text("peptide,rt\nAKLV,1\nAKLV,2\n")
    with pytest.raises(ReadError, match="two retention times"):
        apply_retention_times(spectra(), read_retention_times(path))
    # different columns, the same number of them: refused, both rows named
    path.write_text("peptide,charge,sample,rt\nAKLV,2,,1\nAKLV,,WT,2\n")
    with pytest.raises(ReadError, match=r"\(peptide AKLV, charge 2, rt 1.*\(sample WT, "
                                        r"peptide AKLV, rt 2.*on 2 column"):
        apply_retention_times(spectra(), read_retention_times(path))
    for bad in ("peptide\nAKLV\n", "sample,rt\nWT,1\n"):
        path.write_text(bad)
        with pytest.raises(ReadError):
            read_retention_times(path)
    # a row with fewer or more cells than the header, or naming no spectrum, is refused
    for bad, match in (("peptide,sample,rt\nAKLV,WT\n", "fewer cells"),
                       ("peptide,rt\nAKLV,1,2\n", "more cells"),
                       ("peptide,sample,rt\n,,5\n", "every spectrum")):
        path.write_text(bad)
        with pytest.raises(ReadError, match=f"line 2: .*{match}"):
            read_retention_times(path)


def test_rt_reaches_the_bundle_and_the_species_table(tmp_path):
    from hxex.analysis import species_uptake
    from hxex.io import open_bundle
    from hxex.report import Results
    from hxex.run import run
    from test_run import config, raw_spectra

    raw = raw_spectra(n_series=2)
    rt = np.where(raw.data_id == 0, 20.0, 31.0) + 0.1 * np.arange(len(raw))
    raw = dataclasses.replace(raw, rt=rt)
    out = tmp_path / "bundle"
    result = run(raw, config(bootstrap=None), out=out)
    assert np.array_equal(result.binned.rt, rt)
    with open_bundle(out) as b:
        res = Results.from_bundle(b)
        assert np.array_equal(res.binned.rt, rt)
        up = species_uptake(res, ks=(1,))
    # both series are one species (the same sample and peptide): the median of its spectra
    assert np.allclose(up.rt, np.median(rt))
    assert "rt" in up.columns()
    # over the spectra the table uses: a rejected one does not count
    from hxex.io.bundle import ReviewTable

    rejected = ReviewTable.empty().update(str(res.fit.spectrum_id[0]), status="rejected")
    without = species_uptake(res.with_review(rejected), ks=(1,))
    assert np.allclose(without.rt, np.median(rt[1:])) and not np.allclose(without.rt, up.rt)


def test_kint_fallback_adds_the_retention_time_when_every_control_has_one():
    from hxex.controls import ControlConfig, _kint_predict, fit_controls, series_rates, series_rt
    from hxex.kint import ExchangeConditions
    from test_controls import PEPTIDES, QUENCH, fake_fit

    rates = series_rates(PEPTIDES, ["" for _ in PEPTIDES], range(len(PEPTIDES)),
                         ExchangeConditions(**QUENCH))
    rts = {d: 5.0 + 2.0 * d for d in rates}
    # FD controls that lose more the later the peptide elutes: t = 1500 + 2 x RT(s)
    a, t0, b = 0.85, 1500.0, 2.0
    rows, n = [], []
    for d, k in rates.items():
        fd = float(_kint_predict(a, t0, [k], [rts[d]], b)[0])
        rows += [(d, 0.0, 0.05), (d, 0.0, 0.05)]
        if d != 5:  # series 5 has no FD control
            rows += [(d, 1e6, 0.05 + fd), (d, 1e6, 0.05 + fd)]
        n += [len(k)] * (2 if d == 5 else 4)
    fit, n = fake_fit(rows), np.array(n, float)
    cfg = ControlConfig(fd_fallback="kint", quench=QUENCH)
    with_rt = fit_controls(fit, n, cfg, rates=rates, rt=rts)
    m = with_rt.fd_model
    assert m["b_rt"] == pytest.approx(b, rel=0.05) and m["t0"] == pytest.approx(t0, rel=0.05)
    i = with_rt.rows([5])[0]  # missing: predicted at its own retention time
    want = 0.05 + float(_kint_predict(m["a"], m["t0"], [rates[5]], [rts[5]], m["b_rt"])[0])
    assert with_rt.mfd[i] == pytest.approx(want)
    # some controls without one: taken at the median of the others
    partial = fit_controls(fit, n, cfg, rates=rates, rt={d: v for d, v in rts.items() if d})
    assert partial.fd_model["b_rt"] > 0
    assert partial.fd_model["median_rt"] == pytest.approx(np.median(
        [v for d, v in rts.items() if d and d != 5]))
    # without retention times, or with too few: the plain model
    few = dict(list(rts.items())[:3])
    for given in (None, few):
        assert fit_controls(fit, n, cfg, rates=rates, rt=given).fd_model["b_rt"] is None
    assert series_rt([0, 0, 1, 2], [1.0, 3.0, np.nan, 4.0]) == {0: 2.0, 2: 4.0}
