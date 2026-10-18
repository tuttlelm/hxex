# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise-scale tests: the prototype's values as the oracle, plus the configuration rules."""

import json
from pathlib import Path

import numpy as np
import pytest

from hxex.noise import (
    NoiseConfig,
    control_references,
    kinetics_references,
    noise_scale,
    sample_size,
)

GOLD = np.load(Path(__file__).parent / "golden" / "model_reference.npz")
OBSERVED = [GOLD["obs_padded"][: s, i] for i, s in enumerate(GOLD["sizes"])]
REFS = [(0,), (0,), (0,), (3, 4), (3, 4), None]


@pytest.mark.parametrize(
    "key, config, refs",
    [
        ("noise_gauss_pct", NoiseConfig("gaussian", percent=3.0), REFS),
        ("noise_gauss_self", NoiseConfig("gaussian", percent=5.0), [None] * 6),
        ("noise_gauss_sd_scalar", NoiseConfig("gaussian", sd=123.0, percent=None), None),
        ("noise_gauss_sd_array", NoiseConfig("gaussian", sd=tuple(np.arange(1.0, 7.0)),
                                             percent=None), None),
        ("noise_poisson_neff", NoiseConfig("poisson", n_eff=500.0, percent=None), REFS),
        ("noise_poisson_pct", NoiseConfig("poisson", percent=5.0), REFS),
        ("noise_poisson_none", NoiseConfig("poisson", percent=None), None),
    ],
)
def test_noise_scale_matches_the_prototype(key, config, refs):
    np.testing.assert_allclose(noise_scale(OBSERVED, config, refs), GOLD[key], rtol=1e-13)


def test_noise_scale_by_hand():
    obs = [np.array([1.0, 4.0]), np.array([2.0, 10.0]), np.array([3.0, 3.0])]
    cfg = NoiseConfig("gaussian", percent=10.0)
    np.testing.assert_allclose(noise_scale(obs, cfg, [(0, 1), (0, 1), None]), [0.7, 0.7, 0.3])
    pois = NoiseConfig("poisson", percent=10.0)
    np.testing.assert_allclose(noise_scale(obs, pois, [(0, 1), (0, 1), None]),
                               [0.01 * 7.0, 0.01 * 7.0, 0.01 * 3.0])


def test_reference_is_required_when_the_scale_comes_from_one():
    with pytest.raises(ValueError, match="references are required"):
        noise_scale(OBSERVED, NoiseConfig("gaussian", percent=3.0))
    with pytest.raises(ValueError, match="one entry per measurement"):
        noise_scale(OBSERVED, NoiseConfig(), [(0,)])
    with pytest.raises(ValueError, match="empty reference"):
        noise_scale(OBSERVED, NoiseConfig(), [()] * 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "laplace"},
        {"model": "gaussian", "percent": None},  # needs sd or percent
        {"model": "gaussian", "sd": 5.0},  # both
        {"model": "gaussian", "sd": -1.0, "percent": None},
        {"model": "gaussian", "n_eff": 100.0, "percent": None, "sd": 5.0},
        {"model": "poisson", "sd": 5.0, "percent": None},
        {"model": "poisson", "n_eff": 100.0, "percent": 3.0},
        {"model": "poisson", "n_eff": 0.0, "percent": None},
        {"model": "gaussian", "percent": 0.0},
    ],
)
def test_noise_config_validation(kwargs):
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)


def test_noise_config_defaults_roundtrip_and_flags():
    assert NoiseConfig() == NoiseConfig("gaussian", percent=3.0)
    assert NoiseConfig().needs_reference
    assert not NoiseConfig("gaussian", sd=5.0, percent=None).needs_reference
    assert not NoiseConfig("poisson", percent=None).needs_reference
    for cfg in (NoiseConfig(), NoiseConfig("gaussian", sd=(1.0, 2.0), percent=None),
                NoiseConfig("poisson", n_eff=300.0, percent=None)):
        data = json.loads(json.dumps(cfg.to_dict()))
        assert NoiseConfig.from_dict(data) == cfg
    with pytest.raises(ValueError, match="unknown noise setting"):
        NoiseConfig.from_dict({"sigma": 1.0})


def test_control_references():
    un = "undeut"
    refs = control_references([0, 0, 0, 1, 1, 2], [un, "", un, un, "", ""])
    assert refs == [(0, 2), (0, 2), (0, 2), (3,), (3,), None]
    assert control_references([0], ["fulldeut"]) == [None]  # only the undeuterated serve
    with pytest.raises(ValueError, match="one entry per measurement"):
        control_references([0, 1], [un])
    assert control_references([], []) == []



def test_kinetics_references():
    # series 0: a weak control (0) and kinetics scans of replicates 1 (1, 2, 3) and 2 (4, 5);
    # series 1: a control only
    tallest = [2.0, 90.0, 100.0, 120.0, 60.0, 50.0, 3.0]
    observed = [np.array([0.5, t, 1.0]) for t in tallest]
    refs = kinetics_references([0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 2, 2, 1], observed,
                               [False, True, True, True, True, True, False])
    # the median of 90, 100, 120 is scan 2; of 60 and 50 the lower one, scan 5
    assert refs == [None, (2,), (2,), (2,), (5,), (5,), None]
    assert kinetics_references([0], [1], [np.array([1.0])], [False]) == [None]
    with pytest.raises(ValueError, match="one entry"):
        kinetics_references([0, 0], [1], observed[:2], [True, True])
    scale = noise_scale(observed, NoiseConfig(), refs)  # 3 % of the reference's tallest peak
    np.testing.assert_allclose(scale[1:4], 3.0)


def test_sample_size():
    scaled = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 4.0])]
    np.testing.assert_array_equal(sample_size(scaled, NoiseConfig()), [3.0, 2.0])
    np.testing.assert_array_equal(
        sample_size(scaled, NoiseConfig("poisson", percent=None)), [6.0, 8.0]
    )
