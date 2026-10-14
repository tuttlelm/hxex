# SPDX-License-Identifier: GPL-3.0-or-later
#
# The rate model is ported from pacilab/hdx-rates-mixtures (python/utils_kint.py, utils_pL.py and
# their tables), which is under the MIT License:
#
#   Copyright (c) 2025 pacilab
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#   and associated documentation files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Intrinsic (unprotected) amide hydrogen exchange rates in H2O/D2O mixtures.

The model is that of Grimaldi, Stofella and Paci, J. Phys. Chem. B 130, 2493 (2026), ported from
their code (pacilab/hdx-rates-mixtures): the reference rates and neighbour factors of Bai et al.
(Proteins 17, 75, 1993) and Connelly et al. (Proteins 17, 87, 1993) for the four isotope pairs
(the amide's hydrogen and the catalyst's), the mixture's ionic product (Sweeton et al. 1974; Gold
and Lowe 1967) and the D fractions of L+ and OL- (Gold 1969). Each amide has a forward rate
(H to D, ``kforw``) and a back rate (D to H, ``kback``); in a mixture an unprotected amide relaxes
at ``kforw + kback`` towards the D fraction ``kforw / (kforw + kback)``, so a decay (or rise) of
its deuterium follows that sum, not either rate alone.

Differences from the reference code: sequences are modX residues with terminal groups
(:func:`hxex.chem.split_termini`); an acetylated N-terminus gives residue 1 an amide, with the
acetyl group's factors, and no N-terminal factor on residue 2; a C-terminal ``-NH2`` has no
C-terminal factor (Bai's is for the carboxyl group) and its last residue is marked
``approximate``; a modified residue takes the factors of its standard residue and is marked
``approximate`` (no published factors for acK, pS, pT or pY; HISTORY item 70) unless
:attr:`ExchangeConditions.factors` gives them; a two-residue peptide gets both terminal factors;
no intermediate value is rounded; residues without an amide hydrogen (the first, with a free
amine, and prolines) have NaN rates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType

import numpy as np

from . import chem

REFERENCES = ("PDLA", "3Ala")

_R = 1.987  # cal / (mol K)
_T_REF = 293.0  # K, of the reference rates
_T_REF_PKC = 278.0  # K, of the side-chain pKc values
_EA = {"A": 14000.0, "B": 17000.0, "W": 19000.0}  # cal/mol: acid, base, water catalysis

# log10 of the reference rates (1/min) of poly-DL-alanine, per isotope pair: the amide's
# hydrogen, then the solvent's (HD: an amide H exchanging in D2O)
_LOG_KREF = {
    "HH": (1.39, 10.08, -1.6),
    "DH": (1.40, 10.00, -1.6),
    "HD": (1.62, 10.18, -1.5),
    "DD": (1.63, 10.10, -1.5),
}
_3ALA = (2.34, 1.35, 1.585)  # factors on (kA, kB, kW) for the 3-Ala reference

# Bai et al. (1993) factors, log10: (AL, AR, BL, BR). L: the residue's own side chain on its
# amide; R: the preceding residue's side chain. D, E and H are pH dependent (charged and neutral
# forms below), as is the acid factor of the C-terminal carboxyl.
_FACTORS = {
    "A": (0.00, 0.00, 0.00, 0.00),
    "C": (-0.54, -0.46, 0.62, 0.55),
    "F": (-0.52, -0.43, -0.24, 0.06),
    "G": (-0.22, 0.22, -0.03, 0.17),
    "I": (-0.91, -0.59, -0.73, -0.23),
    "K": (-0.56, -0.29, -0.04, 0.12),
    "L": (-0.57, -0.13, -0.58, -0.21),
    "M": (-0.64, -0.28, -0.01, 0.11),
    "N": (-0.58, -0.13, 0.49, 0.32),
    "P": (np.nan, -0.19, np.nan, -0.24),
    "Q": (-0.47, -0.27, 0.06, 0.20),
    "R": (-0.59, -0.32, 0.08, 0.22),
    "S": (-0.44, -0.39, 0.37, 0.30),
    "T": (-0.79, -0.47, -0.07, 0.20),
    "V": (-0.74, -0.30, -0.70, -0.14),
    "W": (-0.40, -0.44, -0.41, -0.11),
    "Y": (-0.41, -0.37, -0.27, 0.05),
}
_CHARGED = {  # (protonated, neutral) factors and pKc reference: (c in H2O, c in D2O, Ea)
    "D": ((-0.90, -0.12, 0.69, 0.60), (0.90, 0.58, 0.10, -0.18), (-3.87, -4.48, 960.0, 1000.0)),
    "E": ((-0.60, -0.27, 0.24, 0.39), (-0.90, 0.31, -0.11, -0.15), (-4.33, -4.93, 1083.0, 1083.0)),
    "H": ((-0.80, -0.51, 0.80, 0.83), (0.00, 0.00, -0.10, 0.14), (-7.00, -7.42, 7500.0, 7500.0)),
}
_NT = (-1.32, 1.62)  # (AR, BR) of a free N-terminal amine, on residue 2's amide
_AC = (0.29, -0.20)  # (AR, BR) of an N-terminal acetyl group, on residue 1's amide
_CT_ACID = (0.05, 0.96)  # AL of the C-terminal carboxyl, protonated and not (pKc of Glu)
_CT_BASE = -1.80  # BL of the C-terminal carboxyl


@dataclass(frozen=True)
class ExchangeConditions:
    """The solvent an exchange happens in: ``ph_read`` (the glass-electrode reading in the
    mixture, uncorrected), ``temperature`` (degrees Celsius), ``d2o_fraction`` (the deuterium
    atom fraction of the solvent, 0 to 1) and ``reference`` (``'PDLA'``, poly-DL-alanine, the
    default, or ``'3Ala'``). ``factors``: Bai factors ``(AL, AR, BL, BR)``, log10, for modified
    residues (label -> four numbers), replacing those of their standard residue.
    ``initial_fraction``: the D fraction the amides carried when this exchange began (for a
    back-exchange, the labelling solvent's), the start of the theoretical curves
    (:func:`expected_uptake`); None: not known, no curves. :func:`intrinsic_rates` ignores it."""

    ph_read: float
    temperature: float
    d2o_fraction: float
    reference: str = "PDLA"
    factors: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: MappingProxyType({}))
    initial_fraction: float | None = None

    def __post_init__(self):
        for name in ("ph_read", "temperature", "d2o_fraction"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not 0 < self.ph_read < 14:
            raise ValueError(f"ph_read must be between 0 and 14, got {self.ph_read}")
        if not -30 <= self.temperature <= 100:
            raise ValueError(f"temperature is in degrees Celsius (-30 to 100), got "
                             f"{self.temperature}")
        if not 0 <= self.d2o_fraction <= 1:
            raise ValueError(f"d2o_fraction must be between 0 and 1, got {self.d2o_fraction}")
        if self.initial_fraction is not None:
            object.__setattr__(self, "initial_fraction", float(self.initial_fraction))
            if not 0 <= self.initial_fraction <= 1:
                raise ValueError(f"initial_fraction must be between 0 and 1, got "
                                 f"{self.initial_fraction}")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {', '.join(REFERENCES)}")
        factors = {}
        for label, values in dict(self.factors).items():
            values = tuple(float(v) for v in values)
            if len(values) != 4 or not np.all(np.isfinite(values)):
                raise ValueError(f"factors of {label!r} must be four numbers (AL, AR, BL, BR)")
            factors[str(label)] = values
        object.__setattr__(self, "factors", MappingProxyType(factors))

    def to_dict(self) -> dict:
        return {"ph_read": self.ph_read, "temperature": self.temperature,
                "d2o_fraction": self.d2o_fraction, "reference": self.reference,
                "factors": {k: list(v) for k, v in self.factors.items()},
                "initial_fraction": self.initial_fraction}

    @classmethod
    def from_dict(cls, data: Mapping) -> ExchangeConditions:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown exchange setting(s): {', '.join(sorted(unknown))}")
        return cls(**dict(data))


def sample_conditions(
    conditions: ExchangeConditions | Mapping[str, ExchangeConditions] | None, sample: str
) -> ExchangeConditions | None:
    """The conditions of ``sample``: ``conditions`` itself when it is one set for every sample,
    its entry for ``sample`` when it is a mapping (None when the sample has none)."""
    if conditions is None or isinstance(conditions, ExchangeConditions):
        return conditions
    return conditions.get(sample)


def read_conditions(data) -> ExchangeConditions | dict[str, ExchangeConditions] | None:
    """Conditions from plain data (a project file): one set for every sample (a table with
    ``ph_read``), or one per sample (sample -> table); objects are passed through."""
    if data is None or isinstance(data, ExchangeConditions):
        return data
    if not isinstance(data, Mapping):
        raise ValueError("exchange conditions must be a table")
    if "ph_read" in data or not data:
        return ExchangeConditions.from_dict(data)
    out = {}
    for sample, value in data.items():
        if not isinstance(value, (Mapping, ExchangeConditions)):
            raise ValueError(f"exchange conditions of sample {sample!r} must be a table "
                             f"(ph_read, temperature, d2o_fraction)")
        out[str(sample)] = (value if isinstance(value, ExchangeConditions)
                            else ExchangeConditions.from_dict(value))
    return out


def conditions_dict(conditions) -> dict | None:
    """Plain data of :func:`read_conditions`' result."""
    if conditions is None:
        return None
    if isinstance(conditions, ExchangeConditions):
        return conditions.to_dict()
    return {sample: c.to_dict() for sample, c in conditions.items()}


@dataclass(frozen=True)
class IntrinsicRates:
    """Per residue of a sequence: ``residue`` (numbered from the ``start`` given), ``aa`` (the
    modX label), ``kforw`` (H to D) and ``kback`` (D to H) in 1/s (NaN: no amide hydrogen) and
    ``approximate`` (a factor was borrowed: a modified residue without its own factors, on its
    amide or the next, or the last residue before a C-terminal amide)."""

    residue: np.ndarray
    aa: np.ndarray
    kforw: np.ndarray
    kback: np.ndarray
    approximate: np.ndarray
    conditions: ExchangeConditions

    def __len__(self) -> int:
        return len(self.residue)

    @property
    def rate(self) -> np.ndarray:
        """``kforw + kback``: the rate at which an unprotected amide relaxes in this solvent."""
        return self.kforw + self.kback

    @property
    def equilibrium(self) -> np.ndarray:
        """``kforw / (kforw + kback)``: the D fraction an unprotected amide relaxes towards."""
        return self.kforw / self.rate

    def columns(self) -> dict[str, np.ndarray]:
        return {"residue": self.residue, "aa": self.aa, "kforw": self.kforw,
                "kback": self.kback, "rate": self.rate, "equilibrium": self.equilibrium,
                "approximate": self.approximate}

    def at(self, residues: Sequence[int]) -> np.ndarray:
        """The positions of ``residues`` (numbers) in this table."""
        residues = np.asarray(residues, dtype=np.int64)
        pos = residues - int(self.residue[0])
        if np.any((pos < 0) | (pos >= len(self))):
            raise ValueError(f"residues {residues.tolist()} are not all in "
                             f"{int(self.residue[0])}-{int(self.residue[-1])}")
        return pos


def _arrhenius(t: float, t_ref: float, ea: float) -> float:
    return float(np.exp(-ea * (1 / t - 1 / t_ref) / _R))


def _solvent(t: float, x: float, ph_read: float) -> tuple[float, float, float, float]:
    """``[L+]``, ``[OL-]`` and the D fractions of L+ and OL- in the mixture."""
    pkw = 670.86 - 34691.6 / t - 105.15 * np.log(t) + 0.10757 * t + 2358000 / t**2
    pkx = pkw + 0.7282 * x + 0.05122 * x**2 + 0.0826 * x**3
    pl = ph_read * pkx / pkw
    f_l = x * 0.69 / (1 - x + x * 0.69)
    f_ol = x * 0.47 / (1 - x + x * 0.47)
    return 10.0**-pl, 10.0 ** (pl - pkx), f_l, f_ol


def _ionised(protonated, neutral, pkc: float, pl: float) -> tuple[float, ...]:
    """A pH-dependent factor: log10 of the forms' factors weighted by their populations."""
    p, n = np.asarray(protonated), np.asarray(neutral)
    return tuple(np.log10((10.0 ** (p - pl) + 10.0 ** (n - pkc)) / (10.0**-pkc + 10.0**-pl)))


def _tables(t: float, pl: float, factors: Mapping) -> dict[str, dict]:
    """The factor table for amides in H2O-like (``'H'``) and D2O-like (``'D'``) solvent: the
    side-chain pKc values of D, E and H (and so of the C-terminal carboxyl) differ."""
    out = {}
    for solvent, k in (("H", 0), ("D", 1)):
        table = dict(_FACTORS)
        pkc = {}
        for aa, (protonated, neutral, ref) in _CHARGED.items():
            pkc[aa] = -np.log10(10.0 ** ref[k] * _arrhenius(t, _T_REF_PKC, ref[2 + k]))
            table[aa] = _ionised(protonated, neutral, pkc[aa], pl)
        table.update(factors)
        ct_acid = np.log10((10.0 ** (_CT_ACID[0] - pl) + 10.0 ** (_CT_ACID[1] - pkc["E"]))
                           / (10.0 ** -pkc["E"] + 10.0**-pl))
        out[solvent] = {"table": table, "ct": (float(ct_acid), _CT_BASE)}
    return out


def intrinsic_rates(
    sequence: str,
    conditions: ExchangeConditions,
    *,
    start: int = 1,
    registry: chem.ModificationRegistry = chem.DEFAULT_MODIFICATIONS,
) -> IntrinsicRates:
    """The intrinsic forward and back exchange rates of every backbone amide of ``sequence`` (a
    modX sequence, terminal groups included: ``'Ac-PEPpSTIDE-NH2'``) in the solvent of
    ``conditions``; residues are numbered from ``start``. Use the sequence the exchange happened
    in: for fragment ions, their parent's in solution."""
    nterm, _, cterm = chem.split_termini(sequence)
    labels = chem.parse_sequence(sequence, registry)
    n = len(labels)
    own = conditions.factors

    def base(label: str) -> str:
        return label if label in own else registry.base_residue(label)

    t = conditions.temperature + 273.15
    x = conditions.d2o_fraction
    conc_l, conc_ol, f_l, f_ol = _solvent(t, x, conditions.ph_read)
    pl = -np.log10(conc_l)
    tables = _tables(t, pl, own)

    kref = {}
    for pair, logs in _LOG_KREF.items():
        k = np.array([10.0**v / 60 for v in logs])
        k *= [_arrhenius(t, _T_REF, _EA[c]) for c in "ABW"]
        if conditions.reference == "3Ala":
            k *= _3ALA
        kref[pair] = k

    def log_factors(solvent: str) -> tuple[np.ndarray, np.ndarray]:
        tab = tables[solvent]["table"]
        ct_a, ct_b = tables[solvent]["ct"]
        log_a = np.full(n, np.nan)
        log_b = np.full(n, np.nan)
        for i in range(n):
            if i == 0 and nterm != "Ac-":
                continue
            al, _, bl, _ = tab[base(labels[i])]
            if i == 0:
                ar, br = _AC
            else:
                _, ar, _, br = tab[base(labels[i - 1])]
            a, b = al + ar, bl + br
            if i == 1 and nterm == "H-":
                a, b = a + _NT[0], b + _NT[1]
            if i == n - 1 and cterm == "-OH":
                a, b = a + ct_a, b + ct_b
            log_a[i], log_b[i] = a, b
        return 10.0**log_a, 10.0**log_b

    a_h, b_h = log_factors("H")  # pairs HH and DH (solvent hydrogen H)
    a_d, b_d = log_factors("D")  # pairs HD and DD
    hh, dh, hd, dd = (kref[p] for p in ("HH", "DH", "HD", "DD"))  # each (kA, kB, kW)
    kforw = (x * hh[1] * b_h * conc_ol * (1 - f_ol)  # H removed by OH-
             + hd[0] * a_d * conc_l * f_l  # acid catalysis by D+
             + x * hd[1] * b_d * conc_ol * f_ol  # H removed by OD-
             + x * hd[2] * b_d)  # water
    kback = (dh[0] * a_h * conc_l * (1 - f_l)  # acid catalysis by H+
             + (1 - x) * dh[1] * b_h * conc_ol * (1 - f_ol)  # D removed by OH-
             + (1 - x) * dh[2] * b_h  # water
             + (1 - x) * dd[1] * b_d * conc_ol * f_ol)  # D removed by OD-

    modified = np.array([lab != registry.base_residue(lab) and lab not in own for lab in labels])
    approximate = modified.copy()
    approximate[1:] |= modified[:-1]
    if cterm == "-NH2":
        approximate[-1] = True
    approximate &= np.isfinite(kforw)
    return IntrinsicRates(
        residue=np.arange(start, start + n, dtype=np.int64),
        aa=np.array(labels, dtype=str),
        kforw=kforw,
        kback=kback,
        approximate=approximate,
        conditions=conditions,
    )


def expected_uptake(
    rates: IntrinsicRates,
    residues: Sequence[int],
    times,
    initial: float | Sequence[float],
) -> np.ndarray:
    """The deuterium (deuterons) held by the amides of ``residues`` at ``times`` (s) if they
    were unprotected: each relaxes from its ``initial`` D fraction (one value, or one per
    residue) to :attr:`IntrinsicRates.equilibrium` at :attr:`IntrinsicRates.rate`. With
    ``initial`` the labelling solvent's D fraction this is the theoretical reference of a
    back-exchange experiment."""
    pos = rates.at(residues)
    k = rates.rate[pos]
    if not np.all(np.isfinite(k)):
        missing = np.asarray(residues)[~np.isfinite(k)].tolist()
        raise ValueError(f"residues {missing} have no amide hydrogen")
    eq = rates.equilibrium[pos]
    d0 = np.broadcast_to(np.asarray(initial, dtype=float), k.shape)
    t = np.asarray(times, dtype=float)
    return np.sum(eq + (d0 - eq) * np.exp(-np.multiply.outer(t, k)), axis=-1)
