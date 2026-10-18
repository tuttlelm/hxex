# Peak picking

How hxex turns a raw mass spectrum into the binned deuterium sticks that the fit uses, why it does
it this way, and how this differs from pyHXExpress, from which the method descends. The code is in
`hxex.picking` (`pick_spectrum`, `pick_spectra`, `plan_bins`) and the settings in `PickerConfig`.

## Objective

For one ion (a peptide, a fragment or an intact protein at one charge) the picker reports, for
each deuterium level n = 0, 1, 2, ..., the height of the isotope peak at the m/z where that
peak is expected. The fit then explains these heights as a mixture of binomial populations
convolved with the natural isotope envelope. The picker therefore has to (a) predict where each
peak is, and (b) measure the peak found there, without taking intensity from a neighbouring peak or
another species.

## Input

- **A spectrum:** m/z and intensity arrays, profile or centroided, as read from HX-Express
  spreadsheets, HDExaminer SpecExport files or mzML files. For mzML, scans can be summed first
  (see "Summing scans" below).
- **The ion:** its sequence (with modified residues written as modX labels such as `pS` or `acK`),
  charge z, ion type (`M` for the intact peptide or protein, or a fragment type such as `c` or
  `z-dot`) and modifications. From these come the elemental composition, the m/z of the
  undeuterated monoisotopic ion and the number N of exchangeable sites (backbone amides, minus the
  first `nterm_subtract` residues and prolines, plus any user-set `Hex` sites).

## Bins

Bin n is the isotope cluster of nominal mass +n above the undeuterated monoisotopic ion: every
isotopologue of the ion whose extra neutrons (from deuterium and from the natural heavy isotopes
13C, 15N, 18O, 34S, ...) add up to n. Bins are predicted up to N + Hex + (number of natural isotope
clusters - 1) + `zero_filling`, the most the ion could reach; picking normally ends earlier (see
"Extent").

## Where each bin is: the composition ladder

The m/z of bin n is the abundance-weighted mass of its cluster:

    m/z(n) = m/z(0) + M(n) / z

    M(n) = sum_k a(n-k) b(k) [ k * dD + m(n-k) ] / sum_k a(n-k) b(k)

where

- a(j) and m(j) are the abundance and the abundance-weighted mass offset (from the first cluster)
  of the natural isotope cluster j of the ion's non-exchangeable composition (the composition with
  the N exchangeable hydrogens removed). They are computed with the BRAIN algorithm (Dittwald et
  al., Anal. Chem. 2013) as implemented in brainpy;
- b(k) is the probability that k of the N exchangeable sites carry deuterium, taken as a binomial
  distribution with mean d (the deuteration level of the spectrum);
- dD = 1.00627675 Da is the mass of deuterium minus hydrogen.

Bins beyond the reach of both distributions continue by one deuterium step (dD / z) each.

**Why.** A cluster mixes deuterated and naturally heavy isotopologues, and these differ in mass:
one deuterium adds 1.00628 Da, one 13C 1.00336 Da, one 15N 0.99703 Da. The spacing of real peaks
therefore depends on how much of the envelope comes from deuterium. For an undeuterated peptide
the natural clusters are 1.0025 to 1.0029 Da apart (decreasing slowly along the envelope); for a
heavily deuterated small peptide the middle of the envelope is spaced by 1.0063 Da; for an intact
protein, whose natural envelope spans dozens of clusters, the spacing stays close to the natural
one. Measured spacings in real data agree: about 1.001 to 1.0025 Da for undeuterated peptides and
1.0044 to 1.0054 Da for deuterated ones (an unpublished SpecExport peptide dataset).

**The deuteration level.** d is not known before picking. The picker therefore:

1. picks once with the ExMS ladder (below);
2. takes the intensity-weighted centre of the bins above the noise threshold, subtracts the centre
   of the natural envelope, and uses the result (limited to 0 to N) as d;
3. computes the composition ladder for that d and picks again;
4. repeats steps 2 and 3 once more from those picks (a first pass that missed peaks biases d; the
   second estimate removes most of that bias: on synthetic spectra the positions end within 0.4 ppm
   of the true ones).

The data enter only through this single overall level, never through assigning individual peaks
to bins, so a co-eluting species cannot pull the predicted positions toward itself. The binomial is
an approximation for spectra with several populations; the positions depend on d only weakly (the
difference between a natural and a deuterium step, 0.0035 Da per bin, divided by z).

## Ion-type mixtures (fragments)

In fragment spectra one fragment often appears as several ion types that differ by a hydrogen,
for example `c-1`, `c` (c′) and `c+1` in ETD, or `z`, `z+1` (z•) and `z+2` (z′), from hydrogen
transfer between the two fragments. Their envelopes overlap one hydrogen apart, so neither the
picker's positions nor the fit's envelope are those of one ion type. In the undeuterated
reference of a 12-residue peptide fragmented by ETD, short c and z fragments were almost purely
c′ and z•, while the longest ones were about one third c′ (z•) and two thirds c+1 (z′); a single
ion type left a misfit as large as the envelope itself, three types less than 3 %.

The envelope of a series is then a mixture of `components` ion types: the given ion type (the
lightest) and the same plus 1, 2, ... hydrogens, with weights w(i):

    a_mix(n) = sum_i w(i) a(n-i)
    m_mix(n) = sum_i w(i) a(n-i) [ i * mH + m(n-i) ] / a_mix(n)

where a and m are the natural clusters of the lightest type (as above) and mH = 1.00782503 Da is
the mass of a hydrogen atom. The mixed clusters replace the natural ones everywhere: in the
composition ladder (so the bins are placed for the mixture) and as the envelope of the fit.

**Weights.** They are estimated per series from its reference spectra, normally the undeuterated
control: starting from equal weights, the references are picked with the bins of the current
mixture, their envelopes (each scaled to sum to 1) are averaged, and the weights are fitted to
that average by non-negative least squares; this is repeated three times. The fit's relative
residual (the RMS misfit over the RMS of the average envelope) decides whether the reference is
good (`max_residual`, 0.15). If the undeuterated control is missing or poor, the spectra at a
fallback time are used instead (for example a fully exchanged end-state run), and the series is
marked; without either, the series uses its theoretical envelope and is marked too. As in
pyHXExpress, the proportions of the ion types are assumed not to change with deuteration.

**Inferred envelope.** With `mode="infer"` the fit uses the averaged reference envelope itself
instead of the fitted mixture (the bins are still placed with the fitted mixture); it follows
the reference exactly, noise included.

These options follow pyHXExpress's `NA_envelope` settings (`mixedN` and `infer`), with two
differences: the components are one hydrogen mass apart rather than one ExMS step, and the
mixture also places the bins. They are set by `RunConfig.envelope` (`EnvelopeConfig`: `mode`
`theory`, `mixed` or `infer`; `components`; `max_residual`) and `RunConfig.envelope_fallback_time`;
the envelope of every series (weights, residual, source) is stored with the results.

On synthetic spectra of a 1:3 mixture of `c-1` and `c`, the mixture recovered the weights within
0.01 and the fitted deuteration within 0.25 D, while the theoretical envelope of `c-1` alone gave
about one deuterium too much (the extra hydrogen). On the 12-residue peptide's back-exchange time
course (one population, first replicate), the deuteration of c fragments agreed with pyHXExpress's
fits within 0.07 D for 10 of 11 fragments (the eleventh, with two thirds c+1, which pyHXExpress's
two-type mixture could not represent, by -0.37 D), and z fragments were 0.05 to 0.23 D higher;
that difference is not explained yet.

## The ExMS ladder (option)

pyHXExpress, following ExMS (Kan et al., J. Am. Soc. Mass Spectrom. 2011), spaces the bins by a
fixed ladder: 1.003355 Da for the first two steps, 1.004816 Da for the next two and 1.006277 Da for
every later step (divided by z). It is available as `ladder="exms"` and gives the first pass
above. It is close for deuterated small peptides but too wide for undeuterated spectra after the
second step, and much too wide for large ions: on an intact protein (charge 20) the predictions
drifted by about -0.4 ppm per bin, out of the picking window after about 60 bins and by more than
half a bin after about 120.

## The intensity of a bin

Each bin takes the tallest point within `resolution_ppm` of its predicted m/z (default +/- 50 ppm),
but never further than half the isotope spacing, `1.00235 / (2 z)`. The reported m/z of the bin is
the predicted one; where the peak was found is reported separately (see "Where the peaks were
found").

**The half-spacing cap.** A constant ppm window is a mass tolerance with no relation to the peak
spacing: at m/z 1000 and z = 20 a 50 ppm half-width is 50 mDa, the whole isotope spacing, so each
window holds both neighbouring peaks and the shoulder rule (below) zeroes every bin whose neighbour
is taller. On a top-down example at charge 20 that removed a third of the real envelope and moved
its centre by about 9 bins (HISTORY item 99). The cap binds only where m/z times charge exceeds
about 10,000 at 50 ppm, which no peptide dataset reaches, so bottom-up results are unchanged;
pyHXExpress has no cap (a deliberate difference, `docs/PLAN.md`).

**A window from the resolving power (optional).** With `resolving_power` set (R quoted at
`resolving_power_mz`, 200 by default, falling with m/z as `(m0 / m) ** resolving_power_exponent`,
0.5 for an Orbitrap) the half-width is
`window_fwhm` peak widths `m / R(m)`, never below `mass_accuracy_ppm` (the calibration error; 10
ppm by default) and never above half the isotope spacing `1.00235 / (2 z)`, so neighbouring
windows do not overlap. For R = 60,000 at m/z 200 that is 29 ppm at m/z 600, 46 ppm at m/z 1500,
and 25 mDa instead of the 37 mDa peak width at m/z 1000 with z = 20. `resolving_power =
"recorded"` takes each spectrum's value from the scan files: Thermo raw files record it per scan
(`Orbitrap Resolution` or `FT Resolution` in the trailer) with the instrument model, which says
whether it is quoted at m/z 200 or 400 (`hxex check` prints both); mzML files record none, and a
spectrum without a value keeps the `resolution_ppm` window (logged). The spectrum figures draw
the window actually used.

**Shoulder rule (from pyHXExpress).** If the window holds `shoulder_min_points` (10) or more points
and its tallest point is the first or the last of the window, the bin is set to 0, on the reasoning
that the maximum is then the flank of a neighbouring peak. The rule assumes the predicted position
is on the peak: when it is not, it removes real peaks. With the ExMS ladder on an intact protein it
zeroed 339 bins above 5 % of the maximum in 68 spectra (genuine apexes 15 to 20 ppm from their
predictions); with the composition ladder 17. Its worst failure was at high charge, where an
uncapped ppm window held both neighbouring peaks and the rule zeroed every flank bin of the
envelope (HISTORY item 99); the half-spacing cap above removes that case, since a window can then
hold at most one peak's apex. What remains open is a predicted position off the peak (see
"Limitations"). Windows with fewer points (stick or coarse profile data) are not affected.

## Where the peaks were found

The position of each picked peak is recorded (`peak_mz`): the m/z of the tallest point, refined to
the vertex of the parabola through it and its two neighbours when both lie in the same window and
are evenly spaced (profile data; centroided data keep the point itself). Each spectrum gets an
offset, `mz_offset_ppm`: the intensity-weighted median of `(peak - predicted) / predicted` over the
bins at or above the threshold. The picker uses neither; they are a diagnostic.

An offset shared by the spectra of one file is the instrument's calibration. An offset shared by
every spectrum of one series, and not by the others, means the peaks are not where the composition
puts them: a wrong sequence, terminal group, modification or ion type, or another species. Such a
series still picks "successfully" (with a 50 ppm window a peak one composition step away is caught
one bin off), and an ion-type mixture fitted to the undeuterated control can absorb the error, so
the fit alone does not reveal it. The quality screen `mz_shift` flags a series whose median offset
is more than `max_mz_offset_ppm` (15) from the median over all series. Measured (private data):

| Data | Offsets |
|---|---|
| Fragments, right composition (ETD, Orbitrap; well-fitted c and z ions of two peptides) | -3 to +2 ppm |
| The same z ions with a wrong C-terminus (`-NH2` on an acid; 0.984 Da instead of one isotope step) | -23 to -41 ppm |
| Intact peptides (SpecExport, 3,000 spectra) | median +5.5 ppm (calibration); 90 % of spectra from -5 to +14.5 ppm; 2 % of series beyond 15 ppm from the run median, mostly consistent within the series |

## Noise threshold and extent

The threshold is `noise` (an absolute intensity) or, if that is not set, `noise_percent` percent of
the spectrum's maximum. Picking covers every predicted bin and ends `zero_filling` bins after the
last bin that reaches the threshold (or at the last predicted bin). A valley inside a multimodal
envelope therefore never ends picking early (pyHXExpress stopped after a running count of
below-threshold bins). Values below the threshold are kept as they are.

## Output

Per bin: the predicted m/z, the intensity, n and where the peak was found. Per spectrum: the m/z
offset, the envelope width (charge times the
m/z width at `env_threshold` of the tallest bin), the envelope symmetry, the amide count, the
threshold used, and three flags: `empty` (every picked intensity is 0), `no_signal` (no bin reaches
the threshold) and `capped` (the bin limit cut the zero filling short). Nothing is dropped by the
picker; a run drops `empty` and `no_signal` spectra and records them. The picker settings are stored
with the results (the bundle manifest).

## Summing scans (mzML)

Scans can be summed before picking. Profile scans of one acquisition share their point spacing but
not their grid (offsets of up to half a step between scans), so each scan is interpolated onto one
grid with that spacing (the points of the scan with the most points, plus another scan's points
where it has no data), without bridging the stretches the instrument left out; centroided scans are
merged within `sum_ppm`. Sums of 10 MS1 scans agreed with the same sums made by OpenMS
(SpectraMerger) to 0.3 to 0.5 % in area, a profile correlation of 0.9999, picked bins with a median
ratio of 1.00 and an envelope centre within 0.03 to 0.11 bins.

## Validation of the composition ladder

The apex of every strong isotope peak was located in the raw spectrum (the local maximum nearest
each bin, refined with a parabola) and compared with its predicted position.

| Data | Window | ExMS: median offset (5-95 %), share inside the window | Composition: median offset (5-95 %), share inside |
|---|---|---|---|
| Peptides, SpecExport (deuterated), 11,547 peaks | 50 ppm | -1.2 ppm (-39 to +22), 93.7 % | -3.5 ppm (-34 to +20), 93.8 % |
| Peptides, SpecExport (undeuterated), 2,109 peaks | 50 ppm | +1.8 (-18 to +19), 96.3 % | +1.9 (-18 to +19), 96.4 % |
| Intact peptides in ETD scans (mzML), z = 2 and 3 | 40 ppm | +0.3 to +2.2 (spread 2 to 4 ppm), 100 % | -0.3 to -1.0 (spread 1 to 2 ppm), 100 % |
| Fragment ions (mzML), 2,928 peaks | 40 ppm | -29.6 (-51 to +3), 50.0 % | -20.7 (-40 to 0), 94.6 % |
| Intact protein, z = 20 (summed mzML), 6,025 peaks | 20 ppm | -1.6 (-16 to +9), 100 % | -0.1 (-11 to +9), 100 % |

On the Bimodal test set (496 spectra with known populations, picked from the raw spectra and fitted
end to end) the ladder made no difference to accuracy: the correct number of populations for 71.8 %
(ExMS) and 71.2 % (composition) of the spectra, all populations recovered within 0.5 D and 0.10 in
fraction for 29.2 % and 28.6 % (3 of 496 spectra apart).

In short: on data precise enough to show it (intact ions in high-resolution scans) the composition
ladder places peaks within about 1 ppm; for fragments it brings almost all peaks into their windows;
on coarse peptide data, where the scatter (tens of ppm) comes from sampling and calibration, it
changes nothing measurable, and it does not change fit accuracy there.

## Differences from pyHXExpress

| | pyHXExpress | hxex |
|---|---|---|
| Bin positions | ExMS ladder | composition ladder (ExMS as an option) |
| End of picking | after a running count of below-threshold bins | `zero_filling` bins after the last bin above the threshold |
| Empty or signal-less spectra | dropped silently | flagged, dropped by the run and recorded |
| Window half-width | `Peak_Resolution` ppm, no cap | `resolution_ppm`, capped at half the isotope spacing; or from the resolving power (`resolving_power`, a value or `"recorded"`) |
| Maximum, shoulder rule, threshold, predicted m/z reported | as here | unchanged |

With `ladder="exms"` hxex reproduces pyHXExpress's picked bins exactly on its October 2026 runs
(every bin both pickers produce identical in m/z and intensity); the cap never binds on peptide
data (m/z times charge below about 10,000 at 50 ppm), so those runs are unaffected by it.

## Limitations and open questions

- **Per-spectrum offsets.** On coarse profile data the peaks scatter by tens of ppm around their
  predictions (calibration and sampling), and about 6 % of strong peaks lie outside a +/- 50 ppm
  window. The window then holds only a flank of the peak, and the bin is underestimated. hxex does
  not shift spectra to fit the data, because that would assume every peak near a prediction belongs
  to the ion; the offsets are reported instead (see "Where the peaks were found"), and a
  calibration from the undeuterated control (whose isotope pattern is known) is an option to
  consider.
- **The shoulder rule** is unvalidated: it was designed against overlapping species but cannot tell
  their flank from a real peak off its prediction. With the composition ladder and the capped
  window it mostly zeroes small tail bins; the high-charge failure (a window holding both
  neighbours) is gone with the cap.
- **Fragment ions.** Constant offsets of -14 to -49 ppm seen for fragment targets on the
  `c-1`, `z-dot` and `M-NH3` ladders came from the ion-type mixtures above (a single assumed type
  one hydrogen off the real ions): with the right type, fragments of an undeuterated reference sit
  within 1 to 3 ppm. Still open: the z fragments' 0.1 to 0.2 D difference from pyHXExpress, and
  whether a transferred hydrogen can be a deuteron in labelled samples (which would make the
  mixture weights change with deuteration).
- **One deuteration level.** The ladder uses one binomial level per spectrum; spectra with several
  populations are placed by their overall level.

## Settings (`PickerConfig`)

| Setting | Default | Meaning |
|---|---|---|
| `ladder` | `composition` | bin positions: `composition` or `exms` |
| `resolution_ppm` | 50 | half width (ppm) of the window around each predicted m/z, capped at half the isotope spacing |
| `resolving_power` | None | R quoted at `resolving_power_mz`, or `"recorded"`: the window from the peak width instead of `resolution_ppm` |
| `resolving_power_mz` | 200 | m/z the resolving power is quoted at |
| `resolving_power_exponent` | 0.5 | R falls as `(m0 / m) ** e` (Orbitrap 0.5, TOF 0, FT-ICR 1) |
| `window_fwhm` | 1.0 | half width as a multiple of the peak width at half maximum |
| `mass_accuracy_ppm` | 10 | floor of the resolving-power window |
| `noise_threshold` | None | absolute threshold; None or 0 uses `noise_percent` (`noise` is its former name) |
| `noise_percent` | 1.0 | threshold as a percent of the spectrum's maximum |
| `zero_filling` | 4 | bins picked after the last bin above the threshold |
| `env_threshold` | 0.1 | fraction of the tallest bin at which the envelope width is measured |
| `nterm_subtract` | 1 | N-terminal residues whose amides are not counted |
| `shoulder_min_points` | 10 | window size from which the shoulder rule applies |
| `amide_rules` | default | which fragment ion types hold the amide at their cleavage site |
