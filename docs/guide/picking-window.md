# The peak-picking window

*hxex* turns each spectrum into sticks: one intensity per deuterium bin, taken at the m/z
predicted for that bin from the peptide's composition and charge. The **picking window** is how
far from the predicted m/z *hxex* looks for the peak. Two settings in `[run.picker]` choose it.

## The default: a ppm tolerance

```toml
[run.picker]
resolution_ppm = 50.0   # half-width of the window, ppm of the predicted m/z
```

Each bin takes the tallest point within 50 ppm either side of its predicted m/z. This is a
mass-accuracy tolerance (the same as pyHXExpress's `Peak_Resolution`). It is simple and it is
the setting every earlier *hxex* result was made with, so it stays the default.

The window never reaches further than half the spacing of the isotope peaks, about 1 Da divided
by the charge. That matters only at high charge: at m/z 1000 a 50 ppm half-width is 50 mDa, a
tenth of the spacing for a 2+ peptide (peaks 500 mDa apart) but the whole spacing for a 20+
protein (peaks 50 mDa apart). Without the cap each window would hold both neighbouring peaks,
and *hxex* would discard the bins whose neighbour is taller. For peptides the cap never applies.

## A window from the instrument's resolving power

```toml
[run.picker]
resolving_power = 60000.0        # R, quoted at resolving_power_mz
resolving_power_mz = 200.0       # Thermo quotes R at m/z 200 (400 on the Elite/Velos generation)
resolving_power_exponent = 0.5   # how R falls with m/z: 0.5 Orbitrap, 0 TOF, 1 FT-ICR
window_fwhm = 1.0                # half-width in peak widths (full width at half maximum)
mass_accuracy_ppm = 10.0         # the half-width never falls below this
```

With `resolving_power` set, the window follows the peak width. *hxex* takes the resolving power
at each predicted m/z (`R(m) = R0 (m0 / m) ** exponent`, so an Orbitrap's R falls with the
square root of m/z), computes the peak's width at half maximum (`m / R(m)`), and uses
`window_fwhm` of those widths as the half-width. Two guards apply:

- the half-width never falls below `mass_accuracy_ppm`, so a very high resolving power does not
  shrink the window below the instrument's calibration error (series on a well-calibrated
  Orbitrap sit within a few ppm of their predicted m/z; a wrong sequence or modification moves
  them by tens of ppm, which the `mz_shift` quality flag reports);
- the half-width never exceeds half the isotope spacing (`1.00235 / (2 z)` Da), so the windows
  of neighbouring bins never overlap.

For R = 60,000 at m/z 200 that gives a half-width of 29 ppm at m/z 600 and 46 ppm at m/z 1500,
and at m/z 1000 with charge 20 the cap gives 25 mDa where the peak width would be 37 mDa.
`resolution_ppm` is not used while `resolving_power` is set (it remains the fallback, below).

## Taking the resolving power from the raw files

```toml
[run.picker]
resolving_power = "recorded"
```

Thermo `.raw` files record the resolving power of every scan (MS1 and MS2 scans often differ)
and the instrument model. With `"recorded"`, *hxex* uses each spectrum's own value, and the
model tells it whether the value is quoted at m/z 200 or 400. `hxex check` shows what a file
records:

```
run sample_01.raw: sample WT, rep 1, scan times, 885 scans; 12 of 12 targets have scans
  instrument Orbitrap Ascend; resolving power 120000 at MS1, 30000 at MS2 (quoted at m/z
  200); [run.picker] resolving_power = "recorded" would use it
```

mzML files (including those converted from `.raw` with ThermoRawFileParser) and the folder
formats (SpecExport, HX-Express) carry no resolving power. Spectra without a recorded value keep
the `resolution_ppm` window; `hxex run` logs how many. For those formats, set `resolving_power`
to a number instead.

## Seeing the window

The spectrum figures (`hxex report --figures`, `hxex.report.plot_spectrum`) draw every picked
bin as a shaded bar spanning the window *hxex* actually used for that spectrum, and the legend
says where the window came from: "±50 ppm window", "window from R 60000 at m/z 200" or
"window from the recorded R 120000".

## Which to choose

- Intact peptides at low charge on a calibrated instrument: the default is fine.
- High charge states (top-down, large fragments, intact proteins): set `resolving_power`, or
  `"recorded"` for Thermo raw files, so the windows stop overlapping.
- Centroided data or coarse profile data: either works; the shoulder rule (windows of ten or
  more points whose maximum sits at the edge are set to zero) does not apply to windows with
  fewer points.

The two modes do not give identical sticks on the same spectrum, so compare runs made with the
same setting. The results bundle records the picker settings, and `hxex info` shows them.

