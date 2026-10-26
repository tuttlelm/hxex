# The peak-picking window

*hxex* turns each spectrum into sticks: one intensity per deuterium bin, taken at the m/z
predicted for that bin from the peptide's composition and charge. The **picking window** is how
far from the predicted m/z *hxex* looks for the peak. Two settings in `[run.picker]` choose it.

## What a bin takes: a peak

```toml
[run.picker]
pick = "peak"   # the default; "window_max" picks as pyHXExpress does
```

Each bin takes the tallest **peak** whose top lies inside its window. In a profile spectrum a
peak is a point higher than its neighbours, and its top is found from that point and the two
beside it, so it can lie between the recorded points. In a centroided spectrum every point is a
peak.

This matters when a brighter peak sits just outside a window, for example the peak of another
peptide that comes off the column at the same time, a little away from where *hxex* expects this
peptide's peaks. Its side slopes into the window. Taking the tallest point in the window, as
pyHXExpress does (`pick = "window_max"`), picks that side as if it were this peptide's peak;
taking peaks does not, because a side has no top of its own. On coarse profile data (a Q-TOF
records a point about every 20 ppm, so a ±50 ppm window holds only 4 or 5 points) and with
narrow windows (±10 ppm on an Orbitrap) this happened often: on an Orbitrap digest, taking peaks
removed 40 % of the undeuterated controls that seemed to hold two populations (which they
cannot), without changing how well replicates agree.

Set `pick = "window_max"` only to reproduce pyHXExpress's sticks, or results made with *hxex*
before this change. With it, a window of ten or more points whose tallest point sits at its edge
gives 0 (the "shoulder rule", `shoulder_min_points`); windows with fewer points keep their
tallest point.

## The default: a ppm tolerance

```toml
[run.picker]
resolution_ppm = 50.0   # half-width of the window, ppm of the predicted m/z
```

Each bin takes the tallest peak whose top lies within 50 ppm either side of its predicted m/z.
The 50 ppm is a mass-accuracy tolerance (the same as pyHXExpress's `Peak_Resolution`). It is simple and it is
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
  Orbitrap sit within a few ppm of their predicted m/z; a series further away is moved and
  flagged, [below](#a-series-whose-peaks-sit-off-its-mz));
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

## A series whose peaks sit off its m/z

```toml
[run.picker]
off_mz_sd = 6.0   # how far from the run's calibration counts as off, in multiples of its scatter
recentre = true   # move the windows of such a series onto its peaks
```

Before picking, *hxex* looks at the undeuterated controls. In each one it finds where the tops
of the three lightest peaks (M, M+1, and M+2) actually are, searching well beyond the picking
window (at least 40 ppm either side). On a calibrated instrument the peptides of one injection
sit together: their peaks are all off by about the same small amount, the instrument's
calibration, give or take a little scatter. On an Orbitrap the calibration is the same across
the m/z range; on a Q-TOF it drifts by about 1 ppm per 100 m/z, so *hxex* fits a straight line
in m/z for each injection.

A few series sit much further from that line, 10 to 40 ppm, by the same amount in every
control. A peptide does not move that far by itself. Most likely another peptide of almost the
same mass comes off the column at the same time, and the instrument cannot tell the two apart:
what it records is one merged peak whose top lies between them. A narrow window centred on the
predicted m/z then catches only the side of that peak, and a side catches a different share of
each peak, which distorts the envelope.

*hxex* calls such a series **off its m/z** when its controls sit further from the line than
`off_mz_sd` times the injection's scatter (and at least 3 ppm). An injection with fewer than ten
undeuterated controls uses the line and scatter of all of them. Because the limit follows the
scatter, it adapts to the instrument: on an Orbitrap digest the scatter was 0.7 ppm, on a Q-TOF
dataset 2 to 3 ppm.

What happens to a series off its m/z:

- every one of its spectra is flagged `off_mz` in `spectra.csv`, and `hxex run` says how many
  series were flagged. Nothing is removed;
- with `recentre = true` (the default), all its windows, in the controls and in the labelled
  spectra, move by the series' measured offset, so each bin reads the top of the merged peak
  instead of its side. On the Orbitrap digest (a ±10 ppm window) this brought the flagged
  peptides' uptake much closer to the expected values: 82 % of their time points within 0.1 of
  the prediction, against 73 % before. With a wide window (±50 ppm on a Q-TOF) the tops are
  already inside it, and moving it changes nothing.

Moving the window does not take the other peptide out of the spectrum, which is why the flag
stays. Look at a flagged series' spectra. If its uptake disagrees with the same peptide at
another charge, or with its replicates, consider [leaving it out](excluding-spectra.md).

`spectra.csv` shows the check in three columns: `envelope_offset_ppm` (how far the series'
undeuterated peaks sit from the injection's calibration), `envelope_offset_limit_ppm` (the limit
it was compared with), and `window_shift_ppm` (how far its windows moved; 0 when they did not).
Fragment ions are not checked (their own ion-type mixtures move their peaks), nor series without
an undeuterated control; the `mz_shift` flag looks after those.

### Changing these settings

| Setting | What changes |
|---|---|
| `off_mz_sd` lower (4, say) | more series are flagged and moved: those only a little further from the calibration than the rest. On the Orbitrap digest 74 of 527 series instead of 66; on the Q-TOF dataset 22 of 1,320 instead of 4 |
| `off_mz_sd` higher (10, say) | only the series furthest away: 59 and 1 on the same datasets |
| `off_mz_sd = false` | no check: no `off_mz` flag, no window moved, and the three columns are empty; `mz_shift` judges every series instead |
| `recentre = false` | the series are still checked and flagged, but every window stays on the predicted m/z: the picked peaks are the same as with `off_mz_sd = false` |

Only the flagged series change; every other series is picked exactly as before. The settings
act when *hxex* picks the spectra, so they take effect in `hxex run` only (not in `hxex
report`, `hxex resume`, or `hxex analyse`). To apply a change to results you already have,
change the setting in `hxex.toml` and run again from the spectra stored in the bundle, which
does not read the raw files again, into a new folder:

```bash
hxex run hxex.toml --spectra results --bundle results_offmz4
```

The figures and analyses made from the earlier results are then out of date (see [Running
again](outputs.md#running-again)). Results made with a version of *hxex* before this check have
empty columns and no `off_mz` flag until they are run again.

## Seeing the window

The spectrum figures (`hxex report --figures`, `hxex.report.plot_spectrum`) draw every picked
bin as a shaded bar spanning the window *hxex* actually used for that spectrum (moved, for a
series off its m/z), and the legend
says where the window came from: "±50 ppm window", "window from R 60000 at m/z 200" or
"window from the recorded R 120000".

## Which to choose

- Intact peptides at low charge on a calibrated instrument: the default is fine.
- High charge states (top-down, large fragments, intact proteins): set `resolving_power`, or
  `"recorded"` for Thermo raw files, so the windows stop overlapping.
- Centroided data or coarse profile data: either works.
- Do not narrow the window to sharpen the picking on deuterated data. A deuterated peak mixes
  molecules carrying deuterium and carbon-13, which differ slightly in mass, so it does not sit
  exactly where the undeuterated peaks do. On a Q-TOF dataset the undeuterated peaks sat within
  16 ppm of their predicted m/z, but the deuterated ones within 20 to 40 ppm, and a 16 ppm window
  lost real deuterated peaks.

The two modes do not give identical sticks on the same spectrum, so compare runs made with the
same setting. The results bundle records the picker settings, and `hxex info` shows them.

