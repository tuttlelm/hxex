# The peak-picking window

*hxex* turns each spectrum into sticks: one intensity per deuterium bin, taken at the m/z
predicted for that bin from the peptide's composition and charge. The **picking window** is how
far from the predicted m/z *hxex* looks for the peak: `resolution_ppm` in `[run.picker]`, a
tolerance in ppm.

## What a bin takes: a peak

```toml
[run.picker]
pick = "peak"   # the default; "profile_fit" fits the profile instead (see below)
```

Each bin takes the tallest **peak** whose top lies inside its window. In a profile spectrum a
peak is a point higher than its neighbours, and its top is found from that point and the two
beside it, so it can lie between the recorded points. A flat top, several equal points where the
instrument clipped a very bright peak, is one peak whose top is its middle. The first and last
points of a spectrum are not peaks: with nothing recorded beyond them, they may be the side of a
peak the export cut off. In a centroided spectrum every point is a peak.

This matters when a brighter peak sits just outside a window, for example the peak of another
peptide that comes off the column at the same time, a little away from where *hxex* expects this
peptide's peaks. Its side slopes into the window. Taking the tallest point in the window, as
pyHXExpress does, picks that side as if it were this peptide's peak;
taking peaks does not, because a side has no top of its own. On coarse profile data (a Q-TOF
records a point about every 20 ppm, so a ±50 ppm window holds only 4 or 5 points) and with
narrow windows (±10 ppm on an Orbitrap) this happened often: on an Orbitrap digest, taking peaks
removed 40 % of the undeuterated controls that seemed to hold two populations (which they
cannot), without changing how well replicates agree.

Earlier versions of *hxex* could also pick as pyHXExpress does (`pick = "window_max"`, with its
"shoulder rule" and a window from the resolving power). That choice was removed, because taking
peaks was as good or better on every dataset measured. Results picked that way still open and
show their settings; a project file that asks for it is refused, with the reason.

## A peak's height or its area

```toml
[run.picker]
measure = "auto"   # the default; "height" as pyHXExpress, or "area"
area_fwhm = 1.5    # with areas: how far either side of a peak's top, in peak widths
```

Once a bin has its peak, it needs a value: the peak's **height** or its **area**. On the Q-TOF
data tested, weak peaks are wider than tall ones (by 12 to 20 % for peaks at 5 to 30 % of the
tallest), so a weak peak's height reads it low against a tall one, and the heavy side of every
envelope comes out too small. An area counts the whole peak. With areas, the undeuterated
controls matched the isotope pattern of their composition about twice as closely, fewer fully
deuterated controls seemed to hold two populations (8.8 % instead of 14.9 % on one dataset), and
the number of populations of the known-answer set was right more often (78 % instead of 75 %).
An area also does not depend on where the instrument's points happen to fall on a peak: with two
or three points across a peak, its tallest point can miss the top and read up to a tenth low.

On an Orbitrap, weak and tall peaks are equally wide and areas gained nothing, so heights stay.

With `measure = "auto"`, *hxex* decides for each run from its undeuterated controls. It measures
the width at half height of each control's tallest peak and of its weak peaks, and takes areas
when the weak ones are at least 6 % wider (judged on the brighter half of the controls: dim
spectra give unreliable widths). It needs at least 20 undeuterated controls in profile mode; with
fewer, or with centroided spectra, it keeps heights. `hxex run` logs the choice, and `hxex info`
shows it:

```
bin values     areas (weak peaks 1.12 times as wide as the tallest); peak width 48.6 ppm at m/z 1000
```

An area is turned back into a number on the height scale: it is divided by the area that the
run's tallest peaks have per unit of height, so a peak as wide as those keeps its height and a
wider one counts for more. Tables and figures read as before; mostly the weak bins change. On
the spectra page of the figures, a picked value is drawn as an up-pointing triangle when it is
an area and as a circle when it is a height, and the legend says which ("picked values: areas
(height-equivalent)"). A triangle may sit below the top of a narrow, bright peak in the raw
profile or above a wide, weak one: that is the area at work, not a mispicked peak.

The area is taken over 1.5 peak widths either side of the peak's top (never more than 0.45 of
the isotope spacing). That reaches a little further than a height does, so a peak of another
species right beside this peptide's peak adds more to an area than to a height.

Set `measure = "height"` to reproduce pyHXExpress's sticks, or results made with *hxex* before
this change. Set `measure = "area"` to take areas whatever the width check says; *hxex* still
measures the peak width from the undeuterated controls. Like the other picker settings, the
choice acts when the spectra are picked (`hxex run`; see [Changing these
settings](#changing-these-settings) for running again from the stored spectra).

## Fitting the profile instead (opt-in)

```toml
[run.picker]
pick = "profile_fit"
```

With `pick = "profile_fit"`, *hxex* does not take each bin from its own window. It fits the
whole envelope at once: one peak per bin, all with the run's own peak shape, at the positions
the envelope's isotopes put them, and the bins take those peaks' areas. Any peak in the profile
that sits more than one peak width away from every bin gets a peak of its own in the fit (a
"stray" peak), so a neighbouring peptide's peak is given to that neighbour instead of being
counted into this peptide's bins. For each spectrum:

1. The usual pick runs first and gives a rough deuterium distribution.
2. From it and the peptide's composition, *hxex* works out where each bin's peak sits (a
   deuterated bin mixes molecules with deuterium and carbon-13, which differ slightly in mass,
   so it sits a little away from an undeuterated one) and how much wider than a single peak it
   is.
3. It tries a range of small shifts of the whole spectrum in m/z (about its injection's typical
   shift) and keeps the one that fits best.
4. The fit is done twice, the second time with positions from the first fit's values.

The peak shape and width come from the run's undeuterated controls, in three classes of peak
height, because on a Q-TOF weak peaks are wider than tall ones. With fewer than 20 undeuterated
controls the shape is a smooth bell curve of the run's width. Without any undeuterated control
in profile mode the run is refused: nothing tells *hxex* how wide the peaks are.

The values are areas on the height scale (a peak of the run's width keeps its height), so
`measure` and `area_fwhm` do not apply. `hxex check` says so, and `hxex info` shows the peak
width and shape the run measured:

```
bin values     fitted peak areas (pick = profile_fit); peak width 34.9 ppm at m/z 1000; peak shape measured from 517 undeuterated controls (1007, 641, 587 peaks by height class); width at a tenth 1.88 peak widths (a Gaussian's 1.82)
```

Fragment ions, ion types other than M, series with their own envelopes (`[run.envelope]`), and
centroided spectra keep the usual pick. Three columns in `spectra.csv` describe each fitted
spectrum:

- `fit_offset_ppm`: the shift in m/z the fit took;
- `purity`: how much of the profile close to the peptide's peaks (within one peak width of
  each) the peptide's fitted peaks account for, from 0 to 1. Other species resolved between
  the peptide's peaks do not lower it; something sitting beside a peak, under its flank, does.
  On the datasets measured, most spectra read 0.9 or more (median 0.94 to 0.97, on Q-TOF and
  Orbitrap data alike), and spectra below about 0.8 were far more often far from their
  replicates. Something sitting right on a peak (within about a third of a peak width) reads as
  part of it and does not lower purity: no picking can tell those apart.
- `stray_share`: how much of the profile around the envelope the other species' peaks account
  for. A crowded spectrum can have a large share and still a high purity.

Both are only reported, not used to judge anything. To see how the fit read a spectrum, open
its **picking check** tab in the window ([the window](gui.md)), or draw it from Python:

```python
from hxex.report.picking_check import plot_picking_check

fig = plot_picking_check(spectrum_id, results)
```

It shows the raw profile, the peptide's fitted peaks, the other species' peaks the fit set
aside (orange, dashed), and the bin values, with the purity and stray share in its title.

**When it helps.** On Q-TOF data (resolving power about 20,000 to 25,000), the fitted values
were closer to the truth than the pick on every test made: simulated spectra with known
answers, two halves of one elution measured separately, the undeuterated controls against
their isotope pattern, and the charge states of one injection against each other. Fewer
undeuterated controls seemed to hold two populations (from 7.0 to 3.4 % on one dataset, from
11.2 to 5.5 % on a noisy one), and the known-answer set's number of populations was right a
little more often. The values also hardly depend on the window: the window only seeds the fit.

**On an Orbitrap**, keep the usual pick with a narrow window (`resolution_ppm = 10`). There,
other peptides often sit within one peak width of this peptide's peaks, too close for a stray
peak, and a narrow window keeps them out better. The fit was better than a ±50 ppm window but
not better than ±10 ppm.

**It takes longer.** The fit takes about 25 to 50 ms per spectrum on one core, against about
1 ms for the pick. *hxex* spreads it over up to 8 processes: a Q-TOF run of 26,000 spectra spent
about 2 minutes picking instead of under one, an Orbitrap run of 24,000 about 3 minutes. Running again from the stored spectra with the
same picker settings reuses the stored values, so the time is spent once.

On the spectra page of the figures, fitted values are drawn as up-pointing triangles, and the
legend says "fitted values: peak areas (height-equivalent)".

## The default: a ppm tolerance

```toml
[run.picker]
resolution_ppm = 50.0   # half-width of the window, ppm of the predicted m/z
```

Each bin takes the tallest peak whose top lies within 50 ppm either side of its predicted m/z.
The 50 ppm is a mass-accuracy tolerance (the same as pyHXExpress's `Peak_Resolution`). It is simple and it is
the setting every earlier *hxex* result was made with, so it stays the default.

With `pick = "peak"` (the default) this is always the window, and it does not need to change with
m/z. The window only decides whether a peak's top is close enough to count, and how far tops sit
from their predicted m/z depends on the instrument's calibration and on how strong the peak is,
not on how wide it is. Measured after each file's calibration offset: the tops of the taller
peaks (at least 30 % of the envelope's tallest) scattered by about 2 to 3 ppm on an Orbitrap
digest and about 6 ppm on a Q-TOF dataset, the same at m/z 300 as at m/z 1500, although the
Orbitrap's peaks are half again as wide at m/z 1500. Weak peaks scatter more. On an Orbitrap a
narrower window (10 ppm on the digest) keeps the peaks of neighbouring peptides out; series that
sit further off are moved onto their peaks ([below](#a-series-whose-peaks-sit-off-its-mz)).

The window never reaches further than half the spacing of the isotope peaks, about 1 Da divided
by the charge. That matters only at high charge: at m/z 1000 a 50 ppm half-width is 50 mDa, a
tenth of the spacing for a 2+ peptide (peaks 500 mDa apart) but the whole spacing for a 20+
protein (peaks 50 mDa apart). Without the cap each window would hold both neighbouring peaks,
and *hxex* would discard the bins whose neighbour is taller. For peptides the cap never applies.

## The instrument's resolving power

Thermo `.raw` files record the resolving power of every scan (MS1 and MS2 scans often differ)
and the instrument model. `hxex check` shows them:

```
run sample_01.raw: sample WT, rep 1, scan times, 885 scans; 12 of 12 targets have scans
  instrument Orbitrap Ascend; resolving power 120000 at MS1, 30000 at MS2 (quoted at m/z 200)
```

*hxex* does not size the picking window from it. A bin takes a peak whose top lies in the
window, so the window is a mass tolerance: how far a peak's top may sit from its predicted m/z.
That distance is a few ppm whatever the peak width. Two findings on an Orbitrap digest, from
earlier versions that could size the window from the resolving power:

- a window one peak width either side accepts the top of a neighbouring peak one and a half peak
  widths away, which is a different peak. With the recorded resolving power, the undeuterated
  controls that seemed to hold two populations went from 21 % to 48 %, and the labelled
  spectra from 13 % to 27 %, against a ±10 ppm window;
- the resolving power a Thermo file records is the setting the scan was acquired with, not the
  width of the peaks. On that digest the peaks were 1.5 times narrower than the recorded value
  implies, at every m/z.

The width of the peaks that *hxex* does use (for areas, above) is measured from the run's own
undeuterated controls.

## A series whose peaks sit off its m/z

```toml
[run.picker]
off_mz_sd = 6.0   # how far from the run's calibration counts as off, in multiples of its scatter

[analysis.filters]
off_mz = true     # leave such series out of the analysis
```

Before picking, *hxex* looks at the undeuterated controls. In each one it finds where the tops
of the three lightest peaks (M, M+1, and M+2) actually are, searching well beyond the picking
window (at least 40 ppm either side). The tops have to agree (within 5 ppm, or a quarter of the
picking window's half-width if that is more) to place the envelope. When one of the three lands
on a peak of something else and the other two agree, those two place it. Such a control counts
for its own series, but not for the injection's calibration below, which uses only controls whose
three tops agree, so that the limit stays as tight as the data allow. On a calibrated instrument the peptides of one injection
sit together: their peaks are all off by about the same small amount, the instrument's
calibration, give or take a little scatter. With scan files (mzML or Thermo raw) an injection is
one undeuterated control file; with the other formats, which do not say which file a spectrum came
from, it is the undeuterated controls of one sample and replicate. On an Orbitrap the calibration is the same across
the m/z range; on a Q-TOF it drifts by about 1 ppm per 100 m/z, so *hxex* fits a straight line
in m/z for each injection.

A few series sit much further from that line, 10 to 50 ppm, by the same amount in every
control. A peptide does not move that far by itself: the envelope there belongs to another
species. On the Orbitrap digest we checked against the dataset's own identification runs, most
such series had a retention time in the peptide table that pointed at another peptide of almost
the same mass; the peptide itself came off the column 0.3 to 2 minutes away, at exactly its
predicted m/z. In the rest, another peptide comes off at the same time and its peaks merge with
the peptide's.

*hxex* calls such a series **off its m/z** when its controls sit further from the line than
`off_mz_sd` times the injection's scatter (and at least 3 ppm). An injection with fewer than ten
undeuterated controls uses the line and scatter of all of them. The check needs at least five
undeuterated controls in the project whose peaks were found: the scatter of three or four is no
measure (a line through three points passes through two of them), so with fewer no series is
checked, `hxex run` says so, and `envelope_offset_ppm` is empty. Because the limit follows the
scatter, it adapts to the instrument: on an Orbitrap digest the scatter was 0.7 ppm, on a Q-TOF
dataset 2 to 3 ppm.

What happens to a series off its m/z:

- every one of its spectra is flagged `off_mz` in `spectra.csv`, and `hxex run` says how many
  series were flagged;
- its windows stay on the predicted m/z, so its bins hold the side of the other species' peaks,
  or nothing;
- `hxex analyse` leaves all its spectra out, its controls included, and lists them in
  `left_out_spectra.csv` with the reason `off its m/z`. Its fully deuterated control then does
  not stand in for other series either.

What to do about a flagged series: check its retention time. With scan files and `find_peak`,
*hxex* looks for where the peptide's exact m/z elutes and lists what it found in
`retention_suggestions.csv` ([When the peptide table's time is wrong](inputs.md#when-the-peptide-tables-time-is-wrong)).
If your peptide list came from an identification run, look there for where the peptide was identified, and whether a modification
differs between that sample and the HDX samples (cysteines alkylated in one but not the other,
for example). Correct the time in your peptide table and run again. If you have checked a series
and want it kept, set its `review_status` to `accepted` in the review table.

Earlier versions moved the windows of such a series onto the envelope they found (`recentre`).
That setting has been removed: on the Orbitrap digest the envelope it moved onto was another
peptide in most series, and its uptake can look perfectly plausible, so nothing in the results
showed it.

`spectra.csv` shows the check in two columns: `envelope_offset_ppm` (how far the series'
undeuterated peaks sit from the injection's calibration) and `envelope_offset_limit_ppm` (the
limit it was compared with). A third, `window_shift_ppm`, is 0 except in results made with
`recentre`. Fragment ions are not checked (their own ion-type mixtures move their peaks), nor
series without an undeuterated control; the `mz_shift` flag looks after those.

### Changing these settings

| Setting | What changes |
|---|---|
| `off_mz_sd` lower (4, say) | more series are flagged: those only a little further from the calibration than the rest. On the Orbitrap digest 74 of 527 series instead of 66; on the Q-TOF dataset 22 of 1,320 instead of 4 |
| `off_mz_sd` higher (10, say) | only the series furthest away: 59 and 1 on the same datasets |
| `off_mz_sd = false` | no check: no `off_mz` flag, and the columns are empty; `mz_shift` judges every series instead |
| `[analysis.filters] off_mz = false` | the flagged series stay in the analysis, controls included |

No setting changes how any series is picked. `off_mz_sd` acts when *hxex* picks the spectra,
so it takes effect in `hxex run` only (not in `hxex report` or `hxex analyse`);
`off_mz` acts in `hxex analyse`. To apply a change of `off_mz_sd` to results you already
have, change the setting in `hxex.toml` and run again from the spectra stored in the bundle,
which does not read the raw files again, into a new folder:

```bash
hxex run hxex.toml --spectra results --bundle results_offmz4
```

The figures and analyses made from the earlier results are then out of date (see [Running
again](outputs.md#running-again)). Results made with a version of *hxex* before this check have
empty columns and no `off_mz` flag until they are run again.

## One spectrum off its injection's m/z

```toml
[run.quality]
spectrum_off_mz_sd = 6.0   # how far from its injection counts as off, in multiples of the scatter
```

The check above judges a whole series, from its undeuterated controls. A single spectrum can be
off on its own: a labelled replicate whose retention window landed on a neighbouring ion, or an
exported spectrum summed over the wrong scans. Its replicates are fine, so the series passes.

*hxex* therefore also compares every spectrum with the other spectra of its injection (one scan
file, or for the other formats one sample, time point, and replicate). Within an injection the
peptides' peaks all sit off their predicted m/z by about the same amount, the instrument's
calibration, and a spectrum far from that is suspect. How far is too far depends on how bright
the spectrum is: the peaks of a dim spectrum land less precisely, so *hxex* measures the scatter
separately for dim and bright spectra in each run and holds each spectrum to the scatter of
spectra like it. A spectrum further from its injection's median than `spectrum_off_mz_sd` times
that scatter is flagged `spectrum_off_mz`.

On the datasets we measured, the flag marked about 1 % of the spectra, and about half of them
were more than 0.05 D per amide away from their replicates, against 3 % of all spectra. It
caught every exported spectrum that had been summed over another ion's scans, and on the tutorial
data the one series that reads another species in every sample. It is a flag only: nothing is
left out. Look at a flagged spectrum and its elution trace, and leave it out with
`review_status = "rejected"` if it is not your peptide.

`spectra.csv` shows the check in `spectrum_offset_ppm` (how far the spectrum's peaks sit from
its injection's median) and `spectrum_offset_limit_ppm` (the limit for a spectrum of its
brightness). Fragment ions are not checked, and nor are spectra in an injection with fewer than
ten spectra. `spectrum_off_mz_sd = false` turns the check off. It is set in `hxex run`, which
also adds it to results made before it existed.

## Seeing the window

The spectrum figures (`hxex report --figures`, `hxex.report.plot_spectrum`) draw every picked
bin as a shaded bar spanning the window *hxex* actually used for that spectrum (moved, for a
series off its m/z), and the legend
says where the window came from: "±50 ppm window" (results picked by earlier versions with a
window from the resolving power say "window from R 60000 at m/z 200" or "window from the
recorded R 120000").

## Which to choose

- Intact peptides at low charge on a calibrated instrument: the default is fine.
- High charge states (top-down, large fragments, intact proteins): the ppm window is capped at
  half the isotope spacing, so the windows never overlap.
- An Orbitrap: a narrower ppm window, such as 10 ppm, keeps neighbouring peptides out.
- Do not narrow the window to sharpen the picking on deuterated data. A deuterated peak mixes
  molecules carrying deuterium and carbon-13, which differ slightly in mass, so it does not sit
  exactly where the undeuterated peaks do. On a Q-TOF dataset the undeuterated peaks sat within
  16 ppm of their predicted m/z, but the deuterated ones within 20 to 40 ppm, and a 16 ppm window
  lost real deuterated peaks.

Results picked by earlier versions with `pick = "window_max"` do not give the sticks of today's
picking on the same spectrum, and heights and areas differ too, so compare runs made with the same settings. The results bundle records the picker settings, and `hxex info` shows them.

