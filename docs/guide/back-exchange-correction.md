# Back-exchange correction

Some of the deuterium a protein takes up is lost again before it is measured: during the
quench, the digestion, and the LC. *hxex* corrects for this loss with each series' two
controls. A value at the level of the undeuterated control becomes 0, a value at the level of
the fully deuterated control becomes the number of exchangeable amides, and everything in
between is scaled in proportion.

The correction runs only when the project file has a `[run.controls]` table; without one, the
values stay in deuterons, as measured. Fragment ions are never corrected, with or without the
table (see [Experiment types](experiments.md)). Where the controls come from, and how to use a
time point as one, is in [Controls and times](controls.md).

## How *hxex* judges the controls

The back-exchange correction scales each series between its two controls, so a bad control
distorts every value of its series. *hxex* checks every control before using it. A control
that fails is not used: the series takes a stand-in value instead (see
[The order of the values](#the-order-of-the-values) below) and is marked in the tables
(`control_fallback`, with the reason in the `controls` column of `series.csv`).

**The undeuterated control** is rejected when its mean is above 0.5 deuterons beyond natural
abundance (`m0_max`; flagged `m0_high`). An undeuterated sample holds no deuterium, so a mean
that high means the peaks *hxex* picked are not the peptide's alone: most often another species
of almost the same m/z, a few mass units heavier or at another charge, sits in the picking
windows, and it is usually in the labelled spectra too.

**The fully deuterated control** is judged by its *retention*: the fraction of the label it
still holds when it is measured, (mFD − m0) / number of amides. Retention is always below the
D₂O fraction of the labelling buffer, because some deuterium exchanges back during the quench,
digestion, and LC. How much depends on the workflow, and on the peptide (its sequence and how
long it stays on the column), so there is no single right value: in the B5 tutorial data it runs
from 0.34 to 0.64 with an 85 % D₂O buffer. A fully deuterated control is rejected when:

- its retention is 0 or less, or above 1.1 (`retention_max`): it cannot be right;
- it is below the highest time point of its own series (`fd_below_data`): the sample did not
  fully deuterate. A little below is allowed (`time_course_tolerance`, 0.1 D per amide), and
  each time point counts by the median of its replicates, so one bad replicate does not
  decide;
- its retention is far below the run's other controls (`retention_outlier`): more than four
  robust standard deviations below their median (`outlier_sd`). This adapts to your workflow:
  a run with much back-exchange has a low median, and its controls are judged against that.
- with the quench conditions given (`quench` in `[run.controls]`), its retention misses the
  retention predicted from the peptide's sequence by more than 0.15 per amide
  (`fd_off_model`, `kint_tolerance`). This check replaces the previous one for every peptide
  the prediction covers.

Two more checks only warn, and the control is still used. They are listed in the
`control_notes` column of `series.csv`:

- `fd_disagrees`: the same peptide's fully deuterated controls in the other samples have a
  clearly different retention (by more than 0.1, `sample_tolerance`). In one run they should
  agree closely, since the peptide and the workflow are the same.
- `fd_multimodal` or `m0_multimodal`: most of a control's replicate spectra were fitted best
  with more than one population: the envelope is wider than one population fits, for example
  because the molecules kept different amounts of deuterium. The correction uses the
  control's average, its retention, which can still be right. For a fully deuterated control,
  `series.csv` says whether that retention was checked against the same peptide's controls in
  the other samples: `fd_multimodal (retention agrees)` means it matches theirs and the
  correction is sound; `fd_multimodal (retention differs)` (with `fd_disagrees`) means it does not,
  which can mean an incompletely deuterated sample. With no other sample to compare (a run with
  one sample, say), it is compared with the same peptide's controls at its other charge states:
  `fd_multimodal (retention agrees with its other charge states)`. `fd_multimodal (retention
  unchecked)` means there was nothing to compare it with, or its other charge states disagree,
  so look at its spectra.
- `data_below_m0`: a time point of the series reads below its undeuterated control. The
  fastest time points are the hardest to measure, so it is the time point that is suspect,
  not the control: look at that time point's spectra.

The overview figure (`hxex report hxex.toml --figures`, file `overview`) shows the retention of
every measured control with the limits used, and counts the reasons. If you prefer a fixed
lower limit, set it: `retention_min = 0.3` in `[run.controls]` replaces the comparison with the
run.

## The order of the values

The correction needs two values for every series (one peptide at one charge in one sample):
its undeuterated value and its fully deuterated value. For each, *hxex* takes the first of
these that it has. Fragment ions are left out: they are not corrected, and their controls
take no part in the run's median or prediction.

Before a control's replicates are averaged, a replicate is left out when its deuterium is
above the peptide's amide count, or when it lies far from the series' other replicates of
that control (`replicate_outlier`, 6 typical replicate standard deviations, with three or more
replicates). `control_notes` then says `m0_replicate_left_out` or `fd_replicate_left_out`.

**The undeuterated value**

1. The series' own undeuterated control (its replicates averaged), if it is accepted.
2. Otherwise 0, natural abundance (flagged `m0_missing` or `m0_high`).

**The fully deuterated value**

1. The peptide's accepted fully deuterated controls in the same sample, at all its charge
   states: their mean retention. Retention belongs to the peptide, not to its charge state.
   In the B5 tutorial data a peptide's charge states agree to within 0.01, in a larger
   dataset 90 % agree to within 0.05, and in a test where all charge states were measured in
   the same scans their deuterium differed by at most about 1.5 %. A charge state without a
   good control of its own takes this value too (`fd_other_charge`) and counts as measured.
   Charge states whose retentions differ by more than `sample_tolerance` (0.1) are still
   combined, with the warning `fd_charges_disagree` in `control_notes`. The combined value is
   checked against each charge state's own time course, as a measured control is: a charge
   state whose time points it would put below keeps its own control if that passed its
   checks; one without a good control of its own is rejected (`fd_below_data`) and takes the
   run's value (step 2) instead.

   To correct each charge state with its own control instead, set `combine_charges = false`
   in `[run.controls]`. A charge state without one then takes the mean of the peptide's other
   charge states (`fd_other_charge`, a stand-in, checked against its time course in the same
   way), unless `fd_other_charge = false` too.
2. A value for the whole run, chosen by `fd_fallback`:
   - Without quench conditions (the default, `"median"`): the median retention of the run's
     accepted controls (`fd_median`). When no control is accepted: the D₂O fraction of the
     labelling buffer (`d2o_fraction`), which means no back-exchange correction.
   - With quench conditions (`quench` in `[run.controls]`; the default then is `"kint"`): the
     retention predicted from the peptide's sequence, fitted to the run's accepted controls
     (`fd_kint`). The prediction needs the intrinsic exchange rate of every amide, so a
     peptide with extra exchangeable sites (`Hex` in its modification) takes the median
     retention instead (`fd_median`). So does every series when fewer than 10 controls are
     accepted (`kint_min_series`): with fewer, one bad control can mislead the prediction.
     On the B5 data, from 10 controls on, the prediction beat the median in more than nine
     cases out of ten, even with one bad control among them.
   - `fd_fallback = "d2o"`: the D₂O fraction.

Step 2 is a stand-in, so the series is marked `control_fallback`. A value from another
charge state of the same peptide (step 1) is the peptide's own measurement, so it is not a
stand-in, and the protection-factor fit treats it as measured (see
[Protection factors](protection-factors.md)).

`hxex analyse` leaves the series with a stand-in out of its tables, figures, and comparisons
(`stand_in_controls = "exclude"` in `[analysis.filters]`, the default): on a large dataset
they were the series whose corrected values most often went above full deuteration. A sample
with no measured fully deuterated control at all keeps them, since every one of its series
uses one. Set `stand_in_controls = "include"` to keep them everywhere. The run's own tables
and uptake plots always show every series.

The same setting leaves out a series whose undeuterated control was rejected as too heavy
(`m0_high`), with the reason `stand-in control (undeuterated)` in `left_out_spectra.csv`:
replacing its m0 by 0 fixes the starting point, but the other species is still in its labelled
spectra. On a public Orbitrap digest, these were most of the peptides whose uptake differed from
another program's by several deuterons. Here there is no exception for a sample: if half of a
sample's series or more are left out this way, `hxex analyse` warns, since that usually means a
labelled file was marked as undeuterated. A missing undeuterated control (`m0_missing`) is not
a reason: 0 is then a fair stand-in. A series just over the limit can be a good one (a long
peptide at 0.6 deuterons, say): look at its undeuterated spectra, and to keep it, mark its
spectra `accepted` in the [user review table](project-file.md#the-user-review-table).

A control that shows another species in its spectrum is left out of the correction in
`hxex analyse` (`spectrum_quality_filtering`; see
[Spectrum quality screens](outputs.md#spectrum-quality-screens)). If that leaves a series
without a measured fully deuterated control, it takes a stand-in as above, and is then left
out with the reason `stand-in control (screened)`.

The same control settings in `[analysis.controls]` recompute the values for an analysis of an
existing bundle.
