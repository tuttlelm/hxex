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
abundance (`m0_max`).

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
  with more than one population, which can mean an incompletely deuterated or a mixed sample.
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

**The undeuterated value**

1. The series' own undeuterated control (its replicates averaged), if it is accepted.
2. Otherwise 0, natural abundance (flagged `m0_missing` or `m0_high`).

**The fully deuterated value**

1. The series' own fully deuterated control: the same peptide, at the same charge, in the same
   sample (its replicates averaged), if it is accepted.
2. The same peptide's accepted fully deuterated controls at its other charge states, in the
   same sample: their mean retention (`fd_other_charge`). Retention belongs to the peptide,
   not to its charge state. In the B5 tutorial data a peptide's charge states agree to within
   0.01, and in a test where all charge states were measured in the same scans their
   deuterium differed by at most about 1.5 %. To turn this step off, set
   `fd_other_charge = false` in `[run.controls]`.
3. A value for the whole run, chosen by `fd_fallback`:
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

Steps 2 and 3 are stand-ins, so the series is marked `control_fallback`. A stand-in from
step 2 is the same peptide's own measurement, so the protection-factor fit treats it as a
measured control (see [Protection factors](protection-factors.md)). The same settings in
`[analysis.controls]` recompute the values for an analysis of an existing bundle.
