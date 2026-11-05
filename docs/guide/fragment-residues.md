# Residue-level rates and protection factors from fragment ladders

With fragment ions (ETD, ECD, and other low-scrambling activations), consecutive fragments of
a ladder differ by one residue, so the difference of their deuterium follows that residue
alone. With `[analysis.fragments] fit_rates = true`, `hxex analyse` fits an exponential decay
(or rise) to every such single-residue step and writes the rate to `ladder_rates.csv`. With
`kint = true` as well and the kinetics' solvent in `[analysis.fragments.exchange]`, each step
also gets the residue's intrinsic rate and its protection factor with a standard error
(`log10_protection`, `log10_protection_se`).

**The rate fits are opt-in.** Fitting every step takes time: several minutes on a large run.
Without `fit_rates = true`, `hxex analyse` still writes `fragment_uptake.csv`, `closure.csv`,
the ladder figures, and the closure figures, but the ladder figures show the measured points
and differences without fitted curves, and there is no `ladder_rates.csv`,
`ladder_residues.csv`, or residue-rates figure. Turn the fits on when you want the rates:

```toml
[analysis.fragments]
fit_rates = true
```

Everything below about rates, precise steps, and fitted curves applies only with
`fit_rates = true`.

**The ladder figure.** `ladder_<parent>_<ion>_z<charge>` shows the precursor in its top
row, then a row for each step of the ladder, read top to bottom. A ladder too long for
`[analysis] page_size` (16 by 12 inches: the precursor and three steps) goes on over further
pages, `..._z1_2`, `..._z1_3`, each with the precursor row again at its top and its page
number in that row's title. A taller page (`page_size = [16, 30]`) holds more steps a page.

**Which steps count.** A step is `precise` when its decay is visible in the time range and
its rate's relative error is at most `max_rate_error` (0.5 by default):

```toml
[analysis.fragments]
fit_rates = true
max_rate_error = 0.3      # stricter: rate_se / rate at most 30 %
```

**One value per residue.** `ladder_residues.csv` combines a residue's precise estimates from
the N-terminal ladder (c ions, say) and the C-terminal ladder (z ions), weighted by their
errors. It never combines different activation conditions (gentle and harsh), precursor
charges, fragment charges or runs, and never two ion types of the same ladder (c and c-1
come from the same peaks, so they are not independent): with both c and c-1 you get one row
for c with z and one for c-1 with z. `agreement` says whether the two ladders agree
(`agree`), disagree beyond their errors (`disagree`: look at both before using the mean), or
whether only one ladder measures the residue (`single`).

**Error ranges in the figures.** Every point of the ladder and closure figures carries a
light bar of plus and minus one standard error, when the points have one (they need a
bootstrap or replicates; a kinetics run fitted without a bootstrap has none). Every fitted
curve of the ladder figure has a shaded band of plus and minus one standard error of the
fit, whatever the points carry; with many points the band is narrow. In the residue-rates
figure, the rate, the amplitude and the protection factor of each step have error bars.
