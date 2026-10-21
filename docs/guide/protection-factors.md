# Protection factors

Protection factors (`[analysis] protection = true`) compare how fast each peptide exchanges
with how fast the same amides would exchange unprotected (their intrinsic rates). They work
for forward labelling (the protein goes into D2O and gains deuterium) and for back-exchange
kinetics: the protein is deuterated first, then put into a solvent with less deuterium, and
loses it over time.

You do not name the direction. It follows from the labelling conditions: for back-exchange,
add `initial_fraction`, the D fraction the amides started with, above the `d2o_fraction` of
the solvent the exchange happens in.

```toml
[analysis]
protection = true

[analysis.labelling]       # the solvent the back-exchange happens in
ph_read = 7.0
temperature = 20.0
d2o_fraction = 0.1         # 90 % D2O diluted ten-fold into H2O
initial_fraction = 0.9     # the amides began in 90 % D2O
```

Leave `initial_fraction` out for forward labelling. If the data go the other way from the
conditions (deuterium falls with time but no `initial_fraction` is set, or rises when it
is), *hxex* stops and says which setting to change.

**Where the exchange starts.** By default *hxex* fits each peptide's starting deuteration
(`fD0` in `protection.csv`) along with its protection factor, because the protein may not
have been fully deuterated when the exchange began. If the run has a measured fully
deuterated control (a run with `role = "fulldeut"`), each peptide is taken to start there
(`fD0 = 1`). Early points of a kinetics run are often unreliable: leave them out with
`[analysis] exclude_times`. *hxex* never drops them on its own.

**Without a fully deuterated control.** Back-exchange samples without one are fitted on the
absolute scale (deuterons per amide), relaxing to the solvent's equilibrium deuteration;
`protection.csv` says `corrected = false`. (Forward-labelling samples without controls get no
protection factors.)

**Reading the results.** `direction` says `back`, `equilibrium` is the deuteration each
peptide relaxes to, and `fD0` its fitted start with `fD0_se`. A peptide already at
equilibrium at the first time point gets only an upper limit (`bound = upper`); one that did
not exchange gets a lower limit. A fitted start is trustworthy when the loss of deuterium is
visible within your time points. When most of it happened before the first point, the start
is poorly known and its error is too small. Peptides with fewer than three time points get
no value when the start is fitted.

**Per residue.** Protection factors from peptides are per peptide only. Overlapping peptides
do not determine single residues reliably, so *hxex* does not offer that. Residue-level rates
and protection factors come from fragment ladders ([Residue-level rates and protection factors from fragment
ladders](fragment-residues.md)).

