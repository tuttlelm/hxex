hxex tutorial: HSPB5 and two disease-associated variants
=========================================================

This folder holds the data for the hxex tutorial. The tutorial itself is in two parts:

    https://github.com/tuttlelm/hxex/blob/main/docs/guide/tutorial.md        (part 1)
    https://github.com/tuttlelm/hxex/blob/main/docs/guide/tutorial-fits.md   (part 2)

The same steps are in the two Jupyter notebooks in this folder, for those who prefer to work
in a notebook: hxex_b5_tutorial_1_basics.ipynb, then hxex_b5_tutorial_2_fits.ipynb.


What is in this folder
----------------------

SpecExport/              HDX-MS spectra exported from HDExaminer 3.0 ("SpecExport"):
  WT/, D109H/, R120G/    one folder per sample, one folder per peptide inside it, one
                         file per time point, replicate, and charge state
  WT.fasta, ...          the protein sequence of each sample
hxex_b5_tutorial_1_basics.ipynb   part 1 of the tutorial as a Jupyter notebook
hxex_b5_tutorial_2_fits.ipynb     part 2, which continues from the results of part 1
tutorial_helpers.py      small helpers the notebooks use to change the project file
README.txt               this file
LICENSE.txt              the licence of the data (CC BY 4.0)


The data
--------

The spectra are from the hydrogen/deuterium exchange experiments of:

    Woods CN, Janowska MK, Ulmer LD, Kaur Sidhu J, Stone NL, James EI, Guttman M, Bush MF,
    Klevit RE. Activation mechanism of small heat shock protein HSPB5 revealed by
    disease-associated mutants. Proc Natl Acad Sci USA 122(20), e2425061122 (2025).
    https://doi.org/10.1073/pnas.2425061122

Please cite this paper when you use the data.

In short: HSPB5 (wild type, D109H and R120G; 0.04 mg/mL) labelled at room temperature in
85 % D2O PBS, pH 7.5, for 4 s, 1 min, 30 min, and 20 h; fully deuterated controls made in
85 % D2O at 75 degrees C for 30 min; quenched with ice-cold 1.6 % formic acid; measured on a
Waters Synapt G2-Si. The folders also hold 0.1 s and 0.8 s time points (quench-flow) that
are not part of the paper; the tutorial shows how to leave them out.


Licence
-------

The data in this folder are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
You may share and adapt them for any purpose, provided you give appropriate credit (cite
the paper above).
