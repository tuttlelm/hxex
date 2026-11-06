# hxex

Standalone Python package for multimodal binomial-mixture analysis of HDX-MS spectra: spectra
to binned deuterium sticks to fitted populations with uncertainties and plots. Successor to
pyHXExpress (it does not depend on it). Early development: Phases 0 to 3 are done (scaffold,
`chem`, readers and peak picker, then the engine: `model`, `noise`, `fit`, `bootstrap`, the
results bundle and CPU/GPU parity, then controls, quality flags and the `run` stage). Phase 4
(`report`: plots, tables and figure files) is in progress.

The plan, decisions, phases, status and the deliberate differences from pyHXExpress are in
`docs/PLAN.md`. Read it before designing anything, and do not copy it here. The dated findings
and measurements behind them are in `docs/HISTORY.md`; read the relevant item when a task needs
the evidence. Release steps are in `docs/RELEASING.md`.

## Setup and commands

Python 3.11 or newer (CI runs 3.11 to 3.14 on Linux, Windows and macOS).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest            # tests
.venv/bin/ruff check .      # lint; must pass
```

`.venv` is the CPU-only PyTorch build and matches CI. A GPU build lives in a separate,
git-ignored `.venv-gpu` (see the README for the CUDA wheel index).

## Conventions (already decided; the reasons are in docs/PLAN.md)

- Every `.py` file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- Sequences use modX labels (`acK`, `pS`, `pT`, `pY`). Tokenise into residues; never count or
  slice a sequence by characters. Modifications go in a private registry passed to pyteomics
  through `aa_comp`; never edit pyteomics' global tables.
- Chemistry takes an ion type (default `M`).
- No global state. Behaviour is configured by explicit, serialisable config objects.
- No pandas DataFrames in the core: NumPy arrays plus a typed index in memory, Parquet
  (pyarrow) on disk, one row per spectrum for raw spectra.
- Long operations take progress and cancellation hooks. Use `logging`, never `print`. Plot
  functions return figures and never call `show()`.
- Controls are identified by user-set times (undeuterated and fully deuterated), with replicates
  averaged and flagged fallbacks for missing or bad controls.
- Match the behaviour of pyHXExpress's `hxex_updating.py` unless `docs/PLAN.md` lists the
  difference as deliberate. Document any new difference there.

## Working rules

- Run `pytest` and `ruff check .` before saying work is done, and report failures with their
  output.
- Do not add a dependency without discussing it. Its license must be compatible with GPL v3, and
  the lower bounds in `pyproject.toml` are tested by the minimum-versions CI job (keep the
  `torch==2.5.0` pin in `.github/workflows/ci.yml` equal to the torch lower bound).
- Commit only when asked. The repository is configured to commit with the GitHub no-reply
  address; do not change git configuration or use a private email. Do not push; the owner pushes.
- Do not publish to PyPI or change repository settings.

## Concurrent sessions

Several sessions may work on this repository at once.

- A session that edits code works in its own git worktree on its own branch
  (`git worktree add ../hxex-<topic> -b <topic>`); the owner merges. Sessions that only read,
  analyse data or write scratch outside the repository may stay on `main`.
- Before editing, run `git status` and `git log -3`. If another session has uncommitted changes
  in the tree, or commits appear on the branch during the session, stop and tell the owner.
- Stage with `git add <file>` only after checking (`git diff <file>`) that the file holds only
  this session's changes; never `git add -A`, `git stash`, `git checkout` or `git reset` in a
  shared tree.
- New `docs/HISTORY.md` items are written as `pending.` and numbered only at commit time: the
  next free number after the last item in `git log` and the file, with cross-references updated.
- `ListAgents` shows peer sessions; tell an active one which files you are about to touch.

## Layout

- `src/hxex/`: one placeholder module per planned component (`chem`, `io`, `picking`, `model`,
  `fit`, `report`, `noise`, `controls`, `bootstrap`, `pipeline` (the run stage; `hxex.run` is its entry point), `cli`). Each docstring says what
  it will contain and in which phase. `gui` is the Qt window (optional `gui` extra; `hxex gui`;
  docs/PLAN.md "GUI"; its `session` and `figures` modules import without Qt; tests run
  offscreen). `fragments` makes fragment-ion entries and mzML
  targets from a parent peptide. `kint` gives intrinsic exchange rates in H2O/D2O mixtures
  (ported from pacilab/hdx-rates-mixtures, MIT). `deamidation` measures Asn-Gly deamidation of
  heated fully deuterated controls from the peaks' m/z. `config` reads the TOML project file that `cli` (`hxex init`,
  `set`, `check`, `run`, `resume`, `report`, `figures`, `info`) runs; `project_edit` changes one setting
  of it in place (`hxex set`). `report.catalogue` lists every kind of figure (the guide's `figures.md`, `hxex figures`). Shared contracts: `spectra.py` (spectrum and entry
  types) and `hooks.py` (progress and cancellation). `picking` produces `BinnedSet`.
- `tests/`, `docs/` (`PLAN.md`, `HISTORY.md`, `ANALYSIS.md` (post-fitting analysis design:
  read it before working on that), `BUNDLE.md`, `PEAK_PICKING.md`, `RELEASING.md`,
  `AUDIT_2026-10-17.md`, `AUDIT_2026-10-28.md` and `AUDIT_2026-11-01.md` (audit findings with the owner's decisions and the execution order;
  read it before fixing anything it lists), `guide/` (the user guide: add or update a page
  for every user-facing feature; its `settings.md` and `commands.md` are generated by
  `python -m hxex._reference docs/guide`, and a test fails when a setting or a command-line
  argument has no description or a page is out of date);
  `images/` holds the logo, regenerated by `images/make_logo.py`; `tutorial/` holds the
  two tutorial notebooks (one per page: `guide/tutorial.md` and `guide/tutorial-fits.md`),
  their `tutorial_helpers.py`, the download's README and licence, and `make_tutorial.py`, which
  builds the download and redraws `images/tutorial_*.png`; `tests/test_tutorial.py` keeps each
  page and its notebook in step),
  `.github/workflows/` (`ci.yml`, `publish.yml`).

Machine-specific notes (local paths to the pyHXExpress reference code and test data, tooling
quirks) are in `CLAUDE.local.md`, which is not committed.
