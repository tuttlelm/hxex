# hxex

Multimodal binomial-mixture analysis of HDX-MS spectra: from spectra (raw or exported) to
binned deuterium sticks to fitted populations with uncertainties and plots.

**Status: early development (Phase 0, repository scaffold).** Nothing is usable yet. The
project plan is in [`docs/PLAN.md`](docs/PLAN.md).

`hxex` is the planned successor to
[pyHXExpress](https://github.com/tuttlelm/pyHXExpress). It is standalone and does not depend
on it.

## Requirements

- Python 3.11 or newer
- Runs on CPU or GPU (PyTorch)

## Development install

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only PyTorch (smaller than the default Linux wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

For an NVIDIA GPU, install a CUDA build of PyTorch that matches your driver instead of the CPU
line. Tested: the CUDA 12.6 build on an RTX A4500 with driver 535 (CUDA 12 minor-version
compatibility):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

These wheels are several gigabytes; on a slow connection add `--default-timeout=600` to `pip`.

## License

GNU General Public License v3 (see [`LICENSE`](LICENSE)), the same license as pyHXExpress.
