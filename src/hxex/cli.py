"""Command-line entry point (`hxex`).

Only `--version` exists so far; the real commands arrive with the pipeline stages.
"""

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hxex",
        description="Multimodal binomial-mixture analysis of HDX-MS spectra (early development).",
    )
    parser.add_argument("--version", action="version", version=f"hxex {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
