"""hxex: multimodal binomial-mixture analysis of HDX-MS spectra.

Early development: see docs/PLAN.md.  The subpackages are placeholders for the planned
modules and contain no functionality yet.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hxex")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
