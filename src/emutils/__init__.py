"""
    EmUtils: Emanuele's Utilities

    This package contains a bunch of utilities for several use cases and libraries.
"""

import pkg_resources
from pathlib import Path
import os

# Monkey Patch broken packages

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0"

PACKAGE_DATA_FOLDER = os.path.abspath(os.path.join(Path(__file__).parent, 'data'))