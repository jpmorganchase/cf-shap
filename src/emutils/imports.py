"""
    Author: Emanuele Albini

    This module contains a list of popular imports for a ML / Data Science project
    Useful to rapidly start a Notebook without writing all imports manually.

    It will also set up the main logging.Logger

    ## Usage

    ```python
        from emutils.imports import *
    ```

    ## Imports
    - Python Standard Modules: os, sys, time, platform, gc, math, random, collections, itertools, re, urllib, importlib, warnings, logging
    - Several common function/class from Pyton Standard Library
    - Several common function from emutils
    - Utilities: tqdm
    - Machine Learning: xgboost, sklearn
    - Data Science: numpy, pandas, networkx
    - Plotting: matplotlib, seaborn, plotly, graphviz

"""

# Python 3 stuff
import os
import sys
import time
import platform
import gc
import math
import random
import pickle
import collections
import itertools
import functools
import re
import logging
import urllib
import importlib
import warnings
import json

from itertools import product, combinations, permutations, combinations_with_replacement, chain
from math import floor, ceil, isnan, isinf, log, log10, exp, sqrt
from collections import defaultdict, Counter
from enum import Enum
from functools import partial
from typing import List, Dict, Union, Iterable, Any, Tuple, Callable
from argparse import ArgumentParser

from emutils.utils import (
    prod,
    attrdict,
)
from emutils.ipython import (
    display,
    in_ipynb,
    import_tqdm,
)

print('Python ', platform.python_version())
print('Python Executable:', sys.executable)

print("CWD = ", os.getcwd())


# Logging
def __setup_logging(logging_format='[%(asctime)s] %(levelname)s:\t %(message)s', time_format="%Y-%m-%d %H:%M:%S"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(logging_format, time_format))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


__setup_logging()

# Utilities
try:
    # Import iPy or command-line TQDM
    tqdm = import_tqdm()
    # Common typo
    tqmd = tqdm
except (ModuleNotFoundError, ImportError):
    pass

# DataScience Basics
try:
    import numpy
    np = numpy
    print('NumPy', np.__version__, end="")

    try:
        import pandas
        pd = pandas
        print(' | Pandas', pd.__version__, end="")
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        import scipy
        sp = scipy
        import scipy.stats as stats
        print(' | SciPy', sp.__version__, end="")
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        import networkx
        nx = networkx
        print(' | NetworkX', nx.__version__, end="")
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        import statsmodels
        import statsmodels as sm
        print(' | StatsModels', statsmodels.__version__, end="")
    except (ModuleNotFoundError, ImportError):
        pass

except (ModuleNotFoundError, ImportError):
    pass
# DataScience Basics End

print()

# Machine Learning
try:
    import sklearn
    print('scikit-learn', sklearn.__version__, end=" | ")
except (ModuleNotFoundError, ImportError):
    pass

try:
    import xgboost
    xgb = xgboost
    print('xgboost', xgb.__version__, end=" | ")
except (ModuleNotFoundError, ImportError):
    pass

print("\b\b")

# Visualization
# try:
import matplotlib

mpl = matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.ticker as mtick

print('MatPlotLib', mpl.__version__, end=" | ")
try:
    import seaborn
    sns = seaborn
    print('Seaborn', sns.__version__, end=" | ")
except (ModuleNotFoundError, ImportError):
    pass

try:
    import graphviz
    print('GraphViz', graphviz.__version__, end=" | ")
except (ModuleNotFoundError, ImportError):
    pass

try:
    import plotly.offline as pyo
    if in_ipynb():
        pyo.init_notebook_mode(connected=True)
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # print('Plotly', plotly.__version__, end=f" (mode = {'notebook' if in_ipynb() else 'script'})")
    # copy_plotlyjs('.')

except (ModuleNotFoundError, ImportError):
    pass

print("\n")
