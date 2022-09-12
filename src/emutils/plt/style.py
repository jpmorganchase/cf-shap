""""
    Author: Emanuele Albini

    Plot styling utilities.
"""

import matplotlib.pyplot as plt
# import matplotlib as mpl

from emutils.latex import set_latex

__all__ = [
    'nice_style',
]


def activate_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}\usepackage{amssymb}",
        "font.sans-serif": ["Helvetica", "Avant Garde", "Computer Modern Sans Serif"],
        "font.serif": ["Bookman", "Computer Modern Roman", "Times"],
        "font.family": "serif",
        "legend.labelspacing": .1,
    })
    set_latex(True)


def deactivate_latex():
    plt.rcParams.update({
        "text.usetex": False,
    })
    set_latex(False)


def nice_style(latex=False, size=10):

    plt.style.use('bmh')

    # Set the default text font size
    plt.rc('font', size=size)
    # Set the axes title font size
    plt.rc('axes', titlesize='large')
    # Set the axes labels font size
    plt.rc('axes', labelsize='medium')
    # Set the font size for x tick labels
    plt.rc('xtick', labelsize='medium')
    # Set the font size for y tick labels
    plt.rc('ytick', labelsize='medium')
    # Set the legend font size
    plt.rc('legend', fontsize='medium')
    # Set the font size of the figure title
    plt.rc('figure', titlesize='large')

    if latex:
        activate_latex()
    else:
        deactivate_latex()