"""
    Author: Emanuele Albini

    Utilities to write LaTex code in Python strings.
"""

import functools

# Global config
LATEX = False


# Change config
def set_latex(latex):
    global LATEX
    LATEX = latex


def latex_config_decorator(func):
    # This make the resulting avoid the lost of 'identity' in the resulting function
    @functools.wraps(func)
    def decorated_latex_func(*args, latex=None, **kwargs):
        latex = LATEX if latex is None else latex
        return func(*args, latex=latex, **kwargs)

    return decorated_latex_func


@latex_config_decorator
def bold(s, latex):
    if latex:
        return '\\textbf{' + s + '}'
    else:
        return s


@latex_config_decorator
def exp(s, e, latex):
    if latex:
        return '{' + f"{s}" + '}^{' + f"{e}" + '}'


@latex_config_decorator
def under(s, u, latex):
    if latex:
        return '{' + f"{s}" + '}_{' + f"{u}" + '}'


@latex_config_decorator
def math_bold(s, latex):
    if latex:
        return '\\bm{' + s + '}'
    else:
        return s


@latex_config_decorator
def percentage_sign(latex):
    if latex:
        return r'\%'
    else:
        return '%'


@latex_config_decorator
def serif(s, latex):
    if latex:
        return '\\emph{' + s + '}'
    else:
        return s


@latex_config_decorator
def equation(s, latex):
    if latex:
        return '$' + s + '$'
    else:
        return s
