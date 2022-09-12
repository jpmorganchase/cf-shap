"""
    Author: Emanuele Albini

    Threshold tuning strategies.
"""

import numpy as np

__all__ = [
    'find_best_curve_threshold_by_derivative',
    'find_best_threshold_by_rate_equivalence',
    'find_best_curve_threshold_by_sum',
]


def find_best_curve_threshold_by_derivative(x, y, thr, m=1, precision=1e-10):

    assert x.shape == y.shape == thr.shape
    assert len(x.shape) == 1

    nb_points = x.shape[0]

    # The basic line with the right inclination
    base_line = np.linspace(0, m, nb_points)
    start_q = min(0, 1 - m)

    # Find the line just above the curve
    vl = 1.0 + precision
    vr = start_q - precision
    while vl - vr > precision:
        vm = (vl + vr) / 2
        vline = vm + base_line
        if np.all(vline > y):
            vl = vm
        else:
            vr = vm

    # Calculate the line just above the curve
    vline = vl + base_line

    # Find best threshold
    best_index = np.argmin(vline - y)
    best_x = x[best_index]
    best_y = y[best_index]
    best_thr = thr[best_index]

    return best_x, best_y, best_thr


def find_best_threshold_by_rate_equivalence(model, X, y):
    y = np.asarray(y).flatten()

    ratio0 = 1 - y.sum() / y.shape[0]

    y_pred_proba = model.predict_proba(X)
    thresh_ = np.sort(y_pred_proba[:, 1])[int(np.round(ratio0 * y.shape[0]))]

    return thresh_


def find_best_curve_threshold_by_sum(x, y, thr, m=1):

    assert x.shape == y.shape == thr.shape
    assert len(x.shape) == 1

    # Find best threshold
    best_index = np.argmax(m * y - x)
    best_x = x[best_index]
    best_y = y[best_index]
    best_thr = thr[best_index]

    return best_x, best_y, best_thr
