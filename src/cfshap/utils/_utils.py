"""
    Author: Emanuele Albini

    This module implements some general utilities for explanations
"""

import numpy as np

from ..base import Model, XGBWrapping, ListOf2DArrays

__all__ = [
    'get_shap_compatible_model',
    'get_top_counterfactuals',
    'expand_dims_counterfactuals',
    'get_top_counterfactual_costs',
]


def get_top_counterfactuals(
    XX_C: ListOf2DArrays,
    X: np.ndarray,
    n_top: int = 1,
    nan: bool = True,
    return_2d=True,
):
    """Extracts the top-x counterfactuals from the input

    Args:
        XX_C (ListOf2DArrays): List of arrays (for each sample) of counterfactuals (multiple)
        X (np.ndarray): The query instances/samples
        n_top (int, optional): Number of counterfactuals to extract. Defaults to 1.
        nan (bool, optional): Return NaN. Defaults to True.
        return_2d (bool, optional): Return a 2D array if n_top = 1. Defaults to True.

    Returns:
        ListOf2DArrays: The top-counterfactuals
            If n_top = 1 and return_2d = True:
                nb_samples x nb_features
                Note: if nan = False, nb_samples may be less that X.shape[0]
            Else:
                nb_samples (List) x nb_counterfactuals x nb_features
                Note: if nan = False, nb_counterfactuals may be less than n_top

    """
    assert X.shape[0] == len(XX_C)

    if n_top == 1 and return_2d is True:
        XC = np.full(X.shape, np.nan)
        for i, XC_ in enumerate(XX_C):
            if XC_.shape[0] > 0:
                XC[i] = XC_[0]
        if nan is False:
            XC = XC[np.any(np.isnan(XC), axis=1)]
        return XC
    else:
        if nan is True:
            XX_C_top = [np.full((n_top, X.shape[1]), np.nan)]
            for i, XC_ in enumerate(XX_C):
                if XC_.shape[0] > 0:
                    n_xc = min(n_top, XC_.shape[0])
                    XX_C_top[i][:n_xc] = XC_[:n_xc]
            return XX_C_top
        else:
            XX_C = [XC[~np.any(np.isnan(XC), axis=1)] for XC in XX_C]
            return [XC[:n_top] for XC in XX_C]


def expand_dims_counterfactuals(X):
    """Expand dimensions of a list of counterfactuals

    Args:
        X (np.ndarray): The counterfactuals

    Returns:
        List[np.ndarray]: A list of arrays of counterfactuals. All arrays will have length 1.

    """

    if isinstance(X, np.ndarray):
        return [x.reshape(-1, 1) for x in X]
    else:
        return [np.array([x]) for x in X]


def get_top_counterfactual_costs(costs: ListOf2DArrays, X: np.ndarray, nan: bool = True):
    """Extract the costs of the top-x counterfactuals

        Pre-condition: Counterfactuals costs for each sample must be ordered (ascending).

    Args:
        costs (ListOf2DArrays): List of arrays (for each sample) of counterfactuals costs (multiple).
        X (np.ndarray): The query instances/samples
        nan (bool, optional): If True, return NaN cost if no counterfactual exists, skips the sample otherwise. Defaults to True.

    Returns:
        np.ndarray: Array of the costs of the best counterfactuals
    """
    if nan:
        return np.array([c[0] if len(c) > 0 else np.nan for c in costs])
    else:
        raise NotImplementedError()


def get_shap_compatible_model(model: Model):
    """Get the SHAP-compatible model

    Args:
        model (Model): A model

    Raises:
        ValueError: If the model is not supported

    Returns:
        object: SHAP-compatible model.
    """
    # Extract XGBoost-wrapped models
    if isinstance(model, XGBWrapping):
        model = model.get_booster()

        # Explicit error handling
        if not (type(model).__module__ == 'xgboost.core' and type(model).__name__ == 'Booster'):
            raise ValueError("'get_booster' must return an XGBoost Booster object.")

    return model
