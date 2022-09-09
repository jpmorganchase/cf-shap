"""
    Author: Emanuele Albini
    
    Basic evaluation utilities for feature attributions 
"""

from typing import Union
import numpy as np
import pandas as pd

__all__ = [
    'feature_attributions_statistics',
]


def feature_attributions_statistics(phi: np.ndarray, mean=False) -> Union[pd.DataFrame, pd.Series]:
    """Calculate some statistics on feature attributions

    Args:
        phi (np.ndarray: nb_samples x nb_features): The feature attributions
        mean (bool, optional): Calculate the mean. Defaults to True.

    Returns:
        The statistics
            pd.DataFrame: By default
            pd.Series: If mean is True.
    """
    stats = pd.DataFrame()
    stats['Non-computable Phi (NaN)'] = np.any(np.isnan(phi), axis=1)
    stats['Non-Positive Phi'] = np.all(phi < 0, axis=1)
    stats['Non-Negative Phi'] = np.all(phi > 0, axis=1)
    stats['All Zeros Phi'] = np.all(phi == 0, axis=1)
    stats['Number of Feature with Positive Phi'] = np.sum(phi > 0, axis=1)
    stats['Number of Feature with Negative Phi'] = np.sum(phi < 0, axis=1)
    stats['Number of Feature with Zero Phi'] = np.sum(phi == 0, axis=1)

    if mean:
        # Rename columns accordingly
        def __rename(name):
            if name.startswith('Number'):
                return 'Average ' + name
            else:
                return 'Percentage of ' + name

        stats = stats.rename(columns={name: __rename(name) for name in stats.columns.values})

        # Mean over all the attributions in the dataset
        stats = stats.mean()
    return stats