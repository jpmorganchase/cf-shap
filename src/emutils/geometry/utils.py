"""
Author: Emanuele Albini

This module contains basic and utility functions for geometric transformations.

"""

from typing import Union
import numpy as np
from sklearn.base import BaseEstimator

__all__ = [
    'rotate_2d',
    'scaled_linspace',
]


def rotate_2d(
    X: Union[np.ndarray, list],
    degrees: int = 90,
    origin: Union[np.ndarray, list] = [0, 0],
) -> np.ndarray:
    """Rotate a 2D point around a given origin

    Args:
        X (np.ndarray): Array of points to rotate
        degrees (int, optional): Degrees to rotate the points. Defaults to 90.
        origin (list, optional): Origin coordinates. Defaults to [0, 0].

    Returns:
        _type_: _description_
    """
    origin = np.asarray(origin)
    X = np.asarray(X)

    # Convert to complex
    X = X[:, 0] + 1j * X[:, 1]
    origin = origin[0] + 1j * origin[1]

    # Convert to radiants
    angle = np.deg2rad(degrees)

    # User Euler formula
    X = (X - origin) * np.exp(complex(0, angle)) + origin

    # Convert back to reals
    X = X.view("(2,)float")

    return X


def scaled_linspace(x: np.ndarray, y: np.ndarray, num: int, scaler: BaseEstimator) -> np.ndarray:
    """Generate a linspace, evenly spaced according to the scaling of the data (enacted by the scaler) 

        Args:
            x (np.ndarray): First point
            y (np.ndarray): Sencond point
            num (int): Number of points (in between the two points)
            method (str): Normalization method

        Returns:
            np.ndarray: Sequence of points evenly spaced
        """
    # Normalize the points
    x = scaler.transform([x])[0]
    y = scaler.transform([y])[0]

    # Generate the linspace
    ls = np.linspace(x, y, num=num + 1, endpoint=True)

    # Unnormalize the points
    ls = scaler.inverse_transform(ls)

    return ls
