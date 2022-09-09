"""
    Author: Emanuele Albini

    Random sampling utilities.
"""

from typing import Union
import numpy as np
import pandas as pd

__all__ = [
    'sample_data',
    'np_sample',
]


def sample_data(
    X: Union[pd.DataFrame, np.ndarray],
    n=None,
    frac=None,
    random_state=None,
    replace=False,
    **kwargs,
) -> Union[pd.DataFrame, np.ndarray]:
    assert frac is None or n is None, "Cannot specify both `n` and `frac`"
    assert not (frac is None and n is None), "One of `n` or `frac` must be passed."

    if isinstance(X, pd.DataFrame):
        return X.sample(
            n=n,
            frac=frac,
            random_state=random_state,
            replace=replace,
            **kwargs,
        )
    elif isinstance(X, np.ndarray):
        if frac is not None:
            n = int(np.ceil(len(X) * frac))
        return np_sample(X, n=n, replace=replace, random_state=random_state, **kwargs)
    else:
        raise NotImplementedError('Unsupported dataset type.')


def np_sample(
    a: Union[np.ndarray, int],
    n: Union[int, None],
    replace: bool = False,
    seed: Union[None, int] = None,
    random_state: Union[None, int] = None,
    safe: bool = False,
) -> np.ndarray:
    """Randomly sample on axis 0 of a NumPy array

    Args:
        a (Union[np.ndarray, int]): The array to be samples, or the integer (max) for an `range`
        n (int or None): Number of samples to be draw. If None, it sample all the samples.
        replace (bool, optional): Repeat samples or not. Defaults to False.
        seed (Union[None, int], optional): Random seed for NumPy. Defaults to None.
        random_state (Union[None, int], optional): Alias for seed. Defaults to None.
        safe (bool, optional) : Safely handle `n` or not. If True and replace = False, and n > len(a), then n = len(a)

    Returns:
        np.ndarray: A random sample
    """
    assert random_state is None or seed is None

    if random_state is not None:
        seed = random_state

    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random

    # Range case
    if isinstance(a, int):
        if safe and n > a:
            n = a
        return random_state.choice(a, n, replace=replace)
    # Array sampling case
    else:
        if n is None:
            n = len(a)
        if safe and n > len(a):
            n = len(a)
        return a[random_state.choice(a.shape[0], n, replace=replace)]