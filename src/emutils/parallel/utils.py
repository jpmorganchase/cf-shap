"""
    Author: Emanuele Albini

    General utilities needed when parallelizing (e.g., determining number of CPUs, split an array)
"""

from typing import Union, List, Tuple
import itertools
import multiprocessing

import pandas as pd
import numpy as np

__all__ = [
    'max_cpu_count',
]

Element = Union[list, np.ndarray, pd.DataFrame]
Container = Union[List[Element], np.ndarray]


def max_cpu_count(reserved=1):
    count = multiprocessing.cpu_count()
    return max(1, count - reserved)


def nb_splits(iterable: Element, split_size: int) -> int:
    return int(np.ceil(len(iterable) / split_size))


def split_indexes(iterable: Element, split_size: int) -> List[Tuple[int, int]]:
    nb = nb_splits(iterable, split_size)
    split_size = int(np.ceil(len(iterable) / nb))  # Equalize sizes of the splits
    return [(split_id * split_size, (split_id + 1) * split_size) for split_id in range(0, nb)]


def split(iterable: Element, split_size) -> Container:
    """
        iterable : The objet that should be split in pieces
        split_size: Maximum split size (the size is equalized among all splits)
    """
    if isinstance(iterable, (np.ndarray, pd.DataFrame)):
        return np.array_split(iterable, nb_splits(iterable, split_size))
    else:
        return [iterable[a:b] for a, b in split_indexes(iterable, split_size)]


def join(iterable: Container, axis: int = 0) -> Element:
    # Get rid of iterators
    iterable = list(iterable)

    assert len(set((type(a) for a in iterable))) == 1, "Expecting a non-empty iterable of objects of the same type."

    if isinstance(iterable[0], pd.DataFrame):
        return pd.concat(iterable, axis=axis)
    elif isinstance(iterable[0], np.ndarray):
        return np.concatenate(iterable, axis=axis)
    elif isinstance(iterable[0], list):
        return list(itertools.chain.from_iterable(iterable))
    else:
        raise TypeError('Can handle only pandas, numpy and lists. No iterators.')
