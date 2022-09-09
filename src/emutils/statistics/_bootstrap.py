from typing import Iterable, Union, Callable

import numpy as np
from tqdm import tqdm

from emutils.profiling.time import estimate_iterative_function_runtime

__all__ = [
    'bootstrap_statistics',
]


def bootstrap_statistics(
    f: Callable,
    X: Iterable,
    n: Union[None, int] = None,
    max_time: float = 30.0,
    min_n: int = 1,
    max_n: int = 100000,
    random_state: int = 0,
    show_progress: bool = False,
    **kwargs,
):
    """Bootstrap the output of a function

    Args:
        f (Callable): The function
        X (Iterable): Arguments of the function, i.e., the function will be called as f(X[:])
        n (int, optional): Number of samples (to boostrap on). Defaults to None.
        max_time (int, optional): Maximum amount of time (in seconds) to bootstrap. Defaults to 30.
        max_n (int, optional): Maximum amount of sample (to bootstrap on). Defaults to 100000.
        random_state (int, optional): Random State. Defaults to 0.
        show_progress (bool, optional): If True will show tqdm progress bar. Defaults to False.

    Returns:
        np.ndarray: Bootstrap results (output of the function for each bootstrap sample)
    """

    # Estimate runtime
    if n is None:
        runtime = estimate_iterative_function_runtime(lambda X: f(X, **kwargs), X, start=min_n)
        n = min(max_n, max(min_n, int(max_time / max(1e-16, runtime))))

    # Setup random state
    random_state = np.random.RandomState(random_state)
    lenX = len(X)

    # Tranform X to an array to allow indexing
    X = np.array(X)

    # Setup iterator
    iters = range(n)
    if show_progress:
        iters = tqdm(iters, desc='Bootstrap')

    # Bootstrap
    results = []
    for _ in iters:
        indexes = random_state.randint(0, lenX, size=lenX)
        results.append(f(X[indexes], **kwargs))

    # Return
    return np.array(results)