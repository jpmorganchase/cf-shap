"""
    Author: Emanuele Albini

    Execution time profiling utilities.
"""

import time
import functools
from typing import Optional
import numpy as np

__all__ = [
    'estimate_iterative_function_runtime',
    'estimate_parallel_function_linear_plateau',
    'timer',
]

# Function profiler Python 3.6
# import cProfile
# import pstats
# cProfile.run('3 + 4', 'restats')
# pstats.Stats('restats').strip_dirs().sort_stats('cumtime').print_stats()


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        return value

    return wrapper_timer


def estimate_iterative_function_runtime(f, X, n=None, start=1, precision=1e-1, concatenate=False):
    """Estimate the runtime of an iterative function (a function that execute on operation on an array iteratively)
        NOTE: The assumption is that, at the limit the execution time of the function over each sample is similar.


    Args:
        f (lambda): The function
        X (Iterable): Arguments of the function
        n (int, optional): Number of samples for which you want to know the runtime. Defaults to None.
            - n=None: All X, it will return the runtime of the function over X.
            - n=1: It will return the runtime of the function over a single sample.
        start (int, optional): Number of samples from which to start the bisection algorithm. Defaults to 1.
        precision (float, optional): Precision (seconds). The execution will take approximately 2 * precision seconds. Defaults to 1e-1.
        concatenate (bool, optional): Allow to concatenate the arguments if the required precision has not been reached. Defaults to False.

    Returns:
        float: Runtime of the function over n arguments.
    """
    # Prevent in-place changes
    if concatenate:
        X = X.copy()

    # Setup
    n = n or len(X)
    m = start
    rtime = -np.inf
    while True:
        # Slice
        X_ = X[:m]

        # Run
        start = time.perf_counter()
        f(X_)
        rtime = time.perf_counter() - start

        # Break conditions
        if rtime > precision:
            break
        if not concatenate and m > len(X):
            break

        # Increment size
        m = m * 2

        # Concatenate dataset if necessary
        if concatenate and m > len(X):
            X = np.concatenate([X, X], axis=0)

    rtime = rtime * n / m

    return rtime


def estimate_parallel_function_linear_plateau(
    f: callable,
    X: np.ndarray,
    r: int = 7,
    n: Optional[int] = None,
    b: float = 0.1,
    precision: float = 1e-1,
    start: int = 1,
    ratio: float = .5,
    window_size: int = 3,
    ret_details: bool = False,
    verbose: bool = False,
) -> int:
    """Estimate the number of samples after which the runtime of a parallel function (a function that execute on operations on an array in parallel) reaches a linear plateau.
        >> linear plateau: the runtime of the function over n samples is approximately proportional to n. 
            Typically this happens when the function is given a large number of samples and the runtime dominates over the overhead of the parallelization.

    Args:
        f (callable): A callable function
        X (np.ndarray): An array of samples 
        r (int, optional): Number of times to repeat an estimation. Defaults to 7.
        n (Optional[int], optional): Number of samples over which to call a function to estimate the runtime. Defaults to None (auto).
        b (float, optional): Fraction of increment in size over iterations. Defaults to 0.1 (10% increment per iteration).
        precision (float, optional): The minimum runtime to run a function for (in seconds). Defaults to 1e-1.
        start (int, optional): The number of samples to run the function on to start with. Defaults to 1.
        ratio (float, optional): The threshold in the ratio of the increase in runtime over the increase in the number of samples the function is called on. Defaults to .5 (50%).
        window_size (int, optional): The window size. Defaults to 3.
        ret_details (bool, optional): Return a more verbose output. Defaults to False.
        verbose (bool, optional): Verbose. Defaults to False.

    Returns:
        int: _description_
    """
    assert b > 0, 'The increment b must be positive'

    # Prevent in-place changes
    X = X.copy()

    # Setup
    m = start
    rtimes = [np.inf] * window_size
    ms = [np.nan] * window_size
    rratios = []

    while True:
        # Take a slice of size m
        X_ = X[:m]

        rtime = []  # Runtimes
        ns = []  # Number of samples over which the function has been called

        # Repeat for r times
        for _ in range(r):
            if n is None:
                n_ = 0
                start = time.perf_counter()

                # Run the function until the required precision is reached
                while time.perf_counter() - start < precision:
                    f(X_)
                    n_ += 1

                # Record runtime and number of samples over which the function has been called
                rtime.append((time.perf_counter() - start) / n_)
                ns.append(n_ * m)
            else:

                # Precision is not ignored
                n_ = int(np.ceil(n / m))

                # Run the function over (at least) n samples
                start = time.perf_counter()
                for _ in range(n_):
                    f(X_)

                # Record runtime and number of samples over which the function has been called
                rtime.append((time.perf_counter() - start) / n_)
                ns.append(n)

        ns = np.mean(ns)
        rtime = np.min(rtime)

        # Compute the fraction of increase in the current runtime compared to the previous ones (if any) in a sliding window
        rratio_time = np.array([(rtime - rtimes[-j - 1]) / rtime for j in range(window_size)])

        # Compute the increase in the number of samples over which the function has been called compared to the previous ones (if any) in a sliding window
        rratio_m = np.array([(m - ms[-j - 1]) / m for j in range(window_size)])

        # Compute the ratio between the increase in runtime and the increase in number of samples
        rratio = np.mean(rratio_time / rratio_m)

        # Store them
        rtimes.append(rtime)
        rratios.append(rratio)
        ms.append(m)

        if verbose:
            print('m=', m, 'T=', rtime, "N=", ns, 'Ratio=', rratio, "(", np.round(rratio_time, 3), "/",
                  np.round(rratio_m, 3), ")")

        # Break conditions
        # We stop when the ratio between the increase in runtime and the increase in number of samples is larger than `ratio`
        if rratio > ratio:
            break

        # Increment size
        m = int(np.ceil(m * (1 + b)))

        # Concatenate dataset if necessary
        if m > len(X):
            X = np.concatenate([X] * int(np.ceil(1 + b)), axis=0)

    if ret_details:
        return m, np.array(ms[window_size:]), np.array(rtimes[window_size:]), np.array(rratios)
    else:
        return m
