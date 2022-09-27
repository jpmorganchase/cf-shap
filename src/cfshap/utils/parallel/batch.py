"""
    Author: Emanuele Albini

    Parallelization Utilities for batch processing with JobLib.
"""

import logging
from typing import Union

from emutils.utils import import_tqdm
from emutils.parallel.utils import Element, split, join, max_cpu_count

__all__ = ['batch_process']


def _parallel_joblib(f, splits, n_jobs):
    # pylint: disable=import-outside-toplevel
    from joblib import Parallel, delayed
    # pylint: enable=import-outside-toplevel
    return Parallel(n_jobs=n_jobs)(delayed(f)(s) for s in splits)


def batch_process(
    f,
    iterable: Element,
    split_size: int = None,
    batch_size: int = None,
    desc='Batches',
    verbose=1,
    n_jobs: Union[None, int, str] = None,
) -> Element:
    '''
        Batch-processing for iterable that support list comprehension.
        
        f : function
        iterable : An iterable to be batch processed (list of numpy)
        split_size/batch_size : size of the batch
    '''

    assert split_size is None or batch_size is None
    if split_size is None:
        split_size = batch_size

    if n_jobs is None:
        n_jobs = 1
    if n_jobs == 'auto':
        n_jobs = n_jobs or (max_cpu_count() - 1)

    if verbose > 0:
        tqdm = import_tqdm()

    # Split
    if verbose > 1:
        logging.info('Splitting the iterable for batch process...')
    splits = split(iterable, split_size)
    if verbose > 1:
        logging.info('Splitting done.')

    # Attach TQDM
    if desc is not None and desc is not False and verbose > 0:
        splits = tqdm(splits, desc=str(desc))

    # Batch process
    if n_jobs > 1 and len(iterable) > split_size:
        logging.info(f'Using joblib with {n_jobs} jobs')
        results = _parallel_joblib(f, splits, n_jobs)
    else:
        results = [f(s) for s in splits]

    # Join
    if verbose > 1:
        logging.info('Joining the iterable after batch process...')
    results = join(results)
    if verbose > 1:
        logging.info('Joining done.')

    # Return
    return results