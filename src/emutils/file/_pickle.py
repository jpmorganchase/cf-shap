import os
import time
import pickle
import numpy as np
from typing import Optional, Union
import logging
import warnings
from enum import Enum

from ..human import duration_to_long_string

__all__ = [
    'ComputeRequest',
    'save_pickle',
    'load_pickle',
    'compute_or_load',
    'run_or_load',
]


class ComputeRequest(Enum):
    RUN_OR_RESUME_NOSAVE = RESUME_OR_RUN_NOSAVE = 180
    RUN_OR_RUN_LOAD_PASS = RUN_OR_RESUME = RESUME_OR_RUN = 175
    RUN_OVERRIDE = OVERRIDE = 150
    RUN_OR_SKIP = SKIP_OR_RUN = 125
    RUN_ONLY = 100
    LOAD_OR_RUN_NOSAVE = RUN_OR_LOAD_NOSAVE = 80
    LOAD_OR_RUN = RUN_OR_LOAD = 50
    LOAD_OR_SKIP = SKIP_OR_LOAD = 40
    LOAD = 10
    SKIP = 0


def save_pickle(obj, filename, verbose=2, use_joblib=False, ret=False):
    """"
        Note: joblib can be potentially VERY slow.
    """
    with open(filename, 'wb') as file:
        start = time.perf_counter()
        if verbose >= 2:
            logging.info(f'Dumping PICKLE to {filename}...')
        if use_joblib:
            warnings.warn('Joblib is slower in newer versions of Python.')
            import joblib
            joblib.dump(obj, file)
        else:
            # Use most recent protocol (fastest)
            pickle.dump(obj, file, protocol=-1)
        if verbose >= 2:
            logging.info(f'Dump saved in {np.round(time.perf_counter()-start, 4)} seconds.')
        if verbose >= 1:
            logging.info(f'Dumped PICKLE {filename}')

    if ret:
        return obj


def is_file_too_old(filename, time_since_modification=None, ret_time=False):
    time_since_modification_ = time.time() - os.path.getmtime(filename)
    ret = time_since_modification_ > time_since_modification

    if ret_time:
        return ret, time_since_modification_
    else:
        return ret


def load_pickle(
    filename: Union[str, os.PathLike],
    verbose: int = 2,
    use_joblib: bool = False,
    time_since_modification: Optional[float] = None,
):
    """
        Note: joblib can be potentially VERY slow.
    """
    if time_since_modification is not None:
        too_old, time_since_modification_ = is_file_too_old(filename, time_since_modification, ret_time=True)
        if too_old:
            raise FileNotFoundError(
                f'File {filename} is too old ({duration_to_long_string(time_since_modification_)} ago).')

    with open(filename, 'rb') as file:
        if verbose >= 2:
            start = time.perf_counter()
            logging.info(f'Loading PICKLE from {filename}...')
        if use_joblib:
            warnings.warn('Joblib is slower in newer versions of Python.')
            import joblib
            obj = joblib.load(file)
        else:
            try:
                obj = pickle.load(file)
            except EOFError as e:
                logging.error(f'Load FAILED for {filename}.')
                raise e
        if verbose >= 2:
            logging.info(f'Load done in {np.round(time.perf_counter()-start, 4)} seconds.')
        return obj


def compute_or_load(
    pkl_filename,
    func,
    *args,
    forcegen: Union[bool, None] = None,
    force=None,
    verbose=2,
    request: ComputeRequest = None,
    expect_nb_rets=1,
    use_joblib=False,
    resume_loaded_argument_name=None,
    **kwargs,
):
    '''
        Function that:
         - Loads a pickle if it exists
         - Compute the passe function otherwise


        Deprecated:
            forcegen : if True behaves as ComputeRequest.RUN_OVERRIDE
            force : Same as forcegen
    '''

    pickle_kwargs = dict(verbose=verbose, use_joblib=use_joblib)

    def __load():
        if isinstance(pkl_filename, (list, tuple)):
            return tuple([load_pickle(f, **pickle_kwargs) for f in pkl_filename])
        else:
            return load_pickle(pkl_filename, **pickle_kwargs)

    def __save(obj):
        if isinstance(pkl_filename, (list, tuple)):
            assert len(pkl_filename) == len(obj)
            for f, o in zip(pkl_filename, obj):
                save_pickle(o, f, **pickle_kwargs)
        else:
            save_pickle(obj, pkl_filename, **pickle_kwargs)

    def __exists():
        if isinstance(pkl_filename, (list, tuple)):
            return all([os.path.exists(f) for f in pkl_filename])
        else:
            return os.path.exists(pkl_filename)

    def skip():
        if expect_nb_rets == 1:
            return None
        else:
            return tuple([None] * expect_nb_rets)

    def load():
        loaded = __load()
        if expect_nb_rets != 1:
            if not isinstance(loaded, tuple):
                raise TypeError(f'Loaded pickle is not a tuple. Expecting {expect_nb_rets} arguments in a tuple.')
            if len(loaded) < expect_nb_rets:
                loaded = loaded + tuple([None] * (expect_nb_rets - len(loaded)))
        return loaded

    def compute(save=True):
        if verbose >= 3:
            logging.info('Computation...')
        computed = func(*args, **kwargs)
        if save:
            __save(computed)
        return computed

    def resume(save=True):
        loaded = load()
        if verbose >= 1:
            logging.info('Resuming computation...')
        computed = func(*args, **{resume_loaded_argument_name: loaded}, **kwargs)
        if save:
            __save(computed)
        return computed

    # Retro-compatible
    if forcegen is not None and force is not None:
        raise ValueError("Both forcegen and force set!")
    if forcegen is not None:

        force = forcegen

    # Cases
    if request is not None:
        if force is not None:
            logging.warning('Warning: Ignoring force. request (ComputeRequest) is superseeding force!')

        if (request is ComputeRequest.LOAD_OR_RUN) or (request is ComputeRequest.LOAD_OR_RUN_NOSAVE):
            if __exists():
                return load()
            else:
                return compute(save=(request is ComputeRequest.LOAD_OR_RUN))

        if (request is ComputeRequest.RUN_OR_RESUME) or (request is ComputeRequest.RUN_OR_RESUME_NOSAVE):
            if resume_loaded_argument_name is None:
                raise ValueError(
                    'You must provide the name of the argument to pass the loaded obejct using "resume_loaded_argument_name" when using RUN_OR_RUN_LOAD_PASS'
                )
            if __exists():
                return resume(save=(request is ComputeRequest.RUN_OR_RESUME))
            else:
                return compute()

        elif request is ComputeRequest.RUN_OVERRIDE:
            return compute()

        elif request is ComputeRequest.RUN_ONLY:
            return func(*args, **kwargs)

        if request is ComputeRequest.RUN_OR_SKIP:
            if __exists():
                logging.info('File exists, returning none.')
                return skip()
            else:
                logging.info('File does not exists, computing...')
                return compute()

        elif request is ComputeRequest.LOAD_OR_SKIP:
            if __exists():
                return load()
            else:
                return skip()

        elif request is ComputeRequest.LOAD:
            return load()

        elif request is ComputeRequest.SKIP:
            return skip()

        else:
            raise ValueError('Unrecognized request.')

    else:
        if force is None:
            force = False
        if force:
            if verbose >= 1:
                logging.info('Forcing computation.')
            return compute()
        elif __exists():
            return load_pickle(pkl_filename, verbose=verbose, use_joblib=use_joblib)
        else:
            return compute()


run_or_load = compute_or_load