# %%
from collections import defaultdict
import numpy as np
import scipy as sp
import scipy.spatial

from emutils.utils import keydefaultdict
from emutils.parallel import batch_process

NORM_NAMES = keydefaultdict(lambda l: f'L{l} Norm')


def nnhamming(x, y):
    return (x != y).sum()


CUSTOM_DISTANCE_NAMES = {
    'L0': 'hamming',
    'L1': 'cityblock',
    'L2': 'euclidean',

    # Alias
    'manhattan': 'cityblock',
    "cosine_distance": "cosine",

    # Minkowski np.inf
    'Linf': 'chebyshev',
    'max': 'chebyshev',

    # Weighted
    'wmanhattan': 'wcityblock',
    'wL1': 'wcityblock',
    'wL2': 'weuclidean',

    # Hamming distance (without normalization)
    'nnhamming': nnhamming,
}

CUSTOM_DISTANCE_NAMES_SET = set(list(CUSTOM_DISTANCE_NAMES))

PROXY_DISTANCES = {
    'wchebyshev': ('wminkowski', {
        'p': np.inf
    }),
    'weuclidean': ('wminkowski', {
        'p': 2
    }),
    'wcityblock': ('wminkowski', {
        'p': 1
    }),
}

RELATIVE_DISTANCE_NORM = defaultdict(
    lambda X, Y: np.ones(X.shape[0]), {
        'euclidean': lambda X, Y: np.linalg.norm((X + Y) / 2, ord=2, axis=1),
        'cityblock': lambda X, Y: np.linalg.norm((X + Y) / 2, ord=1, axis=1),
        'chebyshev': lambda X, Y: np.max((X + Y) / 2, axis=1),
    })

RELATIVE_DISTANCE_NORM2 = {
    'euclidean': lambda X, Y: (np.linalg.norm(X, ord=2, axis=1) + np.linalg.norm(Y, ord=2, axis=1)) / 2,
    'cityblock': lambda X, Y: (np.linalg.norm(X, ord=1, axis=1) + np.linalg.norm(Y, ord=1, axis=1)) / 2,
    'chebyshev': lambda X, Y: np.maximum(np.max(X, axis=1), np.max(Y, axis=1)),
}


def metric_to_function(metric):
    if isinstance(metric, str):
        return getattr(sp.spatial.distance, metric)
    else:
        return metric


def adist(X, Y, metric, **metric_params):
    metric, metric_params = get_metric_name_and_params(metric, **metric_params)
    dist_func = metric_to_function(metric)
    return np.array([dist_func(x, y, **metric_params) for x, y in zip(X, Y)])


def radist(X, Y, metric, norm_abs=False, **metric_params):
    dist = adist(X, Y, metric=metric, **metric_params)
    metric, metric_params = get_metric_name_and_params(metric, **metric_params)
    if norm_abs:
        dist_n = RELATIVE_DISTANCE_NORM2[metric](X, Y)
    else:
        dist_n = RELATIVE_DISTANCE_NORM[metric](X, Y)

    return dist / dist_n


def norm_distance(x, y, **kwargs):
    return np.linalg.norm(x - y, **kwargs)


def get_metric_name(metric):
    if metric in CUSTOM_DISTANCE_NAMES_SET:
        return CUSTOM_DISTANCE_NAMES[metric]
    return metric


def get_metric_params(metric, **kwargs):
    metric_params = {}
    if metric == 'mahalanobis':
        if 'IV' in kwargs:
            metric_params.update({'VI': kwargs['IV']})
        # Use the data
        elif 'data' in kwargs:
            metric_params.update({'VI': sp.linalg.inv(np.cov(kwargs['data'], rowvar=False))})
        # Use the multiscaler
        elif 'multiscaler' in kwargs:
            method = kwargs['method'] if 'method' in kwargs else None
            lib = kwargs['lib'] if 'lib' in kwargs else 'np'
            metric_params.update({'VI': kwargs['multiscaler'].covariance_matrix(data=None, method=method, lib=lib)})
    if metric == 'minkowski':
        metric_params.update(dict(p=kwargs['p']))

    if metric == 'wminkowski':
        metric_params.update(dict(w=kwargs['weight']))
    return metric_params


def get_metric_name_and_params(metric, **kwargs):
    metric = get_metric_name(metric)
    metric_params = get_metric_params(metric, **kwargs)
    return metric, metric_params


def cdist(
    X,
    Y,
    distance,
    aggregate=None,
    normalize=False,

    # Parallel
    split_size=None,
    desc=None,
    verbose=1,
    n_jobs=1,

    # Other metric params
    **kwargs,
):
    """
        Returns:
            X.shape[0] x Y.shape[0] if aggregate is None
            X.shape[0] otherwise
    """

    if Y is None:
        Y = X

    # Batching
    if split_size is not None:
        return batch_process(
            # Recursive call
            f=lambda X: cdist(
                X=X,
                Y=Y,
                distance=distance,
                aggregate=aggregate,
                normalize=normalize,
                split_size=None,
                **kwargs,
            ),
            iterable=X,
            split_size=split_size,
            desc=desc,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # Get distance and params
    distance, metric_params = get_metric_name_and_params(distance, **kwargs)

    # Computation
    ret = sp.spatial.distance.cdist(
        X,
        Y,
        metric=distance,
        **metric_params,
    )

    if normalize:
        if distance == 'cityblock':
            ret = ret / X.shape[1]
        elif distance == 'euclidean':
            ret = ret / np.sqrt(X.shape[1])
        elif distance == 'cosine':
            pass
        else:
            raise ValueError('Normalization is unsupported for this distance.')

    # Aggregate distance
    if aggregate is not None:
        return aggregate(np.array(ret), axis=1)
    else:
        return np.array(ret)


pairwise_distance = cdist

# %%
