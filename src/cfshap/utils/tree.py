"""
    Author: Emanuele Albini

    Utilities for tree-based models explainability.
"""

from collections import defaultdict
import numpy as np

from emutils.preprocessing import process_data
from ..utils import get_shap_compatible_model

try:
    # Newer versions of SHAPss
    from shap.explainers._tree import TreeEnsemble
except:
    # Older versions of SHAP
    from shap.explainers.tree import TreeEnsemble

__all__ = [
    'TreeEnsemble',
    'get_splits',
    'splits_to_values',
    'SplitTransformer',
]


def get_splits(model, nb_features):
    """Extract feature splits from a tree-based model
        In order to extract the trees this method rely on the SHAP API:
        any model supported by SHAP is supported.

    Args:
        model (Any tree-based model): The model
        nb_features (int): The number of features (sometimes it cannot be induced by the model)

    Returns:
        List[np.ndarray]: A list of splits for each of the features
            nb_features x nb_splits (it may different for each feature)
    """

    # Convert the model to something that SHAP API can understand
    model = get_shap_compatible_model(model)

    # Extract the trees (using TreeEnseble API from SHAP)
    ensemble = TreeEnsemble(model)

    splits = defaultdict(list)
    for tree in ensemble.trees:
        for i in range(len(tree.features)):
            if tree.children_left[i] != -1:  # i.e., it is a split (and not a leaf)
                splits[tree.features[i]].append(tree.thresholds[i])

    assert len(splits) <= nb_features
    assert max(list(splits.keys())) < nb_features

    return [np.sort(np.array(list(set(splits[i])))) for i in range(nb_features)]


def __splits_to_values(splits, how: str, eps: float):
    if len(splits) == 0:
        return [0]

    if how == 'left':
        return ([splits[0] - eps] + [(splits[i] + eps) for i in range(len(splits) - 1)] + [splits[-1] + eps])
    elif how == 'right':
        return ([splits[0] - eps] + [(splits[i + 1] - eps) for i in range(len(splits) - 1)] + [splits[-1] + eps])
    elif how == 'center':
        return ([splits[0] - eps] + [(splits[i] + splits[i + 1]) / 2
                                     for i in range(len(splits) - 1)] + [splits[-1] + eps])
    else:
        raise ValueError('Invalid mode.')


def splits_to_values(splits, how: str = 'center', eps: float = 1e-6):
    """Convert lists of splits in values (in between splits)

    Args:
        splits (List[np.ndarray]): List of splits (one per each feature) obtained using get_feature_splits
        how (str): Where should the values be wrt to the splits. Default to 'center'.
            'center': precisely in between the two splits (except for the first and last split)
            'right': split - eps
            'left': split + eps
            e.g. if we indicate with | the splits and the values with x
            'center': x|    x    |   x   |  x  |     x    |x
            'left':   x|x        |x      |x    |x         |x
            'right':  x|        x|      x|    x|         x|x

        eps (float, optional): epsilon to use for the conversion of splits to values. Defaults to 1e-6.

    Returns:
        List[np.ndarray]: List of values for the features supported based on the splits (one list per feature)
    """

    return [np.unique(np.array(__splits_to_values(s, how=how, eps=eps))) for s in splits]


class SplitTransformer:
    def __init__(self, model, nb_features=None):
        """The initialization is lazy because we may not know yet the number of features"""
        if hasattr(model, 'get_booster'):
            model = model.get_booster()
        self.model = model
        self.nb_features = nb_features
        self.values = None
        self.splits = None

        self._initialize()

    def _initialize(self, X=None):
        if self.nb_features is None and X is not None:
            self.nb_features = X.shape[1]
        if (self.values is None or self.splits is None) and (self.nb_features is not None):
            self.splits = get_splits(self.model, self.nb_features)
            self.values = splits_to_values(self.splits, how='center')

    def get_nb_value(self, i):
        return len(self.splits[i]) + 1

    def transform(self, X):
        """Takes inputs and transform them into discrete splits"""
        A = process_data(X, ret_type='np')
        self._initialize(A)
        for i in range(A.shape[1]):
            if len(self.splits[i]) > 0:
                A[:, i] = np.digitize(A[:, i], self.splits[i])
            else:
                A[:, i] = 0
        return process_data(A, Xret_type=X, Xindex=X, Xnames=X).astype(int)

    def inverse_transform(self, X):
        """Takes splits and transform them into continuous inputs
            Note: The post-condition T^(-1)(T(X)) == X may not hold.
        """
        X_ = process_data(X, ret_type='np').astype(int)
        self._initialize(X)
        A = np.full(X.shape, np.nan)
        for i in range(X.shape[1]):
            A[:, i] = self.values[i][X_[:, i]]
        return process_data(A, Xret_type=X, Xindex=X, Xnames=X)


# import numba
# try:
#     from shap.explainers._tree import TreeEnsemble
# except:
#     # Older versions
#     from shap.explainers.tree import TreeEnsemble
# @numba.jit(nopython=True, nogil=True)
# def predict_tree(x, region, features, values, thresholds, children_left, children_right, children_default):
#     n = 0
#     while n != -1:  # -1 => Leaf
#         f = features[n]
#         t = thresholds[n]
#         v = x[f]

#         if v < t:  # Left
#             # Constrain region
#             region[f, 1] = np.minimum(region[f, 1], t)
#             #                 print(f'T{i}: x_{f} = {v} < {t}')

#             # Go to child
#             n = children_left[n]

#         elif v >= t:  # Right
#             # Constrain region
#             region[f, 0] = np.maximum(region[f, 0], t)
#             #                 print(f'T{i}: x_{f} = {v} >= {t}')

#             # Go to child
#             n = children_right[n]

#         else:  # Missing
#             n = children_default[n]
#             region[f] = np.nan

#     return region

# def predict_region(x, region, ensemble):
#     for i, tree in enumerate(ensemble.trees):
#         region = predict_tree(
#             x, region, tree.features, tree.values, tree.thresholds, tree.children_left, tree.children_right,
#             tree.children_default
#         )
#     return region

# def predict_regions(X, model):
#     ensemble = TreeEnsemble(model)
#     regions = np.tile(
#         np.array([-np.inf, np.inf], dtype=ensemble.trees[0].thresholds.dtype), (X.shape[0], X.shape[1], 1)
#     )
#     return np.stack([predict_region(x, region, ensemble) for x, region in zip(X, regions)])