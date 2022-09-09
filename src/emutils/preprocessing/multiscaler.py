"""
    Author: Emanuele Albini

    This module implements the MultiScaler.
    The multi scaler is a scaler that allows for different scaling within the same class through an argument passed to the `transform` methods.
    e.g., STD, MAD, Quantile, etc.

"""

from typing import Union

import numpy as np
import scipy as sp

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import keydefaultdict

from .madscaler import (
    MedianAbsoluteDeviationScaler,
    MeanAbsoluteDeviationScaler,
    MeanAbsoluteDeviationFromMedianScaler,
    MedianAbsoluteDeviationFromMeanScaler,
)
from .quantiletransformer import EfficientQuantileTransformer

__all__ = [
    'MultiScaler',
    'IdentityScaler',
    'get_scaler_name',
    'SCALERS',
]

# NOTE: This will be deprecated, it is confusing
_DISTANCE_TO_SCALER = {
    'euclidean': 'std',
    'manhattan': 'mad',
    'cityblock': 'mad',
    'percshift': 'quantile',
}

_SCALER_TO_NAME = {
    'quantile': 'Quantile',
    'std': 'Standard',
    'mad': 'Median Absolute Dev. (from median)',
    'Mad': 'Mean Absolute Dev. (from mean)',
    'madM': 'Median Absolute Dev. from mean',
    'Madm': 'Mean Absolute Dev. from median',
    'minmax': 'Min Max',
    'quantile_nan': 'Quantile w/NaN OF',
    'quantile_sum': 'Quantile w/OF-Σ',
    None: 'Identity',
}

SCALERS = list(_SCALER_TO_NAME.keys())


def _get_method_safe(method):
    if method is None or method in SCALERS:
        return method
    # NOTE: This will be deprecated, it is confusing
    elif method in list(_DISTANCE_TO_SCALER.keys()):
        return _DISTANCE_TO_SCALER[method]
    else:
        raise ValueError('Invalid normalization method.')


def get_scaler_name(method):
    return _SCALER_TO_NAME[_get_method_safe(method)]


class IdentityScaler(TransformerMixin, BaseEstimator):
    """A dummy/identity scaler compatible with the sklearn interface for scalers
        It returns the same input it receives.

    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def get_transformer_class(method):
    method = _get_method_safe(method)

    if method is None:
        return IdentityScaler
    elif method == 'std':
        return StandardScaler
    elif method == 'minmax':
        return MinMaxScaler
    elif method == 'mad':
        return MedianAbsoluteDeviationScaler
    elif method == 'Mad':
        return MeanAbsoluteDeviationScaler
    elif method == 'Madm':
        return MeanAbsoluteDeviationFromMedianScaler
    elif method == 'madM':
        return MedianAbsoluteDeviationFromMeanScaler
    elif method == 'quantile':
        return EfficientQuantileTransformer  # PercentileShifterCached
    elif method == 'quantile_nan':
        return EfficientQuantileTransformer
    elif method == 'quantile_sum':
        return EfficientQuantileTransformer


def get_transformer_kwargs(method):
    method = _get_method_safe(method)

    if method == 'quantile_nan':
        return dict(overflow="nan")
    elif method == 'quantile_sum':
        return dict(overflow="sum")
    else:
        return dict()


class MultiScaler:
    """Multi-Scaler

    Raises:
        Exception: If an invalid normalization is used

    """
    # Backward compatibility
    NORMALIZATION = SCALERS

    def __init__(self, data: np.ndarray = None):
        """Constructor

            - The data on which to train the scalers can be passed here (in the constructor), or
            - It can also be passe later using the .fit(data) method.

        Args:
            data (pd.DataFrame): A dataframe with the features, optional. Default to None.

        """

        if data is not None:
            return self.fit(data)

        self.__suppress_warning = False

    def fit(self, data: np.ndarray):
        self.data = np.asarray(data)

        self.transformers = keydefaultdict(lambda method: get_transformer_class(method)
                                           (**get_transformer_kwargs(method)).fit(self.data))

        def single_transformer(method, f):
            return get_transformer_class(method)(**get_transformer_kwargs(method)).fit(self.data[:, f].reshape(-1, 1))

        self.single_transformers = keydefaultdict(lambda args: single_transformer(*args))
        self.data_transformed = keydefaultdict(lambda method: self.transform(self.data, method))

        self.covs = keydefaultdict(lambda method, lib: self.__compute_covariance_matrix(self.data, method, lib))

        self._lower_bounds = keydefaultdict(lambda method: self.data_transformed[method].min(axis=0))
        self._upper_bounds = keydefaultdict(lambda method: self.data_transformed[method].max(axis=0))

    def lower_bounds(self, method):
        return self._lower_bounds[method]

    def upper_bounds(self, method):
        return self._upper_bound[method]

    def suppress_warnings(self, value=True):
        self.__suppress_warning = value

    def __compute_covariance_matrix(self, data, method, lib):
        if lib == 'np':
            return sp.linalg.inv(np.cov((self.transform(data, method=method)), rowvar=False))
        elif lib == 'tf':
            from ..tf import inv_cov as tf_inv_cov
            return tf_inv_cov(self.transform(data, method=method))
        else:
            raise ValueError('Invalid lib.')

    def transform(self, data: np.ndarray, method: str, **kwargs):
        """Normalize the data according to the "method" passed

        Args:
            data (np.ndarray): The data to be normalized (nb_samples x nb_features)
            method (str, optional): Normalization (see class documentation for details on the available scalings). Defaults to 'std'.

        Raises:
            ValueError: Invalid normalization

        Returns:
            np.ndarray: Normalized array
        """
        method = _get_method_safe(method)
        return self.transformers[method].transform(data)

    def inverse_transform(self, data: np.ndarray, method: str = 'std'):
        """Un-normalize the data according to the "method" passes

        Args:
            data (np.ndarray): The data to be un-normalized (nb_samples x nb_features)
            method (str, optional): Normalization (see class documentation for details on the available scalings). Defaults to 'std'.

        Raises:
            ValueError: Invalid normalization

        Returns:
            np.ndarray: Un-normalized array
        """
        method = _get_method_safe(method)
        return self.transformers[method].inverse_transform(data)

    def feature_deviation(self, method: str = 'std', phi: Union[float, int] = 1):
        """Get the deviation of each feature according to the normalization method

        Args:
            method (str): method (str, optional): Normalization (see class documentation for details on the available scalings). Defaults to 'std'.
            phi (Union[float, int]): The fraction of the STD/MAD/MINMAX. Default to 1.

        Raises:
            ValueError: Invalid normalization

        Returns:
            np.ndarray: Deviations, shape = (nb_features, )
        """
        method = _get_method_safe(method)
        transformer = self.transformers[method]
        if 'scale_' in dir(transformer):
            return transformer.scale_ * phi
        else:
            return np.ones(self.data.shape[1]) * phi

    def feature_transform(self, x: np.ndarray, f: int, method: str):
        x = np.asarray(x)
        transformer = self.get_feature_transformer(method, f)
        return transformer.transform(x.reshape(-1, 1))[:, 0]

    def value_transform(self, x: float, f: int, method: str):
        x = np.asarray([x])
        transformer = self.get_feature_transformer(method, f)
        return transformer.transform(x.reshape(-1, 1))[:, 0][0]

    def shift_transform(self, X, shifts, method, **kwargs):
        transformer = self.get_transformer(method)
        if 'shift' in dir(transformer):
            return transformer.shift_transform(X, shifts=shifts, **kwargs)
        else:
            return X + shifts

    def move_transform(self, X, costs, method, **kwargs):
        transformer = self.get_transformer(method)
        assert costs.shape[0] == X.shape[1]
        return transformer.inverse_transform(transformer.transform(X) + np.tile(costs, (X.shape[0], 1)))

    def get_transformer(self, method: str):
        return self.transformers[_get_method_safe(method)]

    def get_feature_transformer(self, method: str, f: int):
        return self.single_transformers[(_get_method_safe(method), f)]

    def single_transform(self, x, *args, **kwargs):
        return self.transform(np.array([x]), *args, **kwargs)[0]

    def single_inverse_transform(self, x, *args, **kwargs):
        return self.inverse_transform(np.array([x]), *args, **kwargs)[0]

    def single_shift_transform(self, x, shift, **kwargs):
        return self.shift_transform(np.array([x]), np.array([shift]), **kwargs)[0]

    # NOTE: This must be deprecated, it does not fit here.
    def covariance_matrix(self, data: np.ndarray, method: Union[None, str], lib='np'):
        if data is None:
            return self.covs[(method, lib)]
        else:
            return self.__compute_covariance_matrix(data, method, lib)

    # NOTE: This must be deprecated, it does not fit here.
    def covariance_matrices(self, data: np.ndarray, methods=None, lib='np'):
        """Compute the covariance matrices

        Args:
            data (np.ndarray): The data from which to extract the covariance

        Returns:
            Dict[np.ndarray]: Dictionary (for each normalization method) of covariance matrices
        """
        # If no method is passed we compute for all of them
        if methods is None:
            methods = self.NORMALIZATION

        return {method: self.covariance_matrix(data, method, lib) for method in methods}