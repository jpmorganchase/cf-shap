"""
    This module implements the (Mean/Median) Absolute Deviation Scalers

    NOTE: The acronym MAD (Mean/Median Absolute Deviation) has many different meaning.
    This module implements all the four possible alternatives:
    - Mean Absolute Deviation Scaler from the Mean
    - Median Absolute Deviation Scaler from the Median
    - Mean Absolute Deviation Scaler from the Median
    - Median Absolute Deviation Scaler from the Mean

"""

__all__ = [
    'MeanAbsoluteDeviationFromMeanScaler',
    'MeanAbsoluteDeviationFromMedianScaler',
    'MedianAbsoluteDeviationFromMeanScaler',
    'MedianAbsoluteDeviationFromMedianScaler',
    'MedianAbsoluteDeviationScaler',
    'MeanAbsoluteDeviationScaler',
]
__author__ = 'Emanuele Albini'

from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (check_is_fitted, FLOAT_DTYPES)
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils import check_array


def mean_absolute_deviation(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def mean_absolute_deviation_from_median(data, axis=None):
    return np.mean(np.absolute(data - np.median(data, axis)), axis)


def median_absolute_deviation_from_mean(data, axis=None):
    return np.median(np.absolute(data - np.mean(data, axis)), axis)


class AbsoluteDeviationScaler(TransformerMixin, BaseEstimator, ABC):
    """
        This class is the interface and base class for the Mean/Median Absolute Deviation Scalers.
        It implements the scikit-learn API for scalers.

        This class is based on scikit-learn StandardScaler and RobustScaler.
    """
    def __init__(self, *, copy=True, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.copy = copy

    @abstractmethod
    def _center(self, X):
        pass

    @abstractmethod
    def _scale(self, X):
        pass

    def _check_inputs(self, X):
        try:
            X = self._validate_data(X,
                                    accept_sparse='csc',
                                    estimator=self,
                                    dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan')
        except AttributeError:
            X = check_array(X,
                            accept_sparse='csr',
                            copy=self.copy,
                            estimator=self,
                            dtype=FLOAT_DTYPES,
                            force_all_finite='allow-nan')
        return X

    def fit(self, X, y=None):
        X = self._check_inputs(X)

        if self.with_centering:
            if sparse.issparse(X):
                raise ValueError("Cannot center sparse matrices: use `with_centering=False`"
                                 " instead. See docstring for motivation and alternatives.")
            self.center_ = self._center(X)
        else:
            self.center_ = None

        if self.with_scaling:
            self.scale_ = self._scale(X)
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self, ['center_', 'scale_'])
        X = self._check_inputs(X)

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, 1.0 / self.scale_)
        else:
            if self.with_centering:
                X -= self.center_
            if self.with_scaling:
                X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The rescaled data to be transformed back.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self, ['center_', 'scale_'])
        X = check_array(X,
                        accept_sparse=('csr', 'csc'),
                        copy=self.copy,
                        estimator=self,
                        dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_scaling:
                X *= self.scale_
            if self.with_centering:
                X += self.center_
        return X

    def _more_tags(self):
        return {'allow_nan': True}


class MeanAbsoluteDeviationFromMeanScaler(AbsoluteDeviationScaler):
    """Mean absolute deviation scaler (from the mean)
        It scales using the the MEAN deviation from the MEAN
    """
    def _center(self, X):
        return np.nanmean(X, axis=0)

    def _scale(self, X):
        return mean_absolute_deviation(X, axis=0)


class MedianAbsoluteDeviationFromMedianScaler(AbsoluteDeviationScaler):
    """Median absolute deviation scaler (from the median)
        It scales using the the MEDIAN deviation from the MEDIAN
    """
    def _center(self, X):
        return np.nanmedian(X, axis=0)

    def _scale(self, X):
        return stats.median_absolute_deviation(X, axis=0)


class MeanAbsoluteDeviationFromMedianScaler(AbsoluteDeviationScaler):
    """Mean absolute deviation scaler (from the median)
        It scales using the the MEAN deviation from the MEDIAN
    """
    def _center(self, X):
        return np.nanmean(X, axis=0)

    def _scale(self, X):
        return mean_absolute_deviation_from_median(X, axis=0)


class MedianAbsoluteDeviationFromMeanScaler(AbsoluteDeviationScaler):
    """Median absolute deviation scaler (from the mean)
        It scales using the the MEDIAN deviation from the MEAN
    """
    def _center(self, X):
        return np.nanmean(X, axis=0)

    def _scale(self, X):
        return median_absolute_deviation_from_mean(X, axis=0)


# Aliases
# By default we return the matching absolute scalers (median-median and mean-mean).
MeanAbsoluteDeviationScaler = MeanAbsoluteDeviationFromMeanScaler
MedianAbsoluteDeviationScaler = MedianAbsoluteDeviationFromMedianScaler