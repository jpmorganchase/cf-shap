"""
    Author: Emanuele Albini

    This module implements an efficient and exact version of the scikit-learn QuantileTransformer.
    
    Note: This module has been inspired from scikit-learn QuantileTransformer
    (and it is effectively an extension of QuantileTransformer API)
    See https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/preprocessing
    
"""
# %%
import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import (check_is_fitted, FLOAT_DTYPES)
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_array

__all__ = ['EfficientQuantileTransformer']


class EfficientQuantileTransformer(QuantileTransformer):
    """
        This class directly extends and improve the efficiency of scikit-learn QuantileTransformer

        Note: The efficient implementation will be only used if:
        - The input are NumPy arrays (NOT scipy sparse matrices)
        The flag self.smart_fit_ marks when the efficient implementation is being used.
        
    """
    def __init__(
        self,
        *,
        subsample=np.inf,
        random_state=None,
        copy=True,
        overflow=None,  # "nan" or "sum"
    ):
        """Initialize the transformer

        Args:
            subsample (int, optional): Number of samples to use to create the quantile space. Defaults to np.inf.
            random_state (int, optional): Random seed (sampling happen only if subsample < number of samples fitted). Defaults to None.
            copy (bool, optional): If False, passed arrays will be edited in place. Defaults to True.
            overflow (str, optional): Overflow strategy. Defaults to None.
            When doing the inverse transformation if a quantile > 1 or < 0 is passed then:
                - None > Nothing is done. max(0, min(1, q)) will be used. The 0% or 100% reference will be returned.
                - 'sum' > It will add proportionally, e.g., q = 1.2 will result in adding 20% more quantile to the 100% reference.
                - 'nan' > It will return NaN
        """
        self.ignore_implicit_zeros = False
        self.n_quantiles_ = np.inf
        self.output_distribution = 'uniform'
        self.subsample = subsample
        self.random_state = random_state
        self.overflow = overflow
        self.copy = copy

    def _smart_fit(self, X, random_state):
        n_samples, n_features = X.shape
        self.references_ = []
        self.quantiles_ = []
        for col in X.T:
            # Do sampling if necessary
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(n_samples, size=self.subsample, replace=False)
                col = col.take(subsample_idx, mode='clip')
            col = np.sort(col)
            quantiles = np.sort(np.unique(col))
            references = 0.5 * np.array(
                [np.searchsorted(col, v, side='left') + np.searchsorted(col, v, side='right') for v in quantiles]
            ) / n_samples
            self.quantiles_.append(quantiles)
            self.references_.append(references)

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        y : None
            Ignored.
        Returns
        -------
        self : object
           Fitted transformer.
        """

        if self.subsample <= 1:
            raise ValueError(
                "Invalid value for 'subsample': %d. "
                "The number of subsamples must be at least two." % self.subsample
            )

        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        if n_samples <= 1:
            raise ValueError(
                "Invalid value for samples: %d. "
                "The number of samples to fit for must be at least two." % n_samples
            )

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.smart_fit_ = not sparse.issparse(X)
        if self.smart_fit_:  # <<<<<- New case
            self._smart_fit(X, rng)
        else:
            raise NotImplementedError('EfficientQuantileTransformer handles only NON-sparse matrices!')

        return self

    def _smart_transform_col(self, X_col, quantiles, references, inverse):
        """Private function to transform a single feature."""

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        # Simply Interpolate
        if not inverse:
            X_col[isfinite_mask] = np.interp(X_col_finite, quantiles, references)
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, references, quantiles)

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        try:
            X = self._validate_data(
                X, reset=in_fit, accept_sparse=False, copy=copy, dtype=FLOAT_DTYPES, force_all_finite='allow-nan'
            )
        except AttributeError:  # Old sklearn version (_validate_data do not exists)
            X = check_array(X, accept_sparse=False, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if (
                not accept_sparse_negative and not self.ignore_implicit_zeros and
                (sparse.issparse(X) and np.any(X.data < 0))
            ):
                raise ValueError('QuantileTransformer only accepts' ' non-negative sparse matrices.')

        # check the output distribution
        if self.output_distribution not in ('normal', 'uniform'):
            raise ValueError(
                "'output_distribution' has to be either 'normal'"
                " or 'uniform'. Got '{}' instead.".format(self.output_distribution)
            )

        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.
        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        for feature_idx in range(X.shape[1]):
            X[:, feature_idx] = self._smart_transform_col(
                X[:, feature_idx], self.quantiles_[feature_idx], self.references_[feature_idx], inverse
            )

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self, ['quantiles_', 'references_', 'smart_fit_'])
        X = self._check_inputs(X, in_fit=False, copy=self.copy)
        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The data used to scale along the features axis.
            
        Returns
        -------
        Xt : {ndarray, sparse matrix} of (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self, ['quantiles_', 'references_', 'smart_fit_'])
        X = self._check_inputs(X, in_fit=False, accept_sparse_negative=False, copy=self.copy)

        if self.overflow is None:
            T = self._transform(X, inverse=True)
        elif self.overflow == 'nan':
            NaN_mask = np.ones(X.shape)
            NaN_mask[(X > 1) | (X < 0)] = np.nan
            T = NaN_mask * self._transform(X, inverse=True)

        elif self.overflow == 'sum':
            ones = self._transform(np.ones(X.shape), inverse=True)
            zeros = self._transform(np.zeros(X.shape), inverse=True)

            # Standard computation
            T = self._transform(X.copy(), inverse=True)

            # Deduct already computed part
            X = np.where((X > 0), np.maximum(X - 1, 0), X)

            # After this X > 0 => Remaining quantile > 1.00
            # and X < 0 => Remaining quantile < 0.00

            T = T + (X > 1) * np.floor(X) * (ones - zeros)
            X = np.where((X > 1), np.maximum(X - np.floor(X), 0), X)
            T = T + (X > 0) * (ones - self._transform(1 - X.copy(), inverse=True))

            T = T - (X < -1) * np.floor(-X) * (ones - zeros)
            X = np.where((X < -1), np.minimum(X + np.floor(-X), 0), X)
            T = T - (X < 0) * (self._transform(-X.copy(), inverse=True) - zeros)

            # Set the NaN the values that have not been reached after a certaing amount of iterations
            # T[(X > 0) | (X < 0)] = np.nan

        else:
            raise ValueError('Invalid value for overflow.')

        return T


# %%
