"""
    Author: Emanuele Albini

    This module provides the wrappers for ML models implementing the standard interface (scikit-learn and XGBoost)
"""

from types import MethodType
import warnings

import numpy as np
# from emutils.math import sigmoid
from emutils.preprocessing import process_data
from cached_property import cached_property

# Fail-safe imports (for the case in which XGBoost is not installed)
try:
    import xgboost as xgb
except (ModuleNotFoundError, ImportError):
    pass

__all__ = [
    'UniversalSKLWrapper',
    'XGBClassifierSKLWrapper',
]


class UniversalSKLWrapper(object):
    """Universal Wrapper for models to an SKL (scikit-learn).

        Usage:
            new_model = UniversalSKLWrapper(model, *args, **kwargs)
            new_model will be a wrapper or monkey-patched version of the original model.
        
        It will implement:
        - predict : That returns the prediction
        - predict_proba : That returns the probabilities of each class
        - decision_function (optional) : That return the raw results of the decision function

    """
    def __new__(cls, model, *args, **kwargs):
        """Wraps or Monkey-patch an existing model

        Args:
            model : The model
        
        Based on the type of model different arguments will be available.
        Based on the type of model a different model will be returned.
        
        - xgboost.Booster : An XGBClassifierSKLWrapper will be returned.
                            The arguments will be passed to the constructor of XGBClassifierSKLWrapper.
                            See XGBClassifierSKLWrapper for more details on the available arguments.

        """
        module_name, class_name = model.__class__.__module__, model.__class__.__name__

        # XGBoost Booster (binary)
        if (module_name, class_name) == ('xgboost.core', 'Booster'):
            return XGBClassifierSKLWrapper(model, *args, **kwargs)

        # SKL Classifiers (binary)
        elif module_name.startswith('sklearn') and hasattr(model, 'predict_proba'):
            # Get the number of classes
            if hasattr(model, 'n_classes_'):
                nclasses = model.n_classes_
            elif hasattr(model, 'classes_'):
                nclasses = len(model.classes_)
            else:
                nclasses = None

            if nclasses == 2:
                return skl_binary_wrapper(model, *args, **kwargs)
            else:
                raise NotImplementedError('This SKL model class is not supported.')

        else:
            raise NotImplementedError('Model type not supported.')


def skl_predict_binary(self, X, *args, **kwargs):
    return 1 * (self.predict_proba(X, *args, **kwargs)[:, 1] > self.threshold)


# Monkey patch-er for SKL binary prediction
def skl_binary_wrapper(model, threshold=.5, **kwargs):
    if len(kwargs) > 0:
        warnings.warn(f'Ignoring unsupported kwargs {list(kwargs.values())}')
    model.threshold = threshold
    model.predict = MethodType(skl_predict_binary, model)

    return model


class XGBClassifierSKLWrapper:
    """
        Model wrapper that provide the `predict` and `predict_proba` interface
        for compatibility with all methods expecting SKLEARN-like output
    """
    def __init__(self, booster, features, classes=2, ntree_limit=0, threshold=.5, missing=None):

        self.booster = booster
        self.ntree_limit = ntree_limit
        self.threshold = threshold
        self.features = features

        self.missing = missing

        # scikit-learn compatible attributes
        self.classes_ = np.arange(classes) if isinstance(classes, int) else np.array(classes)
        self.n_classes_ = classes if isinstance(classes, int) else len(classes)
        self.n_features_ = features if isinstance(features, int) else len(features)
        self.n_outputs_ = 1

    @cached_property
    def dmatrix_kwargs(self):
        kwargs = {}
        if hasattr(self, 'missing') and self.missing is not None:
            kwargs['missing'] = self.missing
        return kwargs

    def predict(self, X, *args, **kwargs) -> np.ndarray:
        """
            Returns:
                [np.ndarray] : shape = (n_samples)
                The prediction (0 or 1), it returns 1 iff `probability of class 1 > self.threshold`
        """
        X = process_data(X, ret_type='np', names=self.features)
        return (self.booster.predict(xgb.DMatrix(X, **self.dmatrix_kwargs), ntree_limit=self.ntree_limit) >
                self.threshold) * 1

    def predict_margin(self, X, *args, **kwargs):
        X = process_data(X, ret_type='np', names=self.features)
        return self.booster.predict(xgb.DMatrix(X, **self.dmatrix_kwargs),
                                    ntree_limit=self.ntree_limit,
                                    output_margin=True)

    def decision_function(self, *args, **kwargs) -> np.ndarray:
        return self.predict_margin(*args, **kwargs)

    def predict_probs(self, X, *args, **kwargs):
        """
            Returns:
                [np.ndarray] : shape = (n_samples)
                It is the probability of class 1
        """
        X = process_data(X, ret_type='np', names=self.features)
        return self.booster.predict(xgb.DMatrix(X, **self.dmatrix_kwargs), ntree_limit=self.ntree_limit)

    def predict_proba(self, X, *args, **kwargs) -> np.ndarray:
        """
            Returns:
                [np.ndarray] : shape = (n_samples, 2)
                [:, 0] is the probability of class 0
                [:, 1] is the probability of class 1
        """
        ps = self.predict_probs(X, *args, **kwargs).reshape(-1, 1)
        return np.hstack([1 - ps, ps])

    # def margin_to_probs(self, margin, *args, **kwargs):
    #     return sigmoid(margin)

    # def margin_to_prob(self, margin, *args, **kwargs):
    #     return sigmoid(margin)

    def prob_to_predict(self, prob):
        assert isinstance(prob, np.number)
        return 1 * (prob > self.threshold)

    def probs_to_predict(self, probs):
        assert len(probs.shape) == 1
        return 1 * (probs > self.threshold)

    def proba_to_predict(self, proba):
        assert len(proba.shape) == 2
        return 1 * (proba[:, 1] > self.threshold)

    def get_booster(self):
        return self.booster
