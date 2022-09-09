"""
    Author: Emanuele Albini

    SHAP Wrapper
"""

from abc import ABC
from typing import Union
import numpy as np
import pandas as pd

from shap.maskers import Independent as SHAPIndependent
from shap import TreeExplainer as SHAPTreeExplainer
from shap.explainers import Exact as SHAPExactExplainer

from ..base import (BaseExplainer, BaseSupportsDynamicBackground, Model, TrendEstimatorProtocol)
from ..utils import get_shap_compatible_model

__all__ = [
    'TreeExplainer',
    'ExactExplainer',
]


class SHAPExplainer(BaseExplainer, BaseSupportsDynamicBackground, ABC):
    @property
    def explainer(self):
        if self._data is None:
            self._raise_data_error()
        return self._explainer

    def get_backgrounds(self, X):
        return np.broadcast_to(self.data, [X.shape[0], *self.data.shape])

    def get_trends(self, X):
        if self.trend_estimator is None:
            return np.full(X.shape, np.nan)
        else:
            return np.array([self.trend_estimator(x, self.data) for x in X])


class TreeExplainer(SHAPExplainer):
    """Monkey-patched and improved wrapper for (Interventional) TreeSHAP
    """
    def __init__(
        self,
        model: Model,
        data: Union[None, np.ndarray] = None,
        max_samples: int = 1e10,
        class_index: int = 1,
        trend_estimator: Union[None, TrendEstimatorProtocol] = None,
        **kwargs,
    ):
        """Create the TreeSHAP explainer.

        Args:
            model (Model): The model
            data (Union[None, np.ndarray], optional): The background dataset. Defaults to None.
                If None it must be set dynamically using self.data = ...
            max_samples (int, optional): Maximum number of (random) samples to draw from the background dataset. Defaults to 1e10.
            class_index (int, optional): Class for which SHAP values will be computed. Defaults to 1 (positive class in a binary classification setting).
            trend_estimator (Union[None, TrendEstimatorProtocol], optional): The feature trend estimator. Defaults to None.
            **kwargs: Other arguments to be passed to shap.TreeExplainer.__init__

        """
        super().__init__(model, None)

        self.max_samples = max_samples
        self.class_index = class_index
        self.trend_estimator = trend_estimator
        self.kwargs = kwargs

        # Allow only for interventional SHAP
        if ('feature_perturbation' in kwargs) and (kwargs['feature_perturbation'] != 'interventional'):
            raise NotImplementedError()

        self.data = data

    @BaseSupportsDynamicBackground.data.setter
    def data(self, data):
        if data is not None:
            if not hasattr(self, '_explainer'):
                self._explainer = SHAPTreeExplainer(
                    get_shap_compatible_model(self.model),
                    data=SHAPIndependent(self.preprocess(data), max_samples=self.max_samples),
                    **self.kwargs,
                )

            # Copied and adapted from SHAP 0.39.0 __init__
            ### Start of copy ###
            self._explainer.data = self._explainer.model.data = self._data = SHAPIndependent(
                self.preprocess(data), max_samples=self.max_samples).data
            self._explainer.data_missing = self._explainer.model.data_missing = pd.isna(self._explainer.data)
            try:
                self._explainer.expected_value = self._explainer.model.predict(self._explainer.data).mean(0)
            except ValueError:
                raise Exception("Currently TreeExplainer can only handle models with categorical splits when " \
                                "feature_perturbation=\"tree_path_dependent\" and no background data is passed. Please try again using " \
                                "shap.TreeExplainer(model, feature_perturbation=\"tree_path_dependent\").")
            if hasattr(self._explainer.expected_value, '__len__') and len(self._explainer.expected_value) == 1:
                self._explainer.expected_value = self._explainer.expected_value[0]

            # if our output format requires binary classification to be represented as two outputs then we do that here
            if self._explainer.model.model_output == "probability_doubled" and self._explainer.expected_value is not None:
                self._explainer.expected_value = [1 - self._explainer.expected_value, self._explainer.expected_value]
            ### End of copy ####
        else:
            self._data = None

    def get_attributions(self, *args, **kwargs) -> np.ndarray:
        """Compute SHAP values
            Proxy for shap.TreeExplainer.shap_values(*args, **kwargs)

        Returns:
            np.ndarray: nb_samples x nb_features array with SHAP values of class self.class_index
        """
        r = self.explainer.shap_values(*args, **kwargs)

        # Select Shapley values of a specific class
        if isinstance(r, list):
            return r[self.class_index]
        else:
            return r


class ExactExplainer(SHAPExplainer):
    def __init__(self, model, data=None, max_samples=1e10, function: str = 'predict', **kwargs):
        """Monkey-patched and improved constructor for TreeSHAP

        Args:
            model (any): A model
            data (np.ndarray, optional): Reference data. Defaults to None.
            max_samples (int, optional): Maximum number of samples in the reference data. Defaults to 1e10.
            function (str): the name of the function to call on the model.
            **kwargs: Other arguments to be passed to shap.ExactExplainer.__init__
        """

        super().__init__(model, None)

        self.max_samples = max_samples
        self.function = function
        self.kwargs = kwargs

        if ('identity' in kwargs) and (kwargs['link'] != 'identity'):
            raise NotImplementedError()

        self.data = data

    @BaseSupportsDynamicBackground.data.setter
    def data(self, data):
        if data is not None:
            self._data = self.preprocess(data)
            self._explainer = SHAPExactExplainer(
                lambda X: getattr(self.model, self.function)(X),
                SHAPIndependent(self._data, max_samples=self.max_samples),
                **self.kwargs,
            )
        else:
            self._data = None

    def get_attributions(self, *args, **kwargs):
        """Compute SHAP values
            Proxy for shap.ExactExplainer.shap_values(*args, **kwargs)

        Returns:
            np.ndarray: nb_samples x nb_features array with SHAP values
        """
        return self.explainer(*args, **kwargs).values