"""
    Counterfactuals as projection onto the decision boundary (estimated by bisection).
"""

__author__ = 'Emanuele Albini'
__all__ = ['BisectionProjectionDBCounterfactuals']

import numpy as np
from tqdm import tqdm

from ..base import BaseMultipleCounterfactualMethod
from ..utils import keydefaultdict
from ..utils.project import find_decision_boundary_bisection


class BisectionProjectionDBCounterfactuals(BaseMultipleCounterfactualMethod):
    def __init__(self,
                 model,
                 data,
                 num_iters=100,
                 earlystop_error=1e-8,
                 scaler=None,
                 max_samples=1e10,
                 nb_diverse_counterfactuals=None,
                 random_state=0,
                 verbose=1,
                 **kwargs):

        if scaler is not None:
            raise NotImplementedError('Scaling is not supported by Cone.')

        super().__init__(model, scaler, random_state=random_state)

        self.num_iters = num_iters
        self.earlystop_error = earlystop_error
        self.kwargs = kwargs

        self._max_samples = min(max_samples or np.inf, nb_diverse_counterfactuals or np.inf)

        self.data = data
        self.verbose = verbose

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is not None:
            self._raw_data = self.preprocess(data)
            self._preds = self.model.predict(self._raw_data)

            self._data = keydefaultdict(
                lambda pred: self.sample(self._raw_data[self._preds != pred], n=self._max_samples))
        else:
            self._raw_data = None
            self._preds = None
            self._data = None

    def __get_counterfactuals(self, x, pred):
        if self.data is not None:
            return np.array(
                find_decision_boundary_bisection(
                    x=x,
                    Y=self.data[pred],
                    model=self.model,
                    scaler=self.scaler,
                    num=self.num_iters,
                    error=self.earlystop_error,
                    method='counterfactual',
                    desc=None,
                    # model_parallelism=1,
                    # n_jobs=1,
                    **self.kwargs,
                ))
        else:
            raise ValueError('Invalid state. `self.data` must be set.')

    def get_multiple_counterfactuals(self, X):
        X = self.preprocess(X)
        preds = self.model.predict(X)
        X_counter = []
        iters = tqdm(zip(X, preds)) if self.verbose else zip(X, preds)
        for x, pred in iters:
            X_counter.append(self.__get_counterfactuals(x, pred))

        return self.diverse_postprocess(X, X_counter)
