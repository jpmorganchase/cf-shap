"""
    Author: Emanuele Albini

    Adapter from a counterfactual method to background generator.
"""
from typing import Union
import numpy as np

from ..base import (BaseBackgroundGenerator, BackgroundGenerator, CounterfactualMethod, MultipleCounterfactualMethod,
                    ListOf2DArrays)

from ..utils import (
    get_top_counterfactuals,
    expand_dims_counterfactuals,
)

__all__ = ['CounterfactualMethodBackgroundGeneratorAdapter']


class CounterfactualMethodBackgroundGeneratorAdapter(BaseBackgroundGenerator, BackgroundGenerator):
    """Adapter to make a counterfactual method into a background generator"""
    def __init__(
            self,
            counterfactual_method: CounterfactualMethod,
            n_top: Union[None, int] = None,  # By default: All
    ):
        """

        Args:
            counterfactual_method (CounterfactualMethod): The counterfactual method
            n_top (Union[None, int], optional): Number of top-counterfactuals to select as background. Defaults to None (all).
        """
        self.counterfactual_method = counterfactual_method
        self.n_top = n_top

    def get_backgrounds(self, X: np.ndarray) -> ListOf2DArrays:
        """Generate the background datasets for each query instance

        Args:
            X (np.ndarray): The query instances

        Returns:
            ListOf2DArrays: A list/array of background datasets
                nb_query_intances x nb_background_points x nb_features
        """

        X = self.preprocess(X)

        # If we do not have any background then we compute it
        if isinstance(self.counterfactual_method, MultipleCounterfactualMethod):
            if self.n_top is None:
                return self.counterfactual_method.get_multiple_counterfactuals(X)
            elif self.n_top == 1:
                return expand_dims_counterfactuals(self.counterfactual_method.get_counterfactuals(X))
            else:
                return get_top_counterfactuals(
                    self.counterfactual_method.get_multiple_counterfactuals(X),
                    X,
                    n_top=self.n_top,
                    nan=False,
                )
        elif isinstance(self.counterfactual_method, CounterfactualMethod):
            if self.n_top is not None and self.n_top != 1:
                raise ValueError('Counterfactual methodology do not supportthe generation of multiple counterfactuals.')
            return np.expand_dims(self.counterfactual_method.get_counterfactuals(X), axis=1)
        else:
            raise NotImplementedError('Unsupported CounterfactualMethod.')