"""
    Author: Emanuele Albini

    CF-SHAP Explainer, it uses a CF generator togheter with an explainer that supports at-runtime background distributions.
"""

from abc import abstractmethod, ABC
from typing import Iterable, Union, List
import numpy as np

from ..base import (
    BaseExplainer,
    ExplainerSupportsDynamicBackground,
    BackgroundGenerator,
    CounterfactualMethod,
    ListOf2DArrays,
)

from ..background import CounterfactualMethodBackgroundGeneratorAdapter

from emutils.utils import (
    attrdict,
    import_tqdm,
)

tqdm = import_tqdm()

__all__ = [
    'CompositeExplainer',
    'CFExplainer',
]


class CompositeExplainer(BaseExplainer):
    """CompositeExplainer allows to compose:
        - a background generator (or a counterfactual generation method), and
        - an explainer (i.e., feature importance method) that suppports a background distributions
        into a new explainer that uses the dataset obtained with the background generator 
        as background for the feature importance method.

    """
    def __init__(
        self,
        background_generator: Union[CounterfactualMethod, BackgroundGenerator],
        explainer: ExplainerSupportsDynamicBackground,
        n_top: Union[None, int] = None,
        verbose: bool = True,
    ):
        """

        Args:
            background_generator (Union[CounterfactualMethod, BackgroundGenerator]): A background generator.
            explainer (ExplainerSupportsDynamicBackground): An explainer supporting a dynamic background dataset.
            n_top (Union[None, int], optional): Number of top counterfactuals (only if background_generator is a counterfactual method). Defaults to None.
            verbose (bool, optional): If True, will be verbose. Defaults to True.

        """

        super().__init__(background_generator.model)
        if isinstance(background_generator, CounterfactualMethod):
            background_generator = CounterfactualMethodBackgroundGeneratorAdapter(background_generator, n_top=n_top)
        else:
            if n_top is not None:
                raise NotImplementedError('n_top is supported only for counterfactual methods.')

        self.background_generator = background_generator
        self.explainer = explainer
        self.verbose = verbose

    def get_backgrounds(
        self,
        X: np.ndarray,
        background_data: Union[None, ListOf2DArrays] = None,
    ) -> ListOf2DArrays:
        """Generate the background datasets for the query instances.

        Args:
            X (np.ndarray): The query instances
            background_data (Union[None, ListOf2DArrays], optional): The background datasets. Defaults to None.
                By default it will recompute the background for each query instance using the backgroud generator passed in the constructor.
                One may consider passing the background_data here to accelerate the execution if the method is called multiple times on the same query instances.

        Returns:
            ListOf2DArrays : A list/array of background datasets
                nb_query_instances x nb_background_samples x nb_features
        """
        X = self.preprocess(X)

        # If we do not have any background then we compute it
        if background_data is None:
            return self.background_generator.get_backgrounds(X)

        # If we have some background we use it
        else:
            assert len(self.background_data) == X.shape[0]
            return background_data

    def _get_backgrounds_iterator(self, X, background_data):
        iters = zip(X, background_data)
        if len(X) > 100 and self.verbose:
            iters = tqdm(iters)

        return iters

    def get_attributions(
        self,
        X: np.array,
        background_data: Union[None, ListOf2DArrays] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate the feature attributions for the query instances.

        Args:
            X (np.ndarray): The query instances
            background_data (Union[None, ListOf2DArrays], optional): The background datasets. Defaults to None.
                By default it will recompute the background for each query instance using the backgroud generator passed in the constructor.
                One may consider passing the background_data here to accelerate the execution if the method is called multiple times on the same query instances.

        Returns:
            np.ndarray : An array of feature attributions
                nb_query_instances x nb_features
        """
        X = self.preprocess(X)
        backgrounds = self.get_backgrounds(X, background_data)

        shapvals = []
        for x, background in self._get_backgrounds_iterator(X, backgrounds):
            if len(background) > 0:
                # Set background
                self.explainer.data = background
                # Compute Shapley values
                shapvals.append(self.explainer.get_attributions(x.reshape(1, -1), **kwargs)[0])
            else:
                shapvals.append(np.full(X.shape[1], np.nan))
        return np.array(shapvals)

    def get_trends(
        self,
        X: np.array,
        background_data: Union[None, ListOf2DArrays] = None,
    ):
        """Generate the feature trends for the query instances.

        Args:
            X (np.ndarray): The query instances
            background_data (Union[None, ListOf2DArrays], optional): The background datasets. Defaults to None.
                By default it will recompute the background for each query instance using the backgroud generator passed in the constructor.
                One may consider passing the background_data here to accelerate the execution if the method is called multiple times on the same query instances.

        Returns:
            np.ndarray : An array of feature trends (+1 / -1)
                nb_query_instances x nb_features
        """
        X = self.preprocess(X)
        backgrounds = self.get_backgrounds(X, background_data)

        trends = []
        for x, background in self._get_backgrounds_iterator(X, backgrounds):
            if len(background) > 0:
                self.explainer.data = background
                trends.append(self.explainer.get_trends(x.reshape(1, -1))[0])
            else:
                trends.append(np.full(X.shape[1], np.nan))
        return np.array(trends)

    def __call__(
        self,
        X: np.array,
        background_data: Union[None, ListOf2DArrays] = None,
    ):
        """Generate the explanations for the query instances.

        Args:
            X (np.ndarray): The query instances
            background_data (Union[None, ListOf2DArrays], optional): The background datasets. Defaults to None.
                By default it will recompute the background for each query instance using the backgroud generator passed in the constructor.
                One may consider passing the background_data here to accelerate the execution if the method is called multiple times on the same query instances.

        Returns:
            attrdict : The explanations.

            See BaseExplainer.__call__ for more details on the output format.
        """
        X = self.preprocess(X)
        backgrounds = self.get_backgrounds(X, background_data)

        shapvals = []

        # Check if trends is implemented
        try:
            self.get_trends(X[:1])
            trends = []
        except NotImplementedError:
            trends = None

        for x, background in self._get_backgrounds_iterator(X, backgrounds):
            if len(background) > 0:
                self.explainer.data = background
                shapvals.append(self.explainer.get_attributions(x.reshape(1, -1))[0])
                if trends is not None:
                    trends.append(self.explainer.get_trends(x.reshape(1, -1))[0])
            else:
                shapvals.append(np.full(X.shape[1], np.nan))
                if trends is not None:
                    trends.append(np.full(X.shape[1], np.nan))
        return attrdict(
            backgrounds=backgrounds,
            values=np.array(shapvals),
            trends=np.array(trends) if trends is not None else None,
        )


# Alias
CFExplainer = CompositeExplainer