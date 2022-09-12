import logging
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from emutils.utils import attrdict
from emutils.preprocessing.multiscaler import MultiScaler

from ....base import Model
from ....utils.tree import TreeEnsemble, get_shap_compatible_model
from ._utils import (
    transform_trees,
    mask_values,
    inverse_actionabiltiy,
)

__all__ = ['TreeInducedCounterfactualGeneratorV2']


class TreeInducedCounterfactualGeneratorV2:
    def __init__(
        self,
        model: Model,
        data: np.ndarray = None,
        multiscaler: np.ndarray = None,
        global_feature_trends: Union[None, list, np.ndarray] = None,
        random_state: int = 0,
    ):
        assert data is not None or multiscaler is not None

        self.model = model
        self.data = data
        self.multiscaler = multiscaler
        self.global_feature_trends = global_feature_trends
        self.random_state = random_state

    @property
    def multiscaler(self):
        return self._multiscaler

    @multiscaler.setter
    def multiscaler(self, multiscaler):
        self._multiscaler = multiscaler

    @property
    def data(self):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        if data is not None:
            self._multiscaler = MultiScaler(self._data)

    @property
    def model(self):
        return self._model

    @property
    def shap_model(self):
        return self._shap_model

    @model.setter
    def model(self, model):
        self._model = model
        self._shap_model = get_shap_compatible_model(self.model)

    def transform(
        self,
        X: np.ndarray,
        explanations: attrdict,
        K: Tuple[int, float] = (1, 2, 3, 4, 5, np.inf),
        action_cost_normalization: str = 'quantile',
        action_cost_aggregation: str = 'L1',  # L0, L1, L2, Linf
        action_strategy: str = 'proportional',  # 'equal', 'proportional', 'random'
        action_scope: str = 'positive',  # 'positive', 'negative', 'all'
        action_direction: str = 'local',  # 'local', 'global', 'random'
        precision: float = 1e-4,
        show_progress: bool = True,
        starting_sample: int = 0,
        max_samples: Union[int, None] = None,
        nan_explanation: str = 'raise',  # 'ignore'
        counters: np.ndarray = None,
        costs: np.ndarray = None,
        desc: str = "",
    ):

        # Must pass trends if global trends are used
        assert self.global_feature_trends is not None or action_direction != 'global'
        # Must pass local trends if local trends are used
        assert hasattr(explanations, 'trends') or action_direction != 'local'

        assert (counters is None and costs is None) or (counters is not None and costs is not None)

        assert action_cost_aggregation[0] == 'L'
        action_cost_aggregation = float(action_cost_aggregation[1:])

        random_state = np.random.RandomState(self.random_state)

        # Compute stuff from the results
        Xn = self.multiscaler.transform(X, method=action_cost_normalization)
        Pred = self.model.predict(X)
        Phi = explanations.values

        # Let's raise some warning in case of NaN
        nb_PhiNaN = np.any(np.isnan(Phi), axis=1).sum()
        if nb_PhiNaN > 0:
            logging.warning(f'There are {nb_PhiNaN} NaN explanations.')
        if nb_PhiNaN == len(X):
            logging.error('All explanations are NaN.')

        # Pre-compute directions of change
        if action_direction == 'local':
            Directions = explanations.trends
        elif action_direction == 'global':
            Directions = (-1 * (np.broadcast_to(Pred.reshape(-1, 1), X.shape) - .5) * 2 *
                          np.broadcast_to(np.array(self.global_feature_trends), X.shape)).astype(int)
        elif action_direction == 'random':
            Directions = random_state.choice((-1, 1), X.shape)
        else:
            raise ValueError('Invalid direction_strategy.')

        # If we don't have a direction we set Phi to 0
        Phi = Phi * (Directions != 0)

        logging.info(f"There are {np.any(Directions == 0, axis = 1).sum()}/{len(X)} samples with some tau_i = 0")

        assert np.all((Directions == -1) | (Directions == +1) | (np.isnan(Phi))
                      | ((Directions == 0) & (Phi == 0))), "Some trends are not +1 or -1"

        # Compute the order of feature-change
        # NOTE: This must be done before the action_strategy computation below!
        if action_scope == 'all':
            Order = np.argsort(-1 * np.abs(Phi), axis=1)
        elif action_scope == 'positive':
            Order = np.argsort(-1 * Phi, axis=1)
            Phi = Phi * (Phi > 0)
        elif action_scope == 'negative':
            Order = np.argsort(+1 * Phi, axis=1)
            Phi = Phi * (Phi < 0)
        else:
            raise ValueError('Invalid action_scope.')

        assert np.all((Phi >= 0) | (np.isnan(Phi))), "Something weird happend: Phi should be >= 0 at this point"

        # Modify Phi based on the action strategy
        if action_strategy == 'proportional':
            pass
        elif action_strategy == 'equal':
            Phi = 1 * (Phi > 0)
        elif action_strategy == 'random':
            Phi = random_state.random(Phi.shape) * (Phi != 0) * (~np.isnan(Phi))
        elif action_strategy == 'noisy':
            Phi = Phi * random_state.random(Phi.shape)
        else:
            raise ValueError('Invalid action_strategy.')

        if len(np.unique(Pred)) != 1:
            raise RuntimeError('Samples have different predicted class! This is weird: stopping.')

        assert np.all((Phi >= 0) | (np.isnan(Phi))), "Something weird happend: Phi should still be >= 0 at this point"

        # Do some shape checks
        assert X.shape == Xn.shape
        assert X.shape == Phi.shape
        assert X.shape == Order.shape
        assert X.shape == Directions.shape
        assert (X.shape[0], ) == Pred.shape
        assert np.all(self.model.predict(X) == Pred)
        assert np.all(Pred % Pred.astype(np.int) == 0)

        # Compute trees (and cost-normalized version)
        trees = TreeEnsemble(self.shap_model).trees
        ntrees = transform_trees(trees, self.multiscaler, action_cost_normalization)

        # Run the experiments from scratch
        if counters is None:
            counters = []
            costs = []

        # Resume the experiments
        else:
            assert counters.shape[0] == costs.shape[0] == (min(X[starting_sample:].shape[0], max_samples or np.inf))
            assert counters.shape[1] == costs.shape[1]
            assert counters.shape[1] >= len(K)
            assert counters.shape[2] == X.shape[1]

            counters = counters.copy()
            costs = costs.copy()

            counters = counters.transpose((1, 0, 2))  # Top-K x Samples x nb_features
            costs = costs.transpose((1, 0))  # Top-K x Samples
            recompute_mask = np.isinf(costs)

        for kidx, k in enumerate(K):
            # Mask only top-k features
            PhiMasked = mask_values(Phi, Order, k)

            # Create object to be iterated
            iters = np.array(list(zip(X, Xn, PhiMasked, Directions, Pred))[starting_sample:], dtype=np.object)
            if max_samples is not None:
                iters = iters[:max_samples]
            if not isinstance(counters, list):
                iters = iters[recompute_mask[kidx]]

            if show_progress:
                iters = tqdm(
                    iters,
                    desc=(f'{desc} - (A)St={action_strategy}/D={action_direction}/Sc={action_scope}/K={k} '
                          '(C)N={action_cost_normalization}/Agg={action_cost_aggregation}'),
                )

            # Iterate over all samples
            results = []

            for a, args in enumerate(iters):
                # print(a) # TQDM may not be precise
                results.append(
                    inverse_actionabiltiy(
                        *args,
                        model=self.model,
                        multiscaler=self.multiscaler,
                        normalization=action_cost_normalization,
                        trees=trees,
                        ntrees=ntrees,
                        degree=action_cost_aggregation,
                        precision=precision,
                        error=nan_explanation,
                    ))

            counters_ = [r[0] for r in results]
            costs_ = [r[1] for r in results]
            if isinstance(counters, list):
                counters.append(counters_)
                costs.append(costs_)
            else:
                if recompute_mask[kidx].sum() > 0:
                    counters[kidx][recompute_mask[kidx]] = np.array(counters_)
                    costs[kidx][recompute_mask[kidx]] = np.array(costs_)

        # Counterfactuals
        if isinstance(counters, list):
            counters = np.array(counters)  # Top-K x Samples x nb_features
        counters = counters.transpose((1, 0, 2))  # Samples x Top-K x nb_features

        # Costs
        if isinstance(costs, list):
            costs = np.array(costs)  # Top-K x Samples
        costs = costs.transpose((1, 0))  # Samples x Top-K

        return counters, costs