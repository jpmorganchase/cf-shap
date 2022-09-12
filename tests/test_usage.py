"""
    Limited-tests of basic usage.

    TODO: Improve tests coverage.
"""

import pytest
import itertools

from emutils.datasets import load_dataset
from emutils.model.train import train_model
from emutils.preprocessing.quantiletransformer import EfficientQuantileTransformer
from cfshap.counterfactuals import KNNCounterfactuals
from cfshap.attribution import TreeExplainer, CompositeExplainer
from cfshap.trend import TrendEstimator

dataset_names = ['boston', 'breastcancer']
model_types = ['xgb', 'rf']
random_seeds = [0]


@pytest.mark.parametrize('dataset_name, model_type, random_seed',
                         itertools.product(dataset_names, model_types, random_seeds))
def test_usage(dataset_name, model_type, random_seed):
    X, y = load_dataset(dataset_name, task='classification', return_X_y=True)
    model = train_model(X, y, model_type=model_type, params=dict(random_state=random_seed))

    scaler = EfficientQuantileTransformer()
    scaler.fit(X)

    trend_estimator = TrendEstimator(strategy='mean')

    explainer = CompositeExplainer(
        KNNCounterfactuals(
            model=model,
            X=X,
            n_neighbors=100,
            distance='cityblock',
            scaler=scaler,
            max_samples=10000,
        ),
        TreeExplainer(
            model,
            data=None,
            trend_estimator=trend_estimator,
            max_samples=10000,
        ),
    )

    return explainer(X[:10])
