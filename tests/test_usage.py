"""
    Tests of basic usage.
"""

import pytest
import itertools

from sklearn.datasets import load_boston
from xgboost import XGBClassifier

from cfshap.utils.preprocessing import EfficientQuantileTransformer
from cfshap.counterfactuals import KNNCounterfactuals
from cfshap.attribution import TreeExplainer, CompositeExplainer
from cfshap.trend import TrendEstimator


def test_usage():

    dataset = load_boston()
    X = dataset.data
    y = (dataset.target > 21).astype(int)

    model = XGBClassifier()
    model.fit(X, y)

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
