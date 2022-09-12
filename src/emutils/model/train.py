"""
    Author: Emanuele Albini

    Model Training Utilties.
"""

import logging
import os
import time
import random
import warnings

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from emutils.file import save_pickle, load_pickle, save_json, load_json
from emutils.parallel import max_cpu_count

from .wrappers import XGBClassifierSKLWrapper

__all__ = [
    'train_model',
    'train_xgb',
]


def _parse_param_and_delete(params, name, default):
    if name in params:
        value = params[name]
        del params[name]
    else:
        value = default
    return params, value


def _parse_param_and_keep(params, name, default):
    if name not in params:
        params[name] = default
    return params


def _parse_param_and_preprocess(params, name, func):
    if name in params:
        params[name] = func(params[name])
    return params


def _parse_alias(params, name, alias):
    if alias in params:
        if name in params:
            raise ValueError('{name} and {alias} are the same parameter. Set only one.')
        params[name] = params[alias]
        del params[alias]
    return params


def parse_param_and_separate(params, other_params, name, default, ignore_default=False):
    if name in params:
        value = params[name]
        del params[name]
        other_params[name] = value
    else:
        if not ignore_default:
            other_params[name] = default

    return params, other_params


def train_sklearn(Classifier, X, y, save_path=None, params={}):
    X = np.asarray(X)
    y = np.asarray(y).flatten()

    model = Classifier(**params)
    model.fit(X, y.flatten())
    if save_path is not None:
        save_pickle(model, save_path)

    return model


def train_decision_tree(*args, **kwargs):
    return train_sklearn(DecisionTreeClassifier, *args, **kwargs)


def train_adoboost(*args, **kwargs):
    return train_sklearn(AdaBoostClassifier, *args, **kwargs)


def train_rf(*args, **kwargs):
    return train_sklearn(RandomForestClassifier, *args, **kwargs)


def train_xgb(X, y, params, save_path=None, wrapper=False):
    from xgboost import XGBClassifier
    from xgboost import Booster, DMatrix

    if save_path is not None:
        assert save_path.endswith('.pkl') or save_path.endswith('.pickle'), 'The model filename must be .pkl or .pickle'

    # Pre-process parameters
    params = _parse_alias(params, name='n_estimators', alias='num_boost_round')

    # the threshold is not handled by XGB interface (but by our)
    params, binary_threshold = _parse_param_and_delete(params, 'binary_threshold', .5)

    # n_jobs is handled by XGB SKL interface
    params = _parse_param_and_keep(params, name='n_jobs', default=max(1, min(max_cpu_count() - 1, 24)))

    # This is to avoid XGB complaining
    params = _parse_param_and_keep(params, name='use_label_encoder', default=False)

    X = np.asarray(X)
    y = np.asarray(y).flatten()
    assert X.shape[0] == y.shape[0]

    if wrapper:
        if not tuple(np.sort(np.unique(y))) == (0, 1):
            raise NotImplementedError('XGB Wrapper currently only support binnary classification.')

    # Fit the model
    model = XGBClassifier()
    model = clone(model)
    model.set_params(**params)

    logging.info('Training...')
    model.fit(
        X,
        y,
        # early_stopping_rounds=10,
        verbose=True,
    )

    if wrapper:
        # Save and re-load (feature-agnostic model)
        temp_file = f'temp-{time.perf_counter()}-{random.random()}.bin'
        model.get_booster().save_model(temp_file)
        booster = Booster(model_file=temp_file)
        os.remove(temp_file)

        if binary_threshold == 'auto':
            p_ = booster.predict(DMatrix(X))
            p_ = np.sort(p_)
            binary_threshold = p_[int((y == 0).sum())]

        logging.info(f'Using a binary_threshold = {binary_threshold}')

        # Wrap
        model = XGBClassifierSKLWrapper(booster, features=X.shape[1], threshold=binary_threshold)

    # Save
    if save_path is not None:
        save_pickle(model, save_path)
        save_booster(model.get_booster(), '.'.join(save_path.split('.')[:-1]))
    return model


def save_booster(model, path):
    """Save the booster model in BIN, JSON and TXT

    Args:
        model (xgboost.Booster): The model
        path (str): The path without extension or '.bin'
    """
    if path.endswith('.bin') or path.endswith('.pkl'):
        path = path[:-4]

    model.save_model(path + '.bin')
    model.dump_model(path + '.json')
    model.dump_model(path + '.txt')


def train_xgbbooster(X, y, params, save_path=None):
    import xgboost

    if save_path is not None:
        assert save_path.endswith('.bin')

    # Pre-process parameters
    train_params = {}

    params = _parse_alias(params, name='num_boost_round', alias='n_estimators')

    params, train_params = parse_param_and_separate(params,
                                                    train_params,
                                                    name='num_boost_round',
                                                    default=None,
                                                    ignore_default=True)

    # Prepare data
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    assert X.shape[0] == y.shape[0]
    dtrain = xgboost.DMatrix(data=X, label=y)

    # Train
    model = xgboost.train(params=params, dtrain=dtrain, **train_params)

    # Save
    if save_path is not None:
        save_booster(model, save_path)

    return model


def get_model_json_filename(model_filename):
    if model_filename is not None:
        return model_filename.replace('.pkl', '') + '.json'
    return None


# from .utils import get_model_type, ModelType

# def load_model(model_filename):
#     model = load_pickle(model_filename)
#     model_type = get_model_type(model)

#     if model_type == ModelType.XGBOOST_WRAPPER:
#         model.get_booster().set_param({'nthread': min(15, max_cpu_count() - 1)})
#     elif model_type == ModelType.XGBOOST_BOOSTER:
#         model.set_param({'nthread': min(15, max_cpu_count() - 1)})
#       TODO: Wrap model with wrappers? I'd say so.
#     return model


def train_model(
        X,
        y,
        model_type: str,
        model_filename: str = None,
        params: dict = dict(),
        override=True,
        override_params=True,
):

    if model_filename is None:
        logging.warning('No model_filename passed. The model (binary) will not be saved.')
    else:
        if not model_filename.endswith('.pkl'):
            raise ValueError('Model filename must end with .pkl')

    model_parameters_filename = get_model_json_filename(model_filename)

    # We try to use saved model
    if not override:
        if model_filename is not None:
            try:  # Load model from PKL
                model = load_pickle(model_filename)
                logging.info('MODEL PKL FOUND: loading saved model.')
                return model
            except FileNotFoundError:
                logging.info('MODEL PKL NOT FOUND: model will be trained from scratch.')

    # We try to use save model parameters
    if not override_params:
        try:  # Load model parameters from JSON
            params = load_json(model_parameters_filename)
            logging.info('PARAMETERS found: loading saved parameters.')
        except FileNotFoundError:
            logging.info('PARAMETERS NOT FOUND: using passed parameters.')

    # Cast to dict (e.g., from attrdict, defaultdict)
    params = dict(params)

    # Train
    if model_type in ['dt', 'tree']:
        model = train_decision_tree(X, y, save_path=model_filename, params=params)
    elif model_type == 'ada':
        model = train_adoboost(X, y, save_path=model_filename, params=params)
    elif model_type == 'rf':
        model = train_rf(X, y, save_path=model_filename, params=params)
    elif model_type == 'xgbclassifier':
        model = train_xgb(X, y, params=params, save_path=model_filename, wrapper=False)
    elif model_type == 'xgbwrapper':
        model = train_xgb(X, y, params=params, save_path=model_filename, wrapper=True)
    elif model_type == 'xgb':
        warnings.warn("model_type = 'xgb' is interpreted as 'xgbwrapper (our binary wrapper)'.")
        model = train_xgb(X, y, params=params, save_path=model_filename, wrapper=True)
    elif model_type == 'xgbbooster' or model_type == 'xgbooster':
        model = train_xgbbooster(X, y, params, save_path=model_filename)
    else:
        raise ValueError('Invalid model type!')

    # Save parameters used for training
    if model_parameters_filename is not None:
        save_json(params, model_parameters_filename)

    return model