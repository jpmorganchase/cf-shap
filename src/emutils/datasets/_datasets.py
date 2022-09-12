"""
    Loading and pre-processing of many popular datasets.

    NOTE/TODO: Some of the loading can be devolved to uci.py for some of these datasets
    (this would save space in the repo).
"""

__all__ = ['load_dataset', 'get_dataset_name']
__author__ = 'Emanuele Albini'

from typing import Dict, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml

from ..utils import attrdict
from ..preprocessing import data_astype

DATASET_NAMES = {
    'bivariate': 'Toy/Bivariate',
    'moons': 'Toy/Moons',
    'gmsc': 'GMSC (Give Me Some Credit)',
    'heloc': 'HELOC (Home Equity Line of Credit)',
    'lendingclub': 'Lending Club (2007-2018)',
    'wines': 'Wine Quality (White)',
    'wines_red': 'Wine Quality (Red)',
    'adult': 'Adult (Census Income)',
}


def get_dataset_name(
    key: str,
    datasets_names: Optional[Dict[str, str]] = None,
):
    """Returns the long name of the dataset.

    Args:
        key (str): The short name / key / handle of the dataset.
        datasets_names (Optional[Dict[str, str]], optional): A map for the long names of the datasets. Defaults to None.

    Returns:
        str: The long name of the dataset (if available, else the short name).
    """
    if datasets_names is not None and key in datasets_names:
        return datasets_names[key]
    elif key in DATASET_NAMES:
        return DATASET_NAMES[key]
    else:
        return key


def _default_names(n, prefix, suffix, start=0):
    return [f'{prefix} {i+start} {suffix}' for i in range(n)]


def _default_target_names(n, **kwargs):
    if n > 1:
        return _default_names(n, 'Target', **kwargs)
    else:
        return ['target']


def _default_feature_names(n, **kwargs):
    return _default_names(n, 'Feature', **kwargs)


# def _default_class_names(n, **kwargs):
#     return _default_names(n, 'Class', **kwargs)


def _infer_target_names(dt):

    # Get
    if 'target_names' in dt:
        targets = np.array(dt.target_names)
        # If dt.target is there, check that the names are consistent
        assert 'target' not in dt or not isinstance(dt.target, pd.DataFrame) or (len(
            dt.target.columns) == len(targets) and np.all(dt.target.columns.values == targets))
        # If dt.target is NOT there, then dt.data must be a DataFrame and the target names must be consistent
        assert 'target' in dt or (isinstance(dt.data, pd.DataFrame)
                                  and all([t in dt.data for t in targets])), 'Target names not found in data.'
        return targets
    elif 'target_name' in dt:
        # If dt.target is there (as a DataFrame or a Series), check that the names are consistent
        assert 'target' not in dt or not isinstance(dt.target, pd.DataFrame) or (
            len(dt.target.columns) == 1
            and dt.target.columns[0] == dt.target_name), ('target_name and target column name do not match.')
        assert 'target' not in dt or not isinstance(dt.target, pd.Series) or dt.target.name == dt.target_name, (
            'target_name and target series name do not match.')
        # If dt.target is NOT there, then dt.data must be a DataFrame and the target name must be consistent
        assert 'target' in dt or (isinstance(dt.data, pd.DataFrame)
                                  and dt.target_name in dt.data), 'Target name not found in data.'
        return np.array([dt.target_name])

    # Inference
    else:
        if 'target' in dt:
            if isinstance(dt.target, pd.DataFrame):
                return dt.target.columns.values
            elif isinstance(dt.target, pd.Series):
                return np.array(dt.target.name)
            else:  # np.ndarray
                warnings.warn('Target names(s) could not be inferred from the data. Using defaults.')
                n_targets = dt.target.shape[1] if len(dt.target.shape) > 1 else 1
                return _default_target_names(n_targets)
        elif 'feature_names' in dt:
            return np.array([c for c in dt.data.columns
                             if c not in dt.feature_names])  # This could return an empty array (no target)
        else:
            return np.array([])  # No target (all the data is constituted of features, e.g., autoencoders)


def _infer_feature_names(dt):

    # Get
    if 'feature_names' in dt:
        features = np.array(dt.feature_names)
        assert not isinstance(dt.data, pd.DataFrame) or all([f in dt.data.columns.values
                                                             for f in features]), 'Feature names not found in data.'
        return features

    # Inference
    if isinstance(dt.data, pd.DataFrame):
        if 'target_name' in dt:
            return np.array([c for c in dt.data.columns if c != dt.target_name])
        elif 'target_names' in dt:
            return np.array([c for c in dt.data.columns if c not in dt.target_names])
        else:
            return dt.data.columns.values  # No target
    else:  # np.ndarray
        warnings.warn('Feature names could not be inferred from the data. Using defaults.')
        return _default_feature_names(dt.data.shape[1])


def load_dataset(dataset='boston', task='classification', as_frame=False, return_X_y=False, **kwargs):
    """Load the dataset in a standard format.


    It returns an attrict with the following fields:
    ----
    - data (np.ndarray, pd.DataFrame): the dataset features. If as_frame=True, it is a DataFrame, a Numpy array otherwise.
    - target (np.ndarray, pd.DataFrame): the dataset target(s). OPTIONAL (e.g., unsupervised learning). If as_frame=True, it is a DataFrame, a NumPy array otherwise.
    - frame (pd.DataFrame): the dataset features and target(s) concatenated. Only returned when as_frame=True.
    ----
    - target_name (str): the name of the target column
    - class_names (List[str]): the names of the classes (if there is only one target)
    - target_names (List[str]): the names of the target labels
    ----
    - feature_names (List[str]): the names of the features
    ----
    - name (str): the name of the dataset (handle/key)
    - long_name (str): the long name of the dataset
    - task (str): the task of the dataset (classification, regression, etc.)

    Args:
        dataset (str, optional): Key/handle of the dataset. Defaults to 'boston'.
        task (str, optional): ML Task. Defaults to 'classification'.
        as_frame (bool, optional): Whether to return the data as a pandas DataFrame. Defaults to False.

    Returns:
        attrdict: Dataset in a standard format.


    NOTE: Imports are done locally to avoid unnecessary dependencies on unused modules.

    """

    if task == 'classification':
        if dataset == 'boston':
            dt = fetch_openml('boston', version=1, **kwargs)
            dt.target = 1 * (dt.target >= 21.2)
            dt.data = dt.data.astype(float)
        elif dataset == 'breastcancer':
            dt = load_breast_cancer(**kwargs)
        elif dataset == 'adult':
            from .preprocessing import load_adult
            dt = load_adult(**kwargs)
        elif dataset == 'lendingclub':
            from .preprocessing.lendingclub import load_lendingclub
            dt = load_lendingclub(**kwargs)
        elif dataset == 'heloc':
            from .preprocessing.heloc import load_heloc
            dt = load_heloc(**kwargs)
        elif dataset == 'gmsc':
            from .preprocessing import load_give_me_some_credit
            dt = load_give_me_some_credit(**kwargs)
        elif dataset == 'wines':
            from .preprocessing.winequality import load_winequality
            dt = load_winequality(type='white', **kwargs)
        elif dataset == 'wines_red':
            from .preprocessing.winequality import load_winequality
            dt = load_winequality(type='white', **kwargs)
        else:
            raise NotImplementedError('Not supported dataset so far.')
    else:
        raise NotImplementedError('Only classification supported so far.')

    # No dataset returned.
    if dt is None:
        raise FileNotFoundError('The dataset is supported but could not be loaded.')

    # Cast to attrdict
    dt = attrdict(dt)

    # Check that the dataset is valid.
    assert 'data' in dt, 'The dataset must have a "data" field (either a DataFrame or a Numpy array).'
    assert 'target_name' not in dt or 'target_names' not in dt, 'target_name and target_names cannot be both present in the dataset.'

    # target_names to class_names if required
    # Some people call classes targets, and viceversa. We want to standardize this.
    if 'target_names' in dt:
        if 'target' in dt and dt.target.ndim == 1 and len(dt.target_names) > 1:
            dt.class_names = dt.target_names
            del dt.target_names

    # Infer target_name(s)
    targets = _infer_target_names(dt)
    if len(targets) == 0:  # No target
        warnings.warn('Data has not target. Assuming all the data containts only features.')
    elif len(targets) == 1:  # Single target
        dt.target_name = targets[0]
    else:  # Multiple targets
        dt.target_names = targets

    # Infer feature_names
    dt.feature_names = _infer_feature_names(dt)

    # Split X and y
    if 'target' not in dt:
        if isinstance(dt.data, pd.DataFrame):
            if 'target_name' in dt:
                dt.target = dt.data[[dt.target_name]]
                dt.data = dt.data.drop(columns=[dt.target_name])
            else:  # target_names
                dt.target = dt.data[dt.target_names]
                dt.data = dt.data.drop(columns=dt.target_names)

    # Cast to Numpy
    dt.data = data_astype(dt.data, ret_type='pd' if as_frame else 'np', names=dt.feature_names, inplace=True)
    if 'target' in dt:
        dt.target = data_astype(dt.target, ret_type='pd' if as_frame else 'np', names=targets, inplace=True)

    # Return also full frame
    if as_frame:
        dt.frame = dt.data
        if 'target' in dt:
            dt.frame = pd.concat([dt.frame, dt.target], axis=1)

    if 'target' in dt:
        if dt.target.shape[1] == 1:
            if isinstance(dt.target, pd.DataFrame):
                dt.target = dt.target.iloc[:, 0]  # Return Series
            else:
                dt.target = dt.target[:, 0]  # Return 1D array

    # Add dataset name, task, etc.
    assert 'name' not in dt, 'Dataset name already present.'
    assert 'long_name' not in dt, 'Dataset long name already present.'
    assert 'task' not in dt, 'Dataset task already present.'

    dt.name = dataset
    dt.long_name = get_dataset_name(dataset)
    dt.task = task

    # Return only X and y, if requested.
    if return_X_y:
        if 'target' in dt:
            return dt.data, dt.target
        else:
            return dt.data

    return dt


load_toy_dataset = load_dataset