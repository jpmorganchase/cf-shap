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
from sklearn.datasets import load_boston, load_breast_cancer

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


def load_dataset(dataset='boston', task='classification', **kwargs):
    """Load the dataset in a standard format.


    It returns an attrict with the following fields:
    ----
    - data (pd.DataFrame): the dataset as a pandas DataFrame
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

    Returns:
        attrdict: Dataset in a standard format.


    NOTE: Imports are done locally to avoid dependencies when not necessary.

    """

    if task == 'classification':
        if dataset == 'boston':
            data = load_boston(**kwargs)
            data.target = 1 * (data.target >= 21.2)
        elif dataset == 'breastcancer':
            data = load_breast_cancer(**kwargs)
        elif dataset == 'adult':
            from .preprocessing import load_adult
            data = load_adult(**kwargs)
        elif dataset == 'lendingclub':
            from .preprocessing.lendingclub import load_lendingclub
            data = load_lendingclub(**kwargs)
        elif dataset == 'heloc':
            from .preprocessing.heloc import load_heloc
            data = load_heloc(**kwargs)
        elif dataset == 'gmsc':
            from .preprocessing import load_give_me_some_credit
            data = load_give_me_some_credit(**kwargs)
        elif dataset == 'wines':
            from .preprocessing.winequality import load_winequality
            data = load_winequality(type='white', **kwargs)
        elif dataset == 'wines_red':
            from .preprocessing.winequality import load_winequality
            data = load_winequality(type='white', **kwargs)

        else:
            raise NotImplementedError('Not supported dataset so far.')
    else:
        raise NotImplementedError('Only classification supported so far.')

    if data is None:
        raise FileNotFoundError('The dataset is supported but could not be loaded.')

    assert 'target_name' not in data or 'target_names' not in data, 'target_name and target_names cannot be both present in the dataset.'

    # Infer target_name or target_names
    if 'feature_names' not in data:
        if 'target_name' in data:
            data.feature_names = np.array([c for c in data.data.columns if c != data.target_name])
        elif 'target_names' in data:
            data.feature_names = np.array([c for c in data.data.columns if c not in data.target_names])
        else:
            warnings.warn('Could not infer feature names.')

    # Infer feature_names
    if 'target_name' not in data and 'target_names' not in data:
        if 'feature_names' in data:
            targets = np.array([c for c in data.data.columns if c not in data.feature_names])
            if len(targets) == 1:
                data.target_name = targets[0]
            else:
                data.target_names = targets
        else:
            warnings.warn('Could not infer target(s) name.')

    # Add dataset name, task, etc.
    assert 'name' not in data, 'Dataset name already present.'
    assert 'long_name' not in data, 'Dataset long name already present.'
    assert 'task' not in data, 'Dataset task already present.'

    data.name = dataset
    data.long_name = get_dataset_name(dataset)
    data.task = task

    return data


load_toy_dataset = load_dataset