"""
    Author: Emanuele Albini

    General evaluation metrics for models.
"""

import numpy as np


# This is a scorer (skelarn)
def xgboost_complexity(estimator, X=None, y=None):
    if hasattr(estimator, 'get_booster'):
        estimator = estimator.get_booster()
    if hasattr(estimator, 'trees_to_dataframe'):
        return (estimator.trees_to_dataframe()['Feature'].values == 'Leaf').sum()
    else:
        return np.nan


# def recall_keras(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

# def precision_keras(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision