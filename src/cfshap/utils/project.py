"""Functions to find the decision boundary of a classifier using bisection.
"""

__author__ = 'Emanuele Albini'
__all__ = [
    'find_decision_boundary_bisection',
]

import numpy as np

from ..base import Model, Scaler
from .parallel import batch_process


def find_decision_boundary_bisection(
    x: np.ndarray,
    Y: np.ndarray,
    model: Model,
    scaler: Scaler = None,
    num: int = 1000,
    n_jobs: int = 1,
    error: float = 1e-10,
    method: str = 'mean',
    mem_coeff: float = 1,
    model_parallelism: int = 1,
    desc='Find Decision Boundary Ellipse (batch)',
) -> np.ndarray:
    """Find the decision boundary between x and multiple points Y using bisection.

    Args:
        x (np.ndarray): A point.
        Y (np.ndarray): An array of points.
        model (Model): A model (that implements model.predict(X))
        scaler (Scaler, optional): A scaler. Defaults to None (no scaling).
        num (int, optional): The (maximum) number of bisection steps. Defaults to 1000.
        n_jobs (int, optional): The number of parallel jobs. Defaults to 1.
        error (float, optional): The early stopping error for the bisection. Defaults to 1e-10.
        method (str, optional): The method to find the point. Defaults to 'mean'.
            - 'left': On the left of the decision boundary (closer to x).
            - 'right': On the right of the decision boundary (closer to y).
            - 'mean': The mean of the two points (it can be either on the left of right side).
            - 'counterfactual': alias for 'right'.
        mem_coeff (float, optional): The coefficient for the job split (the higher the bigger every single job will be). Defaults to 1.
        model_parallelism (int, optional): Factor to enable parallel bisection. Defaults to 1.
        desc (str, optional): TQDM progress bar description. Defaults to 'Find Decision Boundary Ellipse (batch)'.

    Returns:
        np.ndarray: The points close to the decision boundary.
            
    NOTE: This function perform much better when num << 1,000,000 
    because it batches together multiple points when predicting
    (this is also done to avoid memory issues).
    """

    assert model_parallelism >= 1
    assert method in ['mean', 'left', 'right', 'counterfactual']

    if scaler is not None:
        assert hasattr(scaler, 'transform'), 'Scaler must have a `transform` method.'
        assert hasattr(scaler, 'inverse_transform'), 'Scaler must have a `inverse_transform` method.'

    def _find_decision_boundary_bisection(x: np.ndarray, Y: np.ndarray):

        x = np.asarray(x).copy()
        Y = np.asarray(Y).copy()
        X = np.tile(x, (Y.shape[0], 1))
        Yrange = np.arange(Y.shape[0])

        model_parallelism_ = int(np.ceil(model_parallelism / len(Y)))

        # Compute the predictions for the two points and check that they are different
        pred_x = model.predict(np.array([x]))
        pred_Y = model.predict(np.array(Y))
        assert np.all(pred_x != pred_Y), 'The predictions for x and Y are the same. They must be different.'

        for _ in range(num):
            # 'Pure' bisection
            if model_parallelism_ == 1:
                if scaler is None:
                    M = (X + Y) / 2
                else:
                    X_ = scaler.transform(X)
                    Y_ = scaler.transform(Y)
                    M_ = (X_ + Y_) / 2
                    M = scaler.inverse_transform(M_)

                # Cast to proper type (e.g., if X and/or Y are integers) with proper precision
                if M.dtype != X.dtype:
                    X = X.astype(M.dtype)
                if M.dtype != Y.dtype:
                    Y = Y.astype(M.dtype)

                preds = (model.predict(M) != pred_x)
                different_indexes = np.argwhere(preds)
                non_different_indexes = np.argwhere(~preds)

                Y[different_indexes] = M[different_indexes]
                X[non_different_indexes] = M[non_different_indexes]

            # Parallel bisection
            else:
                if scaler is None:
                    M = np.concatenate(np.linspace(X, Y, model_parallelism_ + 1, endpoint=False)[1:])
                else:
                    X_ = scaler.transform(X)
                    Y_ = scaler.transform(Y)
                    M_ = np.concatenate(np.linspace(X_, Y_, model_parallelism_, endpoint=False))
                    M = scaler.inverse_transform(M_)

                # Predict
                preds = (model.predict(M) != pred_x).reshape(model_parallelism_, -1).T

                # Rebuild M
                M = np.concatenate([
                    np.expand_dims(X, axis=0),
                    M.reshape(model_parallelism_, Y.shape[0], -1),
                    np.expand_dims(Y, axis=0)
                ])

                # Find next index
                left_index = np.array([np.searchsorted(preds[i], 1) for i in range(len(Y))])
                right_index = left_index + 1

                # Update left and right
                X = M[left_index, Yrange]
                Y = M[right_index, Yrange]

            # Early stopping
            err = np.max(np.linalg.norm((X - Y), axis=1, ord=2))
            if err < error:
                break

        # The decision boundary is in between the change points
        if method == 'mean':
            return (X + Y) / 2
        # Same predictions
        elif method == 'left':
            return X
        # Counterfactual
        elif method == 'right' or method == 'counterfactual':
            return Y
        else:
            raise ValueError('Invalid method.')

    return batch_process(
        lambda Y: list(_find_decision_boundary_bisection(x=x, Y=Y)),
        iterable=Y,
        # Optimal split size (cpu-mem threadoff)
        split_size=int(max(1, 100 * 1000 * 1000 / num / (x.shape[0]))) * mem_coeff,
        desc=desc,
        n_jobs=min(n_jobs, len(Y)),
    )