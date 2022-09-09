"""Funtions to:
- Generate samples of the surface of a cone.
- Project points over the decision boundary of a model.

TODO: This module needs refactoring. The functions about the model should be put in a separate module.
"""

from typing import Union

import numpy as np

from emutils.parallel.batch import batch_process
from emutils.utils import import_tqdm

from .utils import scaled_linspace

tqdm = import_tqdm()


def generate_cone(
    vertex,
    base_points,
    n,
    multiscaler,
    method,
    reversed_cone=0.0,
    axis='lines',
):
    lines = np.array([
        scaled_linspace(
            x=vertex + (vertex - x) * reversed_cone,
            y=x,
            num=n,
            scaler=multiscaler.get_transformer(method),
        ) for x in tqdm(base_points, desc='Ellipses Generation')
    ])

    # Transpose if necessary
    if axis == 'ellipses':
        lines = lines.swapaxes(1, 0)
    elif axis == 'lines':
        pass
    else:
        raise ValueError('Invalid main axis name.')

    return lines


def find_decision_boundary_between_two_points(x, y, model, multiscaler, num, norm_method, method='mean'):
    # Compute the predictions for the two points and check that they are different
    pred_x = model.predict(np.array([x]))
    pred_y = model.predict(np.array([y]))
    assert pred_x != pred_y

    # Generate the linspace between the points
    points = scaled_linspace(x, y, num=num, scaler=multiscaler.get_transformer(norm_method))

    # Compute the predictions for the linspace

    preds = 1 * (model.predict(points) != pred_x)

    # We get the index at which the prediction changes
    pred_change_i = np.searchsorted(preds, v=1, side='left')

    # The decision boundary is in between the change points
    if method == 'mean':
        return (points[pred_change_i] + points[pred_change_i - 1]) / 2
    elif method == 'left':
        return points[pred_change_i - 1]
    elif method == 'right':
        return points[pred_change_i]
    else:
        raise ValueError('Invalid method.')


def _find_decision_boundary_between_multiple_points(
    x,
    Y,
    predict_lambda,
    num,
    multiscaler,
    norm_method,
    error,
    method='mean',
    debug=False,
    cast=True,
    model_parallelism=1,
    # Y_already_normalized=False,
):
    """
        This function perform much better when num << 1,000,000
        beacause it batch together mutiple points when predicting
    """
    assert model_parallelism >= 1
    assert method in ['mean', 'left', 'right', 'counterfactual']

    x = np.asarray(x).copy()
    Y = np.asarray(Y).copy()
    X = np.tile(x, (Y.shape[0], 1))
    Yrange = np.arange(Y.shape[0])

    model_parallelism = int(np.ceil(model_parallelism / len(Y)))

    # Compute the predictions for the two points and check that they are different
    pred_x = predict_lambda(np.array([x]))
    if debug:
        pred_Y = predict_lambda(np.array(Y))
        assert np.all(pred_x != pred_Y)

    if multiscaler is not None:
        transformer = multiscaler.get_transformer(norm_method)
        if 'scale_' in dir(transformer):
            transformer = None
    else:
        transformer = None

    for _ in range(num):
        # Generate the linspace between the points
        # pointss = np.vstack(
        #     [scaled_linspace(x, y, num=num, scaler=multiscaler.get_transformer(norm_method)) for (x, y) in zip(X, Y)]
        # )

        # Compute the predictions for the linspace
        # predss = 1 * (predict_lambda(pointss) != pred_x)

        # # Reshape
        # pointss = pointss.reshape(Y.shape[0], num + 1, Y.shape[1])
        # predss = predss.reshape(Y.shape[0], num + 1)

        # # We get the index at which the prediction changes
        # for j, (preds, points) in enumerate(zip(predss, pointss)):
        #     pred_change_i = np.searchsorted(preds, v=1, side='left')
        #     Y[j] = points[pred_change_i]
        #     X[j] = points[pred_change_i - 1]
        if model_parallelism == 1:
            if transformer is None:
                M = (X + Y) / 2
            else:
                X_ = transformer.transform(X)
                Y_ = transformer.transform(Y)
                M_ = (X_ + Y_) / 2
                M = transformer.inverse_transform(M_)

            # Cast to proper type (e.g., if X and/or Y are integers)
            if cast is True:
                if M.dtype != X.dtype:
                    X = X.astype(M.dtype)
                if M.dtype != Y.dtype:
                    Y = Y.astype(M.dtype)

            preds = (predict_lambda(M) != pred_x)
            different_indexes = np.argwhere(preds)
            non_different_indexes = np.argwhere(~preds)

            Y[different_indexes] = M[different_indexes]
            X[non_different_indexes] = M[non_different_indexes]
        else:
            if transformer is None:
                M = np.concatenate(np.linspace(X, Y, model_parallelism + 1, endpoint=False)[1:])
            else:
                X_ = transformer.transform(X)
                Y_ = transformer.transform(Y)
                M_ = np.concatenate(np.linspace(X_, Y_, model_parallelism, endpoint=False))
                M = transformer.inverse_transform(M_)

            # Predict
            preds = (predict_lambda(M) != pred_x).reshape(model_parallelism, -1).T

            # Rebuild M
            M = np.concatenate(
                [np.expand_dims(X, axis=0),
                 M.reshape(model_parallelism, Y.shape[0], -1),
                 np.expand_dims(Y, axis=0)])

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


def find_decision_boundary_between_multiple_points(
    x,
    Y,
    num,
    model,
    n_jobs=1,
    mem_coeff=1,
    desc='Find Decision Boundary Ellipse (batch)',
    **kwargs,
):
    if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'inplace_predict') and hasattr(
            model, 'threshold'):
        booster = model.get_booster()
        predict_lambda = lambda X: 1 * (booster.inplace_predict(X) > model.threshold)
    else:
        predict_lambda = model.predict

    return batch_process(
        lambda Y: list(
            _find_decision_boundary_between_multiple_points(
                x=x,
                Y=Y,
                num=num,
                predict_lambda=predict_lambda,
                **kwargs,
            )),
        iterable=Y,
        # Optimal split size (cpu-mem threadoff)
        split_size=int(max(1, 100 * 1000 * 1000 / num / (x.shape[0]))) * mem_coeff,
        desc=desc,
        n_jobs=min(n_jobs, len(Y)),
    )


def generate_decision_boundary_cone_points(
    X: np.ndarray,
    vertex: np.ndarray,
    model,
    multiscaler,
    num: int = 100,
    num_search: int = 1000,
    norm_method_ellipses: Union[None, str] = 'mad',
    norm_method_search: Union[None, str] = None,
    axis='ellipses',
    reversed_cone: float = 0.0,
):
    """Generate the "ellipses" points of the sections of a cone
        that has as base an ellipses induced by projecting a set of points on the
        decision boundary.

    Args:
        X (np.ndarray): Points to be projected (size = n x m)
        vertex (np.ndarray): Vertex of the cone (size = m)
        model ([type]): model (must implement .predict)
        multiscaler ([type]): multiscaler object to multiscaler the data
        num (int, optional): Number of ellipses regions to generate. Defaults to 100.
        num_search (int, optional): Number of points to use to induce the decision boundary. Defaults to 10000.
        norm_method (str, optional): Nomrmalizations technique. Defaults to 'mad'.
        axis (str, optional): Axis on which to return the results. Default to 'ellipses'.
        reversed_cone (float, optional): Portion of the (reversed) cone after the vertex. Default to 0.
            If 1.0 the reversed cone will have the same height of the "non-reversed" part.

    Returns:
        np.ndarray : The points on the ellipses.
            (A) If axis = 'ellipses' (Default) then shape = (num + 1) x n x m
            Ellipse 0 -> vertex
            ...
            Ellipse num -> closest to the decision boundary
            (B) If axis = 'lines' then shape = n x (num + 1) x m
    """

    if axis not in ['ellipses', 'lines']:
        raise ValueError(f'Invalid value for axis ({axis}). Expected "ellipses" or "lines".')

    # Find the points sitting on the ellipses at the base of the cone lying in the vicinity of the decision boundary
    base_ellipses = find_decision_boundary_between_multiple_points(
        x=vertex,
        Y=X,
        model=model,
        multiscaler=multiscaler,
        num=num_search,
        method='left',
        norm_method=norm_method_search,
    )

    # Return the ellipses
    all_ellipses = generate_cone(
        vertex=vertex,
        base_points=base_ellipses,
        n=num,
        multiscaler=multiscaler,
        method=norm_method_ellipses,
        reversed_cone=reversed_cone,
        axis=axis,
    )

    return all_ellipses
