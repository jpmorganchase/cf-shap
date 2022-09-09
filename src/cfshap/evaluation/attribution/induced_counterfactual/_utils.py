import logging
import time
import warnings
from copy import deepcopy

import numba
import numpy as np


__all__ = ['transform_trees', 'mask_values', 'inverse_actionabiltiy']

EPSILON = np.finfo(float).eps


def mask_values(Phi, Order, k):
    if np.isinf(k):
        k = Phi.shape[1]

    # Select the subset
    SubsetCols = Order[:, :k]
    SubsetRows = np.tile(np.arange(Phi.shape[0]), (SubsetCols.shape[1], 1)).T

    # Create a mask
    OrderMask = np.zeros(Phi.shape, dtype=np.int)
    OrderMask[SubsetRows, SubsetCols] = 1

    # Apply the mask
    PhiMasked = Phi * OrderMask

    return PhiMasked


def transform_trees(trees, multiscaler, method):
    ntrees = deepcopy(trees)
    for i in range(len(ntrees)):
        for j in range(len(ntrees[i].features)):
            if ntrees[i].children_left[j] != -1:  # A split (not a leaf)
                ntrees[i].thresholds[j] = multiscaler.value_transform(ntrees[i].thresholds[j], ntrees[i].features[j],
                                                                      method)
    return ntrees


def cost_trees(trees, ntrees, xn):
    ctrees = deepcopy(trees)
    for i in range(len(ctrees)):
        ctrees[i].thresholds_c = np.zeros(ctrees[i].thresholds.shape)
        for j in range(len(ctrees[i].thresholds_c)):
            if ctrees[i].children_left[j] != -1:
                ctrees[i].thresholds_c[j] = ntrees[i].thresholds[j] - xn[ctrees[i].features[j]]
    return ctrees


@numba.jit(nopython=True, nogil=True)
def filter_region(region, x, phi, tau, f, mc, Mc, m, M):
    # Get contraint
    t = tau[f]
    v = phi[f]

    if v == 0.0:
        # Must be 0-cost
        if (Mc < 0.0) or (mc > 0.0):
            return None
    else:
        # Must be trend/sign compatible (direction of the change to the feature)
        if (t > 0 and Mc <= 0.0) or (t < 0 and mc >= 0.0):
            return None

        # This is a super-rare error caused by np.float32 <> np.float64 conflicts
        # TreeEnsemble returns np.float32, but the dataset could be np.float64
        # The multiscaler then passed with a np.float32 (64) spits a np.float32 (64), it does not change type
        # This results in a wrong mc/Mc = thr_n - xn, i.e., sgn(thr_n - xn) != sgn(thr - x)
        if (t > 0 and M <= x[f]) or (t < 0 and m >= x[f]):
            return None

        if t > 0:
            mc = np.maximum(0.0, mc)
            m = np.maximum(x[f], m)
        if t < 0:
            Mc = np.minimum(0.0, Mc)
            M = np.minimum(x[f], M)

        assert mc < Mc and m < M

        # Precompute direction constraints
        if mc == 0.0 or Mc == 0.0:
            # We need this because inf * 0 = Nan (raise error)
            vm = 0.0
        elif mc * Mc > 0.0:
            vm = np.minimum(np.abs(mc), np.abs(Mc))
        else:
            vm = 0.0
        vM = np.maximum(np.abs(mc), np.abs(Mc))
        vm, vM = vm / v, vM / v

    for f_ in range(region.shape[0]):
        mc_, Mc_, m_, M_ = region[f_][0], region[f_][1], region[f_][2], region[f_][3]
        # Must be overlapping
        if f == f_:
            if M <= m_ or m >= M_:
                return None

        # Must be direction-compatible (proportion between features)
        v_ = phi[f_]
        if v != 0.0 and v_ != 0.0:
            if mc_ == 0.0 or Mc_ == 0.0:
                # We need this because inf * 0 = Nan (raise error)
                vm_ = 0.0
            elif mc_ * Mc_ > 0.0:
                vm_ = np.minimum(np.abs(mc_), np.abs(Mc_))
            else:
                vm_ = 0.0
            vM_ = np.maximum(np.abs(mc_), np.abs(Mc_))
            vm_, vM_ = vm_ / v_, vM_ / v_
            if (vM <= vm_ or vm >= vM_):
                return None

    nregion = region.copy()
    nregion[f][0] = np.maximum(region[f][0], mc)
    nregion[f][1] = np.minimum(region[f][1], Mc)
    nregion[f][2] = np.maximum(region[f][2], m)
    nregion[f][3] = np.minimum(region[f][3], M)
    return nregion


@numba.jit(nopython=True, nogil=True)
def filter_children_regions(tree_features, tree_children_left, tree_children_right, tree_thresholds, tree_thresholds_c,
                            region, x, phi, tau, n):

    # Non-compatible: return empty array
    if region is None:
        return np.zeros((0, x.shape[0], 4))

    # Recurse
    if tree_children_left[n] != -1:
        f = tree_features[n]
        l = tree_children_left[n]
        r = tree_children_right[n]
        t = tree_thresholds[n]
        tc = tree_thresholds_c[n]

        #         print('Region', region)

        regionl = filter_region(region, x, phi, tau, f, -np.inf, tc, -np.inf, t)
        regionr = filter_region(region, x, phi, tau, f, tc, np.inf, t, np.inf)

        #         print('Region L', regionl)
        #         print('Region R', regionr)

        return np.concatenate((filter_children_regions(tree_features, tree_children_left, tree_children_right,
                                                       tree_thresholds, tree_thresholds_c, regionl, x, phi, tau, l),
                               filter_children_regions(tree_features, tree_children_left, tree_children_right,
                                                       tree_thresholds, tree_thresholds_c, regionr, x, phi, tau, r)))

    # Recurse termination
    else:
        return np.expand_dims(region, 0)


@numba.jit(nopython=True, nogil=True)
def filter_tree_regions(tree_features, tree_children_left, tree_children_right, tree_thresholds, tree_thresholds_c,
                        regions, x, phi, tau):
    cregions = np.zeros((0, x.shape[0], 4))
    for region in regions:
        filtered_children = filter_children_regions(tree_features, tree_children_left, tree_children_right,
                                                    tree_thresholds, tree_thresholds_c, region, x, phi, tau, 0)
        cregions = np.concatenate((cregions, filtered_children))
    return cregions


def filter_regions(trees, x, phi, tau):
    regions = np.tile(np.array([-np.inf, np.inf, -np.inf, np.inf]), (1, x.shape[0], 1))
    for t, tree in enumerate(trees):
        regions = filter_tree_regions(tree.features, tree.children_left, tree.children_right, tree.thresholds,
                                      tree.thresholds_c, regions, x, phi, tau)
        # There must be at least a region
        assert len(regions) > 0
    regions = [{i: tuple(v) for i, v in enumerate(region)} for region in regions]
    return np.array(regions)


def assert_regions(regions):
    for region in regions:
        for f, (mc, Mc, m, M) in region.items():
            assert mc < Mc and m < M
    return regions


def cost_in_region(region, cost):
    for f, (mc, Mc, _, _) in region.items():
        if cost[f] > Mc or cost[f] < mc:
            return False
    return True


def point_in_region(region, point):
    for f, (_, _, m, M) in region.items():
        if point[f] > M or point[f] < m:
            return False
    return True


def find_region_candidates(region, x, xn, phi, tau, multiscaler, normalization, precision, bypass):
    X_costs = []

    #     print('>>>>>>>>>>>> REGION', region, '<<<<<<<<<<<<<<<<<<<<')

    # Compute candidates in the region
    for f, (mc, Mc, m, M) in region.items():
        #         print(f, mc, Mc, phi[f], end=' / ')
        if phi[f] == 0.0:  # => f is not a constraining variable
            #             print('NC phi=0.0')
            continue
        if Mc < 0:  # => decrease f
            delta_star = Mc - precision
            if delta_star < mc:
                delta_star = (mc + Mc) / 2
        elif mc > 0:  # => increase f
            delta_star = mc + precision
            if delta_star > Mc:
                delta_star = (mc + Mc) / 2
        elif mc <= 0 and Mc >= 0:  # keep f the same (not constraining)
            #             print('NC mc <= 0 and Mc >= 0')
            continue
        else:
            raise AssertionError('This should never happen.')

#         print(delta_star)
#         print('Dstar=', delta_star, 'Tau=', tau, '/ phi=', phi)
        X_costs.append(tau * phi / phi[f] * np.abs(delta_star))

    # Means that the regions is the current region (i.e., the region where x lies)
    if len(X_costs) == 0:
        return None, None

#     print('costs prev:', X_costs)

# Filter points outside the region (in terms of cost)
    X_costs = np.array([cost for cost in X_costs if cost_in_region(region, cost)])

    #     print('costs post 1:', X_costs)

    if len(X_costs) > 1:
        logging.debug(
            f'More than one direction-compatible point in the region (cost-space) with precision {precision}.')

    if len(X_costs) == 0:
        logging.debug(
            f'Less than one direction-compatible point in the region (cost-space) with precision {precision}.')
        return X_costs, X_costs  # Return empty (both X_p and X_costs)

    # Filter points outside the region (on the input space)
    X_p = multiscaler.inverse_transform(np.tile(xn, (X_costs.shape[0], 1)) + X_costs, method=normalization)
    X_p = np.where(X_costs == 0.0, np.tile(x, (X_costs.shape[0], 1)), X_p)

    if bypass is True:
        return X_p, X_costs


#     print('X_p post 1:', X_p)

    mask = np.array([point_in_region(region, x) for x in X_p])
    X_p = X_p[mask]
    X_costs = X_costs[mask]

    #     print('costs post 2:', X_costs)
    #     print('X_p post 2:', X_p)

    if len(X_costs) > 1:
        logging.debug(f'More than one direction-compatible point in the region (x-space) with precision {precision}.')

    if len(X_costs) == 0:
        logging.debug(f'Less than one direction-compatible point in the region (x-space) with precision {precision}.')

    return X_p, X_costs


def _find_point_in_region_with_min_cost(region, x, xn, phi, tau, multiscaler, normalization, precision, bypass):
    X_p = np.zeros((0, x.shape[0]))
    X_costs = np.zeros((0, x.shape[0]))

    # There are 0 or more than 1 candidates with this precision
    while len(X_costs) != 1 and precision > EPSILON:
        # print(f'Precision {precision}')
        X_p_, X_costs_ = find_region_candidates(region, x, xn, phi, tau, multiscaler, normalization, precision, bypass)
        precision = precision / 2

        # Means that the regions is the current regions
        if X_costs_ is None:
            return np.array([x]), np.zeros((1, x.shape[0]))

        # More than one with the same cost and its fine
        if len(X_costs_) == 0 and len(X_costs) > 1:
            break

        X_costs = X_costs_
        X_p = X_p_
    return X_p, X_costs


def find_point_in_region_with_min_cost(region, x, xn, phi, tau, degree, multiscaler, normalization, precision):
    X_p, X_costs = _find_point_in_region_with_min_cost(region,
                                                       x,
                                                       xn,
                                                       phi,
                                                       tau,
                                                       multiscaler,
                                                       normalization,
                                                       precision,
                                                       bypass=False)

    if len(X_costs) == 0:
        logging.warning(
            f'Less than one direction-compatible point in the region with precision {precision}/{EPSILON}: point will be (slightly) out of the region.'
        )
        X_p, X_costs = _find_point_in_region_with_min_cost(region,
                                                           x,
                                                           xn,
                                                           phi,
                                                           tau,
                                                           multiscaler,
                                                           normalization,
                                                           precision,
                                                           bypass=True)

    if len(X_costs) == 0:
        logging.error(
            f'No solution. Less than one direction-compatible point in the region with precision {precision}/{EPSILON}.'
        )
        raise RuntimeError()

    # Compute costs
    costs = np.linalg.norm(np.abs(np.array(X_costs)), degree, axis=1)
    index_min = np.argmin(costs)
    # costs_min = X_costs[index_min]
    cost_min = costs[index_min]
    x_min = X_p[index_min]
    # x_min = multiscaler.single_inverse_transform(xn + costs_min, method=normalization)
    # x_min = np.where(cost_min == 0.0, x, x_min)

    #     print('x/c', x_min, cost_min)

    # quantile_overflow handling: will return none when cost exceed 1.0 or less than 0.0
    if np.any(np.isnan(x_min)):
        return None, np.inf

    if not point_in_region(region, x_min):
        logging.warning(f"Point not in region: x = {x_min} / Region = {region}")

    return x_min, cost_min


def find_points_in_regions(regions, x, xn, phi, tau, degree, multiscaler, normalization, precision):
    X_prime, dist_prime = [], []
    for region in regions:
        x_min, cost_min = find_point_in_region_with_min_cost(region, x, xn, phi, tau, degree, multiscaler,
                                                             normalization, precision)
        if x_min is not None:
            X_prime.append(x_min)
            dist_prime.append(cost_min)
    return np.array(X_prime), np.array(dist_prime)


def inverse_actionabiltiy(
    x,
    xn,
    phi,
    tau,
    y_pred,
    model,
    multiscaler,
    trees,
    ntrees,
    degree,
    normalization,
    precision,
    error,
):

    x_counter, cost_counter = np.full(x.shape, np.nan), np.inf

    if np.any(np.isnan(phi)):
        if error == 'raise':
            raise RuntimeError(f'NaN feature attribution.')
        elif error == 'warning':
            warnings.warn(f'NaN feature attribution.')
        elif error == 'ignore':
            pass
        else:
            raise ValueError('Invalid error argument.')
        return x_counter, cost_counter

    # Compute costs in trees
    ctrees = cost_trees(trees, ntrees, xn)

    # Find regions (of the input space) that are phi-compatible
    regions = filter_regions(ctrees, x, phi=phi, tau=tau)
    assert_regions(regions)

    # Fint the regions counterfactual and cost
    X_prime, costs_prime = find_points_in_regions(
        regions,
        x,
        xn,
        phi,
        tau,
        degree=degree,
        multiscaler=multiscaler,
        normalization=normalization,
        precision=precision,
    )

    # Order the regions based on the cost
    costs_sort = np.argsort(costs_prime)
    X_prime, costs_prime = X_prime[costs_sort], costs_prime[costs_sort]

    # Generate the prediction points corresponding to regions of the input
    start = time.perf_counter()
    y_prime = model.predict(X_prime)
    logging.debug(f'Prediction of X_prime.shape = {X_prime.shape} took {time.perf_counter()-start}')

    # Find first occurence of change in output (with min cost)
    for xp, yp, cp in zip(X_prime, y_prime, costs_prime):
        if yp != y_pred:
            x_counter = xp
            cost_counter = cp
            break

    return x_counter, cost_counter
