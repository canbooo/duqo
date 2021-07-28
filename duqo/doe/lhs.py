# -*- coding: utf-8 -*-
"""
Generate a design of experiments using Latin Hypercube or orthogonal sampling
i.e. with non-uniform marginal distributions with the desired correlation.


Created on Mon May  2 21:28:53 2016

@author: Bogoclu
"""


from copy import deepcopy
import warnings
import numpy as np
from scipy.special import comb as combine
from scipy.stats import uniform
from scipy.linalg import eigh, cholesky, inv, pinv
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", message="overflow encountered")


def uniform_lhs(lower_bounds, upper_bounds, num_sample, corr_mat=0,
                num_iter=100):
    """
    Creates a uniformly distributed LHS with central points. Implements the method proposed in [1]

    [1] D. Roos, LATIN HYPERCUBE SAMPLING BASED ON ADAPTIVE ORTHOGONAL DECOMPOSITION, ECCOMAS 2016

    Input
    -----
    lower_bounds : np.ndarray
        Lower bounds. Shape = (n_dim,)
    upper_bounds : np.ndarray
        Upper bounds. Shape = (n_dim,)
    num_sample : int
        Number of Samples
    corr_mat : float or 2-D np.ndarray
        Correlation matrix. If an array, it must be symmetrical with shape=(n_dim, n_dim). If scalar, the whole matrix
        except the diagonal will be filled with this value. Default is 0 meaning no correlation.
    num_iter : int
        The number of iterations. Default is 100.

    Returns
    -------
    fDoEPts : np.ndarray
        Optimized design of experiments matrix with the shape=(num_sample, n_dim)
    """
    num_var = lower_bounds.shape[0]
    # Sanity checks
    if not num_var == upper_bounds.shape[0]:
        err_msg = 'Lower bounds should have the same number of entries'
        err_msg += '\n as the upper bounds.'
        raise ValueError(err_msg)
    interval = (upper_bounds - lower_bounds)
    if (interval < 0).any():
        err_msg = 'Upper bounds should be greater than the lower bounds.'
        raise ValueError(err_msg)
    if np.isscalar(corr_mat):
        corr_mat = np.eye(num_var) * (1 - corr_mat) + np.ones((num_var, num_var)) * corr_mat
    if np.max(np.abs(corr_mat)) > 1:
        err_msg = 'Correlations should be in the interval [-1,1].'
        raise ValueError(err_msg)
    if not (corr_mat.shape[0] == corr_mat.shape[1] and corr_mat.shape[0] == num_var):
        err_msg = 'Inconsistent number of correlations and bounds.'
        raise ValueError(err_msg)
    if not np.isscalar(num_iter):
        raise ValueError('num_iter must be a scalar.')
    if num_iter < 1:
        num_iter = 1
    num_iter = int(np.round(num_iter))

    # Create start solution. Anything goes
    doe_curr = uniform.rvs(size=(num_sample, num_var))
    doe_curr = (np.argsort(doe_curr, axis=0) - 0.5) / num_sample
    old_score = np.inf
    #
    # array instead of switching just the first column
    for _ in range(num_iter):
        cur_rho = np.corrcoef(doe_curr, rowvar=False)
        cur_score = np.sum(np.abs(cur_rho - corr_mat), axis=0)
        # print(iIter,np.max(fScores))
        if np.abs(np.max(cur_score) - old_score) < 1e-16:
            break
        core_orders = np.argsort(cur_score)[::-1]
        ord_rho, ord_cur_rho = (np.zeros(cur_rho.shape), np.zeros(cur_rho.shape))
        for i_row, i_row_order in enumerate(core_orders):
            for i_col, i_col_order in enumerate(core_orders):
                ord_rho[i_row, i_col] = corr_mat[i_row_order, i_col_order]
                ord_cur_rho[i_row, i_col] = cur_rho[i_row_order, i_col_order]
        doe_curr = doe_curr[:, core_orders]
        doe_curr = (doe_curr - np.mean(doe_curr, axis=0)) / np.std(doe_curr, axis=0)
        try:
            chol_mat = cholesky(ord_rho, lower=False)
        except np.linalg.LinAlgError:
            eig_val, eig_vec = eigh(ord_rho)
            chol_mat = np.dot(np.diag(np.sqrt(eig_val)), eig_vec.T)

        try:
            chol_cur_mat = cholesky(ord_cur_rho, lower=False)
        except np.linalg.LinAlgError:
            eig_val, eig_vec = eigh(ord_cur_rho)
            chol_cur_mat = np.dot(np.diag(np.sqrt(eig_val)), eig_vec.T)
            chol_cur_mat = np.nan_to_num(chol_cur_mat)

        try:
            chol_cur_mat_inv = inv(chol_cur_mat)
        except np.linalg.LinAlgError:
            chol_cur_mat_inv = pinv(chol_cur_mat)
        doe_curr = np.dot(doe_curr, np.dot(chol_cur_mat_inv, chol_mat))
        doe_curr = np.argsort(np.argsort(doe_curr, axis=0), axis=0)
        doe_curr = (doe_curr + 0.5) / num_sample
        doe_curr = doe_curr[:, np.argsort(core_orders)]
        doe_curr = doe_curr * interval + lower_bounds
        old_score = np.max(cur_score)
    return doe_curr


def orthogonal_sampling(margs, num_sample: int, corr_mat=0, num_iter=100):
    """
    Creates an arbitrarily distributed LHS.

    This function creates a uniform LHS with UniformLHS-function
    and uses inverse transform sampling to convert the uniform LHS
    to an arbitrarily distributed one.

    Parameters
    ----------
    margs : list of distributions
        List of marginal distribution objects with .ppf method each object
        corresponds one random variable.

    num_sample : int
        Number of samples

    corr_mat : float or np.ndarray
        Correlation matrix. If an array, it must be symmetrical with shape=(n_dim, n_dim). If scalar, the whole matrix
        except the diagonal will be filled with this value. Default is 0 meaning no correlation.

    num_iter : int
        Number of iterations to compute the uniform LHS

    Returns
    -------
    fDoEPts : 2-D numpy.ndarray
        Optimized design of experiments matrix with the shape
        ```python
        (num_sample, len(margs))
        ```
    """

    def _calc_score(doe_curr, corrs, dist_max):
        dist_score = _calc_distscore(doe_curr, dist_max)
        corr_score = _calc_corrscore(doe_curr, corrs)
        return 5 * dist_score + corr_score

    num_var = len(margs)
    # Sanity checks
    if np.isscalar(corr_mat):
        corr_mat = np.eye(num_var) * (1 - corr_mat) + np.ones((num_var, num_var)) * corr_mat
    if not (corr_mat.shape[0] == corr_mat.shape[1] and corr_mat.shape[0] == num_var):
        msg = 'Inconsistent number of correlations and distibution'
        msg += '\n objects.'
        raise ValueError(msg)
    if num_iter < 0:
        msg = f"num_iter must be >= 0. Passed {num_iter}."
        raise ValueError(msg)
    num_iter = max(1, num_iter)
    best_probs = None
    best_score = np.inf
    log_d_max = np.log(np.sqrt(len(margs)))
    n_iter = int(np.sqrt(num_iter))
    for n_iter in range(n_iter):
        probs = uniform_lhs(np.zeros(num_var), np.ones(num_var), num_sample,
                            corr_mat=corr_mat, num_iter=n_iter)
        score = float(_calc_score(probs, corr_mat, log_d_max))
        if best_probs is None or score < best_score:
            best_probs = np.copy(probs)
            best_score = score

    doe = np.zeros(best_probs.shape)

    # Inverse transform sampling
    for i_var in range(num_var):
        doe[:, i_var] = margs[i_var].ppf(best_probs[:, i_var])
    return doe


def _switch_rows(doe_curr, column=None, col_row_pairs=()):
    """
    Randomly switches the values of a numpy array along the second axis
    This is the permutation function of OptimizeLHS.

    Parameters
    -----
    doe_curr : np.ndarray
        shape = (num_sample, n_dim)
    column : int
        The number of column, along which the switching is done. If
        not given, it will be chosen randomly.

    Returns
    -------
    doe_perturbed : np.ndarray
        perturbed DoE with shape (num_sample, num_dim)
        ```python

        ```
    """

    num_sample, num_var = doe_curr.shape
    max_combs_per_column = combine(num_sample, 2)
    max_combs_per_row = num_sample - 1
    if col_row_pairs:
        pairs = np.array(col_row_pairs, dtype=int)
    else:
        pairs = np.empty((0, 3), dtype=int)
    if column:
        cur_column = column
    else:
        uniques, col_counts = np.unique(pairs[:, 0], return_counts=True)
        uniques = uniques[col_counts >= max_combs_per_column].tolist()
        possible_cols = [i_col for i_col in np.arange(num_var) if i_col not in uniques]
        cur_column = np.random.choice(possible_cols)
    pairs = pairs[pairs[:, 0] == cur_column, 1:]
    fulls_1, row_counts = np.unique(pairs[:, 0], return_counts=True)
    fulls_1 = fulls_1[row_counts >= max_combs_per_row - fulls_1].tolist()
    row_inds = np.arange(num_sample - 1).tolist()
    possible_rows = [i_row for i_row in row_inds if i_row not in fulls_1]
    row_1 = np.random.choice(possible_rows)
    possible_rows = np.arange(row_1, num_sample)
    fulls_2 = pairs[pairs[:, 0] == row_1, 1]
    possible_rows = [i_row for i_row in possible_rows if i_row not in fulls_2]
    row_2 = np.random.choice(possible_rows)
    if row_1 > row_2:
        row_1, row_2 = row_2, row_1  # always same order
    doe_curr[row_1, cur_column], doe_curr[row_2, cur_column] = \
        doe_curr[row_2, cur_column], doe_curr[row_1, cur_column]
    return doe_curr, (cur_column, row_1, row_2)


def optimize_doe(doe_start, corr_mat=0, doe_old=None, num_tries: int = 10000,
                 decay: float = .95, max_steps: int = 20, sim_time: float = 25.,
                 verbose=0):
    """
    Optimizes a start LHS with simulated annealing to minimize
    the maximum correlation error.

    Inputs
    ------
    doe_start : numpy.ndarray
        used as the started solution with shape=(num_sample, num_vars)

    corr_mat : float or numpy.ndarray
        Correlation matrix. It must be symmetrical. If scalar, the whole matrix
        except the diagonal will be filled with this value. Default is 0 meaning
        no correlation.

    num_tries : int
        Maximum number of tries

    sim_time : float
        Time for the annealing algorithm.

    decay : float
        Step size for fTime for the annealing algorithm. Must be smaller than 1.

    max_steps : int
        Maximum number of steps for each time step


    Returns
    -------
    doe_final : numpy.ndarray
        Optimized design of experiments matrix with the shape=(num_sample, n_dim)
    """
    num_var = doe_start.shape[1]

    def _calc_score(doe_curr, corr_mat, dist_max, appender):
        dist_score = _calc_distscore(appender(doe_curr), dist_max)
        corr_score = _calc_corrscore(appender_loc(doe_curr), corr_mat)
        return 5 * dist_score + corr_score

    if doe_start.shape[0] == 1:
        return doe_start
    # Sanity checks
    if np.isscalar(corr_mat):
        corr_mat = np.eye(num_var) * (1 - corr_mat) + np.ones((num_var, num_var)) * corr_mat
    if not (corr_mat.shape[0] == corr_mat.shape[1] and corr_mat.shape[0] == num_var):
        msg = 'Inconsistent number of correlations and number of'
        msg += '\n variables in the LHS.'
        raise ValueError(msg)
    if num_tries < 1:
        num_tries = 1
    num_tries = np.round(num_tries)
    if sim_time < 1e-16:
        sim_time = 1.
    if decay <= 0 or decay >= 1:
        raise ValueError('fFac lie in the interval (0,1).')
    if max_steps < 1:
        max_steps = 1.
    if doe_old is None:
        appender = appender_loc = lambda x: x
    else:
        locs = [doe_start.min(0, keepdims=True), doe_start.max(0, keepdims=True)]
        locs = np.logical_and((doe_old >= locs[0]).all(1),
                              (doe_old <= locs[1]).all(1))
        appender_loc = lambda x: np.append(doe_old[locs].reshape((-1, x.shape[1])), x, axis=0)
        appender = lambda x: np.append(doe_old, x, axis=0)  # will be used for calculating score

    doe_final = deepcopy(doe_start)
    dist_max = np.max(appender(doe_start), axis=0) - np.min(appender(doe_start), axis=0)
    dist_max = np.log(np.sqrt(np.sum(dist_max ** 2)))
    best_score = _calc_score(doe_final, corr_mat, dist_max, appender)
    start_score = best_score
    max_cr_pair = doe_start.shape[1] * combine(doe_start.shape[0], 2)
    i_step = 0
    if verbose > 0:
        dist_score = _calc_distscore(appender(doe_final), dist_max)
        corr_score = _calc_corrscore(doe_final, corr_mat)
        print(f"Start loss - total: {best_score:.4f} dist: {5 * dist_score:.4f} corr: {corr_score:.4f}")
    cr_pairs = []
    old_cr_pairs = []
    for i_try in range(num_tries):
        doe_try, pair = _switch_rows(doe_start, col_row_pairs=cr_pairs)
        cr_pairs.append(pair)
        curr_score = _calc_score(doe_try, corr_mat, dist_max, appender)
        anneal_prob = 0.
        if sim_time > 1e-5:
            anneal_prob = np.exp(-(curr_score - start_score) / sim_time)

        if curr_score <= start_score or np.random.random() <= anneal_prob:  # pylint: disable=no-member
            doe_start = deepcopy(doe_try)
            old_cr_pairs = deepcopy(cr_pairs)
            cr_pairs = []
            start_score = curr_score
            i_step = 0
            sim_time *= decay
            if start_score < best_score:
                doe_final = deepcopy(doe_start)
                best_score = start_score
                if verbose > 1:
                    dist_score = _calc_distscore(appender(doe_final), dist_max)
                    corr_score = _calc_corrscore(doe_final, corr_mat)
                    print(f"{i_try + 1} - total: {best_score:.4f} dist: {5 * dist_score:.4f} corr: {corr_score:.4f}")
        i_step += 1
        if i_step >= max_steps:
            sim_time *= decay
            # Bound Randomness by setting back to best result
            # This may help convergence
            doe_start = deepcopy(doe_final)
            cr_pairs = deepcopy(old_cr_pairs)
            start_score = best_score
            i_step = 0
        if len(cr_pairs) >= max_cr_pair:
            # switch_twice = True
            break

    if verbose > 0:
        dist_score = _calc_distscore(appender(doe_final), dist_max)
        corr_score = _calc_corrscore(doe_final, corr_mat)
        print(f"Final loss - total: {best_score:.4f} dist: {5 * dist_score:.4f} corr: {corr_score:.4f}")
    return doe_final


def _calc_distscore(doe_cur, log_d_max):
    """ Calculate quality score used for DoE optimization"""
    obj_d = log_d_max - np.log(np.min(pdist(doe_cur)))
    return obj_d


def _calc_corrscore(doe_cur, corr_mat):
    d_cor = np.max(np.abs(np.corrcoef(doe_cur, rowvar=False) - corr_mat))
    if d_cor <= 1e-8:
        d_cor = 1e-8
    return np.log(d_cor)


def inherit_lhs(num_sample, empty_bins, bounds_l, bounds_u):
    """
    Add new samples to lhs

    Parameters
    ----------
    num_sample : int
        Number of samples

    empty_bins : np.ndarray
        Boolean mask of empty bins with shape=(n_bins, n_dims)

    bounds_l : np.ndarray
        Lower bounds with shape=(n_dims,)

    bounds_u : np.ndarray
        Upper bounds with shape=(n_dims,)

    Returns
    -------
    candidates: np.ndarray
        candidate new samples placed at empty bins with shape=(num_sample, n_dims)
    """
    num_bins, num_dims = empty_bins.shape
    v = (np.arange(num_bins) + 0.5) / num_bins  # get bin mids i.e. 0.5/num_sample, 1.5/num_sample...
    n_empty = empty_bins.sum(0)
    lb, ub = np.array(bounds_l), np.array(bounds_u)
    frees = np.empty((num_sample, num_dims))
    for i_dim in range(num_dims):
        cur_bins = v[empty_bins[:, i_dim]]
        n_diff = int(num_sample - n_empty[i_dim])
        while n_diff != 0:
            if n_diff > 0:
                extras = np.random.choice(v, size=n_diff, replace=False)
                cur_bins = np.append(cur_bins, extras)
                n_diff -= num_sample
            else:
                extras = np.random.choice(v, size=abs(n_diff), replace=False).tolist()
                cur_bins = np.array([c for i, c in enumerate(cur_bins) if i not in extras])
                n_diff = 0
        frees[:, i_dim] = cur_bins[np.random.permutation(num_sample)]
    return frees * (ub - lb) + lb


def make_doe(num_sample, margs=None, corr_mat=0, num_tries=None,
             lower_bound=None, upper_bound=None, verbose=0):
    """
    Makes an LHS with desired distributions and correlation

    Parameters
    ----------
    num_sample : int
        Number of samples

    margs : list
        List of marginal distribution objects with .ppf method. Each object
        corresponds one random variable.
    corr_mat : float or 2-D np.ndarray
        Correlation matrix. If an array, it must be symmetrical with shape=(n_dim, n_dim). If scalar, the whole matrix
        except the diagonal will be filled with this value. Default is 0 meaning no correlation.

    lower_bound : np.ndarray
        Lower bounds. Shape = (n_dim,)

    upper_bound : np.ndarray
        Upper bounds. Shape = (n_dim,)


    Returns
    -------
    doe_final : 2-D numpy.ndarray
        Optimized design of experiments matrix with the shape (num_sample, len(margs))

    """
    if margs is None:
        margs = [uniform(lb, ub - lb)
                 for lb, ub in zip(lower_bound, upper_bound)]
    if num_sample == 1:
        return np.array([marg.rvs(1) for marg in margs]).reshape((1, -1))
    if num_tries is None:
        if num_sample < 100:
            num_tries = 20000
        else:
            num_tries = 2000
    if margs is None and (lower_bound is None or upper_bound is None):
        raise ValueError("Either marginal distributions or bounds must be passed")

    if lower_bound is not None and upper_bound is not None:
        if np.any(lower_bound >= upper_bound):
            raise ValueError("Lower bound must be strictly smaller than the upper bound")
    num_dims = len(margs)
    if np.isscalar(corr_mat):
        corr_mat = np.eye(num_dims) * (1 - corr_mat) + np.ones((num_dims, num_dims)) * corr_mat
    n_iter = num_tries // 5
    doe_final = orthogonal_sampling(margs, num_sample, corr_mat, num_iter=n_iter)

    msg1 = ''
    if lower_bound is not None:
        for i_dim in range(num_dims):
            locs = doe_final[:, i_dim] < lower_bound[i_dim]
            num_inds = np.sum(locs)
            if num_inds > 0:
                if num_inds > 1:
                    if not msg1:
                        msg1 += 'Error in setting the lower bounds.\n'
                    msg1 += 'Please expand the lower bound ' + \
                            'for the dimension %d.\n' % i_dim
                else:
                    doe_final[locs, i_dim] = lower_bound[i_dim]
    msg2 = ''
    if upper_bound is not None:
        for i_dim in range(num_dims):
            locs = doe_final[:, i_dim] > upper_bound[i_dim]
            num_inds = np.sum(locs)
            if num_inds > 0:
                if num_inds > 1:
                    if not msg2:
                        msg2 += 'Error in setting the upper bounds.\n'
                    msg2 += 'Please expand the lower bound ' + \
                            'for the dimension %d.\n' % i_dim
                else:
                    doe_final[locs, i_dim] = upper_bound[i_dim]
    err_msg = msg1 + msg2
    if err_msg:
        raise ValueError(err_msg)
    doe_final = optimize_doe(doe_final, corr_mat, num_tries=num_tries, verbose=verbose)
    return doe_final


def find_empty_bins(doe, n_bins, lower_bound, upper_bound):
    """
    Find empty bins in an LHS
    Parameters
    ----------
    doe : np.ndarray
        Array containing samples with shape=(n_samples, n_dim)
    n_bins : in
        The number of bins in the LHS
    lower_bound : np.ndarray
        Lower bounds. Shape = (n_dim,)

    upper_bound : np.ndarray
        Upper bounds. Shape = (n_dim,)

    Returns
    -------
    empty_bins : np.ndarray
        Boolean mask of empty bins with shape=(n_bins, n_dims)
    """
    n_dims = len(lower_bound)
    lb, ub = np.array(lower_bound).ravel(), np.array(upper_bound).ravel()
    active_mask = np.logical_and((doe >= lb).all(1), (doe <= ub).all(1))
    empty_bins = np.ones((n_bins, n_dims), dtype=bool)
    probs = (doe[active_mask].reshape((-1, n_dims)) - lb) / (ub - lb)
    # probs = np.sort(probs, axis=0)
    edges = np.arange(n_bins + 1) / n_bins
    for i_bin in range(n_bins):
        condition = np.logical_and(probs >= edges[i_bin],
                                   probs <= edges[i_bin + 1])
        empty_bins[i_bin, :] = np.logical_not(condition.any(0))
    return empty_bins
