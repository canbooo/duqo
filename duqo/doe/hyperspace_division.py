# -*- coding: utf-8 -*-
"""
A heuristic estimation of the Fekete points. Fekete points are
acquired by solving the Fekete problem i.e. distributing points with minimal
energy on a surface

This seemed to perform better than other tested methods regarding the difference
between the minimum and the maximum (cosine and euclidean) distance between
the points on higher dimensional spaces (n > 10), which is used as a measure
of the uniformity. A very old code so caution is advised.

Created on Wed Dec 16 01:39:14 2015

@author: Bogoclu
"""

import functools

import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform


def scaled_standard_normal(n_dim, n_dir):
    P = norm.rvs(size=(n_dir, n_dim))
    return P / np.linalg.norm(P, axis=1).reshape((P.shape[0], 1))


def _skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


def _heuristic_fekete(n_dim, n_dir, max_iterations=100, tolerance=1e-18):
    """
    Calculates the fekete points heuristically using the
    zero centered repulsive power

    Parameters
    ----------
    n_dim : int 
        number of dimensions
    n_dir : int
        number of directions
    max_iterations : int
        maximum iterations for fekete points optimization
    tolerance : float
        tolerance for fekete points optimization

    Returns
    -------
    Fekete points:  np.ndarray with each row corresponding to
                    a point coordinate on unit sphere

    """
    alpha, alpha_cur = 2., 1.
    alpha_tol = 1e-12
    # base_points = adv_hs_div(n_dim, n_dir, golden=True)
    base_points = scaled_standard_normal(n_dim, n_dir)
    dist_min = np.min(pdist(base_points))
    print("start", dist_min)
    for i_iter in range(max_iterations):
        forces = pdist(base_points, "cosine")
        forces[forces < 5e-8] = 5e-8
        forces = squareform(forces ** (-i_iter / 2 - 1))
        forces = base_points[:, np.newaxis, :] * (forces[:, :, np.newaxis])
        inds = np.arange(n_dir)
        forces = forces.sum(0) - forces[inds, inds, :]
        # subtraction removes the diagonals from the distance matrix
        # which would have infinite magnitude
        amps = np.linalg.norm(forces, axis=1, keepdims=True)
        amps[amps < 5e-8] = 5e-8
        forces = forces / amps
        dist_min_tmp = 0
        alpha_cur = alpha * alpha_cur
        # above is reasonable since generally step size
        # should be decreasing  but we look for a better starting from
        # slightly larger value as the last one
        while dist_min_tmp <= dist_min and alpha_cur > alpha_tol:
            tmp_points = (1 + alpha_cur) * base_points - alpha_cur * forces
            tmp_points = tmp_points / np.linalg.norm(tmp_points, axis=1, keepdims=True)
            dist_min_tmp = np.min(pdist(tmp_points))
            alpha_cur = alpha_cur * .9

        if dist_min_tmp < dist_min:
            print('No more improvement could be achieved with Fekete points after', i_iter, "iterations.")
            break

        base_points = tmp_points.copy()
        if abs(dist_min_tmp - dist_min) < tolerance:
            print('Fekete points converged.', f" Change in min. dist: {abs(dist_min_tmp - dist_min)}")
            break
        dist_min = dist_min_tmp
    print("final", dist_min_tmp)
    return base_points


def comb(N, k):
    return np.math.factorial(N) / (np.math.factorial(N - k) * np.math.factorial(k))


@functools.lru_cache(maxsize=1)
def fekete_points(n_dim, n_dir, max_iters=500, tolerance=1e-18, n_try=3):
    """
    Parameters
    ----------
    n_dim  :  number of dimensions
    n_dir  :  number of directions
    max_iters  :  maximum iterations for fekete points optimization
    tolerance  :  tolerance for fekete points optimization

    Returns
    -------
    x - 2 -d numpy array
        Fekete points with each row corresponding to
        a point coordinate on unit sphere
    """
    if n_dim == 1:
        return np.array([-1, 1])
    if n_dim == 2:
        phi = np.linspace(0, 2 * np.pi, n_dir)
        return np.c_[np.cos(phi), np.sin(phi)]
    best_dist, best_dirs = 0, None
    for _ in range(n_try):
        cur_dir = _heuristic_fekete(n_dim, n_dir, max_iters, tolerance)
        cur_dist = np.min(pdist(cur_dir))
        if cur_dist > best_dist:
            best_dist = cur_dist
            best_dirs = cur_dir.copy()
    return best_dirs
