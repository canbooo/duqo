# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:30:39 2016

WARNING: VERY OLD CODE BUT SEEMS TO WORK SO FAR

@author: Bogoclu
"""


from scipy import stats
from scipy.stats.distributions import rv_frozen
import numpy as np


def frozen_scipy_dist(dist) -> rv_frozen:
    """
    Create a frozen scipy distribution from a UniVar instance

    Inputs
    ------
    distribution: duqo.variables.UniVar
        A distribution object

    Returns
    -------
    distribution: rv_frozen
        a frozen scipy distribution
    """

    if dist.name.lower() == 'exponential' or dist.name.lower() == 'expon':
        return stats.expon(dist.mean - dist.std, dist.std)

    if dist.name.lower() == 'gumbel':
        scale = dist.std * np.sqrt(6) / np.pi
        loc = dist.mean - scale * np.euler_gamma
        return stats.gumbel_r(loc, scale)

    if dist.name.lower() == 'lognormal' or dist.name.lower() == 'lognorm':
        sigma = np.sqrt(np.log((dist.std / dist.mean) ** 2 + 1))
        logmean = np.log(dist.mean / np.sqrt((dist.std / dist.mean) ** 2 + 1))
        return stats.lognorm(sigma, 0, np.exp(logmean))

    if dist.name.lower() == 'normal' or dist.name.lower() == 'norm':
        return stats.norm(dist.mean, dist.std)

    if dist.name.lower() == 'uniform':
        args = (dist.lower_bound, dist.upper_bound - dist.lower_bound)
        return stats.uniform(*args)

    if dist.name.lower() == 'triangular':
        if not dist.params:
            consta = 0.5
            scale = np.sqrt(18 * (dist.std ** 2) / (consta ** 2 - consta + 1))
            loc = dist.mean - (consta + 1) * scale / 3
        else:
            mean_tmp = dist.mean
            mid_point = dist.params[0]
            aux_var = (-18 * (dist.std ** 2) + mid_point * (2 * mid_point - 3 * mean_tmp))
            aux_var = (9 * (mean_tmp ** 2) - 6 * mid_point * mean_tmp +
                       (mid_point ** 2) + aux_var) / 3
            aux_var = np.sqrt((9 * mean_tmp ** 2 - 6 * mid_point * mean_tmp
                               + mid_point ** 2) / 4 - aux_var)
            loc = (3 * mean_tmp - mid_point) / 2 + aux_var
            scale = 3 * mean_tmp - 2 * loc - mid_point
            if scale < 0:
                loc = (3 * mean_tmp - mid_point) / 2 - aux_var
                scale = 3 * mean_tmp - 2 * loc - mid_point
            consta = (mid_point - loc) / scale
        return stats.triang(consta, loc, scale)

    if dist.name.lower() == 'truncnormal' or dist.name.lower() == 'truncnorm':
        a = (dist.lower_bound - dist.mean) / dist.std
        b = (dist.upper_bound - dist.mean) / dist.std
        args = (a, b, dist.mean, dist.std)
        return stats.truncnorm(*args)
    if dist.name.lower() == 'bernoulli':
        if not dist.params:
            return stats.bernoulli(0.5)
        cond = np.isfinite(dist.params[0]) and dist.params[0] > 0
        if cond and dist.params[0] < 1:
            return stats.bernoulli(dist.params[0])
        raise ValueError("Distribution parameters are invalid for Bernoulli.")

    #######################################
    #           FRECHET AND WEIBULL missing among others
    #########################################
    msg = '%s distribution is not supported yet.' % dist.name
    #        warnings.warn(sWarnMsg)
    raise NotImplementedError(msg)
