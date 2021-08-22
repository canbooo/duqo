# -*- coding: utf-8 -*-
"""
Estimation of the probability of failure using the most probable point (MPP)
i.e. the closest point to the mean that lies in the region of failure.

A Multistart First Order Reliability Method (FORM) and Importance Sampling
Procedure Using Design point (ISPUD) are also implemented to evaluate the
probability of failure given MPP


Created on Tue Jul 30 02:06:47 2019

@author: Bogoclu
"""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from joblib import Parallel, delayed
from .integrator import GenericIntegrator, to_safety_index
from ..doe.lhs import make_doe

def _mpp_obj(std_norm_input):
    """Objective for MPP search """
    return np.sum(std_norm_input**2)

def _mpp_jac(std_norm_input, *args):
    """Objective derivative for MPP search """
    return 2*std_norm_input

def _call_opt(limit_state, x_start, cons, method='SLSQP', bounds=None):
    """ calls scipy optimizer """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bounds is None:
            res = minimize(_mpp_obj, x_start, jac=_mpp_jac, method=method, 
                           constraints=cons)
        else:
            res = minimize(_mpp_obj, x_start, jac=_mpp_jac, method=method,
                           constraints=cons, bounds=bounds)
    return res

def _get_mpp(limit_state, x_start: np.ndarray, bounds=None, give_vals: bool = True):
    """ Get MPP using SLSQP and if that fails, slower cobyla"""
    def neg_limit_state(x):
        return -limit_state(x)
    
    cons = ({'type': 'eq', 'fun': limit_state})
    try:
        res = _call_opt(limit_state, x_start, cons, bounds=bounds)
        success = res.success
    except ValueError:
        success = False
    if not success or res.get("status") in [5,6]:
        cons = ({'type': 'ineq', 'fun': limit_state},
                {'type': 'ineq', 'fun': neg_limit_state})
        res = _call_opt(limit_state, x_start, cons, method="COBYLA")
    res_input = res.get("x")
    if give_vals:
        return res_input.ravel(), _mpp_obj(res_input), limit_state(res_input)
    return res_input.ravel()


def most_probable_failure_point(const_env_stdnorm, num_dim: int,
                                num_starts: int = 12, num_parallel: int = 2):
    """Compute the most probable failure point MPP

    Parameters:
    ----------
        const_env_stdnorm : instance of any class from integrate Module
            used for computing the lower envelope of the constraints in the
            standard normal space

        num_dim : int
            number of input dimensions in the stochastic space

        num_starts : int
            number of starts for finding MPP

        num_parallel : int
            number of parallel starts for finding MPP. Only used
            if num_starts > 1.

    Returns:
    --------
        mpp : numpy.ndarray
            Most probable point of failure in the standard normal space
    """
    num_tolerance = 1e-10

    lim = 8 # because of scipy.stats precision, even this results in errors

    bounds = [(-lim, lim) for k in range(num_dim)]
    x_starts = np.zeros((1, num_dim))
    if num_starts <= 1:
        x_best = _get_mpp(const_env_stdnorm, x_starts,
                          bounds=bounds, give_vals=False)

    else:
        margs = [stats.uniform(-1, 2) for _ in range(num_dim)]
        x_starts = np.append(x_starts, make_doe(num_starts - 1, margs,
                                                num_tries=1), axis=0)
        x_best = None
        obj_best = np.inf
        if num_parallel == 1:
            for x_start in x_starts:
                x_cur, obj, con = _get_mpp(const_env_stdnorm, x_start, bounds=bounds)
                if obj < obj_best and abs(con) < num_tolerance:                    
                    x_best = np.copy(x_cur)
                    print(x_best)
                    obj_best = obj
        else:
            with Parallel(n_jobs=num_parallel, backend="loky") as para:
                x_all = para(delayed(_get_mpp)(const_env_stdnorm,
                                               x_starts[[i_st]], bounds=bounds,
                                               give_vals=False)
                             for i_st in range(x_starts.shape[0]))
                x_all = np.squeeze(np.array([x.ravel() for x in x_all]))
                if x_all.ndim < 2:
                    # must be one dimension since there were more than 1 starts
                    x_all = x_all.reshape((num_starts, 1))
                x_best = x_all[[np.argmin(np.sum(x_all**2, axis=1))], :]

    return x_best


class FORM(GenericIntegrator):
    """First order reliability method

        Assumes P(F) = phi_inv(mpp) where phi_inv is the inverse of the
        standard normal distribution and mpp is in the standard normal space.
        Thus this is a linear estimation, which is inacurrate for multimodal
        limit state function/constraints.
        
        
    """

    def integrate(self, num_parallel: int = 2, post_processing: int = True, probability_tolerance: bool = 1e-8,
                  **kwargs):
        """ Calculate the failure probability based on FORM on the most
        probable point of failure (MPP).
        
        Parameters
        ----------
        num_parallel: int
            number of parallel starts
        probability_tolerance : bool
            If true, sampling points will be accumulated to the attributes
            x_lsf, x_safe and x_fail and also will return mpp, conv_mu, conv_var,
            conv_x
            
        Note that the estimation variance is always returned as -1 but is not 
        removed for conformity with other pipeline objects
        
        Returns
        -------
        fail_prob_mu : float
            estimation of the expected probability of failure

        fail_prob_var : float
            estimation variance of the probability of failure

        Following are only retuned if post_proc = True

        safety_index : float
            Safety index, also known as the sigma level. It is equal to
            Phi_inv(1-fail_prob_mu), where Phi_inv is the inverse of the CDF
            of standard normal distribution

        mpp : 2-D numpy.ndarray
            Most probable point of failure among the used samples. It may
            slightly differ if calculated with optimization directly since
            no additional samples are generated to find it. If you need this
            use the mpp module
        """
        self._post_proc = probability_tolerance
        self._n_parallel = num_parallel
        mpp = most_probable_failure_point(self.const_env_stdnorm, self._n_dim,
                                          num_starts, num_parallel=num_parallel)
        if mpp is None:
            return 0.0, None, np.inf, None
        safety_index = np.linalg.norm(mpp)
        fail_prob_mu = stats.norm._cdf(-safety_index)
        if np.isnan(fail_prob_mu):
            return 0.0, -1, np.inf, None
        return fail_prob_mu, -1, safety_index, self.u2x(mpp)








class ISPUD(GenericIntegrator):
    """Importance Sampling Procedure Using Design point

     Importance sampling uses an auxilary distribution q* to estimate the
     integral with lower variance compared to MC. ISPUD transforms the space
     to the standard normal and uses MPP as the mean of q*, which is estimated
     as a normal distribution with unit variance.
     
     
    """


    def integrate(self, num_parallel: int = 2, post_processing: int = True, probability_tolerance: int = 1e-8,
                  **kwargs: bool):
        """ Calculate the failure probability based on ISPUD using MPP.
        
        Parameters
        ----------
        probability_tolerance: int
            number of parallel starts
        
        
        Returns
        -------
        fail_prob_mu : float
            estimation of the expected probability of failure

        fail_prob_var : float
            estimation variance of the probability of failure

        Following are only retuned if post_proc = True

        safety_index : float
            Safety index, also known as the sigma level. It is equal to
            Phi_inv(1-fail_prob_mu), where Phi_inv is the inverse of the CDF
            of standard normal distribution

        mpp : 2-D numpy.ndarray
            Most probable point of failure among the used samples. It may
            slightly differ if calculated with optimization directly since
            no additional samples are generated to find it. If you need this
            use the mpp module
        
        
        """
        self._post_proc = post_proc
        self._n_parallel = probability_tolerance
        mpp = most_probable_failure_point(self.const_env_stdnorm, self._n_dim,
                                          num_starts, num_parallel=probability_tolerance)
        if mpp is None:
            return 0.0, None, np.inf, None      
        mpp = mpp.ravel()
        aux_margs = [stats.norm(x_i, 1.) for x_i in mpp]
        if doe is None:
            aux_doe = make_doe(num_samples, aux_margs, num_tries=1000)
        else:
            aux_doe = doe + mpp
        fails = np.array(self.const_env_stdnorm(aux_doe) < 0, dtype=np.float64)
        weights = np.prod(stats.norm._pdf(aux_doe), axis=1)
        denom = np.zeros(aux_doe.shape)
        for i_dim in range(self._n_dim):
            denom[:, i_dim] = stats.norm(mpp[i_dim], 1.).pdf(aux_doe[:, i_dim])
        weights /= np.prod(denom, axis=1)
        fails *= weights
        fail_prob_mu = np.mean(fails)
        fail_prob_var = np.var(fails, ddof=1)
        safety_index = to_safety_index(fail_prob_mu)
        return fail_prob_mu, fail_prob_var, safety_index, self.u2x(mpp)
