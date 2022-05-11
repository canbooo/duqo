# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:05:41 2019

@author: Bogoclu
"""
import numpy as np
from scipy.stats import norm


class GenericIntegrator:
    """ Base class for an integrator

        This is used for inheritance only

    Parameters
    ----------
    multivariate:  pyRDO.uncertainty.model.Multivariate instance
        Defines the stochastic model to be used.

    constraints:  list
        list of constraint functions. Each must accept a 2-d numpy as input.


    const_args   :  list of additional arguments for the constraint
                    functions. each element of the list are omitted
                    to the constraints as in constraints[0](x,args).
                    Will not be passed if not provided as in
                    constraints[0](x) which is the default behaviour.

    std_norm_to_orig: None or function
            If not None, this will be called instead of the own function based
            on Nataf transformation for transforming the random variables to
            the standard normal space i. e. norm(0,1)


    The following only available after calling the calc_fail_prob method with
    the post_proc=True. Limited support for num_parallel > 2.

    Attributes
    ----------

    x_safe : numpy.ndarray
        points sampled in the safe domain

    x_fail : numpy.ndarray
        points sampled in the failure domain

    x_lsf : numpy.ndarray
        zero-crossing points on the limit state

    Look at the definition of calc_fail_prob in other integrators for further
    information
    """

    def __init__(self, multivariate, constraints, constraint_args=None,
                 std_norm_to_orig=None, orig_to_std_norm=None):
        self.mulvar = multivariate
        self.margs = multivariate.margs[:]  # make copy for faster access
        self._n_dim = _sanity_check_margs(self.mulvar.margs)
        if not isinstance(constraints, list):
            constraints = [constraints]
        self.constraints = constraints
        self._n_cons = len(self.constraints)
        self.const_args = _get_const_args(constraint_args, self._n_cons)
        if std_norm_to_orig is None:  # use our own
            self.corr_transform, self.icorr_transform = multivariate.nataf_mats()
            self.u2x = self._u2x
        else:
            self.u2x = std_norm_to_orig

        if orig_to_std_norm is None:  # use our own
            self.corr_transform, self.icorr_transform = multivariate.nataf_mats()
            self.x2u = self._x2u
        else:
            self.x2u = orig_to_std_norm

        self._post_proc = False
        self._n_parallel = 1

        self.x_lsf = np.empty((0, self._n_dim), dtype=np.float32)
        self.x_fail = np.empty((0, self._n_dim), dtype=np.float32)
        self.x_safe = np.empty((0, self._n_dim), dtype=np.float32)
        self.num_eval = 0

    def const_env(self, input_points):
        """
        Returns the lower envelope of all passed constraints i.e.
        min([constraints[i](x) for i in range(len(consmtraints])))

        Arguments
        ---------

        input_points  :  numpy array with the point coordinates. if 1-D, it
                       will be assumed a row array (one point)
        """

        if input_points.ndim < 2:
            input_points = input_points.reshape((1, -1))
        self.num_eval += input_points.shape[0]
        res = np.ones(input_points.shape[0]) * np.inf
        for i_con in range(self._n_cons):
            # if self.const_args[i_con]:
            cur = self.constraints[i_con](input_points,
                                          *self.const_args[i_con])
            # else:
            #     cur = self.constraints[i_con](input_points)
            cur = cur.reshape(res.shape)
            locs = res > cur
            res[locs] = cur[locs]
        if self._post_proc and self._n_parallel == 1:
            safe_inds = np.isclose(res, 0, atol=1e-5)
            if safe_inds.any():
                self.x_lsf = np.append(self.x_lsf,
                                       input_points[safe_inds, :], axis=0)
            safe_inds = res < 0
            if safe_inds.any():
                self.x_fail = np.append(self.x_fail,
                                        input_points[safe_inds, :], axis=0)
            safe_inds = res > 0
            if safe_inds.any():
                self.x_safe = np.append(self.x_safe,
                                        input_points[safe_inds, :], axis=0)
        return res

    # @property
    # def num_eval(self):
    #     if self._post_proc and self._n_parallel == 1:
    #         return self.x_lsf.shape[0] + self.x_fail.shape[0] + self.x_safe.shape[0]
    #     msg = "post_proc must be True and number of parallel computations "
    #     msg += "must be one to use the num_evals attribute. Returnin -1"
    #     print(msg)
    #     return -1

    def const_env_stdnorm(self, std_norm_input):
        """const_env in the standard normal space"""
        return self.const_env(self.u2x(std_norm_input))

    def _u2x(self, std_norm_input):
        """
        Transforms the standard normal variable u to the original
        space defined by the marginal distributions of x as passed
        in margs
        """
        corr_input = np.dot(std_norm_input, self.corr_transform)
        if corr_input.ndim < 2:  # assuming single dimensionals to be a single point
            corr_input = corr_input.reshape((1, self._n_dim))
        orig_input = np.zeros(corr_input.shape, dtype=np.float64)
        for k in range(self._n_dim):
            orig_input[:, k] = self.margs[k].ppf(norm._cdf(corr_input[:, k]))
        return orig_input

    def _x2u(self, input_points):
        """
        Transforms the standard normal variable u to the original
        space defined by the marginal distributions of x as passed
        in margs
        """
        ucorr_input = np.dot(input_points, self.icorr_transform)
        if ucorr_input.ndim < 2:  # assuming single dimensionals to be a single point
            ucorr_input = ucorr_input.reshape((1, self._n_dim))
        std_norm_input = np.zeros(ucorr_input.shape, dtype=np.float64)
        for k in range(self._n_dim):
            std_norm_input[:, k] = norm._ppf(self.margs[k].cdf(ucorr_input[:, k]))
        return std_norm_input


def to_safety_index(fail_prob):
    return -norm._ppf(fail_prob)


def _sanity_check_margs(margs):
    """
    Sanity check for the margs, returns number of entries in margs as n_dim
    """
    try:
        _ = iter(margs)
    except:
        err_msg = "margs must be an iterable with each entry "
        err_msg += "corresponding a marginal distribution of an "
        err_msg += "input dimension."
        raise ValueError(err_msg)
    n_dim = len(margs)
    if n_dim < 1:
        err_msg = "Number of dimensions cannot be smaller than 1 as "
        err_msg += "given in margs."
        raise ValueError(err_msg)
    return n_dim


def _get_const_args(const_args, n_const):
    """
    check and return const_args from the passed value to DirectionalSimulator
    """
    if const_args is None:
        const_args = [[]] * n_const
    try:
        const_args = list(const_args)
    except TypeError:
        const_args = [[const_args]] * n_const

    if len(const_args) != n_const:
        err_msg = f"Number of constraints ({n_const}) and the constraint arguments "
        err_msg += f"({len(const_args)}) do not match."
        raise ValueError(err_msg)
    return const_args
