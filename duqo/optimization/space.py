# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:53:34 2019

@author: Bogoclu
"""
import typing

import numpy as np

from duqo.stoch.model import UniVar, MultiVar


def _sane_inds(ids, n_ele):
    if n_ele == 0:
        return []
    if ids is None:
        ids = np.arange(n_ele)
    ids = [int(i) for i in ids]
    if not ids:
        return []
    maxi = np.max(ids)
    if maxi >= n_ele:
        raise ValueError(f"Max. index {maxi} is larger than max. dim {n_ele}")
    return ids


def _make_bool(inds, n_ele):
    if isinstance(inds, np.ndarray) and inds.dtype == bool:
        return inds
    res = np.zeros(n_ele, dtype=bool)
    res[inds] = True
    return res


def _make_bool_d(dic, n_ele):
    for name in dic:
        dic[name] = _make_bool(dic[name], n_ele)
    return dic


class InputSpace:
    """
    Parameters
    ----------

    mulvar : MultiVar instance
        Joint probability distribution of the inputs

    num_inp : int
        total number of parameters (deterministic + stochastic) to
        pass to objectives and constraints. If not submitted, the number
        of variables in mulvar will be used.

    opt_inps : None or iterable
        The indices of the inputs used for the optimization inputs in the full
        space. If submitted, the missing indices will be kept constant during
        the optimization as their mean value since they are assumed to be
        stochastic, otherwise it will be assumed that all inputs are used for
        the optimization.

    stoch_inps : None or iterable
        The indices of the stochastic inputs in the full space. If submitted,
        the missing indices  will be kept constant during the stochastic
        assessments, otherwise it will be assumed that all inputs are
        stochastic.

    stoch_obj_inps : None or iterable
        The indices of the stochastic inputs in the full space. If not submitted,
        stoch_inps will be used. This provides granularity for which inputs
        to use for the stochastic evaluation of the objectives i. e. for
        estimating the mean and the variance of the objectives conditional on
        the inputs defined by these indexes.

    stoch_con_inps : None or iterable
        The indices of the stochastic inputs of the constraints in the full space.
        If not submitted, stoch_inps will be used. This provides finer
        granularity for which inputs to use for the stochastic evaluation
        of the constraints i.e. for estimating the failure probability
        contitional on the inputs defined by these indexes.

    """

    def __init__(self, mulvar: MultiVar, num_inp: int = None, opt_inps: list = None,
                 sto_inps: list = None, sto_obj_inps: list = None,
                 sto_con_inps: list = None):
        self.mulvar = mulvar
        if num_inp is None:
            num_inp = len(mulvar.dists)
        self.dims = num_inp
        if num_inp < 1:
            raise ValueError("At least one input is required")

        opt_inps = _sane_inds(opt_inps, num_inp)
        num_sto_inp = mulvar.mean.shape[0]
        stoch_inps = _sane_inds(sto_inps, num_sto_inp)

        if sto_obj_inps is None:
            sto_obj_inps = stoch_inps
        stoch_obj_inps = _sane_inds(sto_obj_inps, num_sto_inp)

        if sto_con_inps is None:
            sto_con_inps = stoch_inps
        stoch_con_inps = _sane_inds(sto_con_inps, num_sto_inp)
        # Make a guide to input space
        self.inds = {"opt": opt_inps,
                     "sto": list(set(stoch_obj_inps + stoch_con_inps)),
                     "sto_obj": stoch_obj_inps,
                     "sto_con": stoch_con_inps}
        self.inds = _make_bool_d(self.inds, num_inp)

        self._cov_i, self._covs = [], []
        for dist in self.mulvar.dists:
            if dist.var_coef is not None:
                self._covs.append(dist.var_coef)
                self._cov_i.append(True)
            else:
                self._covs.append(0.)  # So it raises an error, if one tries
                # to use it. Also makes it easier to slice
                self._cov_i.append(False)
        self._cov_i = _make_bool(self._cov_i, num_inp)
        self._covs = np.array(self._covs)

    def _get_bound(self, prob, opt_bound, quantile_fun):
        if opt_bound is None:
            return opt_bound
        model_bound = np.array(opt_bound, dtype=float)
        mulvar = self.opt_mulvar(model_bound, domain="sto")
        model_bound = self.opt2full(model_bound).ravel()
        model_bound[self.inds["sto"]] = getattr(mulvar, quantile_fun)(prob)
        if not np.isfinite(model_bound).all():
            raise ValueError(f"Resulting DoE bound from {quantile_fun} is not finite!")
        return model_bound

    def doe_bounds(self, prob, lower=None, upper=None):
        """
        Generates bounds for SML model given the deterministic bounds
        and a probability tolerance

        Parameters
        ----------
        lower : np.ndarray
            1-D array or None of lower optimization bound.
            If None, lower bound is not returned
        upper : np.ndarray
            1-D array or None of lower optimization bound.
            If None, lower bound is not returned
        prob_tol : float
            probability tolerance to use. For symmetric
            alpha-level confidence bounds,
            prob_tol == (1 - alpha) / 2

        Returns
        -------
        if both passed, lower and upper, else only the passed bound.
        if None passed, raises an error.

        """
        lower = self._get_bound(prob, lower, "lower_quantile")
        upper = self._get_bound(prob, upper, "upper_quantile")
        if lower is None and upper is None:
            raise ValueError("Either lower or upper bound must be passed!")
        if lower is None:
            return upper
        if upper is None:
            return lower
        return lower, upper

    def opt2full(self, x_opt):
        """A wrapper for stochastic assessments on user functions

        converts optimization inputs to full inputs by inserting
        the non optimization i.e. only stochastic inputs

        Parameters
        ----------
        x_opt : np.ndarray
            The array containing the sampling points in the optimization space with
            shape=(n_samples, len(self.inds['opt']))

        Returns
        -------
        x_full : np.ndarray with shape=(n_samples, n_inp)
            Full set of input points to be passed to objectives and constraints
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "opt2full")
        x_full = np.zeros((x_opt.shape[0], self.dims))
        # to get non opt stochastics
        x_full[:, self.inds["sto"]] = self.mulvar.mean
        x_full[:, self.inds["opt"]] = x_opt
        return x_full

    def opt2stoch(self, x_opt, domain: str = "sto"):
        """A wrapper to convert optimization to stochastic space

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape=(n_samples, len(self.inds['opt'])).

        domain : str
            one of sto_con, sto_opt, sto or opt. Defines the context
            of the transformation

        Returns
        -------
        x_stoch : numpy.ndarray
            Transformed points x_opt in the stochastic space
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "opt2stoch")
        return self.opt2full(x_opt)[:, self.inds[domain]]

    def stoch2full(self, x_stoch, x_opt, domain: str = "sto"):
        """A wrapper for stochastic assessments on user functions

        converts stochastic inputs to full inputs by inserting the references

        Parameters
        ----------
        x_stoch : np.ndarray
            The array containing the sampling points in the stochastic space with
            shape=(n_samples, len(self.inds['sto']))

        x_opt : np.ndarray
            The array containing the reference point in the optimization space with
            shape=(n_samples, len(self.inds['opt'])).

        domain : str
            one of sto_con, sto_opt, sto or opt. Defines the context
            of the transformation

        Returns
        -------
        x_full : numpy-array with n_samples x n_inp
            Full set of input points to be passed to objectives and constraints
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "stoch2full.x_opt")
        x_stoch = check_shape(x_stoch, self.inds[domain].sum(),
                              "opt_mulvar.x_stoch")
        x_full = self.opt2full(x_opt * np.ones((x_stoch.shape[0], x_opt.shape[1])))
        #        x_full = np.repeat(x_ref, x_stoch.shape[0], axis=0) ## too slow
        x_full[:, self.inds[domain]] = x_stoch
        return x_full

    def opt_moms(self, x_opt, domain: str = "sto_obj"):
        """Get the transformed mean and std. dev for x_opt

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape=(n_samples, len(self.inds['opt'])).

        domain : str
            one of sto_con, sto_opt, sto or opt. Defines the context
            of the transformation

        Returns
        -------
        new_mu : float
            Mean values of the inputs

        new_sigma : float
            Std. deviation of the inputs
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "opt_moms")
        mv_inds = self.mv_inds(domain=domain)
        new_mu = self.opt2stoch(x_opt, domain=domain).ravel()
        new_sigma = self.mulvar.std[mv_inds]
        cur_inds = self._cov_i[mv_inds]
        if cur_inds.any():
            cur_inds = np.logical_and(cur_inds, self.inds[domain][mv_inds])
            new_sigma[cur_inds] = self._covs[mv_inds][cur_inds] * new_mu[cur_inds]
        return new_mu, np.absolute(new_sigma)  # in case mu changes sign

    def opt_mulvar(self, x_opt, domain: str = "sto_con"):
        """return MultiVar with transformed mean and std. dev for x_opt

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape=(n_samples, len(self.inds['opt'])).

        domain : str
            one of sto_con, sto_opt, sto or opt. Defines the context
            of the transformation

        Returns
        -------
        mulvar : MultiVar
            Multivariate at the optimization point x_opt
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "opt_mulvar")
        mean, std = self.opt_moms(x_opt, domain)
        return self.mulvar.new(mean, std, self.mv_inds(domain))

    def sto_obj_doe(self, x_opt, num_points=100, num_iters=5000):
        """ Get a design of experiments at x_opt

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape=(n_samples, len(self.inds['opt'])).

        num_points : int
            number of points in the DoE

        num_iters : int
            number of iterations for the optimization of the DoE

        Returns
        -------
        doe - numpy.ndarray
            an optimized LHS for the stochastic objectives. Only the
            stochastic variables defined for the objectives i.e.
            sto_obj_inps are varied.
        """
        x_opt = check_shape(x_opt, self.inds["opt"].sum(), "sto_obj_doe")
        opt_mv = self.opt_mulvar(x_opt, "sto_obj")
        return opt_mv.opt_lhs(num_points=num_points, num_iters=num_iters)

    def sto_obj_base_doe(self, num_points=100, num_iters=5000):
        """ Get a design of experiments at x_opt

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape=(n_samples, len(self.inds['opt'])).

        num_points : int
            number of points in the DoE

        num_iters : int
            number of iterations for the optimization of the DoE

        Returns
        -------
        doe - numpy.ndarray
            an optimized LHS for the stochastic objectives. Only the
            stochastic variables defined for the objectives i.e.
            sto_obj_inps are varied.

        """
        mv_inds = self.mv_inds("sto_obj")
        n_univar = len(mv_inds)
        mu = np.zeros(n_univar)
        sig = np.ones(n_univar)
        dnames = self.mulvar.names
        dnames = [dnames[ind] for ind in mv_inds]
        inds = [i for i, x in enumerate(dnames) if "log" in x or "trunc" in x]
        # Handle Lognormal as uniform to use as probabilities
        scaled_mv = self.mulvar.new(mu, sig, mv_inds)
        for ind in inds:
            scaled_mv.dists[ind] = UniVar("uniform", lower_bound=0.0,
                                          upper_bound=1.0)
        return scaled_mv.opt_lhs(num_points=num_points, num_iters=num_iters)

    def mv_inds(self, domain="sto_obj"):
        """Gets the mv indexes of a given domain"""
        sto_inds = np.where(self.inds["sto"])[0].tolist()
        full_inds = np.where(self.inds[domain])[0].tolist()
        return [sto_inds.index(fui) for fui in full_inds]


class FullSpace():
    """Stochastic space of user functions

    Parameters
    ----------

    inp_space : MultiVar or InpSpace instance
        The definition of the input space

    num_obj : int
        total number of objective functions (deterministic + stochastic) to
        pass to objectives and constraints.

    num_con : int
        total number of constraint functions (deterministic + stochastic) to
        pass to objectives and constraints.

    obj_fun : function
        The objective function to evaluate. It should accept the input points,
        obj_args and the indices obj_inds of the objectives required as input
        i.e. obj_fun(x, obj_args, obj_inds) and return a numpy array
        with shape (n_points, len(obj_inds)) as output. The unintiutive
        formatting is for a better support of multi task models,
        that can approximate all objectives at once.

    con_fun : function
        The constraint function. Please read obj_fun for details.

    obj_args : any
        The arguments to pass to the objective function obj_fun

    con_args : any
        The arguments to pass to the constraint function con_fun

    sto_obj : None or list
        Indices of the stochastic objectives. If None, all objectives
        are assumed to be stochastic. This will be passed to the objective
        function, whenever a stochastic evaluation is conducted

    con_obj : None or list
        Indices of the stochastic constraints. If None, all constraints
        are assumed to be stochastic. This will be passed to the constraint
        function, whenever a stochastic evaluation is conducted

    """

    def __init__(self, inp_space: typing.Union[MultiVar, InputSpace], num_obj: int, num_con: int,
                 obj_fun=None, obj_arg=(), con_fun=None, con_arg=(),
                 sto_objs: list = None, sto_cons: list = None):
        if num_obj + num_con < 1:
            raise ValueError("At least one function is required")
        if num_obj > 0 and obj_fun is None:
            raise TypeError("obj_fun must be defined for num_obj > 0.")
        if num_con > 0 and con_fun is None:
            raise TypeError("con_fun must be defined for num_con > 0.")

        self.dims = (num_obj, num_con)
        if isinstance(inp_space, MultiVar):
            self.inp_space = InputSpace(inp_space)
        else:
            self.inp_space = inp_space
        self.obj = obj_fun
        self.con = con_fun
        if not obj_arg:
            obj_arg = tuple()
        if not con_arg:
            con_arg = tuple()
        self.obj_arg = obj_arg
        self.con_arg = con_arg
        stoch_obj_locs = _sane_inds(sto_objs, num_obj)
        stoch_con_locs = _sane_inds(sto_cons, num_con)
        self.obj_inds = {"full": [i for i in range(self.dims[0])],
                         "sto": stoch_obj_locs}
        self.con_inds = {"full": [i for i in range(self.dims[1])],
                         "sto": stoch_con_locs}

    def det_obj(self, x_opt):
        """ objective function for the deterministic assessment

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization with shape shape=(n_samples, len(self.inds['opt'])).
            The non-optimization variables are parsed from the MultiVar

        Returns
        -------
        obj_vals : numpy.ndarray
            the values of the objective functions with the shape
            (x_opt.shape[0], num_obj)
        """
        if self.dims[0] == 0:
            raise TypeError("This space does not have an objective function")
        res = self.obj(self.inp_space.opt2full(x_opt), *self.obj_arg,
                       self.obj_inds["full"])
        return check_shape(res, self.dims[0], parent="det_obj")

    def det_con(self, x_opt):
        """constraint function for the deterministic assessment

        Parameters
        ----------
        x_opt : numpy.ndarray
            the coordinates of the variables used in optimization. The non
            optimization variables are parsed from the MultiVar

        Returns
        -------
        con_vals : numpy.ndarray
            the values of the constraint functions with the shape
            (x_opt.shape[0], num_con)
        """
        if self.dims[1] == 0:
            raise TypeError("This space does not have a constraint function")
        res = self.con(self.inp_space.opt2full(x_opt), *self.con_arg,
                       self.con_inds["full"])
        return check_shape(res, self.dims[1], parent="det_con")

    def sto_obj(self, x_sto, x_opt):
        """ objective function for the stochastic assessment

        Parameters
        ----------
        x_sto : numpy.ndarray
            The values of the stochastic variables, as used for the objectives

        x_opt : numpy.ndarray
            the current coordinates of the variables used in optimization.

        Returns
        -------
        obj_vals : numpy.ndarray
            the values of the objective functions with the shape
            (x_opt.shape[0], num_stochastic_con)
        """
        if self.dims[0] == 0 or not self.obj_inds["sto"]:
            err = "This space does not have a stochastic objective function"
            raise TypeError(err)
        x_full = self.inp_space.stoch2full(x_sto, x_opt, domain="sto_obj")
        res = self.obj(x_full, *self.obj_arg, self.obj_inds["sto"])
        return check_shape(res, len(self.obj_inds["sto"]), parent="sto_obj")

    def sto_con(self, x_stoch, x_opt, envelope=True):
        """ constraint (limit state) function for the stochastic assessment """
        if self.dims[1] == 0 or not self.con_inds["sto"]:
            err = "This space does not have a stochastic constraint function"
            raise TypeError(err)
        x_full = self.inp_space.stoch2full(x_stoch, x_opt, domain="sto_con")
        res = self.con(x_full, *self.con_arg, self.con_inds["sto"])
        res = check_shape(res, len(self.con_inds["sto"]), parent="sto_con")
        if envelope:
            return np.min(res, axis=1, keepdims=True)
        return res


def check_shape(res, expected_dim, parent="somewhere"):
    res = np.array(res)
    if res.ndim == 1:
        res = res.reshape((-1, expected_dim))
    if res.shape[1] != expected_dim:
        msg = f"Dimension mismatch ({res.shape[1]}!={expected_dim}) "
        msg += "in " + parent
        raise ValueError(msg)
    return res
