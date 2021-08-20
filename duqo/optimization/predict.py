# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:33:47 2019

@author: Bogoclu
"""
import typing
import multiprocessing as mp
import warnings

import numpy as np
from scipy import stats

from .space import FullSpace
from duqo.proba import DS, MC
from duqo.doe.lhs import make_doe


def _check_obj_wgt(obj_weights, num_obj):
    """ Check obj_wgt argument passed to CondMom """
    if obj_weights is None:
        return None
    try:
        _ = obj_weights[0]
    except (TypeError, IndexError):
        obj_weights = np.ones(num_obj) * obj_weights
    if len(obj_weights) != num_obj:
        msg = f"Mismatch between the number of entries ({len(obj_weights)} in "
        msg += f"obj_wgt and the number of stochastic objectives ({num_obj})."
        raise ValueError(msg)
    return np.array(obj_weights).ravel()


def _check_std_inds(use_std, num_obj):
    """ Check use_std argument passed to CondMom and
        convert it to a slice definition
    """
    if isinstance(use_std, bool):
        inds = [use_std] * num_obj
    if len(inds) != num_obj:
        msg = "Mismatch between the number of entries in "
        msg += "use_std and the number of stochastic objectives."
        raise ValueError(msg)
    return np.array(use_std, dtype=bool)


def _find_integrator_cls(integrator):
    """
        Find the Integrator class as defined by the string integrator
    """
    integrator = integrator.upper()
    if integrator == "DS":
        return DS
    elif integrator == "MC":
        return MC
    msg = f"Requested integrator {integrator} is not found."
    raise ValueError(msg)


def _make_chain(methods: list):
    """Makes the chain given a list of method names"""
    try:
        first = methods[0]
    except TypeError:
        raise TypeError(f"methods must be a list of strings or classes, not {type(methods)}")
    try:
        _ = first.upper()
    except AttributeError:
        return methods

    return [_find_integrator_cls(name.upper()) for name in methods]


def _n_para_chk(num_parallel: int = None):
    """ Check the num_parallel argument as passed to CondProb """
    n_procs = max(1, mp.cpu_count())  # could cpu_count ever be < 1?
    if num_parallel is None or num_parallel > n_procs:
        print(f"Number of parallel processes was set to {n_procs}")
        return n_procs
    return num_parallel


def _default_init(targ_prob: float, acc_max: float, num_inp: int,
                  num_para: int):
    """Decide the default integrator chain methods and arguments depending
    on the problem

    Parameters
    ----------
    targ_prob : float
        target failure probability
    acc_max : float
        target tolerance for the estimation
    num_inp : int
        number of stochastic inputs of the constraints
    num_para : int
        number of parallel processes to use

    Returns
    -------
    integrators : list
        Integrator classes, that are to be initiated
    int_args : dict
        Keyword arguments to pass to integrators
    """
    if num_inp < 15:
        integrators = ["DS"]
        int_args = {"num_starts": 1, "multi_region": True}
    else:
        integrators = ["MC"]
        int_args = {"num_starts": 1, "batch_size": 1e5}

    print("Using", integrators, "as default chain.")
    return integrators, int_args


def _is_worker(workers, name):
    """ check if name is in workers list of classes"""
    for worker in workers:
        wname = read_integrator_name(worker)
        if name.upper() in wname.upper():
            return True
    return False


def read_integrator_name(worker):
    """ read the name of the integrator instance worker """
    name = str(worker).split(".")[-1]
    return "".join([c for c in name if c.isalnum()])


class CondMom:
    """Class to estimate conditional means

    full_space : FullSpace instance
        The definition of the optimization and stochastic spaces

    base_doe : int or np.ndarray
        set if a new doe should be calculated or the same one should
        be transformed during the optimization.
        if array, it should  have zero mean and unit variance
        but the original marginal distributions and correlation.
        it should have same number of columns as stochastic variables
        used in the objective. If integer, a base_doe with that number of
        samples will be created

    doe_size : int
        The size of the doe to use. If base_doe is a numpy array, this
        has no effect and doesn't have to be passed.

    obj_wgt : float or iterable of floats:
        If not None, these weights will be used for combining the
        estimated mean and the variance/std. dev. If iterable, it
        must be the same length as the number of stochastic input
        variables as used for the objective function.
        If None, the variances are returned separetly

    use_std : bool or iterable of bools
        Flag to use standard deviation (True) or the variance for the
        estimation. If iterable, it must be the same length as the number
        of stochastic input variables as used for the objective function.



    """

    def __init__(self, full_space: FullSpace, base_doe: typing.Union[bool, np.ndarray] = True,
                 doe_size: int = 100, obj_wgt: typing.Optional[typing.Union[float, list, np.ndarray]] = None,
                 use_std: typing.Union[bool, list] = False):
        self.full_space = full_space
        num_obj = len(self.full_space.obj_inds["sto"])
        self._use_std = _check_std_inds(use_std, num_obj)
        self._obj_wgt = _check_obj_wgt(obj_wgt, num_obj)
        self._doe_size = None
        self._base_doe = None
        self.doe_size = doe_size
        self.base_doe = base_doe

    @property
    def base_doe(self):
        """Base doe to use for the moment estimation
        
        Don't set this to an array with truncnorm and lognormal distributions
        in the MultiVariate if you don't know exactly what you are doing.
        """
        return self._base_doe

    @base_doe.setter
    def base_doe(self, new_doe):
        """Base doe to use for the moment estimation
        
        Don't set this to an array with truncnorm and lognormal distributions
        in the MultiVariate if you don't know exactly what you are doing.
        """
        # Sanity checks for base_doe. Using parameters with multiple valid types
        # may be an antipattern but it makes configuration easier from
        # the user point of view. Tolerate this for a better user experience.
        if isinstance(new_doe, np.ndarray):
            if self._is_valid_base(new_doe):  # raises errors
                self._base_doe = new_doe.copy()  # Make our copy.
            return
        try:
            make_base_doe = bool(new_doe)
        except ValueError:
            return

        if make_base_doe:
            # Prepare doe with zero mean and unit variance
            doe = self.full_space.inp_space.sto_obj_base_doe(self.doe_size)
            self._base_doe = doe
            return
        # if not bool(new_doe); remake new doe so set base_doe to None
        self._base_doe = None
        return

    def _is_valid_base(self, new_doe):
        # Assume numpy array
        n_sto_obj_inps = len(self.full_space.inp_space.inds["sto_obj"])
        if new_doe.shape[1] != n_sto_obj_inps:
            msg = "base_doe must be one of None, bool or a 2d array "
            msg += f"with shape (num_samples, num_stochastic_objective_variables={n_sto_obj_inps})."
            raise TypeError(msg)
        if max(abs(new_doe.mean(0).max()), abs(1 - new_doe.std(0).max())) > 0.5:
            msg = "base_doe must have zero mean and unit variance."
            raise ValueError(msg)
        return True

    @property
    def doe_size(self):
        """Size of the base doe to use for the moment estimation"""
        return self._doe_size

    @doe_size.setter
    def doe_size(self, new_size):
        """Size of the base doe to use for the moment estimation"""
        self._doe_size = new_size
        if self.base_doe is not None:
            self.base_doe = new_size

    @property
    def obj_wgt(self):
        """Weights for the linear combination of cond. moments"""
        return self._obj_wgt

    @obj_wgt.setter
    def obj_wgt(self, new_obj_wgt):
        """Weights for the linear combination of cond. moments"""
        n_obj = len(self.full_space.obj_inds["sto"])
        self._obj_wgt = _check_obj_wgt(new_obj_wgt, n_obj)

    @property
    def use_std(self):
        """Indexes to use std. dev. instead of variance"""
        return self._use_std

    @use_std.setter
    def use_std(self, new_std):
        """Indexes to use std. dev. instead of variance"""
        n_obj = len(self.full_space.obj_inds["sto"])
        self._use_std = _check_std_inds(new_std, n_obj)

    def gen_doe(self, x_opt):
        """Get DoE for the Moment estimation for x_opt"""
        if x_opt.ndim == 1:
            x_opt = x_opt.reshape((1, -1))
        if self.base_doe is None:
            return self.full_space.inp_space.sto_obj_doe(x_opt, self._doe_size)
        mean, std = self.full_space.inp_space.opt_moms(x_opt)
        names = self.full_space.inp_space.mulvar.names
        names = [names[i] for i in self.full_space.inp_space.mv_inds("sto_obj")]
        # Translating is not sufficient for lognormal and truncated normal
        inds = [i for i, x in enumerate(names) if "log" in x or "trunc" in x]
        if not inds:
            return self.base_doe * std + mean
        # Handle Lognormal
        binds = np.ones(self.base_doe.shape[1], dtype=bool)
        binds[inds] = False
        base_doe = self.base_doe.copy()
        base_doe[:, binds] = base_doe[:, binds] * std[binds] + mean[binds]
        mean = mean[inds]
        std = std[inds]
        cur_mv = self.full_space.inp_space.opt_mulvar(x_opt, domain="sto_obj")
        for ind in inds:
            base_doe[:, ind] = cur_mv.dists[ind].marg.ppf(base_doe[:, ind])
        return base_doe

    def est_mom(self, x_opt):
        """ Estimate conditional moments for a single optimization point x_opt

            Conditional moments are E[Y | x_opt] and Var[Y | x_opt]

            Parameters
            ----------
            x_opt : numpy.ndarray
                the coordinates of the optimization variables to compute
                the moments
            Returns
            -------
            mus : numpy.ndarray
                Estimated means, or if obj_wgt was not None,
                the combined mu + obj_wgt * sigma

            sigmas : numpy.ndarray
                Estimated variances or std. dev. depending on the settings.
                only returned if obj_wgt is None.
        """
        if x_opt.ndim == 1:
            x_opt = x_opt.reshape((1, -1))
        doe = self.gen_doe(x_opt)
        res = self.full_space.sto_obj(doe, x_opt)
        mus = np.mean(res, axis=0)
        sigmas = np.zeros(mus.shape)
        std_inds = self.use_std
        sigmas[std_inds] = np.std(res[:, std_inds], axis=0, ddof=1)
        var_inds = np.logical_not(std_inds)
        sigmas[var_inds] = np.var(res[:, var_inds], axis=0, ddof=1)
        if self.obj_wgt is None:
            return mus, sigmas
        return mus + self.obj_wgt * sigmas


class CondProba:
    """A chain of integtrators for the calculation of the probability

    This starts with a fast integrator to get an initial guess. If the
    guess is too far away from target_pf, this stops further calculations
    and returns the failure probability. Used for accelerating the
    optimization process. Chains with a single element are also possible.

    Parameters
    ----------
    num_inputs : int
        Number of stochastic inputs used for the constraints

    target_fail_prob : float
        Target failure probability. If unsure, just set it sufficiently low
        i.e. >=1e-6. Note that Numerical unstabilities start at 1e-9 due to
        scipy stats returning nans and infs

    num_parallel : int
        Number of parallel computations, if the used integrator supports it.
        If passed, the entry in call_args will override this.

    methods : None or list of str
        Names of the methods to use for the estimation. If None, a default
        chain will be selected depending the problem definition, which is
        recommended for new users.
        Currently the following names are supported:
            MC - Crude Monte Carlo
            DS - Directional simulation
            FORM - First order reliability method
            ISPUD - Importance sampling using design point (MPP)


    call_args : None or list
        keyword argument dict to pass to the integrator calc_prob_fail
        as call arguments. Any argument in this will override the
        initialization arguments with the same name i.e. target_fp and
        num_parallel

    target_tol : float
        Target tolerance for the failure probability. Also used
        for stopping the chain, if the computed failure probability
        is either smaller than target_fp * target_tol or larger than
        target_fp / target_tol.



    """

    def __init__(self, target_fail_prob: float, num_inputs: int, num_parallel: int = 4,
                 methods: typing.Optional[typing.Union[str, list]] = None, call_args: typing.Optional[dict] = None,
                 target_tol: float = 0.01):
        self.n_inp = num_inputs
        num_para = _n_para_chk(num_parallel)
        cargs = {"num_parallel": num_para, "multi_region": True}
        if methods is None:
            methods, cargs = _default_init(target_fail_prob, target_tol,
                                           num_inputs, num_para)
        if call_args is None:
            self.call_args = {**cargs}
        else:
            self.call_args = {**cargs, **call_args}
        self._tar_fp = target_fail_prob
        self._tar_tol = target_tol
        self.workers = _make_chain(methods)
        self._prob_tol()
        if "doe" in self.call_args.keys():
            doe = self.call_args["doe"]
            if doe.shape[1] != self.n_inp:
                msg = f"Shape mismatch between the number of inputs ({self.n_inp}) "
                msg += f"and the DoE {doe.shape[1]}"
                raise ValueError()
            mu_max = np.max(np.mean(doe, axis=0))
            sig_max = np.max(np.std(doe, axis=0))
            if abs(mu_max) > 1e-10 or abs(sig_max - 1) > 1e-10:
                msg = "Zero mean and unit variance is required for doe "
                msg += "in call_args, found mean == {mu_max} and "
                msg += "sigma == {sig_max} columns"
                raise ValueError(msg)
        elif _is_worker(self.workers, "ISPUD"):
            margs = [stats.norm() for k in range(self.n_inp)]
            self.call_args["doe"] = make_doe(100, margs, num_tries=1000)
        self.call_args["post_proc"] = False
        self.call_args["num_parallel"] = num_para

    @property
    def target_fail_prob(self):
        """target failure probability"""
        return self._tar_fp

    @target_fail_prob.setter
    def target_fail_prob(self, new_fp):
        """target failure probability"""
        if new_fp <= 0 or new_fp > 0.9:
            msg = "Target failure probability should lie in the interval (0,0.9]"
            raise ValueError(msg)
        self._tar_fp = new_fp
        self._prob_tol()

    @property
    def target_tol(self):
        """Target accuracy for failure probability estimation"""
        return self._tar_tol

    @target_tol.setter
    def target_tol(self, new_tol):
        """Target accuracy for failure probability estimation"""
        if new_tol <= 0 or new_tol > 0.9:
            msg = "Target probability accuracy should lie in the interval (0,0.9]"
            raise ValueError(msg)
        self._tar_tol = new_tol
        self._prob_tol()

    def _prob_tol(self):
        prob_tol = self._tar_fp * self._tar_tol
        if _is_worker(self.workers, "MC") and prob_tol < 1e-6:
            msg = "Crude Monte Carlo can be very inefficient for "
            msg += "such low probabilities of failure."
            warnings.warn(msg)
        self.call_args["prob_tol"] = prob_tol

    def calc_fail_prob(self, input_mv, constraints, const_args, verbose: int = 0):
        """ Calculate failure probability using the worker chain

        Parameters
        ----------

        input_mv : MultiVar instance
            Definition of the multivariate input

        constraints : list
            constraint functions to initialize the integrator

        const_args : None or list
            arguments to pass to the constraints

        Returns:
        --------
        pof : float
            probability of failure

        feasible : bool
            pof <= target_pf

        """
        if not self.workers:
            raise ValueError("No estimators defined")

        for worker in self.workers:
            estimator = worker(input_mv, constraints, const_args)
            try:
                pof = estimator.calc_fail_prob(**self.call_args)[0]
            except ValueError:
                if worker == self.workers[-1]:
                    print("Fatal error while calculating probability of failure with", worker)
                    print(input_mv)
                    print("Setting it to 100%.")
                    pof = 1.
                continue

            if verbose > 1:
                name = read_integrator_name(worker)
                print(f"{name} estimated the failure probability as {pof:.2e}.")

            if pof > self._tar_fp:
                prob_ratio = self._tar_fp / pof
            else:
                prob_ratio = pof / self._tar_fp
            if prob_ratio <= self._tar_tol:
                break

        if verbose > 0:
            try:
                name = read_integrator_name(worker)
                print(f"{name} estimated the failure probability as {pof:.2e}.")
            except NameError:
                pass
        return pof, pof <= self._tar_fp
