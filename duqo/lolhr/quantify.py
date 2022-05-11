# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:41:27 2020

@author: CanBo
"""
from __future__ import print_function, division
from copy import deepcopy
from typing import Callable, Optional, Tuple, List, Iterable

import numpy as np
from scipy import stats
from scipy.optimize import brentq
import pandas as pd

from duqo.stoch.model import MultiVar
from duqo.proba import MC, SUSE, DS
from duqo.proba.generic_integrator import GenericIntegrator
from duqo.doe.lhs import make_doe
from .optimize import adapt_doe

# from ..doe.lhs import inherit_lhs, optimize_doe, find_empty_bins, make_doe
# from ._integrator_base import GenericIntegrator
# from .ds import DS
# from .mc import MC
# from pyRDO.proba.suse_old.suse_old import SUSE
# from .clustering import get_dbclusters
# from sklearn.neighbors import NearestNeighbors

def _get_default_integrator(num_dims, prob_tol):
    if prob_tol > 1e-5:
        print("Using MC")
        return MC
    if num_dims <= 5:
        print("Using DS")
        return DS
    print("Using SUSE")
    return SUSE


def _is_converged_tscore(tolerance, pf, pf_std, pfs, n_samps, *args, **kwargs):
    if pfs and pf > 0:
        res = stats.ttest_ind_from_stats(pfs[-1]["mu"],
                                         pfs[-1]["sigma"],
                                         n_samps,
                                         pf, pf_std, n_samps)
        conv = res.pvalue > tolerance
        if conv:
            print("Early convergence due to t-test score", res.pvalue)
        return conv, res.pvalue
    return False, 0


# def _is_converged_reltscore(tolerance, pf, pf_std, pfs, n_samps, *args, **kwargs):
#     if pfs and pf > 0:
#         res = stats.ttest_ind_from_stats(pfs[-1]["mu"],
#                                          pfs[-1]["sigma"],
#                                          n_samps,
#                                          pf, pf_std, n_samps)
#         conv = res.pvalue > tolerance
#         if conv:
#             print("Early convergence due to t-test score", res.pvalue)
#         return conv, res.pvalue
#     return False, 0

def _is_converged_reldif(tolerance, pf, pf_std, pfs, *args, **kwargs):
    if pfs and pf > 0:
        diff = np.abs(pf - pfs[-1]["mu"]) / pf
        conv = diff < tolerance
        if conv:
            print("Early convergence due to relative difference", diff)
        return conv, diff
    return False, np.inf


class LoLHR4RA:
    """
    Reliability assessment using Local Latin hypercube sampling
    
    This class implements a surrogate based reliability analysis method using
    the method from [1].
    [1] C. Bogoclu, D. Roos, Reliability analysis of non-linear and multimodal limit state functions using adaptive
        kriging, 2017, ICOSSAR
    """
    def __init__(self, multivariate: MultiVar, constraints: List[Callable], constraint_args: Optional[Iterable] = None,
                 std_norm_to_orig: Optional[Callable] = None, orig_to_std_norm: Optional[Callable] = None):
        self.multivariate = multivariate

        self.constraints = constraints
        self.constraint_args = constraint_args
        self.model_constraints = None  # This will be generated after sampling
        self.model_constraint_args = None
        self.std_norm_to_orig = std_norm_to_orig
        self.orig_to_std_norm = orig_to_std_norm

    def calc_fail_prob(self, model_trainer: Callable, start_doe: Optional[Tuple[np.ndarray, np.ndarray]]=None,
                       step_size: int = 4, model_trainer_args: Optional[Iterable] = None,
                       max_evals: float = np.inf, convergence_test: str = "t-test",
                       prob_tol: float = 1e-6,
                       integrator: Optional[GenericIntegrator] = None,
                       **kwargs):
        """
        Model list output from model_trainer will be passed as the last argument
        to the passed limit state functions, which should be in form


        Parameters
        ----------
        model_trainer : Callable
            model_trainer recieves input and output coordinates as its first arguments and returns
            a list of surrogate models, which implement a predict function i.e. y = model.predict(x).
        start_doe : Optional[Tuple(np.ndarray, np.ndarray)],
            If passed, the first element will be used as the inputs and the second element as the outputs
            of the initial model. Otherwise, a start_doe will be generated automatically. The default is None.
        step_size : int, optional
            DESCRIPTION. The default is 4.
        prob_tol : float, optional
            DESCRIPTION. The default is 1e-6.
        num_samples : int, optional
            DESCRIPTION. The default is 4.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if start_doe is None:
            start_doe = max(2 * len(self.multivariate), 10)
            # num_start = 4
        if isinstance(start_doe, int):
            lower, upper = self.multivariate.quantile_bounds(prob_tol)
            margs = [stats.uniform(l, u - l) for l, u in zip(lower, upper)]
            start_doe = make_doe(start_doe, margs)
        if start_doe.shape[0] < 2:
            raise ValueError("Start doe has too few samples")
        cur_doe = start_doe.copy()
        if integrator is None:
            integrator = _get_default_integrator(cur_doe.shape[1], prob_tol)
        n_samps = cur_doe.shape[0]
        pfs = []
        base_args = deepcopy(self.const_args)
        iteration = 0
        if convergence_test == "t-test":
            conv_checker = _is_converged_tscore
            tol = 0.95
            if integrator == MC:
                tol = 0.9
        else:
            conv_checker = _is_converged_reldif
            tol = 0.01
            if integrator == MC:
                tol = 0.1
        while n_samps <= max_evals:
            print()
            print("Starting model update...")
            models = model_trainer(cur_doe, *model_trainer_args)
            print("Model update complete")
            self.const_args = [args + [models] for args in base_args]
            (pf,
             pf_var,
             inter) = self.model_fail_prob(cur_doe, integrator=integrator,
                                           prob_tol=prob_tol,
                                           ttest=convergence_test == "t-test",
                                           **kwargs)

            pf_std = np.sqrt(pf_var)
            n_samps = cur_doe.shape[0]
            isconv, score = conv_checker(tol, pf, pf_var, pfs, n_samps)

            msg = f"Iter. {iteration} {n_samps} samp.- P(F): {pf:.4e} "
            if pfs and pf > 0:
                msg += f"rel. change: {(pfs[-1]['mu'] - pf) / pf:.4f} "
                if convergence_test != "t-test":
                    score = -score
                msg += f"score: {score:.4f}"
            print(msg)
            iteration += 1
            pfs.append({"mu": pf, "sigma": pf_std})

            if isconv:
                # Message printed during isconv check
                break
            if max_evals - n_samps < step_size:
                print("Maximum allowed number of iterations has been reached.")
                break
            lower, upper = self.multivariate.get_probability_bounds(prob_tol)
            cur_doe = adapt_doe(lower, upper, cur_doe, inter.x_lsf, inter.x_fail,
                                num_samples=step_size, prob_tol=prob_tol,
                                return_update_only=False,
                                **kwargs)
        self.const_args = base_args
        return pf, pf_var, cur_doe, pfs, models

    def model_fail_prob(self, doe, integrator=None, prob_tol=1e-6,
                        ttest=True, **kwargs):
        if integrator is None:
            integrator = _get_default_integrator(doe.shape[1], prob_tol)
        inter = integrator(self.multivariate, self.constraints,
                           constraint_args=self.const_args,
                           std_norm_to_orig=self.u2x,
                           orig_to_std_norm=self.x2u)

        if integrator == MC:
            prob_tol = prob_tol * kwargs.get("CoV", 0.1) ** -2
        if ttest:
            kwargs["converge"] = False
        fail_prob, fail_prob_var = inter.calc_fail_prob(prob_tol=prob_tol, multi_region=True,
                                                        post_proc=True, num_parallel=1,
                                                        **kwargs)[:2]
        return fail_prob, fail_prob_var, inter

    def model_constraints(self):
        raise NotImplementedError


