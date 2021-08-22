# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:42:00 2019

@author: Bogoclu
"""

from copy import deepcopy
import numpy as np

from duqo.optimization.predict import CondProba, CondMom, read_integrator_name
from duqo.optimization.space import FullSpace, check_shape


class RRDO:
    """ Stochatic Optimization Framework
        Computes the conditional moments of the objectives
        and conditional failure probabilities of the constraints
        in the context of optimization

    full_space : FullSpace
        Definition of the stochastic and optimization spaces

    target_fp : float
        Target failure probability. If not None, co_fp will use this
        value, otherwise it will be read from co_fp

    co_fp : CondProba
        Integrator chain to use for estimating the probability of failure.
        If None, a default chain will be generated based on the problem.


    co_mom : CondMom
        Estimator for the conditional mean and variance. If None, one with the
        default settings will be created. Note that the default behaviour is to
        return the variances of the objectives separatly on the right hand side
        of the output matrix in self.obj and self.obj_con

    opt_chk : function
        If not None, this will be used to check if a point is optimal. If not
        the estimation of the failure probability will be skipped and the point
        will be declared unfeasible. It must accept two arguments namely the
        input point and the corresponding objective values. Only relevant for
        the obj_con method


    After initialization, use obj, con or obj_con functions methods.
    """

    def __init__(self, full_space: FullSpace, targ_fp: float = None,
                 co_fp: CondProba = None, co_mom: CondMom = None, opt_chk=None):
        self.full_space = full_space
        if co_mom is None:
            co_mom = CondMom(full_space)
        if not hasattr(co_mom, "est_mom"):
            raise TypeError("If passed, co_mom must be a CondMom instance.")
        self.co_mom = co_mom
        if co_fp is None:
            if targ_fp is None:
                targ_fp = 1e-2
            n_sto_con_inps = np.sum(full_space.inp_space.inds["sto_con"])
            if n_sto_con_inps > 1:
                co_fp = CondProba(targ_fp, n_sto_con_inps)
        elif targ_fp is not None:
            co_fp.set_target_prob(targ_fp)
        if not hasattr(co_fp, "calc_fail_prob"):
            raise TypeError(f"If passed, co_fp must be a CondProba instance. Got  {type(co_fp)}.")
        self.co_fp = co_fp
        self.opt_chk = opt_chk

    def obj(self, x_opt, det_objs=None):
        """ Evaluates all objectives in deterministic and stochastic
        space

        Parameters
        ----------
        x_opt : numpy.ndarray
            A single point in the optimization space

        det_objs : None or numpy.ndarray
            Deterministic objective values. If None, will be computed
            from the user supplied function obj_fun. It passed, must be
            a 2-d array with shape (num_sample, num_objectives)

        Returns
        -------
        objs : numpy.ndarray
            Values of the objective functions. Columns correspond to the
            individual objectives and rows to the individual points in
            the same order as x_opt. If stochastic objectives are
            defined, the shape of this will differ from the original
            number of objectives, if AlgoSettings.objective_weights
            is not passed.
        """
        x_opt = check_shape(x_opt, self.full_space.inp_space.inds["opt"].sum(),
                            parent="obj")
        rob_w = self.co_mom.obj_wgt
        if det_objs is None:
            det_objs = self.full_space.det_obj(x_opt)

        if not self.full_space.obj_inds["sto"]:
            # Why are you using this again?
            return det_objs

        if rob_w is None:
            objs = np.hstack((det_objs, det_objs[:, self.full_space.obj_inds["sto"]]))
        else:
            objs = det_objs
        return self._stoch_obj(x_opt, objs)

    def _stoch_obj(self, x_opt, objs):
        rob_w = self.co_mom.obj_wgt
        for i_design in range(x_opt.shape[0]):
            x_curr = x_opt[[i_design], :]
            if rob_w is None:
                mean, sigma = self.co_mom.est_mom(x_curr)
                objs[i_design, self.full_space.obj_inds["sto"]] = mean
                objs[i_design, self.full_space.dims[0]:] = sigma
            else:
                objs[i_design, self.full_space.obj_inds["sto"]] = \
                    self.co_mom.est_mom(x_curr)
        return objs

    def con(self, x_opt, det_cons=None, verbose=0):
        """ Evaluates all constraints in deterministic
        and stochastic space

        Parameters
        ----------
        x_opt : numpy.ndarray
            A single point in the optimization space

        det_cons : None or numpy.ndarray
            Deterministic evaluation of the constraints for x_opt. If None, it
            will be computed from the user supplied function con_fun

        verbose : int or float
            a number to adjust the verbosity. Currently, anything > 1
            displays all prints.

        Returns
        -------
        feasible : numpy.ndarray(dtype=bool)
            A boolean array corresponding to the feasibility of each passed
            candidate in x_opt.  It checks for deterministic and stochastic
            constraints
        fail_probs : numpy.ndarray
            Calculated probabilities of failure for each candidate. Note that
            if the evaluation is skipped due to infeasibility, the probability
            of failure is returned as 1 (i.e. 100 %)
        det_cons : numpy.ndarray
            Values of the deterministic evaluation of the constraints.
            The columns correspond to the constraints and
        """
        x_opt = check_shape(x_opt, self.full_space.inp_space.inds["opt"].sum(),
                            parent="con")
        if det_cons is None:
            det_cons = self.full_space.det_con(x_opt)
        if det_cons.ndim == 1:
            det_cons = det_cons.reshape((1, -1))

        feasible = np.min(det_cons, axis=1) >= 0
        fail_probs = np.array(np.logical_not(feasible),
                              dtype=np.float64).reshape((-1, 1))

        if not self.full_space.con_inds["sto"]:  # Why are you using this again?
            return feasible, det_cons, fail_probs

        for i_design in range(x_opt.shape[0]):
            if not feasible[i_design]:
                fail_probs[i_design, :] = 1.
            x_curr = x_opt[[i_design], :]
            curr_mv = self.full_space.inp_space.opt_mulvar(x_curr)

            fprob, feas = self.co_fp.calc_fail_prob(curr_mv,
                                                    [self.full_space.sto_con],
                                                    [x_curr], verbose=verbose)
            fail_probs[i_design, :] = fprob
            feasible[i_design] = feas
        return feasible, fail_probs, det_cons

    def obj_con(self, x_opt, verbose: int = 0):
        """Evaluates the defined problem
        Should only be called after define

        Parameters
        ----------
        x_opt : numpy.ndarray
            a 2-D array with rows corresponding to the individual points or
            candidates and the columns corresponding to the individual
            dimensions in the optimization space.
        verbose : int or float
            a number to adjust the verbosity. Currently, anything > 1
            displays all prints.

        Returns
        -------
        objs : numpy.ndarray
            Values of the objective functions. Columns correspond to the
            individual objectives and rows to the individual points in
            the same order as x_opt. If stochastic objectives are
            defined, the shape of this will differ from the original
            number of objectives, if AlgoSettings.objective_weights
            is not passed.
            In that case, the estimated means are returned in the
            same columns as the stochastic objectives. Furthermore, the matrix
            will be extendend with the same number of columns as the number
            of stochastic objectives. These last columns will correspond
            the variances (or std. deviations if settings.use_objective_std
            is True).
        feasible : numpy.ndarray(dtype=bool)
            A boolean array corresponding to the feasibility of each passed
            candidate in x_opt.  It checks for deterministic and stochastic
            constraints
        det_cons : numpy.ndarray
            Values of the deterministic evaluation of the constraints.
            The columns correspond to the constraints and
        fail_probs : numpy.ndarray
            Calculated probabilities of failure for each candidate. Note that
            if the evaluation is skipped due to infeasibility, the probability
            of failure is returned as 1 (i.e. 100 %)

        All returned arrays except feasible are two-dimensional
        """
        x_opt = check_shape(x_opt, self.full_space.inp_space.inds["opt"].sum(),
                            parent="obj_con")
        if self.full_space.dims[1] > 0:
            det_cons = self.full_space.det_con(x_opt)
        else:
            det_cons = np.zeros(x_opt.shape[0], 1)
        if det_cons.ndim == 1:
            det_cons = det_cons.reshape((1, -1))
        objs = self.full_space.det_obj(x_opt)
        fail_probs = np.ones((x_opt.shape[0], 1), dtype=np.float64)
        feasible = np.min(det_cons, axis=1) >= 0
        if not self.full_space.obj_inds["sto"] and \
                not self.full_space.con_inds["sto"]:
            # Why are you using this again?
            return objs, feasible, det_cons, fail_probs

        if self.co_mom.obj_wgt is None and self.full_space.obj_inds["sto"]:
            objs = np.hstack((objs, objs[:, self.full_space.obj_inds["sto"]]))

        for i_design in range(x_opt.shape[0]):
            if not feasible[i_design] and verbose > 0:
                print(f"Skipping design {i_design} due to infeasibility.")
            if feasible[i_design]:
                x_curr = x_opt[[i_design], :]
                if self.full_space.obj_inds["sto"]:
                    curr_obj = objs[[i_design], :]
                    objs[i_design, :] = self._stoch_obj(x_curr, curr_obj)
                is_opt = True
                if self.opt_chk is not None:
                    x_f = self.full_space.inp_space.opt2full(x_curr)
                    is_opt = self.opt_chk(x_f, objs[i_design, :])
                    if not is_opt and verbose > 0:
                        print(f"Skipping candidate {i_design} due to inoptimality")
                        feasible[i_design] = False  # To match returned failure probability
                if self.full_space.con_inds["sto"] and is_opt:
                    if verbose > 0:
                        print(f"Candidate {i_design}:", end=" ")
                    res = self.con(x_curr, det_cons[[i_design], :],
                                   verbose=verbose)[:2]
                    fail_probs[i_design, :] = res[1]
                    feasible[i_design] = res[0]
        return objs, feasible, det_cons, fail_probs


    def gen_post_proc(self, x_opt, moment=True, proba=True):
        """Gather all samples used for the stochastic assessments
        for an optimization candidate x_curr.

        Parameters
        ----------
        x_opt : list or numpy.ndarray
            A single point defined in the optimization space i.e. containing
            the values of the optimization parameters. The stochastic values
            are read from the MultiVariate model.

        Returns
        -------
        proba_res : dict
            results of the probabilistic assesment with the following keys for
            each integrator:
                *`fail_prob` -- estimated probability of failure
                *`est_var`-- variance of the estimation
                *`safety_index` -- Safety index, also known as the sigma level.
                It is equal to Phi_inv(1-fail_prob_mu), where Phi_inv is the
                inverse of the CDF of standard normal distribution
                *`mpp` -- Most probable point of failure as found by the integrator
                *'safe_samp` - generated samples in the safe domain (`min(c)>=0`)
                *'fail_samp` - generated samples in the failure domain (`min(c)<0`)
                *'limit_state` - generated samples on the limit state (`min(c)==0`)

        Note that this method will recompute the post processing data, as the
        data is not saved during the optimization.
        """

        x_cur = np.array(x_opt)
        if x_cur.ndim > 1 and x_cur.shape[0] > 1:
            raise ValueError(
                f"x_opt must be a single point, not {x_cur.shape[0]}.")
        x_cur = x_cur.reshape((1, -1))
        proba_res, rob_res = {}, {}
        if self.full_space.con_inds["sto"] and proba:
            pp_int = deepcopy(self.co_fp)
            pp_int.call_args["post_proc"] = True
            pp_int.call_args["num_parallel"] = 1
            cur_mv = self.full_space.inp_space.opt_mulvar(x_cur)
            # Hack for Diss
            workers = [pp_int.workers[-1]]
            for int_lib in workers:
                estimator = int_lib(cur_mv, [self.full_space.sto_con], [x_cur])
                try:
                    res = estimator.calc_fail_prob(**pp_int.call_args)[:4]
                except ValueError:
                    pass
                else:
                    name = read_integrator_name(int_lib)
                    proba_res[name] = {}
                    proba_res[name]["fail_prob"] = res[0]
                    proba_res[name]["est_var"] = res[1]
                    proba_res[name]["safety_index"] = res[2]
                    proba_res[name]["mpp"] = res[3]
                    if estimator.x_safe.size > 0:
                        samps = np.unique(self.full_space.inp_space.stoch2full(estimator.x_safe, x_cur,
                                                                               domain="sto_con"), axis=0)
                    else:
                        samps = np.empty((0, self.full_space.inp_space.dims))
                    proba_res[name]["safe_samp"] = samps
                    if estimator.x_fail.size > 0:
                        samps = np.unique(
                            self.full_space.inp_space.stoch2full(estimator.x_fail, x_cur, domain="sto_con"),
                            axis=0)
                    else:
                        samps = np.empty((0, self.full_space.inp_space.dims))
                    proba_res[name]["fail_samp"] = samps
                    if estimator.x_lsf.size > 0:
                        samps = np.unique(
                            self.full_space.inp_space.stoch2full(estimator.x_lsf, x_cur, domain="sto_con"),
                            axis=0)
                    else:
                        samps = np.empty((0, self.full_space.inp_space.dims))
                    proba_res[name]["limit_state"] = samps
                    proba_res[name]["num_eval"] = estimator.num_eval
        if self.full_space.obj_inds["sto"] and moment:
            doe = self.co_mom.gen_doe(x_cur)
            rob_res["input"] = self.full_space.inp_space.stoch2full(doe, x_cur,
                                                                    "sto_obj")
            obj_wgt = None
            if self.co_mom.obj_wgt is not None:
                # To get individual moments
                obj_wgt = self.co_mom.obj_wgt.copy()
                self.co_mom.obj_wgt = None
            mus, sigs = self.co_mom.est_mom(x_cur)
            obj = None
            if obj_wgt is not None:
                self.co_mom.obj_wgt = obj_wgt
                obj = self.co_mom.est_mom(x_cur)
            rob_res["output"] = [mus, sigs, obj]
        return proba_res, rob_res
