# -*- coding: utf-8 -*-
"""
Proposed LoLHR strategy
Bogoclu et. al., Local Latin Hypercube Refinement for Multi-Objective Design Uncertainty Optimization (2021)

Uses GP as model

Created on Sun Jun 14 15:17:20 2020
@author: CanBo
"""
import os
import pickle
import random
import string
import sys

import numpy as np

from duqo import ConditionalProbability, ConditionalMoment
from duqo import RRDO, UniVar, MultiVar, InputSpace, FullSpace
from duqo.lolhr import lolhr4rrdo
from .inspyred_optimizer import InspyredOptimizer
from ..trainers.gpsklearn import model_trainer


def direct_rrdo(objectives, constraints, trainer, trainer_args, start_samples,
                step_size, max_samples,
                model_objectives, model_constraints, num_obj, num_con,
                n_inp_total, lower, upper, margs,
                sto_inps=None, opt_inps=None, scale_objs=False, ra_methods=None,
                obj_arg=None, con_arg=None, sto_obj_inds: list = None,
                sto_con_inds: list = None, obj_wgt=1.96, base_doe=True,
                target_fail_prob=None, pop_size=100, max_gens=100,
                punish_factor=100, pareto_size=1000, verbose=0, start_gen=None,
                res_key=None):
    dists = [UniVar(m["name"], **m["kwargs"]) for m in margs]
    mv = MultiVar(dists)  # no correlation assumed
    if sto_inps is None:
        sto_inps = np.arange(len(mv))
    if opt_inps is None:
        opt_inps = np.arange(len(mv))
    if sto_obj_inds is None:
        sto_obj_inds = np.arange(num_obj)
    if sto_con_inds is None:
        sto_con_inds = np.arange(num_con)
    inp_space = InputSpace(mv, num_inp=n_inp_total,
                           opt_inps=opt_inps, sto_inps=sto_inps)

    models = None  # overwritten in lolhr4rrdo
    if obj_arg is None:
        model_obj_arg = [models]
    else:
        try:
            model_obj_arg = list(obj_arg) + [models]
        except TypeError:
            obj_arg = [obj_arg]
            model_obj_arg = obj_arg + [models]
    if con_arg is None:
        model_con_arg = [models]
    else:
        try:
            model_con_arg = list(con_arg) + [models]
        except TypeError:
            con_arg = [con_arg]
            model_con_arg = con_arg + [models]

    full_space = FullSpace(inp_space, num_obj, num_con,
                           obj_fun=model_objectives, obj_arg=model_obj_arg,
                           con_fun=model_constraints, con_arg=model_con_arg,
                           sto_objs=sto_obj_inds, sto_cons=sto_con_inds)

    problem = make_problem(full_space, obj_wgt, target_fail_prob,
                           base_doe, ra_methods)

    if res_key is None:
        res_key = ''.join(
            random.choice(string.ascii_lowercase) for i in range(6))
    # Do here to avoid errors after computation
    res_key = str(res_key)
    if "ex3" in res_key:
        def obj_con(x, *args, **kwargs):
            objs, feasible, det_cons, fail_probs = problem.obj_con(x, *args, **kwargs)
            fail_probs = np.maximum(target_fail_prob / 100, np.minimum(0.5, fail_probs.reshape((-1, 1))))
            objs = np.c_[objs, fail_probs]
            return objs, np.c_[det_cons, (target_fail_prob - fail_probs) / target_fail_prob]
    else:
        def obj_con(x, *args, **kwargs):
            objs, feasible, det_cons, fail_probs = problem.obj_con(x, *args, **kwargs)
            if target_fail_prob is None:
                return objs, det_cons
            return objs, np.c_[det_cons, (target_fail_prob - fail_probs) / target_fail_prob]

    if obj_wgt is None:
        num_obj += len(sto_obj_inds)

    opter = InspyredOptimizer(obj_con, lower, upper, method="NSGA", scale_objs=scale_objs, verbose=verbose)
    opter_args = dict(pop_size=pop_size, max_gens=max_gens,
                      punish_factor=punish_factor,
                      pareto_size=pareto_size, verbose=verbose,
                      start_gen=start_gen)

    res = lolhr4rrdo(problem, lower, upper, opter, trainer, step_size,
                     max_samples, start_doe=start_samples, optimizer_kwargs=opter_args,
                     model_trainer_args=trainer_args)

    cur_doe, cands, _, cons = res
    nit, nfev = opter.nit, opter.nfev

    results = []
    for i, c in enumerate(cands):
        print("Computing final result for pareto design", i + 1, "of", len(cands))
        proba_res, rob_res = problem.gen_post_proc(c)
        results.append({"proba": proba_res,
                        "rob": rob_res})
    save_res_pred = {"candidates": cands,
                     "results": results,
                     "num_opt_it": nit,
                     "num_opt_fev": nfev}

    if not res_key.endswith(f"_adapt_gp_{start_samples}_{step_size}_{max_samples}_pred_res.pkl"):
        res_key += f"_adapt_gp_{start_samples}_{step_size}_{max_samples}_pred_res.pkl"
    with open(res_key, "wb") as f:
        pickle.dump(save_res_pred, f)

    full_space = FullSpace(inp_space, num_obj, num_con,
                           obj_fun=objectives, obj_arg=obj_arg,
                           con_fun=constraints, con_arg=con_arg,
                           sto_objs=sto_obj_inds, sto_cons=sto_con_inds)
    problem = make_problem(full_space, obj_wgt, target_fail_prob,
                           base_doe, ra_methods)
    res_key = res_key.replace("pred", "true")
    results = []
    for i, c in enumerate(cands):
        print("Computing final result for pareto design", i + 1, "of", len(cands))
        proba_res, rob_res = problem.gen_post_proc(c)
        results.append({"proba": proba_res,
                        "rob": rob_res})
    save_res_true = {"candidates": cands,
                     "results": results,
                     "doe": cur_doe,
                     "num_opt_it": nit,
                     "num_opt_fev": nfev}

    with open(res_key, "wb") as f:
        pickle.dump(save_res_true, f)

    return save_res_pred, save_res_true


def make_problem(full_space, obj_wgt, target_fail_prob, base_doe, ra_methods,
                 **kwargs):
    cmom = ConditionalMoment(full_space, obj_wgt=obj_wgt, base_doe=base_doe)
    cprob = None
    if target_fail_prob is not None:
        cprob = ConditionalProbability(target_fail_prob,
                                       len(full_space.con_inds["sto"]),
                                       call_args=kwargs,
                                       methods=ra_methods,
                                       )
    problem = RRDO(full_space, co_fp=cprob, co_mom=cmom)
    return problem


def main(exname, save_dir=".", force_pop_size=None, force_opt_iters=None):
    if exname == "ex1":
        from ..definitions.example1 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, opt_inps, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con
    elif exname == "ex2":
        from ..definitions.example2 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, opt_inps, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con
    elif exname == "ex3":
        from ..definitions.example3 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, opt_inps, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con
    else:
        raise ValueError(exname + " not recognized.")

    if force_pop_size is not None:
        popsize = force_pop_size
    if force_opt_iters is not None:
        maxgens = force_opt_iters

    save_dir = os.path.join(save_dir, "results")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    try:
        res_key = sys.argv[1]
    except IndexError:
        res_key = None

    res_key = exname + (res_key if res_key is not None else "")
    res_key = os.path.join(save_dir, res_key)
    return direct_rrdo(obj_fun, con_fun, model_trainer, funs,
                       n_start, n_step, n_stop,
                       model_obj, model_con, n_obj, n_con, n_var,
                       lower, upper, margs, target_fail_prob=target_pf,
                       verbose=1, scale_objs=scale_objs, ra_methods=ra_methods,
                       pop_size=2 * popsize, max_gens=2 * maxgens, res_key=res_key, opt_inps=opt_inps)


if __name__ == "__main__":
    _ = main("ex1")
