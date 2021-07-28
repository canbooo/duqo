# -*- coding: utf-8 -*-
"""
Direct optimization strategy. Does not use a surrogate model.

Created on Sun Jun 14 15:17:20 2020

@author: Bogoclu
"""
import os
import pickle
import random
import string
import sys

import numpy as np

from duqo import ConditionalProbability, ConditionalMoment
from duqo import RRDO, UniVar, MultiVar, InputSpace, FullSpace
from .inspyred_optimizer import InspyredOptimizer


def direct_rrdo(objectives, constraints, num_obj, num_con, n_inp_total,
                lower, upper, margs, sto_inps=None, opt_inps=None,
                ra_methods=None, scale_objs=False,
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
    full_space = FullSpace(inp_space, num_obj, num_con,
                           obj_fun=objectives, obj_arg=obj_arg,
                           con_fun=constraints, con_arg=con_arg,
                           sto_objs=sto_obj_inds, sto_cons=sto_con_inds)
    cmom = ConditionalMoment(full_space, obj_wgt=obj_wgt, base_doe=base_doe)
    cprob = None
    if target_fail_prob is not None:
        cprob = ConditionalProbability(target_fail_prob,
                                       len(full_space.con_inds["sto"]),
                                       call_args={"multi_region": True},
                                       methods=ra_methods,
                                       )
    problem = RRDO(full_space, co_fp=cprob, co_mom=cmom)
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
    res = opter.optimize(pop_size=pop_size, max_gens=max_gens,
                         punish_factor=punish_factor,
                         pareto_size=pareto_size, verbose=verbose,
                         start_gen=start_gen)

    cands = [np.array(r.candidate) for r in res]

    nit, nfev = opter.nit, opter.nfev

    results = []
    for i, c in enumerate(cands):
        print("Computing final result for pareto design", i + 1, "of", len(cands))
        proba_res, rob_res = problem.gen_post_proc(c)
        results.append({"proba": proba_res,
                        "rob": rob_res})
    save_res = {"candidates": cands,
                "results": results,
                "num_opt_it": nit,
                "num_opt_fev": nfev}

    if not res_key.endswith("_direct_res.pkl"):
        res_key += "_direct_res.pkl"
    with open(res_key, "wb") as f:
        pickle.dump(save_res, f)
    return save_res


def main(exname, save_dir=".", force_pop_size=None, force_opt_iters=None):
    if exname == "ex1":
        from ..definitions.example1 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, \
            ra_methods, scale_objs, obj_fun, con_fun, opt_inps
    elif exname == "ex2":
        from ..definitions.example2 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, \
            ra_methods, scale_objs, obj_fun, con_fun, opt_inps
    elif exname == "ex3":
        from ..definitions.example3 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, \
            ra_methods, scale_objs, obj_fun, con_fun, opt_inps
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
    return direct_rrdo(obj_fun, con_fun, n_obj, n_con, n_var,
                       lower, upper, margs, target_fail_prob=target_pf,
                       verbose=1, ra_methods=ra_methods, scale_objs=scale_objs,
                       pop_size=popsize, max_gens=maxgens, opt_inps=opt_inps,
                       res_key=res_key)


if __name__ == "__main__":
    _ = main("ex1")
