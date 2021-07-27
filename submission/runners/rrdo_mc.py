# -*- coding: utf-8 -*-
"""
Random sampling. Does not use optimization or surrogate models.

Created on Sun Jun 14 15:17:20 2020
@author: Bogoclu
"""
import os
import pickle
import random
import string
import sys

import numpy as np

from pyRDO import ConditionalProbability, ConditionalMoment
from pyRDO import RRDO, UniVar, MultiVar, InputSpace, FullSpace


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
    n_opt = len(lower)

    nit, nfev = 0, 0
    cands = np.random.rand(128, len(lower)) * (np.array(upper) - np.array(lower)) + np.array(lower)
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
    save_key = res_key
    if not save_key.endswith("_mc_res.pkl"):
        save_key += "_mc_res.pkl"
    with open(save_key, "wb") as f:
        pickle.dump(save_res, f)

    return save_res


def main(exname, save_dir="."):
    if exname == "ex1":
        from ..definitions.example1 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun
    elif exname == "ex2":
        from ..definitions.example2 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun
    elif exname == "ex3":
        from ..definitions.example3 import n_var, n_obj, n_con, target_pf, margs, lower, upper, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun
    else:
        raise ValueError(exname + " not recognized.")
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
                       pop_size=popsize, max_gens=maxgens,
                       res_key=res_key)


if __name__ == "__main__":
    _ = main("ex1")
