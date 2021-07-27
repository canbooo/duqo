import os
from pathlib import Path
import pickle
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from pyRDO.lolhr.rrdo_lolhr import is_pareto

PRED_KEYS = {"mc_res": "Random",
             "direct_res": "Direct",
             "direct_short_res": "Direct (short)",
             "stat_gp_pred_res": "Stationary GP",
             "adapt_mean_gp_*_pred_res": "Gu et. al (GP)",
             "adapt_gp_*_pred_res": "LoLHR (GP)",
             "stat_svr_pred_res": "Stationary SVR",
             "adapt_mean_svr_*_pred_res": "Gu et. al (SVR)",
             "adapt_svr_*_pred_res": "LoLHR (SVR)",
             }


def load_data(pred_path):
    if not os.path.isfile(pred_path):
        return None, None
    with open(pred_path, "rb") as f:
        data = pickle.load(f)
    return data["candidates"], data["results"]


def to_pred_path(ex_name, key, res_dir):
    filename = "_".join([ex_name, key]) + ".pkl"
    full_path = os.path.join(res_dir, filename)
    if "*" not in full_path:
        return full_path
    full_path = Path(full_path)
    res = list(full_path.parent.glob(full_path.name))
    return res[0] if res else ""


ParetoData = namedtuple("ParetoData", ["mus", "sigmas", "objs", "fail_probs", "fails", "candidates"])


def get_objs_cons(candidates, results, target_pf, ex_name, con_fun, ref):
    if results is None:
        return None
    mus, sigs, objs, fail_probs = [], [], [], []
    for candidate, fitness in zip(candidates, results):
        r = fitness["rob"]["output"]
        try:
            p = fitness["proba"]["DS"]
        except KeyError:
            print(fitness["proba"].keys())
            raise
        mus.append(r[0])
        sigs.append(r[1])
        if ex_name == "ex3":
            obj = [np.prod(candidate[-2:]), p["fail_prob"]]
        else:
            obj = r[2]
        objs.append(obj)
        fail_probs.append(p["fail_prob"])
    mus, sigs, objs, fail_probs = [np.array(a) for a in [mus, sigs, objs, fail_probs]]
    fail_probs = fail_probs.ravel()
    fails = fail_probs > target_pf
    if "ex3" in ex_name:
        """Only example 3 has deterministic constraints"""
        cons = con_fun(to_fullspace(candidates, ref))
        det_fails = np.min(cons, axis=1) < 0
        print("det fails", det_fails.sum())
        print("prob fails", fails.sum())
        fails = np.logical_or(det_fails, fails)
    print(fails.sum(), "fails of", objs.shape[0])
    candidates = np.array(candidates)
    if candidates.ndim < 2:
        candidates = candidates.reshape((1, -1))
    return ParetoData(mus, sigs, objs, fail_probs, fails, candidates)


def to_fullspace(candidates, ref):
    fullx = np.ones((len(candidates), len(ref)))
    fullx *= ref
    fullx[:, -2:] = np.array(candidates)
    return fullx


def plot_surf_onfig(ax, ex_name, obj_fun, con_fun, lower, upper, ref):
    # return
    x = np.linspace(lower[0], upper[0])
    y = np.linspace(lower[1], upper[1])
    X, Y = np.meshgrid(x, y)
    inps = np.c_[X.ravel(), Y.ravel()]
    colors = ["g", "r", "b", "k"]
    if ex_name == "ex3":
        inps = to_fullspace(inps, ref)
    obs = obj_fun(inps)
    proxies = {}
    for i_obj in range(obs.shape[1]):
        o1 = obs[:, [i_obj]].reshape(X.shape)
        c = colors.pop()
        ax.contour(X, Y, o1, colors=c)
        proxies[f"Obj. {i_obj + 1}"] = plt.Line2D((0, 0), (0, 0), color=c)
    # o2 = obs[:, 1].reshape(X.shape)
    cns = con_fun(inps)
    for i_con in range(cns.shape[1]):
        c1 = cns[:, [i_con]].reshape(X.shape)
        c = colors.pop()
        ax.contour(X, Y, c1, levels=[0], colors=c)
        proxies[f"Con. {i_con + 1}"] = plt.Line2D((0, 0), (0, 0), color=c)
    return proxies


def plot_archives(plot_data, ex_name, res_dir, obj_fun, con_fun, lower, upper, ref):
    fig, ax = plt.subplots(figsize=(12, 7))
    proxies = plot_surf_onfig(ax, ex_name, obj_fun, con_fun, lower, upper, ref)
    # ax.legend(list(proxies.values()), list(proxies.keys()))
    for name, data in plot_data.items():
        # plt.scatter(objs[fails, 0], objs[fails, 1], label="fail",s=50, marker="s")
        cands = data.candidates
        proxies[name] = ax.scatter(cands[:, 0], cands[:, 1], label=name)
    ax.legend(list(proxies.values()), list(proxies.keys()))
    ax.set_title("Pareto Designs - Example " + ex_name[-1])
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    plt.savefig(os.path.join(res_dir, ex_name + "_pareto_designs.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_pareto_frontiers(plot_data, ex_name, res_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, data in plot_data.items():
        safe = np.logical_not(data.fails)
        if not safe.any():
            continue
        objs = data.objs[safe]
        mask = is_pareto(objs)
        ax.scatter(objs[mask, 0], objs[mask, 1], label=name)
    ax.set_title("Pareto Frontiers - Example " + ex_name[-1])
    ax.legend()
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel("f_1(x)")
    ax.set_ylabel("f_2(x)")
    plt.savefig(os.path.join(res_dir, ex_name + "_pareto_frontiers.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot(ex_name, res_dir):
    if ex_name == "ex1":
        from .definitions.example1 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, ref, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con, \
            lower, upper
    elif ex_name == "ex2":
        from .definitions.example2 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, ref, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con, \
            lower, upper
    elif ex_name == "ex3":
        from .definitions.example3 import n_var, n_obj, n_con, target_pf, margs, lower, upper, n_start, ref, \
            n_step, n_stop, popsize, maxgens, ra_methods, scale_objs, obj_fun, con_fun, funs, model_obj, model_con, \
            lower, upper
    else:
        raise ValueError(ex_name + " not recognized.")
    plot_data = {name: get_objs_cons(*load_data(to_pred_path(ex_name, key, res_dir)),
                                     target_pf, ex_name, con_fun, ref)
                 for key, name in PRED_KEYS.items()}

    plot_pareto_frontiers(plot_data, ex_name, res_dir)
    plot_archives(plot_data, ex_name, res_dir, obj_fun, con_fun, lower, upper, ref)
