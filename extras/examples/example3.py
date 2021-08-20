# -*- coding: utf-8 -*-
# %%
"""
Created on Sat Jul 11 19:12:16 2020

@author: Admin
"""
import numpy as np


def oblique_bending(x):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    # M1 - 0 | M2 - 1 | P/F - 2 | f_y - 3 | b - 4 | h - 5
    x[:, 3] = np.maximum(x[:, 3], 1e-6)
    bhfy = np.prod(x[:, [3, 4, 5]], axis=1)
    return 1 - (4 * x[:, 0] / bhfy / x[:, 5]) - (4 * x[:, 1] / bhfy / x[:, 4]) - (x[:, 2] / bhfy) ** 2


def obj_fun(x, locs=(0,)):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    objs = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            objs[:, i_loc] = x[:, 4] * x[:, 5]
    return objs


def con_fun(x, locs=(0, 1)):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    cons = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            cons[:, i_loc] = oblique_bending(x)
        elif loc == 1:
            prop = x[:, 4] / x[:, 5]
            cons[:, i_loc] = np.minimum(prop - 0.5, 2 - prop)
    return cons


def model_obj(x, **kwargs):
    return obj_fun(x)


def model_con(x, models=None, locs=(0, 1)):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    if not np.isfinite(x).all():
        print("model_con", x[np.logical_not(np.isfinite(x).any(1))])
    cons = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            cons[:, i_loc] = models[0].predict(x).ravel()
        elif loc == 1:
            prop = x[:, 4] / x[:, 5]
            cons[:, i_loc] = np.minimum(prop - 0.5, 2 - prop)
    return cons


n_var = 6
n_obj = 1
n_con = 2
obj_wgt = None
target_pf = 1.35e-3

means = [a * 1e6 for a in [250, 125, 2.5]] + [40]
covs = [0.3, 0.3, 0.2, 0.1]
margs = [{"name": "lognorm",
          "kwargs": {"mean": mu, "CoV": cov}} for mu, cov in zip(means, covs)]
for k in range(2):
    margs.append({"name": "norm",
                  "kwargs": {"mean": 500, "CoV": 0.01}})

ref = [m["kwargs"]["mean"] for m in margs]
lower = [100, 100]
upper = [1000, 1000]
n_start = 64
n_step = 16
n_stop = 128
popsize = 100
maxgens = 100
ra_methods = ["DS"]
scale_objs = True
sto_obj_inds = []
sto_con_inds = [0]
opt_inps = [4, 5]
funs = [oblique_bending]  # Model order
ref = [m["kwargs"]["mean"] for m in margs]
