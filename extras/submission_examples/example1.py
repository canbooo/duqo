# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:12:16 2020

@author: Admin
"""
import numpy as np


def scale_styta(x):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) / 180


def scale_lin(x):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    return (5 * np.sqrt(x.shape[1]) - x.sum(1)) / 7


def himmblau(x, gamma=1):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    res = ((x[:, 0] ** 2 + x[:, 1]) / 1.81 - 11) ** 2
    res += ((x[:, 0] + x[:, 1] ** 2) / 1.81 - 7) ** 2
    return res - 45 * gamma


def obj_fun(x, locs=(0, 1)):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    objs = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            objs[:, i_loc] = scale_lin(x)
        elif loc == 1:
            objs[:, i_loc] = scale_styta(x)
    return objs


def con_fun(x, locs=(0, )):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    cons = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            cons[:, i_loc] = himmblau(x)
    return cons


def model_obj(x, models=None, locs=(0, 1)):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    if not np.isfinite(x).all():
        print("model_obj", x[np.logical_not(np.isfinite(x).any(1))])
    objs = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            objs[:, i_loc] = models[0].predict(x).ravel()
        elif loc == 1:
            objs[:, i_loc] = models[1].predict(x).ravel()
    return objs


def model_con(x, models=None, locs=(0, )):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    if not np.isfinite(x).all():
        print("model_con", x[np.logical_not(np.isfinite(x).any(1))])
    cons = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            cons[:, i_loc] = models[2].predict(x).ravel()
    return cons


n_var = 2
n_obj = 2
n_con = 1
obj_wgt = 1.96
target_pf = 1e-6
margs = [{"name": "norm",
          "kwargs": {"mean": 0., "std": 0.2}} for _ in range(n_var)]
lower = [-5, -5]
upper = [5, 5]
n_start = 32
n_step = 8
n_stop = 64
popsize = 100
maxgens = 100
ra_methods = ["DS"]
scale_objs = True
opt_inps = None  # Everything is an opt_inp
funs = [scale_lin, scale_styta, himmblau]  # Model order
ref = [m["kwargs"]["mean"] for m in margs]
