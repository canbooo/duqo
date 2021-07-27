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


def scale_square(x):
    return np.sum((x - 2.25) ** 2, axis=1) / 50


def scale_tricky_cos(x):
    alpha = 1.475
    return 7. - np.sum((x / alpha) ** 2 - 5 * np.cos(2 * np.pi * x / alpha), axis=1)


def obj_fun(x, locs=[0, 1]):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    objs = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            objs[:, i_loc] = scale_styta(x)
        elif loc == 1:
            objs[:, i_loc] = scale_square(x)
    return objs


def con_fun(x, locs=[0]):
    if x.ndim < 2:
        x = x.reshape((1, -1))
    cons = np.zeros((x.shape[0], np.array(locs).size))
    for i_loc, loc in enumerate(locs):
        if loc == 0:
            cons[:, i_loc] = scale_tricky_cos(x)
    return cons


def model_obj(x, models=None, locs=[0]):
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


def model_con(x, models=None, locs=[0]):
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
target_pf = 1e-2
margs = [{"name": "norm",
          "kwargs": {"mean": 0., "std": 0.15}},
         {"name": "uniform",
          "kwargs": {"lower_bound": -0.25, "upper_bound": 0.25}}]
lower = [-4.5, -4.5]
upper = [4.5, 4.5]
n_start = 64
n_step = 16
n_stop = 128
popsize = 10  # This should be set to 100 for a reproduction of the results in the paper
maxgens = 10  # This should be set to 100 for a reproduction of the results in the paper
ra_methods = ["DS"]
scale_objs = True
funs = [scale_styta, scale_square, scale_tricky_cos]  # Model order
ref = [m["kwargs"]["mean"] if "mean" in m["kwargs"] else (m["kwargs"]["lower_bound"] + m["kwargs"]["upper_bound"]) / 2
       for m in margs]
