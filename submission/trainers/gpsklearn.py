# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:58:02 2019

@author: Bogoclu
"""

import itertools
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel
from sklearn.pipeline import Pipeline


def model_trainer(doe, *functions):
    models = []
    for func in functions:
        output = func(doe).reshape((doe.shape[0], -1))
        print(f"Training GP for function {func.__name__} with {output.shape[0]} samples")
        models.append(fit_gpr(doe, output))
    return models


def fit_gpr(X, y, noise=False, anisotropy=True, kernel_names=None, scale_output=True):
    """
    Fit GP to inputs X and the output y after scaling X.
    sklearn scales y internally.

    Parameters
    ----------
    X : np.ndarray
        input array with shape (num_sample, num_dimensions)

    y : np.ndarray
        output array with shape (num_sample, num_outputs)

    noise : bool
        Controls whether to train with noise aka. additive white kernel

    anisotropy : bool
        Controls if the anisotropic kernels should be trained too.
        Isotropic versions will be trained in any case

    kernel_names : iterable
        Iterable of kernel names to test. Leaving this None is
        recommended. Possible kernels:
            'RBF', 'Matern0.5', 'Matern1.5', 'Matern2.5', 'RationalQuadratic'
        Notice no space after Matern kernels.Other Kernel names from sklearn as 
        strings are also allowed with limited functionality of gradients.
    scale_output : bool
        If True, the output is scaled. Defaults to False


    Returns
    -------
    gp : sklearn.Pipeline
        Fitted GP model with input scaling

    Usage
    ------
    >>> gp = fit_gpr(X_train, y_train)
    >>> y_pred = gp.predict(y_test)
    """
    kern_names, kernel_classes = _kernelize(kernel_names, anisotropy)
    n_dim = X.shape[1]
    restarts = 50 if X.shape[0] < 256 else 10
    theta_start = 1e-3

    pipe = _make_sum_pipeline(kern_names, kernel_classes, noise, n_dim,
                              theta_start, scale_output, restarts=restarts,
                              )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return pipe.fit(X, y)


def _kernelize(kernel_names, anisotropy):
    if kernel_names is None:
        kernel_names = ["RBF", "RationalQuadratic", "Matern0.5", "Matern1.5", "Matern2.5"]
    kern_names = kernel_names[:]
    globs = globals()
    kernel_classes = [globs[k] if "Matern" not in k else globs[k[:-3]] for k in kern_names]
    if anisotropy:
        kern_names += [n + " anisotropic" for n in kern_names if n != "RationalQuadratic"]
        kernel_classes += [k for k in kernel_classes if "RationalQuadratic" not in str(k)]
    return kern_names, kernel_classes


def _make_sum_pipeline(kernel_names, kernel_classes, noise, n_dim,
                       theta=1e-3, scale_output=False, restarts=10,
                       ):
    """ Create Pipline Object from inputs"""

    bounds = tuple((1e-5, 1e5))
    isotropes, anisotropes = {}, {}
    for name, kernel_class in zip(kernel_names, kernel_classes):
        if "anisotropic" in name.lower():
            anisotropes[name.split()[0]] = kernel_class

        else:
            isotropes[name] = kernel_class
    isotropes = {name: val for name, val in isotropes.items() if name not in anisotropes}

    kernel = 0
    for kernel_name, kernel_class in isotropes.items():
        if "matern" in kernel_name.lower():
            # also this
            nu = float(kernel_name[-3:])
            kernel += kernel_class(length_scale=theta, nu=nu,
                                   length_scale_bounds=bounds)
        else:
            kernel += kernel_class(theta)
    theta = theta * np.ones(n_dim)
    bounds = [bounds] * n_dim
    for kernel_name, kernel_class in anisotropes.items():
        if "matern" in kernel_name.lower():
            # also this
            nu = float(kernel_name[-3:])
            kernel += kernel_class(length_scale=theta, nu=nu,
                                   length_scale_bounds=bounds)
        else:
            kernel += kernel_class(length_scale=theta, length_scale_bounds=bounds)

    kernel = kernel * ConstantKernel()

    if noise:
        kernel += WhiteKernel()
    kernel_name = ""
    for name in itertools.chain(isotropes.keys(), anisotropes.keys()):
        kernel_name += name + " + "
    kernel_name = kernel_name[:-3]  # remove last plus

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=restarts,  # alpha=1e-10,
                                   normalize_y=scale_output, optimizer=_gp_optimizer)
    pipe = Pipeline([("input_scaler", StandardScaler()),
                     ("GP " + kernel_name, gpr)])
    return pipe


def _gp_optimizer(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be minimized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True,
                       bounds=bounds,
                       options={"maxiter": 1e5, "maxfun": 2e5, "maxls": 1000,
                                "gtol": 2.220446049250313e-09, 'disp': 0})
    if not opt_res.success:
        print("Optimization failed")
        print(opt_res.message)
    theta_opt, func_min = opt_res.x, opt_res.fun

    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.

    return theta_opt, func_min
