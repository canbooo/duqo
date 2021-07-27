# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:01:17 2020

@author: Bogoclu
"""

import numpy as np
from sklearn import svm, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize


def skopt_svr(X, Y):
    space = [Real(0.9, 2, name='C'),
             Real(10 ** -5, 10 ** 1, "log-uniform", name='epsilon'),
             Real(10 ** -5, 10 ** 1, "log-uniform", name='gamma'),
             ]

    # # this decorator allows your objective function to receive a the parameters as
    # # keyword arguments. This is particularly convenient when you want to set
    # # scikit-learn estimator parameters
    @use_named_args(space)
    def objective(**params):
        pipe[1].set_params(**params)
        pipe.fit(X, Y.ravel())
        return -np.mean(cross_val_score(pipe, X, Y.ravel(), cv=5, n_jobs=-1,
                                        scoring="neg_mean_squared_error"))
        # return -pipe.score(X, Y)

    model = svm.SVR(C=0.9999, cache_size=2000,  # break_ties=True,
                    # class_weight=cweights
                    kernel="rbf")
    pipe = pipeline.Pipeline([("input_scaler", StandardScaler()),
                              ("SVR", model)])

    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
    pipe[1].set_params(C=res_gp.x[0], epsilon=res_gp.x[1], gamma=res_gp.x[2])
    pipe.fit(X, Y.ravel())
    return pipe


def model_trainer(doe, *funcs, slices=None):
    models = []
    doe = doe[np.isfinite(doe).all(1)]
    for i_fun, func in enumerate(funcs):
        output = func(doe).reshape((-1, 1))  # Assume scalar
        inds = np.isfinite(output).all(1)
        # print(np.sum(output <= 0), "fails in DoE.")
        # output = output >= 0
        scaler = StandardScaler()
        output = scaler.fit_transform(output[inds])
        if slices is None:
            model = OutputScaledModel(skopt_svr(doe[inds], output[inds].ravel()), scaler)
        else:
            model = OutputScaledModel(skopt_svr(doe[inds, slices[i_fun]], output[inds].ravel()), scaler)
            model = SlicedModel(model, slices[i_fun])
        models.append(model)
    return models


class OutputScaledModel:

    def __init__(self, pipeline, scaler):
        self.model = pipeline
        self.scaler = scaler

    def predict(self, X):
        return self.scaler.inverse_transform(self.model.predict(X).reshape((X.shape[0], -1)))


class SlicedModel:
    def __init__(self, model, input_slice):
        self.model = model
        self.slice = input_slice

    def predict(self, x, *args, **kwargs):
        return self.model.predict(x[:, self.slice], *args, **kwargs)
