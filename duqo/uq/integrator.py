# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:05:41 2019

@author: Bogoclu
"""
import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from .lsf import LimitStateFunction


@dataclass
class ConvergencePlot:
    mean_convergence: np.ndarray
    x_axis: np.ndarray
    var_convergence: Optional[np.ndarray] = None


@dataclass
class UQResult:
    failure_probability: float
    estimation_variance: float
    mpp: Optional[np.ndarray] = None
    plot: Optional[ConvergencePlot] = None

    @property
    def safety_index(self):
        return -norm.ppf(self.failure_probability)


class GenericIntegrator(abc.ABC):
    """ Base class for an integrator

        This is used for inheritance only

    The following only available after calling the calc_fail_prob method with
    the post_proc=True. Limited support for num_parallel > 2.


    Look at the definition of calc_fail_prob in other integrators for further
    information.
    """

    def __init__(self, post_process: bool = False):
        self.post_process = False

    @abc.abstractmethod
    def integrate(self, limit_state_function: LimitStateFunction) -> UQResult:
        pass