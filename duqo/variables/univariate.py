# -*- coding: utf-8 -*-
"""
Implements a 1-D random variable as modelled by scipy distributions

Created on Tue Jul 30 12:48:25 2019

@author: Bogoclu
"""

from typing import Union, List, Iterable, Optional

import numpy as np
from scipy.stats.distributions import rv_frozen

from .get_scipy_dist import frozen_scipy_dist


class UniVariate:
    """
    A continuous uni variate distribution.


    Parameters
    ----------
    name : str
        name of the distribution, best as used in scipy dists but some
        checks get made to also accept e.g. normal for norm. Default: "norm"

    mean : float
        mean value, ignored for uniform and bernoulli. Also, if
        lower_bound and upper_bound is passed for uniform, this will
        be recalculated from it. Otherwise, the bounds will be calculated
        from this and std. Default: 0.

    std : float
        standard deviation, see mean for the behaviour in uniform. Default: 1.

    lower_bound : Optional[float]
        lower bound, ignored for unbounded distributions. Default: None

    upper_bound : Optional[float]
        upper bound, ignored for unbounded distributions. Default: None
    variation_coefficient : Optional[Union[bool, float]]
        Decides if the coefficient of variation should be used for further
        transformations in rdo/rbdo. If True, variation_coefficient will be computed from the passed
        mean and std as
        ```
        variation_coefficient = mean / std
        ```
        otherwise it will be None. If passed as float, variation_coefficient overrides std.
    params : Iterable
        Other parameters used in the scipy distribution object.

    move_bounds : bool
        If True, lower_bound and upper_bound will be moved upon setting mean or standard deviation, if these
        were passed.
    """

    def __init__(self, name: str = "norm", mean: Union[int, float] = 0., std: Union[int, float] = 1.,
                 lower_bound: Union[int, float, None] = None, upper_bound: Union[int, float, None] = None,
                 variation_coefficient: Union[bool, int, float, None] = None, params: Iterable = (),
                 move_bounds: bool = True,
                 ) -> None:
        self.move_bounds = move_bounds
        self.name = name.lower()
        self._mu = mean

        if std <= 0:
            raise ValueError(f"Standard deviation std must be strictly positive, got {std}.")
        self._std = std
        if self.name == "uniform" and None not in [lower_bound, upper_bound]:
            self._mu = (lower_bound + upper_bound) / 2
            self._std = np.sqrt((upper_bound - lower_bound) ** 2 / 12)
        self._variation_coefficient = None
        # noinspection PyTypeChecker
        self.variation_coefficient = variation_coefficient

        if self.name == "uniform":
            if lower_bound is None:
                # compute it from the mean and std_dev
                # sigma = ub - lb / sqrt(12), mu = (ub + lb) / 2
                # => lb = 2mu - ub => sigma = (ub - mu) / np.sqrt(3)
                # => lb, ub = mu +- np.sqrt(3) sigma
                lower_bound = mean - np.sqrt(3) * self._std
            if upper_bound is None:
                upper_bound = mean + np.sqrt(3) * self._std
            self._mu = (lower_bound + upper_bound) / 2
            self._std = np.sqrt((upper_bound - lower_bound) ** 2 / 12)

        # self.lower_bound, self.upper_bound = None, None
        if lower_bound is None:
            lower_bound = self._mu - 1e9 * self._std  # Just some numeric bound
        if upper_bound is None:
            upper_bound = self._mu + 1e9 * self._std  # same reason as lower_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = params
        # to get the resulting ones
        self._distribution = None
        self._reset()

    @property
    def variation_coefficient(self) -> Union[None, int, float]:
        return self._variation_coefficient

    @variation_coefficient.setter
    def variation_coefficient(self, new_variation_coefficient: Union[bool, int, float, None]) -> None:
        if not new_variation_coefficient:
            self._variation_coefficient = None
            return
        if isinstance(new_variation_coefficient, bool):
            new_variation_coefficient = self._mu / self._std

        if np.sign(self._mu * new_variation_coefficient) < 0:
            new_variation_coefficient = -new_variation_coefficient
        self._variation_coefficient = new_variation_coefficient
        self._std = self._mu * new_variation_coefficient
        self._reset()

    @property
    def distribution(self) -> rv_frozen:
        """get a list of scipy.stats distributions"""
        return self._distribution

    def upper_quantile(self, probability: Union[int, float]) -> Union[int, float]:
        """Return the value x, where CDF(x) = 1 - prob_tol"""
        return self.distribution.ppf(1. - probability)

    def lower_quantile(self, probability: Union[int, float]) -> Union[int, float]:
        """Return the value x, where CDF(x) = prob_tol"""
        return self.distribution.ppf(probability)

    def confidence_bounds(self, probability: Union[int, float]) -> List[Union[int, float]]:
        """Two sided confidence bounds x_l <= x <= x_u, where
                CDF(x_l) = prob_tol / 2
           and
                CDF(x_u) = (1 - prob_tol / 2)

        """
        return [self.lower_quantile(probability / 2), self.upper_quantile(probability / 2)]

    def _reset(self) -> None:
        self._distribution = frozen_scipy_dist(self)
        self._mu = self._distribution.mean()
        self._std = self._distribution.std()

    @property
    def mean(self) -> Union[int, float]:
        """get mean of the dsit"""
        return self._mu

    @mean.setter
    def mean(self, new_mean: Union[int, float]) -> None:
        """set new mean"""
        mu_dist = new_mean - self._mu
        self._mu = new_mean
        if self.move_bounds:
            self.lower_bound += mu_dist
            self.upper_bound += mu_dist
        self._reset()

    @property
    def std(self) -> Union[int, float]:
        """get standard deviation of all variables"""
        return self._std

    # noinspection PyTypeChecker
    @std.setter
    def std(self, new_std: Union[int, float]) -> None:
        """set new mean and new standard deviation"""
        if new_std <= 0:
            raise ValueError("Standard deviation std must be strictly positive")
        sig_ratio = new_std / self._std
        self._std = new_std
        if self.move_bounds:
            self.lower_bound = (self.lower_bound - self._mu) * sig_ratio + self._mu
            self.upper_bound = (self.upper_bound - self._mu) * sig_ratio + self._mu
        self._reset()

    @property
    def var(self) -> Union[int, float]:
        """get standard deviation of all variables"""
        return self._std ** 2

    @var.setter
    def var(self, new_var: Union[int, float]) -> None:
        """get standard deviation of all variables"""
        self.std = np.sqrt(new_var)

    @property
    def moments(self) -> Iterable[Union[int, float]]:
        """ Mean and Standard deviation"""
        return self._mu, self._std

    @moments.setter
    def moments(self, new_moments: List[Union[int, float]]) -> None:
        """Set new mean and new standard deviation passed in this order"""
        self.std = new_moments[1]
        self.mean = new_moments[0]

    def __repr__(self) -> str:
        if self.name == "uniform":
            return f"UniVar({self.name}, lb={self.lower_bound:.2e}, ub={self.upper_bound:.2e})"
        return f"UniVar({self.name}, mean={self.mean:.2e}, std={self.std:.2e})"

    def cdf(self, values: Union[int, float, np.ndarray, List[Union[int, float]]]) -> Union[float, np.ndarray]:
        """Cumulative density function"""
        return self._distribution._cdf(values)

    def ppf(self, values: Union[int, float, np.ndarray, List[Union[int, float]]]) -> Union[float, np.ndarray]:
        """Inverse cumulative distribution function"""
        return self._distribution._ppf(values)

    def pdf(self, values: Union[int, float, np.ndarray, List[Union[int, float]]]) -> Union[float, np.ndarray]:
        """Probability density function"""
        return self._distribution._pdf(values)
