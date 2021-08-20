# -*- coding: utf-8 -*-
"""
Stochastic modeling for univariate and multivariate random variables

Currently using scipy.stats and own implementation of Nataf. If you want to
use an own MultiVariate model, take a look at these classes and inherit/overwrite
the correponding methods.

Created on Tue Jul 30 12:48:25 2019

@author: Bogoclu
"""
from copy import deepcopy
import typing

import numpy as np
from scipy.linalg import eigh, cholesky

from ..doe.lhs import make_doe
from .get_margs import sp_margs
from .copula import NatafTransformation


def _get_corr_mat(corr_transform, n_dim):
    """ Input check for the arguments passed to DirectionalSimulator"""
    if corr_transform is None:
        return np.eye(n_dim)
    if not isinstance(corr_transform, np.ndarray) or corr_transform.ndim < 2:
        err_msg = "corr_transform must be a 2-D numpy array"
        raise ValueError(err_msg)
    if corr_transform.shape[0] != n_dim:
        err_msg = "Inconsistent number of marginal distributions and "
        err_msg += "corr_transform shape"
        raise ValueError(err_msg)
    if corr_transform.shape[0] != corr_transform.shape[1]:
        err_msg = "corr_transform must be square"
        raise ValueError(err_msg)
    if not (corr_transform == corr_transform.T).all():
        err_msg = "corr_transform must be symmetrical"
        raise ValueError(err_msg)
    return corr_transform


class UniVar:
    """
    Database object for a marginal distribution.
    This will be used to generate distribution objects from various sources.
    Currently, it only support scipy

    Parameters
    ----------
    name : str
        name of the distribution, best as used in scipy dists but some
        checks get made to also accept e.g. normal for norm

    mean : float
        mean value, ignored for uniform and bernoulli. Also, if
        lower_bound and upper_bound is passed for uniform, this will
        be recalculated from it. Otherwise, the bounds will be calculated
        from this and std

    std : float
        standard deviation, see mean for behaviour in uniform and bernoulli.

    lower_bound : float
        lower bound, ignored for unbounded distributions.

    upper_bound : float
        upper bound, ignored for unbounded distributions
    CoV : bool of float
        Decides if the coefficient of variation should be used for further
        transformations in rdo/rbdo. If true self.CoV will be computed as
        ```
        CoV = mean/std
        ```
        otherwise it will be None. If passed as float, CoV overrides std.
    """

    def __init__(self, name: str = "norm", mean: float = 0.0, std: float = 1.0,
                 lower_bound: float = None, upper_bound: float = None,
                 CoV: typing.Union[bool, float] = False, params: tuple = (), move_bounds=True,
                 ):
        self.move_bounds = move_bounds
        self.name = name.lower()
        self._mu = mean

        if std <= 0:
            raise ValueError("Standard deviation std must be strictly positive, got {std}")
        self._std = std
        if self.name == "uniform" and None not in [lower_bound, upper_bound]:
            self._mu = (lower_bound + upper_bound) / 2
            self._std = np.sqrt((upper_bound - lower_bound) ** 2 / 12)
        self._var_coef = None
        if CoV:
            self.var_coef = self._std / self._mu
            if isinstance(CoV, (int, float)):
                self.var_coef = CoV
            if self._std < 0:
                self.var_coef = -CoV

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
            lower_bound = self._mu - 1e9 * self._std
        if upper_bound is None:
            upper_bound = self._mu + 1e9 * self._std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = params
        # to get the resulting ones
        self._get_moments()

    @property
    def var_coef(self):
        return self._var_coef

    @var_coef.setter
    def var_coef(self, value):
        if value is None:
            self._var_coef = None
            return
        self._var_coef = value
        self._std = self._mu * self.var_coef

    @property
    def marg(self):
        """get a list of scipy.stats distributions"""
        return sp_margs(self)

    def _get_moments(self):
        dis = self.marg
        self._mu = dis.mean()
        self._std = dis.std()

    @property
    def mean(self):
        """get mean of the dsit"""
        return self._mu

    @mean.setter
    def mean(self, mean_new):
        """set new mean"""
        mu_dist = mean_new - self._mu
        self._mu = mean_new
        if self.move_bounds:
            self.lower_bound += mu_dist
            self.upper_bound += mu_dist

    @property
    def std(self):
        """get standard deviation of all variables"""
        return self._std

    @std.setter
    def std(self, std_new):
        """set new mean and new standard deviation"""
        if std_new <= 0:
            raise ValueError("Standard deviation std must be strictly positive")
        sig_ratio = std_new / self._std
        self._std = std_new
        if self.move_bounds:
            self.lower_bound = (self.lower_bound - self._mu) * sig_ratio + self._mu
            self.upper_bound = (self.upper_bound - self._mu) * sig_ratio + self._mu

    @property
    def var(self):
        """get standard deviation of all variables"""
        return self._std ** 2

    @var.setter
    def var(self, var_new):
        """get standard deviation of all variables"""
        self.std = np.sqrt(var_new)

    @property
    def moments(self):
        """ Mean and Standard deviation"""
        return self._mu, self._std

    @moments.setter
    def moments(self, moments):
        """set new mean and new standard deviation"""
        self.std = moments[1]
        self.mean = moments[0]

    def __repr__(self):
        if self.name == "uniform":
            msg = f"UniVar: {self.name} - LB: {self.lower_bound:.2e}  "
            msg += f"UB: {self.upper_bound:.2e}"
            return msg
        return f"UniVar: {self.name} - Mean: {self.mean:.2e}  Std. Dev: {self.std:.2e}"


class MultiVar:
    """
    A Multivariate stochastic model.

    Parameters
    ----------

    univars : list
        A list of Distribution instances

    corr_mat : np.ndarray
        The correlation matrix. If not passed, identity matrix is assumed.
    """

    def __init__(self, univars: list, corr_mat: np.ndarray = None):
        self.dists = univars
        self.rho = _get_corr_mat(corr_mat, len(self.dists))
        self._read_moments()

    @property
    def names(self):
        """ Names of the distributions"""
        return [dist.name for dist in self.dists]

    def __len__(self):
        return len(self.names)

    @names.setter
    def names(self, new_names):
        """ Names of the distributions"""
        if len(new_names) != len(self.dists):
            raise ValueError("Number of names does not match the number of distributions.")

        for name, dist in zip(new_names, self.dists):
            dist.name = name
        self._read_moments()

    def _read_moments(self):
        self._mu = np.array([d.mean for d in self.dists]).ravel()
        self._std = np.array([d.std for d in self.dists]).ravel()

    @property
    def margs(self):
        """Get marginal distributions"""
        return [dist.marg for dist in self.dists]

    def upper_quantile(self, prob_tol):
        return [dist.ppf(1. - prob_tol) for dist in self.margs]

    def lower_quantile(self, prob_tol):
        return [dist.ppf(prob_tol) for dist in self.margs]

    def quantile_bounds(self, prob_tol):
        """
        Quantile bounds of marginal distribution
     
        Parameters
        ----------
    
        prob_tol : float
            probability tolerance to use. For symmetric
            alpha-level confidence bounds,
            prob_tol == (1 - alpha) / 2
        """
        lower, upper = [], []
        for dist in self.margs:
            lower.append(dist.ppf(prob_tol))
            upper.append(dist.ppf(1 - prob_tol))
        return lower, upper

    @property
    def cov_inds(self):
        """Get the indexes of the random variables, that are defined over CoV"""
        return [dist.var_coef is not None for dist in self.dists]

    @property
    def var_inds(self):
        """Get the indexes of the random variables, that are defined over CoV"""
        return [not res for res in self.cov_inds()]

    @property
    def mean(self):
        """Get input means, thus the design point"""
        return self._mu

    @mean.setter
    def mean(self, new_mean):
        """Get input means, thus the design point"""
        for new_mu, dist in zip(new_mean, self.dists):
            dist.mean = new_mu
        self._read_moments()

    @property
    def var(self):
        """Get input variations"""
        return self._std ** 2

    @var.setter
    def var(self, new_var):
        for sigmasq, dist in zip(new_var, self.dists):
            dist.var = sigmasq
        self._read_moments()

    @property
    def std(self):
        """Get input std_devs"""
        return self._std

    @std.setter
    def std(self, new_std):
        for sigma, dist in zip(new_std, self.dists):
            dist.std = sigma
        self._read_moments()

    @property
    def is_corr(self):
        """Return True if any correlation is nonzero"""
        return not np.array_equal(self.rho, np.eye(self.rho.shape[0]))

    def transform_mats(self, verbose=False):
        """Matrices for the linear correlation transformation

        Returns
        -------

        corr_trans : np.ndarray
            To be used for transforming from the uncorrelated space to the
            correlated space

        inv_corr_trans : np.ndarray
            To be used for transforming from the correlated space to the
            uncorrelated space
        """
        try:
            corr_trans = cholesky(self.rho, lower=True)
        except np.linalg.LinAlgError:
            if verbose:
                print('Cholesky factorization failed.')
                print('Continuing with PCA.')
            eivals, eivecs = eigh(self.rho)
            corr_trans = np.dot(np.diag(np.sqrt(eivals)), eivecs.T)
        try:
            inv_corr_trans = np.linalg.inv(corr_trans)
        except np.linalg.LinAlgError:
            if verbose:
                print('linalg.inv failed.')
                print('Continuing with linalg.pinv.')
            inv_corr_trans = np.linalg.pinv(corr_trans)
        return corr_trans, inv_corr_trans

    def nataf_mats(self, verbose=False):
        """Matrices for the Nataf transformation
            Returns
            -------

            corr_trans : np.ndarray
                To be used for transforming from the uncorrelated space to the
                correlated space

            inv_corr_trans : np.ndarray
                To be used for transforming from the correlated space to the
                uncorrelated space
        """
        copula = NatafTransformation(self.margs, rho_x=self.rho,
                                     verbose=verbose)
        return copula.corr_transform, copula.inv_corr_transform

    def opt_lhs(self, num_points: int = 100, lower_bounds=None, upper_bounds=None,
                num_iters: int = 5000):
        """ Get an orthogonal sampling design of experiments for the MultiVariate

        Parameters
        ----------
        num_points : int
            number of samples to be generated

        lower_bounds : None or iterable
            if passed, the points in the generated DoE will be moved so that
            each dimension is larger than lower_bounds

        lower_bounds : None or iterable
            if passed, the points in the generated DoE will be moved so that
            each dimension is smaller than upper_bounds

        num_iters : int
            number of iterations to optimize the DoE for distance and target
            correlation criteria
        """
        return make_doe(num_points, self.margs, corr_mat=self.rho,
                        num_tries=num_iters, lower_bound=lower_bounds,
                        upper_bound=upper_bounds)

    def new(self, mean: typing.Union[list, np.ndarray] = None, std: typing.Union[list, np.ndarray] = None,
            inds: list = None):
        """
        Return new MultiVariate model with the original distribution families
        with different means and standard deviations as long as these were
        submitted
        """
        dists = deepcopy(self.dists)

        if inds:
            dists = [dists[ind] for ind in inds]
            rho = self.rho[inds, :][:, inds]
        else:
            rho = self.rho

        for i_var, dist in enumerate(dists):
            if std is not None:
                dist.std = std[i_var]
            if mean is not None:
                dist.mean = mean[i_var]
        return MultiVar(dists, rho)

    def standard_lhs(self, num_points: int = 100, num_iters: int = 5000,
                     lower_bounds=None, upper_bounds=None):
        """Returns get_lhs with mean=0 and std=1"""
        mus = np.zeros(self._mu.shape[0])  
        sigs = np.ones(self._std.shape[0])  
        tmp_mv = self.new(mus, sigs)
        return tmp_mv.opt_lhs(num_points, lower_bounds, upper_bounds, num_iters)

    def __repr__(self):
        return "MultiVar: " + " | ".join([str(d) for d in self.dists])

    def __str__(self):
        n_list = [dist.name for dist in self.dists]
        names = " | ".join(n_list)
        msg = f"{len(n_list)} dimensional multivariate random model\n Distributions: \n"
        msg += f"{names}\n uses nataf copula to simulate independency\n"
        msg += f"Means: {self.mean}\n Std. Devs. {self.std} \n"
        return msg
