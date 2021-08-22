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
from typing import Union, List, Iterable, Optional

import numpy as np

from scipy.linalg import eigh, cholesky
from scipy.stats import norm

from ..doe.lhs import make_doe
from .univariate import UniVariate
from .copula import NatafTransformation


class MultiVariate:
    """
    A Multivariate stochastic model.

    Parameters
    ----------

    marginals : List[UniVariate]
        A list of UniVar instances

    corr_mat : np.ndarray
        The correlation matrix. If not passed, identity matrix is assumed.
    """

    def __init__(self, marginals: Iterable[UniVariate], corr_mat: Optional[np.ndarray] = None) -> None:
        self.marginals = marginals
        self.rho = _get_corr_mat(corr_mat, len(self.marginals))
        self.copula = NatafTransformation(self.marginals, rho_x=self.rho)
        self._reset()

    @property
    def names(self) -> List[str]:
        """ Names of the distributions"""
        return [dist.name for dist in self.marginals]

    @names.setter
    def names(self, new_names: List[str]) -> None:
        """ Names of the distributions"""
        if len(new_names) != len(self.marginals):
            raise ValueError("Number of names does not match the number of distributions.")

        for name, dist in zip(new_names, self.marginals):
            dist.name = name
        self._reset()

    def __len__(self) -> int:
        return len(self.names)

    def _reset(self) -> None:
        self._mu = np.array([d.mean for d in self.marginals])
        self._std = np.array([d.std for d in self.marginals])

    @property
    def distributions(self):
        """Get marginal distributions"""
        return self.marginals

    def upper_quantile(self, probability: Union[int, float]) -> np.ndarray:
        return np.array([dist.upper_quantile(probability) for dist in self.marginals])

    def lower_quantile(self, probability: Union[int, float]) -> np.ndarray:
        return np.array([dist.lower_quantile(probability) for dist in self.marginals])

    def confidence_bounds(self, probability: Union[int, float]) -> List[np.ndarray]:
        """
        Confidence bounds of the marginal distributions
     
        """
        lower, upper = [], []
        for dist in self.marginals:
            l, b = dist.confidence_bounds(probability)
            lower.append(l)
            upper.append(b)
        return [np.array(lower), np.array(upper)]

    @property
    def variation_coefficient_mask(self) -> np.ndarray:
        """Get the indexes of the random variables, that are defined over CoV"""
        return np.array([dist.variation_coefficient is not None for dist in self.marginals], dtype=bool)

    @property
    def variance_mask(self) -> np.ndarray:
        """Get the indexes of the random variables, that are defined over CoV"""
        return np.array([not res for res in self.variation_coefficient_mask], dtype=bool)

    @property
    def mean(self) -> np.ndarray:
        """Get input means, thus the design point"""
        return self._mu

    @mean.setter
    def mean(self, new_mean: Union[List[Union[int, float]], np.ndarray]) -> None:
        """Get input means, thus the design point"""
        for new_mu, dist in zip(new_mean, self.marginals):
            dist.mean = new_mu
        self._reset()

    @property
    def var(self) -> np.ndarray:
        """Get input variations"""
        return self._std ** 2

    @var.setter
    def var(self, new_var: Union[List[Union[int, float]], np.ndarray]) -> None:
        for sigma2, dist in zip(new_var, self.marginals):
            dist.var = sigma2
        self._reset()

    @property
    def std(self) -> np.ndarray:
        """Get input std_devs"""
        return self._std

    @std.setter
    def std(self, new_std: Union[List[Union[int, float]], np.ndarray]) -> None:
        for sigma, dist in zip(new_std, self.marginals):
            dist.std = sigma
        self._reset()

    @property
    def is_correlated(self) -> bool:
        """Return True if any correlation is nonzero"""
        return not np.array_equal(self.rho, np.eye(self.rho.shape[0]))

    def moment_trans(self, verbose=False) -> List[np.ndarray]:
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
        return [corr_trans, inv_corr_trans]

    @property
    def nataf_trans(self) -> List[np.ndarray]:
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
        return [self.copula.corr_transform, self.copula.inv_corr_transform]

    def from_normal_space(self, std_norm_input: np.ndarray) -> np.ndarray:
        """
        Transforms input points from standard normal space to the multivariate space
        """
        corr_input = np.dot(std_norm_input, self.copula.corr_transform)
        if corr_input.ndim < 2:  # assuming single dimensional arrays to represent a single point
            corr_input = corr_input.reshape((1, -1))
        orig_input = np.zeros(corr_input.shape, dtype=np.float64)
        for k, dist in enumerate(self.marginals):
            orig_input[:, k] = dist.ppf(norm._cdf(corr_input[:, k]))
        return orig_input

    def to_normal_space(self, input_points: np.ndarray) -> np.ndarray:
        """
        Transforms input points from the multivariate space to standard normal space
        """
        ucorr_input = np.dot(input_points, self.copula.inv_corr_transform)
        if ucorr_input.ndim < 2:  # assuming single dimensionals to be a single point
            ucorr_input = ucorr_input.reshape((1, -1))
        std_norm_input = np.zeros(ucorr_input.shape, dtype=np.float64)
        for k, dist in enumerate(self.marginals):
            std_norm_input[:, k] = norm._ppf(dist.cdf(ucorr_input[:, k]))
        return std_norm_input

    def optimized_lhs(self, num_point: int = 100, num_iteration: int = 5000,
                      lower_bounds: Optional[Iterable[Union[int, float]]] = None,
                      upper_bounds: Optional[Iterable[Union[int, float]]] = None,
                      ) -> np.ndarray:
        """ Get an orthogonal sampling design of experiments for the MultiVariate

        Parameters
        ----------
        num_point : int
            number of samples to be generated

        num_iteration : int
            number of iterations to optimize the DoE for distance and target
            correlation criteria

        lower_bounds : Optional[Iterable[Union[int, float]]]
            if passed, the points in the generated DoE will be moved so that values in
            each dimension is larger than lower_bounds

        upper_bounds : Optional[Iterable[Union[int, float]]]
            if passed, the points in the generated DoE will be moved so that values in
            each dimension is smaller than upper_bounds


        """
        return make_doe(num_point, self.marginals, corr_mat=self.rho,
                        num_tries=num_iteration, lower_bound=lower_bounds,
                        upper_bound=upper_bounds)

    def new(self, mean: Union[List[Union[int, float]], np.ndarray] = None,
            std: Union[List[Union[int, float]], np.ndarray] = None,
            ids: Optional[List[int]] = None):
        """
        Return new MultiVariate model with the original distribution families
        with different means and standard deviations as long as these were
        submitted. If ids are passed, only the distributions at those indexes will be used
        """
        dists = deepcopy(self.marginals)

        if ids:
            dists = [dists[ind] for ind in ids]
            rho = self.rho[ids, :][:, ids]
        else:
            rho = self.rho

        for i_var, dist in enumerate(dists):
            if std is not None:
                dist.std = std[i_var]
            if mean is not None:
                dist.mean = mean[i_var]
        return MultiVariate(dists, rho)

    def standard_normal_lhs(self, num_point: int = 100, num_iteration: int = 5000,
                            lower_bounds: Optional[Iterable[Union[int, float]]] = None,
                            upper_bounds: Optional[Iterable[Union[int, float]]] = None, ):
        """Returns get_lhs with mean=0 and std=1"""
        mus = np.zeros(self._mu.shape[0])
        sigs = np.ones(self._std.shape[0])
        tmp_mv = self.new(mus, sigs)
        return tmp_mv.optimized_lhs(num_point, lower_bounds, upper_bounds, num_iteration)

    def __repr__(self) -> str:
        return "MultiVar(" + " ,".join([str(d) for d in self.marginals]) + ")"


def _get_corr_mat(corr_transform: Optional[Union[np.ndarray, int, float]], n_dim: int) -> np.ndarray:
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
