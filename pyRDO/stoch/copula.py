# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:39:28 2015
Nataf Transformation

WARNING: VERY OLD CODE BUT SEEMS TO WORK SO FAR
@author: Bogoclu
Last Modified: 02.12.15
"""

import warnings
from scipy import stats
from scipy.linalg import eigh, cholesky
import numpy as np


def _force2d(arr_like):
    arr_like = np.array(arr_like)
    if arr_like.ndim == 1:
        return arr_like.reshape((1, -1))
    return arr_like


class NatafTransformation:
    """
    Defines the object for Nataf transformation.

    Input Parameters
    ----------------

        margs : Marginal distributions of the variables.
                Margs is a list of objects that have
                the methods pdf, cdf, ppf (inverse of cdf),
                mean and std i.e.
                [scipy.stats.norm(),scipy.stats.expon()]
                (Python List or Tuple of length n_vars)

        rho_x : Correlation matrix of the input variables
                (Tested with  Pearson correlation) If rho_x
                is set to None or not given, it will be
                set to identity matrix.
                ([n_vars x n_vars] numpy Array)

        verbose : If set to True, information about the
                   transformation will be printed (bool)

    Other Details
    -------------
        Notation
        --------
        n_vars - Number of variables
        n_samples - Number of samples
        prig_var - Input variable ([n_samples x n_vars] numpy Array)
        std_norm_Var - the standard normal variable
        ind_std_norm_var - independent standard normal variable.

        All variables are [n_samples x n_vars] numpy
        arrays. Transformation is written as Input2Output
        or InputOutput i.e. u2x transforms u to x and
        Jux is the gradient of this transformation

    Methods
    -------

        If there is no correlation, following methods
        are sufficient:

        x2u(x) : Converts arbitrarily distributed variable
                 x to standardnormal variable u

        u2x(u) : Converts standardnormal variable u to
                 arbitrarily distributed variable x

        If there is correlation, use:

        x2zunc(x) : Converts correlated and arbitrarily
                    distributed variable x to independent
                    standardnormal variable zunc

        zunc2x(zunc) : Converts independent standardnormal
                       variable zunc to correlated and
                       arbitrarily distributed variable x.

    Attributes
    ----------
        rho_u - Modified correlation matrix

        corr_transform - Gradient for converting zunc to u; cholesky
                         transformation of rho_u [n_vars x n_vars]

        inv_corr_transform - Gradient for converting u to zunc;
                             inverse of the cholesky transformation
                             of rho_u [n_vars x n_vars]
    """

    def __init__(self, margs, rho_x=None, verbose=False):
        """
        Initialise Model and solve Nataf Copula
        """
        self.margs = margs
        self.n_var = len(self.margs)

        if self.n_var < 2:
            err_msg = 'If there is only one variable,'
            err_msg += 'use inverse transform sampling.'
            raise ValueError(err_msg)
        if rho_x is None:
            rho_x = np.eye(self.n_var)
        else:
            if rho_x.shape[0] == self.n_var:
                rho_x = rho_x
            else:
                err_msg = 'Correlation matrix does not have the expected '
                err_msg += 'dimensions %i x %i .' % (self.n_var, self.n_var)
                raise ValueError(err_msg)
        self._corr_trans_nataf(rho_x, order=11, rho_tol=1e-6,
                               max_iters=30, verbose=verbose)

    def x2u(self, orig_var):
        """
        Converts arbitrarily distributed variable x
        to standardnormal variable u.

        Parameters
        ----------
        orig_var - Variable with the arbitrary initial marginal
                   distribution (margs), numpy array
                   [n_samples x n_vars]
        Returns
        -------
        std_norm_var - Standardnormal variable, numpy
                       array [n_samples x n_vars]
        """
        orig_var = _force2d(orig_var)
        std_norm_var = []
        for i in range(self.n_var):
            std_norm_var.append(stats.norm.ppf(self.margs[i].cdf(orig_var[:,
                                                                 i])))
        return np.asarray(std_norm_var).T

    def u2x(self, std_norm_var):
        """
        Converts standardnormal variable u
        to arbitrarily distributed variable x.

        Parameters
        ----------
        std_norm_var - Standardnormal variable, numpy
                       array [n_samples x n_vars]
        Returns
        -------
        orig_var - Variable with the initial marginal
                   distribution (Margs), numpy array
                   [n_samples x n_vars]
        """
        std_norm_var = _force2d(std_norm_var)
        orig_var = []
        for i in range(self.n_var):
            orig_var.append(self.margs[i].ppf(stats.norm.cdf(std_norm_var[:,
                                                             i])))
        return np.asarray(orig_var).T

    def x2zunc(self, orig_var):
        """
        Converts correlated and arbitrarily
        distributed variable x to independent
        standardnormal variable zunc

        Parameters
        ----------
        orig_var - Correlated variable with the initial
                   marginal distribution (Margs), numpy
                   array  [n_samples x n_vars]
        Returns
        -------
        ind_std_norm_var - independent standardnormal variable,
                           numpy array [n_samples x n_vars]
        """
        return np.dot(self.x2u(orig_var), self.inv_corr_transform)

    def zunc2x(self, ind_std_norm_var):
        """
        Converts independent standard normal
        variable zunc to correlated and
        arbitrarily distributed variable x.

        Parameters
        ----------
        ind_std_norm_var - independent standardnormal
                           variable, numpy array [n_samples x n_vars]
        Returns
        -------
        orig_var - Variable with the initial marginal
                   distribution (Margs) and Correlation
            (rho_x), numpy array [n_samples x n_vars]
        """
        return self.u2x(np.dot(ind_std_norm_var, self.corr_transform))

    def _corr_trans_nataf(self, rho_x_orig, order=11, rho_tol=1e-6,
                          max_iters=30, verbose=False):
        """
        Calculates the modified correlation matrix
        with Nataf Assumption.

        sets the attributes corr_transform and inv_corr_transform and rho_u
        """
        # sanity checks
        if order <= 1:
            err_msg = 'The specified integration order ' + str(order)
            err_msg += 'must be larer than 1!'
            raise ValueError(err_msg)
        rho_x_tol = rho_z_tol = rho_tol  # Doesn't have to be like this
        herm_coords, herm_weights = np.polynomial.hermite.hermgauss(order)
        u1_coords, u2_coords = np.meshgrid(herm_coords, herm_coords)
        u1_coords, u2_coords = np.sqrt(2) * u1_coords, np.sqrt(2) * u2_coords
        weights = np.dot(np.transpose([herm_weights]), [herm_weights])
        std_norm = stats.norm(0., 1.)
        z_mod_rho = np.eye(self.n_var)
        for row_no in range(self.n_var):
            for col_no in range(row_no):
                rho_x = rho_x_orig[row_no, col_no]
                rho_z = rho_x
                if (np.abs(rho_x) > 0.05 and np.abs(rho_x) < 0.99) \
                        and \
                        not (self.margs[row_no].dist.name == 'norm' and \
                             self.margs[col_no].dist.name == 'norm'):
                    iter_counter = 0
                    rho_x_acc = np.inf
                    rho_z_acc = np.inf
                    denom = np.pi * self.margs[row_no].std() * \
                            self.margs[col_no].std()
                    while iter_counter <= max_iters and \
                            (rho_x_acc > rho_x_tol or rho_z_acc > rho_z_tol):
                        rho_z_sqr = np.sqrt(1.0 - rho_z * rho_z)
                        z1_coords = u1_coords
                        z2_coords = rho_z * u1_coords + rho_z_sqr * u2_coords

                        # Transform into the initial distribution space
                        x1_coords = self.margs[row_no].ppf( \
                            std_norm.cdf(z1_coords))
                        x2_coords = self.margs[col_no].ppf( \
                            std_norm.cdf(z2_coords))
                        x1_stds = (x1_coords - self.margs[row_no].mean())
                        x2_stds = (x2_coords - self.margs[col_no].mean())

                        # Calculate the result of the integral as in C-Script
                        rho_x_new = np.sum(x1_stds * x2_stds * weights) / denom

                        # Calculate derivative
                        d_rho_x = (u1_coords - rho_z * u2_coords / rho_z_sqr)
                        d_rho_x *= std_norm.pdf(z2_coords) / \
                                   self.margs[col_no].pdf(x2_coords)
                        d_rho_x = np.sum(d_rho_x * weights * x1_stds) / denom

                        # Evaluate the new dZZCorr while making sure that
                        # is stays between [-1,+1]
                        rho_z_old = rho_z
                        rho_z = rho_z_old + (rho_x - rho_x_new) / d_rho_x
                        if np.abs(rho_z) > 1.0:
                            rho_z = 0.5 * (rho_z_old + np.sign(rho_z))

                        # Calculate the accuracies
                        rho_x_acc = np.abs(rho_x - rho_x_new)
                        rho_z_acc = np.abs(rho_z - rho_z_old)
                        iter_counter += 1

                    # Should this be an Error or a Warning?
                    if rho_x_acc > rho_x_tol or rho_z_acc > rho_z_tol:
                        err_msg = 'Optimization not converged for'
                        err_msg += 'variables' + str(row_no) + 'and'
                        err_msg += str(col_no) + '.'
                        warnings.warn(err_msg)
                z_mod_rho[row_no, col_no] = rho_z
        self.rho_u = z_mod_rho + np.tril(z_mod_rho, -1).T  # pylint: disable=no-member
        try:
            self.corr_transform = cholesky(self.rho_u, lower=False)
        except np.linalg.LinAlgError:
            if verbose:
                print('Cholesky factorization failed.')
                print('Continuing with PCA.')
            w_z, v_z = eigh(self.rho_u)
            self.corr_transform = np.dot(v_z, np.diag(np.sqrt(w_z))).T

        try:
            self.inv_corr_transform = np.linalg.inv(self.corr_transform)
        except np.linalg.LinAlgError:
            if verbose:
                print('linalg.inv failed.')
                print('Continuing with linalg.pinv.')
            self.inv_corr_transform = np.linalg.pinv(self.corr_transform)
