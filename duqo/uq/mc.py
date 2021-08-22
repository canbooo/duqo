# -*- coding: utf-8 -*-
"""
Crude Monte Carlo simulation

Created on Fri Feb 22 15:01:06 2019

@author: Bogoclu
"""
import warnings
from typing import Union, Optional
import numpy as np


from .lsf import LimitStateFunction
from .integrator import GenericIntegrator, UQResult


class MC(GenericIntegrator):
    """
    Crude Monte Carlo simulation using batches for the integration of the
    probability of failure. Batches are recommended especially for low probabilities
    of failure for avoiding memory problems.

    Parameters
        ----------

        probability_tolerance : float
            Target estimation coefficient of variation i. e. E[P(F)]/STD(P(F)). Is used to
            estimate the maximum number of samples as well as for the convergence
            criteria is converge is True


        Returns
        -------
        fail_prob_mu : float
            estimation of the expected probability of failure

        fail_prob_var : float
            estimation variance of the probability of failure

        Following are only retuned if post_proc = True

        safety_index :
            Safety index, also known as the sigma level. It is equal to
            Phi_inv(1-fail_prob_mu), where Phi_inv is the inverse of the CDF
            of standard normal distribution


        mpp : 2-D numpy.ndarray
            Most probable point of failure among the used samples. It may
            slightly differ if calculated with optimization directly since
            no additional samples are generated to find it. If you need this
            use the mpp module

        conv_mu : numpy.ndarray
            y-axis values of the convergence plot of the estimation of expected
            probability of failure.

        conv_var :numpy.ndarray
            y-axis values of the convergence plot of the variance of the estimation.

        conv_x : numpy.ndarray
            x-axis of the convergence plots.

        Unlike other Integrators, this will generate the post processing only
        for the last batch.

    """
    def __init__(self, post_process, doe: Optional[np.ndarray] = None, probability_tolerance: float = 1e-8,
                 estimation_cov_limit: float = 0.1, max_mc_samples: Optional[int] = None, batch_size: Optional[int] = None,
                 converge: bool = True, verbose=0):
        super(MC, self).__init__(post_process)
        self.doe = doe
        self.probability_tolerance = probability_tolerance
        self.estimation_cov_limit = estimation_cov_limit
        self.sample_limit = int(np.ceil((estimation_cov_limit**-2) / self.probability_tolerance))
        if max_mc_samples is not None:
            self.sample_limit = max_mc_samples
        batch_size = batch_size or 0
        if batch_size < 1 or batch_size > self.sample_limit:
            batch_size = self.sample_limit
        else:
            batch_size = int(batch_size)
        self.batch_size = batch_size
        self.converge = converge
        self.verbose = verbose

    def integrate(self, limit_state_function: LimitStateFunction):
        """ Estimate the probability of failure P(F)

        """
        lsf = limit_state_function.to_sp()
        is_corr = lsf.multivariate.is_correlated
        corr_mat = lsf.multivariate.moment_trans()[0]
        fails = None
        if self.doe is not None:
            fails = lsf(self.doe) < 0
            fail_prob = np.mean(fails)
            fail_prob_var = np.var(fails, ddof=1)
        else:
            total_samples = 0
            fail_prob = 0
            fail_prob_var = 0
            fp_vars, batch_sizes = np.empty(0), np.empty(0)
            batch_size = self.batch_size
            while total_samples < self.sample_limit:
                remaining_samples = self.sample_limit - total_samples
                if remaining_samples < self.batch_size:
                    batch_size = remaining_samples
                doe = np.zeros((self.batch_size, self._n_dim), dtype=np.float64)
                for i_dim in range(self._n_dim):
                    doe[:, i_dim] = self.margs[i_dim].rvs(batch_size)
                # Check because multiplication takes time
                if is_corr:
                    doe = np.dot(doe, corr_mat)
                total_samples += batch_size
                fails = self.const_env(doe) < 0
                # Sum now divide later
                fail_prob += np.sum(fails)
                # Estimate total variance
                fp_vars = np.append(fp_vars, np.var(fails))
                # ddof==1 and sum(wi)==1 so w_i = n_i - 1
                batch_sizes = np.append(batch_sizes, batch_size)
                fail_prob_var = np.sum(fp_vars * batch_size / (batch_sizes.sum() - 1))
                if self.verbose:
                    print(f"{total_samples:.4e}, samples computed")
                if self.converge and fail_prob > 0:
                    cov = np.sqrt(fail_prob_var) / (fail_prob / total_samples)
                    if cov <= self.estimation_cov_limit:
                        break
            fail_prob = fail_prob / total_samples
        return self.to_uq_result(fail_prob, fail_prob_var)


    def _gen_post_proc(self, fails, n_conv=100):
        """ Generate post processing. Will only process the last batch
        Use calc_fail_prob(..., post_proc=True) instead of this.
        """
        safety_index = to_safety_index(fail_prob)
        mpp = np.empty((0, self._n_dim))
        if post_proc:
            mpp, conv_mu, conv_var, conv_x = self._gen_post_proc(fails)
        #     return fail_prob, fail_prob_var, safety_index, mpp, conv_mu, conv_var, conv_x
        # return fail_prob, fail_prob_var, safety_index, mpp
        if fails is None or len(fails) < 1:
            warnings.warn("No samples were computed with the current settings. Post processing elements are empty.")
            return np.empty((0, self._n_dim)), np.empty(0), np.empty(0), np.empty(0)
        means = [marg.mean() for marg in self.margs]
        means = np.array(means).reshape((1, -1))
        d_best = np.inf
        mpp = np.inf*np.ones_like(means)
        if self.x_lsf.size > 0:
            distance = np.linalg.norm(means - self.x_lsf, axis=1)
            loc = np.argmin(distance)
            d_best = distance[loc]
            mpp = self.x_lsf[loc, :].reshape((1, -1))
        if self.x_fail.size > 0:
            distance = np.linalg.norm(means - self.x_fail, axis=1)
            loc = np.argmin(distance)
            if distance[loc] < d_best:
                mpp = self.x_fail[loc, :].reshape((1, -1))

        if n_conv < 1:
            n_conv = fails.shape[0]
        n_win = fails.shape[0] // n_conv
        conv_x = np.array([n_win * i_conv for i_conv in range(1, n_conv + 1)])
        conv_x[-1] = fails.shape[0]
        conv_mu = np.zeros(n_conv)
        conv_var = np.zeros(conv_mu.shape[0])
        for k in range(1, n_conv+1):
            curr_fails = fails[:(k + 1)*n_win]
            conv_mu[k-1] = np.mean(curr_fails)
            conv_var[k-1] = np.var(curr_fails, ddof=1)
        return mpp, conv_mu, conv_var, conv_x
