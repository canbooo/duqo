# -*- coding: utf-8 -*-
"""
Crude Monte Carlo simulation

Created on Fri Feb 22 15:01:06 2019

@author: Bogoclu
"""

import typing
import numpy as np


from ._integrator_base import GenericIntegrator, to_safety_index


class MC(GenericIntegrator):
    """
    Crude Monte Carlo simulation using batches for the integration of the
    probability of failure. Batches are recommended especially for low probabilities
    of failure for avoiding memory problems
    """

    def calc_fail_prob(self, prob_tol: float = 1e-4, mc_batch_size: typing.Union[int, float] = 1e5,
                       CoV: float = 0.1, max_mc_samples: typing.Union[int, float] = None,
                       post_proc: bool = False, doe=None,
                       converge: bool = True, **kwargs):
        """ Estimate the probability of failure P(F)
        Parameters
        ----------

        prob_tol : float
            Defines the accuracy of the estimated failure probability in terms
            of number of total samples

        mc_batch_size : int
            the maximum number of samples to be calculated in one call.
            If 0, the all samples are calculated at once, although note that
            for larger number of samples, memory errors are possible. To avoid
            this, set this to a large number, that your memory can handle.
        CoV : float
            Target estimation covariance i. e. E[P(F)]/STD(P(F)). Is used to
            estimate the maximum number of samples as well as for the convergence
            criteria is converge is True
        
        max_mc_samples : int or None
            Maximum number of samples. If passed, this will override the
            estimation using CoV
        
        post_proc : bool
            If true, sampling points will be accumulated to the attributes
            x_lsf, x_safe and x_fail and also will return mpp, conv_mu, conv_var,
            conv_x
        
        doe : None or 2d numpy array
            If passed, this will be used for the estimation instead of
            generating random samples
        
        converge : bool
            If True, a convergence check will be done after each batch. 
            Recommended for small probabilities of failure 

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
        is_corr = self.mulvar.is_corr
        self._post_proc = post_proc
        corr_mat = self.mulvar.transform_mats()[0]
        if doe is not None:
            fails = self.const_env(doe) < 0
            fail_prob = np.mean(fails)
            fail_prob_var = np.var(fails, ddof=1)
        else:
            sample_limit = int(np.ceil((CoV**-2)/prob_tol))
            if max_mc_samples is not None:
                sample_limit = max_mc_samples
            if mc_batch_size < 1 or mc_batch_size > sample_limit:
                mc_batch_size = sample_limit
            else:
                mc_batch_size = int(mc_batch_size)
            total_samples = 0
            fail_prob = 0
            fail_prob_var = 0
            fp_vars, batch_sizes = np.empty(0), np.empty(0)
            while total_samples < sample_limit:
                
                remaining_samples = sample_limit - total_samples
                if remaining_samples < mc_batch_size:
                    mc_batch_size = remaining_samples
                doe = np.zeros((mc_batch_size, self._n_dim), dtype=np.float64)
                for i_dim in range(self._n_dim):
                    doe[:, i_dim] = self.margs[i_dim].rvs(mc_batch_size)
                # Check because multiplication takes time
                if is_corr:
                    doe = np.dot(doe, corr_mat)
                total_samples += mc_batch_size
                fails = self.const_env(doe) < 0
                # Sum now divide later
                fail_prob += np.sum(fails)
                # Estimate total variance
                fp_vars = np.append(fp_vars, np.var(fails, ddof=1))
                # ddof==1 and sum(wi)==1 so w_i = n_i - 1
                batch_sizes = np.append(batch_sizes, mc_batch_size - 1)
                fail_prob_var = np.sum(fp_vars * mc_batch_size / batch_sizes.sum())
                if mc_batch_size < sample_limit:
                    print(f"{total_samples:.4e}, samples computed")
                if converge and fail_prob > 0 :
                    if fail_prob_var / fail_prob <= CoV:
                        break
            fail_prob = fail_prob / total_samples
        safety_index = to_safety_index(fail_prob)
        mpp = None
        if post_proc:
            mpp, conv_mu, conv_var, conv_x = self._gen_post_proc(fails)
            return fail_prob, fail_prob_var, safety_index, mpp, conv_mu, conv_var, conv_x
        return fail_prob, fail_prob_var, safety_index, mpp

    def _gen_post_proc(self, fails, n_conv=100):
        """ Generate post processing. Will only process the last batch
        Use calc_fail_prob(..., post_proc=True) instead of this.
        """
        means = [marg.mean() for marg in self.margs]
        means = np.array(means).reshape((1, -1))
        d_best = np.inf
        mpp = np.inf*np.ones_like(means)
        if self.x_lsf.size > 0:
            distance = np.sum((means - self.x_lsf)**2, axis=1)
            loc = np.argmin(distance)
            d_best = distance[loc]
            mpp = self.x_lsf[loc, :].reshape((1, -1))
        if self.x_fail.size > 0:
            distance = np.sum((means - self.x_fail)**2, axis=1)
            loc = np.argmin(distance)
            if distance[loc] < d_best:
                mpp = self.x_fail[loc, :].reshape((1, -1))

        if n_conv < 1:
            n_conv = fails.shape[0]
        n_win = fails.shape[0] // n_conv
        conv_x = np.array([n_win*i_conv for i_conv in range(1, n_conv + 1)])
        conv_x[-1] = fails.shape[0]
        conv_mu = np.zeros(n_conv)
        conv_var = np.zeros(conv_mu.shape[0])
        for k in range(1, n_conv+1):
            curr_fails = fails[:(k + 1)*n_win]
            conv_mu[k-1] = np.mean(curr_fails)
            conv_var[k-1] = np.var(curr_fails, ddof=1)
        return mpp, conv_mu, conv_var, conv_x
