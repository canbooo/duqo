from __future__ import print_function, division
import warnings
import numpy as np
from scipy import stats

from pyRDO.proba._integrator_base import GenericIntegrator, to_safety_index
from pyRDO.doe.lhs import make_doe


class SUSE(GenericIntegrator):
    """
    Subset simulation based on adaptive conditional sampling as proposed in

    1."MCMC algorithms for subset simulation"
       Papaioannou et al.
       Probabilistic Engineering Mechanics 41 (2015) 83-103.
    """

    # def __init__(self, multivariate, constraints, constraint_args=None,
    #              std_norm_to_orig=None, orig_to_std_norm=None):
    #
    #     super(SUSE, self).__init__(multivariate, constraints, constraint_args, std_norm_to_orig, orig_to_std_norm)

    def calc_fail_prob(self, init_doe=None, prob_tol=1e-9, num_subset_points: int = 1e3,
                       inter_prob: float = 0.1, max_subsets=50, post_proc: bool = False,
                       init_var=1., use_covariate=True, **kwargs):
        """ Estimate the probability of failure P(F)
        Parameters
        ----------

        prob_tol : float
            Defines the accuracy of the estimated failure probability in terms
            of number of total samples. Does not have an effect yet

        batch_size : int
            the maximum number of samples to be calculated in one call.
            If 0, the all samples are calculated at once, although note that
            for larger number of samples, memory errors are possible. To avoid
            this, set this to a large number, that your memory can handle.

        max_samples : int or None
            Maximum number of samples. If passed, this will override the
            estimation using CoV

        post_proc : bool
            If true, sampling points will be accumulated to the attributes
            x_lsf, x_safe and x_fail and also will return mpp, conv_mu, conv_var,
            conv_x


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

        use_covariate : bool
        if True, following reference is used for the estimation
            Abdollahi et.al. A refined subset simulation for the reliability analysis using the subset
            control variate, 2020

        Unlike other Integrators, this will generate the post processing only
        for the last batch.

        """
        if self._n_parallel != 1:
            print("A parallel implementation of subset simulation is missing.")
            print("setting num_parallel to 1")
            self._n_parallel = 1
        self._post_proc = post_proc
        num_subset_points = int(num_subset_points)
        assert 0 < inter_prob < 1

        # Generate initial population
        if init_doe is None:
            # mv_norm = stats.multivariate_normal(mean=np.zeros(self._n_dim)) # Why not an lhs
            # doe = mv_norm.rvs(num_subset_points)
            doe = make_doe(num_subset_points, [stats.norm() for _ in range(self._n_dim)], num_tries=10)
        else:
            doe = init_doe
            if doe.shape[0] != num_subset_points:
                warnings.warn(f"Mismatch between the passed doe shape {doe.shape} "
                              f"and num_subset_points ({num_subset_points}).")

        outputs = self.const_env_stdnorm(doe)
        num_seeds = int(np.ceil(num_subset_points * inter_prob))
        doe_cur, outputs_cur, g_cur = _get_worst_n(doe, outputs, num_seeds)
        if g_cur <= 0:  # Failure region has already been reached
            fails = outputs < 0
            fail_prob = np.mean(fails)
            fail_prob_var = _subset_cov(fail_prob, doe.shape[0])
            safety_index = to_safety_index(fail_prob)
            if post_proc:
                mpp, conv_mu, conv_var, conv_x = self._gen_post_proc(fails)
                return fail_prob, fail_prob_var, safety_index, mpp, conv_mu, conv_var, conv_x
            return fail_prob, fail_prob_var, safety_index, None
        alphas = []
        subset_counter = 0
        fail_probs = [inter_prob]
        deltas = [_subset_cov(inter_prob, num_subset_points)]
        lamb = 0.6  # recommended initial value for lambda
        while g_cur > 0 and subset_counter <= max_subsets:
            # np.random.shuffle(doe_cur)
            doe_cur, outputs_cur, lamb = parallel_adaptive_conditional_sampling(doe_cur, outputs_cur, num_subset_points,
                                                                                self.const_env_stdnorm, g_cur, lamb,
                                                                                init_var)
            doe_cur = doe_cur.reshape(-1, self._n_dim)
            doe_cur, outputs_next, g_next = _get_worst_n(doe_cur, outputs_cur.ravel(), num_seeds)
            # print(g_cur, g_next, outputs_cur.min(), outputs_cur.std())
            indicators = outputs_cur < g_next
            fail_probs.append(indicators.mean())
            # print(fail_probs)
            if use_covariate:
                fail_probs_old = (outputs_cur < g_cur).mean()
                if fail_probs_old == fail_probs[-1] or fail_probs_old == 0:
                    alphas.append(1)
                else:
                    alphas.append(fail_probs[-1] / fail_probs_old)
            g_cur = np.copy(g_next)
            gamma = _corr_factor_gamma(indicators, fail_probs[-1])
            # print(gamma)
            deltas.append(_subset_cov(fail_probs[-1], num_subset_points, gamma))
            outputs_cur = outputs_next.copy()
            subset_counter += 1
            if g_cur == 0:
                break

        deltas = np.array(deltas)
        if use_covariate:
            fail_prob = np.prod(alphas) * fail_probs[0]
        else:
            fail_prob = np.prod(fail_probs)
        fail_prob_var = np.sum(deltas ** 2) * fail_prob ** 2
        safety_index = to_safety_index(fail_prob)
        if post_proc:
            mpp, conv_mu, conv_var, conv_x = self._gen_post_proc()
            return fail_prob, fail_prob_var, safety_index, mpp, conv_mu, conv_var, conv_x
        return fail_prob, fail_prob_var, safety_index, None

        # fail_prob = fail_prob / total_samples
        # safety_index = -stats.norm._ppf(fail_prob)
        # mpp = None
        # if post_proc:
        #     mpp, conv_mu, conv_var, conv_x = self._gen_post_proc(fails)
        #     return fail_prob, fail_prob_var, safety_index, mpp, conv_mu, conv_var, conv_x
        # return fail_prob, fail_prob_var, safety_index, mpp

    def _gen_post_proc(self, n_conv=100):
        """ Generate post processing. Will only process the last batch
        Use calc_fail_prob(..., post_proc=True) instead of this.
        """
        means = [marg.mean() for marg in self.margs]
        means = np.array(means).reshape((1, -1))
        d_best = np.inf
        mpp = np.inf * np.ones_like(means)
        if self.x_lsf.size > 0:
            distance = np.sum((means - self.x_lsf) ** 2, axis=1)
            loc = np.argmin(distance)
            d_best = distance[loc]
            mpp = self.x_lsf[loc, :].reshape((1, -1))
        if self.x_fail.size > 0:
            distance = np.sum((means - self.x_fail) ** 2, axis=1)
            loc = np.argmin(distance)
            if distance[loc] < d_best:
                mpp = self.x_fail[loc, :].reshape((1, -1))

        # if n_conv < 1:
        #     n_conv = fails.shape[0]
        # n_win = fails.shape[0] // n_conv
        # conv_x = np.array([n_win*i_conv for i_conv in range(1, n_conv + 1)])
        # conv_x[-1] = fails.shape[0]
        # conv_mu = np.zeros(n_conv)
        # conv_var = np.zeros(conv_mu.shape[0])
        # for k in range(1, n_conv+1):
        #     curr_fails = fails[:(k + 1)*n_win]
        #     conv_mu[k-1] = np.mean(curr_fails)
        #     conv_var[k-1] = np.var(curr_fails, ddof=1)
        conv_mu, conv_var, conv_x = None, None, None
        return mpp, conv_mu, conv_var, conv_x


def _get_worst_n(inputs, outputs, n_worst):
    order = np.argsort(outputs)
    g_cur = outputs[order[n_worst - 1]]
    if g_cur <= 0:
        g_cur = 0
        return inputs[order], outputs[order], g_cur
    ids = order[:n_worst]
    doe_cur = inputs[ids]
    outputs_cur = outputs[ids]
    return doe_cur, outputs_cur, g_cur


def _subset_cov(fail_prob, num_samples, gamma=0):
    if fail_prob > 0:
        return np.sqrt((1 - fail_prob) * (1 + gamma) / num_samples / fail_prob)
    return np.inf


def _corr_factor_gamma(indicator, fail_prob):
    """ Compute auto correlation factor of mcmc chains using the
    regular matrix indicator=g<0
    TODO: Check corr_factor_beta in UQpy.Reliability.py on https://github.com/SURGroup/UQpy
    """
    n_samples, n_chains = indicator.shape
    r = np.zeros(n_samples)
    pf_2 = fail_prob ** 2
    for lag in range(n_samples):
        if lag:
            r[lag] = (indicator[:-lag] * indicator[lag:]).mean()
        else:
            r[lag] = np.maximum((indicator**2).mean(), 1e-12)
    r -= pf_2
    r = r[1:] / r[0]
    scales = (1 - np.arange(1, n_samples) / n_samples)
    return 2 * np.sum(scales * r)


def parallel_adaptive_conditional_sampling(seeds, performances, num_samples, limit_state_fun, limit_state_value,
                                           lambda_prev=0.6, init_var="auto"):
    num_chains, num_dims = seeds.shape
    num_chain_samples = int(np.floor(num_samples / num_chains))
    inputs = np.zeros((num_chain_samples + 1, num_chains, num_dims))
    outputs = np.zeros((num_chain_samples + 1, num_chains))
    inputs[0] = seeds
    outputs[0] = performances.ravel()
    lambda_cur = np.copy(lambda_prev)
    accepts = np.zeros_like(outputs)
    num_adapts = int(0.1 * np.ceil(num_chain_samples))
    hat_a = []  # average acceptance rate of the chains

    star_a = 0.44
    if init_var == "auto":
        stds = np.std(seeds, axis=0, ddof=1)
    else:
        stds = np.ones(num_dims)
        if init_var is not None and not np.any(init_var == 0):
            stds *= init_var
    stds = np.repeat(stds.reshape((1, -1)), num_chains, 0)

    def get_sigma_rho(lamb):
        sig = np.minimum(lamb * stds, np.ones_like(stds))  # Ref. 1 Eq. 23
        return sig, np.sqrt(1 - sig ** 2)

    sigma, rho = get_sigma_rho(lambda_cur)
    i_adapt = 0
    # mu_acc = 0
    # print("Chains", num_chains, "g_max", limit_state_value, "seeds", seeds)
    # print(num_chain_samples, num_chains)
    for i_sample in range(num_chain_samples):
        candidates = np.random.normal(loc=rho * inputs[i_sample], scale=sigma)
        performance = limit_state_fun(candidates).ravel()
        improves = performance <= limit_state_value
        inputs[i_sample + 1, improves] = candidates[improves]
        outputs[i_sample + 1, improves] = performance[improves]
        accepts[i_sample + 1] = improves
        not_improves = np.logical_not(improves)
        inputs[i_sample + 1, not_improves] = inputs[i_sample, not_improves]
        outputs[i_sample + 1, not_improves] = outputs[i_sample, not_improves]
        # print("rho", rho, "sigma", sigma, f"accepts = {accepts[i_sample + 1]}")
        # print("cur_xs", inputs[i_sample + 1], "cur_gs", outputs[i_sample + 1])

        # average of the accepted samples for each seed 'mu_acc'
        # here the warning "Mean of empty slice" is not an issue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_acc = np.minimum(1, np.mean(accepts[:i_sample + 2]))
            # print("mu_acc", mu_acc)
        if i_sample and np.mod(i_sample, num_adapts) == 0:
            # c. evaluate average acceptance rate
            # print("adapting lamb", i_adapt, i_sample, num_adapts)
            lambda_prev = np.copy(lambda_cur)
            # d. compute new scaling parameter
            zeta = 1 / np.sqrt(i_adapt + 1)
            lambda_cur = np.exp(np.log(lambda_prev) + zeta * (mu_acc - star_a))  # Ref. 1 Eq. 26
            # print("lambda", lambda_cur)
            sigma, rho = get_sigma_rho(lambda_cur)
            # print("sigma", sigma, "rho", rho)
            i_adapt += 1

    # compute mean acceptance rate of all chains
    # if i_adapt != 0:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         accept_rate = np.mean(hat_a[:i_adapt - 1])
    # else:  # no adaptation
    #     accept_rate = mu_acc
    return inputs[1:], outputs[1:], lambda_cur
