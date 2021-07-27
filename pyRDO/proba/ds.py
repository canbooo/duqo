# -*- coding: utf-8 -*-
"""
Directional simulation using Fekete points and Nataf copula

Created on Tue Apr  4 23:44:58 2017

@author: Bogoclu
"""

import multiprocessing as mp
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from joblib import Parallel, delayed

from ..doe.hyperspace_division import fekete_points
from ._integrator_base import GenericIntegrator, to_safety_index


def _sanity_check_calc(multi_region, num_presearch, prob_tol, num_parallel,
                       post_proc):
    """
    Input check for the arguments passed to
    calculate_failure_probability
    """
    if not isinstance(multi_region, bool):
        err_msg = "multi_region must be of type bool "
        err_msg += "got: " + type(multi_region)
        raise ValueError(err_msg)
    if not isinstance(prob_tol, float):
        err_msg = "prob_tol must be a floating number, "
        err_msg += "got: " + type(prob_tol)
        raise ValueError(err_msg)
    if (not isinstance(num_parallel, int)) or num_parallel < 1:
        err_msg = "num_parallel must be a positive integer defining "
        err_msg += "the number of parallel execution, got:" + num_parallel
        raise ValueError(err_msg)
    if (not isinstance(num_presearch, int)) or num_presearch < 1:
        err_msg = "num_presearch must be a positive integer defining "
        err_msg += "the number of parallel execution, got:" + num_presearch
        raise ValueError(err_msg)
    if not isinstance(post_proc, bool):
        err_msg = "post_proc must be of type bool "
        err_msg += "got: " + type(post_proc)
        raise ValueError(err_msg)


def _sane_fekete_points(directions, n_dim):
    """
    get fekete points for DirectionalSimulator object.
    use get_directions function for other use cases.
    """

    if directions is None:
        n_dir = n_dim * 80
    elif isinstance(directions, int):
        n_dir = directions
    else:
        try:
            n_dir, n_dim_dir = directions.shape
        except AttributeError:
            err_msg = "Only an integer or a numpy array is accepted as "
            err_msg += "directions."
            raise TypeError(err_msg)

        if n_dim != n_dim_dir:
            err_msg = "Number of dimensions of the directions does not "
            err_msg += "match the number of marginal distributions"
            raise ValueError(err_msg)
        return directions
    return fekete_points(n_dim, n_dir, max_iters=100, tolerance=1e-12)


class DS(GenericIntegrator):
    r"""
    Compute the expected probability of failure as well as the estimation
    variance using directional sampling with Fekete points. As described
    in J. Nie and B. R. Ellingwood. “Directional methods for structural
    reliability analysis”. In: Structural Safety 22 (2000)

    The probability of failure P(F) is defined as the probability of
    exceedance for the minimum of the limit state functions,
    passed as the variable constraints. P(F) = P(min(constraints(x))<0)

    Joint probability of failure of all limit state functions constraints
    is calculated by finding the distance R_i to the region of failure in
    each direction in the standard normal space. Inverse transform sampling
    is used for transforming non-normal variables. For defining correlation
    a 2-D np.array corr_transform can be passed, that transforms the
    variables in the standard normal space. This can be computed i.e.
    by Nataf-Transformation.

    The conditional probability of failure for each direction
    D_i using the Chi^2 distribution can be computed as
        P(F|D_i) = 1 - chi2.cdf(R_i^2)
    And the mean probability of failure using
    P(F) = E[P(F|D_i)]
    With the assumption that Directions are uniformly distributed on a
    unit n-sphere for an n-dimensional problem. Note that the P(F) is
    computed for the lower envelope of all passed constraint functions.
    If individual probabilities of failure are required, each function
    must be analyzed with a new object.

    Furthermore this class can also compute the multipoint approximation
    for a more accurate calculation at the cost of computation time. Only
    recommended for very fast models. This method is not published yet,
    but I am planning it.

    Parameters
    ----------
    multivariate  :  a pyRDO.uncertainty.model.Multivariate object defining
                    the stochastic model to be used. If another object is passed
                    it must have the methods get_margs, which returns a list
                    of marginal distributions with pdf,cdf and ppf functions
                    as well as the method nataf_mats for acquiring the
                    linear transformation matrix for modelling the correlations

    constraints  :  list of constraint functions. Must accept numpy
                    matrix as input.


    const_args   :  list of additional arguments for the constraint
                    functions. each element of the list are omitted
                    to the constraints function after the calculation
                    point x. as in constraints[0](x,args). Will not
                    be passed if not provided, which is the default
                    behaviour.



    Attributes
    ----------
    x_safe : points sampled in the safe domain
             Only collected if post_proc is passed True to calc_failure_prob.
             Limited support for num_parallel > 2.

    x_fail : points sampled in the failure domain
             Only collected if post_proc is passed True to calc_failure_prob.
             Limited support for num_parallel > 2.

    x_lsf : zero-crossing points on the limit state
            Only collected if post_proc is passed True to calc_failure_prob.
            Limited support for num_parallel > 2.
    """

    def _get_start(self):
        start = 0
        zero_value = self._gr(start, 0)
        patience = 0
        while zero_value == 0 and patience < 1000:
            patience += 1
            start += 1e-4
            zero_value = self._gr(start, 0)  # just for numerical consistency
            # print("adjusting", zero_value)
        safe_design = False
        if zero_value > 0:
            safe_design = True
        return safe_design, zero_value

    def _r2u(self, radius, i_dir):
        """
        Converts the radius to a point in the standard normal space
        depending on the direction vector with the index i_dir
        """
        radius = np.array(radius)
        if radius.ndim < 2:
            radius = radius.reshape((-1, 1))
        direction = self.fekete[i_dir, :].reshape((1, -1))
        return (radius * direction).astype(np.float32)

    def _gr(self, radius, i_dir):
        """
        Transformed envelope of the constraint functions
        to the radius (distance) space of each dimension.
        """
        std_norm_var = self._r2u(radius, i_dir)
        return self.const_env_stdnorm(std_norm_var)

    def _get_zero_plane(self, multi_region, pf_min, n_presearch):
        """
        Get the zero plane by calculating the radii to the limit state
        in each direction. uses joblib for parallelization
        """
        r_max = np.sqrt(stats.chi2._ppf(1 - pf_min, df=self._n_dim))
        if not np.isfinite(r_max):
            raise ValueError("Requested probability tolerance is too high!")
        # Make presearch at once with arrays to speed up
        r_grid = np.linspace(r_max / n_presearch,
                             r_max, n_presearch).reshape((-1, 1))
        search_points = np.empty((self.fekete.shape[0] * n_presearch,
                                  self._n_dim))
        for i_dir in range(self.fekete.shape[0]):
            search_points[i_dir * n_presearch:(i_dir + 1) * n_presearch, :] = \
                self._r2u(r_grid, i_dir)

        searchs = self.const_env_stdnorm(search_points).reshape((self.fekete.shape[0], r_grid.shape[0]))

        if not np.isfinite(searchs).all():
            msg = "Infinity or NAN during the transformation."
            msg += "Try decreasing pf_min "
            msg += "or reviewing the marginal distributions."
            raise OverflowError(msg)

        # Now search the directions with failed designs
        if self._n_parallel == 1:
            radii = [self._get_radius(i_dir, r_grid, searchs[i_dir, :],
                                      multi_region)
                     for i_dir in range(self.fekete.shape[0])]
        else:
            with Parallel(n_jobs=self._n_parallel,
                          prefer="processes") as para:
                radii = para(delayed(self._get_radius)(i_dir, r_grid, searchs[i_dir, :],
                                                       multi_region)
                             for i_dir in range(self.fekete.shape[0]))
        return np.array(radii)

    def _optimize1d(self, r_start, r_stop, i_dir):
        """
        Calls the brent optimizer to find the region of failure,
        given r_start and r_stop so that gr(r_start)*gr(r_stop)<=0.
        """
        try:
            return brentq(self._gr, r_start, r_stop, args=(i_dir),
                          full_output=True)[0]
        except ValueError:
            return np.inf

    def _get_radius(self, i_dir, r_grid, search_results, multi_region):
        """
        Get the radius or radii to the limit state (cons_env==0)
        in a direction, given a direction index i_dir,
        the search grid r_grid as well as the search_results
        constraint_radi(r_grid).
        """
        r_zeros = []
        tries_max = 100
        positions = np.where(self._g0 * search_results < 0)[0]
        if positions.size < 1:
            r_zeros.append(np.inf)
        else:
            next_pos = positions[0]
            r_zeros.append(self._optimize1d(0, r_grid[next_pos], i_dir))

        # safe_design assures r_zeros[-1]>0
        if multi_region and self.safe_design and np.isfinite(r_zeros[-1]):
            while r_grid[-1] > r_zeros[-1]:
                r_start = 1.0001 * r_zeros[-1]
                g_start = self._gr(r_start, i_dir)
                zero_tol = 1e-15
                tries = 0
                while np.abs(g_start) < zero_tol and \
                        r_grid[-1] > r_start and tries < tries_max:
                    #                    print("infinity",r_start)
                    r_start *= 1.01
                    tries += 1
                    g_start = self._gr(r_start, i_dir)
                if r_start >= r_grid[-1] or tries >= tries_max:
                    break

                positions = np.where(g_start * search_results < 0)[0]
                next_pos = positions[positions > next_pos]
                if next_pos.size < 1:
                    break
                next_pos = next_pos[0]
                r_zeros.append(self._optimize1d(r_start, r_grid[next_pos],
                                                i_dir))

            # calculate the effective R
            directional_fail_prob = 1 - stats.chi2._cdf(np.array(r_zeros) ** 2, df=self._n_dim)
            signum = np.ones(directional_fail_prob.shape)
            signum[1::2] = -1
            pfd = np.sum(directional_fail_prob * signum)
            r_zeros = [np.sqrt(stats.chi2._ppf(1 - pfd, df=self._n_dim))]
        return r_zeros[0]

    def _get_conv_plot(self, fail_prob):
        conv_mu = np.zeros(self.fekete.shape[0] - 1)
        conv_var = np.zeros(conv_mu.shape[0])
        fail_prob_sort = np.sort(fail_prob)[::-1]
        for k in range(1, self.fekete.shape[0]):
            conv_mu[k - 1] = np.mean(fail_prob_sort[:k + 1])
            conv_var[k - 1] = np.var(fail_prob_sort[:k + 1], ddof=1)
        return conv_mu, conv_var, np.arange(1, conv_mu.shape[0] + 1)

    def _gen_post_proc(self, fail_prob, radii, prob_tol):
        """
        generates variables used for post processing given the
        probability of failure array with same number of entries as
        directions, the radii array
        """
        conv_mu, conv_var, conv_x = self._get_conv_plot(fail_prob)

        # Now make points for plot
        if self._n_parallel > 1:
            r_max = np.sqrt(stats.chi2._ppf(1 - prob_tol, df=self._n_dim))
            if self.safe_design:
                fail_inds = np.isfinite(radii)
                safe_inds = np.logical_not(fail_inds)
                radii[safe_inds] = r_max
            else:
                safe_inds = np.isfinite(radii)
                fail_inds = np.logical_not(safe_inds)
                radii[fail_inds] = r_max
            all_points = self.fekete * radii[:, np.newaxis]

            all_points = self.u2x(all_points)
            self.x_safe = all_points[safe_inds, :]
            self.x_fail = all_points[fail_inds, :]
            self.x_lsf = self.x_fail.copy()

        if self.x_lsf.shape[0] > 1:
            mpp_ind = np.argmin(np.linalg.norm(self.x2u(self.x_lsf), axis=1))
            mpp = self.x_lsf[[mpp_ind], :]
        else:
            mpp = np.empty((0, self._n_dim))
        return mpp, conv_mu, conv_var, conv_x

    def calc_fail_prob(self, multi_region=False, num_presearch=20, directions=None,
                       prob_tol=1e-8, num_parallel=2, post_proc=False,
                       verbose: int = 0, **kwargs):
        """
        Main function to call for calculating the probability of failure.
        As the problem is defined during the init, this only takes the
        computation arguments. The sampling points will not be collected, 
        if num_parallel > 1 or if post_proc is not True

        Parameters
        ----------

        multi_region : bool
            continue to search for safe and unsafe domains beyond the first
            found one. The estimation will be more accurate but will require
            much more samples. Using this without a fast model is not recommended,
            thus defaults to None

        n_presearch : int
            number of search points for each direction. Should be decreased for
            simpler functions. (minimum of 6 is recommended)

        directions : int or 2-D numpy.ndarray
            if None, fekete points with a default number of directions are
            produced. Note that this method becomes expensive for n_dim > 15
            if this is an integer, fekete points with that number of directions
            are computed. if this is a numpy array with the dimensions
            n_dirs x n_vars each row will be used as a direction vector.
            Will default to n_dim * 80 fekete points, However the relationship
            to n_dim is non linear for most examples. The default will work up
            to 5-6 dimensions depending on the problem. Higher dimensional problems
            require more directions for an accurate estimation.

        prob_tol : float
            Defines the accuracy of the estimated failure probability in terms
            of number of total samples

        post_proc : bool
            If true, sampling points will be accumulated to the attributes
            x_lsf, x_safe and x_fail and also will return mpp, conv_mu, conv_var,
            conv_x

        num_parallel : int
            number of parallel processes. Reduces performance for values more
            than 1 per core. Defaults to 2.

        Returns
        -------
        fail_prob_mu : float
            estimation of the expected probability of failure

        fail_prob_std : float
            estimation variance of the probability of failure

        Following are only retuned if post_proc = True

        safety_index : float
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

        conv_var : numpy.ndarray
            y-axis values of the convergence plot of the variance of the estimation.

        conv_x : numpy.ndarray
            x-axis of the convergence plots.
        """
        _sanity_check_calc(multi_region, num_presearch, prob_tol,
                           num_parallel, post_proc)

        self._post_proc = post_proc
        self.fekete = np.array(_sane_fekete_points(directions, self._n_dim),
                               dtype=np.float32)
        self.safe_design, self._g0 = self._get_start()
        n_cores = mp.cpu_count()  # Using at least one less core
        # is usually faster
        if num_parallel > n_cores or num_parallel < 1:
            if verbose > 0:
                msg = "Using more processes than cores slows down parallel "
                msg += f"computation because of GIL. Setting it to {n_cores}"
                print(msg)
            num_parallel = n_cores
        self._n_parallel = num_parallel
        if verbose > 0:
            print("Starting limit state search...")
        radii = self._get_zero_plane(multi_region, prob_tol, num_presearch)
        # Calculate the Probability of Failure for each direction
        fail_prob = 1 - stats.chi2._cdf(radii ** 2, df=self._n_dim)
        if not self.safe_design:
            fail_prob = 1 - fail_prob
        fail_prob_mu = np.mean(fail_prob)
        fail_prob_std = np.var(fail_prob, ddof=1)  # Since samples are chisquared var = std
        safety_index = to_safety_index(fail_prob_mu)

        mpp = None
        if post_proc:
            mpp, conv_mu, conv_var, conv_x = self._gen_post_proc(fail_prob, radii, prob_tol)
            return fail_prob_mu, fail_prob_std, safety_index, mpp, conv_mu, conv_var, conv_x
        return fail_prob_mu, fail_prob_std, safety_index, mpp
