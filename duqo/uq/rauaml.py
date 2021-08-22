# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:41:27 2020

@author: CanBo
"""
from __future__ import print_function, division
from copy import deepcopy
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import pandas as pd

from ..doe.lhs import inherit_lhs, optimize_doe, find_empty_bins, make_doe
from .integrator import GenericIntegrator
from .ds import DS
from .mc import MC
from duqo.uq.suse import SUSE
from duqo.misc.clustering import memory_safe_get_clusters
from sklearn.neighbors import NearestNeighbors

def _get_default_integrator(num_dims, prob_tol):
    if prob_tol > 1e-5:
        print("Using MC")
        return MC
    if num_dims <= 5:
        print("Using DS")
        return DS
    print("Using SUSE")
    return SUSE


def _is_converged_tscore(tolerance, pf, pf_std, pfs, n_samps, *args, **kwargs):
    if pfs and pf > 0:
        res = stats.ttest_ind_from_stats(pfs[-1]["mu"],
                                         pfs[-1]["sigma"],
                                         n_samps,
                                         pf, pf_std, n_samps)
        conv = res.pvalue > tolerance
        if conv:
            print("Early convergence due to t-test score", res.pvalue)
        return conv, res.pvalue
    return False, 0


# def _is_converged_reltscore(tolerance, pf, pf_std, pfs, n_samps, *args, **kwargs):
#     if pfs and pf > 0:
#         res = stats.ttest_ind_from_stats(pfs[-1]["mu"],
#                                          pfs[-1]["sigma"],
#                                          n_samps,
#                                          pf, pf_std, n_samps)
#         conv = res.pvalue > tolerance
#         if conv:
#             print("Early convergence due to t-test score", res.pvalue)
#         return conv, res.pvalue
#     return False, 0

def _is_converged_reldif(tolerance, pf, pf_std, pfs, *args, **kwargs):
    if pfs and pf > 0:
        diff = np.abs(pf - pfs[-1]["mu"]) / pf
        conv = diff < tolerance
        if conv:
            print("Early convergence due to relative difference", diff)
        return conv, diff
    return False, np.inf


class RAuAML(GenericIntegrator):
    """
    Reliability assessment using adaptive machine learning
    
    This class implements a surrogate based reliability analysis method using
    the method from [1]. 
    """

    def integrate(self, num_parallel=2, post_processing=True, probability_tolerance=1e-8, **kwargs):
        """
        Model list output from model_trainer will be passed as the last argument
        to the passed limit state functions, which should be in form
        
        def fun(x, *args, models):
            y = [model.predict(x) for model in models]
            ...
            return lsf

        Parameters
        ----------
        probability_tolerance : TYPE, optional
            DESCRIPTION. The default is 4.
        num_samples : int, optional
            DESCRIPTION. The default is 4.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if start_doe is None:
            start_doe = max(2 * len(self.mulvar), 10)
            # num_start = 4
        if isinstance(start_doe, int):
            lower, upper = self.mulvar.quantile_bounds(prob_tol)
            margs = [stats.uniform(l, u - l) for l, u in zip(lower, upper)]
            start_doe = make_doe(start_doe, margs)
        if start_doe.shape[0] < 2:
            raise ValueError("Start doe has too few samples")
        cur_doe = start_doe.copy()
        if integrator is None:
            integrator = _get_default_integrator(cur_doe.shape[1], prob_tol)
        n_samps = cur_doe.shape[0]
        pfs = []
        base_args = deepcopy(self.const_args)
        iteration = 0
        if convergence_test == "t-test":
            conv_checker = _is_converged_tscore
            tol = 0.95
            if integrator == MC:
                tol = 0.9
        else:
            conv_checker = _is_converged_reldif
            tol = 0.01
            if integrator == MC:
                tol = 0.1
        while n_samps <= max_evals:
            print()
            print("Starting model update...")
            models = model_trainer(cur_doe, *model_trainer_args)
            print("Model update complete")
            self.const_args = [args + [models] for args in base_args]
            (pf,
             pf_var,
             inter) = self.model_fail_prob(cur_doe, integrator=integrator,
                                           prob_tol=prob_tol,
                                           ttest=convergence_test == "t-test",
                                           **kwargs)

            pf_std = np.sqrt(pf_var)
            n_samps = cur_doe.shape[0]
            isconv, score = conv_checker(tol, pf, pf_var, pfs, n_samps)

            msg = f"Iter. {iteration} {n_samps} samp.- P(F): {pf:.4e} "
            if pfs and pf > 0:
                msg += f"rel. change: {(pfs[-1]['mu'] - pf) / pf:.4f} "
                if convergence_test != "t-test":
                    score = -score
                msg += f"score: {score:.4f}"
            print(msg)
            iteration += 1
            pfs.append({"mu": pf, "sigma": pf_std})

            if isconv:
                # Message printed during isconv check
                break
            if max_evals - n_samps < probability_tolerance:
                print("Maximum allowed number of iterations has been reached.")
                break
            lower, upper = self.mulvar.get_probability_bounds(prob_tol)
            cur_doe = adapt_doe(lower, upper, cur_doe, inter.x_lsf, inter.x_fail,
                                num_samples=probability_tolerance, prob_tol=prob_tol,
                                return_update_only=False,
                                **kwargs)
        self.const_args = base_args
        return pf, pf_var, cur_doe, pfs, models

    def model_fail_prob(self, doe, integrator=None, prob_tol=1e-6,
                        ttest=True, **kwargs):
        if integrator is None:
            integrator = _get_default_integrator(doe.shape[1], prob_tol)
        inter = integrator(self.mulvar, self.constraints,
                           constraint_args=self.const_args,
                           std_norm_to_orig=self.u2x,
                           orig_to_std_norm=self.x2u)

        if integrator == MC:
            prob_tol = prob_tol * kwargs.get("CoV", 0.1) ** -2
        if ttest:
            kwargs["converge"] = False
        fail_prob, fail_prob_var = inter.integrate(num_parallel=1, probability_tolerance=prob_tol,
                                                   multi_region=True, post_proc=True, **kwargs)[:2]
        return fail_prob, fail_prob_var, inter


def hypercube_size(lower, upper, n_sample):
    return (upper - lower) / n_sample


def adapt_doe(lower, upper, doe, x_lsf, x_fail=None, num_samples=4, prob_tol=1e-6,
              return_update_only=True, beta=2, **kwargs):
    """
    Proposes new samples given the old doe 

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    doe : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if x_fail is None:
        x_fail = np.empty((0, x_lsf.shape[1]))
    fails, all_bounds, points_per_class = get_cluster_bounds(lower, upper, x_fail, x_lsf, num_samples,
                                                             doe.shape[0], beta=beta)

    if fails is None:
        all_bounds = [{"lower": lower,
                       "upper": upper,
                       "inds": None}]  # just for completeness has no effect
    if not all_bounds:
        all_bounds = [{"lower": fails.min(0),
                       "upper": fails.max(0),
                       "inds": np.arange(fails.shape[0])}]  # just for completeness has no effect
    n_clust = len(all_bounds)
    print(f"{n_clust} clusters found on model with {doe.shape[0]} samples")
    # points_per_class = [num_samples // n_clust] * n_clust
    # order = np.argsort([b["count"] for b in all_bounds])[::-1]
    # i_order = 0
    # while sum(points_per_class) < num_samples:
    #     i_clus = order[i_order]
    #     points_per_class[i_clus] += 1
    #     i_order = (i_order + 1) % n_clust
    new_doe = doe.copy()
    for i_class, (bounds, samples) in enumerate(zip(all_bounds, points_per_class)):
        n_bins = samples
        n_empty = 0
        while n_empty < samples:
            empty_bins = find_empty_bins(new_doe, n_bins, bounds["lower"],
                                         bounds["upper"])
            n_empty = np.max(empty_bins.sum(0))
            n_bins += 1
        msg = f"Adapting class {i_class + 1} with {samples} samples"
        corr = 0
        if bounds["inds"] is not None and fails is not None:
            if len(bounds["inds"]) > 2:
                corr = np.corrcoef(fails[bounds["inds"]], rowvar=False)
                max_corr = np.max(np.abs(corr - np.eye(corr.shape[0])))
                msg += f": {len(bounds['inds'])} points with max. corr. {max_corr:.2f}\n"
            relative_size = np.array(bounds["upper"]) - np.array(bounds["lower"])
            # print("upper, lower", upper, lower, bounds["upper"], bounds["lower"])
            relative_size /= (np.array(upper) - np.array(lower))
            mu = (np.array(bounds["upper"]) + np.array(bounds["lower"])) / 2
            msg += f"Rel. size: {relative_size}\n"
            msg += f"Center: {mu}"
        print(msg)
        x_new = inherit_lhs(samples, empty_bins, bounds["lower"],
                            bounds["upper"])

        cur_doe = optimize_doe(x_new, corr_mat=corr, doe_old=new_doe)
        # msg = ""
        # if bounds["inds"] is not None and fails is not None:
        # corr = np.corrcoef(cur_doe, rowvar=False)
        # max_corr =np.max(np.abs(corr- np.eye(corr.shape[0])))
        # msg += f"max. corr: {max_corr:.2f}" 
        new_doe = np.append(new_doe,
                            cur_doe,
                            axis=0)
        # min_dist = np.min(pdist(new_doe))
        # msg += f"min. dist.: {min_dist:.2e}"
        print(msg)
    if return_update_only:
        return new_doe[doe.shape[0]:]
    return new_doe


def get_cluster_bounds(lower, upper, fails, lsf, n_samples, n_old_samples, beta=2):
    x_f, labels, class_names = get_clusters(fails, lsf, n_samples)
    if x_f is None:
        # Get global bounds in the main
        # This is useful for not omitting correlation
        print("Failed to detect clusters")
        return x_f, [-1], [n_samples]
    hc_size = hypercube_size(lower, upper, n_samples + n_old_samples)

    if labels is None:
        labels = np.zeros(x_f.shape[0])
        class_names = [0]
    class_names, counts = filter_tiny(class_names, labels)
    points_per_class = assign_points_per_class(n_samples, len(class_names), counts)
    cluster_bounds = []
    for label, count, sample in zip(class_names, counts, points_per_class):
        locs = labels == label
        cur_fails = x_f[locs]
        mu = (cur_fails.max(0) + cur_fails.min(0)) / 2
        sigma = (cur_fails.max(0) - cur_fails.min(0)) / 2
        delta = np.maximum(sigma, sample * hc_size / 2)
        # print("delta is", delta, sample * hc_size)
        cur_lower = mu - delta
        cur_upper = mu + delta
        # OLD LOGIC
        cur_lower = np.maximum(cur_lower, lower - sample * hc_size / 2)
        cur_upper = np.minimum(cur_upper, upper + sample * hc_size / 2)
        # crit = (cur_upper - cur_lower) / (upper - lower)
        # if np.max(crit) < DIMTOL:
        #     print(f"The largest relative bound {crit.max()} is smaller than the tolerance {DIMTOL}")
        #     continue
        cluster_bounds.append({"lower": cur_lower,
                               "upper": cur_upper,
                               "inds": np.where(locs)[0]})
        print("Appended bounds for cluster", label + 1)
        print("Lower", cluster_bounds[-1]["lower"])
        print("Upper", cluster_bounds[-1]["upper"])
    return x_f, cluster_bounds, points_per_class


def assign_points_per_class(n_samples, n_clust, counts):
    points_per_class = [n_samples // n_clust] * n_clust
    order = np.argsort(counts)[::-1]
    i_order = 0
    while sum(points_per_class) < n_samples:
        i_clus = order[i_order]
        points_per_class[i_clus] += 1
        i_order = (i_order + 1) % n_clust
    return points_per_class


def filter_tiny(class_names, labels):
    names, counts = [], []
    for label in class_names:
        locs = labels == label
        count = locs.sum()
        if count >= 5:
            counts.append(count)
            names.append(label)
    return names, counts


def get_clusters(fails, lsf, max_num_clusters, base_tol=None, max_points=None):
    if base_tol is None:
        base_tol = 1e-6
    if fails.size + lsf.size < 1:
        return None, None, None

    if max_points is None:
        n_dim = fails.shape[1]
        if n_dim <= 10:
            max_points = 25000
        elif n_dim <= 25:
            max_points = 10000
        else:
            max_points = 5000

    # x_f, counts = _filter_points(fails, lsf, None)
    x_f, counts = get_n_points(fails, lsf, max_points)
    # dist_size = calc_size_gb(x_f.shape[0])
    # print(dist_size, x_f.shape)
    try:
        labels, uniques = memory_safe_get_clusters(x_f, sample_weight=True, counts=counts,
                                                   max_num_clusters=max_num_clusters)
    except (SystemError, MemoryError):
        return None, None, None
    return x_f, labels, uniques


def _filter_points(fails, lsf, tol, target_num):
    # print("init _filter_points", fails.shape, lsf.shape)
    x_f = np.empty((0, fails.shape[1]))
    counts = np.empty(0)
    if lsf.size > 0:
        x_f, counts = _get_unique_tol(lsf, tol, target_num)
    # print(x_f.shape, counts.shape)
    if x_f.shape[0] < 6 and fails.size > 0:  # because 3 core points
        unique_xf, counts2 = _get_unique_tol(fails, tol, target_num)
        x_f = np.append(x_f, unique_xf, axis=0)
        counts = np.append(counts, counts2)
    # x_f, counts = _get_unique_tol(x_f, tol) # In case a sample was repeated in both after rounded tolerance

    valid = np.isfinite(x_f).all(1)
    # print(x_f.shape, valid.shape, counts.shape)
    x_f = x_f[valid, :]
    counts = counts[valid]
    # print("final _filter_points", x_f.shape, counts.min(), counts.max())
    return x_f, counts

# def get_n_means(fails, lsf, n_means=25000):
#     orig, counts = _filter_points(fails, lsf, None, None)
#     if orig.shape[0] <= n_means:
#         return orig, counts
#     n_neighbors = orig.shape[0] // n_means
#     neigh = NearestNeighbors(n_neighbors=n_neighbors)
#     nbrs = neigh.fit(orig)
#     distances, indices = nbrs.kneighbors(orig)


def get_n_points(fails, lsf, n_points=25000, conv_tol=0.1):
    def check_tol(n_points_curr):
        return abs((n_points - n_points_curr) / n_points) <= conv_tol

    def obj(tol):
        x_f = _filter_points(fails, lsf, 10**tol, n_points)[0]
        if check_tol(x_f.shape[0]):
            # print("returning", 0)
            return 0
        # print(tol, n_points - x_f.shape[0])
        return n_points - x_f.shape[0]

    def get_interval():
        start = -10
        # print("Getting start")
        while obj(start) > 0:
            start -= 10
            # print(start)
        stop = 10
        # print("Getting stop")
        while obj(stop) < 0:
            stop += 10
            # print(stop)
        return start, stop
    # print("filtering without tol")
    orig, counts = _filter_points(fails, lsf, None, None)
    if orig.shape[0] <= n_points or check_tol(orig.shape[0]):
        return orig, counts
    del orig, counts
    # print("Getting interval")
    a, b = get_interval()
    # print("Starting brentq")
    res = brentq(obj, a, b)
    return _filter_points(fails, lsf, 10**res, None)


def _get_unique_tol(points, tol=None, target_num=None):
    if tol is None:
        return np.unique(points, axis=0, return_counts=True)
    rounded = np.round(points / tol)
    # res, counts = np.unique(rounded, axis=0, return_counts=True)
    # final = np.zeros_like(res)
    df = pd.DataFrame(np.c_[rounded, points])
    df = df.groupby(list(df.columns)[:rounded.shape[1]])
    final = df.agg("mean").values
    counts = df.size().values
    # ids = df.indices
    # n_ids = len(ids)
    # if target_num is None or n_ids > 2 * target_num or n_ids < 0.5 * target_num:
    #     return np.empty((n_ids, points.shape[1])),
    # final, counts = [], []
    # for k, ids in ids.items():
    #     # ids = np.all(r == rounded, axis=1)
    #     final.append(_get_group_stat(points, ids))
    #     counts.append(len(ids))
    # print(counts.shape, counts)
    return final, counts.ravel()

def _get_group_stat(points, ids, agg=np.median):
    return agg(points[ids], axis=0)