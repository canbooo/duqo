# -*- coding: utf-8 -*-
"""
Created on Fri May 29 01:44:29 2020

@author: CanBo
"""
import pickle
import numpy as np
from scipy.spatial.distance import cdist

from duqo.uml.clustering import get_clusters, filter_tiny
from duqo.doe.lhs import inherit_lhs, optimize_doe, find_empty_bins, make_doe
from duqo.optimization.predict import read_integrator_name


def is_pareto(costs, return_mask=True):
    """
    Find the pareto-efficient points.
    Source: https://stackoverflow.com/a/40239615
    Source preferred over older custom implementation due to its performance.

    Parameters
    ----------
    costs : np.ndarray
        An (n_points, n_costs) array with objective values
    return_mask : bool
        True to return a mask instead of indexes
    Returns
    -------
        An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < costs.shape[0]:
        non_dominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        non_dominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[non_dominated_point_mask]  # Remove dominated points
        costs = costs[non_dominated_point_mask]
        next_point_index = np.sum(non_dominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


def set_obj_con_args(problem, models):
    try:
        problem.full_space.obj_arg = list(problem.full_space.obj_arg)
        # assumption, last argument is models
        problem.full_space.obj_arg[-1] = models
    except (TypeError, IndexError):
        problem.full_space.obj_arg = [models]

    try:
        problem.full_space.con_arg = list(problem.full_space.con_arg)
        # assumption, last argument is models
        problem.full_space.con_arg[-1] = models
    except (TypeError, IndexError):
        problem.full_space.con_arg = [models]
    return problem


def get_points_of_interest(problem, pareto, return_results=False):
    n_total_dims = int(problem.full_space.inp_space.dims)
    probas, moms = np.empty((0, n_total_dims)), np.empty((0, n_total_dims))
    objs, cons, fails = [], [], []
    for cand in pareto:
        res_pro, res_mom = problem.gen_post_proc(np.array(cand).reshape((1, -1)))

        if res_pro:
            names = [read_integrator_name(n)
                     for n in problem.co_fp.workers[::-1]]
            for name in names:
                if name in res_pro:
                    res_pro = res_pro[name]
                    break
            else:
                res_pro = res_pro[list(res_pro.keys())[-1]]
            inds = np.isfinite(res_pro["limit_state"]).all(1)
            if inds.any():
                probas = np.append(probas, res_pro["limit_state"][inds], axis=0)
            if res_pro["limit_state"].shape[0] < 10:
                inds = np.isfinite(res_pro["fail_samp"]).all(1)
                if inds.any():
                    probas = np.append(probas, select_samples(res_pro["fail_samp"][inds], 200),
                                       axis=0)
            if not res_mom:
                inds = np.isfinite(res_pro["safe_samp"]).all(1)
                if inds.any():
                    probas = np.append(probas,
                                       select_samples(res_pro["safe_samp"][inds], 200),
                                       axis=0)

            cons.append(res_pro["fail_prob"])
            fails.append((problem.co_fp.target_fail_prob - cons[-1]) < 0)
        if res_mom:
            moms = np.append(moms, res_mom["input"], axis=0)
            objs.append(res_mom["output"])
    x_f = np.append(moms, probas, axis=0)
    x_f = x_f[np.isfinite(x_f).all(1)]
    if objs:
        objs = np.array(objs)
    if cons:
        cons, fails = np.array(cons), np.array(fails)
        safe = np.logical_not(fails)
        pareto, cons = np.array(pareto)[safe], cons[safe]
        if np.size(objs) > 1:
            objs = objs[safe]
    if return_results:
        return x_f, pareto, objs, cons
    return x_f, pareto


def select_samples(samples, max_sample=200):
    res = samples[[0], :]
    while res.shape[0] < min(samples.shape[0], max_sample):
        res = np.append(res, most_distant_sample(samples, res), axis=0)
    return res


def most_distant_sample(choose_from, existing):
    return choose_from[[cdist(choose_from, existing).min(1).argmax()], :]


def square_to_condensed(i, j, n):
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) // 2 + i - 1 - j


def get_pdist_row(i_samp, n_samp):
    return [square_to_condensed(i_samp, other, n_samp)
            for other in range(n_samp) if other != i_samp]


def min_dist_from_pdists(n_samp, pdists):
    scores = np.zeros(n_samp)
    for i_samp in range(n_samp):
        row = get_pdist_row(i_samp, n_samp)
        scores[i_samp] = np.min(pdists[row])
    return scores


def lolhr4rrdo(problem, lower, upper, optimizer, model_trainer, step_size,
               max_evals, start_doe=None, optimizer_kwargs=None, model_trainer_args=(), model_trainer_kwargs=None,
               doe_save_path="", init_bound_tol=1e-3, archive_save_path=""):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if model_trainer_kwargs is None:
        model_trainer_kwargs = {}
    # After generating an initial DoE
    # 0- train models with samples
    # 1- solve problem with optimizer
    # 2- Acquire pareto frontier + relating stoch points
    # 3- cluster pareto frontier
    # 4- Adapt clusters (aka. add new samples)
    # 5- Repeat 0-4 until termination
    if doe_save_path and not doe_save_path.endswith(".npy"):
        doe_save_path += ".npy"
    n_total_dims = int(problem.full_space.inp_space.dims)
    lower_doe, upper_doe = problem.full_space.inp_space.doe_bounds(init_bound_tol,
                                                                   lower,
                                                                   upper)
    if start_doe is None:
        start_doe = 10 * n_total_dims
    if isinstance(start_doe, int):
        cur_doe = make_doe(start_doe, lower_bound=lower_doe,
                           upper_bound=upper_doe)
    else:
        cur_doe = start_doe.copy()

    i_iter = 1
    models = None
    while True:
        if doe_save_path:
            np.save(doe_save_path, cur_doe)
        print(f"Iter. - {i_iter}, samples={cur_doe.shape[0]}")
        try:
            models = model_trainer(cur_doe, *model_trainer_args, **model_trainer_kwargs)
        except RuntimeError as exc:
            print("TrainingError:", exc)
            if models is None:
                raise exc
        print("Training complete")
        problem = set_obj_con_args(problem, models)
        results = optimizer.optimize(**optimizer_kwargs)
        print("Optimization complete")
        pareto = np.array([np.array(res.candidate) for res in results])
        if cur_doe.shape[0] >= max_evals:
            _, pareto, objs, cons = get_points_of_interest(problem, pareto, return_results=True)
            break
        fits = np.array([np.array(res.fitness) for res in results])
        inds = is_pareto(fits)
        print("total", inds.sum(), "of", len(fits), "candidates for adaption")
        step_size = max(1, min(step_size, max_evals - cur_doe.shape[0]))
        new_doe, pareto, objs, cons = rrdouaml_step(problem, pareto[inds], step_size, lower_doe,
                                                    upper_doe, cur_doe)
        if archive_save_path:
            with open(archive_save_path + "." + str(i_iter), "wb") as f:
                pickle.dump({"archive": pareto, "objs": objs, "cons": cons}, f)
        cur_doe = np.append(cur_doe, new_doe, axis=0)
        i_iter += 1

    return cur_doe, pareto, objs, cons


def rrdouaml_step(problem, pareto, step_size, lower_doe, upper_doe, cur_doe):
    x_f, pareto, objs, cons = get_points_of_interest(problem, pareto,
                                                     return_results=True)
    print(f"Got {x_f.shape[0]} points of interest")
    new_doe = adapt_doe(lower_doe, upper_doe, cur_doe, x_f, num_samples=step_size,
                        return_update_only=True, beta=2)
    print(f"Got doe with {new_doe.shape[0]} samples")
    return new_doe, pareto, objs, cons


def hypercube_size(lower, upper, n_sample):
    return (upper - lower) / n_sample


def adapt_doe(lower, upper, doe, x_lsf, x_fail=None, num_samples=4,
              return_update_only=True, **kwargs):
    """
    Proposes new samples given the old doe

    Parameters
    ----------
    lower : list or np.ndarray
        lower doe bound
    upper : list or np.ndarray
        upper doe bound.
    doe : np.ndarray
        Matrix of known samples with shape=(n_old_sample, n_input_dimensions)
    x_lsf : np.ndarray
        Matrix of samples predicted to be on the limit state with shape=(n_lsf_sample, n_input_dimensions)
    x_fail : np.ndarray or None
        Matrix of samples predicted to be in the failure region with shape=(n_fail_sample, n_input_dimensions).
        This is useful in case the number of samples in x_lsf is low. Default: None
    num_samples : int
         Number of new samples to include in the DoE
    return_update_only : bool
        If True, only the new points are returned, otherwise the points will be appended to the passed doe and the
        result is returned

    Returns
    -------
    x_cand : np.ndarray
        Either the matrix of candidate samples or the matrix of the existing and the candidate samples depending on
        the value of return_update_only. shape=(num_sample, n_dim) or (n_old_sample + num_sample, n_dim)


    """
    if x_fail is None:
        x_fail = np.empty((0, x_lsf.shape[1]))
    fails, all_bounds, points_per_class = get_cluster_bounds(lower, upper, x_fail, x_lsf, num_samples,
                                                             doe.shape[0])

    if fails is None:
        all_bounds = [{"lower": lower,
                       "upper": upper,
                       "ids": None}]  # just for completeness has no effect
    if not all_bounds:
        all_bounds = [{"lower": fails.min(0),
                       "upper": fails.max(0),
                       "ids": np.arange(fails.shape[0])}]  # just for completeness has no effect
    n_clust = len(all_bounds)
    print(f"{n_clust} clusters found on model with {doe.shape[0]} samples")

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
        if bounds["ids"] is not None and fails is not None:
            if len(bounds["ids"]) > 2:
                corr = np.corrcoef(fails[bounds["ids"]], rowvar=False)
                max_corr = np.max(np.abs(corr - np.eye(corr.shape[0])))
                msg += f": {len(bounds['ids'])} points with max. corr. {max_corr:.2f}\n"
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
        new_doe = np.append(new_doe,
                            cur_doe,
                            axis=0)
    if return_update_only:
        return new_doe[doe.shape[0]:]
    return new_doe


def get_cluster_bounds(lower, upper, fails, lsf, n_samples, n_old_samples):
    x_f, labels, class_names = get_clusters(fails, lsf, n_samples)
    if x_f is None:
        print("Failed to detect clusters")
        return None, [-1], [n_samples]
    hc_size = hypercube_size(lower, upper, n_samples + n_old_samples)

    if class_names is None:
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
        cur_lower = mu - delta
        cur_upper = mu + delta

        cur_lower = np.maximum(cur_lower, lower - sample * hc_size / 2)
        cur_upper = np.minimum(cur_upper, upper + sample * hc_size / 2)

        cluster_bounds.append({"lower": cur_lower,
                               "upper": cur_upper,
                               "ids": np.where(locs)[0]})
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
