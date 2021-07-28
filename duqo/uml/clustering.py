# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 03:02:14 2016

@author: Bogoclu
Needs:


"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, OPTICS, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors

from scipy.optimize import brentq


def dbscanner(dist_mat_or_points, eps, min_samples, sample_weight, metric="euclidean"):
    if min_samples < 3:
        min_samples = 3
    clus = DBSCAN(eps=max(1e-8, np.abs(eps)), min_samples=min_samples, metric=metric,
                  )
    clus.fit(dist_mat_or_points, sample_weight=sample_weight)
    return clus


def optics(dist_mat_or_points, eps, min_samples, metric="euclidean"):
    if min_samples < 3:
        min_samples = 3
    clus = OPTICS(max_eps=max(1e-8, np.abs(eps)), min_samples=min_samples, metric=metric,
                  cluster_method="xi", min_cluster_size=.1)
    clus.fit(dist_mat_or_points)
    return clus


def kmeans(x_f, n_clust, batch_size=None):
    if batch_size is None:
        clus = KMeans(n_clusters=n_clust)
    else:
        clus = MiniBatchKMeans(n_clusters=n_clust, batch_size=batch_size)
    return clus.fit(x_f)


def _reduce_to_kmeans(x_f, max_points=10000):
    if x_f.shape[0] <= 2 * max_points:
        return x_f
    n_clust = 2 * x_f.shape[1]
    clus = kmeans(x_f, n_clust)
    # means = clus.cluster_centers_
    # stds = clus.inertia_
    labels = np.unique(clus.labels_)
    res = np.empty((0, x_f.shape[1]))
    for lab in labels:
        ids = clus.labels_ == lab
        cur_points = ids.sum()
        cur_target_points = max_points * cur_points // x_f.shape[0]
        step = cur_points // cur_target_points
        res = np.append(res, x_f[ids][::step], axis=0)
    return res

    return res


def get_dbclusters(fails, sample_weight=True,
                   counts=None, max_num_clusters=np.inf):
    n_sample, n_dim = fails.shape
    if n_sample <= 1:
        return None, None
    sample_weight, avg_wgt = _get_sample_weights(sample_weight, fails, counts)
    scaler = StandardScaler()
    fails = scaler.fit_transform(fails)
    if fails.shape[1] > 100:
        metric = "cosine"
        dist_max = 1
    else:
        metric = "euclidean"

    min_samp = avg_wgt * min(2 * n_dim, n_sample // 1000)

    def _eval_db(eps):
        epsilon = eps
        if not np.isfinite(epsilon) and epsilon == 0:
            return [], None, 0, 0
        clus = dbscanner(fails, epsilon, min_samp, sample_weight, metric)
        # clus = optics(fails, epsilon, min_samp, metric)
        unique_labels, cnts = np.unique(clus.labels_, return_counts=True)
        unique_labels = unique_labels.tolist()
        if -1 in unique_labels:
            unique_labels.remove(-1)
            cnts = np.array(cnts[1:])
        if cnts.shape[0] < 1:
            return [], None, 0, 0
        success_rate = (n_sample - np.sum(clus.labels_ == -1)) / n_sample
        size_prop = cnts.min() / n_sample
        return clus.labels_, unique_labels, success_rate, size_prop

    def obj(epsilon):
        _, unique_labels, success_rate, size_prop = _eval_db(epsilon)
        n_classes = len(unique_labels) if unique_labels else 0
        ob1 = epsilon / dist_max
        ob2 = n_classes / max_num_clusters
        infeas = not n_classes or success_rate < 0.9 or size_prop < 0.1 or ob2 > 1
        if infeas:
            return -(success_rate + size_prop)
        return ob1 - ob2 - 3


    dist_min, dist_max = _get_eps_bounds(fails)
    res = minimize_scalar(obj, bounds=(dist_min, dist_max),
                          method="Bounded")
    labels, uniques, sr, sp = _eval_db(res.x)
    return labels, uniques


def _get_eps_bounds(x_f):
    neigh = NearestNeighbors(n_neighbors=1)
    nbrs = neigh.fit(x_f)
    distances, indices = nbrs.kneighbors(x_f)
    eps_min = np.percentile(distances, 0.1)
    eps_max = np.linalg.norm(x_f.max(0) - x_f.min(0))
    return eps_min, eps_max


def filter_tiny(class_names, labels):
    names, counts = [], []
    for label in class_names:
        locs = labels == label
        count = locs.sum()
        if count >= 5:
            counts.append(count)
            names.append(label)
    return names, counts


def get_clusters(fails, lsf, max_num_clusters, max_points=None):
    if fails.size + lsf.size < 5:
        print("No points passed to cluster")
        return None, None, None

    if max_points is None:
        n_dim = fails.shape[1]
        if n_dim <= 5:
            max_points = 25000
        elif n_dim <= 10:
            max_points = 20000
        elif n_dim <= 25:
            max_points = 10000
        else:
            max_points = 5000

    x_f, counts = get_n_points(fails, lsf, max_points)
    try:
        labels, uniques = get_dbclusters(x_f, sample_weight=True,
                                         counts=counts,
                                         max_num_clusters=max_num_clusters)
    except (SystemError, MemoryError) as exc:
        print("Clustering failed due to error:")
        print(exc)
        x_f, _ = _filter_points(fails, lsf, None)
        return x_f, -1 * np.ones(x_f.shape[0], dtype=int), [-1]
    return x_f, labels, uniques


def _filter_points(fails, lsf, tol):
    x_f = np.empty((0, fails.shape[1]))
    counts = np.empty(0)
    if lsf.size > 0:
        x_f, counts = _get_unique_tol(lsf, tol)
    if x_f.shape[0] < 6 and fails.size > 0:  # because 3 core points
        unique_xf, counts2 = _get_unique_tol(fails, tol)
        x_f = np.append(x_f, unique_xf, axis=0)
        counts = np.append(counts, counts2)

    valid = np.isfinite(x_f).all(1)
    x_f = x_f[valid, :]
    counts = counts[valid]
    return x_f, counts


def get_n_points(fails, lsf, n_points=25000, conv_tol=0.1):
    def check_tol(n_points_curr):
        return abs((n_points - n_points_curr) / n_points) <= conv_tol

    def obj(tol):
        x_f = _filter_points(fails, lsf, 10 ** tol)[0]
        if check_tol(x_f.shape[0]):
            return 0
        return n_points - x_f.shape[0]

    def get_interval():
        start = -10
        while obj(start) > 0:
            start -= 10
        stop = 10
        while obj(stop) < 0:
            stop += 10
        return start, stop

    orig, counts = _filter_points(fails, lsf, None)
    if orig.shape[0] <= n_points or check_tol(orig.shape[0]):
        return orig, counts
    del orig, counts
    a, b = get_interval()
    res = brentq(obj, a, b)
    return _filter_points(fails, lsf, 10 ** res)


def _get_unique_tol(points, tol=None):
    if tol is None:
        return np.unique(points, axis=0, return_counts=True)
    rounded = np.round(points / tol)
    df = pd.DataFrame(np.c_[rounded, points])
    df = df.groupby(list(df.columns)[:rounded.shape[1]])
    final = df.agg("mean").values
    counts = df.size().values
    return final, counts.ravel()


def _get_group_stat(points, ids, agg=np.median):
    return agg(points[ids], axis=0)


def _get_sample_weights(sample_weight, fails, counts):
    if not sample_weight or counts is None or np.max(counts) <= 1:
        return np.ones(fails.shape[0]), 1
    sample_weight = (counts - counts.min()) / (counts.max() - counts.min())
    sample_weight[sample_weight < 1e-8] = 1e-8
    return sample_weight, sample_weight.mean()
