from typing import List, Optional, Union

import numpy as np
from scipy import stats

from duqo.doe.lhs import make_doe, find_empty_bins, inherit_lhs, optimize_doe


def create_new_lhs(num_sample: int,
                   margs: Optional[List[stats.rv_continuous]] = None,
                   limit_lower_bound: Optional[np.ndarray] = None,
                   limit_upper_bound: Optional[np.ndarray] = None,
                   corr_mat: Union[np.ndarray, float] = 0,
                   num_tries: Optional[int] = None,
                   central_design: bool = True):
    """
    Makes an LHS with desired distributions and correlation

    Parameters
    ----------
    num_sample : int
       Number of samples

    margs : Optional[list[scipy.stats.rv_continuous]]
       List of marginal distribution objects with .ppf method. Each object
       corresponds one random variable. If None, uniform distribution between limit_lower_bound and limit_upper_bound
       is assumed.

    limit_lower_bound : np.ndarray
       Lower bounds used to limit the minimum value of continuous distributions with infinite support.
       Specifically, values smaller than this value will be clipped. If margs is None, these will be used to set
       the lower bound of the uniform distribution.
       NOTE: This does not set the bounds of the doe! Use margs for this goal.
       Shape = (n_dim,)

    limit_upper_bound : np.ndarray
       Upper bounds used to limit the maximum value of continuous distributions with infinite support.
       Specifically, values larger than this value will be clipped. If margs is None, these will be used to set
       the upper bound of the uniform distribution.
       Shape = (n_dim,)

    corr_mat : float or np.ndarray
       Correlation matrix. If an array, it must be symmetrical with shape=(n_dim, n_dim). If scalar, the whole matrix
       except the diagonal will be filled with this value. Default is 0 meaning no correlation.

    num_tries : Optional[int]
        Number of optimization steps to improve doe metrics such as minimum pairwise distance and correlation. Large
        values can make the algorithm slow depending on num_sample. If None, it will be set to 20000 for
        num_sample < 100 and 2000 otherwise.

    central_design: bool
        If true, the samples are placed in the middle of the bins, which improves the space filling properties. However,
        it should be set to false in case you want to extend this LHS by adding new samples using the extend_lhs
        function while not violating the Latin hypercube design rules.

    Returns
    -------
    doe : np.ndarray
       Optimized design of experiments matrix following Latin hypercube design with the shape (num_sample, len(margs))

  """
    return make_doe(num_sample, margs=margs, corr_mat=corr_mat, num_tries=num_tries, lower_bound=limit_lower_bound,
                    upper_bound=limit_upper_bound, central_design=central_design, verbose=0)


def extend_lhs(doe, num_sample, lower_bound, upper_bound, num_tries: Optional[int] = None):
    """
    Adds new samples to an existing doe while trying to follow Latin hypercube design rules.

     Parameters
    ----------
    doe: np.ndarray
        Design of experiments to be extended. doe should be a matrix with shape (num_sample, num_dim).

    num_sample : int
       Number of samples to be added

    lower_bound, upper_bound: np.ndarray
        Bounds to place new points in to. If they are different than the bounds of the original doe, the Latin hypercube
        design will be limited to this smaller domain. Shape (num_dim,)

    num_tries : Optional[int]
        Number of optimization steps to improve doe metrics such as minimum pairwise distance and correlation. Large
        values can make the algorithm slow depending on num_sample. If None, it will be set to 20000 for
        num_sample < 100 and 2000 otherwise.

    Returns
    -------
    extended_doe : np.ndarray
        Optimized design of experiments matrix following Latin hypercube design with the shape (num_sample, num_dim).
        Specifically, only the new points will be returned.

    """
    n_bins = num_sample
    n_empty = 0
    empty_bins = np.empty(0)
    while n_empty < num_sample:
        empty_bins = find_empty_bins(doe, n_bins, lower_bound, upper_bound)
        n_empty = np.max(empty_bins.sum(0))
        n_bins += 1
    new_points = inherit_lhs(num_sample, empty_bins, lower_bound, upper_bound)
    return optimize_doe(new_points, doe_old=doe, num_tries=num_tries)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lower_bound = -1 * np.ones(2)
    upper_bound = np.ones(2)
    num_start = 8
    num_add = 8
    doe = create_new_lhs(num_start, limit_lower_bound=lower_bound, limit_upper_bound=upper_bound, central_design=False,
                         num_tries=10)  # so that it runs fast
    new_points = extend_lhs(doe, num_add, lower_bound, upper_bound,
                            num_tries=10) # so that it runs fast

    fig, ax = plt.subplots(figsize=(7, 12))
    ax.scatter(doe[:, 0], doe[:, 1], label="Orig. points")
    ax.scatter(new_points[:, 0], new_points[:, 1], label="Added points")
    lhs_grid = np.linspace(-1, 1, num_start + num_add)
    ax.vlines(lhs_grid, -1, 1, colors="k")
    ax.hlines(lhs_grid, -1, 1, colors="k")

    plt.legend()
    plt.show()
