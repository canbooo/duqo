import abc
from abc import ABC
from typing import Union, List, Callable, Iterable, Dict

import numpy as np

from duqo.variables.multivariate import MultiVariate


class LimitStateFunction(ABC):

    @abc.abstractmethod
    def __call__(self, input_points: np.ndarray, envelope: bool = True) -> np.ndarray:
        pass

    @abc.abstractmethod
    def evaluate(self, input_points: np.ndarray, std_norm: bool = False, envelope: bool = True) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def num_eval(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_function(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def x_safe(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def x_fail(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def x_lsf(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_sp(self):
        pass

    @abc.abstractmethod
    def to_mp(self):
        pass


class SPLimitStateFunction(LimitStateFunction):

    def __init__(self, base_functions: Union[Callable, List[Callable]], multivariate: MultiVariate,
                 call_args: Iterable, call_kwargs: Dict, save_points: bool = False):
        if not isinstance(base_functions, list):
            base_functions = [base_functions]
        self.functions = base_functions
        self.multivariate = multivariate
        self._num_function = len(self.functions)
        self.call_args = _get_args(call_args, self._num_function, default_value=tuple())
        self.call_kwargs = _get_args(call_kwargs, self._num_function, default_value=dict())
        self._num_eval = 0

        self.save_points = save_points
        self._x_safe, self._x_fail, self._x_lsf = None, None, None

    def to_mp(self, num_parallel: int = 2) -> LimitStateFunction:
        return MPLimitStateFunction(self.functions, self.multivariate, self.call_args[0], self.call_kwargs[0],
                                    save_points=self.save_points, num_workers=num_parallel)

    def to_sp(self) -> LimitStateFunction:
        return SPLimitStateFunction(self.functions, self.multivariate, self.call_args[0], self.call_kwargs[0],
                                    save_points=self.save_points)

    def __call__(self, input_points: np.ndarray, envelope: bool = True) -> np.ndarray:
        """
        Call functions to calculcate input_points

        Parameters
        ---------
        input_points : np.ndarray(shape=(num_samples, num_dims))
            Array of input point coordinates. if 1-D, it will be assumed a row array (one point)
        envelope : bool
            If True, the minimum of all functions will be returned, otherwise, all results are returned.
        """
        if input_points.ndim < 2:
            input_points = input_points.reshape((1, -1))
        res = np.ones(input_points.shape[0], self._num_function)
        for i_fun, fun in enumerate(self.functions):
            res[:, i_fun] = fun(input_points, *self.call_args[i_fun], **self.call_kwargs[i_fun])
        if envelope:
            return np.min(res, axis=1, keepdims=True)
        return res

    def evaluate(self, input_points: np.ndarray, std_norm: bool = False, envelope: bool = True) -> np.ndarray:
        """
        Evaluate functions at input_points


        Arguments
        ---------
        input_points : np.ndarray(shape=(num_samples, num_dims))
            Array of input point coordinates. if 1-D, it will be assumed a row array (one point)
        std_norm : bool
            If True, input_points are treated to be standard normal and thus transferred to original space
            before passing to self.functions
        envelope : bool
            If True, the minimum of all functions will be returned, otherwise, all results are returned.
        """

        if input_points.ndim < 2:
            input_points = input_points.reshape((1, -1))
        if std_norm:
            input_points = self.multivariate.from_normal_space(input_points)
        res = self(input_points, envelope=envelope)
        self._save_evaluation(input_points, res)
        return res

    def _save_evaluation(self, input_points, responses):
        self._num_eval += input_points.shape[0]
        if not self.save_points:
            return
        ids = np.isclose(responses, 0, atol=1e-5)
        if ids.any():
            self._x_lsf = np.append(self.x_lsf, input_points[ids, :], axis=0)
        ids = responses < 0
        if ids.any():
            self._x_fail = np.append(self.x_fail, input_points[ids, :], axis=0)
        ids = responses > 0
        if ids.any():
            self._x_safe = np.append(self.x_safe, input_points[ids, :], axis=0)

    @property
    def num_eval(self) -> int:
        return self._num_eval

    @property
    def num_function(self) -> int:
        return self._num_function

    @property
    def x_safe(self) -> np.ndarray:
        if self._x_safe is None:
            return np.empty((0, self._num_function), dtype=np.float32)
        return self._x_safe

    @property
    def x_fail(self) -> np.ndarray:
        if self._x_fail is None:
            return np.empty((0, self._num_function), dtype=np.float32)
        return self._x_fail

    @property
    def x_lsf(self) -> np.ndarray:
        if self._x_lsf is None:
            return np.empty((0, self._num_function), dtype=np.float32)
        return self._x_lsf


class MPLimitStateFunction(LimitStateFunction):
    """Only useful if save_points=True since LimitStateFunction cannot collect all evaluations in subprocesses"""
    def __init__(self, base_functions: Union[Callable, List[Callable]], multivariate: MultiVariate,
                 call_args: Iterable, call_kwargs: Dict, save_points: bool = False, num_parallel: int = 2):
        if num_parallel < 1:
            raise ValueError(f"num_parallel must be strictly positive, got {num_parallel}")
        self.workers = [SPLimitStateFunction(base_functions, multivariate, call_args, call_kwargs,
                                             save_points=save_points)
                        for _ in range(num_parallel)]

    def __call__(self, input_points, envelope=True, worker_id=0) -> np.ndarray:
        return self.workers[worker_id](input_points, envelope=envelope)

    def evaluate(self, input_points, std_norm=False, envelope=True, worker_id=0) -> np.ndarray:
        return self.workers[worker_id].evaluate(input_points, std_norm=std_norm, envelope=envelope)

    @property
    def num_eval(self) -> int:
        tot = 0
        for worker in self.workers:
            tot += worker.num_eval
        return tot

    @property
    def num_function(self) -> int:
        return self.workers[0].num_function

    @property
    def x_safe(self) -> np.ndarray:
        return _aggregated_points(self.workers, "x_safe")

    @property
    def x_fail(self) -> np.ndarray:
        return _aggregated_points(self.workers, "x_fail")

    @property
    def x_lsf(self) -> np.ndarray:
        return _aggregated_points(self.workers, "x_lsf")

    def to_sp(self) -> LimitStateFunction:
        return self.workers[0].to_sp()

    def to_mp(self, num_parallel: int = 2) -> LimitStateFunction:
        return self.workers[0].to_mp(num_parallel)


def _aggregated_points(workers, attr_name):
    res = np.empty((0, workers[0].num_function), dtype=np.float32)
    for worker in workers:
        res = np.append(res, getattr(worker, attr_name), axis=0)
    return res


def _get_args(args, n_const, default_value=None):
    """
    Make list of args if passed, else a list of default_value
    """
    if not args:
        args = default_value
    return [args] * n_const
