# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:07:43 2019

@author: Bogoclu
"""
from random import Random
from time import time

import numpy as np
from inspyred import ec
from inspyred.ec import terminators, variators, replacers, Bounder, emo
from inspyred.swarm import PSO, topologies
from scipy.stats import uniform

from duqo.doe.lhs import make_doe


def make_pareto(obs, maximize=False):
    return [emo.Pareto(obj, maximize=maximize) for obj in obs.tolist()]


def _get_method(method):
    if method is None:
        return "NSGA", emo.NSGA2
    method = method.upper()
    if method == "DEA":
        return method, ec.DEA
    if method == "NSGA":
        return method, emo.NSGA2
    if method == "PAES":
        return method, emo.PAES
    if method == "PSO":
        return method, PSO
    if method == "EC":
        return method, ec.EvolutionaryComputation


def _atleast2d(x):
    x = np.array(x)
    if x.ndim < 2:
        x = x.reshape((1, -1))
    return x


class InspyredOptimizer:
    def __init__(self, obj_con, lb, ub, method=None, args=(), maximize=False, obj_con_path="", scale_objs=True,
                 **kwargs):
        """
        

        Parameters
        ----------
        obj_con : function
            the wrapper for the objective and constraint functions
            must be of the following form, where inputs is a numpy
            array with shape = (n_samples, n_dimensions)
            def obj_con(inputs, *args, **kwargs):
                objs, cons = [], []
                for input in inputs
                    objs.append([o(input) for o in objectives])
                    cons.append([c(input) for c in constraints])
            # np.array Not necessary as this will be done internally too
            # it is important to return something even if n_cons = 0
            return np.array(objs), np.array(cons) 
                
        lb : Iterable
            lower optimization bounds
        ub : Iterable
            upper optimization bounds

        method : Optional[str]
            Optimization method to use. Default: 'NSGAII'
        maximize : bool
            If True, maximization is assumed. Default: False

        args : Iterable
            Arguments to pass to obj_con

        obj_con_path : str
            If passed, objectives and constraints of every candidate are written as arrays to path
            obj_con_path + "_gen" + str(self.nit) + ".npz"
            Make sure the directory exists containing the path exists! Default: ""

        scale_objs : bool
            If True, objectives will be scaled. Useful for problems with constraints, where constraints are enforced
            using a punishment function


        **kwargs: dict
            Any keyword arguments that should be passed to obj_con


        Returns
        -------
        None.

        """

        self.bounds = Bounder(lb, ub)
        self.n_dims = len(np.array(lb).ravel())
        self.obj_con = obj_con
        self.method, self.method_ = _get_method(method)
        self.maximize = maximize
        self.args = args
        self.kwargs = kwargs
        self.obj_con_path = obj_con_path
        self.verbose = kwargs.get("verbose", 0)
        self.reset()
        self.scale_objs = scale_objs
        self.obj_scales_ = None
        self.counter_start_pop, self.nfev, self.nit = -1, 0, 0
        self.start_gen, self.archive, self.punish_factor, self.pareto_size = None, None, 1, 1

    def reset(self):
        self.counter_start_pop = -1
        self.nfev, self.nit = 0, 0
        self.archive = None

    def evaluator(self, candidates, args):
        if self.verbose:
            t0 = time()
        candidates = np.array(candidates)
        self.nit += 1
        self.nfev += candidates.shape[0]
        # Just call the evaluate of the problem
        objs, cons = self.obj_con(candidates, *self.args, **self.kwargs)
        objs, cons = _atleast2d(objs), _atleast2d(cons)
        if self.scale_objs:
            if self.obj_scales_ is None:
                self.obj_scales_ = np.abs(np.median(objs, axis=0))
                self.obj_scales_[self.obj_scales_ < 1e-3] = 1e-3
            objs /= self.obj_scales_
        if self.obj_con_path:
            path = self.obj_con_path + "_gen" + str(self.nit) + ".npz"
            np.savez(path, candidates, objs, cons)

        if cons.size > 0:
            objs = self.punish(objs, cons)

        if self.verbose:
            # Don't count initial population in prints to avoid confusion
            print("Its.", self.nit - 1, " Evals.", self.nfev, "-", time() - t0, "seconds.")
            if objs.shape[-1] < 2:
                if self.maximize:
                    best_obj = max(objs)
                else:
                    best_obj = min(objs)
                print("Best objective", best_obj, end="\n\n")
        return make_pareto(objs, self.maximize)

    def generator(self, random, args):
        self.counter_start_pop += 1
        return self.start_gen[self.counter_start_pop].tolist()

    def punish(self, objs, cons):
        fails = cons < 0
        factors = -np.sum(cons * fails, axis=1, keepdims=True) * self.punish_factor
        if self.obj_scales_ is not None:
            factors = np.tile(factors, objs.shape[1])
            factors /= self.obj_scales_
        fails = fails.any(1)
        objs[fails] += factors[fails]
        return objs

    def my_archive_backup(self, population, *args, **kwargs):
        for best in population:
            if self.archive is None:
                self.archive = [best]
            elif best in self.archive:
                continue
            elif len(self.archive) < self.pareto_size:
                self.archive.append(best)
            elif best.fitness > self.archive[-1].fitness:
                self.archive[-1] = best
                self.archive = sorted(self.archive, key=lambda x: x.fitness, reverse=False)
            elif best.fitness < self.archive[-1].fitness:
                pass

        self.archive = sorted(self.archive, key=lambda x: x.fitness, reverse=False)

    def optimize(self, pop_size=100, max_gens=100, punish_factor=100,
                 pareto_size=100, verbose=0, start_gen=None):
        ea = self.init_optimize(pop_size=pop_size, punish_factor=punish_factor, pareto_size=pareto_size,
                                verbose=verbose, start_gen=start_gen)
        return self.run_optimize(ea, pop_size=pop_size, max_gens=max_gens)

    def run_optimize(self, ea, pop_size=100, max_gens=100):
        _ = ea.evolve(generator=self.generator,
                      evaluator=self.evaluator,
                      pop_size=pop_size,
                      maximize=True,  # maximization is handled in make_pareto function...
                      bounder=self.bounds,
                      max_generations=max_gens,
                      neighborhood_size=max(5, pop_size // 20),
                      max_archive_size=max(2 * pop_size, self.pareto_size),
                      num_elites=2)
        return ea.archive

    def init_optimize(self, pop_size=100, punish_factor=100, pareto_size=100, verbose=0, start_gen=None):
        self.reset()
        self.punish_factor = punish_factor
        self.verbose = verbose
        self.pareto_size = pareto_size
        opt_margs = [uniform(self.bounds.lower_bound[k], self.bounds.upper_bound[k] - self.bounds.lower_bound[k])
                     for k in range(self.n_dims)]
        if start_gen is None or start_gen.shape != (pop_size, self.n_dims):
            print("Generating start generation")
            self.start_gen = make_doe(pop_size, opt_margs, num_tries=100)
        else:
            print("Using the passed generation")
            self.start_gen = start_gen
        prng = Random()
        prng.seed(7)

        ea = self.method_(prng)
        ea.terminator = [terminators.generation_termination,
                         terminators.diversity_termination,
                         ]

        ea.variator = [variators.blend_crossover,
                       variators.gaussian_mutation]

        ea.replacer = replacers.generational_replacement

        ea.observer = self.my_archive_backup

        # ea.selector = selectors.rank_selection
        if "PSO" in self.method:
            ea.topology = topologies.ring_topology
        return ea