# pyRDO: A framework for design uncertainty quantification and optimization
This repo is under construction to be submitted to code ocean for reproducibility. Expect big changes after
publication on code ocean. Also expect lolhr4ra, i.e. the proposed method for uncertainty quantification.

# Reliability Analysis and Reliability-based Robust Design Optimization
Generally, given one or more limit state functions of form $`g(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}`$ as well as 
the input distributions $`\mathbf{X} \sim F_{\mathbf{X}}(\cdot, \boldsymbol{\theta}_{\mathbf{X}})`$ as parametrized by $`\boldsymbol{\theta}_{\mathbf{X}}`$, 
uncertainty quantification, i.e. reliability-analysis, seeks to compute the probability of failure $`P(\mathcal{F}) = P(g(\mathbf{X}) < 0)`$. 

To solve the uncertainty optimization, i.e. reliability-based robust design optimization, problem, the evaluation of $`P(\mathcal{F}`)$
as well as the expectations and variances of the objective functions $`f(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}`$
with respect to the distribution parameters $`\theta_{\mathbf{X}}`$ is required. `pyRDO` takes the objectives $`f(\dot)`$ and the limit states $`g(\dot)`$ 
as input and wraps them with `problem.obj_con` to be used by a generic gradient-free optimization algorithm.

# Citation
If this repo helped you, please cite

> C. Bogoclu, T. Nestorović, D. Roos; *Local Latin Hypercube Refinement for Multi-objective Design Uncertainty Optimization*,
Applied Soft Computing (2021)

Contributions welcome as there is a long road ahead to make this research code to a usable one.

