# DUQO: *D*esign *U*ncertainty *Q*uantification and *O*ptimization framework
The repo is under ongoing development. Breaking changes may occur in next versions.

Current stable version can be installed using pip
`pip install duqo`

Alternatively, clone the master branch and use `pip install -e .`

See [examples](https://github.com/canbooo/duqo/tree/master/examples) for example usage. Documentation is on its way.

## Reliability Analysis and Reliability-based Robust Design Optimization
Generally, given one or more limit state functions of form
> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20g(%5Cmathbf%7Bx%7D)%3A%20%5Cmathbb%7BR%7D%5En%20%5Crightarrow%20%5Cmathbb%7BR%7D">
as well as  the input distributions 
> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20%5Cmathbf%7BX%7D%20%5Csim%20F_%7B%5Cmathbf%7BX%7D%7D(%5Ccdot%2C%20%5Cboldsymbol%7B%5Ctheta%7D_%7B%5Cmathbf%7BX%7D%7D)">
as parametrized by <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20%5Cboldsymbol%7B%5Ctheta%7D_%7B%5Cmathbf%7BX%7D%7D">, 
uncertainty quantification, i.e. reliability-analysis, seeks to compute the probability of failure
> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20P(%5Cmathcal%7BF%7D)%20%3D%20P(g(%5Cmathbf%7BX%7D)%20%3C%200)">
whereas robustness analysis investigates the variance of some performance indicator function such as an optimization objective.

To solve the uncertainty optimization, i.e. reliability-based robust design optimization, problem, the evaluation of <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20P(%5Cmathcal%7BF%7D)">
as well as the expectations and variances of the objective functions 
> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20f(%5Cmathbf%7Bx%7D)%3A%20%5Cmathbb%7BR%7D%5En%20%5Crightarrow%20%5Cmathbb%7BR%7D">
and possibly deterministic constraints
> <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20c(%5Cmathbf%7Bx%7D)%3A%20%5Cmathbb%7BR%7D%5En%20%5Crightarrow%20%5Cmathbb%7BR%7D">
with respect to the distribution parameters <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20%5Ctheta_%7B%5Cmathbf%7BX%7D%7D">
is required. Besides the input distributions, `duqo` takes the objectives <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20f_i(%5Ccdot)">,
the limit states <img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20g_j(%5Ccdot)"> and the constraints 
<img src="https://render.githubusercontent.com/render/math?math=%5Ccolor%7Bred%7D%20c_k(%5Ccdot)">
as input and wraps them with `problem.obj_con` to be used by a generic gradient-free optimization algorithm.

## Install
You can install the latest stable release using 
`pip install duqo`
Alternatively, you can install the most recent version using
`pip install git+https://github.com/canbooo/duqo.git`
or without git
`pip install --upgrade https://github.com/canbooo/duqo.git`

## Citation
If this repo helped your research, I would appreciate you citing:

> C. Bogoclu, T. Nestorović, D. Roos; *Local Latin Hypercube Refinement for Multi-objective Design Uncertainty Optimization*,
Applied Soft Computing (2021)

- Article: [https://www.sciencedirect.com/science/article/abs/pii/S1568494621007286](https://www.sciencedirect.com/science/article/abs/pii/S1568494621007286)
- Preprint: [https://arxiv.org/pdf/2108.08890.pdf](https://arxiv.org/pdf/2108.08890.pdf)

## Contribution
Contributions welcome as there is a long road ahead to make this research code to a usable one.

