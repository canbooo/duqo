import os
from submission.runners.rrdo_direct import main as direct
from submission.runners.rrdo_direct_short import main as direct_short
from submission.runners.rrdo_stationary_gp import main as stationary_gp
from submission.runners.rrdo_stationary_svr import main as stationary_svr
from submission.runners.rrdo_adaptive_gp import main as adaptive_gp
from submission.runners.rrdo_adaptive_svr import main as adaptive_svr
from submission.runners.rrdo_adaptive_mean_gp import main as adaptive_mean_gp
from submission.runners.rrdo_adaptive_mean_svr import main as adaptive_mean_svr
from submission.runners.rrdo_mc import main as random_sampling

""" 
Note that the number of generations and the population size for example 1 is decreased from 100 to 10
for a faster reproducibility check at code ocean. Please increase the number of iterations to achieve better results
for all strategies using an optimizer, i.e. all but random_sampling. The definition file for example 1 is found under
submission/definitions/example1.py

"""
try:
    save_dir = os.path.dirname(__file__)
except:
    # In case things go wrong in code ocean
    save_dir = os.path.abspath(".")
example_name = "ex1"
# direct(example_name, save_dir=save_dir)
# direct_short(example_name, save_dir=save_dir)
# random_sampling(example_name, save_dir=save_dir)
# stationary_gp(example_name, save_dir=save_dir)
# stationary_svr(example_name, save_dir=save_dir)
# adaptive_gp(example_name, save_dir=save_dir)
# adaptive_svr(example_name, save_dir=save_dir)
adaptive_mean_gp(example_name, save_dir=save_dir)
adaptive_mean_svr(example_name, save_dir=save_dir)
