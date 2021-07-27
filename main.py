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
from submission.plots import plot


def main(pop_size=None, opt_iters=None):
    """
    Run all scripts

    Parameters
    ----------
    pop_size : Optional[int]
        Population size for optimization. Note that it will be x2 for surrogate based strategies and
        //5 for direct_short
    opt_iters: Optional[int]
        Number of optimization iterations after initial population.
        Note that it will be x2 for surrogate based strategies and //5 for direct_short
    """
    try:
        save_dir = os.path.dirname(__file__)
    except:
        # In case things go wrong in code ocean
        save_dir = os.path.abspath(".")

    for example_name in ["ex1", "ex2", "ex3"]:  #
        print("Testing Random Strategy")
        random_sampling(example_name, save_dir=save_dir)
        print("Testing Direct Strategy")
        direct(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing Direct Short Strategy")
        direct_short(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing Stationary GP Strategy")
        stationary_gp(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing Stationary SVR Strategy")
        stationary_svr(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing LoLHR GP Strategy")
        adaptive_gp(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing LoLHR SVR Strategy")
        adaptive_svr(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing Gu et. Al. (2014) Strategy with GP")
        adaptive_mean_gp(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        print("Testing Gu et. Al. (2014) Strategy with SVR")
        adaptive_mean_svr(example_name, save_dir=save_dir, force_pop_size=pop_size, force_opt_iters=opt_iters)
        plot(example_name, os.path.join(save_dir, "results"))


if __name__ == "__main__":
    main()
