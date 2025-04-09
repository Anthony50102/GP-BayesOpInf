from gpkernels import TORCH_GP
from typing import Iterable
import opinf
import numpy as np
import itertools
import math
import step3_estimate as step3

def torch_fit_single_gaussian_process(
    stateindex: int,
    time_domain_training: np.ndarray,
    time_domain_sampled: np.ndarray,
    state_variable_sampled: np.ndarray,
    config,
    kernel: str = 'rbf',
    gp_regularizer: float = 1e-8,
    return_error: bool = False,
    prior = None
) -> TORCH_GP:
    """Fit a single Gaussian process (GP) to snapshot data for one variable.

    Parameters
    ----------
    stateindex : int
        Index of the state variable.
    time_domain_training : (m',) ndarray
        Time domain at which to estimate states and time derivatives
        for the parameter estimation.
    time_domain_sampled : (m,) ndarray
        Time domain corresponding to the available training snapshots.
    state_variable_sampled : (m,) ndarray
        Observations of a single state variable over the time domain
        ``time_domain_sampled``.
    gp_regularizer : float >= 0
        Regularization hyperparameter for the GP inference in inverting for
        the least-squares weight matrix.

    Returns
    -------
    gpkernels.GP_RBFW
        One-dimensional Gaussian process with parameters fit to training data.
    """
    with opinf.utils.TimedBlock(
        f"\nfitting GP model for state '{config.DIMFMT(stateindex)}'\n"
    ):
        gp = TORCH_GP(
            kernel=kernel,
            period_prior = prior
        )
        error = gp.fit(time_domain_sampled, state_variable_sampled, error=return_error)
        print(f"gp fitted with hyperparameters \n")
        gp.print_kernel_params()

    with opinf.utils.TimedBlock("computing weight matrix", timelimit=600):
        gp.compute_lstsq_matrices(time_domain_training, eta=gp_regularizer)

    if return_error:
        return gp, error
    else:
        return gp


def torch_fit_gaussian_processes(
    time_domain_training: np.ndarray,
    time_domains_sampled: list[np.ndarray],
    snapshots_sampled: np.ndarray,
    config,
    gp_regularizer: float = 1e-8,
    kernel: str = 'rbf',
    return_error: bool = False,
    prior = None
) -> Iterable[TORCH_GP]:
    """Fit Gaussian Process (GP) regression models to the snapshot data,
    one state variable at a time.

    Parameters
    ----------
    time_domain_training : (m',) ndarray
        Time domain at which to estimate states and time derivatives
        for the parameter estimation.
    time_domains_sampled : list of num_variables (m,) ndarrays
        Time domains corresponding to the available training snapshots,
        one for each variable.
    snapshots_sampled : (num_variables, m) ndarray
        Observed training snapshots.
    """
    if prior is not None and 'cos' in kernel:
        print(f"Use prior for kernel: {kernel}")
    else:
        prior = None
    ret_val = [
        torch_fit_single_gaussian_process(
            stateindex=stateindex,
            time_domain_training=time_domain_training,
            time_domain_sampled=time_domains_sampled[stateindex],
            state_variable_sampled=snapshots_sampled[stateindex],
            config=config,
            gp_regularizer=gp_regularizer,
            kernel=kernel,
            prior=prior[stateindex] if prior is not None else prior,
        )
        for stateindex in range(config.NUMVARS)
    ] if not return_error else [torch_fit_single_gaussian_process(
            stateindex=stateindex,
            time_domain_training=time_domain_training,
            time_domain_sampled=time_domains_sampled[stateindex],
            state_variable_sampled=snapshots_sampled[stateindex],
            config=config,
            gp_regularizer=gp_regularizer,
            kernel=kernel,
            prior=prior[stateindex] if prior is not None else prior,
            return_error=True) for stateindex in range(config.NUMVARS)] 
    print(ret_val) 
    return ret_val

def torch_fit_best_gps(
    time_domain_training: np.ndarray,
    time_domains_sampled: list[np.ndarray],
    time_domain_prediction: np.ndarray,
    snapshots_sampled: np.ndarray,
    config,
    gp_regularizer: float = 1e-8,
    prior = None,
    ):
    combinations = ['cos*rbf', 'cos*rq', 'rbf', 'rq', 'cos']

    min_posterior_error = math.inf
    gp_mll_errors = []
    posterior_erros = []
    for combo in combinations:
        fitted_gps = torch_fit_gaussian_processes(
            time_domain_training=time_domain_training,
            time_domains_sampled=time_domains_sampled,
            snapshots_sampled=snapshots_sampled,
            gp_regularizer=gp_regularizer,
            config=config,
            kernel = combo,
            return_error=True,
            prior=prior
        )
        errors = [result[1] for result in fitted_gps]
        gps = [result[0] for result in fitted_gps]
        gp_mll_errors.append(tuple(errors))

    
        # Step 3: Construct the posterior hyperparameters -------------------------
        post_error , bayesian_model = step3.estimate_posterior(
            gps=gps,
            time_domain_prediction=time_domain_prediction,
            config=config,
            return_error=True
        )
        posterior_erros.append(post_error)

        if post_error < min_posterior_error:
            min_kernel = combo
            min_posterior_error = post_error
            mll_for_min_post_error = errors
            best_bayesian = bayesian_model
            best_gps = gps
    print(f"Best kernel has been found to be: {min_kernel} with a posterior error of {min_posterior_error} and a mll error of {mll_for_min_post_error}")
    return best_bayesian, best_gps, min_kernel
    