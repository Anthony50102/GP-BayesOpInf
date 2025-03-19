# step3_estimate.py
"""Estimate system parameters with GP-powered OpInf, using a PyTorch ridge solver."""

__all__ = [
    "estimate_posterior",
]

import logging
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import torch

import opinf

import bayes

__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-40, 20, 44)  # Search grid.

def torch_weighted_leastsq_solve(D, b, Wsqrt):
    r"""
    Solve the unregularized weighted least squares problem:

        min_x  || W^(1/2) (D x - b) ||^2

    which is equivalent to solving the normal equations:

        (D^T W D) x = (D^T W) b,

    where W = Wsqrt.T @ Wsqrt.

    Parameters
    ----------
    D : (m, n) np.ndarray
        Design matrix.
    b : (m,) np.ndarray
        Concatenated derivatives or other right-hand side.
    Wsqrt : (m, m) np.ndarray
        Square-root of the weight matrix, so that W = Wsqrt^T @ Wsqrt.

    Returns
    -------
    x : (n,) np.ndarray
        The least-squares solution vector.
    """
    # Convert to torch Tensors
    D_torch = torch.from_numpy(D).double()
    b_torch = torch.from_numpy(b).double()
    Wsqrt_torch = torch.from_numpy(Wsqrt).double()

    # Weighted design: W^(1/2) * D
    WD = Wsqrt_torch @ D_torch  # shape (m, n)
    # Weighted target: W^(1/2) * b
    wb = Wsqrt_torch @ b_torch  # shape (m,)

    # Normal equations: A x = B, where
    #   A = WD^T WD = D^T W D
    #   B = WD^T wb = D^T W b
    A = WD.t() @ WD
    B = WD.t() @ wb

    # Solve A x = B
    x_solution = torch.linalg.solve(A, B.unsqueeze(-1))  # shape (n, 1)

    return x_solution.squeeze(-1).numpy()  # shape (n,)


def torch_weighted_ridge_solve(D, b, Wsqrt, reg):
    r"""
    Solve the ridge-regularized weighted least squares problem:

        min_x  || W^(1/2) (D x - b) ||^2  +  reg * ||x||^2

    using PyTorch. Here:

        - D is (m x n)
        - b is (m,)
        - Wsqrt is (m x m), the "square root" of the weight matrix
        - reg is a nonnegative scalar lambda

    Returns
    -------
    x : (n,) np.ndarray
        The solution that minimizes the above objective.
    """
    # Convert inputs to double precision torch Tensors
    D_torch = torch.from_numpy(D).double()
    b_torch = torch.from_numpy(b).double()
    Wsqrt_torch = torch.from_numpy(Wsqrt).double()

    # Weighted design: W^(1/2) * D
    WD = Wsqrt_torch @ D_torch  # shape (m, n)
    # Weighted targets: W^(1/2) * b
    wb = Wsqrt_torch @ b_torch  # shape (m,)

    # Build the normal equations: (WD^T WD + reg*I) x = WD^T wb
    A = WD.t() @ WD + reg * torch.eye(D.shape[1], dtype=torch.double)
    B = WD.t() @ wb

    # Solve A x = B
    # For PyTorch 1.9+, can use torch.linalg.solve.  We'll do:
    x_solution = torch.linalg.solve(A, B.unsqueeze(-1))
    # x_solution is shape (n, 1)

    return x_solution.squeeze(-1).numpy()


def _posterior_unregularized_singlepass(
    time_domain_prediction: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    initial_conditions: np.ndarray,
    num_samples: int,
    D: np.ndarray,
    ddt_estimates: np.ndarray,
    Wsqrt: np.ndarray,
    model,
) -> bayes.BayesianODE:
    """
    Build a Bayesian ODE model by a single unregularized weighted least squares solve:

      x* = argmin_x ||W^(1/2)(D x - ddts)||^2

    Then do a quick "instability" check with 'num_samples' draws from the posterior.
    (If you only have a single deterministic solution, that might just repeat.)

    Returns
    -------
    bayes.BayesianODE
        The final ODE model with the posterior mean = x*, and
        precision = (D^T W D).
    """
    # Quick debug: condition number
    cond_num = np.linalg.cond(Wsqrt @ D)
    print(f"\nCondition number of W^(1/2)*D is {cond_num:.4e}")
    if cond_num > 1e12:
        print("WARNING: W^(1/2)*D is extremely ill-conditioned!\n")

    if initial_conditions is None:
        initial_conditions = snapshots_estimated[:, 0]
        print(initial_conditions)
        initial_conditions = np.array([1,1])

    # Solve the unregularized weighted LS problem
    print(ddt_estimates.shape)
    mean = torch_weighted_leastsq_solve(D, ddt_estimates, Wsqrt)
    # mean = np.array([1.5,1,1,3])
    print(f"Unregularized solution x = {mean}")
    model.parameters = mean

    # "Precision" matrix in Bayesian sense is A = D^T W D
    WD = Wsqrt @ D
    A = WD.T @ WD  # shape (n, n)

    # Construct the BayesianODE
    try:
        bayesian_model = bayes.BayesianODE(model, mean, A)
    except np.linalg.LinAlgError as ex:
        if "not positive definite" in str(ex):
            print("WARNING: Weighted design matrix is not SPD!")
            return None
        else:
            raise

    # Optional: check "instability" by calling model.predict repeatedly
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)

    def unstable(sol, size):
        # Example: if the solution time dimension doesn't match -> skip
        if sol.shape[-1] != size:
            return False
        return np.any(np.abs(sol - shift).max(axis=1) > limits)

    draws = []
    for _ in range(num_samples):
        # In reality, you might sample from ~ N(mean, A^-1). But let's just
        # do a single 'predict' call if you're only storing mean in the code.
        for tdom in (time_domain_prediction, time_domain_estimated):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = bayesian_model.predict(initial_conditions, tdom)
            if unstable(sol, len(tdom)):
                print("UNSTABLE -> returning None")
                return None
        draws.append(sol)

    # Just for info: how well do we match the snapshots?
    for draw in draws:
        print(draw.shape)
    mean_sol = np.mean(draws, axis=0)
    err = la.norm(mean_sol - snapshots_estimated) / snapshotnorm
    print(f"Relative error vs snapshots: {err:.2%}")

    return bayesian_model

def estimate_posterior(
    gps,
    time_domain_prediction,
    config,
    initial_conditions=None,
) -> bayes.BayesianODE:
    """
    Construct the posterior parameter distribution, using an *unregularized*
    weighted least squares solver. No lambda, no grid search, no optimization.

    Parameters
    ----------
    gps : list of trained gpkernel.GP_RBFW objects
        ...
    ...
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        # Create your model
        model = config.Model()
        # Gather state estimates from each GP
        state_estimates = np.array([gp.state_estimate for gp in gps])

        # Build the OpInf data matrix, D
        D = model.data_matrix(state_estimates)
        # Concatenate the time derivatives
        ddt_estimates = np.concatenate([gp.ddt_estimate for gp in gps])
        # Block-diagonal of each gp.sqrtW => overall W^(1/2)
        Wsqrt = la.block_diag(*[gp.sqrtW for gp in gps])

        # Single unregularized solve
        return _posterior_unregularized_singlepass(
            time_domain_prediction=time_domain_prediction,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            initial_conditions=initial_conditions,
            num_samples=10,  # or however many draws you want to do
            D=D,
            ddt_estimates=ddt_estimates,
            Wsqrt=Wsqrt,
            model=model,
        )
