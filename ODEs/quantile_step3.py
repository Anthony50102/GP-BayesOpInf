# step3_estimate_quantile.py
"""Estimate system parameters with GP-powered OpInf, using a PyTorch quantile regression solver."""

__all__ = [
    "estimate_posterior",
]

import logging
import warnings
import numpy as np
import scipy.linalg as la
import torch

import opinf
import bayes

__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-40, 20, 44)  # Search grid.

def torch_weighted_quantile_regression_solve(D, b, Wsqrt, tau=0.5, lr=1e-2, num_iter=1000):
    r"""
    Solve the unregularized weighted quantile regression problem:

        min_x  \sum_i ρ_τ( (W^(1/2)*(D x - b))_i )

    where the quantile loss function is defined as

        ρ_τ(u) = u * (τ - I(u < 0)).

    Parameters
    ----------
    D : (m, n) np.ndarray
        Design matrix.
    b : (m,) np.ndarray
        Concatenated derivatives or other right-hand side.
    Wsqrt : (m, m) np.ndarray
        Square-root of the weight matrix, so that W = Wsqrt.T @ Wsqrt.
    tau : float, default=0.5
        The quantile to regress. (τ=0.5 corresponds to median regression.)
    lr : float, default=1e-2
        Learning rate for the optimizer.
    num_iter : int, default=1000
        Number of optimization iterations.

    Returns
    -------
    x : (n,) np.ndarray
        The quantile regression solution vector.
    """
    # Convert to torch Tensors (using double precision)
    D_torch = torch.from_numpy(D).double()
    b_torch = torch.from_numpy(b).double()
    Wsqrt_torch = torch.from_numpy(Wsqrt).double()

    # Form weighted design and target:
    WD = Wsqrt_torch @ D_torch  # shape (m, n)
    wb = Wsqrt_torch @ b_torch  # shape (m,)

    # Initialize parameter vector x (to be optimized)
    x = torch.zeros(D.shape[1], dtype=torch.double, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(num_iter):
        optimizer.zero_grad()
        residual = WD @ x - wb  # shape (m,)
        # Quantile loss: if residual >= 0 then tau*residual, else (tau-1)*residual.
        loss = torch.where(residual >= 0, tau * residual, (tau - 1) * residual)
        loss = loss.sum()
        loss.backward()
        optimizer.step()

    return x.detach().numpy()


def _posterior_quantile_singlepass(
    time_domain_prediction: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    initial_conditions: np.ndarray,
    num_samples: int,
    D: np.ndarray,
    ddt_estimates: np.ndarray,
    Wsqrt: np.ndarray,
    model,
    tau: float,
) -> bayes.BayesianODE:
    """
    Build a Bayesian ODE model by a single unregularized weighted quantile regression solve:

      x* = argmin_x \sum_i ρ_τ(W^(1/2)*(D x - ddts)_i)

    Then perform an "instability" check with 'num_samples' draws from the posterior.
    (If you only have a single deterministic solution, these draws may be nearly identical.)

    Returns
    -------
    bayes.BayesianODE
        The final ODE model with the posterior "estimate" x* and
        a "precision" matrix computed as (D^T W D) (note: this is a placeholder).
    """
    # Quick debug: condition number of the weighted design matrix.
    cond_num = np.linalg.cond(Wsqrt @ D)
    print(f"\nCondition number of W^(1/2)*D is {cond_num:.4e}")
    if cond_num > 1e12:
        print("WARNING: W^(1/2)*D is extremely ill-conditioned!\n")

    if initial_conditions is None:
        initial_conditions = snapshots_estimated[:, 0]
        print("Defaulting to initial conditions:", initial_conditions)
        # If necessary, override with a default (here simply [1, 1])
        initial_conditions = np.array([1, 1])

    # Solve the unregularized weighted quantile regression problem.
    print("Solving weighted quantile regression...")
    quantile_estimate = torch_weighted_quantile_regression_solve(D, ddt_estimates, Wsqrt, tau=tau)
    print(f"Quantile regression solution x = {quantile_estimate}")
    model.parameters = quantile_estimate

    # Compute the "precision" matrix as D^T W D (using the weighted design matrix)
    WD = Wsqrt @ D
    A = WD.T @ WD  # shape (n, n)

    # Construct the BayesianODE model.
    try:
        bayesian_model = bayes.BayesianODE(model, quantile_estimate, A)
    except np.linalg.LinAlgError as ex:
        if "not positive definite" in str(ex):
            print("WARNING: Weighted design matrix is not SPD!")
            return None
        else:
            raise

    # Optional: check "instability" by repeatedly calling model.predict.
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)

    def unstable(sol, size):
        # If the solution's time dimension doesn't match, consider it unstable.
        if sol.shape[-1] != size:
            return False
        return np.any(np.abs(sol - shift).max(axis=1) > limits)

    draws = []
    for _ in range(num_samples):
        for tdom in (time_domain_prediction, time_domain_estimated):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = bayesian_model.predict(initial_conditions, tdom)
            if unstable(sol, len(tdom)):
                print("UNSTABLE -> returning None")
                return None
        draws.append(sol)

    for draw in draws:
        print("Draw shape:", draw.shape)
    mean_sol = np.mean(draws, axis=0)
    err = la.norm(mean_sol - snapshots_estimated) / snapshotnorm
    print(f"Relative error vs snapshots: {err:.2%}")

    return bayesian_model


def estimate_posterior(
    gps,
    time_domain_prediction,
    config,
    initial_conditions=None,
    tau=0.5,
) -> bayes.BayesianODE:
    """
    Construct the posterior parameter distribution using an *unregularized*
    weighted quantile regression solver. (No additional lambda, grid search, or further optimization is performed.)

    Parameters
    ----------
    gps : list of trained gpkernel.GP_RBFW objects
        ...
    time_domain_prediction : np.ndarray
        Time domain for prediction.
    config : configuration object with attribute Model (used to instantiate the model).
    initial_conditions : np.ndarray, optional
        Initial conditions for the ODE.
    tau : float, default=0.5
        Quantile parameter (τ=0.5 corresponds to median regression).

    Returns
    -------
    bayes.BayesianODE
        The final Bayesian ODE model with quantile regression parameter estimates.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters with quantile regression\n"):
        # Create the model instance.
        model = config.Model()
        # Gather state estimates from each GP.
        state_estimates = np.array([gp.state_estimate for gp in gps])

        # Build the OpInf data matrix, D.
        D = model.data_matrix(state_estimates)
        # Concatenate the time derivatives.
        ddt_estimates = np.concatenate([gp.ddt_estimate for gp in gps])
        # Build a block-diagonal weight matrix from each gp.sqrtW.
        Wsqrt = la.block_diag(*[gp.sqrtW for gp in gps])

        # Perform a single unregularized quantile regression solve.
        return _posterior_quantile_singlepass(
            time_domain_prediction=time_domain_prediction,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            initial_conditions=initial_conditions,
            num_samples=10,  # or however many draws you wish to perform
            D=D,
            ddt_estimates=ddt_estimates,
            Wsqrt=Wsqrt,
            model=model,
            tau=tau,
        )
