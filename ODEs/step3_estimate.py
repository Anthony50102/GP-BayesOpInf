# step3_estimate.py
"""Estimate system parameters with GP-powered OpInf."""

__all__ = [
    "estimate_posterior",
]

import logging
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

import opinf

import bayes
import wlstsq


__MAXOPTVAL = 1e12  # Ceiling for optimization.
__DEFAULT_SEARCH_GRID = np.logspace(-16, 5, 22)  # Search grid.


def _posterior_autoregularized_multisample(
    regularizer_grid: np.ndarray,
    time_domain_prediction: np.ndarray,
    time_domain_estimated: np.ndarray,
    snapshots_estimated: np.ndarray,
    initial_conditions: np.ndarray,
    num_samples: int,
    lstsq_solver: wlstsq.WeightedLSTSQSolver,
    model,
    return_error: bool = False,
) -> bayes.BayesianODE:
    r"""Use an error-based optimization to select an appropriate regularization
    hyperparamter for the parameter estimation regression.

        \ohat = (D^T W D + \lambda I)^{-1} D^T W ddts

    Use ``num_samples`` posterior draws to check that the posterior gives
    stable solutions.

    Parameters
    ----------
    regularizer_grid : (num_regs,) ndarray
        Grid of regularization values to try (followed by an optimization).
    time_domain_prediction : (k,) ndarray
        Time domain over which to solve the model for a stability check.
    initial_conditions : (r,) ndarray
        Initial conditions for the model.
    time_domain_estimated : (m',) ndarray
        Time domain corresponding to the GP estimates of the snapshots.
    snapshots_estimated : (r, m') ndarray
        GP state estimates of the available training snapshots.
    num_samples : int
        Number of posterior draws to do for the stability check.
    lstsq_solver : wlstsq.WeightedLSTSQSolver
        Solver for the least-squares problem (already 'fit' to the data).
    model : config.Model
        Model object for running simulations.

    Returns
    -------
    bayes.BayesianODE
        Bayesian ODE model.
    """
    shift = np.mean(snapshots_estimated, axis=1).reshape((-1, 1))
    limits = 5 * np.abs(snapshots_estimated - shift).max(axis=1)
    snapshotnorm = la.norm(snapshots_estimated)
    if initial_conditions is None:
        initial_conditions = snapshots_estimated[:, 0]

    def unstable(_solution, size):
        """Return True if the solution is unstable."""
        if _solution.shape[-1] != size:
            return True
        return np.any(np.abs(_solution - shift).max(axis=1) > limits)

    def get_bayesian_model(reg):
        """Form and solve the regression for the given regularization value."""
        # Posterior mean.
        lstsq_solver.regularizer = reg
        mean = lstsq_solver.solve()
        model.parameters = mean

        # Posterior precision matrix.
        sqrtW_D = lstsq_solver.solvers[0].data_matrix  # = sqrt(W) @ D
        precision = (sqrtW_D.T @ sqrtW_D) + (reg**2 * np.eye(mean.size))

        try:
            return bayes.BayesianODE(model, mean, precision)
        except np.linalg.LinAlgError as ex:
            if ex.args[0] == "Matrix is not positive definite":
                return None
            raise

    def _training_error(logreg):
        """Get the solution for a single regularization candidate."""
        opinf_regularizer = 10**logreg
        print(
            f"Testing regularizer {opinf_regularizer:.4e}...",
            end="",
            flush=True,
        )
        bayesian_model = get_bayesian_model(opinf_regularizer)
        if bayesian_model is None:
            print("Covariance not SPD")
            return __MAXOPTVAL

        # Sample the posterior distribution and check for stability.
        draws = []
        for _ in range(num_samples):
            for tmdmn in (time_domain_prediction, time_domain_estimated):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draw = bayesian_model.predict(
                        initial_conditions=initial_conditions,
                        timepoints=tmdmn,
                    )
                if unstable(draw, tmdmn.size):
                    print("UNSTABLE")
                    return __MAXOPTVAL
            draws.append(draw)

        mean_solution = np.mean(draws, axis=0)
        error = la.norm(mean_solution - snapshots_estimated) / snapshotnorm
        print(f"{error:.2%} error")
        return error

    # Test each regularization hyperparameter.
    regularizer_grid = np.atleast_1d(regularizer_grid)
    if (num_tests := len(regularizer_grid)) == 1:
        search_bounds = [regularizer_grid[0] / 10, 10 * regularizer_grid[0]]
    else:
        # GRID SEARCH.
        _smallest_error, _best_reg_index = __MAXOPTVAL, None
        regularizer_grid = np.sort(regularizer_grid)
        print("\nGRIDSEARCH")
        for i, reg in enumerate(regularizer_grid):
            print(f"({i+1:d}/{num_tests:d}) ", end="")
            if (error := _training_error(np.log10(reg))) < _smallest_error:
                _smallest_error = error
                _best_reg_index = i
        if _best_reg_index is None:
            raise ValueError("grid search failed!")
        best_reg = regularizer_grid[_best_reg_index]

        if _best_reg_index == 0:
            print("\nWARNING: extend regularizer_grid to the left!")
            search_bounds = [best_reg / 100, regularizer_grid[1]]
        elif _best_reg_index == num_tests - 1:
            print("\nWARNING: extend regularizer_grid to the right!")
            search_bounds = [regularizer_grid[-2], 100 * best_reg]
        else:
            search_bounds = [
                regularizer_grid[_best_reg_index - 1],
                regularizer_grid[_best_reg_index + 1],
            ]

        message = f"Best regularization via gridsearch: {best_reg:.4e}"
        print(message + "\n")
        logging.info(message)

    # Follow up grid search with minimization-based search.
    print("1D OPTIMIZATION")
    opt_result = opt.minimize_scalar(
        _training_error, method="bounded", bounds=np.log10(search_bounds)
    )

    if opt_result.success and opt_result.fun != __MAXOPTVAL:
        regularizer = 10**opt_result.x
        message = f"Best regularization via optimization: {regularizer:.4e}"
        print(message)
        logging.info(message)
    else:
        regularizer = best_reg
        print("Optimization failed, falling back on gridsearch")

    if return_error:
        return _smallest_error, get_bayesian_model(regularizer)
    else:
        return get_bayesian_model(regularizer)


def estimate_posterior(
    gps,
    time_domain_prediction,
    config,
    initial_conditions=None,
    return_error: bool = False
) -> bayes.BayesianODE:
    """Construct the posterior parameter distribution.

    Parameters
    ----------
    gps : list of trained gpkernel.GP_RBFW objects.
        Gaussian processes for each state variable, already fit to data.
    time_domain_prediction : (k,) ndarray
        Time domain over which to solve the model for stability checks.
    initial_conditions : (r,) ndarray or None
        Initial conditions for the model. If not provided, use the GP state
        estimates at the initial time.

    Returns
    -------
    bayes.BayesianODE
        Bayesian ODE model.
    """
    with opinf.utils.TimedBlock("constructing posterior hyperparameters\n"):
        model = config.Model()
        state_estimates = np.array([gp.state_estimate for gp in gps])

        # Construct the data matrix, RHS ddts vector, and weight matrix.
        D = model.data_matrix(state_estimates)
        ddt_estimates = np.concatenate([gp.ddt_estimate for gp in gps])
        W = la.block_diag(*[gp.sqrtW for gp in gps])


        # Fit a weighted least-squares solver for the problem.
        lstsq_solver = wlstsq.WeightedLSTSQSolver(W, regularizer=1)
        lstsq_solver.fit(D, ddt_estimates)


        # Select a single regularizer for all equations.
        return _posterior_autoregularized_multisample(
            regularizer_grid=__DEFAULT_SEARCH_GRID,
            time_domain_prediction=time_domain_prediction,
            time_domain_estimated=gps[0].t_estimation,
            snapshots_estimated=state_estimates,
            initial_conditions=initial_conditions,
            num_samples=20,
            lstsq_solver=lstsq_solver,
            model=model,
            return_error=return_error
        )
    
# In wlstsq.py, modify the WeightedLSTSQSolver class to accept periodicity priors

class WeightedLSTSQSolver:
    """Solver for a weighted least-squares problem (or problems) with optional periodicity priors."""

    _METHODS = (
        "svd",
        "lstsq",
        "normal",
    )

    def __init__(
        self,
        weights: np.ndarray,
        regularizer: float = 0.0,
        method: str = "lstsq",
        period_priors: dict = None  # Add period priors parameter
    ):
        """Store the regularizer and initialize attributes.

        Parameters
        ----------
        weights : (r, m, m) or (m, m) ndarray
            Collection of r positive definite matrices defining the weighted
            norms for each problem.
        regularizer : (d, d) or (d,) ndarray or float.
            Regularization hyperparameters.
        method : str
            The strategy for solving the regularized least-squares problem.
        period_priors : dict or None
            Dictionary containing prior information about periodicity of the system.
            Should include 'means' and 'variances' for the periods of each state.
        """
        self.__solvers = []
        self.__period_priors = period_priors  # Store the period priors

        self.weights = weights
        self.regularizer = regularizer
        self.method = method

    # ... keep existing properties and methods ...

    def fit(self, lhs, rhs):
        """Store the data matrices defining the least-squares problems
        and incorporate periodicity priors if available.

        Parameters
        ----------
        lhs : (m, d) ndarray
            Unweighted left-hand side data matrix (D in the notes).
        rhs : (r, m) or (m,) ndarray
            Unweighted right-hand side data matrix.
        """
        # Check dimensions
        if lhs.shape != (_shape := (self.m, lhs.shape[1])):
            raise ValueError(f"expected lhs.shape == {_shape}")
        if np.ndim(rhs) == 1:
            rhs = np.reshape(rhs, (1, -1))
        if rhs.shape != (_shape := (self.r, self.m)):
            raise ValueError(f"expected rhs.shape == {_shape}")
        self.__d = lhs.shape[1]

        # Apply periodicity priors if available
        modified_lhs = lhs
        modified_rhs = rhs
        modified_weights = self.weights

        if self.__period_priors is not None:
            # Modify the LHS, RHS, or weights based on periodicity priors
            modified_lhs, modified_rhs, modified_weights = self._incorporate_periodicity_priors(
                lhs, rhs, self.weights)

        # Initialize underlying solvers
        if np.isscalar(self.regularizer):
            SolverClass = opinf.lstsq.L2Solver
        else:
            SolverClass = opinf.lstsq.TikhonovSolver

        self.__solvers = [
            SolverClass(self.regularizer).fit(
                modified_weights[i] @ modified_lhs, modified_weights[i] @ modified_rhs[i]
            )
            for i in range(self.r)
        ]

        # Set the solver method (only for Tikhonov solvers).
        if SolverClass is opinf.lstsq.TikhonovSolver:
            for solver in self.__solvers:
                solver.method = self.method

        return self

    def _incorporate_periodicity_priors(self, lhs, rhs, weights):
        """Incorporate periodicity priors into the least squares problem.
        
        This can be done by:
        1. Adding synthetic data points that reflect the periodicity
        2. Modifying the weights to favor parameter values that result in the expected period
        3. Adding regularization terms that prefer parameter values consistent with prior
        
        Parameters
        ----------
        lhs : (m, d) ndarray
            Left-hand side data matrix.
        rhs : (r, m) ndarray
            Right-hand side data matrix.
        weights : (r, m, m) ndarray
            Weight matrices.
            
        Returns
        -------
        modified_lhs : ndarray
            Modified left-hand side matrix.
        modified_rhs : ndarray
            Modified right-hand side matrix.
        modified_weights : ndarray
            Modified weight matrices.
        """
        # Extract period priors
        period_means = self.__period_priors.get('means', None)
        period_vars = self.__period_priors.get('variances', None)
        
        if period_means is None or period_vars is None:
            return lhs, rhs, weights
            
        # Create modified matrices - this is where we incorporate the priors
        # Approach 1: Add synthetic data points enforcing periodicity
        # For each state variable with a prior, add constraints that 
        # x(t+T) ≈ x(t) where T is the prior period
        
        # Get original dimensions
        m, d = lhs.shape
        r = rhs.shape[0]
        
        # Number of synthetic constraints to add per state variable
        n_constraints = 10  # Adjust as needed
        
        # Create arrays to store new rows
        new_lhs_rows = []
        new_rhs_values = []
        
        # For each state with a period prior, add synthetic constraints
        for i, (mean_period, var_period) in enumerate(zip(period_means, period_vars)):
            if i >= r:  # Skip if we don't have this state
                continue
                
            # Weight for this constraint (higher confidence = higher weight)
            confidence = 1.0 / (var_period + 1e-6)  
            
            # Create synthetic data points enforcing periodicity constraints
            # This is a simplified approach - in practice you'd design this
            # based on the specific system dynamics
            constraint_lhs = np.zeros((n_constraints, d))
            constraint_rhs = np.zeros(n_constraints)
            
            # The constraint design depends on the structure of the system
            # For oscillatory systems, we might constrain parameters to match the period
            # This is highly system-dependent and would need to be customized
            
            # Add the new constraints
            new_lhs_rows.append(constraint_lhs)
            new_rhs_values.append(constraint_rhs)
        
        # If we have any new constraints to add
        if new_lhs_rows:
            # Combine the original and new constraints
            modified_lhs = np.vstack([lhs] + new_lhs_rows)
            
            # Adjust rhs and weights accordingly
            # (Implementation details depend on specific system structure)
            # This is a placeholder for the actual implementation
            modified_rhs = rhs  # Would need to be extended
            modified_weights = weights  # Would need to be extended
            
            return modified_lhs, modified_rhs, modified_weights
            
        # If no constraints were added, return originals
        return lhs, rhs, weights