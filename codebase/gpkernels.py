# gpkernels.py
"""Convenience classes for Gaussian process kernels."""

__all__ = [
    "GP_RBFW",
    "TORCH_GP_RBFW"
]
### TODO - Numerically determine now

import math
import abc
import joblib
import numpy as np
import scipy.linalg as la

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)

### New Imports
import torch
import numpy as np
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel 

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 constant_bounds=None, length_scale_bounds=None, noise_level_bounds=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.noise_level_bounds = noise_level_bounds
        self.mean_module = ZeroMean()
        self.mean_module = gpytorch.means.ConstantMean()
        # Here we use a ScaleKernel * RBFKernel plus a WhiteNoiseKernel to mimic
        # k(t,t') = constant * exp(-(t-t')^2/(2*length_scale^2)) + noise_level δ(t-t')
        if constant_bounds and length_scale_bounds:
            self.covar_module = ScaleKernel(
                RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(*length_scale_bounds)),
                outputscale_constraint=gpytorch.constraints.Interval(*constant_bounds)
            )
        
        else:
            self.covar_module = ScaleKernel(
                RBFKernel(),
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TorchBaseGP(abc.ABC): 
    """Base class for gpytorch Gaussian process regressor wrappers."""
    def __init__(self):
        self.model = None
        self.likelihood = None
        self.t_training = None  # training input (1D tensor)
        self.y = None         # training target (1D tensor)
        # These will be computed in compute_lstsq_matrices:
        self.state_estimate = None
        self.ddt_estimate = None
        self.ddt_covariance = None
        self.sqrtW = None

    @property
    def nsamples(self):
        return self.t_training.size(0) if self.t_training is not None else 0

    def fit(self, t_training, training_data):
        if training_data.ndim > 1:
            raise ValueError("GP training data must be one-dimensional")
        # Store training data as tensors.
        self.t_training = torch.tensor(t_training, dtype=torch.float32)
        self.y = torch.tensor(training_data, dtype=torch.float32)
        # Create likelihood with noise constraint and model.
        self.likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(*self.noise_level_bounds))
        # Note: the model expects inputs with an extra feature dimension.
        train_x = self.t_training.unsqueeze(-1)
        train_y = self.y
        self.model = ExactGPModel(
            train_x, train_y, self.likelihood,
            # self.constant_bounds, self.length_scale_bounds, self.noise_level_bounds
        )
        # Removed redundant likelihood reassignment.
        # Training mode.
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for j in range(5):
            for i in range(self.training_iter):
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, t):
        # Ensure t is a tensor with shape (n,1)
        t_tensor = (
            torch.tensor(t, dtype=torch.float32).unsqueeze(-1)
            if not torch.is_tensor(t) else t
        )
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(t_tensor))
            # Attach .std for compatibility (sqrt of variance)
            pred.std = pred.variance.sqrt()
        return pred

    def prediction_bounds(self, t, kind="95%"):
        pred = self.predict(t)
        mean, std = pred.mean, pred.std
        if kind == "std":
            width = std
        elif kind == "95%":
            width = 1.96 * std
        elif kind == "2std":
            width = 2 * std
        elif kind == "3std":
            width = 3 * std
        else:
            raise ValueError(kind)
        return mean - width, mean, mean + width

    def __call__(self, t, tprime):
        # Evaluate the kernel by calling the model's covariance module.
        t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(-1) if not torch.is_tensor(t) else t
        tprime_tensor = torch.tensor(tprime, dtype=torch.float32).unsqueeze(-1) if not torch.is_tensor(tprime) else tprime
        self.model.eval()
        with torch.no_grad():
            return self.model.covar_module(t_tensor, tprime_tensor).evaluate()

    def save(self, save_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'training_iter': self.training_iter,
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load(load_path):
        # Loading functionality can be implemented as needed.
        raise NotImplementedError("Loading not implemented for gpytorch version.")

    @abc.abstractmethod
    def compute_lstsq_matrices(self, t_est, **kwargs):
        r"""Compute data needed for the GP-BayesOpInf least squares.
        """
        raise NotImplementedError

    def _compute_estimates_and_weights(self,
                                       K_yy: torch.Tensor,
                                       K_zy: torch.Tensor,
                                       K_zz: torch.Tensor,
                                       kappa_zy: torch.Tensor,
                                       eta: float = 1e-1):
        L = torch.cholesky(K_yy)
        y_unsq = self.y.unsqueeze(-1)  # shape (m, 1)
        K_yy_inv_y = torch.cholesky_solve(y_unsq, L)
        self.state_estimate = (kappa_zy @ K_yy_inv_y).squeeze(-1).detach().numpy().astype(np.float64)
        self.ddt_estimate = (K_zy @ K_yy_inv_y).squeeze(-1).detach().numpy().astype(np.float64)
        K_zy_inv = torch.cholesky_solve(K_zy.T, L)
        ddt_covariance = K_zz - K_zy @ K_zy_inv
        self.ddt_covariance = ddt_covariance
        C = ddt_covariance
        C_reg = C + eta * torch.eye(C.size(0), device=C.device, dtype=C.dtype)
        evals, evecs = torch.linalg.eigh(C_reg)
        if torch.any(evals <= 0):
            raise ValueError("inverse covariance not positive definite, increase eta")
        self.sqrtW = (evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T).detach().numpy().astype(np.float64)


class TORCH_GP_RBFW(TorchBaseGP):
    """Gaussian process regressor with a kernel of the form:

        k(t, t') = constant * exp(-(t - t')^2 / (2 * length_scale^2)) + white_noise.

    The white noise is modeled as an additive component.
    This class mimics the interface of the sklearn wrapper.
    """
    def __init__(self,
                 constant_bounds=(1e-8, 1e5),
                 length_scale_bounds=(.1, 100),
                 noise_level_bounds=(1e-16, .5),
                 training_iter=100):
        super().__init__()
        self.constant_bounds = constant_bounds
        self.length_scale_bounds = length_scale_bounds
        self.noise_level_bounds = noise_level_bounds
        self.training_iter = training_iter

    @property
    def constant(self):
        self.model.eval()
        if hasattr(self.model.covar_module, "kernels"):
            return self.model.covar_module.kernels[0].outputscale
        else:
            return self.model.covar_module.outputscale

    @property
    def length_scale(self):
        self.model.eval()
        if hasattr(self.model.covar_module, "kernels"):
            return self.model.covar_module.kernels[0].base_kernel.lengthscale
        else:
            return self.model.covar_module.base_kernel.lengthscale

    @property
    def noise_level(self):
        self.model.eval()
        noise = self.likelihood.noise
        return noise.item() if torch.is_tensor(noise) else noise

    def __str__(self):
        return "\n\t".join(
            [
                "Gaussian radial basis function kernel (gpytorch)",
                r"k(t, t') = \sigma^2 exp(-(t - t')^2 / (2 \ell^2)) + \chi I",
                rf"\sigma^2 = {float(self.constant):.4e}",
                rf"\ell = {float(self.length_scale):.4e}",
                rf"\chi = {float(self.noise_level):.4e}",
            ]
        )

    def rbf_eval(self, t1, t2):
        """Evaluate the RBF (squared exponential) part of the kernel.

        kappa(t1, t2) = constant * exp( -(t1 - t2)^2 / (2 * length_scale^2) ).

        Parameters
        ----------
        t1 : array-like, shape (m1,)
        t2 : array-like, shape (m2,)

        Returns
        -------
        Tensor of shape (m1, m2) with kernel evaluations.
        """
        t1 = torch.tensor(t1, dtype=torch.float32) if not torch.is_tensor(t1) else t1
        t2 = torch.tensor(t2, dtype=torch.float32) if not torch.is_tensor(t2) else t2
        tdiff = t1.unsqueeze(1) - t2.unsqueeze(0)
        return self.constant * torch.exp(-(tdiff**2) / (2 * self.length_scale**2))

    def compute_lstsq_matrices(self, t_est, eta=1e-1):
        r"""Compute the data needed for the GP-BayesOpInf least squares.

        This sets the following attributes:
        - state_estimate: GP approximation of the state.
        - ddt_estimate: GP approximation of the time derivative.
        - sqrtW: Square root of the (regularized) inverse derivative covariance.

        Parameters
        ----------
        t_est : array-like, shape (m',)
            Time points at which to compute estimates.
        eta : float, optional
            Regularization constant.
        """
        # Evaluate the RBF kernel at training and estimation points.
        rbf_yy = self.rbf_eval(self.t_training, self.t_training)  # (m, m)
        rbf_zy = self.rbf_eval(t_est, self.t_training)             # (m', m)
        rbf_zz = self.rbf_eval(t_est, t_est)                         # (m', m')
        self.t_estimation = t_est

        # Prepare tensor versions.
        t_est_tensor = torch.tensor(t_est, dtype=torch.float32) if not torch.is_tensor(t_est) else t_est
        t_train_tensor = (
            self.t_training if torch.is_tensor(self.t_training)
            else torch.tensor(self.t_training, dtype=torch.float32)
        )
        
        # Compute pairwise differences.
        tprime_minus_tprime = t_est_tensor.unsqueeze(1) - t_est_tensor.unsqueeze(0)
        tprime_minus_t = t_est_tensor.unsqueeze(1) - t_train_tensor.unsqueeze(0)
        ell2 = self.length_scale ** 2
        
        # Add noise to the diagonal.
        noise = self.noise_level
        # If noise is a tensor, convert it to a scalar.
        if torch.is_tensor(noise):
            noise = noise.item()
        K_yy = rbf_yy + torch.diag(torch.full((rbf_yy.size(0),), noise, 
                                                dtype=rbf_yy.dtype, 
                                                device=rbf_yy.device))
        K_zy = -tprime_minus_t * rbf_zy / ell2
        K_zz = (1 - (tprime_minus_tprime ** 2 / ell2)) * rbf_zz / ell2

        self._compute_estimates_and_weights(K_yy, K_zy, K_zz, rbf_zy, eta)
        return self
    
class _BaseGP(abc.ABC):
    """Base class for Gaussian process regressor wrappers."""

    def __init__(self, kernel, n_restarts_optimizer, alpha=0):
        """Initialize the underlying Gaussian process regressor."""
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
        )

    # Properties --------------------------------------------------------------
    @property
    def gpr(self):
        """Underlying scikit-learn GaussianProcessRegressor."""
        return self.__gpr

    @gpr.setter
    def gpr(self, gp):
        """Ensure that the gpr a scikit-learn GaussianProcessRegressor."""
        if not isinstance(gp, GaussianProcessRegressor):
            raise TypeError("expected GaussianProcessRegressor object")
        self.__gpr = gp

    @property
    def nsamples(self):
        """Number of samples used to train the GP."""
        if hasattr(self, "train_indices"):
            return self.t_training.size

    # Main routines -----------------------------------------------------------
    def fit(self, t_training, training_data):
        """Fit the (1D) Gaussian process to compressed snapshot data.

        Parameters
        ----------
        t_snapshots : (nsamples,) ndarray
            Time domain corresponding to the available training snapshots.
        training_data : (nsamples,) ndarray
            Available training snapshots for a single state variable.
        """
        if training_data.ndim > 1:
            raise ValueError("GP training data must be one-dimensional")

        # Fit the Gaussian process to the data.
        self.t_training = t_training
        self.y = training_data
        self.gpr.fit(t_training[:, None], self.y)

        return self

    def predict(self, t):
        """Predict with the trained Gaussian process.

        Parameters
        ----------
        t : (m,) ndarray
            Times at which to evaluate the GP.

        Returns
        -------
        mean : (m,) ndarray
            Mean GP prediction at each point.
        std : (m,) ndarray
            Standard deviation of the GP prediction at each point.
        """
        return self.gpr.predict(t[:, None], return_std=True)

    def prediction_bounds(self, t, kind="95%"):
        """Get prediction bounds.

        Parameters
        ----------
        t : (m,) ndarray
            Times at which to evaluate the GP.
        kind : str
            Type of prediction band to produce.
            * "std": One standard deviation, mean ± std.
            * "95%": 95% confidence interval, mean ± 1.96*std.
            * "2std": Two standard deviations, mean ± 2*std.
            * "3std": Three standard deviations, mean ± 3*std.

        Returns
        -------
        lower : (m,) ndarray
            Lower bound of the GP prediction at each point.
        mean : (m,) ndarray
            Mean GP prediction at each point.
        upper : (m,) ndarray
            Upper bound of the GP prediction at each point.
        """
        mean, std = self.predict(t)

        if kind == "std":
            width = std
        elif kind == "95%":
            width = 1.96 * std
        elif kind == "2std":
            width = 2 * std
        elif kind == "3std":
            width = 3 * std
        else:
            raise ValueError(kind)

        return mean - width, mean, mean + width

    def __call__(self, t, tprime):
        """Evaluate the kernel.

        Parameters
        ----------
        t : (m,) ndarray
            First set of times at which to evaluate the GP.
        tprime : (mprime,) ndarray
            Second set of times at which to evaluate the GP.

        Returns
        -------
        kernel(t, tprime) : (m, mprime) ndarray
            Kernel evaluation.
        """
        return self.gpr.kernel_(t[:, None], tprime[:, None])

    # Persistence -------------------------------------------------------------
    def save(self, save_path):
        """Save the GP regressor to a file with joblib."""
        joblib.dump(self, save_path)

    @staticmethod
    def load(load_path):
        """Load a GP regressor with joblib."""
        return joblib.load(load_path)

    # Abstract methods --------------------------------------------------------
    @abc.abstractmethod
    def compute_lstsq_matrices(self, t_est, **kwargs):
        r"""Compute data needed for the GP-BayesOpInf least squares.

        Parameters
        ----------
        t_est : (trainsize,) ndarray
            Time domain at which to estimate states and time derivatives
            for the parameter estimation.
        """
        raise NotImplementedError

    def _compute_estimates_and_weights(
        self,
        K_yy: np.ndarray,
        K_zy: np.ndarray,
        K_zz: np.ndarray,
        kappa_zy: np.ndarray,
        eta: float = 1e-8,
    ):
        """Compute state and time derivative estimates from kernel matrices.

        Parameters
        ----------
        K_yy : (m, m) ndarray
            kappa(t, t) + noise * I
        K_zy : (m', m) ndarray
            d/d1 kappa(t', t)
        K_zz : (m', m') ndarray
            d^2/d1d2 kappa(t', t')
        kappa_zy : (m', m) ndarray
            Kernel evaluated at the estimation times and the training times.
        eta : float >= 0
            Regularization constant for the computation of the weight matrix.

        Returns
        -------
        Nothing, but the following attributes are set:
        * state_estimate : (m',) ndarray
            Mean estimates for the state at the estimation times.
        * ddt_estimate : (m',) ndarray
            Mean estimates for the time derivatives at the estimation times.
        * ddt_covariance : (m', m') ndarray
            Covariance for the time derivatives at the estimation times.
        * sqrtW : (m', m') ndarray
            Approximate square root of the weight matrix W_zz.
        """
        # Cholesky factor K_yy since we need the action of inv(K_yy).
        K_yy = la.cho_factor(K_yy, overwrite_a=True, check_finite=True)

        # Compute the approximate state over the full time domain.
        K_yy_inv_y = la.cho_solve(K_yy, self.y)
        self.state_estimate = kappa_zy @ K_yy_inv_y

        # Compute the approximate time derivative over the full time domain.
        self.ddt_estimate = K_zy @ K_yy_inv_y

        # Calculate covariance C = inv(W_zz) and stabilize for inversion.
        _Kzyyyyz = K_zy @ la.cho_solve(K_yy, K_zy.T)
        _Kzyyyyz = 0.5 * (_Kzyyyyz + _Kzyyyyz.T)  # Symmetrize.
        self.ddt_covariance = C = K_zz - _Kzyyyyz

        # Compute the sqrt(inv(C)) via truncated eigendecomposition.
        C_evals, C_evecs = la.eigh(
            C + (eta * np.eye(K_zz.shape[0])),
            check_finite=False,
        )
        if np.any(C_evals <= 0):
            raise ValueError(
                "inverse covariance not positive definite, increase eta"
            )
        self.sqrtW = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T


class GP_RBFW(_BaseGP):
    """Gaussian process regressor with the following kernel:

    k(t, t') = constant * (RBF(t, t'; length_scale) + noise_level δ(|t - t'|)).

    This class lightly wraps sklearn.gaussian_process.GaussianProcessRegressor.

    Attributes
    ----------
    gpr : sklearn.gaussian_process.GaussianProcessRegressor
    constant : float >= 0
    length_scale : float >= 0
    noise_level : float >= 0
    train_indices : (m',) ndarray
        Subset of full time domain used to fit GP, i.e., t' = t[train_indices].
    y : (m',) ndarray
        Training data for selecting GP hyperparameters.
    """

    def __init__(
        self,
        constant_bounds=(1e-5, 1e5),
        length_scale_bounds=(1.5e-6, 0.002),
        noise_level_bounds=(1e-14, 1e-10),
        n_restarts_optimizer=50,
    ):
        """Initialize the underlying Gaussian process regressor.

        Parameters
        ----------
        constant_bounds : pair of floats >=0 or "fixed"
            The lower and upper bound on the constatn value. If set to
            "fixed", the constant value is fixed at 1.0.
        length_scale_bounds : pair of floats >= 0 or "fixed"
            Lower and upper bound on 'length_scale' for the radial basis
            function kernel RBf(t, t; length_scale).
            If set to "fixed", 'length_scale' cannot be changed during
            hyperparameter tuning.
        noise_level_bounds : pair of floats >= 0 or "fixed"
            Lower and upper bounds on 'noise_level' for the white noise kernel.
            If set to "fixed", 'noise_level' cannot be changed during
            hyperparameter tuning.
        n_restarts_optimizer : int
            Number of restarts of the optimizer for finding the kernel's
            parameters which maximize the log-marginal likelihood.
        """
        # Kernel.
        constant = ConstantKernel(1.0, constant_value_bounds=constant_bounds)
        rbf = RBF(length_scale_bounds=length_scale_bounds)
        white = WhiteKernel(noise_level_bounds=noise_level_bounds)
        kernel = (constant * rbf) + white

        # Regressor.
        return _BaseGP.__init__(self, kernel, n_restarts_optimizer)

    # Properties --------------------------------------------------------------
    @property
    def constant(self):
        r"""This is :math:`\sigma^2` in the paper."""
        return self.gpr.kernel_.k1.k1.constant_value

    @property
    def length_scale(self):
        r"""This is :math:`\ell` in the paper."""
        return self.gpr.kernel_.k1.k2.length_scale

    @property
    def noise_level(self):
        r"""This is :math:`\chi` in the paper."""
        return self.gpr.kernel_.k2.noise_level

    def __str__(self):
        """String representation: kernel form + list hyperparameters."""
        return "\n\t".join(
            [
                "Gaussian radial basis function kernel",
                r"k(t, t') = \sigma^2 exp(-(t - t')^2 / (2 \ell^2)) + \chi I",
                rf"\sigma^2 = {self.constant:.4e}",
                rf"\ell = {self.length_scale:.4e}",
                rf"\chi = {self.noise_level:.4e}",
            ]
        )

    # Main routines -----------------------------------------------------------
    def rbf_eval(self, t1, t2):
        """Evaluate the RBF (squared exponential) part of the kernel,

            kappa(t1, t2) = sigma^2 exp( (-1/2) (t1 - t2)^2 / ell^2) )

        Parameters
        ----------
        t1 : (m1,) ndarray
            First set of times at which to evaluate the GP.
        t2 : (m2,) ndarray
            Second set of times at which to evaluate the GP.

        Returns
        -------
        RBF(t1, t2; length_scale) : (m1, m2) ndarray
            Kernel evaluation. The (i, j)-th entry is kappa(t1[i], t2[j]).
        """
        tdiff = t1[:, None] - t2
        return self.constant * np.exp(-(tdiff**2) / (2 * self.length_scale**2))

    # Mandatory implementation ------------------------------------------------
    def compute_lstsq_matrices(self, t_est, eta=1e-8):
        r"""Compute the data needed for the GP-BayesOpInf least squares.

        This method sets the following attributes.
        * state_estimate (y_tilde): GP approximation of the state q(t).
        * ddt_estimate (z_tilde): GP approximation of the derivative dq/dt.
        * sqrtW: Square root of (K_zz - K_zy inv(K_yy) K_yr)^{-1}.
        # * yKinvy: y @ K^{-1} @ y, used for computing \sigma_{i}^{2}.

        Parameters
        ----------
        t_est : (m',) ndarray
            Time domain at which to estimate states and time derivatives
            for the parameter estimation.
        eta : float >= 0
            Regularization constant for the computation of the weight matrix.
        """
        # Evaluate the radial basis function kernel.
        rbf_yy = self.rbf_eval(self.t_training, self.t_training)  # m x m
        rbf_zy = self.rbf_eval(t_est, self.t_training)  # m' x m
        rbf_zz = self.rbf_eval(t_est, t_est)  # m' x m'
        self.t_estimation = t_est

        # Calculate the K matrices (uses derivatives of the RBF kernel).
        tprime_minus_tprime = t_est[:, None] - t_est
        tprime_minus_t = t_est[:, None] - self.t_training
        ell2 = self.length_scale**2
        K_yy = rbf_yy + np.diag(np.full(rbf_yy.shape[0], self.noise_level))
        K_zy = -tprime_minus_t * rbf_zy / ell2
        K_zz = (1 - (tprime_minus_tprime**2 / ell2)) * rbf_zz / ell2

        return self._compute_estimates_and_weights(
            K_yy,
            K_zy,
            K_zz,
            rbf_zy,
            eta,
        )
