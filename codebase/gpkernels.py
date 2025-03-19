# gpykernels.py
"""
Convenience classes for Gaussian process kernels.
This version provides a few base manual kernels (RBF, RQ, Cosine/Periodic) and
allows you to define a composite kernel on the fly by passing a string, e.g. "rbf*rq*cos".
"""

import math
import abc
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import Kernel

###############################################################################
# Base Class for Manual Kernels
###############################################################################

class ManualKernelBase(Kernel):
    """
    Base class for manual kernels.
    Provides a common compute_covariance method and helper for input preparation.
    """
    def compute_covariance(self, x1, x2):
        covar = self.forward(x1, x2)
        if hasattr(covar, "evaluate"):
            return covar.evaluate()
        return covar

    @staticmethod
    def _prepare_input(x):
        # If x is a 2D tensor with a single feature column, squeeze it.
        if x.dim() == 2 and x.shape[1] == 1:
            return x.squeeze(-1)
        return x

###############################################################################
# Base Manual Kernels
###############################################################################

class ManualRBFKernel(ManualKernelBase):
    """
    Custom kernel implementing a standard RBF (Gaussian) kernel.
    
    k(x, x') = amplitude * exp(-((x - x')^2) / (2 * lengthscale^2))
    """
    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())
    
    def forward(self, x1, x2, diag=False, **params):
        outputscale = self.raw_outputscale_constraint.transform(self.raw_outputscale)
        lengthscale = self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        
        K = self.manual_rbf_kernel(x1, x2, amplitude=outputscale, lengthscale=lengthscale)
        return torch.diag(K) if diag else K

    @staticmethod
    def manual_rbf_kernel(x, x_prime, amplitude, lengthscale):
        x = ManualKernelBase._prepare_input(x)
        x_prime = ManualKernelBase._prepare_input(x_prime)
        diff = x.unsqueeze(1) - x_prime.unsqueeze(0)
        rbf_part = torch.exp(-0.5 * (diff**2) / (lengthscale**2))
        return amplitude * rbf_part


class ManualRQKernel(ManualKernelBase):
    """
    Custom kernel implementing a Rational Quadratic kernel.
    
    k(x, x') = amplitude * (1 + (x - x')^2 / (2 * alpha * lengthscale^2))^(-alpha)
    """
    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("raw_alpha", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.register_constraint("raw_alpha", gpytorch.constraints.Positive())
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())
    
    def forward(self, x1, x2, diag=False, **params):
        outputscale = self.raw_outputscale_constraint.transform(self.raw_outputscale)
        alpha = self.raw_alpha_constraint.transform(self.raw_alpha)
        lengthscale = self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        
        K = self.manual_rq_kernel(x1, x2, amplitude=outputscale, alpha=alpha, lengthscale=lengthscale)
        return torch.diag(K) if diag else K

    @staticmethod
    def manual_rq_kernel(x, x_prime, amplitude, alpha, lengthscale):
        x = ManualKernelBase._prepare_input(x)
        x_prime = ManualKernelBase._prepare_input(x_prime)
        diff = x.unsqueeze(1) - x_prime.unsqueeze(0)
        diff_sq = diff.pow(2)
        rq_part = (1.0 + diff_sq / (2.0 * alpha * lengthscale**2)).pow(-alpha)
        return amplitude * rq_part


class ManualCosineKernel(ManualKernelBase):
    """
    Custom kernel implementing a Cosine kernel (often used for periodic behavior).
    
    k(x, x') = amplitude * cos(pi*(x - x')/period)
    """
    def __init__(self, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("raw_period", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.register_constraint("raw_period", gpytorch.constraints.Positive())
    
    def forward(self, x1, x2, diag=False, **params):
        outputscale = self.raw_outputscale_constraint.transform(self.raw_outputscale)
        period = self.raw_period_constraint.transform(self.raw_period)
        
        K = self.manual_cosine_kernel(x1, x2, amplitude=outputscale, period=period)
        return torch.diag(K) if diag else K

    @staticmethod
    def manual_cosine_kernel(x, x_prime, amplitude, period):
        x = ManualKernelBase._prepare_input(x)
        x_prime = ManualKernelBase._prepare_input(x_prime)
        diff = x.unsqueeze(1) - x_prime.unsqueeze(0)
        cos_part = torch.cos(math.pi * diff / period)
        return amplitude * cos_part

###############################################################################
# Composite Manual Kernel
###############################################################################

class ManualCompositeKernel(ManualKernelBase):
    """
    Composite manual kernel that multiplies several base manual kernels.
    Instead of returning a gpytorch ProductKernel, it implements its own 
    compute_covariance method by multiplying the covariances from each subkernel.
    """
    def __init__(self, kernels, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.kernels = torch.nn.ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        covar = self.compute_covariance(x1, x2)
        return torch.diag(covar) if diag else covar

    def compute_covariance(self, x1, x2):
        # Start with an all-ones matrix (multiplicative identity)
        covar = torch.ones(x1.shape[0], x2.shape[0], device=x1.device, dtype=x1.dtype)
        for k in self.kernels:
            covar = covar * k.compute_covariance(x1, x2)
        return covar

###############################################################################
# Kernel Composition Helper
###############################################################################

def build_manual_kernel(kernel_str: str):
    """
    Build a composite manual kernel from a string.
    The string should contain base tokens separated by '*' (e.g. "rbf*rq*cos").
    
    Supported tokens:
      - "rbf": ManualRBFKernel
      - "rq": ManualRQKernel
      - "cos" or "periodic": ManualCosineKernel
    """
    token_to_class = {
        "rbf": ManualRBFKernel,
        "rq": ManualRQKernel,
        "cos": ManualCosineKernel,
        "periodic": ManualCosineKernel
    }
    tokens = kernel_str.lower().split("*")
    kernels = []
    for token in tokens:
        if token not in token_to_class:
            raise ValueError(f"Unknown kernel type: {token}")
        kernels.append(token_to_class[token]())
    if len(kernels) == 1:
        return kernels[0]
    else:
        return ManualCompositeKernel(kernels)

###############################################################################
# Generic Exact GP Model
###############################################################################

class GenericExactGP(ExactGP):
    """
    A generic ExactGP model that accepts a custom kernel and mean module.
    """
    def __init__(self, train_x, train_y, likelihood, mean_module=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else ConstantMean()
        self.covar_module = kernel if kernel is not None else ManualRBFKernel()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

###############################################################################
# Base GP Wrapper and TORCH_GP
###############################################################################

class TorchBaseGP(abc.ABC):
    """
    Base class for gpytorch Gaussian process regressor wrappers.
    """
    def __init__(self):
        self.model = None
        self.likelihood = None
        self.t_training = None  # training inputs (1D tensor)
        self.y = None           # training targets (1D tensor)
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
        self.likelihood = GaussianLikelihood()
        # Note: the model expects inputs with an extra feature dimension.
        train_x = self.t_training.unsqueeze(-1)
        train_y = self.y
        # Create a GenericExactGP model with the provided (or default) kernel and mean.
        self.model = GenericExactGP(
            train_x, train_y, self.likelihood,
            mean_module=self.mean_module if hasattr(self, 'mean_module') else None,
            kernel=self.kernel if hasattr(self, 'kernel') else None)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, t):
        t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(-1) if not torch.is_tensor(t) else t
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(t_tensor))
            # Attach .std for compatibility.
            pred.std = pred.stddev
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

    @abc.abstractmethod
    def compute_lstsq_matrices(self, t_est, **kwargs):
        r"""Compute data needed for the GP-BayesOpInf least squares."""
        raise NotImplementedError

    def _compute_estimates_and_weights(self,
                                       K_yy: torch.Tensor,
                                       K_zy: torch.Tensor,
                                       K_zz: torch.Tensor,
                                       kappa_zy: torch.Tensor,
                                       eta: float = 1e-8):
        L = torch.linalg.cholesky(K_yy)
        y_unsq = self.y.unsqueeze(-1)
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
            raise ValueError(f"inverse covariance not positive definite, increase eta...previous value {eta}")
        self.sqrtW = (evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T).detach().numpy().astype(np.float64)


class TORCH_GP(TorchBaseGP):
    """
    Gaussian process regressor with a customizable kernel.
    You can specify the kernel either by passing a kernel instance or a string.
    For example: TORCH_GP(training_iter=100, kernel="rbf*rq*cos")
    """
    def __init__(self, training_iter=100, kernel=None, mean_module=None):
        super().__init__()
        self.training_iter = training_iter
        # Allow passing a custom kernel as an instance or via a string.
        if isinstance(kernel, str):
            self.kernel = build_manual_kernel(kernel)
        else:
            self.kernel = kernel if kernel is not None else ManualRBFKernel()
        # Optionally allow a custom mean module; default to ConstantMean.
        self.mean_module = mean_module if mean_module is not None else ConstantMean()

    def __str__(self):
        return f"TORCH_GP with kernel: {self.kernel.__class__.__name__}"

    def compute_lstsq_matrices(self, t_est, eta):
        self.t_estimation = t_est
        t_est_tensor = torch.tensor(t_est, dtype=torch.float32) if not torch.is_tensor(t_est) else t_est
        t_est_tensor.requires_grad_()
        t_train_tensor = self.t_training if torch.is_tensor(self.t_training) else torch.tensor(self.t_training, dtype=torch.float32)
        kernel = self.model.covar_module
        
        rbf_yy = kernel.compute_covariance(t_train_tensor, t_train_tensor)
        rbf_zy = kernel.compute_covariance(t_est_tensor, t_train_tensor)
        rbf_zz = kernel.compute_covariance(t_est_tensor, t_est_tensor)

        def kernel_func(x1):
            return kernel.compute_covariance(x1, t_train_tensor)
        jacobian_K = torch.autograd.functional.jacobian(kernel_func, t_est_tensor)
        dk_dx1_autograd = torch.zeros_like(rbf_zy)
        for i in range(t_est_tensor.shape[0]):
            dk_dx1_autograd[i, :] = jacobian_K[i, :, i]

        mixed_derivs = torch.zeros_like(rbf_zz)
        for i in range(t_est_tensor.shape[0]):
            for j in range(t_est_tensor.shape[0]):
                xi = t_est_tensor[i:i+1].clone().detach().requires_grad_(True)
                yj = t_est_tensor[j:j+1].clone().detach().requires_grad_(True)
                k_val = kernel.compute_covariance(xi, yj)
                grad_x = torch.autograd.grad(k_val, xi, create_graph=True)[0]
                mixed = torch.autograd.grad(grad_x, yj)[0]
                mixed_derivs[i, j] = mixed

        noise_val = self.model.likelihood.noise.item()
        K_yy = rbf_yy + torch.diag(torch.full((rbf_yy.size(0),), noise_val,
                                                dtype=rbf_yy.dtype,
                                                device=rbf_yy.device))
        K_zy = dk_dx1_autograd
        K_zz = mixed_derivs
        self._compute_estimates_and_weights(K_yy, K_zy, K_zz, rbf_zy, eta)
        return self
