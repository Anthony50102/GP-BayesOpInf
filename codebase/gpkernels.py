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
    
    Supports a prior on the period length to encourage solutions with specific periodicities.
    """
    def __init__(self, period_prior=None, **kwargs):
        """
        Initialize the cosine kernel.
        
        Parameters:
        -----------
        period_prior : tuple or None
            If provided, specifies a Gaussian prior on the period as (mean, std).
            For example, period_prior=(24.0, 2.0) would create a prior centered at 24.0.
        """
        super().__init__(has_lengthscale=False, **kwargs)
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("raw_period", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())
        self.register_constraint("raw_period", gpytorch.constraints.Positive())
        
        # Store the prior information
        self.period_prior = period_prior
    
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
    
    def prior_log_prob(self):
        """
        Calculate the log probability of the current period under the prior.
        
        Returns:
        --------
        log_prob : torch.Tensor
            The log probability of the current period under the prior, or 0 if no prior is set.
        """
        if self.period_prior is None:
            return torch.tensor(0.0, device=self.raw_period.device)
        
        period = self.raw_period_constraint.transform(self.raw_period)
        mean, std = self.period_prior
        
        # Calculate log probability under Gaussian prior
        log_prob = -0.5 * ((period - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)
        return log_prob

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
        print()
        print()
        print(kernels)
        print()
        print()
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

###############################################################################
# New Additive Manual Kernel
###############################################################################

class ManualSumKernel(ManualKernelBase):
    """
    Composite manual kernel that adds the covariances from several base manual kernels.
    """
    def __init__(self, kernels, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)
        self.kernels = torch.nn.ModuleList(kernels)
    
    def forward(self, x1, x2, diag=False, **params):
        covar = None
        for k in self.kernels:
            current = k.compute_covariance(x1, x2)
            covar = current if covar is None else covar + current
        return torch.diag(covar) if diag else covar

###############################################################################
# Kernel Expression Parser and Builder
###############################################################################

class KernelParser:
    """
    A simple recursive descent parser for kernel expressions.
    Supports identifiers (e.g. "rbf"), '+' (addition), '*' (multiplication),
    and parentheses for grouping.
    """
    def __init__(self, s: str):
        self.tokens = self.tokenize(s)
        self.pos = 0
    
    def tokenize(self, s: str):
        tokens = []
        i = 0
        while i < len(s):
            if s[i].isspace():
                i += 1
            elif s[i] in '+*()':
                tokens.append(s[i])
                i += 1
            else:
                j = i
                while j < len(s) and s[j].isalnum():
                    j += 1
                tokens.append(s[i:j].lower())
                i = j
        return tokens
    
    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    
    def consume(self, token: str):
        if self.current_token() == token:
            self.pos += 1
        else:
            raise ValueError(f"Expected token {token} but found {self.current_token()}")
    
    def parse_expression(self):
        node = self.parse_term()
        while self.current_token() == '+':
            self.consume('+')
            right = self.parse_term()
            node = ('+', node, right)
        return node
    
    def parse_term(self):
        node = self.parse_factor()
        while self.current_token() == '*':
            self.consume('*')
            right = self.parse_factor()
            node = ('*', node, right)
        return node
    
    def parse_factor(self):
        token = self.current_token()
        if token == '(':
            self.consume('(')
            node = self.parse_expression()
            self.consume(')')
            return node
        else:
            self.consume(token)
            return token

def build_kernel_from_tree(tree):
    """
    Recursively convert the parse tree into a kernel instance.
    The tree nodes are either:
      - A string: representing a base kernel (e.g., "rbf")
      - A tuple: (operator, left, right)
    """
    token_to_class = {
        "rbf": ManualRBFKernel,
        "rq": ManualRQKernel,
        "cos": ManualCosineKernel,
    }
    
    if isinstance(tree, str):
        if tree not in token_to_class:
            raise ValueError(f"Unknown kernel type: {tree}")
        return token_to_class[tree]()
    
    elif isinstance(tree, tuple):
        op, left, right = tree
        left_kernel = build_kernel_from_tree(left)
        right_kernel = build_kernel_from_tree(right)
        if op == '+':
            return ManualSumKernel([left_kernel, right_kernel])
        elif op == '*':
            return ManualCompositeKernel([left_kernel, right_kernel])
        else:
            raise ValueError(f"Unknown operator: {op}")
    else:
        raise ValueError("Invalid parse tree structure")

def build_manual_kernel(kernel_str: str):
    """
    Build a composite manual kernel from a string expression.
    
    The string can include:
      - Multiplication (e.g., "rbf*rq*cos")
      - Addition (e.g., "rbf+cos")
      - Grouping via parentheses (e.g., "rbf*(rq+cos)")
    
    Supported tokens:
      - "rbf": ManualRBFKernel
      - "rq": ManualRQKernel
      - "cos": ManualCosineKernel
    """
    parser = KernelParser(kernel_str.lower())
    tree = parser.parse_expression()
    if parser.current_token() is not None:
        raise ValueError("Unexpected token at the end of kernel string")
    return build_kernel_from_tree(tree)

###############################################################################
# Generic Exact GP Model
###############################################################################

class GenericExactGP(ExactGP):
    """
    A generic ExactGP model that accepts a custom kernel and mean module.
    Supports kernels with parameter priors.
    """
    def __init__(self, train_x, train_y, likelihood, mean_module=None, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else ConstantMean()
        self.covar_module = kernel if kernel is not None else ManualRBFKernel()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_kernels_prior_log_prob(self):
        """
        Calculate the total log probability of all kernel parameters under their priors.
        
        Returns:
        --------
        total_log_prob : torch.Tensor
            Sum of log probabilities for all kernel parameters with priors.
        """
        total_log_prob = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Process individual kernels
        if hasattr(self.covar_module, "prior_log_prob"):
            total_log_prob += self.covar_module.prior_log_prob()
        
        # Check if this is a composite kernel and process its subkernels
        if hasattr(self.covar_module, "kernels"):
            for kernel in self.covar_module.kernels:
                if hasattr(kernel, "prior_log_prob"):
                    total_log_prob += kernel.prior_log_prob()
        
        return total_log_prob

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

    def instantiate(self, t_training, training_data):
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

    def fit(self, t_training, training_data, error: bool = False):
        self.instantiate(t_training=t_training, training_data=training_data)
        train_x = self.t_training.unsqueeze(-1)
        train_y = self.y
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            
            # Calculate negative log likelihood
            nll = -mll(output, train_y)
            
            # Include prior terms (negative because we're minimizing)
            prior_log_prob = self.model.get_kernels_prior_log_prob()
            loss = nll - prior_log_prob  # Negative log prior for MAP estimation
            
            loss.backward()
            optimizer.step()

        ret_val = self if not error else loss.item()
        return ret_val

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
        y = (self.y - self.model.mean_module.constant).unsqueeze(-1)
        K_yy_inv_y = torch.cholesky_solve(y, L)
        self.state_estimate = self.model.mean_module.constant.detach().numpy() + (kappa_zy @ K_yy_inv_y).squeeze(-1).detach().numpy().astype(np.float64)
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
    def __init__(self, training_iter=100, kernel=None, mean_module=None, period_prior=None):
        super().__init__()
        self.training_iter = training_iter
        self.period_prior = period_prior
        
        # Allow passing a custom kernel as an instance or via a string.
        if isinstance(kernel, str):
            self.kernel = self._build_kernel_with_prior(kernel)
        else:
            self.kernel = kernel if kernel is not None else ManualRBFKernel()
        
        # Optionally allow a custom mean module; default to ConstantMean.
        self.mean_module = mean_module if mean_module is not None else ConstantMean()

    def _build_kernel_with_prior(self, kernel_str):
        """
        Builds a kernel from a string expression and applies priors as needed.
        """
        # First, create the kernel using the existing parser
        kernel = build_manual_kernel(kernel_str)
        
        # Apply period prior to any cosine kernels
        if self.period_prior is not None:
            print("Adding period length prior to GP")
            self._apply_period_prior_to_kernel(kernel)
            
        return kernel
    
    def _apply_period_prior_to_kernel(self, kernel):
        """
        Recursively applies the period prior to any cosine kernels found in a composite kernel.
        """
        if isinstance(kernel, ManualCosineKernel):
            kernel.period_prior = self.period_prior
        
        # Check if this is a composite kernel and apply to its subkernels
        if hasattr(kernel, "kernels"):
            for subkernel in kernel.kernels:
                self._apply_period_prior_to_kernel(subkernel)

    def __str__(self):
        base_str = f"TORCH_GP with kernel: {self.kernel.__class__.__name__}"
        if self.period_prior is not None:
            base_str += f", period prior: mean={self.period_prior[0]}, std={self.period_prior[1]}"
        return base_str

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

    def _print_kernel_params(self, kernel=None, indent=0):
        """Recursively print kernel structure and hyperparameters."""
        if kernel is None:
            if self.model is None:
                print("Model not trained yet. No kernel parameters available.")
                return
            kernel = self.model.covar_module
            print(f"Kernel Structure and Parameters:")
        
        # Print current kernel info
        indent_str = "  " * indent
        kernel_name = kernel.__class__.__name__
        print(f"{indent_str}{kernel_name}")
        
        # Handle specific kernel types
        if isinstance(kernel, ManualRBFKernel):
            outputscale = kernel.raw_outputscale_constraint.transform(kernel.raw_outputscale).item()
            lengthscale = kernel.raw_lengthscale_constraint.transform(kernel.raw_lengthscale).item()
            print(f"{indent_str}  amplitude: {outputscale:.4f}")
            print(f"{indent_str}  lengthscale: {lengthscale:.4f}")
        
        elif isinstance(kernel, ManualRQKernel):
            outputscale = kernel.raw_outputscale_constraint.transform(kernel.raw_outputscale).item()
            alpha = kernel.raw_alpha_constraint.transform(kernel.raw_alpha).item()
            lengthscale = kernel.raw_lengthscale_constraint.transform(kernel.raw_lengthscale).item()
            print(f"{indent_str}  amplitude: {outputscale:.4f}")
            print(f"{indent_str}  alpha: {alpha:.4f}")
            print(f"{indent_str}  lengthscale: {lengthscale:.4f}")
        
        elif isinstance(kernel, ManualCosineKernel):
            outputscale = kernel.raw_outputscale_constraint.transform(kernel.raw_outputscale).item()
            period = kernel.raw_period_constraint.transform(kernel.raw_period).item()
            print(f"{indent_str}  amplitude: {outputscale:.4f}")
            print(f"{indent_str}  period: {period:.4f}")
            if hasattr(kernel, 'period_prior') and kernel.period_prior is not None:
                mean, std = kernel.period_prior
                print(f"{indent_str}  period_prior: mean={mean:.4f}, std={std:.4f}")
        
        # Recursively print subkernels for composite kernels
        if hasattr(kernel, "kernels"):
            print(f"{indent_str}  Subkernels:")
            for subkernel in kernel.kernels:
                self._print_kernel_params(subkernel, indent + 2)

    def print_kernel_params(self):
        """Print the kernel structure and all hyperparameter values."""
        if self.model is None:
            print("Model not trained yet. Run fit() first to train the model.")
            return
        
        # Print likelihood noise parameter
        noise = self.model.likelihood.noise.item()
        print(f"Likelihood noise: {noise:.4f}")
        
        # Print mean module parameter
        if isinstance(self.model.mean_module, ConstantMean):
            mean_constant = self.model.mean_module.constant.item()
            print(f"Mean constant: {mean_constant:.4f}")
        
        # Print kernel parameters
        self._print_kernel_params()