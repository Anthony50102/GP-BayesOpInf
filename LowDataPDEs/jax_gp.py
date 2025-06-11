"""
JAX-based Gaussian Process Implementation with Derivative Support
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import debug
from functools import partial
from typing import Dict, Tuple, Callable, Optional, Union, List

# Default bounds for hyperparameters
DEFAULT_BOUNDS = {
    'lengthscale': (1e-3, 1e2),  # (lower, upper)
    'variance': (1e-6, 1e2),
    'noise': (1e-8, 1e-0)
}

# ------------------------
# Kernel functions
# ------------------------
@jax.jit
def rbf_kernel(x1: jnp.ndarray,
               x2: jnp.ndarray,
               lengthscale: float,
               variance: float) -> jnp.ndarray:
    """
    RBF (Gaussian) kernel function.
    
    Args:
        x1: Input array of shape [N, D]
        x2: Input array of shape [M, D]
        lengthscale: Length scale parameter
        variance: Variance parameter
    
    Returns:
        Kernel matrix K of shape [N, M] with
        K_ij = variance * exp(-0.5 * ||x1[i]-x2[j]||^2 / lengthscale^2)
    """
    sqdist = jnp.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist / (lengthscale**2))

# ------------------------
# Elementary kernel functions for derivatives
# ------------------------
@jax.jit
def k(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    RBF kernel: variance * exp(-(t1 - t2)^2 / (2 * lengthscale^2))
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Kernel value between t1 and t2
    """
    l = params['lengthscale']
    var = params['variance']
    return var * jnp.exp(-(t1 - t2)**2 / (2 * l**2))

@jax.jit
def c_dash(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d/dt2 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Derivative of kernel with respect to t2
    """
    l2 = params['lengthscale']**2
    return (t1 - t2) / l2 * k(params, t1, t2)

@jax.jit
def dash_c(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d/dt1 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Derivative of kernel with respect to t1
    """
    l2 = params['lengthscale']**2
    return -(t1 - t2) / l2 * k(params, t1, t2)

@jax.jit
def c_double_dash(params: Dict[str, float], t1: float, t2: float) -> float:
    """
    d^2/dt1/dt2 of the RBF kernel
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t1: First input point
        t2: Second input point
        
    Returns:
        Second mixed derivative of kernel with respect to t1 and t2
    """
    l2 = params['lengthscale']**2
    diff = t1 - t2
    return (1 / l2 - diff**2 / l2**2) * k(params, t1, t2)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
@jax.jit
def flatten_time(t: jnp.ndarray) -> jnp.ndarray:
    """
    Return t with shape (n,) no matter if (n,), (n,1) or (1,n) was given.
    
    Args:
        t: Input time array
        
    Returns:
        Flattened time array with shape (n,)
    """
    return jnp.ravel(t)

@jax.jit
def rbf_kernel_no_nugget(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Full n×n RBF kernel matrix K_ij = variance * exp(-(t_i-t_j)^2 / (2*ell^2)).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return params["variance"] * jnp.exp(-diff**2 / (2.0 * ell2))

# ----------------------------------------------------------------------
# Kernel matrix functions - each returns an (n, n) array
# ----------------------------------------------------------------------
@jax.jit
def get_c_phi(params: Dict[str, float], t: jnp.ndarray, nugget: float = 1e-4) -> jnp.ndarray:
    """
    Kernel matrix plus nugget on the diagonal.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        nugget: Small value added to diagonal for numerical stability
        
    Returns:
        Kernel matrix of shape [n, n] with nugget on diagonal
    """
    kmat = rbf_kernel_no_nugget(params, t)
    return kmat + nugget * jnp.eye(kmat.shape[0])

@jax.jit
def get_c_phi_dash(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative with respect to the second time argument (dt2).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Derivative kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return (diff / ell2) * rbf_kernel_no_nugget(params, t)

@jax.jit
def get_dash_c_phi(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative with respect to the first time argument (dt1).
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Derivative kernel matrix of shape [n, n]
    """
    return -get_c_phi_dash(params, t)

@jax.jit
def get_c_phi_double_dash(params: Dict[str, float], t: jnp.ndarray) -> jnp.ndarray:
    """
    Second mixed derivative with respect to both time arguments.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance'
        t: Input time array
        
    Returns:
        Second derivative kernel matrix of shape [n, n]
    """
    t = flatten_time(t)
    diff = t[:, None] - t[None, :]
    ell2 = params["lengthscale"] ** 2
    return (1.0 / ell2 - diff**2 / ell2**2) * rbf_kernel_no_nugget(params, t)

# ------------------------
# Parameter transformations
# ------------------------
def bounded(raw: float, a: float, b: float) -> float:
    """Map real number to interval (a,b) via sigmoid."""
    return a + (b - a) * jax.nn.sigmoid(raw)

def inv_bounded(f: float, a: float, b: float) -> float:
    """Inverse of bounded transformation. Maps (a,b) to real numbers."""
    # f must lie strictly between a and b
    y = (f - a) / (b - a)
    return jnp.log(y / (1 - y))

def raw_to_params(raw: Dict[str, float], 
                  bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS) -> Dict[str, float]:
    """Convert unbounded raw parameters to bounded parameters."""
    return {
        'lengthscale': bounded(raw['u_l'], *bounds['lengthscale']),
        'variance': bounded(raw['u_v'], *bounds['variance']),
        'noise': bounded(raw['u_n'], *bounds['noise']),
    }

def params_to_raw(params: Dict[str, float],
                  bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS) -> Dict[str, float]:
    """Convert bounded parameters to unbounded raw parameters."""
    return {
        'u_l': inv_bounded(params['lengthscale'], *bounds['lengthscale']),
        'u_v': inv_bounded(params['variance'], *bounds['variance']),
        'u_n': inv_bounded(params['noise'], *bounds['noise']),
    }

# ------------------------
# GP Core Functions
# ------------------------
@partial(jax.jit, static_argnums=(3,))
def log_marginal_likelihood(params: Dict[str, float],
                           X: jnp.ndarray,
                           y: jnp.ndarray,
                           standardize: bool) -> float:
    """
    Calculate log marginal likelihood for Gaussian Process.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance', 'noise'
        X: Input data of shape [N, D]
        y: Target data of shape [N]
        standardize: Whether to standardize targets
        
    Returns:
        Log marginal likelihood value
    """
    l = params['lengthscale']
    scale = params['variance']
    noise = params['noise']
    N = X.shape[0]
    K = rbf_kernel(X, X, l, scale) + noise * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    
    # Standardize
    if standardize:
        y = (y - jnp.mean(y)) / jnp.std(y)
    
    # alpha = K^{-1} y via Cholesky solves
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    log_lik = -0.5 * y @ alpha \
              - jnp.sum(jnp.log(jnp.diag(L))) \
              - 0.5 * N * jnp.log(2 * jnp.pi)
    return log_lik

@jax.jit
def gp_predict(params: Dict[str, float],
              X_train: jnp.ndarray,
              y_train: jnp.ndarray,
              X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GP posterior predictive distribution.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance', 'noise'
        X_train: Training inputs of shape [N, D]
        y_train: Training targets of shape [N]
        X_test: Test inputs of shape [M, D]
        
    Returns:
        Tuple of (predictive_mean, predictive_covariance)
        - predictive_mean: Array of shape [M]
        - predictive_covariance: Array of shape [M, M]
    """
    l = params['lengthscale']
    scale = params['variance']
    noise = params['noise']
    
    # Train covariance
    K = rbf_kernel(X_train, X_train, l, scale) + noise * jnp.eye(X_train.shape[0])
    L = jnp.linalg.cholesky(K)
    
    # Compute α = K^{-1} y_train
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    
    # Cross-covariances
    K_s = rbf_kernel(X_train, X_test, l, scale)
    
    # Predictive mean
    mu = K_s.T @ alpha
    
    # Solve L v = K_s for v
    v = jax.scipy.linalg.solve_triangular(L, K_s, lower=True)
    
    # Predictive covariance
    K_ss = rbf_kernel(X_test, X_test, l, scale)
    cov = K_ss - v.T @ v
    
    return mu, cov

@jax.jit
def gp_predict_derivative(params: Dict[str, float],
                        X_train: jnp.ndarray,
                        y_train: jnp.ndarray,
                        X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GP posterior predictive derivative distribution.
    
    Args:
        params: Dictionary with keys 'lengthscale', 'variance', 'noise'
        X_train: Training inputs of shape [N, D]
        y_train: Training targets of shape [N]
        X_test: Test inputs of shape [M, D]
        
    Returns:
        Tuple of (predictive_deriv_mean, predictive_deriv_covariance)
        - predictive_deriv_mean: Array of shape [M]
        - predictive_deriv_covariance: Array of shape [M, M]
    """
    # Ensure inputs are 1D for derivative calculations
    if X_train.shape[1] != 1 or X_test.shape[1] != 1:
        raise ValueError("Derivative predictions only supported for 1D inputs (D=1)")
    
    # Flatten inputs
    X_train_flat = flatten_time(X_train)
    X_test_flat = flatten_time(X_test)
    
    # Train covariance matrix K
    K = get_c_phi(params, X_train_flat, nugget=params['noise'])
    L = jnp.linalg.cholesky(K)
    
    # Cross-covariance derivatives: d/dt_test K(X_train, X_test)
    K_ds = get_c_phi_dash(params, jnp.concatenate([X_train_flat, X_test_flat]))
    K_ds = K_ds[:len(X_train_flat), len(X_train_flat):]
    
    # Compute alpha = K^{-1} y_train
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    
    # Predictive derivative mean
    mu_deriv = K_ds.T @ alpha
    
    # Compute derivative covariance
    v = jax.scipy.linalg.solve_triangular(L, K_ds, lower=True)
    
    # Double derivative kernel for test points
    K_ss_dd = get_c_phi_double_dash(params, X_test_flat)
    
    # Predictive derivative covariance
    cov_deriv = K_ss_dd - v.T @ v
    
    return mu_deriv, cov_deriv

# ------------------------
# GP Training
# ------------------------
def create_step_fn(X: jnp.ndarray, y: jnp.ndarray, optimizer: optax.GradientTransformation, standardize: bool = True):
    """
    Create a step function for GP optimization.
    
    Args:
        X: Input data
        y: Target data (can be multi-dimensional [N_samples, N_points])
        optimizer: Optax optimizer
        standardize: Whether to standardize targets
        
    Returns:
        Step function that takes raw_params and optimizer state
    """
    @jax.jit
    def step(raw_params, opt_state):
        def loss_fn(rp):
            p = raw_to_params(rp)
            loss = 0
            # Handle both single and multiple output dimensions
            if y.ndim == 1:
                loss += log_marginal_likelihood(p, X, y, standardize)
            else:
                for j in range(y.shape[0]):
                    loss += log_marginal_likelihood(p, X, y[j,:], standardize)
            return -1 * loss  # Negative because we want to maximize likelihood
            
        loss, grads = jax.value_and_grad(loss_fn)(raw_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_raw_params = optax.apply_updates(raw_params, updates)
        return new_raw_params, new_opt_state, loss
        
    return step


# ------------------------
# GaussianProcess Class
# ------------------------
class GaussianProcess:
    """
    A lightweight class-based wrapper around the JAX GP implementation.
    This provides a more familiar API while keeping the core as functional JAX code.
    """
    
    def __init__(self, 
                 learning_rate: float = 1e-2, 
                 init_params: Optional[Dict[str, float]] = None,
                 bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
                 standardize: bool = True):
        """
        Initialize a Gaussian Process model.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            init_params: Initial parameters. If None, default parameters are used
            bounds: Parameter bounds for transformation
            standardize: Whether to standardize targets during training
        """
        self.bounds = bounds
        self.standardize = standardize
        
        # Initialize parameters
        if init_params is None:
            self.params = {
                'lengthscale': 1.0,
                'variance': 1.0,
                'noise': 1e-2
            }
        else:
            self.params = init_params
            
        self.raw_params = params_to_raw(self.params, self.bounds)
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = None  # Will be initialized during fit
        
    def fit(self, 
            X: Union[np.ndarray, jnp.ndarray], 
            y: Union[np.ndarray, jnp.ndarray], 
            n_iter: int = 100, 
            verbose: bool = True,
            verbose_interval: int = 10) -> List[float]:
        """
        Fit the GP model to data.
        
        Args:
            X: Input data, shape [N, D]
            y: Target data, shape [N] or [M, N] for multi-output
            n_iter: Number of optimization iterations
            verbose: Whether to print progress
            verbose_interval: Print interval
            
        Returns:
            List of loss values during training
        """
        # Fix time dimensions if needed
        if X.shape == (X.size,):
            X = X[:,None]

        # Convert inputs to JAX arrays if needed
        X = self.prep_array(X)
        y = self.prep_array(y)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.raw_params)
        
        # Create step function
        step_fn = create_step_fn(X, y, self.optimizer, self.standardize)
        
        # Training loop
        losses = []
        for i in range(n_iter):
            self.raw_params, self.opt_state, loss = step_fn(self.raw_params, self.opt_state)
            losses.append(float(loss))
            
            if verbose and (i % verbose_interval == 0 or i == n_iter - 1):
                print(f"Iter {i:3d}, neg-log-lik = {loss:.4f}")
                
        # Update params from raw_params
        self.params = raw_to_params(self.raw_params, self.bounds)
        
        if verbose:
            print("Final parameters:")
            for k, v in self.params.items():
                print(f"  {k}: {v:.6f}")
                
        return losses
    
    def predict(self, 
                X_train: Union[np.ndarray, jnp.ndarray], 
                y_train: Union[np.ndarray, jnp.ndarray], 
                X_test: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the GP model.
        
        Args:
            X_train: Training inputs, shape [N, D]
            y_train: Training targets, shape [N] or [M, N] for multi-output
            X_test: Test inputs, shape [M, D]
            
        Returns:
            Tuple of (predictive_mean, predictive_variance)
        """
        # Convert inputs to JAX arrays if needed
        X_train = self.prep_array(X_train)
        y_train = self.prep_array(y_train)
        X_test = self.prep_array(X_test)
        
        mu, cov = gp_predict(self.params, X_train, y_train, X_test)
        std = jnp.sqrt(jnp.diag(cov))
        
        # Convert to numpy for easier handling outside JAX
        return np.array(mu).flatten(), np.array(std).flatten()
    
    def predict_derivative(self,
                          X_train: Union[np.ndarray, jnp.ndarray],
                          y_train: Union[np.ndarray, jnp.ndarray],
                          X_test: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make derivative predictions with the GP model.
        Only works for 1D inputs (time series).
        
        Args:
            X_train: Training inputs, shape [N, 1]
            y_train: Training targets, shape [N]
            X_test: Test inputs, shape [M, 1]
            
        Returns:
            Tuple of (derivative_mean, derivative_std)
        """
        # Convert inputs to JAX arrays if needed
        X_train = self.prep_array(X_train)
        y_train = self.prep_array(y_train)
        X_test = self.prep_array(X_test)
        
        mu_deriv, cov_deriv = gp_predict_derivative(self.params, X_train, y_train, X_test)
        std_deriv = jnp.sqrt(jnp.diag(cov_deriv))
        
        # Convert to numpy for easier handling outside JAX
        return np.array(mu_deriv).flatten(), np.array(std_deriv).flatten()
    
    def predict_all(self, 
                    X_train: Union[np.ndarray, jnp.ndarray], 
                    y_train: Union[np.ndarray, jnp.ndarray], 
                    X_test: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions and return full covariance.
        
        Args:
            X_train: Training inputs, shape [N, D]
            y_train: Training targets, shape [N] or [M, N] for multi-output
            X_test: Test inputs, shape [M, D]
            
        Returns:
            Tuple of (predictive_mean, predictive_std, predictive_covariance)
        """
        # Convert inputs to JAX arrays if needed
        X_train = self.prep_array(X_train)
        y_train = self.prep_array(y_train)
        X_test = self.prep_array(X_test)
        
        mu, cov = gp_predict(self.params, X_train, y_train, X_test)
        std = jnp.sqrt(jnp.diag(cov))
        
        # Convert to numpy for easier handling outside JAX
        return np.array(mu), np.array(std), np.array(cov)
    
    def predict_derivative_all(self,
                              X_train: Union[np.ndarray, jnp.ndarray],
                              y_train: Union[np.ndarray, jnp.ndarray],
                              X_test: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make derivative predictions and return full covariance.
        Only works for 1D inputs (time series).
        
        Args:
            X_train: Training inputs, shape [N, 1]
            y_train: Training targets, shape [N]
            X_test: Test inputs, shape [M, 1]
            
        Returns:
            Tuple of (derivative_mean, derivative_std, derivative_covariance)
        """
        # Convert inputs to JAX arrays if needed
        X_train = self.prep_array(X_train)
        y_train = self.prep_array(y_train)
        X_test = self.prep_array(X_test)
        
        mu_deriv, cov_deriv = gp_predict_derivative(self.params, X_train, y_train, X_test)
        std_deriv = jnp.sqrt(jnp.diag(cov_deriv))
        
        # Convert to numpy for easier handling outside JAX
        return np.array(mu_deriv), np.array(std_deriv), np.array(cov_deriv)
    
    # Kernel matrix utility methods
    def get_kernel_matrix(self, t: Union[np.ndarray, jnp.ndarray], nugget: float = 1e-4) -> np.ndarray:
        """
        Get the kernel matrix for given time points.
        
        Args:
            t: Time points
            nugget: Small value added to diagonal for numerical stability
            
        Returns:
            Kernel matrix of shape [n, n]
        """
        t = self.prep_array(t)
        return jnp.array(get_c_phi(self.params, flatten_time(t), nugget))
    
    def get_kernel_derivative_dt2(self, t: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        """
        Get the kernel derivative matrix with respect to second time argument.
        
        Args:
            t: Time points
            
        Returns:
            Derivative kernel matrix of shape [n, n]
        """
        t = self.prep_array(t)
        return jnp.array(get_c_phi_dash(self.params, flatten_time(t)))
    
    def get_kernel_derivative_dt1(self, t: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        """
        Get the kernel derivative matrix with respect to first time argument.
        
        Args:
            t: Time points
            
        Returns:
            Derivative kernel matrix of shape [n, n]
        """
        t = self.prep_array(t)
        return jnp.array(get_dash_c_phi(self.params, flatten_time(t)))
    
    def get_kernel_double_derivative(self, t: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
        """
        Get the kernel second mixed derivative matrix.
        
        Args:
            t: Time points
            
        Returns:
            Second derivative kernel matrix of shape [n, n]
        """
        t = self.prep_array(t)
        return jnp.array(get_c_phi_double_dash(self.params, flatten_time(t)))

    def get_As(self, t):
        CDashs = get_c_phi_dash(self.params, t)
        DashCs = get_dash_c_phi(self.params, t) 
        CPhis = get_c_phi(self.params, t)
        CDoubleDashs = get_c_phi_double_dash(self.params, t)
        A = []
        # for i in jnp.arange(len(CDashs)):
        A.append(
            CDoubleDashs - jnp.dot(
                DashCs,
                jnp.linalg.solve(CPhis, CDashs)))
        return A
    
    def get_Ds(self,t):
        """
        each entry represents a state
        
        Parameters
        ----------
        DashCs:         list of matrices of shape nTime x nTime
        CInvs:          list of matrices of shape nTime x nTime

        Returns
        ----------
        D:  list of matrices of shape nTime x nTime
            each entry represents one state
        """
        DashCs = get_dash_c_phi(self.params, t)
        CPhis = get_c_phi(self.params, t)
        D = []
        def getProdWithD(x,):
            """
            defines a function to get a product of a vector with the matrix D
            """
            return jnp.dot(DashCs,
                        jnp.linalg.solve(CPhis, x)
                        )
        D.append(getProdWithD)
        return D
        
    @property
    def hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        return dict(self.params)
    
    def prep_array(self, array: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Prepare array for processing, ensuring correct dimensions."""
        array = jnp.asarray(array)
        if array.ndim == 1:
            array = array[:,None]
        return array