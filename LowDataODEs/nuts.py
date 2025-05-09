# -*- coding: utf-8 -*-
# Original Author: Philippe Wenk <philippewenk@hotmail.com>
# JAX conversion: [Your Name]
"""
Helper functions to calculate matrices and densities as described in the
FGPGM paper, implemented in JAX for improved performance and automatic differentiation.
"""
import jax
import jax.numpy as jnp
from functools import partial

def getAs(CDashs, DashCs, CPhis, CDoubleDashs):
    """
    Parameters
    ----------
    CDashs:         list of matrices of shape nTime x nTime
                    each entry represents a state. matrices as returned by
                    kernel.getC_PhiDash
    DashCs:         list of matrices of shape nTime x nTime
    CInvs:          list of matrices of shape nTime x nTime
    CDoubleDashs:   list of matrices of shape nTime x nTime

    Returns
    ----------
    A:  list of matrices of shape nTime x nTime
        each entry represents one state
    """
    A = []
    for i in range(len(CDashs)):
        A.append(
            CDoubleDashs[i] - jnp.dot(
                DashCs[i],
                jax.scipy.linalg.solve(CPhis[i], CDashs[i])))
    return A

def getDs(DashCs, CPhis):
    """
    each entry represents a state
    
    Parameters
    ----------
    DashCs:         list of matrices of shape nTime x nTime
    CInvs:          list of matrices of shape nTime x nTime

    Returns
    ----------
    D:  list of function handles
        each entry represents a function to compute the product with matrix D for one state
    """
    D = []
    for i in range(len(DashCs)):
        # Define a JAX function that can be JIT-compiled
        @partial(jax.jit, static_argnums=(1,))
        def getProdWithD(x, index):
            """
            defines a function to get a product of a vector with the matrix D
            """
            return jnp.dot(DashCs[index],
                          jax.scipy.linalg.solve(CPhis[index], x)
                          )
        
        # Partially apply the function with the specific index
        D.append(lambda x, idx=i: getProdWithD(x, idx))
    
    return D

def getLambdaStars(gamma, nTime):
    """
    each entry represents the LambdaStar of one state, which is the noise
    covariance between the ODEs and the GP estimates of the derivatives
    
    Parameters
    ----------
    gamma:  vector of length nStates
            noise estimate on the derivatives
    nTime:  scalar
            amount of time steps in this experiment
    Returns
    ----------
    Lambdas:    list of nStates matrices of shape nTime x nTime    
    """
    gamma = jnp.asarray(gamma)
    Lambdas = []
    for i in range(gamma.shape[0]):
        Lambdas.append(jnp.eye(nTime) * gamma[i])
    return Lambdas

@jax.jit
def _calculate_gp_post_log_state(x_pret, y_pret, CPhi, sigma, include_det):
    """JIT-compilable function for single state GP posterior calculation"""
    prior_contrib = -1./2 * jnp.dot(
        x_pret,
        jax.scipy.linalg.solve(CPhi, x_pret)
    )
    
    det_argument_prior = jax.lax.cond(
        include_det,
        lambda _: -1./2 * jnp.sum(jnp.linalg.slogdet(CPhi)[1]),
        lambda _: 0.0,
        None
    )
    
    difference = x_pret - y_pret
    
    obs_contrib = -sigma**(-2)/2 * jnp.dot(difference, difference)
    
    det_argument_obs = jax.lax.cond(
        include_det,
        lambda _: -1./2 * jnp.sum(jnp.linalg.slogdet(sigma**2 * jnp.eye(CPhi.shape[0]))[1]),
        lambda _: 0.0,
        None
    )
    
    return prior_contrib + det_argument_prior + obs_contrib + det_argument_obs

def calculateGPPostLog(yNormal, xPret, CPhis, sigmas, mean=None, std=None, includeDet=False):
    """
    calculates the logarithm of the GP posterior of the states
    
    y: unfolded observations
    x: unfolded states, pretreated according to normalization/standardization
    mean: unfolded means
    std:  unfolded stds
    CPhis: list of covariance matrices. Stacked results of kernel.getCPhi(time)
    sigmas: vector of length nStates with the (estimated) stds of the
            observation noise
    """
    yNormal = jnp.asarray(yNormal)
    xPret = jnp.asarray(xPret)
    sigmas = jnp.asarray(sigmas)
    
    if mean is None:
        mean = jnp.zeros_like(yNormal)
    if std is None:
        std = jnp.ones_like(yNormal)
        
    yPret = (yNormal - mean) / std
    
    nTime = CPhis[0].shape[0]
    contributions = []
    
    # iterate through states and calculate respective GP posterior equivalents
    for i in range(len(CPhis)):
        startIndex = nTime * i
        endIndex = nTime * (i + 1)
        currentXPret = xPret[startIndex:endIndex]
        currentYPret = yPret[startIndex:endIndex]
        
        contribution = _calculate_gp_post_log_state(
            currentXPret, currentYPret, CPhis[i], sigmas[i], includeDet
        )
        contributions.append(contribution)
    
    return jnp.sum(jnp.array(contributions))

@jax.jit
def _calculate_f_post_log_state(f_state, mean_state, diff, A, Lambda, include_det):
    """JIT-compilable function for single state F posterior calculation"""
    current_sigma = A + Lambda
    
    prob = -1./2 * jnp.dot(
        diff,
        jax.scipy.linalg.solve(current_sigma, diff)
    )
    
    det_term = jax.lax.cond(
        include_det,
        lambda _: -1./2 * jnp.sum(jnp.linalg.slogdet(current_sigma)[1]),
        lambda _: 0.0,
        None
    )
    
    return prob + det_term

def _vmap_f(x_matrix_normal, theta, f):
    """Helper to vectorize the ODE function application"""
    return jax.vmap(lambda x: f(x, theta))(x_matrix_normal.T).T

def calculateFPostLog(f, xPret, theta, As, Ds, Lambdas, mean=None, std=None, includeDet=False):
    """
    calculates the second component of the density function, which corresponds
    to the logarithm of the posterior on F.

    Parameters
    ----------
    f:  function handle
        function representing the ODEs by mapping x[t] and theta[t] to x_dot[t]
        takes x as first and theta as second argument
    xPret:  vector of shape nTime*nStates
        unfolded states. [x1[t0], x1[t1], ..., x1[tEnd], x2[t0]...]
        pretreated as specified by normalize/standardize
    mean:   Vector of same shape as x
            if None, it is assumed that no GP normalization has been done
            if not None, it is expected to contain the mean of each batch of
            observation as entries.
    std:    Vector of same shape as x
            if None, it is assumed that no standardization has been done
            if not None, it is assumed to contain the std of each batch of
            observations as entries
    """
    xPret = jnp.asarray(xPret)
    nTime = As[0].shape[0]
    nStates = len(As)
    
    if mean is None:
        mean = jnp.zeros_like(xPret)
    if std is None:
        std = jnp.ones_like(xPret)
        
    mean = jnp.asarray(mean)
    std = jnp.asarray(std)
    
    # Calculate true states
    xNormal = std * xPret + mean
    
    # Refold the states to get a matrix of shape nStates x nTime
    xMatrixPret = jnp.zeros((nStates, nTime))
    xMatrixNormal = jnp.zeros((nStates, nTime))
    stdMatrix = jnp.zeros((nStates, nTime))
    
    for i in range(nStates):
        xMatrixPret = xMatrixPret.at[i, :].set(xPret[i*nTime:(i+1)*nTime])
        xMatrixNormal = xMatrixNormal.at[i, :].set(xNormal[i*nTime:(i+1)*nTime])
        stdMatrix = stdMatrix.at[i, :].set(std[i*nTime:(i+1)*nTime])
    
    # Calculate derivatives - using vmap for vectorization across time steps
    f_vectorized = lambda x: f(x, theta)
    fMatrix = jnp.zeros_like(xMatrixPret)
    
    # Use vmap to efficiently apply f across all timesteps
    for t in range(nTime):
        fMatrix = fMatrix.at[:, t].set(f(xMatrixNormal[:, t], theta))
    
    # Normalize derivatives
    fMatrix = fMatrix / stdMatrix
    
    # Calculate probabilities for each state
    probabilities = []
    
    for state in range(nStates):
        currentMean = Ds[state](xMatrixPret[state, :])
        currentDiff = fMatrix[state, :] - currentMean
        
        probability = _calculate_f_post_log_state(
            fMatrix[state, :], 
            currentMean,
            currentDiff,
            As[state], 
            Lambdas[state], 
            includeDet
        )
        probabilities.append(probability)
    
    return jnp.sum(jnp.array(probabilities))

def calculateLogDensity(y, x, CPhis, sigmas, f, theta, As, Ds, Lambdas,
                        mean=None, std=None, includeDet=False):
    """
    calculates a function proportional to the log density for given inputs
    
    Parameters
    ----------
    y:          array, unfolded observations
    x:          array, unfolded states, pretreated according to normalization/standardization
    CPhis:      list of covariance matrices
    sigmas:     vector of length nStates with the (estimated) stds of the observation noise
    f:          function representing the ODEs
    theta:      parameters for the ODE
    As:         list of matrices from getAs
    Ds:         list of function handles from getDs
    Lambdas:    list of matrices from getLambdaStars
    mean:       array of shape like y or None
                means of the observations used in calculating the hyperparams
    std:        array of shape like y or None
                stds of the observations used in calculating the hyperparams
    includeDet: boolean, whether to include determinant terms in the calculation
    
    Returns
    -------
    log_density: scalar, proportional to the log density
    """
    scale = 2 * float(x.size)
    y = jnp.asarray(y)
    x = jnp.asarray(x)
    
    if mean is None:
        mean = jnp.zeros_like(y)
    if std is None:
        std = jnp.ones_like(y)
    
    f_post_log = calculateFPostLog(
        f, x, theta, As, Ds, Lambdas, 
        mean=mean, std=std, includeDet=includeDet
    )
    
    gp_post_log = calculateGPPostLog(
        y, x, CPhis, sigmas, 
        mean=mean, std=std, includeDet=includeDet
    )
    
    return -(f_post_log + gp_post_log) / scale