# -*- coding: utf-8 -*-
# Author: Philippe Wenk <philippewenk@hotmail.com>
"""
Defines the Kernel class. Instances should be implemented in the Kernel folder.
"""

import numpy as np
from scipy.optimize import basinhopping
from RandomDisplacement import RandomDisplacement

class Kernel(object):
    def __init__(self, theta=None, sigma=None, nugget=1e-4):
        """
        Parameters
        ----------
        theta:  vector of shape n_hyperparameters
                vector containing the hyperparameters of this kernel
        sigma:  scalar
                std of the observation noise
        nugget: scalar
                small float that is added to the diagonal of the kernel matrix
                CPhi to guarantee positive definiteness also numerically.
        """
        self.theta = theta
        self.sigma = sigma
        self.nugget = nugget

    def getNTheta():
        """
        Returns the amount of parameters needed by this kernel in the theta
        vector
        """
        raise NotImplementedError(
            "getNTheta has not been implemented for this kernel.")

    def setHyperparams(self, theta, sigma, newNugget=False):
        """
        Sets the hyperparameters of this kernel
        """
        self.theta = theta
        self.sigma = sigma
        if not (isinstance(newNugget, bool) and not newNugget):
            self.nugget = newNugget

    def negLogLikelihood(self, params, time, normalize, standardize, y):
        """
        calculates the negative Log likelihood needed to maximize the evidence
        of the data for the hyperparameters
        """
        # get C matrix without disturbing current hyperparameters
        sigma = params[-1]
        theta = params[:-1]
        thetaOld = self.theta
        sigmaOld = self.sigma
        self.setHyperparams(theta, sigma)
        C = self.getCPhi(time)
        C = C + sigma**2*np.eye(C.shape[0])
        self.setHyperparams(thetaOld, sigmaOld)
        # calculate actual log likelihood
        if standardize:
            yNorm = (y - np.mean(y)) / np.std(y)
        elif normalize:
            yNorm = (y - np.mean(y))
        else:
            yNorm = y
        sum1 = np.prod(np.linalg.slogdet(C))
        sum2 = np.dot(yNorm, np.linalg.solve(C, yNorm))
        assert sum1 is not np.nan
        assert sum2 is not np.nan
        return (sum1 + sum2) / y.size                     

    def getBounds(self, y, time):
        """
        creates the bounds for the optimization of the hyperparameters.
        
        Parameters
        ----------
        y:          vector
                    observation of the states. Target of the regression
        time:       vector
                    time points of the observations. Input of the regression
        Returns
        ----------
        bounds: list of theta.size + 1 pairs of the form 
                (lowerBound, upperBound), representing the bounds on the 
                kernel hyperparameters in theta, while the last one is the
                bound on sigma
        """
        raise NotImplementedError(
            "getBounds has not yet been implemented for this kernel!"
            )
            
    def _adam_optimize(self, objective_func, x0, bounds, max_iter=1000, 
                      learning_rate=0.01, beta1=0.9, beta2=0.999, 
                      epsilon=1e-8, tol=1e-5, verbose=True,
                      callback=None):
        """
        Adam optimizer implementation for hyperparameter optimization.
        
        Parameters
        ----------
        objective_func : callable
            The objective function to minimize
        x0 : array
            Initial guess
        bounds : list of tuples
            List of (min, max) pairs for each parameter
        max_iter : int
            Maximum number of iterations
        learning_rate : float
            Learning rate for the optimizer
        beta1 : float
            Exponential decay rate for first moment estimates
        beta2 : float
            Exponential decay rate for second moment estimates
        epsilon : float
            Small constant to prevent division by zero
        tol : float
            Tolerance for convergence
        verbose : bool
            If True, print progress
        callback : callable
            Called after each iteration with (x, f(x), accepted)
            
        Returns
        -------
        result : dict
            Dictionary containing optimization results
        """
        # Initialize parameters
        x = np.array(x0, dtype=np.float64)
        bounds = np.array(bounds)
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        # Initialize moment estimates
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        # Initial function evaluation
        best_f = objective_func(x)
        best_x = x.copy()
        
        # For convergence check
        prev_f = best_f
        
        if verbose:
            print(f"Starting Adam optimization with initial value: {best_f}")
        
        # Iteration loop
        for t in range(1, max_iter + 1):
            # Compute gradient using finite differences
            grad = np.zeros_like(x)
            h = np.maximum(1e-6, np.abs(x) * 1e-6)  # Step size for numerical gradient
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += h[i]
                f_plus = objective_func(x_plus)
                
                x_minus = x.copy()
                x_minus[i] -= h[i]
                f_minus = objective_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h[i])
            
            # Update moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Compute update step
            delta = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Update parameters with bounds constraint
            x_new = np.clip(x - delta, lower_bounds, upper_bounds)
            x = x_new
            
            # Evaluate function at new point
            f = objective_func(x)
            
            # Update best solution
            if f < best_f:
                best_f = f
                best_x = x.copy()
                accepted = True
            else:
                accepted = False
            
            # Call callback if provided
            if callback is not None:
                callback(x, f, accepted)
            
            # Print progress
            if verbose and t % 10 == 0:
                print(f"Iteration {t}: f = {f}, best_f = {best_f}")
            
            # Check for convergence
            if abs(f - prev_f) < tol:
                if verbose:
                    print(f"Converged at iteration {t}")
                break
                
            prev_f = f
        
        # Prepare result similar to basinhopping output
        result = {
            'x': best_x,
            'fun': best_f,
            'nit': t,
            'message': 'Optimization terminated successfully.',
            'success': True
        }
        
        return result

    def learnHyperparams(self, theta0, sigma0, y, time, normalize=False,
                         standardize=False, T=1, newNugget=False, anneal=False,
                         annealArgs={}, basinIter=100, optimizer='basinhopping',
                         adam_params=None):
        """
        Learns the hyperparameters by maximizing the marginal likelihood of the
        data y

        Parameters
        ----------
        theta0:     vector
                    initial guess for parameters for optimization
        sigma0:     scalar
                    initial guess for noise for optimization
        y:          vector of length nObs or array of shape nObs x nReps
                    observation of the states. Target of the regression
                    if y is an array, it is assumed that the observations
                    come from different, independent experiments on the same
                    time scale.
        time:       vector of length nObs
                    time points of the observations. Input of the regression
        normalize:  boolean
                    if True, hyperparameters will be optimized for the
                    mean corrected observation.
                    if False, hyperparameters will be optimized directly
        standardize:    boolean
                        if True, hyperparameters will be optimized for the
                        standardized observations. normalize will be ignored
                        if False, hyperparameters will be optimized as
                        specified by normalize keyword.
        T:          scalar
                    Temperature for the basinhopping optimization
        newNugget:  False or float
                    if false, the old nugget will be used
                    if float, the old nugget will be overwritten
                    nugget is the small number that is added to the GP prior
                    covariance matrix to guarantee positive definiteness
                    also numerically
        anneal:     bool
        annealArgs: dict
        basinIter:  scalar
                    if no annealing is performed, basinIter iterations of
                    basinhopping will be done instead
        optimizer:  str
                    The optimizer to use. One of 'basinhopping' or 'adam'.
        adam_params: dict or None
                    Parameters for the Adam optimizer if used. If None, defaults will be used.
                    Valid keys are: max_iter, learning_rate, beta1, beta2, epsilon, tol
        """
        # define optimization target
        def negLogLikelihood(params):
            # for multiple trajectories, just add likelihood of each
            # run. Assumes one GP per trajectory and mean likelihood as
            # optimization target
            likelihoods = 0
            if y.ndim == 1:
                # Handle the case of a single trajectory
                likelihoods = self.negLogLikelihood(
                    params, time, normalize, standardize, y)
            else:
                # Handle multiple trajectories
                for i in np.arange(y.shape[1]):
                    likelihoods += self.negLogLikelihood(
                        params, time, normalize, standardize, y[:, i])
            return likelihoods

        # set optimizer settings
        bounds = self.getBounds(y, time)
        print(bounds)
                  
        # set nugget
        if not (isinstance(newNugget, bool) and not newNugget):
            self.nugget = newNugget
        else:
            print(newNugget)
        
        # Initialize parameters for optimization
        x0 = np.zeros(theta0.size + 1)
        x0[:-1] = theta0
        x0[-1:] = sigma0
        if sigma0 < 1e-3:
            sigma0 = 1e-3
            
        def printAcceptance(x, f, accept):
            if accept:
                print("YES: {} @ {}".format(f, x))
            else:
                print("Nope: {} @ {}".format(f, x))
                
        # Choose the optimizer to use
        if optimizer.lower() == 'adam':
            print("using Adam as hyperparameter optimizer")
            
            # Set default Adam parameters if not provided
            default_adam_params = {
                'max_iter': 1000,
                'learning_rate': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                'tol': 1e-5,
                'verbose': True
            }
            
            # Update with user-provided parameters if any
            if adam_params is not None:
                default_adam_params.update(adam_params)
                
            # Run Adam optimization
            minimum = self._adam_optimize(
                negLogLikelihood, 
                x0, 
                bounds,
                max_iter=default_adam_params['max_iter'],
                learning_rate=default_adam_params['learning_rate'],
                beta1=default_adam_params['beta1'],
                beta2=default_adam_params['beta2'],
                epsilon=default_adam_params['epsilon'],
                tol=default_adam_params['tol'],
                verbose=default_adam_params['verbose'],
                callback=printAcceptance
            )
        else:
            # Use the original basin hopping approach
            print("using L-BFGS-B as hyperparameter optimizer with basin hopping")
            # include method and bounds
            args = dict(method="L-BFGS-B", bounds=bounds)
            # default options from scipy
            options={'disp': None,
                    'maxls': 20,
                    'iprint': -1,
                    'gtol': 1e-05,
                    'eps': 1e-08,
                    'maxiter': 15000,
                    'ftol': 2.220446049250313e-09,
                    'maxcor': 10,
                    'maxfun': 15000}
            # ftol: relative difference in function value accepted for convergence
            options['ftol'] = 2.220446049250313e-09
            # maximum number of function evaluations
            options['maxfun'] = 200000
            # flag to control showing of convergence messages
            args['options'] = options
            
            minimum = basinhopping(negLogLikelihood, x0, T=T,
                                  minimizer_kwargs=args,
                                  take_step=RandomDisplacement(bounds=bounds),
                                  niter=int(basinIter),
                                  callback=printAcceptance
                                  )
                                    
        print("Kernel optimization output: ")
        print(minimum)
        print("\n")
        
        # Extract the optimized parameters
        if optimizer.lower() == 'adam':
            optVector = minimum['x']
        else:
            optVector = minimum.x
            
        self.theta = optVector[:-1]
        self.sigma = optVector[-1]
        
        # check for positive semidefinite
        C = self.getCPhi(time)
        minEigenvalue = np.min(np.linalg.eig(C)[0])
        print("minimum eigenvalue = {}".format(minEigenvalue))
        if (minEigenvalue < 1e-5):
            print("\n\nRECOMMENDATION: USE BIGGER NUGGET\n\n")
        C = C + (self.sigma**2)*np.eye(C.shape[0])
        try:
            # test for psd
            np.linalg.cholesky(C)
        except:
            print("matrix not positive semidefinite")

    def k(self, time1, time2):
        """
        returns the correlation between time1 and time2 for the specific kernel
        this does not yet add the observation noise
        """
        raise NotImplementedError("k has not been implemented for this kernel")
    
    def CDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time2, used in the C_Phi' matrix
        """
        raise NotImplementedError(
            "CDash has not been implemented for this kernel")
    
    def DashC(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time1, used in the 'C_Phi matrix
        """
        raise NotImplementedError(
            "DashC has not been implemented for this kernel")
    
    def CDoubleDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to both times, used in the C_Phi'' matrix
        """
        raise NotImplementedError(
            "CDoubleDash has not been implemented for this kernel")
    
    def getCPhi(self, time):
        """
        returns the correlation matrix of the GP using this kernel
        """
        C_Phi = np.zeros([time.shape[0], time.shape[0]])
        for i in np.arange(time.shape[0]):
            for j in np.arange(time.shape[0]):
                C_Phi[i, j] = self.k(time[i], time[j])
        return C_Phi + self.nugget*np.eye(time.shape[0])

    def getCPhiDash(self, time):
        """
        returns the derivative of C_Phi w.r.t. the second time argument
        """
        C_PhiDash = np.zeros([time.shape[0], time.shape[0]])
        for i in np.arange(time.shape[0]):
            for j in np.arange(time.shape[0]):
                C_PhiDash[i, j] = self.CDash(time[i], time[j])
        return C_PhiDash

    def getDashCPhi(self, time):
        """
        returns the derivative of C_Phi w.r.t. the first time argument
        """
        DashC_Phi = np.zeros([time.shape[0], time.shape[0]])
        for i in np.arange(time.shape[0]):
            for j in np.arange(time.shape[0]):
                DashC_Phi[i, j] = self.DashC(time[i], time[j])
        return DashC_Phi

    def getCPhiDoubleDash(self, time):
        """
        returns the derivative of C_Phi w.r.t. both time arguments
        """
        C_PhiDoubleDash = np.zeros([time.shape[0], time.shape[0]])
        for i in np.arange(time.shape[0]):
            for j in np.arange(time.shape[0]):
                C_PhiDoubleDash[i, j] = self.CDoubleDash(time[i], time[j])
        return C_PhiDoubleDash
    
        
    def predict(self, time_train, y_train, time_test, return_cov=False):
        """
        Gaussian‐process prediction at new points.
        
        Parameters
        ----------
        time_train : array, shape (n_train,)
            The time points you fitted on.
        y_train : array, shape (n_train,)
            The observations at train points.
        time_test : array, shape (n_test,)
            The new time points where you want predictions.
        return_cov : bool, default=False
            If True, also return the posterior covariance matrix.
        
        Returns
        -------
        mu_star : array, shape (n_test,)
            Posterior mean at each time in time_test.
        cov_star : array, shape (n_test, n_test), optional
            Posterior covariance (only if return_cov=True).
        """
        # 1) Build the train/train covariance (with noise)
        K = self.getCPhi(time_train)              # this has nugget on the diagonal
        K += (self.sigma**2) * np.eye(len(time_train))

        # 2) Build the cross‐covariances
        K_star = np.zeros((len(time_train), len(time_test)))
        for i, t_i in enumerate(time_train):
            for j, t_j in enumerate(time_test):
                K_star[i, j] = self.k(t_i, t_j)

        # 3) Build the test/test covariance (no noise)
        K_star_star = np.zeros((len(time_test), len(time_test)))
        for i, ta in enumerate(time_test):
            for j, tb in enumerate(time_test):
                K_star_star[i, j] = self.k(ta, tb)

        # 4) Do the usual Cholesky trick
        L = np.linalg.cholesky(K)                 # K = L Lᵀ
        # Solve α = K^{-1} y  via two triangular solves
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # 5) Compute posterior mean
        mu_star = K_star.T.dot(alpha)

        if not return_cov:
            return mu_star

        # 6) Compute posterior covariance
        #    cov = K**−1_star_star = K_ss − K_*ᵀ K^−1 K_*
        #   but more stably via the intermediate v = L⁻¹ K_star
        v = np.linalg.solve(L, K_star)
        print(L.shape, K_star.shape, v.shape)
        cov_star = K_star_star - v.T.dot(v)

        return mu_star, cov_star