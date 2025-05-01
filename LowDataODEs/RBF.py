# -*- coding: utf-8 -*-
# Author: Philippe Wenk <philippewenk@hotmail.com>
"""
Implements the necessary functions of the Kernel class for the RBF kernel,
with an added option to use Adam optimizer.
"""
import numpy as np
from Kernel import Kernel

class RBF(Kernel):
    def __init__(self, theta=None, sigma=None, nugget=1e-4):
        """
        Parameters
        ----------
        theta:  vector of shape n_hyperparameters
                vector containing the hyperparameters of this kernel
                theta[0] is the multiplicative constant
                theta[1] is the lengthscale
        sigma:  scalar
                std of the observation noise
        nugget: scalar
                small float that is added to the diagonal of the kernel matrix
                CPhi to guarantee positive definiteness also numerically.
        """
        self.theta = theta
        self.sigma = sigma
        self.nugget = nugget
        
    def getNTheta(self):
        """
        Returns the amount of parameters needed by this kernel in the theta
        vector
        """
        return 2
        
    def k(self, time1, time2):
        """
        returns the correlation between time1 and time2 for the specific kernel
        this does not yet add the observation noise
        """
        time1 = float(time1)
        time2 = float(time2)
        return self.theta[0]*np.exp(-(time1 - time2)**2/(2*self.theta[1]**2))
    
    def CDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time2, used in the C_Phi' matrix
        """
        return 1./self.theta[1]**2*(time1 - time2) * self.k(time1, time2)
    
    def DashC(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to time1, used in the 'C_Phi matrix
        """
        return -1./self.theta[1]**2*(time1 - time2) * self.k(time1, time2)
        
    def CDoubleDash(self, time1, time2):
        """
        returns the derivative of the correlation between time1 and time2 with
        respect to both times, used in the C_Phi'' matrix
        """
        return (1./self.theta[1]**2 - (time1 - time2)**2/self.theta[1]**4)* \
            self.k(time1, time2)
            
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
        upperBoundSigmaF = (np.max(y) - np.min(y))**2
        upperBoundLengthscale = time[1]*100
        upperBoundStd = np.max(y) - np.min(y)
        lowerBoundLengthscale = time[1]
        bounds = [(1e-4, upperBoundSigmaF),
                  (lowerBoundLengthscale, upperBoundLengthscale),
                  (1e-3, upperBoundStd)
                  ]
        print(bounds)
        return bounds
        
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