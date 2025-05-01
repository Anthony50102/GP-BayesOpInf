# -*- coding: utf-8 -*-
# Author: Philippe Wenk <philippewenk@hotmail.com>

from scipy.integrate import odeint
import numpy as np
import os
from matplotlib import pyplot as plt

class Experiment(object):
    def __init__(self):
        pass
    
    def f(self, x, theta):
        """
        representing the ODEs. Will return a vector with same shape as the
        state vector x, containing the derivatives of each state w.r.t. time
        for the given ODE parameters theta
        """
        raise NotImplementedError(
            "f has not been implemented for this Experiment")

    def getConstraints(self, nStates, nParams):
        # currently untested and unused
        """
        returns a dict of functions that receive a vector with flattened
        states and parameters, representing constraints as required by the
        optimizer
        (See: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize)
        
        Input vector of the functions will be a vector [unfoldedX, theta]
        where theta are the ODE parameters as taken by theta and unfoldedX are
        the states unfolded like this:
        unfoldedX = [x1[t0], x1[t1], ..., x1[tEnd], x2[t0]...]
        """
        raise NotImplementedError(
            "constraints have not been implemented for this Experiment")

    def getBounds(self, nStates, nParams):
        """
        returns a list of bounds for states and parameters to constrain
        optimization
        """
        raise NotImplementedError(
            "bounds have not been implemented for this Experiment")

#    def sampleTrajectory(self, XInit, tEnd, dt, theta, obsNoiseStd):
#        """
#        Creates a trajectory using a numerical integrator
#        
#        Parameters
#        ----------
#        XInit:          vector of length nStates
#                        initial values of the states at time zero
#        tEnd:           scalar
#                        end time of the experiment
#        dt:             scalar
#                        time steps at which an observation should be sampled
#        theta:          vector of length nParameters
#                        parameters for the ODE
#        obsNoiseStd:    scalar
#                        std of the noise on observations
#        Returns
#        ----------
#        x:  array of shape nTime x nStates
#            true states as returned by the integrator
#        y:  array of shape nTime x nStates
#            noisy observations of the true states
#        """
#        def fODE(x, time):
#            return self.f(x, theta)
#        time = np.arange(0, tEnd+0.5*dt, dt)
#        x = odeint(fODE, XInit, time, rtol=1e-8, mxstep=5000000) # huge for stiff problems
#        noise = np.random.randn(x.shape[0], x.shape[1])
#        noise = noise*obsNoiseStd
#        y = x + noise
#        return x, y
    
    def sampleTrajectoryNonUniform(self, XInit, theta, time,
                                obsNoiseStd=None, SNR=None,
                                noisePct=None, plotting=None):
        """
        Creates a trajectory using a numerical integrator

        Parameters
        ----------
        XInit:      vector of length nStates
                    initial values of the states at time zero
        theta:      vector of ODE parameters
        time:       array of time points at which to observe
        obsNoiseStd: scalar or None
                        absolute noise std to use (overrides SNR & noisePct)
        SNR:        scalar or None
                    signal-to-noise ratio (std(signal)/std(noise))
        noisePct:   scalar or None
                    if provided, noise std = noisePct/100 * mean(signal)
                    over the time series
        plotting:   None or str
                    directory to save state vs time plots

        Returns
        -------
        x : ndarray, shape (nTime, nStates)
            true states
        y : ndarray, shape (nTime, nStates)
            noisy observations
        """
        def fODE(x, t):
            return self.f(x, theta)

        # integrate
        x = odeint(fODE, XInit, time, rtol=1e-8, mxstep=5000000)

        # prepare noise‐std per state per time
        if obsNoiseStd is not None:
            # absolute noise override
            obsStds = np.ones_like(x) * obsNoiseStd

        elif noisePct is not None:
            # percent‐of‐mean noise
            means = np.mean(x, axis=0)                     # shape (nStates,)
            pct_stds = (noisePct/100.0) * means            # shape (nStates,)
            obsStds = np.repeat(pct_stds[np.newaxis, :],   # shape (nTime, nStates)
                                x.shape[0], axis=0)

        else:
            # fallback to SNR
            if SNR is None:
                raise ValueError("Either obsNoiseStd, noisePct, or SNR must be provided")
            sig_stds = np.std(x, axis=0)                   # shape (nStates,)
            noise_stds = sig_stds / np.sqrt(SNR)           # shape (nStates,)
            obsStds = np.repeat(noise_stds[np.newaxis, :],
                                x.shape[0], axis=0)

        # sample noise and add to signal
        noise = np.random.randn(*x.shape) * obsStds
        y = x + noise

        # optional plotting (unchanged)
        if plotting:
            if not os.path.exists(plotting):
                os.makedirs(plotting)
            for s in range(x.shape[1]):
                fig, ax = plt.subplots()
                ax.scatter(time, y[:, s], marker='.', s=50, linewidths=1)
                ax.plot(time, x[:, s], 'r-', linewidth=2)
                ax.set_xlabel("time", fontsize=16)
                ax.set_ylabel(f"state {s+1}", fontsize=16)
                plt.tight_layout()
                fig.savefig(os.path.join(plotting, f"state{s+1}.png"), dpi=300)
                plt.close(fig)

        return x, y
