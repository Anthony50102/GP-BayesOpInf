# base_ode.py
"""Base class for ODE models."""

__all__ = [
    "SIR",
    "SEIR",
    "SEIRD",
    "SVIR",
    "Lorenz",
    "LotkaVolterra",
]

import abc
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate


# Base classes ================================================================
class _BaseODE(abc.ABC):
    """Base class for systems of ordinary differential equations.

    Child classes must have
        * a ``LABELS`` attribute listing the names of each state variable.
        * a ``_DEFAULT_PARAMETER_VALUES`` attribute of default parameter.
        * a ``derivative()`` method defining dX/dt.

    Parameters
    ----------
    parameters : tuple
        System parameters affecting the definition of dX/dt.
    """

    # Properties --------------------------------------------------------------
    LABELS = NotImplemented  # Names of the state variables.
    _DEFAULT_PARAMETER_VALUES = NotImplemented  # Default system parameters.

    @property
    def num_variables(self) -> int:
        """Number of state variables."""
        return len(self.LABELS)

    @property
    def parameters(self):
        """System parameters."""
        return self.__params

    @parameters.setter
    def parameters(self, params):
        """Set the system parameters."""
        if len(params) != (num_params := len(self._DEFAULT_PARAMETER_VALUES)):
            raise ValueError(f"expected exactly {num_params} parameters")
        self.__params = np.array(params)

    # Constructor -------------------------------------------------------------
    def __init__(self, parameters=None):
        """Set the system parameters."""
        if parameters is None:
            parameters = self._DEFAULT_PARAMETER_VALUES
        self.parameters = parameters

    # Differential equation ---------------------------------------------------
    @abc.abstractmethod
    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (num_variables,) ndarray
            State [X1(t), X2(t), ...] at time ``t``.

        Returns
        -------
        (num_variables,) ndarray
            State derivative [dX1/dt, dX2/dt, ...] at time ``t``
        """
        raise NotImplementedError

    def solve(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        method: str = "RK45",
        rtol: float = 1e-5,
        atol: float = 1e-8,
        **kwargs: dict,
    ) -> np.ndarray:
        """Solve the model with scipy.integrate.solve_ivp().

        Parameters
        ----------
        initial_conditions : (num_variables,) ndarray
            Initial condition to start the simulation from.
        timepoints : (k,) ndarray
            Time domain over which to solve the equations.

        The following are arguments for ``scipy.integrate.solve_ivp()``.

        method : str
            Integration strategy.
        rtol : float > 0
            Relative error tolerance.
        atol : float > 0
            Absolute error tolerance.
        kwargs : dict
            Additional arguments for ``solve_ivp()``.

        Returns
        -------
        Q : (num_variables, k) ndarray
            Solution to the ODE over the discretized space-time domain.
        """
        if len(initial_conditions) != (nvars := self.num_variables):
            raise ValueError(
                f"expected initial conditions for exactly {nvars} variables"
            )

        return integrate.solve_ivp(
            fun=self.derivative,
            t_span=[timepoints[0], timepoints[-1]],
            y0=np.array(initial_conditions),
            method=method,
            t_eval=timepoints,
            rtol=rtol,
            atol=atol,
            **kwargs,
        ).y

    # Noise model -------------------------------------------------------------
    @abc.abstractmethod
    def noise(self, states: np.ndarray, noise_level=0) -> np.ndarray:
        """Add noise to the ODE solution.

        Parameters
        ----------
        states : (num_variables, k) ndarray
            Solution to the ODE over the discretize time domain.
        noise_level : float
            Noise percentage to add to the solution.

        Returns
        -------
        (num_variables, k) ndarray
            Solution array with added noise.
        """
        raise NotImplementedError

    # Visualization -----------------------------------------------------------
    @classmethod
    def plot(cls, time_domain, states, ls=".", ax=None):
        """Plot the ODE solution with state variables overlapping on one axes.
        This is a class method, use plot() for instantiated objects.

        Parameters
        ----------
        time_domain : (k,) ndarray
            Time domain over which to plot the states.
        states : (num_variables, k) ndarray
            State data to plot.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6))

        for statevar, label in zip(states, cls.LABELS):
            ax.plot(time_domain, statevar, ls, lw=2, label=label)

        ax.set_xlim(left=time_domain[0])
        ax.set_xlabel("$t$")
        ax.set_ylabel("States")
        ax.legend()

        return ax.get_figure(), ax

    @classmethod
    def plot_phase(cls, t, states, variables=(0, 1), fig=None):
        """Plot a single trajectory of two state variables.
        This is a static method, use plot_phase() for instantiated objects.

              +----------------------+    +------------+
            x |    First variable    |    |            |
              + ---------------------+    |    Phase   |
              +----------------------+  y |    plot    |
            y |    Second variable   |    |            |
              +----------------------+    +------------+
                        t                       x

        Parameters
        ----------
        t : (k,) ndarray
            Time domain of the trajectory.
        states : (2+, k) ndarray
            State data to plot.
        variables : tuple of 2 ints
            Which state variables to plot in phase space.
        fig : figure with 3 Axes in the grid format drawn above.
        """
        if len(states) != 2:
            states = np.array(
                [
                    states[variables[0]],
                    states[variables[1]],
                ]
            )
        if fig is None:
            # Make a grid of trajectories
            fig = plt.figure(constrained_layout=True, figsize=(9, 4))
            spec = fig.add_gridspec(
                nrows=2,
                ncols=2,
                hspace=0.1,
                wspace=0.15,
                width_ratios=[1.5, 1],
                height_ratios=[1, 1],
            )
            fig.add_subplot(spec[0, 0])
            fig.add_subplot(spec[1, 0])
            fig.add_subplot(spec[:, 1])

        axes = fig.axes
        if len(axes) != 3:
            raise ValueError("figure should have 3 Axes")

        # Plot trajectories.
        axes[0].plot(t, states[0], "C0", lw=1)
        axes[0].plot([t[0]], [states[0, 0]], "ko")
        axes[1].plot(t, states[1], "C1", lw=1)
        axes[1].plot([t[0]], [states[1, 0]], "ko")
        axes[2].plot(states[0], states[1], "C3", lw=1)
        axes[2].plot([states[0, 0]], [states[1, 0]], "ko")

        axes[0].set_xticks([])
        axes[0].set_ylabel(cls.LABELS[variables[0]])

        axes[1].set_xlabel("$t$")
        axes[1].set_ylabel(cls.LABELS[variables[1]])
        fig.align_ylabels([axes[0], axes[1]])

        axes[2].set_xlabel(cls.LABELS[variables[0]])
        axes[2].set_ylabel(cls.LABELS[variables[1]])
        axes[2].set_title("Phase plot")

        return fig


class _InferenceMixin(abc.ABC):
    """Old Mixin with checks for data matrix construction when the parameter
    inference decouples into individual problems.
    """

    @classmethod
    @abc.abstractmethod
    def data_matrices(
        cls,
        states: np.ndarray,
        ddts: np.ndarray,
        weights: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Construct the data matrix for the parameter estimation.

        Parameters
        ----------
        states : (num_variables, k) ndarray
            State snapshots.
        ddts : (num_variables, k) ndarray
            Time derivative measurements for the state snapshots.
        weights : list of num_variables (k, k) ndarrays
            Least-squares weighting matrices.

        Returns
        -------
        List of d (n_i * k, n_p) ndarrays
            Data matrices where sum_i(n_i) = num_variables,
            sum_p(n_p) = num_params.
        List of d (n_i * k,) ndarrays
            Right-hand side regression vectors (ddts minus constant terms)
            such that sum(ni) = num_variables.
        List of d (n_i * k, n_i * k) ndarrays
            Least-squares weighting matrices such that
            sum_i(n_i) = num_variables.
        """
        raise NotImplementedError

    @classmethod
    def _validate_data_matrices_inputs(cls, states, ddts, weights) -> int:
        if len(states) != (NUMVARS := len(cls.LABELS)):
            raise ValueError(
                f"expected data for exactly {NUMVARS} state variables"
            )
        if len(ddts) != NUMVARS:
            raise ValueError(
                f"expected data for exactly {NUMVARS} ddt variables"
            )
        if len(weights) != NUMVARS:
            raise ValueError(f"expected exactly {NUMVARS} weight matrices")
        k = len(states[0])
        for state, ddt, weight in zip(states, ddts, weights):
            if len(state) != k:
                raise ValueError(f"expected {k} state snapshots")
            if len(ddt) != k:
                raise ValueError(f"expected {k} state ddts")
            if weight.shape != (k, k):
                raise ValueError(f"expected {(k, k)} weight matrices")
        return k

    @classmethod
    def _validate_data_matrices_outputs(cls, Ds, rhss, Ws, numsnapshots):
        """Check shape consistency for the outputs of _data_matrices()."""
        if (d := len(Ds)) != (d2 := len(rhss)):
            raise RuntimeError(f"len(rhss) = {d2} != {d} = len(Ds)")
        if (d3 := len(rhss)) != d:
            raise RuntimeError(f"len(Ws) = {d3} != {d} = len(Ds)")
        for Di, ddti, Wi in zip(Ds, rhss, Ws):
            if Di.shape[0] != ddti.shape[0]:
                raise RuntimeError("data matrix & ddt shape[0] not aligned")
            if Di.shape[0] != Wi.shape[0]:
                raise RuntimeError("data matrix & W shape[0] not aligned")
            if Wi.shape[0] != Wi.shape[1]:
                raise RuntimeError("nonsquare weight matrix")

        NUMVARS = len(cls.LABELS)
        NUMPARAMS = len(cls._DEFAULT_PARAMETER_VALUES)
        if sum([Di.shape[0] for Di in Ds]) != numsnapshots * NUMVARS:
            raise RuntimeError("D.shape[0]s do not sum to k*NUMVARS")
        if sum([Di.shape[1] for Di in Ds]) != NUMPARAMS:
            raise RuntimeError("D.shape[1]s do not sum to NUMPARAMS")

        return Ds, rhss, Ws


# SIR models ==================================================================
class _SIRModel(_BaseODE):
    """Base class for SIR-type models."""

    def solve(
        self,
        initial_conditions: np.ndarray,
        timepoints: np.ndarray,
        *,
        strict: bool = False,
        **kwargs: dict,
    ) -> np.ndarray:
        """Solve the model with ``scipy.integrate.solve_ivp()``.

        Parameters
        ----------
        initial_conditions : (num_variables,) ndarray
            Initial condition to start the simulation from.
        timepoints : (k,) ndarray
            Time domain over which to solve the equations.
        stric : bool
            If ``True``, raise an exception if the initial conditions do not
            sum to N.
        kwargs : dict
            Additional arguments for ``solve_ivp()``.

        Returns
        -------
        Q : (num_variables, k) ndarray
            Solution to the ODE over the discretized space-time domain.
        """
        N = self.N if hasattr(self, "N") else 1
        if strict and (total := sum(initial_conditions)) != N:
            raise ValueError(f"initial conditions sum to {total}, not {N}")

        dkwargs = dict(method="RK45", rtol=1e-5, atol=1e-8)
        dkwargs.update(kwargs)
        return _BaseODE.solve(self, initial_conditions, timepoints, **dkwargs)

    def noise(self, states, noise_level: float = 0.0) -> np.ndarray:
        """Add noise to the ODE solution.

        This noise model is chosen so that the noisy states are
        always positive. Noise also corrupts the initial condition.

        Parameters
        ----------
        states : (num_variables, m) ndarray
            Solution array without noise.
        noise_level : float
            Noise percentage to add to the solution.

        Returns
        -------
        (num_variables, m) ndarray
            Solution array with added noise.
        """
        if not noise_level:
            return states

        # Add noise from a truncated normal distribution.
        iszero = np.abs(states) < 5e-16
        noise_std = np.abs(noise_level * states)
        noise_std[iszero] = 0.001
        # minimum value = 0; maximum value = 3 standard deviations
        a = np.minimum(np.zeros(states.shape), -states / noise_std)
        b = np.maximum(np.zeros(states.shape), (1 - states) / noise_std)
        # b = 3 * np.ones(states.shape)
        states_noised = stats.truncnorm.rvs(
            a,
            b,
            loc=states,
            scale=noise_std,
            size=states.shape,
        )
        states_noised[iszero] = 0
        return states_noised


# SIR -------------------------------------------------------------------------
class SIR(_SIRModel):
    """Susceptible-Infected-Recovered (SIR) ODE model.

    dS / dt = -beta S I / N
    dI / dt = (beta S I / N) - gamma I
    dR / dt = gamma I
    """

    LABELS = (
        "Susceptible",
        "Infected",
        "Recovered",
    )

    _DEFAULT_PARAMETER_VALUES = (
        1000.0,  # N
        0.25,  # beta
        0.1,  # gamma
    )

    @property
    def N(self) -> float:
        """Total population."""
        return self.parameters[0]

    @property
    def beta(self) -> float:
        """Infection rate, i.e., the expected number of people an infected
        person infects per day.
        """
        return self.parameters[1]

    @property
    def gamma(self) -> float:
        """Recovery rate, i.e., the proportion of recovered individuals
        per day.
        """
        return self.parameters[2]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (3,) ndarray
            State values at time t, i.e., [S(t), I(t), R(t)].

        Returns
        -------
        (3,) ndarray
            State derivative [dS/dt, dI/dt, dR/dt].
        """
        S, I, _ = state
        N, beta, gamma = self.parameters

        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I

        return np.array([dSdt, dIdt, dRdt])


class SIR2(_SIRModel):
    """Susceptible-Infected-Recovered (SIR) ODE model, reparameterized to have
    only two parameters.
    """

    _DEFAULT_PARAMETER_VALUES = (
        0.00025,  # p1 = beta / N
        0.1,  # p2 = gamma
    )

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (3,) ndarray
            State values at time t, i.e., [S(t), I(t), R(t)].

        Returns
        -------
        (3,) ndarray
            State derivative [dS/dt, dI/dt, dR/dt].
        """
        S, I, _ = state
        p1, p2 = self.parameters

        dSdt = -p1 * S * I
        dIdt = p2 * S * I - p2 * I
        dRdt = p2 * I

        return np.array([dSdt, dIdt, dRdt])

    @staticmethod
    def convert_parameters(parameter_values: np.ndarray) -> np.ndarray:
        """Convert the parameters of SEIRD to those of SEIRD2."""
        N, beta, gamma = parameter_values
        return np.array([beta / N, gamma])

    @staticmethod
    def _data_matrix(states: np.ndarray) -> np.ndarray:
        """Construct the 3k x 2 data matrix for the single coupled problem."""
        S, I, _ = states
        SI = S * I
        Z = np.zeros_like(S)
        data_dSdt = np.column_stack((-SI, Z))
        data_dIdt = np.column_stack((SI, -I))
        data_dRdt = np.column_stack((Z, I))
        return np.vstack((data_dSdt, data_dIdt, data_dRdt))


# SEIR ------------------------------------------------------------------------
class SEIR(_SIRModel):
    """Susceptible-Exposed-Infected-Recovered COVID-19 model.

    dS / dt = -beta S I / N
    dE / dt = (beta S I / N) - epsilon E
    dI / dt = epsilon E - gamma I
    dR / dt = gamma I
    """

    LABELS = (
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
    )

    _DEFAULT_PARAMETER_VALUES = (
        1000.0,  # N
        0.25,  # beta
        0.1,  # gamma
        0.1,  # eps
    )

    @property
    def N(self) -> float:
        """Total population."""
        return self.parameters[0]

    @property
    def beta(self) -> float:
        """Infection rate, i.e., the expected number of people an infected
        person infects per day.
        """
        return self.parameters[1]

    @property
    def gamma(self) -> float:
        """Recovery rate, i.e., the proportion of recovered individuals
        per day.
        """
        return self.parameters[2]

    @property
    def eps(self) -> float:
        """Incubation period for exposed individuals."""
        return self.parameters[3]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (4,) ndarray
            State values at time t, i.e., [S(t), E(t), I(t), R(t)].

        Returns
        -------
        (4,) ndarray
            State derivative [dS/dt, dE/dt, dI/dt, dR/dt].
        """
        S, E, I, _ = state
        N, beta, gamma, eps = self.parameters
        epsE = eps * E

        dSdt = -beta * S * I / N
        dEdt = -dSdt - epsE
        dRdt = gamma * I
        dIdt = epsE - dRdt

        return np.array([dSdt, dEdt, dIdt, dRdt])


# SEIRD -----------------------------------------------------------------------
class SEIRD(_SIRModel):
    """Susceptible-Exposed-Infections-Recovered-Deceased (SEIRD) CoVid model.

    dS / dt = -beta S I / N
    dE / dt = (beta S I / N) - delta E
    dI / dt = delta E - (1 - alpha) gamma I - alpha rho I
    dR / dt = (1 - alpha) gamma I
    dD / dt = alpha rho I
    """

    # LABELS = (
    #     "Susceptible",
    #     "Exposed",
    #     "Infected",
    #     "Recovered",
    #     "Deceased",
    # )

    LABELS = (
        r"$q_{S}(t)$",
        r"$q_{E}(t)$",
        r"$q_{I}(t)$",
        r"$q_{R}(t)$",
        r"$q_{D}(t)$",
    )

    _DEFAULT_PARAMETER_VALUES = (
        1000.0,  # N
        0.25,  # beta
        0.1,  # delta
        0.1,  # gamma
        0.01,  # alpha
        0.05,  # rho
    )

    @property
    def N(self) -> float:
        """Total population."""
        return self.parameters[0]

    @property
    def beta(self) -> float:
        """Infection rate, i.e., the expected number of people an infected
        person infects per day."""
        return self.parameters[1]

    @property
    def delta(self) -> float:
        """Recovery rate, i.e., the proportion of recovered individuals
        per day.
        """
        return self.parameters[2]

    @property
    def gamma(self) -> float:
        """Incubation period for exposed individuals."""
        return self.parameters[3]

    @property
    def alpha(self) -> float:
        """Fataility rate due to the infection."""
        return self.parameters[4]

    @property
    def rho(self) -> float:
        """Inverse of the average number of days for an infected person to die
        if they do not recover.
        """
        return self.parameters[5]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (5,) ndarray
            State values at time t, i.e., [S(t), E(t), I(t), R(t), D(t)].

        Returns
        -------
        (5,) ndarray
            State derivative [dS/dt, dE/dt, dI/dt, dR/dt, dD/dt].
        """
        S, E, I, _, _ = state
        N, beta, delta, gamma, alpha, rho = self.parameters
        deltaE = delta * E

        dSdt = -beta * S * I / N
        dEdt = -dSdt - deltaE
        dDdt = alpha * rho * I
        dRdt = (1 - alpha) * gamma * I
        dIdt = deltaE - dRdt - dDdt

        return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])


class SEIRD2(_SIRModel):
    """Susceptible-Exposed-Infections-Recovered-Deceased (SEIRD) CoVid model,
    reparameterized to have only four parameters.

        dS / dt = -p1 S I
        dE / dt = p1 S I - p2 E
        dI / dt = p2 E - p3 I - p4 I
        dR / dt = p3 I
        dD / dt = p4 I

    In terms of the SEIRD class, we have
        * p1 = beta / N,
        * p2 = delta
        * p3 = (1 - alpha) gamma
        * p4 = alpha rho.
    """

    LABELS = (
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Deceased",
    )

    _DEFAULT_PARAMETER_VALUES = (
        0.00025,  # p1 = beta / N
        0.10000,  # p2 = delta
        0.09900,  # p3 = (1 - alpha) gamma
        0.00500,  # p4 = alpha rho
    )

    def __init__(self, parameters: np.ndarray = None):
        """Set the system parameters."""
        self.N = 1
        if parameters is not None and len(parameters) == 6:
            self.N = parameters[0]
            parameters = self.convert_parameters(parameters)
        return _SIRModel.__init__(self, parameters=parameters)

    @staticmethod
    def convert_parameters(parameter_values: np.ndarray) -> np.ndarray:
        """Convert the parameters of SEIRD to those of SEIRD2."""
        N, beta, delta, gamma, alpha, rho = parameter_values
        return np.array([beta / N, delta, (1 - alpha) * gamma, alpha * rho])

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (5,) ndarray
            State values at time t, i.e., [S(t), E(t), I(t), R(t), D(t)].

        Returns
        -------
        (5,) ndarray
            State derivative [dS/dt, dE/dt, dI/dt, dR/dt, dD/dt].
        """
        S, E, I, _, _ = state
        p1, p2, p3, p4 = self.parameters
        deltaE = p2 * E

        dSdt = -p1 * S * I
        dEdt = -dSdt - deltaE
        dRdt = p3 * I
        dDdt = p4 * I
        dIdt = deltaE - dRdt - dDdt

        return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])


# SVIR ------------------------------------------------------------------------
class SVIR(_SIRModel):
    """Susceptible-Vaccinated-Infected-Recovered (SVIR) CoVid model.

    (SVIR) CoVid model.

        dS / dt = -(beta S I / N) - nu S
        dV / dt = nu S - (betaV V I / N) - gammaV V
        dI / dt = (beta S I / N) + (betaV V I / N) - gamma I
        dR / dt = gammaV V + gamma I
    """

    LABELS = [
        "Susceptible",
        "Vaccinated",
        "Infected",
        "Recovered",
    ]

    _DEFAULT_PARAMETER_VALUES = (
        1000.0,  # N
        0.1,  # nu
        0.25,  # beta
        0.05,  # betaV
        0.1,  # gamma
        0.3,  # gammaV
    )

    @property
    def N(self) -> float:
        """Total population."""
        return self.parameters[0]

    @property
    def nu(self) -> float:
        """Rate at which susceptible population becomes vaccinated."""
        return self.parameters[1]

    @property
    def beta(self) -> float:
        """Expected number of susceptible people an infected person infects
        per day (including vaccinated and unvaccinated people).
        """
        return self.parameters[2]

    @property
    def betaV(self) -> float:
        """Expected number of vaccinated people an infected person infects
        per day.
        """
        return self.parameters[3]

    @property
    def gamma(self) -> float:
        """Proportion of recovered individuals per day from the infected
        population (vaccinated and unvaccinated).
        """
        return self.parameters[4]

    @property
    def gammaV(self) -> float:
        """Proportion of recovered individuals per day from the vaccinated
        population (by gaining immunity).
        """
        return self.parameters[5]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (4,) ndarray
            State values at time t, i.e., [S(t), V(t), I(t), R(t)].

        Returns
        -------
        (4,) ndarray
            State derivative [dS/dt, dV/dt, dI/dt, dR/dt].
        """
        S, V, I, _ = state
        N, nu, beta, betaV, gamma, gammaV = self.parameters

        nuS = nu * S
        bSIN = beta * S * I / N
        bVIN = betaV * V * I / N
        gI = gamma * I
        gV = gammaV * V

        dSdt = -bSIN - nuS
        dVdt = nuS - bVIN - gV
        dIdt = bSIN + bVIN - gI
        dRdt = gV + gI

        return np.array([dSdt, dVdt, dIdt, dRdt])


class SVIR2(_SIRModel):
    """Susceptible-Vaccinated-Infected-Recovered (SVIR) CoVid model
    with an alternative parameterization.

        dS / dt = -p1 S I - p3 S
        dV / dt = p3 S - p2 V I - p5 V
        dI / dt = p1 S I + p2 V I - p4 I
        dR / dt = p5 V + p4 I

    p1 = beta / N
    p2 = betaV / N
    p3 = nu
    p4 = gamma
    p5 = gammaV
    """

    LABELS = [
        "Susceptible",
        "Vaccinated",
        "Infected",
        "Recovered",
    ]

    _DEFAULT_PARAMETER_VALUES = (
        0.00025,  # p1 = beta / N
        0.00005,  # p2 = betaV / N
        0.1,  # p3 = nu
        0.1,  # p4 = gamma
        0.3,  # p5 = gammaV
    )

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (4,) ndarray
            State values at time t, i.e., [S(t), V(t), I(t), R(t)].

        Returns
        -------
        (4,) ndarray
            State derivative [dS/dt, dV/dt, dI/dt, dR/dt].
        """
        S, V, I, _ = state
        p1, p2, p3, p4, p5 = self.parameters

        nuS = p3 * S
        bSIN = p1 * S * I
        bVIN = p2 * V * I
        gI = p4 * I
        gV = p5 * V

        dSdt = -bSIN - nuS
        dVdt = nuS - bVIN - gV
        dIdt = bSIN + bVIN - gI
        dRdt = gV + gI

        return np.array([dSdt, dVdt, dIdt, dRdt])


# Lorenz equations ============================================================
class Lorenz(_BaseODE):
    """Lorenz-1963 system.

    dX / dt = sigma (y - x)
    dY / dt = x (rho - z) - y
    dZ / dt = xy - beta z
    """

    LABELS = (
        "$X(t)$",
        "$Y(t)$",
        "$Z(t)$",
    )

    _DEFAULT_PARAMETER_VALUES = (
        10,  # sigma
        28,  # rho
        8 / 3.0,  # beta
    )

    @property
    def sigma(self) -> float:
        return self.parameters[0]

    @property
    def rho(self) -> float:
        return self.parameters[1]

    @property
    def beta(self) -> float:
        return self.parameters[2]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (3,) ndarray
            State values at time t, i.e., [X(t), Y(t), Z(t)].

        Returns
        -------
        (3,) ndarray
            State derivative [dX/dt, dY/dt, dZdt].
        """
        X, Y, Z = state

        dXdt = self.sigma * (Y - X)
        dYdt = self.rho * X - (X * Z) - Y
        dZdt = (X * Y) - (self.beta * Z)

        return np.array([dXdt, dYdt, dZdt])

    def noise(self, states, noise_level: float = 0.0) -> np.ndarray:
        """Add noise to the ODE solution.

        Parameters
        ----------
        noise_level : float
            Noise percentage to add to the solution (truncated normal).
        rseed : int or None
            Random seed for the additive noise.

        Returns
        -------
        (num_variables, k) ndarray
            Solution array with added noise.
        """
        if not noise_level:
            return states

        # Add noise from a Normal distribution.
        maxval = np.abs(states - np.mean(states)).max()
        noise_std = noise_level * maxval
        thenoise = np.random.normal(0, noise_std, size=states.shape)
        return states + thenoise

    def jacobian(self, t, state):
        """Compute the Jacobian of the state at the given time / state."""
        X, Y, Z = state
        return np.array(
            [
                [-self.sigma, self.sigma, 0],
                [self.rho - Z, -1, X],
                [Y, X, -self.beta],
            ]
        )


# Lotka-Volterra predator-prey system =========================================
class LotkaVolterra(_BaseODE):
    """Lotka-Volterra prey-predator model.

    dX / dt = alpha X - beta X Y    (prey)
    dY / dt = deta X Y - gamma Y    (predator)
    """

    LABELS = (
        "Prey",
        "Predator",
    )

    _DEFAULT_PARAMETER_VALUES = (
        1.1,  # alpha
        0.4,  # beta
        0.1,  # delta
        0.4,  # gamma
    )

    @property
    def alpha(self) -> float:
        """Maximum per capita growth rate for the prey."""
        return self.parameters[0]

    @property
    def beta(self) -> float:
        """Effect of the presence of predators on the prey population."""
        return self.parameters[1]

    @property
    def delta(self) -> float:
        """Effect of the presence of prey on the predator population."""
        return self.parameters[2]

    @property
    def gamma(self) -> float:
        """Preditor per capita death rate."""
        return self.parameters[3]

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the derivative of the state at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        state : (2,) ndarray
            State values at time t, i.e., [X(t), Y(t)].

        Returns
        -------
        (2,) ndarray
            State derivative [dX/dt, dY/dt].
        """
        X, Y = state
        alpha, beta, delta, gamma = self.parameters
        XY = X * Y

        dXdt = alpha * X - beta * XY
        dYdt = delta * XY - gamma * Y

        return np.array([dXdt, dYdt])

    def jacobian(self, t, state):
        """Compute the Jacobian of the state at the given time / state."""
        alpha, beta, delta, gamma = self.parameters
        x, y = state
        return np.array(
            [[alpha - beta * y, -beta * x], [delta * y, delta * x - gamma]]
        )

    def noise(self, states, noise_level: float = 0.0) -> np.ndarray:
        """Add noise to the ODE solution.

        Parameters
        ----------
        noise_level : float
            Noise percentage to add to the solution (truncated normal).

        Returns
        -------
        (num_variables, k) ndarray
            Solution array with added noise.
        """
        if not noise_level:
            return states

        # Add noise from a Normal distribution.
        noise_std = noise_level * np.mean(states, axis=1)
        thenoise = np.array(
            [
                np.random.normal(0, noise_std[0], size=states.shape[1]),
                np.random.normal(0, noise_std[1], size=states.shape[1]),
            ]
        )
        noised = states + thenoise

        # Force the noised states to be positive.
        noised[noised <= 0] = 0.001
        return noised
