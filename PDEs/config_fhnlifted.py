# config_fhnlifted.py
"""Configuration for FitzHugh-Nagumo experiments with quadratic lifting.

This experiment reduces the lifted variables (q1, q2, q3=q1^2) jointly and
learns a ROM with the quadratic structure
dq/dt = c + Aq + H[q x q] + B[u] + N[u x q].
"""

__all__ = [
    # Simulation specifics
    "spatial_domain",
    "time_domain",
    # Simulation classes
    "monolithic",
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    # GP kernel fitting hyperparameters
    "CONSTANT_VALUE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "N_RESTARTS_OPTIMIZER",
]

import numpy as np

import opinf

import pde_models_fn as pdes


# Simulation specifications  --------------------------------------------------
spatial_domain = np.linspace(0, 1, 512)  # Spatial domain x.
time_domain = np.linspace(0, 4, 401)  # Temporal domain t.
initial_conditions = None
a = 50000.0  # first parameter for Neumann BC.
b = 15.0  # second parameter for Neumann BC.


# Simulation classes ----------------------------------------------------------
class FullOrderModel(pdes.FitzHughNagumo):
    """Full-order model for this problem."""

    def __init__(self):
        """Initialized solver with default parameters."""
        super().__init__(spatial_domain, a=a, b=b)


class Basis(opinf.basis.PODBasis):
    """Basis for states of the form (q1, q2, q1^2).
    A separate POD basis is used for each state variable.
    """

    # def fit(self, states, r):
    #     """Construct the bases."""
    #     q1, q2 = np.split(states, 2, axis=0)

    #     return super().fit(
    #         np.concatenate((q1, q2, q1**2)),
    #         r,
    #     )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_vectors = kwargs['num_vectors']

    def fit(self, states):
        """Construct the bases."""
        q1, q2 = np.split(states, 2, axis=0)

        print(q1.shape, q2.shape)
        return super().fit(
            np.concatenate((q1, q2, q1**2)),
            )


    def compress(self, states):
        """Map high-dimensional states to low-dimensional coordinates."""
        q1, q2 = np.split(states, 2, axis=0)
        return super().compress(
            np.concatenate((q1, q2, q1**2)),
        )

    def decompress(self, states_compressed, **kwargs):
        """Map low-dimensional coordinates to high-dimensional states."""
        q = super().decompress(states_compressed)
        q1, q2, _ = np.split(q, 3, axis=0)
        return np.concatenate((q1, q2))


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for this problem."""

    ivp_method = "Radau"
    input_dimension = 1

    def __init__(self, *args, **kwargs):
        # ensure that the base class sees your default operator string
        kwargs.setdefault('operators', "cAHBN")
        super().__init__(*args, **kwargs)

    @staticmethod
    def input_func(t):
        return FullOrderModel.left_neumann_condition(t, a, b)
    
    @staticmethod
    def input_func_jax(t):
       return FullOrderModel.left_neumann_condition_jax(t, a, b) 

    @staticmethod
    def full_rhs(t):
        pass


monolithic = True


# Gaussian process kernel fitting hyperparameters -----------------------------
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100
