import ode_models as odes
import numpy as np

from config import *

alpha = 1.5#1.5
beta = 1 #1
delta = 1 #1
gamma = 3 #3
x0 = 1
y0 = 1
# TODO - Me thinks?
time_domain = np.linspace(0, 20, 500)
# true_parameters = np.array([1.5, 1, 1, 1.0, 3, 1, 1]) # alpha, beta, delta, gamma, x0, y0
# initial_conditions = np.array([.994, .992])
initial_conditions = np.array([x0, y0])
test_initial_conditions = np.array([.994,.992])

class Model(odes.LotkaVolterra):
    num_equations = 1

    def __init__(self):
        """Set the system parameters."""
        super().__init__([alpha,beta,delta,gamma])

    @staticmethod
    def data_matrix(states: np.ndarray) -> np.ndarray:
        """Construct the 5k x 4 data matrix for the single coupled problem."""
        Prey, Pred = states
        Z = np.zeros_like(Prey)

        data_dSdt = np.column_stack((Prey, -Prey*Pred, Z, Z))
        data_dEdt = np.column_stack((Z, Z, Prey*Pred, -Pred))

        return np.vstack(
            [data_dSdt, data_dEdt]
        )

NUMVARS = len(Model.LABELS)

def DIMFMT(stateindex: int) -> str:
    """String format for state variable index."""
    return Model.LABELS[stateindex]