# Full-order Models

Notebooks demonstrating the ODE and (full-order) PDE models from the numerical examples.

- [`euler.ipynb`](./euler.ipynb): Compressible Euler equations.
- [`heat.ipynb`](./heat.ipynb): Cubic heat equation.
- [`seird.ipynb`](./seird.ipynb): Susceptible-Exposed-Infected-Recovered-Deceased ODE model.

The models themselves are defined in the following files.

- [`pde_models.py`](../PDEs/pde_models.py): Partial differential equation models (Euler, heat).
- [`ode_models.py`](../ODEs/ode_models.py): Ordinary differential equation models (SEIRD).
