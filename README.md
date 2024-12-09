# GP-BayesOpInf

This repository contains the source code for the numerical experiments in the paper

[**Bayesian learning with Gaussian processes for low-dimensional representations of time-dependent nonlinear systems**](https://arxiv.org/abs/2408.03455)

by\
[S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ) (Sandia National Laboratories),\
[A. Chaudhuri](https://scholar.google.com/citations?user=oGL9YJIAAAAJ) (The University of Texas at Austin),\
[K. E. Willcox](https://kiwi.oden.utexas.edu/) (The University of Texas at Austin), and\
[M. Guo](https://scholar.google.com/citations?user=eON6MykAAAAJ) (Lund University).

<details><summary>BibTex</summary><pre>
@misc{mcquarrie2024gpbayesopinf,
    title = {Bayesian learning with {G}aussian processes for low-dimensional representations of time-dependent nonlinear systems},
    author = {Shane A. McQuarrie and Anirban Chaudhuri and Karen E. Willcox and Mengwu Guo},
    year = {2024},
    eprint = {2408.03455},
    archivePrefix = {arXiv},
}
</pre></details>

## Contents

- [**codebase/**](./codebase/): implementation of the main elements of GP-BayesOpInf.
- [**models/**](./models/): full-order models used to generate data for the numerical experiments.
- [**PDEs/**](./PDEs/): implementation of GP-BayesOpInf for PDE problems with a single trajectory of training data.
- [**PDEsMulti/**](./PDEsMulti/): implementation of GP-BayesOpInf for PDE problems with multiple trajectories of training data.
- [**ODEs/**](./ODEs/): implementation of GP-based Bayesian parameter estimation for ODE problems.

## Installation

This repository uses the standard Python scientific stack (NumPy, SciPy, Scikit-Learn, etc.) and the [`opinf`](https://willcox-research-group.github.io/rom-operator-inference-Python3) package.
We recommend installing the required packages in a new [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```shell
$ conda deactivate
$ conda create -n gpbayesopinf python=3.12
$ conda activate gpbayesopinf
(gpbayesopinf) $ pip install -r requirements.txt
```

## Reproducing Numerical Results

Each of the three numerical experiments detailed in the paper is contained in its own folder.

- Compressible Euler equations: [**PDEs/**](./PDEs/)
- Nonlinear diffusion-reaction equation: [**PDEs/**](./PDEsMulti/)
- SEIRD epidemiological model: [**ODEs/**](./ODEs/)

To reproduce the figures in the paper, navigate to the directory and run `experiments.sh`.

```shell
cd PDEs/
./experiments.sh
```
