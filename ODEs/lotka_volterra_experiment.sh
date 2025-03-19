#! /bin/zsh
# Experiments for GP-BayesOpInf for ODE parameter estimation.

set -e

experiment() {
    python3 lotka_volterra_experiment.py $@ --ndraws 60 --noopen
}

# Noisy data.
experiment 007 100 .3 300 --exportto data/lotka_volterra/ex1a # span samples noise regression points
# experiment 060 060 .10 240 --exportto data/seird/ex1c
# experiment 120 120 .10 480 --exportto data/seird/ex1d

# # Sparse data.
# experiment 120 010 .05 480 --exportto data/seird/ex2a
# experiment 060 010 .05 240 --exportto data/seird/ex2c
# experiment 090 010 .05 360 --exportto data/seird/ex2d

# python3 plots_paper.py lotka_volterra
