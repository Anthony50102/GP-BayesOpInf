#! /bin/zsh
# Experiments for GP-BayesOpInf for ODE parameter estimation.

set -e

experiment() {
    python3 main.py $@ --ndraws 600 --noopen
}

# Noisy data.
experiment 090 090 .10 360 --exportto data/seird/ex1a
experiment 060 060 .10 240 --exportto data/seird/ex1c
experiment 120 120 .10 480 --exportto data/seird/ex1d

# Sparse data.
experiment 120 010 .05 480 --exportto data/seird/ex2a
experiment 060 010 .05 240 --exportto data/seird/ex2c
experiment 090 010 .05 360 --exportto data/seird/ex2d

python3 plots_paper.py
