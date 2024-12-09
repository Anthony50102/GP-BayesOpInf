#! /bin/zsh
# Experiments for GP-BayesOpInf for PDEs with one data trajectory.

set -e

experiment() {
    python3 main.py $@ --ndraws 600 --noopen
}

# Paper plots =================================================================

# Noisy data.
experiment 0.06 200 .03 0400 6 --exportto data/euler/ex1a --ddtdata
experiment 0.06 200 .03 0050 6 --exportto data/euler/ex1b
experiment 0.06 200 .03 3200 6 --exportto data/euler/ex1c
experiment 0.06 200 .01 0400 6 --exportto data/euler/ex1d
experiment 0.06 200 .05 0400 6 --exportto data/euler/ex1e
# Singular value decay
experiment 0.06 200 .03 400 8 --exportto data/euler/ex1r8

# Sparse data.
experiment 0.06 50 .01 0400 6 --exportto data/euler/ex2a --ddtdata
experiment 0.06 50 .01 0050 6 --exportto data/euler/ex2b
experiment 0.06 50 .01 3200 6 --exportto data/euler/ex2c
experiment 0.06 20 .01 0400 6 --exportto data/euler/ex2d
experiment 0.06 80 .01 0400 6 --exportto data/euler/ex2e

# Plot results.
python3 plots_paper.py

exit 0
