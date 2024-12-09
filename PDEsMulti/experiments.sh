#! /bin/zsh
# Experiments for GP-BayesOpInf for PDEs with multiple data trajectories.

set -e

python3 main.py 1 20 .05 80 5 --ndraws 600 --noopen --exportto data/heat3/ex3
python3 plots_paper.py
