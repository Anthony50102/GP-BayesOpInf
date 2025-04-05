#! /bin/zsh
# Experiments for Lotka Volterra for ODE parameter estimation.

set -e

experiment() {
    python3 lotka_volterra_experiment.py $@ --ndraws 60 --noopen
}

# Define the grid
noise_levels=(0.0 .1 .2)
data_points=(20 100 200)

for nl in "${noise_levels[@]}"; do
    for np in "${data_points[@]}"; do
        echo "Running with noise_level=$nl, data_points=$np"
        experiment 008 $np $nl 300 --exportto data/lotka_volterra/ex1a  --ensemble  --custom_save "/Users/anthonypoole/Repositories/GP-BayesOpInf/ODEs/menguo_plots"  # span samples noise regression points
        break
    done
done
