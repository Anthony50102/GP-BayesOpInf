import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import norm
from gpkernels import TORCH_GP

def period_length_prior(prior: tuple, sampled_time: list, sampled_states: list, save_loc: str, kernel: str = 'cos*rq'):
    """
    Plot Gaussian distribution of period length and visualize GP predictions with/without a period prior.
    """
    # Set consistent seaborn style
    sns.set_theme(style="whitegrid")
    
    # Loop over each state variable
    for state_var in range(len(prior)):
        mean, std_dev = prior[state_var]
        
        # Plot the prior distribution for period length
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        y = norm.pdf(x, mean, std_dev)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=y)
        plt.title(f"Prior Distribution on Period Length for State Variable {state_var}")
        plt.xlabel("Period")
        plt.ylabel("Probability Density")
        plt.axvline(x=mean, color='green', linestyle='--', label='Prior Mean')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, f"period_length_prior_{state_var}.pdf"))
        plt.close()
        
        # Create GP instances
        no_prior_gp = TORCH_GP(kernel=kernel)
        no_prior_gp.instantiate(sampled_time[state_var], sampled_states[state_var])
        
        # Note: if dividing by 2 is intentional, leave as is; otherwise, verify this transformation.
        prior_gp = TORCH_GP(kernel=kernel, period_prior=(mean / 2, std_dev))
        prior_gp.instantiate(sampled_time[state_var], sampled_states[state_var])
        
        # Generate a time domain for predictions
        t = np.linspace(sampled_time[state_var][0], sampled_time[state_var][-1], 500)
        gp_preds = [gp.predict(t) for gp in (no_prior_gp, prior_gp)]
        gp_means = np.array([pred.mean for pred in gp_preds])
        gp_stds = np.array([pred.stddev for pred in gp_preds])
        
        # Plot and compare GP predictions in subplots
        plot_gp_comparison(t, gp_means, gp_stds, sampled_time[state_var],
                           sampled_states[state_var], state_var, prior[state_var], save_loc)
        
def plot_gp_comparison(t, gp_means, gp_stds, sampled_time, sampled_states, state_var, prior_params, save_loc, num_samples=5):
    """Plot comparisons of GP predictions for No Prior and With Period Prior cases."""
    gp_types = ["No Prior", "With Period Prior"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for gp_idx, gp_type in enumerate(gp_types):
        ax = axes[gp_idx]
        # Observed data points
        ax.scatter(sampled_time, sampled_states, color='red', label='Observations', zorder=10)
        
        # GP Mean and 95% confidence interval
        ax.plot(t, gp_means[gp_idx], 'b-', label='GP Mean', linewidth=2)
        ax.fill_between(t, 
                        gp_means[gp_idx] - 2 * gp_stds[gp_idx], 
                        gp_means[gp_idx] + 2 * gp_stds[gp_idx], 
                        color='b', alpha=0.15, label='95% Confidence')
        
        # Draw a vertical line for the expected period (for illustration)
        ax.axvline(x=prior_params[0], color='green', linestyle='--', label='Expected Period')
        
        # Generate and plot sample paths. Ideally, replace with multivariate sampling for correlated samples.
        for i in range(num_samples):
            sample = np.random.normal(gp_means[gp_idx], gp_stds[gp_idx])
            ax.plot(t, sample, 'k-', alpha=0.5, linewidth=1, 
                    label='GP Sample' if i == 0 else None)
        
        ax.set_title(f"State Variable {state_var} - {gp_type}")
        ax.set_xlabel("Time")
        ax.set_ylabel("State Value")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, f"gp_comparison_state{state_var}.pdf"))
    plt.close()
