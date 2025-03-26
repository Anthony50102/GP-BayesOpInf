# main.py
"""Do a single numerical experiment from start to finish."""

import os
import itertools
import math
import numpy as np

import opinf

import utils
import config_lotka_volterra as config
import step1_generate_data as step1
import step2_fitgps as step2
import step3_estimate as step3
import step4_plot as step4

def main(
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float,
    num_regression_points: int,
    kernel:str,
    gp_regularizer: float = 1e-8,
    ndraws: int = 100,
    exportto: str = None,
    openonsave: bool = True,
    ensemble: bool = False,
):
    r"""Do a single trial from start to finish (do not save intermediate data).

    Parameters
    ----------
    training_span : (float, float)
        Time domain over which to sample solution data.
    num_samples : int > 0
        Number of snapshots to sample.
    noiselevel : float >= 0
        Percentage of noise applied to the training snapshots.
    num_regression_points : int > 0
        Number of points at which to evaluate the GP state and derivative
        estimates.
    gp_regularizer : float >= 0
        Regularization hyperparameter for the GP inference in inverting for
        the least-squares weight matrix.
    ndraws : int
        Number of draws from the posterior distribution.
    exportto : str
        If given, write experiment data to an HDF5 files with ``exportto`` as
        the prefix.
    openonsave : bool
        If ``True`` (default), open figures as they are created.
    """
    # Report on experimental scenario.
    utils.summarize_experiment(
        training_span=training_span,
        num_samples=num_samples,
        noiselevel=noiselevel,
        num_regression_points=num_regression_points,
        gp_regularizer=gp_regularizer,
        opinf_regularizer=None,
        ndraws=ndraws,
    )

    # Step 1: Generate data ---------------------------------------------------
    sampler = step1.TrajectorySampler(
        training_span=training_span,
        num_samples=num_samples,
        noiselevel=noiselevel,
        num_regression_points=num_regression_points,
        synced=False,
        integersonly=False,
        config=config
    )

    (
        truthmodel,
        time_domain_prediction,
        true_states,
        time_domains_sampled,
        snapshots_sampled,
    ) = sampler.sample()
    true_parameters = np.copy(truthmodel.parameters)

    # Step 2: Fit Gaussian processes to data ----------------------------------
    time_domain_training = np.linspace(
        training_span[0],
        training_span[-1],
        num_regression_points,
    )

    if ensemble == True:
        kernels = ['cos', 'rbf', 'rq']
        operators = ['+', '*']
        combinations = []
        # For each possible number of kernels (1 to len(kernels))
        for r in range(1, len(kernels) + 1):
            # Generate all orderings (permutations) of r kernels from the list.
            for perm in itertools.permutations(kernels, r):
                # If there's only one kernel, just add it.
                if r == 1:
                    combinations.append(perm[0])
                else:
                    # For each placement of operators between the kernels.
                    for ops in itertools.product(operators, repeat=r-1):
                        expr = perm[0]
                        for op, ker in zip(ops, perm[1:]):
                            expr += op + ker
                        combinations.append(expr)

        # Print the list of generated combinations
        min_error = math.inf
        for combo in combinations:
                gps = step2.torch_fit_gaussian_processes(
                    time_domain_training=time_domain_training,
                    time_domains_sampled=time_domains_sampled,
                    snapshots_sampled=snapshots_sampled,
                    gp_regularizer=gp_regularizer,
                    config=config,
                    kernel = combo
                )

            
                # Step 3: Construct the posterior hyperparameters -------------------------
                error , bayesian_model = step3.estimate_posterior(
                    gps=gps,
                    time_domain_prediction=time_domain_prediction,
                    config=config
                )
                if error < min_error:
                    min_kernel = combo
                    min_error = error
        print(f"Ensemble modeling has found best model to be: {min_kernel} with a error of {min_error}")

    else:
        gps = step2.torch_fit_gaussian_processes(
            time_domain_training=time_domain_training,
            time_domains_sampled=time_domains_sampled,
            snapshots_sampled=snapshots_sampled,
            gp_regularizer=gp_regularizer,
            config=config,
            kernel = kernel
        )

    
        # Step 3: Construct the posterior hyperparameters -------------------------
        error , bayesian_model = step3.estimate_posterior(
            gps=gps,
            time_domain_prediction=time_domain_prediction,
            config=config
        )

    utils.summarize_posterior(true_parameters, bayesian_model)

    # Draw samples from the posterior.
    ICs = true_states[:, 0]
    with opinf.utils.TimedBlock("\nsampling posterior distribution"):
        draws = bayesian_model.solution_posterior(
            initial_conditions=ICs,
            timepoints=time_domain_prediction,
            ndraws=ndraws,
        )

    # Step 4: plot results ----------------------------------------------------
    gp_predictions = [gp.predict(time_domain_training) for gp in gps]
    gp_means = np.array([ms.mean for ms in gp_predictions])
    gp_stds=np.array([ms.stddev for ms in gp_predictions])

    plotter = step4.ODEPlotter(
        sampling_time_domain=time_domains_sampled,
        training_time_domain=time_domain_training,
        prediction_time_domain=time_domain_prediction,
        snapshots=snapshots_sampled,
        true_states=true_states,
        gp_means = gp_means,
        gp_stds = gp_stds,
        draws=draws,
        labels=truthmodel.LABELS,
        config=config
    )

    # If desired, export experimental data to HDF5 files for later.
    if exportto is not None:
        os.makedirs(os.path.dirname(exportto), exist_ok=True)
        dfile = f"{exportto}_data.h5"
        with opinf.utils.TimedBlock(f"exporting data to {dfile}"):
            plotter.save(dfile, overwrite=True)

    # Plot and save results.
    with opinf.utils.TimedBlock("\nplotting GP training fit\n"):
        # Gaussian process fit.
        plotter.plot_gp_training_fit(width=3)
        utils.save_figure(f"train_{kernel.replace("*", "_")}.pdf", andopen=openonsave)

        # Bayesian model performance.
        for k, flag in enumerate((True, False)):
            plotter.plot_posterior(individual=flag)
            utils.save_figure(f"predict{k}_{kernel.replace("*", "_")}.pdf", andopen=openonsave)

    # Prediction at different initial conditions.
    if config.test_initial_conditions is None:
        return
    test_trajectory = truthmodel.solve(
        config.test_initial_conditions,
        time_domain_prediction,
        strict=True,
    )
    with opinf.utils.TimedBlock("sampling posterior distribution"):
        draws = bayesian_model.solution_posterior(
            initial_conditions=config.test_initial_conditions,
            timepoints=time_domain_prediction,
            ndraws=ndraws,
        )

    fig = plotter.plot_posterior_newICs(draws, truth=test_trajectory)
    utils.save_figure(f"newtrajectory_{kernel.replace("*", "_")}.pdf", andopen=openonsave, fig=fig)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fname = os.path.basename(__file__)
    spc = " " * len(fname)
    parser.usage = f""" python3 {fname} --help
        python3 {fname} T_MAX NUMSAMPLES NOISELEVEL NUMPTS
                {spc} [--gpreg ETA] [--opinfreg RHO] [--ndraws NDRAWS]
                {spc} [--exportto PREFIX] [--noopen]
    """

    parser.add_argument(
        "t_max",
        type=float,
        help="upper bound on the training time domain",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="number of training snapshots to sample",
    )
    parser.add_argument(
        "noiselevel",
        type=float,
        help="percentage of noise added to training snapshots",
    )
    parser.add_argument(
        "num_regression_points",
        type=int,
        help="number of points to use in the OpInf regression0",
    )
    parser.add_argument(
        "--kernel", "-k",
        type=str,
        default='rbf'
    )
    parser.add_argument(
        "--gpreg",
        type=float,
        # TODO - Maybe reduce this
        default=5e-4,
        help="regularization for GP matrices (eta)",
    )
    parser.add_argument(
        "--ndraws",
        type=int,
        default=100,
        help="number of posterior model draws",
    )
    parser.add_argument(
        "--exportto",
        help="prefix for HDF5 files to save plot data to",
    )
    parser.add_argument(
        "--noopen",
        action="store_true",
        help="do not open figures automatically",
    )
    parser.add_argument(
        "--ensemble",
        action='store_true'
    )

    args = parser.parse_args()
    main(
        training_span=[0, args.t_max],
        num_samples=args.num_samples,
        noiselevel=args.noiselevel,
        num_regression_points=args.num_regression_points,
        kernel=args.kernel,
        gp_regularizer=args.gpreg,
        ndraws=args.ndraws,
        exportto=args.exportto,
        openonsave=not args.noopen,
        ensemble=args.ensemble
    )