from functools import partial

import torch
import mlxp
import matplotlib.pyplot as plt

from hnpe.misc import make_label
from hnpe.inference import run_inference

from viz import get_posterior
from viz import display_posterior, display_posterior_mlxp
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel, prior_ToyModel, get_ground_truth

"""
In this example, we consider the ToyModel setting in which the simulator has
two input parameters [alpha, beta] and generates x = alpha * beta^gamma + eps,
where gamma is a fixed known parameter of the simulator, and eps is a Gaussian
white noise with standard deviation sigma. Because the observation is a product
of two parameters, we may expect an indeterminacy when trying to estimate them
from a given observation xo. To try and break this, we consider that each x0 is
accompanied by a few other observations x1, ..., xN which all share the same
parameter beta but with different values for alpha. Our goal then is to use
this extra information to obtain the posterior distribution of
p(alpha, beta | x0, x1, ..., xN)
"""

def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


@mlxp.launch(config_path='./configs/')
def main(ctx: mlxp.Context):

    cfg = ctx.config
    logger = ctx.logger
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description='Run inference on the Toy Model'
    # )
    # parser.add_argument('--alpha', '-a', type=float, default=0.5,
    #                     help='Ground truth value for alpha.')
    # parser.add_argument('--beta', '-b', type=float, default=0.5,
    #                     help='Ground truth value for beta.')
    # parser.add_argument('--gamma', '-y', type=float, default=1.0,
    #                     help='Ground truth value for gamma.')
    # parser.add_argument('--noise', type=float, default=0.0,
    #                     help='Standard deviation of the Gaussian noise.')
    # parser.add_argument('--summary', '-s', type=str, default='Identity',
    #                     help='Architecture used to compute summary features.')
    # parser.add_argument('--viz', action='store_true',
    #                     help='Show results from previous run.')
    # parser.add_argument('--naive', action='store_true',
    #                     help='Use naive posterior estimation.')
    # parser.add_argument('--aggregate', action='store_true',
    #                     help='Aggregate the extra observations.')
    # parser.add_argument('--round', '-r', type=int, default=0,
    #                     help='Show results from previous inference run.')
    # parser.add_argument('--nextra', '-n', type=int, default=0,
    #                     help='How many extra observations to consider.')
    # parser.add_argument('--ntrials', '-t', type=int, default=1,
    #                     help='How many trials to consider.')
    # parser.add_argument('--nrounds', '-nrd', type=int, default=1,
    #                     help='How many training rounds to consider.')
    # parser.add_argument('--nsim', '-nsr', type=int, default=10000,
    #                     help='How many data to simulate per round.')
    # parser.add_argument('--dry', action='store_true')

    # args = parser.parse_args()

    if cfg.dry:
        # the dryrun serves just to check if all is well
        nrd = 1
        nsr = 10
        maxepochs = 0
        saverounds = False
    else:
        nrd = cfg.nrounds #1 #NBR ROUND
        nsr = cfg.nsim #NBR SIMU PER ROUND
        maxepochs = 100 #None
        saverounds = True

    # setup the parameters for the example
    meta_parameters = {}
    # how many extra observations to consider
    meta_parameters["n_extra"] = cfg.nextra
    # how many trials for each observation
    meta_parameters["n_trials"] = cfg.ntrials
    # what kind of summary features to use
    meta_parameters["summary"] = cfg.summary
    # the parameters of the ground truth (observed data)
    meta_parameters["theta"] = torch.tensor([cfg.alpha, cfg.beta])
    # gamma parameter on x = alpha * beta^gamma + w
    meta_parameters["gamma"] = cfg.gamma
    # standard deviation of the noise added to the observations
    meta_parameters["noise"] = cfg.noise
    # which example case we are considering here
    meta_parameters["case"] = ''.join([
        "Flow/ToyModel_",
        f"naive_{cfg.naive}_",
        f"ntrials_{meta_parameters['n_trials']:02}_",
        f"nextra_{meta_parameters['n_extra']:02}_",
        f"alpha_{meta_parameters['theta'][0]:.2f}_",
        f"beta_{meta_parameters['theta'][1]:.2f}_",
        f"gamma_{meta_parameters['gamma']:.2f}_",
        f"noise_{meta_parameters['noise']:.2f}"])
    # number of rounds to use in the SNPE procedure
    meta_parameters["n_rd"] = nrd
    # number of simulations per round
    meta_parameters["n_sr"] = nsr
    # number of summary features to consider
    meta_parameters["n_sf"] = 1
    # label to attach to the SNPE procedure and use for saving files
    meta_parameters["label"] = make_label(meta_parameters)
    # set prior distribution for the parameters
    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0]))

    # choose how to setup the simulator
    simulator = partial(simulator_ToyModel,
                        n_extra=meta_parameters["n_extra"],
                        n_trials=meta_parameters["n_trials"],
                        p_alpha=prior,
                        gamma=meta_parameters["gamma"],
                        sigma=meta_parameters["noise"])

    # choose the ground truth observation to consider in the inference
    ground_truth = get_ground_truth(meta_parameters, p_alpha=prior)

    # choose how to get the summary features
    summary_net = IdentityToyModel()

    # choose a function which creates a neural network density estimator
    # il s'agit d'une fonction !!!
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=cfg.naive,
                                 aggregate=cfg.aggregate)
    #print(build_nn_posterior)
    # decide whether to run inference or viz the results from previous runs
    if not cfg.viz:
        # run inference procedure over the example
        posteriors = run_inference(simulator=simulator,
                                   prior=prior,
                                   build_nn_posterior=build_nn_posterior,
                                   ground_truth=ground_truth,
                                   meta_parameters=meta_parameters,
                                   summary_extractor=summary_net,
                                   save_rounds=saverounds,
                                   device='cpu',
                                   max_num_epochs=maxepochs)
    else:
        print("avant get posterior")
        posterior = get_posterior(
            simulator, prior, build_nn_posterior,
            meta_parameters, round_=cfg.round
        )
        df, fig, ax = display_posterior_mlxp(posterior, prior, meta_parameters)
        # store the posterior plot in the corresponding logs
        logger.log_artifacts(fig, artifact_name=f"posterior_plot_{nrd}_rounds_{nsr}_simperround_{cfg.nextra}_nextra.png",
                            artifact_type='image')
        logger.register_artifact_type("pickle", save_pickle, load_pickle)
        logger.log_artifacts(df, f"estimated_posterior_samples_{cfg.nextra}_nextra_{cfg.nsim}_sim.pkl", "pickle")
        #ess=logger.load_artifacts(f"estimated_posterior_samples_{cfg.nextra}_nextra_{cfg.nsim}_sim.pkl", "pickle")

if __name__ == "__main__":
    main()

