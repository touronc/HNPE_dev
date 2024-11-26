from functools import partial
from jax import random

import torch
import mlxp
import matplotlib.pyplot as plt
import pandas as pd
from hnpe.misc import make_label
from hnpe.inference import run_inference

from viz import get_posterior
from viz import display_posterior_mlxp, display_posterior_from_file
from posterior import build_flow, IdentityToyModel
from simulator import simulator_ToyModel, prior_ToyModel, get_ground_truth
from get_true_posterior_samples import sample_true_posterior, pairplot_samples
from c2st_analysis import c2st_score_df
from get_true_samples_nextra_obs import get_posterior_samples, plot_true_posterior


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
import numpyro

numpyro.set_host_device_count(4)

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
    #torch.manual_seed(42)

    cfg = ctx.config
    logger = ctx.logger

    if cfg.dry:
        # the dryrun serves just to check if all is well
        nrd = 1
        nsr = 10
        maxepochs = 0
        saverounds = False
    else:
        nrd = cfg.nrounds #1 #NBR ROUND
        nsr = cfg.nsim #NBR SIMU PER ROUND
        maxepochs = 500 #None
        saverounds = True
        num_samples = 10000

    # setup the parameters for the example
    meta_parameters = {}
    # which kind of flow to build
    meta_parameters["naive"] = cfg.naive
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
    # if meta_parameters["n_extra"]>1:
    #     print(ground_truth["observation"].squeeze())
    #     df = pd.DataFrame(ground_truth["observation"].squeeze(), columns=["xobs"])
    #     df.to_csv(meta_parameters["label"].split("/")[1]+f"_agg_{cfg.aggregate}.csv", index=False)
    
    # choose how to get the summary features
    summary_net = IdentityToyModel()

    # choose a function which creates a neural network density estimator
    # il s'agit d'une fonction !!!
    build_nn_posterior = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=cfg.naive,
                                 aggregate=cfg.aggregate)
    # decide whether to run inference or viz the results from previous runs
    #if not cfg.viz:
        # run inference procedure over the example
    run_inference(simulator=simulator,
                            prior=prior,
                            build_nn_posterior=build_nn_posterior,
                            ground_truth=ground_truth,
                            meta_parameters=meta_parameters,
                            summary_extractor=summary_net,
                            save_rounds=saverounds,
                            device='cpu',
                            max_num_epochs=maxepochs)
    #print("avant get posterior")
    posterior = get_posterior(
        simulator, prior, build_nn_posterior,
        meta_parameters, round_=cfg.round
    )

    # sample from the estimated posterior and plot the distributions
    estim_samples, df, fig, ax = display_posterior_mlxp(posterior, prior, meta_parameters, num_samples)
    print(estim_samples.size())
    logger.log_artifacts(fig, artifact_name=f"posterior_plot_naive_{cfg.naive}_{nrd}_rounds_{nsr}_simperround_{cfg.nextra}_nextra.png",
                        artifact_type='image')
    logger.register_artifact_type("pickle", save_pickle, load_pickle)
    logger.log_artifacts(df, f"estimated_posterior_samples_naive_{cfg.naive}_{cfg.nextra}_nextra_{cfg.nsim}_sim.pkl", "pickle")

    # simulate true posterior samples and store them
    true_nextra = ground_truth["observation"].squeeze()
    df_true_samples, true_samples = get_posterior_samples([meta_parameters["n_extra"]],meta_parameters["theta"],true_nextra,num_samples)
    logger.log_artifacts(df_true_samples, f"true_posterior_samples_{cfg.noise}_scale_{cfg.nextra}_nextra.pkl", "pickle")
    
    # plot the true posterior
    fig1, ax1 = plot_true_posterior(meta_parameters["theta"],true_samples)
    logger.log_artifacts(fig1, artifact_name=f"true_plot_scale_{cfg.noise}_{cfg.nextra}_nextra.png",
                        artifact_type='image')
   
    # compute the c2st score between the true and estimated samples
    # print("C2ST computation running :")
    # acc = c2st_score_df(df_true_samples, df)
    # logger.log_metrics({"accuracy":acc.item()}, log_name="c2st")
    print("Variance computation :")
    true_samples = torch.cat((torch.tensor(df_true_samples["alpha"]).unsqueeze(1),torch.tensor(df_true_samples["beta"]).unsqueeze(1)),dim=1)
    true_var = torch.var(true_samples, dim=0)
    logger.log_metrics({"true_variance_alpha":true_var[0].item()}, log_name="c2st")
    logger.log_metrics({"true_variance_beta":true_var[1].item()}, log_name="c2st")
    estimated_var = torch.var(estim_samples, dim=1)
    logger.log_metrics({"estimated_variance_alpha":estimated_var[0].item()}, log_name="c2st")
    logger.log_metrics({"estimated_variance_beta":estimated_var[1].item()}, log_name="c2st")




if __name__ == "__main__":
    main()
    