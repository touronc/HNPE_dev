from pathlib import Path

import torch
import sbi
import matplotlib.pyplot as plt
import pandas as pd
from sbi import utils as sbi_utils
from sbi.analysis import plot
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from get_true_samples_nextra_obs import true_marginal_0_obs, true_marginal_alpha_nextra_obs, true_marginal_beta_nextra_obs


def get_posterior(simulator, prior, build_nn_posterior, meta_parameters,
                  round_=0):

    folderpath = Path.cwd() / "results" / meta_parameters["label"]
    print(folderpath)

    # load ground truth
    ground_truth = torch.load(folderpath / "ground_truth.pkl",
                              map_location="cpu")
    # Construct posterior
    batch_theta = prior.sample((2,))
    batch_x = simulator(batch_theta)
    nn_posterior = build_nn_posterior(batch_theta=batch_theta,
                                      batch_x=batch_x)
    nn_posterior.eval()
    posterior = DirectPosterior(
        #method_family="snpe",
        #neural_net=nn_posterior,
        posterior_estimator=nn_posterior,
        prior=prior,
        x_shape=ground_truth["observation"].shape
    )

    # Load learned posterior
    state_dict_path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
    posterior.posterior_estimator.load_state(state_dict_path)

    # Set the default conditioning as the observation
    posterior = posterior.set_default_x(ground_truth["observation"])

    return posterior


def display_posterior(posterior, prior, metaparameters):

    alpha=metaparameters["theta"][0]
    beta=metaparameters["theta"][1]
    n_samples = 100_000
    n_extras = metaparameters["n_extra"] #nb of extra conditional obs
    n_sim = metaparameters["n_sr"] #nbr of simulations per round

    samples = posterior.sample((n_samples,))#.unsqueeze(1) #, sample_with=False)
    df = pd.DataFrame(data=samples, columns=["beta","alpha"])
    df.to_csv(f"results/estimated_posterior_samples_naive_{metaparameters["naive"]}_{n_extras}_nextra_{n_sim}_sim.csv",index=False)
    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    fig, axes = plot.pairplot(samples, limits=xlim)
    condition_title = r"$x_0$"
    if metaparameters["n_extra"]>0:
        condition_title += rf"$, x_1, x_2,...,x_{metaparameters["n_extra"]}$"
    print(condition_title)
    axes[0][0].set_title(r"$p(\alpha|$"+condition_title+")")
#    axes[0][0].set_title(r"$p(\alpha|x_0$)")
    axes[0][0].set_xlabel(r"$\alpha$")
    axes[0][0].axvline(x=alpha, linestyle='dotted', color="orange", lw=2)


    axes[0][1].set_title(r"$p(\alpha,\beta|$"+condition_title+")")
#    axes[0][1].set_title(r"$p(\alpha,\beta|x_0$)")

    axes[0][1].set_xlabel(r"$\alpha$")
    axes[0][1].set_ylabel(r"$\beta$")
    axes[0][1].scatter(x=alpha, y=beta, color="orange")
    axes[0][1].axvline(x=alpha, linestyle='dotted', color="orange")
    axes[0][1].axhline(y=beta, linestyle='dotted', color="orange")

    axes[1][1].set_title(r"$p(\beta|$"+condition_title+")")
   # axes[1][1].set_title(r"$p(\beta|x_0$)")

    axes[1][1].set_xlabel(r"$\beta$")
    axes[1][1].axvline(x=beta, linestyle='dotted', color="orange", lw=2)

    return fig, axes

def display_posterior_mlxp(posterior, prior, metaparameters, num_samples, true_nextra):
    #print("in sampling true nextra",true_nextra)
    alpha=metaparameters["theta"][0]
    beta=metaparameters["theta"][1]
    n_extra = metaparameters["n_extra"] #nb of extra conditional obs
    n_sim = metaparameters["n_sr"] #nbr of simulations per round
    samples = posterior.sample((num_samples,))#.unsqueeze(1) #, sample_with=False)
    #df = pd.DataFrame(data=samples, columns=["beta","alpha"]) 
    df = pd.DataFrame(data=samples, columns=["alpha","beta"]) ### ATTENTION à l'ordre des paramètres dans la df
    
    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    # import seaborn as sns
    # fig1 = plt.figure()
    # sns.kdeplot(samples[:,[0]])
    # param, density_alpha = true_marginal_alpha_nextra_obs(true_nextra)
    # plt.plot(param,density_alpha)
    # fig1.savefig("test_alpha")
    fig, axes = plot.pairplot(samples, limits=xlim, diag="kde")
    condition_title = r"$x_0$"
    if metaparameters["n_extra"]>0:
        condition_title += rf"$, x_1, x_2,...,x_{metaparameters["n_extra"]}$"
    print(condition_title)
    if n_extra==0:
        param, densities = true_marginal_0_obs(true_nextra)
        axes[0][0].plot(param,densities.squeeze(), color="red")
        axes[1][1].plot(param, densities.squeeze(), color="red")
    else:
        param, density_alpha = true_marginal_alpha_nextra_obs(true_nextra)
        axes[0][0].plot(param, density_alpha, color="red")
        param, density_beta = true_marginal_beta_nextra_obs(true_nextra)
        axes[1][1].plot(param, density_beta, color="red") 
    
    axes[0][0].set_title(r"$p(\alpha|$"+condition_title+")")
    axes[0][0].set_xlabel(r"$\alpha$")
    axes[0][0].axvline(x=alpha, linestyle='dotted', color="orange", lw=2)

    axes[0][1].set_title(r"$p(\alpha,\beta|$"+condition_title+")")
    axes[0][1].set_ylabel(r"$\alpha$")
    axes[0][1].set_xlabel(r"$\beta$")
    axes[0][1].scatter(y=alpha, x=beta, color="orange")
    axes[0][1].axvline(x=beta, linestyle='dotted', color="orange")
    axes[0][1].axhline(y=alpha, linestyle='dotted', color="orange")

    axes[1][1].set_title(r"$p(\beta|$"+condition_title+")")
    axes[1][1].set_xlabel(r"$\beta$")
    axes[1][1].axvline(x=beta, linestyle='dotted', color="orange", lw=2)

    return samples, df, fig, axes

def display_posterior_from_file(path):
    """
    plot the marginals and joint posterior distribution from a csv file
    """

    ### /!\ PLEASE change the parameters below to suit your experiment ###

    alpha=0.5 #metaparameters["theta"][0]
    beta=0.5#metaparameters["theta"][1]
    n_extras = 10#metaparameters["n_extra"] #nb of extra conditional obs
    n_sim = 10000 #metaparameters["n_sr"] #nbr of simulations per round

    df_samples = pd.read_csv(path)
    alpha_samples = torch.tensor(df_samples["alpha"].values).unsqueeze(1)
    beta_samples=torch.tensor(df_samples["beta"].values).unsqueeze(1)
    samples = torch.cat((beta_samples,alpha_samples), dim=1)
    xlim = [[0.0,1.0],[0.0,1.0]]
    fig, axes = plot.pairplot(samples, limits=xlim)
    condition_title = r"$x_0$"
    if n_extras>0:
        condition_title += rf"$, x_1, x_2,...,x_{n_extras}$"
    print(condition_title)
    axes[0][0].set_title(r"$p(\alpha|$"+condition_title+")")
#    axes[0][0].set_title(r"$p(\alpha|x_0$)")
    axes[0][0].set_xlabel(r"$\alpha$")
    axes[0][0].axvline(x=alpha, linestyle='dotted', color="orange", lw=2)


    axes[0][1].set_title(r"$p(\alpha,\beta|$"+condition_title+")")
#    axes[0][1].set_title(r"$p(\alpha,\beta|x_0$)")

    axes[0][1].set_xlabel(r"$\alpha$")
    axes[0][1].set_ylabel(r"$\beta$")
    axes[0][1].scatter(x=alpha, y=beta, color="orange")
    axes[0][1].axvline(x=alpha, linestyle='dotted', color="orange")
    axes[0][1].axhline(y=beta, linestyle='dotted', color="orange")

    axes[1][1].set_title(r"$p(\beta|$"+condition_title+")")
   # axes[1][1].set_title(r"$p(\beta|x_0$)")

    axes[1][1].set_xlabel(r"$\beta$")
    axes[1][1].axvline(x=beta, linestyle='dotted', color="orange", lw=2)
    plt.savefig(f"results/posterior_plot_1_rounds_{n_sim}_simperround_{n_extras}_nextra")

    return fig, axes