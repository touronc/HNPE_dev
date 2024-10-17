from pathlib import Path

import torch
import sbi
from sbi import utils as sbi_utils
from sbi.analysis import plot
from sbi.inference.posteriors.direct_posterior import DirectPosterior


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

    samples = posterior.sample((n_samples,))#.unsqueeze(1) #, sample_with=False)
    print("sample size", samples.size())
    xlim = [[prior.support.base_constraint.lower_bound[i],
             prior.support.base_constraint.upper_bound[i]]
            for i in range(2)]
    fig, axes = plot.pairplot(samples, limits=xlim)

    axes[0][0].set_title(r"$p(\alpha|x_0$)")
    axes[0][0].set_xlabel(r"$\alpha$")
    axes[0][0].axvline(x=alpha, linestyle='dotted', color="orange", lw=2)


    axes[0][1].set_title(r"$p(\alpha,\beta|x_0$)")
    axes[0][1].set_xlabel(r"$\alpha$")
    axes[0][1].set_ylabel(r"$\beta$")
    axes[0][1].scatter(x=alpha, y=beta, color="orange")
    axes[0][1].axvline(x=alpha, linestyle='dotted', color="orange")
    axes[0][1].axhline(y=beta, linestyle='dotted', color="orange")

    axes[1][1].set_title(r"$p(\beta|x_0$)")
    axes[1][1].set_xlabel(r"$\beta$")
    axes[1][1].axvline(x=beta, linestyle='dotted', color="orange", lw=2)

    return fig, axes
