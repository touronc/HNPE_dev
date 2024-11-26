import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from jax import random
import matplotlib.pyplot as plt
import pandas as pd
from sbi.analysis import plot
import torch

numpyro.set_host_device_count(4)
rng_key = random.PRNGKey(1)
np.random.seed(42)


def model(x_obs=None, n_extra=0, eps=0.01):
    """
    A probabilistic model that generates observations based on a set of parameters.

    Parameters:
    -----------
    x_obs : array-like, optional
        Observed data to condition on. Default is None.
    n_extra : int, optional
        Number of additional observations to generate. Default is 0.
    eps : float, optional
        Standard deviation of the Gaussian noise added to the observations. Default is 0.01.

    Returns:
    --------
    None
        This function does not return any value. It defines a probabilistic model using numpyro.
    """

    alpha = numpyro.sample(
        "alpha",
        dist.Uniform(low=jnp.zeros(n_extra+1), high=jnp.ones(n_extra+1))
    )

    beta = numpyro.sample(
        "beta",
        dist.Uniform(low=jnp.zeros(1), high=jnp.ones(1))
    )

    # create z_0, z_1, ..., z_nextra observations, each associated to a
    # different alpha_i but everyone with the same beta
    z = alpha * beta

    # add gaussian perturbation to all observations
    numpyro.sample(
        "obs",
        dist.Normal(z, scale=eps),
        obs=x_obs
    )

def get_posterior_samples(n_extra,true_theta,true_nextra,nb_samples=100000):
    
    rng_key = random.PRNGKey(1)
    # fix the parameters of the ground truth
    alpha_star = jnp.concatenate(
        [jnp.array([true_theta[0]]),
        jnp.array(np.random.rand(n_extra))])
    beta_star = jnp.array([true_theta[1]])

    # generating and saving new observations
    predictive = Predictive(
        condition(model, {"alpha": alpha_star, "beta": beta_star}),
        num_samples=1)
    rng_key, subkey = random.split(rng_key)
    #data = predictive(subkey, n_extra=n_extra)
    # x_obs = data['obs']
    # print("x_obs",jnp.shape(x_obs))
    x_obs = jnp.expand_dims(jnp.array(true_nextra),0)
    #print("x_obs",jnp.shape(x_obs))

    # print("x_obs",jnp.shape(x_obs))

    kernel = NUTS(
        model,
        init_strategy=init_to_value(
            None, values={"alpha": alpha_star, "beta": beta_star}))
    num_chains = 4
    mcmc = MCMC(
        kernel,
        num_warmup=1_000,
        num_chains=num_chains,
        num_samples=int(nb_samples/num_chains),
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(
        rng_key=subkey,
        x_obs=x_obs,
        n_extra=n_extra)

    mcmc_alpha = mcmc.get_samples()['alpha'][:, 0]
    mcmc_beta = mcmc.get_samples()['beta'][:, 0]
    #dict of samples for each value of n_extra
    samples_mcmc = np.array(jnp.stack([mcmc_alpha, mcmc_beta]).T)
    #print(samples_mcmc)
    df = pd.DataFrame(samples_mcmc,columns=["alpha","beta"])
    #df.to_csv(f"true_samples_alpha_{true_theta[0]}_beta_{true_theta[1]}_nextra_{n_extra}.csv",index=False)
    return df, samples_mcmc

def plot_true_posterior(n_extra, true_theta, samples_mcmc):
    # when we want to plot several joint posteriors at the same time
    
    # if len(samples_mcmc) >1: 
    #     fig, ax = plt.subplots(figsize=(13.3, 4.0), ncols=len(samples_mcmc.keys()))
    #     #print("keys",len(samples_mcmc.keys()))
    #     for i, n_extra in enumerate(samples_mcmc.keys()):
    #         ax[i].scatter(x=samples_mcmc[n_extra][:, 0],
    #                     y=samples_mcmc[n_extra][:, 1])
    #         ax[i].scatter(true_theta[0],true_theta[1], color="orange", s=100)
    #         #ax[i].hexbin(x=samples_mcmc[n_extra][:, 0],
    #                     #  y=samples_mcmc[n_extra][:, 1],
    #                     #  extent=(0, 1, 0, 1))
    #         ax[i].set_xlim(0, 1)
    #         ax[i].set_ylim(0, 1)
    #         ax[i].set_xlabel(r"$\alpha$")
    #         ax[i].set_ylabel(r"$\beta$")
    #         ax[i].set_title(f'n_extra = {n_extra}')
    #     plt.savefig("true_posterior_samples_multiple_nextra")
    print(samples_mcmc.shape)
    print(torch.tensor(samples_mcmc).size())
    xlim = [[0.0,1.0],[0.0,1.0]]
    fig, ax = plot.pairplot(samples_mcmc, limits=xlim)
    condition_title = r"$x_0$"
    if n_extra>0:
        condition_title += rf"$, x_1, x_2,...,x_{n_extra}$"
    print(condition_title)
    ax[0][0].set_title(r"$p(\alpha|$"+condition_title+")")
    ax[0][0].set_xlabel(r"$\alpha$")
    ax[0][0].axvline(x=true_theta[0], linestyle='dotted', color="orange", lw=2)

    ax[0][1].axvline(x=true_theta[1], ls='dotted', c='orange', lw=2.0)
    ax[0][1].axhline(y=true_theta[0], ls='dotted', c='orange', lw=2.0)
    ax[0][1].set_ylabel(r"$\alpha$")
    ax[0][1].set_xlabel(r"$\beta$")
    ax[0][1].scatter(true_theta[1],true_theta[0], c='orange', s=100, zorder=2)
    ax[0][1].set_title(r"$p(\beta, \alpha_0|$"+condition_title+")")
    
    ax[1][1].set_title(r"$p(\beta|$"+condition_title+")")
    ax[1][1].set_xlabel(r"$\beta$")
    ax[1][1].axvline(x=true_theta[1], linestyle='dotted', color="orange", lw=2)

    #fig.savefig(fname=f'true_posterior_samples_alpha_{true_theta[0]}_beta_{true_theta[1]}_nextra_{n_extra}.pdf', format='pdf')
    return fig, ax
    #fig.show()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(
#         description='Sample from the true posterior on the Toy Model with n_extra observations'
#     )
#     parser.add_argument('--alpha', '-a', type=float, default=0.5,
#                         help='Ground truth value for alpha.')
#     parser.add_argument('--beta', '-b', type=float, default=0.5,
#                         help='Ground truth value for beta.')
#     parser.add_argument('--n_extra', '-n',  nargs='+', type=int, default=[0, 5, 10],
#                         help='How many extra observations to consider.')
#     parser.add_argument('--nb_samples', '-nsp', type=int, default=100000,
#                         help='How many parameters to sample.')
#     parser.add_argument('--viz', action='store_true',
#                         help='Only show a pairplot of posterior samples from a csv file.')
#     args = parser.parse_args()
#     file = "ToyModel_naive_False_ntrials_01_nextra_10_alpha_0.50_beta_0.50_gamma_1.00_noise_0.01_agg_False.csv"
#     rng_key = random.PRNGKey(1)
#     true_nextra = pd.read_csv(file)["xobs"]
#     n_extra_list = args.n_extra
    
#     true_theta=[args.alpha,args.beta]
#     print(true_theta)
#     samples = get_posterior_samples(n_extra_list,true_theta,true_nextra,nb_samples=args.nb_samples)
#     plot_true_posterior(true_theta,samples)
    

