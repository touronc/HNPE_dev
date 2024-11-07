# This code will implement a simulator for the throwball example
# The idea is that we have a projectile being launched with some initial
# angle and velocity and we want to know how far it will land from the origin.
# The projectile is subject to air drag proportional to its velocity. The
# equations describing this dynamics can be found in this video:
# https://www.youtube.com/watch?v=Tr_TpLk3dY8

from jax import numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from numpyro.handlers import condition
import pandas as pd
import torch
import numpy as np
from sbi.analysis import plot


def model_hnpe(x_obs=None, solver='analytic'):
    """
    define the simulator for the toy example of HNPE
    x = alpha*beta
    The parameters are theta = (alpha, beta)
    """
    print("scale model", scale)
    # parameters alpha, beta from uniform distributions
    theta = numpyro.sample("theta", dist.Uniform(low=jnp.array([0.0, 0.0]),
                     high=jnp.array([1.0, 1.0])))

    alpha = theta[0]
    beta = theta[1]
  
    if solver == 'analytic': #get a sample from a gaussian N(alpha*beta,scale^2)
        x = alpha*beta
    numpyro.sample("obs", dist.Normal(x,scale=scale),obs=x_obs)
    #print("obs",numpyro.sample("obs", dist.Normal(x,scale=0.10),obs=x_obs))


def generate_posterior_samples(
        rng_key, x_obs, theta_obs, num_samples=100_000):
    """
    generate samples from the posterior distribution p(theta|x)
    with MCMC procedure for a given obs x
    """

    samples_mcmc = {}

    #use no U-turn sampler as transition kernel for MCMC
    kernel = NUTS(
        model_hnpe,
        init_strategy=init_to_value(None, values={'theta': theta_obs})
    )
    num_chains = 4
    mcmc = MCMC(
        kernel,
        num_warmup=2000, #nbr of iterates from which the chain is supposed to have converged to the target distribution
        num_chains=num_chains, #nbr of chains in parallel
        num_samples=int(num_samples/num_chains), #each chain provides a prop of final samples
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(
        rng_key=subkey,
        x_obs=x_obs) #all chains start from xobs ?

    samples_mcmc = mcmc.get_samples()['theta']

    samples = jnp.asarray(samples_mcmc)
    return samples


def sample_true_posterior(rng_key, theta_obs_list, num_samples):
    """sample from the true posterior with MCMC 
    and store the samples in a csv file
    """
    df_list = []
    for i in range(len(theta_obs_list)):

        #parameter theta_0 = (alpha_0, beta_0)
        theta_obs = jnp.array(theta_obs_list[i])

        predictive = Predictive(
            condition(model_hnpe, {"theta": theta_obs}),
            num_samples=1)
        rng_key, subkey = random.split(rng_key)

        #corresponding obs x_0
        data = predictive(subkey)
        x_obs = data['obs']
        print("xobs",x_obs)
        #generates samples from the true posterior wrt x_0
        rng_key, subkey = random.split(rng_key)
        samples = generate_posterior_samples(
            rng_key=subkey,
            x_obs=x_obs,
            theta_obs=theta_obs,
            num_samples=num_samples
        )
        df = pd.DataFrame(data=samples, columns=["alpha","beta"])

        #record the samples for different theta_0
        df['example'] = [i] * len(samples)
        df_list.append(df)

    df = pd.concat(df_list)
    df.to_csv(f"true_posterior_samples_{scale}.csv",index=False)
    return df

def pairplot_samples(theta_obs, df_samples, resample=False):
    """
    Plot the marginals and the joint posterior distribution 
    and save the figure
    """
    if not resample :
        df_samples = pd.read_csv(f"true_posterior_samples_alpha_{theta_obs[1]}_beta_{theta_obs[0]}_scale_{scale}.csv")
    alpha = torch.tensor(df_samples["alpha"].values).unsqueeze(1)
    beta=torch.tensor(df_samples["beta"].values).unsqueeze(1)
    samples = torch.cat((beta,alpha), dim=1)
    #fig, ax = plt.subplots(figsize=(5.4, 5.4))
    #dfi = df[df['example'] == i]
    xlim = [[0.0,1.0],[0.0,1.0]]
    fig, ax = plot.pairplot(samples, limits=xlim)
    # plt.hexbin(
    #     x=dfi.values[:, 0],
    #     y=dfi.values[:, 1],
    #     gridsize=(25, 25),
    #     bins=None,
    #     #mincnt=1,
    #     extent=(0, 1, 0, 1),
    #     #extent=(-jnp.pi/2, +jnp.pi/2, 0, 10),
    # )
    ax[0][0].set_title(r"$p(\alpha|x_0)$")
    ax[0][0].set_xlabel(r"$\alpha$")
    ax[0][0].axvline(x=theta_obs[1], linestyle='dotted', color="orange", lw=2)

    ax[0][1].axvline(x=theta_obs[1], ls='dotted', c='orange', lw=2.0)
    ax[0][1].axhline(y=theta_obs[0], ls='dotted', c='orange', lw=2.0)
    ax[0][1].set_xlabel(r"$\alpha$")
    ax[0][1].set_ylabel(r"$\beta$")
    ax[0][1].scatter(theta_obs[1],theta_obs[0], c='orange', s=100, zorder=2)
    ax[0][1].set_title(r"$p(\beta, \alpha_0|x_0)$")
    
    ax[1][1].set_title(r"$p(\beta|x_0)$")
    ax[1][1].set_xlabel(r"$\beta$")
    ax[1][1].axvline(x=theta_obs[0], linestyle='dotted', color="orange", lw=2)

    fig.savefig(fname=f'true_posterior_samples_alpha_{theta_obs[1]}_beta_{theta_obs[0]}_scale_{scale}.pdf', format='pdf')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Sample from the true posterior on the Toy Model'
    )
    parser.add_argument('--alpha', '-a', type=float, default=0.5,
                        help='Ground truth value for alpha.')
    parser.add_argument('--beta', '-b', type=float, default=0.5,
                        help='Ground truth value for beta.')
    parser.add_argument('--nsample', '-nsp', type=int, default=100000,
                        help='How many parameters to sample.')
    parser.add_argument('--viz', action='store_true',
                        help='Only show a pairplot of posterior samples from a csv file.')
    args = parser.parse_args()
    
    rng_key = random.PRNGKey(1)

    theta_obs=torch.tensor([args.beta,args.alpha])
    theta_obs_list=[theta_obs]
    num_samples=args.nsample
    scale=0.001
    if not args.viz :
        df_samples = sample_true_posterior(rng_key, theta_obs_list, num_samples)
        pairplot_samples(theta_obs=theta_obs, df_samples=df_samples, resample=True)

    else :
        pairplot_samples(theta_obs=theta_obs, df_samples=None)


    