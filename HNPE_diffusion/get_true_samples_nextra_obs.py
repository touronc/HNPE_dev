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
    if 0:
        mcmc_alpha = mcmc.get_samples(group_by_chain=True)['alpha']#[:, 0]
        # print(mcmc_alpha.shape)
        mcmc_beta = mcmc.get_samples(group_by_chain=True)['beta']#[:, 0]
    else:
        mcmc_alpha = mcmc.get_samples()['alpha'][:, 0]
        # print(mcmc_alpha.shape)
        mcmc_beta = mcmc.get_samples()['beta'][:, 0]
    print(mcmc_beta.shape, mcmc_alpha.shape)
    #dict of samples for each value of n_extra
    samples_mcmc = np.array(jnp.stack([mcmc_alpha, mcmc_beta]).T)
    #print(samples_mcmc)
    df = pd.DataFrame(samples_mcmc,columns=["alpha","beta"])
    # df.to_csv(f"true_samples_alpha_{true_theta[0]}_beta_{true_theta[1]}_nextra_{n_extra}.csv",index=False)
    return df, samples_mcmc
    # return mcmc_alpha, mcmc_beta

def model_unif_gauss(x_obs=None, nextra=0, scale=0.2):
    """
    define the simulator for a linear gaussian example
    The parameters are theta in 2d
    nextra is the number of additional observations drawn from the same parameter theta
    """
    theta = numpyro.sample("theta", dist.Uniform(low=jnp.array([0.0, 0.0]),
                     high=jnp.array([1.0, 1.0])))
    # theta = numpyro.sample("theta", dist.MultivariateNormal(loc=jnp.ones(2),covariance_matrix = 3*jnp.eye(2)))

    x = numpyro.sample("obs", dist.MultivariateNormal(loc=theta,covariance_matrix = scale*jnp.eye(2)).expand([nextra+1]),obs=x_obs)

def get_posterior_unif_gauss(x_obs, theta_obs, nextra, num_samples=100_000):
    rng_key = random.PRNGKey(1)
    # fix the parameters of the ground truth
    x_obs = jnp.array(x_obs)

    kernel = NUTS(
        model_unif_gauss,
        init_strategy=init_to_value(
            None, values={"theta": jnp.array(theta_obs)}))
    num_chains = 4
    mcmc = MCMC(
        kernel,
        num_warmup=1_000,
        num_chains=num_chains,
        num_samples=int(num_samples/num_chains),
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(
        rng_key=subkey,
        x_obs=x_obs,
        nextra = nextra)
    post_samples = mcmc.get_samples()["theta"]
    post_samples = np.array(post_samples)
    df = pd.DataFrame(post_samples)
    return post_samples

def model_hnpe_gauss(x_obs=None, n_extra=0, eps=0.01):

    alpha = numpyro.sample(
        "alpha",
        dist.Normal(loc=jnp.zeros(n_extra+1), scale=jnp.ones(n_extra+1))
    )

    beta = numpyro.sample(
        "beta",
        dist.Normal(loc=jnp.zeros(1), scale=jnp.ones(1))
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

def get_posterior_samples_hnpe_gauss(n_extra,theta_o,x_o_long,nb_samples=100000):
    
    rng_key = random.PRNGKey(1)
    # fix the parameters of the ground truth
    alpha_star = jnp.concatenate(
        [jnp.array([theta_o[0]]),
        jnp.array(np.random.randn(n_extra))])
    beta_star = jnp.array([theta_o[1]])

    # x_obs = jnp.expand_dims(jnp.array(x_o_long),0)
    x_obs = jnp.array(x_o_long)
    kernel = NUTS(
        model_hnpe_gauss)
        # init_strategy=init_to_value(
        #     None, values={"alpha": alpha_star, "beta": beta_star}))
    num_chains = 4
    init_alpha = jnp.concatenate([jnp.tile(alpha_star,(num_chains//2,1)),jnp.tile(-alpha_star,(num_chains//2,1))],axis=0)
    init_beta = jnp.concatenate([jnp.tile(beta_star,(num_chains//2,1)),jnp.tile(-beta_star,(num_chains//2,1))],axis=0)
    init_params={"alpha": init_alpha, "beta": init_beta}
  
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
        n_extra=n_extra,
        init_params=init_params)

    mcmc_alpha = mcmc.get_samples()['alpha'][:, 0]
    print(mcmc_alpha.shape)
    mcmc_beta = mcmc.get_samples()['beta'][:, 0]
    print(mcmc_beta.shape)
    samples_mcmc = np.array(jnp.stack([mcmc_alpha, mcmc_beta]).T)
    df = pd.DataFrame(samples_mcmc,columns=["alpha","beta"])
    return df, samples_mcmc

def true_marginal_0_obs(x_0):
    param = torch.linspace(x_0.item(),1,500)
    return param, -1.0/(torch.log(x_0)*param)

def true_marginal_alpha_nextra_obs(nextra_obs):
    N = nextra_obs.size(1)-1 # only the nb of EXTRA obs
    mu = torch.max(nextra_obs)
    print("check",nextra_obs.size())
    nu = torch.max(nextra_obs[:,1:])
    x_0 = nextra_obs.squeeze()[0]
    alpha = torch.linspace(x_0.item(),torch.min(torch.tensor([1.0,x_0/nu])).item(),500)
    return alpha, N*alpha**(N-1)/((1/mu**N-1)*x_0**N)

def true_marginal_beta_nextra_obs(nextra_obs):
    N = nextra_obs.size(1)-1
    mu = torch.max(nextra_obs)
    beta = torch.linspace(mu.item(), 1, 500)
    return beta, N/(beta**(N+1)*(1/mu**N-1))

def plot_true_posterior(true_nextra, true_theta, samples_mcmc):
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
    n_extra = true_nextra.size(1)-1
    xlim = [[0.0,1.0],[0.0,1.0]]
    fig, ax = plot.pairplot(samples_mcmc, limits=xlim, diag="kde")
    condition_title = r"$x_0$"
    if n_extra>0:
        condition_title += rf"$, x_1, x_2,...,x_{n_extra}$"
    print(condition_title)
    #x_0 = true_theta[0]*true_theta[1]
    ax[0][0].set_title(r"$p(\alpha|$"+condition_title+")")
    ax[0][0].set_xlabel(r"$\alpha$")
    ax[0][0].axvline(x=true_theta[0], linestyle='dotted', color="orange", lw=2)
    if n_extra==0:
        param, densities = true_marginal_0_obs(true_nextra)
        ax[0][0].plot(param,densities.squeeze(), color="red")
        ax[1][1].plot(param, densities.squeeze(), color="red")
    else:
        param, density_alpha = true_marginal_alpha_nextra_obs(true_nextra)
        ax[0][0].plot(param, density_alpha, color="red")
        param, density_beta = true_marginal_beta_nextra_obs(true_nextra)
        ax[1][1].plot(param, density_beta, color="red")  
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
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Sample from the true posterior on the Toy Model with n_extra observations'
    )
    parser.add_argument('--alpha', '-a', type=float, default=0.5,
                        help='Ground truth value for alpha.')
    parser.add_argument('--beta', '-b', type=float, default=0.5,
                        help='Ground truth value for beta.')
    parser.add_argument('--n_extra', '-n',  nargs='+', type=int, default=[0, 5, 10],
                        help='How many extra observations to consider.')
    parser.add_argument('--nb_samples', '-nsp', type=int, default=10000,
                        help='How many parameters to sample.')
    parser.add_argument('--viz', action='store_true',
                        help='Only show a pairplot of posterior samples from a csv file.')
    args = parser.parse_args()
    #file = "ToyModel_naive_False_ntrials_01_nextra_10_alpha_0.50_beta_0.50_gamma_1.00_noise_0.01_agg_False.csv"
    rng_key = random.PRNGKey(1)

    #true_nextra = pd.read_csv(file)["xobs"]
    n_extra_list = args.n_extra[0]
    
    true_theta=[args.alpha,args.beta]
    # print(true_theta)
    # true_nextra = torch.tensor([0.2500]).unsqueeze(1)
    true_alpha = jnp.concatenate(
        [jnp.array([true_theta[0]]),
        jnp.array(np.random.rand(n_extra_list))])
    true_nextra = (true_alpha*true_theta[1])
    import seaborn as sns
    # plt.figure()
    # samples_alpha, samples_beta = get_posterior_samples(n_extra_list,true_theta,true_nextra,nb_samples=args.nb_samples)
    # samples_alpha=samples_alpha[:,:,0].reshape(-1)
    # print(samples_alpha.shape)
    # samples_beta = samples_beta.squeeze(2).reshape(-1)
    # param,density = true_marginal_alpha_nextra_obs(torch.tensor(np.array(true_nextra)).unsqueeze(0))
    # param_b, density_b = true_marginal_beta_nextra_obs(torch.tensor(np.array(true_nextra)).unsqueeze(0))
    # print(density_b)
    # sns.kdeplot(samples_beta)
    # plt.plot(param_b,density_b)
    # plt.show()


    nb_nextra = [0,2,5,10]
    fig, axes = plt.subplots(len(nb_nextra),2,figsize=(16,8.5))
    fig2, axes2 = plt.subplots(len(nb_nextra),2,figsize=(16,8.5))
    for j in range(len(nb_nextra)):
        stop = nb_nextra[j]
        true_nextra_j = true_nextra[:stop+1]
        print(true_nextra_j)

        samples_alpha, samples_beta = get_posterior_samples(stop,true_theta,true_nextra_j,nb_samples=args.nb_samples)
        print("size alpha",samples_beta.shape)
        true_nextra_j = torch.tensor(np.array(true_nextra_j))
        print(true_nextra_j.size())

        mu = torch.max(true_nextra_j)
        if true_nextra_j.shape[0]>1:
            nu = torch.max(true_nextra_j[1:])
            paramb, densityb = true_marginal_beta_nextra_obs(true_nextra_j.unsqueeze(0))
            parama, densitya = true_marginal_alpha_nextra_obs(true_nextra_j.unsqueeze(0))
        else:
            nu=mu
            paramb, densityb = true_marginal_0_obs(true_nextra_j[0])
            parama, densitya = true_marginal_0_obs(true_nextra_j[0])

        axes2[j,0].plot(parama,densitya, color="red", label="analytical")
        sns.kdeplot(samples_alpha[:,:,0].reshape(-1), ax=axes2[j,0], label="MCMC")
        axes2[j,0].axvline(x=true_theta[0],ls="dashed",color="orange", label=r"$\alpha_0$")
        axes2[j,0].legend()
        axes2[j,0].set_ylabel(fr"${stop}$ extra obs")
        axes2[j,1].set_ylabel("")
        axes2[j,1].plot(paramb,densityb, color="red", label="analytical")
        sns.kdeplot(samples_beta[:,:,0].reshape(-1), ax=axes2[j,1], label="MCMC")
        axes2[j,1].axvline(x=true_theta[1],ls="dashed",color="orange", label=r"$\beta_0$")
        axes2[j,1].legend()
        axes2[0,0].set_title(r"Evolution of $\alpha$")
        axes2[0,1].set_title(r"Evolution of $\beta$")

        ax = axes[j,0]
        color=["red","green","orange","blue"]
        for i in range(samples_alpha.shape[0]):
            ax.plot(samples_alpha[i,:,0], color=color[i], label=f"chain {i+1}")
        ax.axhline(y=true_theta[0], color="black", ls="dashed", label=r"$\alpha_0$")
        ax.axhline(y=true_nextra_j[0], color="black")
        # print("pblm",true_nextra_j[0]/nu)
        # print("mu", mu)
        # print("mu", mu)
        # print("x0", true_nextra_j[0])
        ax.axhline(y=torch.min(torch.tensor([1.0,true_nextra_j[0]/nu])).item(), color="black")
        if j==len(nb_nextra)-1:
            ax.set_xlabel("nb of MCMC iterations")
        ax.set_ylabel(fr"${stop}$ extra obs")
        ax.legend()
        ax = axes[j,1]
        for i in range(samples_beta.shape[0]):
            ax.plot(samples_beta[i,:,0], color=color[i], label=f"chain {i+1}")
        ax.axhline(y=true_theta[1], color="black", ls="dashed", label=r"$\beta_0$")
        ax.axhline(y=1, color="black")
        ax.axhline(y=mu.item(), color="black")
        if j==len(nb_nextra)-1:
            ax.set_xlabel("nb of MCMC iterations")
        ax.legend()
    axes[0,0].set_title(r"Evolution of $\alpha$")
    axes[0,1].set_title(r"Evolution of $\beta$")
    
    fig.suptitle(rf"$\alpha={true_theta[0]}$, $\beta={true_theta[1]}$")
    fig2.suptitle(rf"$\alpha={true_theta[0]}$, $\beta={true_theta[1]}$")
    # sns.kdeplot(x=samples[:,0], y=samples[:,1])
    # plt.axhline(y=true_theta[1])
    # plt.axvline(x=true_theta[0])
    # plot_true_posterior(true_nextra,true_theta,samples)
    plt.tight_layout()

    fig.savefig(f"trace_plot_hnpe_toymodel_alpha_{true_theta[0]}_beta_{true_theta[1]}.png")
    fig2.savefig(f"marginal_plot_hnpe_toymodel_alpha_{true_theta[0]}_beta_{true_theta[1]}.png")

    # plt.savefig(f"trace_plot_hnpe_toymodel_alpha_{true_theta[0]}_beta_{true_theta[1]}.png")
    plt.show()




    # df.to_csv(f"true_samples_toy_model_{n_extra_list+1}_obs_beta_{args.beta}.csv")
    # from sbi.utils.metrics import c2st
    # nextra = 10
    # cov_prior = 3*torch.eye(2)
    # cov_lik = 0.2*jnp.eye(2)
    # mu_prior = torch.ones(2)
    # theta_o = jnp.array([args.alpha,args.beta])
    # torch_cov_lik = torch.from_numpy(cov_lik.__array__())
    # x_o = dist.MultivariateNormal(loc=theta_o,covariance_matrix = cov_lik).sample(key=rng_key,sample_shape=(nextra+1,))
    # # print(jnp.squeeze(x_o,axis=0).shape)
    # true_samples = get_posterior_unif_gauss(x_o, theta_o, nextra = nextra, num_samples=10000)
    # print(true_samples.shape)
    # cov_tall_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+(nextra+1)*torch.linalg.inv(torch_cov_lik))
    # # cov_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+torch.linalg.inv(torch_cov_lik))
    # # mu_post = cov_post@(torch.linalg.inv(torch_cov_lik)@torch.from_numpy(x_o.__array__()).reshape(2,1) + torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1))
    # def mu_tall_post(x):
    #     "mean of the true tall posterior p(beta|x0,...xn)"
    #     tmp = torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1)
    #     for i in range(x.shape[0]):
    #         tmp += torch.linalg.inv(torch_cov_lik)@torch.from_numpy(x.__array__())[i,:].reshape(2,1)
    #     return cov_tall_post@tmp
    
    # anal_samples = dist.MultivariateNormal(loc=jnp.array(mu_tall_post(x_o).squeeze(1).numpy()),covariance_matrix = jnp.array(cov_tall_post.numpy())).sample(sample_shape=(10000,),key=rng_key)
    # print(anal_samples.shape)
    # acc = c2st(torch.from_numpy(anal_samples.__array__()),torch.from_numpy(true_samples))
    # print(acc)

