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


rng_key = random.PRNGKey(1)

# b = 5.0
# m = 10.0
# g = 10.0
# h = 1.8


# def dz_dt(z, t):
#     """
#     compute the derivative of z wrt t
#     """

#     A = jnp.zeros((4, 4))
#     A = A.at[0, 2].set(1.0)
#     A = A.at[1, 3].set(1.0)
#     A = A.at[2, 2].set(-b/m)
#     A = A.at[3, 3].set(-b/m)

#     c = jnp.array([0.0, 0.0, 0.0, -g])

#     dz_dt = A @ z + c

#     return dz_dt


# def model(x_obs=None, solver='analytic'):
#     """

#     """

#     # measurement times
#     ts = jnp.linspace(0, 3, 100)

#     # parameters alpha, beta, gamma, delta of dz_dt
#     theta = numpyro.sample(
#         "theta",
#         dist.Uniform(low=jnp.array([-jnp.pi, 0.0]),
#                      high=jnp.array([jnp.pi, 10.0]))
#     )

#     angle = theta[0]
#     speed = theta[1]

#     if solver == 'numeric':

#         z_init = jnp.array([
#             0.0,  # x(0)
#             h,  # y(0)
#             speed * jnp.cos(angle),  # vx(0)
#             speed * jnp.sin(angle),  # vy(0)
#         ])

#         # integrate dz/dt, the result will have shape N x 2
#         z = odeint(
#             dz_dt,
#             z_init,
#             ts,
#             rtol=1e-8,
#             atol=1e-8,
#             mxstep=1000
#         )
#         # get the value of x in which the projectile touches the ground
#         t_hit = jnp.sum(z[:, 1] > 0)
#         x = z[t_hit, 0]

#     elif solver == 'analytic':

#         x_array = (m*speed/b)*jnp.cos(angle)*(1 - jnp.exp(-b/m * ts))
#         y_array = (m*speed/b*jnp.sin(angle) + (m/b)**2*g)*(1 - jnp.exp(-b/m * ts)) + h - (m/b)*g*ts  # noqa
#         t_hit = (y_array > 0).sum()
#         x = x_array[t_hit]

#     numpyro.sample(
#         "obs",
#         dist.Normal(
#             x,
#             scale=0.10
#         ),
#         obs=x_obs
#     )

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
  
    if solver == 'analytic':
        x = alpha*beta
    numpyro.sample("obs", dist.Normal(x,scale=scale),obs=x_obs)
    #print("obs",numpyro.sample("obs", dist.Normal(x,scale=0.10),obs=x_obs))


def generate_posterior_samples(
        rng_key, x_obs, theta_obs, num_samples=10_000):
    """
    generate samples from the posterior distribution p(theta|x)
    with MCMC procedure
    """

    samples_mcmc = {}

    #use no U-turn sampler
    kernel = NUTS(
        model_hnpe,
        init_strategy=init_to_value(None, values={'theta': theta_obs})
    )
    num_chains = 4
    mcmc = MCMC(
        kernel,
        num_warmup=2000,
        num_chains=num_chains,
        num_samples=int(num_samples/num_chains),
    )
    rng_key, subkey = random.split(rng_key)
    mcmc.run(
        rng_key=subkey,
        x_obs=x_obs)

    samples_mcmc = mcmc.get_samples()['theta']

    samples = jnp.asarray(samples_mcmc)
    return samples


# prior = dist.Uniform(
#     low=jnp.array([-jnp.pi/2, 0.0]),
#     high=jnp.array([jnp.pi/2, 10.0])
# )

# theta_obs_list = [
#     [0.0, 2.0]
# ]

theta_obs_list=[torch.tensor([0.5,0.5])]
scale=0.001

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

    #generates samples from the true posterior wrt x_0
    rng_key, subkey = random.split(rng_key)
    samples = generate_posterior_samples(
        rng_key=subkey,
        x_obs=x_obs,
        theta_obs=theta_obs,
        num_samples=10_000
    )
    df = pd.DataFrame(data=samples, columns=["alpha","beta"])
    # results = {}
    # results['samples'] = df.values
    # results['theta_obs'] = np.array(theta_obs_list[i])
    # results['x_obs'] = np.array(x_obs)
    # fname_save = f'true_posterior_example_{i:02}.pkl'
    # torch.save(results, fname_save)

    #record the samples for different theta_0
    df['example'] = [i] * len(samples)
    df_list.append(df)

df = pd.concat(df_list)
df.to_csv(f"true_posterior_samples_{scale}.csv",index=False)

fig, ax = plt.subplots(figsize=(5.4, 5.4))
dfi = df[df['example'] == i]
plt.hexbin(
    x=dfi.values[:, 0],
    y=dfi.values[:, 1],
    gridsize=(25, 25),
    bins=None,
    #mincnt=1,
    extent=(0, 1, 0, 1),
    #extent=(-jnp.pi/2, +jnp.pi/2, 0, 10),
)
ax.axvline(x=theta_obs_list[i][0], ls='--', c='k', lw=2.0)
ax.axhline(y=theta_obs_list[i][1], ls='--', c='k', lw=2.0)
ax.scatter(*theta_obs_list[i], c='r', s=100, zorder=2)
ax.set_xlabel('alpha', fontsize=14)
ax.set_ylabel('beta', fontsize=14)
# ax.set_xlim(-jnp.pi/2, +jnp.pi/2)
ax.set_xlim(0,1)
# ax.set_ylim(0, 10)
ax.set_ylim(0, 1)
plt.colorbar()
fig.savefig(fname=f'posterior_samples_mcmc_scale_{scale}.pdf', format='pdf')
plt.show()
