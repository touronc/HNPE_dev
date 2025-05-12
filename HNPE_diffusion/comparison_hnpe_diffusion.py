import sbi
from sbi.inference import NPSE, SNPE_C
import torch
import mlxp
from sbi.utils import BoxUniform
from simulator import simulator_ToyModel, prior_ToyModel
import seaborn as sns
import matplotlib.pyplot as plt
from posterior import ToyModelFlow_diffusion_nflows
from posterior import build_flow, IdentityToyModel
from functools import partial
from inference import run_inference
from sbi import inference as sbi_inference
from get_true_samples_nextra_obs import get_posterior_samples
from sbi.utils.metrics import c2st
from sbi.utils.metrics import unbiased_mmd_squared
import pandas as pd
import ot

def true_marginal_beta_nextra_obs(nextra_obs):
        N = nextra_obs.size(0)-1
        mu = torch.max(nextra_obs)
        beta = torch.linspace(mu.item(), 1, 500)
        if N>0:
            return beta, N/(beta**(N+1)*(1/mu**N-1))
        else:
            return beta, -1/torch.log(mu)*1/beta

def true_marginal_alpha_nextra_obs(nextra_obs):
    print(nextra_obs)
    N = nextra_obs.size(0)-1
    mu = torch.max(nextra_obs)
    nu = mu
    if N>0:
        nu = torch.max(nextra_obs[1:])
    x_0 = nextra_obs[0]
    if x_0/nu.item()<1:
        mini = x_0/nu.item()
    else:
        mini = torch.ones(1,)
    if N>0:
        alpha = torch.linspace(x_0.item(), mini.item(), 500)
        return alpha, N*alpha**(N-1)/(x_0**N*(1/mu**N-1))
    else:
        alpha = torch.linspace(mu.item(), 1, 500)
        return alpha, -1/torch.log(mu)*1/alpha

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    L, V = torch.linalg.eig(matrix)
    L = L.real
    V = V.real
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)

def wasserstein_dist(mu_1,mu_2,cov_1,cov_2):
    """Compute the Wasserstein distance between 2 Gaussians"""
    # G.Peyré & M. Cuturi (2020), Computational Optimal Transport, eq 2.41 
    sqrtcov1 = _matrix_pow(cov_1, 0.5)
    covterm = torch.trace(
        cov_1 + cov_2 - 2 * _matrix_pow(sqrtcov1 @ cov_2 @ sqrtcov1, 0.5)
    )
    return ((mu_1-mu_2)**2).sum() + covterm

@mlxp.launch(config_path='./configs/')
def main(ctx: mlxp.Context):

    cfg = ctx.config
    logger = ctx.logger

    torch.manual_seed(cfg.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_train = cfg.num_train
    prior_beta = BoxUniform(low=torch.zeros(1), high=torch.ones(1)) # prior on the parameters beta
    # prior = BoxUniform(low=torch.zeros(2),high=torch.ones(2))
    prior = prior_ToyModel(low=torch.tensor([0.0, 0.0]),
                           high=torch.tensor([1.0, 1.0]),
                          )
    summary_net = IdentityToyModel()

    theta_train = prior.sample((num_train,)) # dataset for the simulator
    beta_train = theta_train[:,-1:] # dataset for the diffusion part
    alpha_train = theta_train[:,:-1] # dataset for the normalizing flow
    x_train_hnpe = simulator_ToyModel(theta_train,p_alpha=prior, n_extra=cfg.nextra) # observations HNPE 
    x_train_diff = x_train_hnpe[:,0,0][:,None] # observations for diffusion
    
    print("Training dimensions : \n theta : ", theta_train.size(),
        "\n x : ",x_train_diff.size(),"\n beta : ",beta_train.size(),
        "\n alpha : ",alpha_train.size())

    print("############### TRAINING DIFFUSION ###############")
    # diffusion model for beta parameters
    inference_beta = NPSE(prior=prior_beta, sde_type="vp")
    inference_beta.append_simulations(beta_train, x_train_diff)
    score_estimator = inference_beta.train()
    posterior_beta = inference_beta.build_posterior(score_estimator, sample_with="sde")
    # logger.log_artifacts({"post":posterior_beta}, artifact_name=f"posterior",
                            # artifact_type='pickle')
    
    print("############### TRAINING NORMALIZING FLOW ###############")
    # normalizing flow for the alpha parameters
    build_nn_posterior = partial(ToyModelFlow_diffusion_nflows, embedding_net=torch.nn.Identity())
    inference_alpha = SNPE_C(
            prior=prior_beta,
            density_estimator=build_nn_posterior,
            show_progress_bars=True)

    condition = torch.cat((x_train_diff,beta_train),dim=1)

    nn_posterior = inference_alpha.append_simulations(alpha_train, condition, proposal=None).train(
                #num_atoms=10,
                #learning_rate=1e-4,
                training_batch_size=200,
                #use_combined_loss=True,
                discard_prior_samples=True,
                max_num_epochs=10000,
                stop_after_epochs=20,
                show_train_summary=True
            ) # TYPE : FACTORIZED FLOW
    nn_posterior.zero_grad()
    posterior = inference_alpha.build_posterior(nn_posterior)

    print("############### TRAINING HNPE FLOW ###############")
    # #HNPE training
    build_nn_posterior_hnpe = partial(build_flow,
                                 embedding_net=summary_net,
                                 naive=False,
                                 aggregate=True)
    inference_hnpe = SNPE_C(
        prior=prior,
        density_estimator=build_nn_posterior_hnpe,
        show_progress_bars=True)
    
    nn_posterior_hnpe = inference_hnpe.append_simulations(theta_train, x_train_hnpe, proposal=None).train(
            num_atoms=10,
            #learning_rate=1e-4,
            training_batch_size=200,
            #use_combined_loss=True,
            discard_prior_samples=True,
            max_num_epochs=10000,
            stop_after_epochs=20,
            show_train_summary=True
        ) # TYPE : FACTORIZED FLOW
    nn_posterior_hnpe.zero_grad()
        #nn_posterior=ConditionalDensityEstimator(net=nn_posterior, input_shape=nn_posterior.input_shape, condition_shape=nn_posterior.condition_shape)
    posterior_hnpe = inference_hnpe.build_posterior(nn_posterior_hnpe)
    logger.log_artifacts({"posterior_beta":posterior_beta,
                          "posterior_alpha":nn_posterior,
                          "posterior_hnpe":posterior_hnpe}, artifact_name="posteriors.pkl",
                            artifact_type='pickle')

    print("############### INFERENCE PART ###############")
    # inference
    ground_truth = cfg.beta
    num_samples = cfg.num_samples
    theta_o = torch.tensor([cfg.alpha,cfg.beta],dtype=torch.float64)
    x_o = (theta_o[0]*theta_o[1]).reshape(1,1)

    if cfg.nextra==0:
        # sampling with 0 extra observation
        samples_beta = posterior_beta.sample((num_samples,), x=x_o, predictor="euler_maruyama", corrector="langevin",show_progress_bars=False)
        cond_flow = torch.cat((x_o.repeat(num_samples,1),samples_beta),dim=1)
        samples_alpha = nn_posterior.sample((num_samples,),condition=cond_flow)
        # samples_alpha = posterior.sample((num_samples,),x=cond_flow)
        samples_hnpe = posterior_hnpe.sample((num_samples,),x=x_o)
        df_true, true_samples_0_obs = get_posterior_samples(cfg.nextra,theta_o,x_o,num_samples)

        # df_est = pd.DataFrame(torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1),columns=["alpha","beta"])
        # df_true.to_csv(f"true_samples_{cfg.nextra+1}_obs.csv")
        # df_est.to_csv(f"est_samples_{cfg.nextra+1}_obs.csv")
        acc_alpha_diff = c2st(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_alpha.detach())
        acc_beta_diff = c2st(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_beta.detach())
        acc_diff  = c2st(torch.from_numpy(true_samples_0_obs),torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1),classifier="mlp")
        acc_alpha_hnpe = c2st(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_hnpe[:,0].detach())
        acc_beta_hnpe = c2st(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_hnpe[:,1].detach())
        acc_hnpe  = c2st(torch.from_numpy(true_samples_0_obs),samples_hnpe.detach(),classifier="mlp")
        logger.log_metrics({"c2st_alpha_diff":acc_alpha_diff.item(),"c2st_beta_diff":acc_beta_diff.item(),"c2st_pair_diff":acc_diff.item(),
                            "c2st_alpha_hnpe":acc_alpha_hnpe.item(),"c2st_beta_hnpe":acc_beta_hnpe.item(),"c2st_pair_hnpe":acc_hnpe.item()},
                             log_name="0_obs")
        
        mmd_alpha_diff = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_alpha.detach())
        mmd_beta_diff = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_beta.detach())
        mmd_diff = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs),torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1))
        mmd_alpha_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_hnpe[:,0].detach())
        mmd_beta_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_hnpe[:,1].detach())
        mmd_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs),samples_hnpe.detach())
        logger.log_metrics({"mmd_alpha_diff":mmd_alpha_diff.item(),"mmd_beta_diff":mmd_beta_diff.item(),"mmd_pair_diff":mmd_diff.item(),
                            "mmd_alpha_hnpe":mmd_alpha_hnpe.item(),"mmd_beta_hnpe":mmd_beta_hnpe.item(),"mmd_pair_hnpe":mmd_hnpe.item()},
                             log_name="0_obs")
        
        cov_true = torch.cov(torch.from_numpy(true_samples_0_obs).T)
        cov_est_diff = torch.cov(torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1).T)
        cov_est_hnpe = torch.cov(samples_hnpe.T)
        diff_cov = torch.mean((cov_true-cov_est_diff)**2)
        diff_cov_hnpe = torch.mean((cov_true-cov_est_hnpe)**2)
        logger.log_metrics({"diff_cov":diff_cov.item(),"diff_cov_hnpe": diff_cov_hnpe.item()}, log_name="0_obs")

        a = torch.ones((samples_hnpe.shape[0],)) / samples_hnpe.shape[0]
        b = torch.ones((true_samples_0_obs.shape[0],)) / true_samples_0_obs.shape[0]
        M = ot.dist(samples_hnpe.detach(),torch.from_numpy(true_samples_0_obs))
        wass_dist_hnpe = ot.emd2(a, b, M)
        M = ot.dist(torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1),torch.from_numpy(true_samples_0_obs))
        wass_dist_diff = ot.emd2(a, b, M)
        logger.log_metrics({"wass_diff":wass_dist_diff.item(),"wass_hnpe": wass_dist_hnpe.item()}, log_name="0_obs")

        param, density_beta = true_marginal_beta_nextra_obs(x_o)

        fig = plt.figure(figsize=(11,8))
        plt.subplot(121)
        sns.kdeplot(samples_alpha.detach().squeeze(1), color="blue", label="Diffusion")
        sns.kdeplot(samples_hnpe[:,0].detach(), color="green", label="HNPE")
        plt.plot(param,density_beta, color="red", label="analytical")
        plt.axvline(x=theta_o[0], ls="dashed", color="orange", label=r"$\alpha_0$")
        plt.xlabel(r"$\alpha$")
        plt.legend()
        plt.title(r"$p(\alpha|x_0)$")
        plt.subplot(122)
        sns.kdeplot(samples_beta.detach().squeeze(1), color="blue", label="Diffusion")
        sns.kdeplot(samples_hnpe[:,1].detach(), color="green", label="HNPE")
        plt.axvline(x=theta_o[1], ls="dashed", color="orange", label=r"$\beta_0$")
        plt.plot(param,density_beta, color="red", label="analytical")
        plt.xlabel(r"$\beta$")
        plt.title(r"$p(\beta|x_0)$")
        plt.legend()
        logger.log_artifacts(fig, artifact_name=f"marginals_1_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                            artifact_type='image')

        fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_alpha.detach().squeeze(1), y=samples_beta.detach().squeeze(1), cmap="Blues", ax=ax[0,0], fill=True, alpha=0.5,label="estimated")
        ax[0,0].axvline(x=theta_o[0], ls="dashed", color="orange")
        ax[0,0].axhline(y=theta_o[1], ls="dashed", color="orange")
        ax[0,0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        ax[0,0].set_xlabel(r"$\alpha$")
        ax[0,0].set_ylabel(r"$\beta$")
        ax[0,0].legend()
        ax[0,0].set_title(r"$p_{\text{Diff}}(\alpha,\beta|x_0)$")
        
        sns.kdeplot(x=samples_hnpe[:,0].detach(), y=samples_hnpe[:,1].detach(), cmap="Greens", ax=ax[0,1], fill=True, alpha=0.5,label="estimated")
        ax[0,1].axvline(x=theta_o[0], ls="dashed", color="orange")
        ax[0,1].axhline(y=theta_o[1], ls="dashed", color="orange")
        ax[0,1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        ax[0,1].set_xlabel(r"$\alpha$")
        ax[0,1].set_ylabel(r"$\beta$")
        ax[0,1].legend()
        ax[0,1].set_title(r"$p_{\text{HNPE}}(\alpha,\beta|x_0)$")

        sns.kdeplot(x=true_samples_0_obs[:,0], y=true_samples_0_obs[:,1], cmap="Reds", ax=ax[1,0],fill=True, label="true", alpha=0.5)
        ax[1,0].axvline(x=theta_o[0], ls="dashed", color="orange")
        ax[1,0].axhline(y=theta_o[1], ls="dashed", color="orange")
        ax[1,0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        ax[1,0].set_xlabel(r"$\alpha$")
        ax[1,0].set_ylabel(r"$\beta$")
        ax[1,0].legend()
        ax[1,0].set_title(r"$p_{\text{true}}(\alpha,\beta|x_0)$")

        ax[0,0].set_aspect('equal')
        ax[1,0].set_aspect('equal')
        ax[0,1].set_aspect('equal')
        ax[1,1].set_aspect('equal')
        plt.tight_layout()
        plt.show()
        logger.log_artifacts(fig, artifact_name=f"2D_plot_1_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                            artifact_type='image')
    else:
        # inference with several additional observations
        n_obs = cfg.nextra
        extra_alpha = torch.rand(n_obs).unsqueeze(1)
        extra_obs = torch.cat((extra_alpha,torch.tensor([ground_truth]).repeat(n_obs,1)),dim=1)
        theta_o_long = torch.cat((theta_o.unsqueeze(0),extra_obs),dim=0)
        x_o_long = simulator_ToyModel(theta_o_long,sigma=0.1).squeeze(1)
        print(theta_o.size(), x_o_long.size())
        df_true, true_samples = get_posterior_samples(n_obs,theta_o,x_o_long.squeeze(1), num_samples)
        
        samples_long_gauss = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="gauss", predictor="euler_maruyama", corrector="langevin")
        samples_long_auto = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="auto_gauss", predictor="euler_maruyama", corrector="langevin")
        samples_long_fnpe = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="fnpe", predictor="euler_maruyama", corrector="langevin")
        samples_long_jac = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="jac_gauss", predictor="euler_maruyama", corrector="langevin")

        cond_flow_gauss = torch.cat((x_o.repeat(num_samples,1),samples_long_gauss),dim=1)
        cond_flow_auto = torch.cat((x_o.repeat(num_samples,1),samples_long_auto),dim=1)
        cond_flow_fnpe = torch.cat((x_o.repeat(num_samples,1),samples_long_fnpe),dim=1)
        cond_flow_jac = torch.cat((x_o.repeat(num_samples,1),samples_long_jac),dim=1)
        
        samples_alpha_gauss = nn_posterior.sample((num_samples,),condition=cond_flow_gauss)
        samples_alpha_auto = nn_posterior.sample((num_samples,),condition=cond_flow_auto)
        samples_alpha_fnpe = nn_posterior.sample((num_samples,),condition=cond_flow_fnpe)
        samples_alpha_jac = nn_posterior.sample((num_samples,),condition=cond_flow_jac)
    
        # df_est = pd.DataFrame(torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1),columns=["alpha","beta"])
        # df_true.to_csv(f"true_samples_{cfg.nextra+1}_obs.csv")
        # df_est.to_csv(f"est_samples_{cfg.nextra+1}_obs.csv")
        samples_hnpe = posterior_hnpe.sample((num_samples,),x=x_o_long.reshape(1,n_obs+1))

        acc_alpha_gauss = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_gauss.detach())
        acc_alpha_auto = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_auto.detach())
        acc_alpha_jac = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_jac.detach())
        acc_alpha_fnpe = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_fnpe.detach())
        acc_alpha_hnpe = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_hnpe[:,0].detach())
        acc_beta_gauss = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_gauss.detach())
        acc_beta_auto = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_auto.detach())
        acc_beta_jac = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_jac.detach())
        acc_beta_fnpe = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_fnpe.detach())
        acc_beta_hnpe = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_hnpe[:,1].detach())
        acc_gauss  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1),classifier="mlp")
        acc_auto  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1),classifier="mlp")
        acc_jac  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1),classifier="mlp")
        acc_fnpe  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1),classifier="mlp")
        acc_hnpe  = c2st(torch.from_numpy(true_samples),samples_hnpe.detach(),classifier="mlp")
        logger.log_metrics({"c2st_alpha_gauss":acc_alpha_gauss.item(),"c2st_beta_gauss":acc_beta_gauss.item(),"c2st_pair_gauss":acc_gauss.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_auto":acc_alpha_auto.item(),"c2st_beta_auto":acc_beta_auto.item(),"c2st_pair_auto":acc_auto.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_jac":acc_alpha_jac.item(),"c2st_beta_jac":acc_beta_jac.item(),"c2st_pair_jac":acc_jac.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_fnpe":acc_alpha_fnpe.item(),"c2st_beta_fnpe":acc_beta_fnpe.item(),"c2st_pair_fnpe":acc_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_hnpe":acc_alpha_hnpe.item(),"c2st_beta_hnpe":acc_beta_hnpe.item(),"c2st_pair_hnpe":acc_hnpe.item()}, log_name=f"{n_obs+1}_obs")

        mmd_alpha_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_gauss.detach())
        mmd_alpha_auto = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_auto.detach())
        mmd_alpha_jac = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_jac.detach())
        mmd_alpha_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_fnpe.detach())
        mmd_alpha_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_hnpe[:,0].detach())
        mmd_beta_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_gauss.detach())
        mmd_beta_auto = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_auto.detach())
        mmd_beta_jac = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_jac.detach())
        mmd_beta_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_fnpe.detach())
        mmd_beta_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_hnpe[:,1].detach())
        mmd_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1))
        mmd_auto = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1))
        mmd_jac = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1))
        mmd_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1))
        mmd_hnpe = unbiased_mmd_squared(torch.from_numpy(true_samples),samples_hnpe.detach())
        logger.log_metrics({"mmd_alpha_gauss":mmd_alpha_gauss.item(),"mmd_beta_gauss":mmd_beta_gauss.item(),"mmd_pair_gauss":mmd_gauss.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_auto":mmd_alpha_auto.item(),"mmd_beta_auto":mmd_beta_auto.item(),"mmd_pair_auto":mmd_auto.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_jac":mmd_alpha_jac.item(),"mmd_beta_jac":mmd_beta_jac.item(),"mmd_pair_jac":mmd_jac.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_fnpe":mmd_alpha_fnpe.item(),"mmd_beta_fnpe":mmd_beta_fnpe.item(),"mmd_pair_fnpe":mmd_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_hnpe":mmd_alpha_hnpe.item(),"mmd_beta_hnpe":mmd_beta_hnpe.item(),"mmd_pair_hnpe":mmd_hnpe.item()}, log_name=f"{n_obs+1}_obs")
        
        cov_true = torch.cov(torch.from_numpy(true_samples).T)
        cov_est_gauss = torch.cov(torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1).T)
        cov_est_auto = torch.cov(torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1).T)
        cov_est_jac = torch.cov(torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1).T)
        cov_est_fnpe = torch.cov(torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1).T)
        cov_est_hnpe = torch.cov(samples_hnpe.detach().T)
        diff_gauss = torch.mean((cov_true-cov_est_gauss)**2)
        diff_auto = torch.mean((cov_true-cov_est_auto)**2)
        diff_jac = torch.mean((cov_true-cov_est_jac)**2)
        diff_fnpe = torch.mean((cov_true-cov_est_fnpe)**2)
        diff_hnpe = torch.mean((cov_true-cov_est_hnpe)**2)
        logger.log_metrics({"diff_gauss":diff_gauss.item(),"diff_auto": diff_auto.item(),
                            "diff_jac":diff_jac.item(),
                            "diff_fnpe":diff_fnpe.item(),
                            "diff_hnpe":diff_hnpe.item()}, log_name=f"{n_obs+1}_obs")

        a = torch.ones((samples_hnpe.shape[0],)) / samples_hnpe.shape[0]
        b = torch.ones((true_samples.shape[0],)) / true_samples.shape[0]
        M = ot.dist(samples_hnpe.detach(),torch.from_numpy(true_samples))
        wass_dist_hnpe = ot.emd2(a, b, M)
        M = ot.dist(torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1),torch.from_numpy(true_samples))
        wass_dist_gauss = ot.emd2(a, b, M)
        M = ot.dist(torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1),torch.from_numpy(true_samples))
        wass_dist_auto = ot.emd2(a, b, M)
        M = ot.dist(torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1),torch.from_numpy(true_samples))
        wass_dist_jac = ot.emd2(a, b, M)
        M = ot.dist(torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1),torch.from_numpy(true_samples))
        wass_dist_fnpe = ot.emd2(a, b, M)
        logger.log_metrics({"wass_gauss":wass_dist_gauss.item(),"wass_hnpe": wass_dist_hnpe.item(),
                            "wass_auto":wass_dist_auto.item(),"wass_jac":wass_dist_jac.item(),
                            "wass_fnpe":wass_dist_fnpe.item()}, log_name=f"{n_obs+1}_obs")

        param_beta,density_beta = true_marginal_beta_nextra_obs(x_o_long)
        param_alpha,density_alpha = true_marginal_alpha_nextra_obs(x_o_long)
    
        fig = plt.figure(figsize=(12,8))
        plt.subplot(121)
        sns.kdeplot(true_samples[:,0], color="black", label="MCMC")
        sns.kdeplot(samples_alpha_auto.detach().squeeze(1), color="green", label="auto")
        sns.kdeplot(samples_alpha_fnpe.detach().squeeze(1), color="gray", label="Geffner")
        sns.kdeplot(samples_alpha_gauss.detach().squeeze(1), color="orange", label="GAUSS")
        sns.kdeplot(samples_alpha_jac.detach().squeeze(1), color="blue", label="JAC")
        sns.kdeplot(samples_hnpe[:,0].detach(), color="purple", label="HNPE")
        plt.plot(param_alpha,density_alpha, color="red", label="analytical")
        plt.axvline(x=theta_o[0], ls="dashed", color="brown", label=r"$\alpha_0$")
        plt.legend()
        plt.title(fr"$p(\alpha|x_0,...,x_{{{n_obs}}})$")

        plt.subplot(122)
        sns.kdeplot(true_samples[:,1], color="black", label="MCMC")
        sns.kdeplot(samples_long_auto.detach().squeeze(1), color="green", label="auto")
        sns.kdeplot(samples_long_fnpe.detach().squeeze(1), color="grey", label="Geffner")
        sns.kdeplot(samples_long_gauss.detach().squeeze(1), color="orange", label="GAUSS")
        sns.kdeplot(samples_long_jac.detach().squeeze(1), color="blue", label="JAC")
        sns.kdeplot(samples_hnpe[:,1].detach(), color="purple", label="HNPE")
        plt.plot(param_beta,density_beta, color="red", label="analytical")
        plt.axvline(x=theta_o[1], ls="dashed", color="brown", label=r"$\beta_0$")
        plt.title(fr"$p(\beta|x_0,...,x_{{{n_obs}}})$")
        plt.legend()
        plt.show()
        logger.log_artifacts(fig, artifact_name=f"marginals_{n_obs+1}_obs_{num_train}_train_{ground_truth}_beta.png",
                            artifact_type='image')
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_alpha_gauss.detach().squeeze(1), y=samples_long_gauss.detach().squeeze(1), cmap="Oranges", ax=axes[1][0], fill=True, alpha=0.5)
        axes[1][0].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[1][0].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[1][0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[1][0].set_xlabel(r"$\alpha$")
        axes[1][0].set_ylabel(r"$\beta$")
        axes[1][0].legend()
        axes[1][0].set_title(fr"$p_{{\text{{GAUSS}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_alpha_auto.detach().squeeze(1), y=samples_long_auto.detach().squeeze(1), cmap="Greens", ax=axes[0][1], fill=True, alpha=0.5)
        axes[0][1].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[0][1].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[0][1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[0][1].set_xlabel(r"$\alpha$")
        axes[0][1].set_ylabel(r"$\beta$")
        axes[0][1].legend()
        axes[0][1].set_title(fr"$p_{{\text{{AUTO}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")
                            
        sns.kdeplot(x=true_samples[:,0], y=true_samples[:,1], cmap="Reds", ax=axes[0][0],fill=True, alpha=0.5)
        axes[0][0].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[0][0].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[0][0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[0][0].set_xlabel(r"$\alpha$")
        axes[0][0].set_ylabel(r"$\beta$")
        axes[0][0].legend()
        axes[0][0].set_title(fr"$p_{{\text{{MCMC}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_alpha_jac.detach().squeeze(1), y=samples_long_jac.detach().squeeze(1), cmap="Blues", ax=axes[2][0], fill=True, alpha=0.5)
        axes[2][0].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[2][0].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[2][0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[2][0].set_xlabel(r"$\alpha$")
        axes[2][0].set_ylabel(r"$\beta$")
        axes[2][0].legend()
        axes[2][0].set_title(fr"$p_{{\text{{JAC}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_alpha_fnpe.detach().squeeze(1), y=samples_long_fnpe.detach().squeeze(1), cmap="Greys", ax=axes[1][1], fill=True, alpha=0.5)
        axes[1][1].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[1][1].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[1][1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[1][1].set_xlabel(r"$\alpha$")
        axes[1][1].set_ylabel(r"$\beta$")
        axes[1][1].legend()
        axes[1][1].set_title(fr"$p_{{\text{{FNPE}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_hnpe[:,0].detach(), y=samples_hnpe[:,1].detach(), cmap="Purples", ax=axes[2][1], fill=True, alpha=0.5)
        axes[2][1].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[2][1].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[2][1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label=r"$(\alpha_0,\beta_0)$")
        axes[2][1].set_xlabel(r"$\alpha$")
        axes[2][1].set_ylabel(r"$\beta$")
        axes[2][1].legend()
        axes[2][1].set_title(fr"$p_{{\text{{HNPE}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        axes[0][0].set_aspect('equal')
        axes[0][1].set_aspect('equal')
        axes[1][0].set_aspect('equal')
        axes[1][1].set_aspect('equal')
        axes[2][0].set_aspect('equal')
        axes[2][1].set_aspect('equal')
        plt.tight_layout()
        plt.show()
        logger.log_artifacts(fig, artifact_name=f"2D_plot_{n_obs+1}_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                        artifact_type='image')

if __name__ == "__main__":
    main()
    # import ot
    # x = torch.randn(1000,2)
    # y = torch.randn(1000,2)
    # mu_x = torch.mean(x,dim=0)
    # mu_y = torch.mean(x,dim=0)
    # cov_x = torch.cov(x.T)
    # cov_y = torch.cov(y.T)
    # print(mu_x.size())
    # print(cov_x.size())
    # M = ot.dist(x,y, metric='euclidean')**2
    # M2 = ot.dist(x,y)
    # a = torch.ones((x.shape[0],)) / x.shape[0]
    # b = torch.ones((y.shape[0],)) / y.shape[0]
    # wass_dist = ot.emd2(a, b, M)
    # wass_dist2 = ot.emd2(a, b, M2)
    # manu = wasserstein_dist(mu_x,mu_y,cov_x,cov_y)
    # print(wass_dist)
    # print(wass_dist2**0.5)
    # print(manu)
    # print(manu)

    # ground_truth = torch.tensor([0.5])
    # num_samples = 100
    # theta_o = torch.tensor([0.5,0.5],dtype=torch.float64)
    # x_o = (theta_o[0]*theta_o[1]).reshape(1,1)
    # extra_alpha = torch.rand(2).unsqueeze(1)
    # extra_obs = torch.cat((extra_alpha,torch.tensor([ground_truth]).repeat(2,1)),dim=1)
    # theta_o_long = torch.cat((theta_o.unsqueeze(0),extra_obs),dim=0)
    # x_o_long = simulator_ToyModel(theta_o_long,sigma=0.1).squeeze(1)
    # import pickle
    # post = pickle.load(open("/home/ctouron/codedev/sbi_hackathon/HNPE_diff/logs/325/artifacts/pickle/posteriors.pkl","rb"))
    # samples=post["posterior_hnpe"].sample((num_samples,),x=x_o_long.reshape(1,3))
    # print(samples)