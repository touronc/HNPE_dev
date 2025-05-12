import sbi
from sbi.inference import NPSE, SNPE_C
import torch
import mlxp
from sbi.utils import BoxUniform

from simulator import simulator_ToyModel
import seaborn as sns
import matplotlib.pyplot as plt
from posterior import ToyModelFlow_diffusion_nflows
from posterior import build_flow, IdentityToyModel
from functools import partial
from inference import run_inference
from sbi import inference as sbi_inference
from get_true_samples_nextra_obs import get_posterior_samples_hnpe_gauss
from sbi.utils.metrics import c2st
from sbi.utils.metrics import unbiased_mmd_squared
import pandas as pd

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

@mlxp.launch(config_path='./configs/')
def main(ctx: mlxp.Context):

    cfg = ctx.config
    logger = ctx.logger

    torch.manual_seed(cfg.seed)
    num_train = cfg.num_train
    prior_beta = torch.distributions.MultivariateNormal(loc=torch.zeros(1), covariance_matrix=torch.eye(1))

    beta_train = prior_beta.sample((num_train,))
    alpha_train = prior_beta.sample((num_train,))
    theta_train = torch.cat((alpha_train,beta_train),dim=1)
    x_train = simulator_ToyModel(theta_train,p_alpha=prior_beta).mean(dim=1)

    print("Training dimensions : \n theta : ", theta_train.size(),
        "\n x : ",x_train.size(),"\n beta : ",beta_train.size(),
        "\n alpha : ",alpha_train.size())

    # diffusion model for beta parameters
    inference_beta = NPSE(prior=prior_beta, sde_type="vp")
    inference_beta.append_simulations(beta_train, x_train)
    score_estimator = inference_beta.train()
    posterior_beta = inference_beta.build_posterior(score_estimator, sample_with="sde")

    # normalizing flow for the alpha parameters
    build_nn_posterior = partial(ToyModelFlow_diffusion_nflows, embedding_net=torch.nn.Identity())
    inference_alpha = SNPE_C(
            prior=prior_beta,
            density_estimator=build_nn_posterior,
            show_progress_bars=True)

    condition = torch.cat((x_train[:,:1],beta_train),dim=1)
    nn_posterior = inference_alpha.append_simulations(alpha_train, condition, proposal=None).train(
                #num_atoms=10,
                #learning_rate=1e-4,
                training_batch_size=200,
                #use_combined_loss=True,
                discard_prior_samples=True,
                max_num_epochs=5000,
                stop_after_epochs=20,
                show_train_summary=True
            ) # TYPE : FACTORIZED FLOW
    nn_posterior.zero_grad()
    # # posterior = inference_alpha.build_posterior(nn_posterior)

    # # inference
    ground_truth = cfg.beta
    num_samples = cfg.num_samples
    theta_o = torch.tensor([cfg.alpha,cfg.beta],dtype=torch.float64)
    theta_o = prior_beta.sample((2,)).squeeze(1)
    print(theta_o.size())
    x_o = simulator_ToyModel(theta_o).mean(dim=1)
    print(x_o.size())
    if cfg.nextra==0:
        # sampling with 0 extra observation
        samples_beta = posterior_beta.sample((num_samples,), x=x_o, predictor="euler_maruyama", corrector="langevin")
        cond_flow = torch.cat((x_o.repeat(num_samples,1),samples_beta),dim=1)
        samples_alpha = nn_posterior.sample((num_samples,),condition=cond_flow)

        df_true, true_samples_0_obs = get_posterior_samples_hnpe_gauss(cfg.nextra,theta_o,x_o,num_samples)
        # # df_est = pd.DataFrame(torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1),columns=["alpha","beta"])
        # # df_true.to_csv(f"true_samples_{cfg.nextra+1}_obs.csv")
        # # df_est.to_csv(f"est_samples_{cfg.nextra+1}_obs.csv")
        acc_alpha = c2st(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_alpha.detach())
        acc_beta = c2st(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_beta.detach())
        acc  = c2st(torch.from_numpy(true_samples_0_obs),torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1),classifier="mlp")
        logger.log_metrics({"c2st_alpha":acc_alpha.item(),"c2st_beta":acc_beta.item(),"c2st_pair":acc.item()}, log_name="0_obs")
        
        mmd_alpha = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,0]).unsqueeze(1),samples_alpha.detach())
        mmd_beta = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs[:,1]).unsqueeze(1),samples_beta.detach())
        mmd = unbiased_mmd_squared(torch.from_numpy(true_samples_0_obs),torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1))
        logger.log_metrics({"mmd_alpha":mmd_alpha.item(),"mmd_beta":mmd_beta.item(),"mmd_pair":mmd.item()}, log_name="0_obs")
        
        cov_true = torch.cov(torch.from_numpy(true_samples_0_obs).T)
        cov_est = torch.cov(torch.cat((samples_alpha.detach(),samples_beta.detach()),dim=1).T)
        diff_cov = torch.mean((cov_true-cov_est)**2)
        logger.log_metrics({"diff_cov":diff_cov.item()}, log_name="0_obs")

        fig = plt.figure(figsize=(11,8))
        plt.subplot(121)
        sns.kdeplot(samples_alpha.detach().squeeze(1), color="blue", label="estimated")
        sns.kdeplot(true_samples_0_obs[:,0], color="red", label="true")
        plt.axvline(x=theta_o[0], ls="dashed", color="orange", label="ground truth")
        plt.xlabel(r"$\alpha$")
        plt.legend()
        plt.title(r"$p(\alpha|x_0)$")
        plt.subplot(122)
        sns.kdeplot(samples_beta.detach().squeeze(1), color="blue", label="estimated")
        plt.axvline(x=theta_o[1], ls="dashed", color="orange", label="ground truth")
        sns.kdeplot(true_samples_0_obs[:,1], color="red", label="true")
        plt.xlabel(r"$\beta$")
        plt.title(r"$p(\beta|x_0)$")
        plt.legend()
        logger.log_artifacts(fig, artifact_name=f"marginals_1_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                            artifact_type='image')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_alpha.detach().squeeze(1), y=samples_beta.detach().squeeze(1), cmap="Blues", ax=ax1, fill=True, alpha=0.5,label="estimated")
        ax1.axvline(x=theta_o[0], ls="dashed", color="orange")
        ax1.axhline(y=theta_o[1], ls="dashed", color="orange")
        ax1.scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel(r"$\beta$")
        ax1.legend()
        ax1.set_title(r"$p_{\text{estim}}(\alpha,\beta|x_0)$")

        sns.kdeplot(x=true_samples_0_obs[:,0], y=true_samples_0_obs[:,1], cmap="Reds", ax=ax2,fill=True, label="true", alpha=0.5)
        ax2.axvline(x=theta_o[0], ls="dashed", color="orange")
        ax2.axhline(y=theta_o[1], ls="dashed", color="orange")
        ax2.scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"$\beta$")
        ax2.legend()
        ax2.set_title(r"$p_{\text{true}}(\alpha,\beta|x_0)$")

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
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
        x_o_long = simulator_ToyModel(theta_o_long).squeeze(1)
        
        df_true, true_samples = get_posterior_samples_hnpe_gauss(n_obs,theta_o,x_o_long.squeeze(1), num_samples)
        
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

        acc_alpha_gauss = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_gauss.detach())
        acc_alpha_auto = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_auto.detach())
        acc_alpha_jac = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_jac.detach())
        acc_alpha_fnpe = c2st(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_fnpe.detach())
        acc_beta_gauss = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_gauss.detach())
        acc_beta_auto = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_auto.detach())
        acc_beta_jac = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_jac.detach())
        acc_beta_fnpe = c2st(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_fnpe.detach())
        acc_gauss  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1),classifier="mlp")
        acc_auto  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1),classifier="mlp")
        acc_jac  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1),classifier="mlp")
        acc_fnpe  = c2st(torch.from_numpy(true_samples),torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1),classifier="mlp")
        logger.log_metrics({"c2st_alpha_gauss":acc_alpha_gauss.item(),"c2st_beta_gauss":acc_beta_gauss.item(),"c2st_pair_gauss":acc_gauss.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_auto":acc_alpha_auto.item(),"c2st_beta_auto":acc_beta_auto.item(),"c2st_pair_auto":acc_auto.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_jac":acc_alpha_jac.item(),"c2st_beta_jac":acc_beta_jac.item(),"c2st_pair_jac":acc_jac.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"c2st_alpha_fnpe":acc_alpha_fnpe.item(),"c2st_beta_fnpe":acc_beta_fnpe.item(),"c2st_pair_fnpe":acc_fnpe.item()}, log_name=f"{n_obs+1}_obs")

        mmd_alpha_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_gauss.detach())
        mmd_alpha_auto = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_auto.detach())
        mmd_alpha_jac = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_jac.detach())
        mmd_alpha_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,0]).unsqueeze(1),samples_alpha_fnpe.detach())
        mmd_beta_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_gauss.detach())
        mmd_beta_auto = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_auto.detach())
        mmd_beta_jac = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_jac.detach())
        mmd_beta_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples[:,1]).unsqueeze(1),samples_long_fnpe.detach())
        mmd_gauss = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1))
        mmd_auto = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1))
        mmd_jac = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1))
        mmd_fnpe = unbiased_mmd_squared(torch.from_numpy(true_samples),torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1))
        logger.log_metrics({"mmd_alpha_gauss":mmd_alpha_gauss.item(),"mmd_beta_gauss":mmd_beta_gauss.item(),"mmd_pair_gauss":mmd_gauss.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_auto":mmd_alpha_auto.item(),"mmd_beta_auto":mmd_beta_auto.item(),"mmd_pair_auto":mmd_auto.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_jac":mmd_alpha_jac.item(),"mmd_beta_jac":mmd_beta_jac.item(),"mmd_pair_jac":mmd_jac.item()}, log_name=f"{n_obs+1}_obs")
        logger.log_metrics({"mmd_alpha_fnpe":mmd_alpha_fnpe.item(),"mmd_beta_fnpe":mmd_beta_fnpe.item(),"mmd_pair_fnpe":mmd_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        
        cov_true = torch.cov(torch.from_numpy(true_samples).T)
        cov_est_gauss = torch.cov(torch.cat((samples_alpha_gauss.detach(),samples_long_gauss.detach()),dim=1).T)
        cov_est_auto = torch.cov(torch.cat((samples_alpha_auto.detach(),samples_long_auto.detach()),dim=1).T)
        cov_est_jac = torch.cov(torch.cat((samples_alpha_jac.detach(),samples_long_jac.detach()),dim=1).T)
        cov_est_fnpe = torch.cov(torch.cat((samples_alpha_fnpe.detach(),samples_long_fnpe.detach()),dim=1).T)
        diff_cov_gauss = torch.mean((cov_true-cov_est_gauss)**2)
        diff_cov_auto = torch.mean((cov_true-cov_est_auto)**2)
        diff_cov_jac = torch.mean((cov_true-cov_est_jac)**2)
        diff_cov_fnpe = torch.mean((cov_true-cov_est_fnpe)**2)

        logger.log_metrics({"diff_cov_gauss":diff_cov_gauss.item(),
                            "diff_cov_auto":diff_cov_auto.item(),
                            "diff_cov_jac":diff_cov_jac.item(),
                            "diff_cov_fnpe":diff_cov_fnpe.item()}, log_name=f"{n_obs+1}_obs")
    
        fig = plt.figure(figsize=(12,8))
        plt.subplot(121)
        sns.kdeplot(samples_alpha_gauss.detach().squeeze(1),bw_method=0.3, color="orange", label="GAUSS")
        sns.kdeplot(samples_alpha_auto.detach().squeeze(1),bw_method=0.3, color="green", label="auto")
        sns.kdeplot(samples_alpha_fnpe.detach().squeeze(1),bw_method=0.3, color="grey", label="Geffner")
        sns.kdeplot(samples_alpha_jac.detach().squeeze(1),bw_method=0.3, color="blue", label="JAC")
        sns.kdeplot(true_samples[:,0], color="red", label="true")
        plt.legend()
        plt.title(fr"$p(\alpha|x_0,...,x_{{{n_obs}}})$")

        plt.subplot(122)
        sns.kdeplot(samples_long_gauss.detach().squeeze(1),bw_method=0.3, color="orange", label="GAUSS")
        sns.kdeplot(samples_long_auto.detach().squeeze(1),bw_method=0.3, color="green", label="auto")
        sns.kdeplot(samples_long_fnpe.detach().squeeze(1),bw_method=0.3, color="grey", label="Geffner")
        sns.kdeplot(samples_long_jac.detach().squeeze(1),bw_method=0.3, color="blue", label="JAC")
        sns.kdeplot(true_samples[:,1], color="red", label="true")
        plt.title(fr"$p(\beta|x_0,...,x_{{{n_obs}}})$")
        plt.legend()
        plt.show()
        logger.log_artifacts(fig, artifact_name=f"marginals_{n_obs+1}_obs_{num_train}_train_{ground_truth}_beta.png",
                            artifact_type='image')
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_alpha_gauss.detach().squeeze(1), y=samples_long_gauss.detach().squeeze(1), cmap="Greens", ax=axes[0][0], fill=True, alpha=0.5)
        axes[0][0].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[0][0].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[0][0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        axes[0][0].set_xlabel(r"$\alpha$")
        axes[0][0].set_ylabel(r"$\beta$")
        axes[0][0].legend()
        axes[0][0].set_title(fr"$p_{{\text{{GAUSS}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")
                            
        sns.kdeplot(x=true_samples[:,0], y=true_samples[:,1], cmap="Reds", ax=axes[0][1],fill=True, alpha=0.5)
        axes[0][1].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[0][1].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[0][1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        axes[0][1].set_xlabel(r"$\alpha$")
        axes[0][1].set_ylabel(r"$\beta$")
        axes[0][1].legend()
        axes[0][1].set_title(fr"$p_{{\text{{true}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_alpha_jac.detach().squeeze(1), y=samples_long_jac.detach().squeeze(1), cmap="Blues", ax=axes[1][0], fill=True, alpha=0.5)
        axes[1][0].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[1][0].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[1][0].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        axes[1][0].set_xlabel(r"$\alpha$")
        axes[1][0].set_ylabel(r"$\beta$")
        axes[1][0].legend()
        axes[1][0].set_title(fr"$p_{{\text{{JAC}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_alpha_fnpe.detach().squeeze(1), y=samples_long_fnpe.detach().squeeze(1), cmap="Greys", ax=axes[1][1], fill=True, alpha=0.5)
        axes[1][1].axvline(x=theta_o[0], ls="dashed", color="orange")
        axes[1][1].axhline(y=theta_o[1], ls="dashed", color="orange")
        axes[1][1].scatter(theta_o[0], theta_o[1], color="orange", marker="o", label="ground truth")
        axes[1][1].set_xlabel(r"$\alpha$")
        axes[1][1].set_ylabel(r"$\beta$")
        axes[1][1].legend()
        axes[1][1].set_title(fr"$p_{{\text{{FNPE}}}}(\alpha,\beta|x_0,...,x_{{{n_obs}}})$")

        axes[0][0].set_aspect('equal')
        axes[0][1].set_aspect('equal')
        axes[1][0].set_aspect('equal')
        axes[1][1].set_aspect('equal')
        plt.tight_layout()
        plt.show()
        logger.log_artifacts(fig, artifact_name=f"2D_plot_{n_obs+1}_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                        artifact_type='image')

if __name__ == "__main__":
    main()