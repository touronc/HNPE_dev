from sbi.inference import NPSE, SNPE_C
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sbi import inference as sbi_inference
import mlxp
from sbi.utils.metrics import c2st, unbiased_mmd_squared
#from get_true_samples_nextra_obs import get_posterior_samples

mu_prior = torch.ones(2)
cov_prior = torch.eye(2)*3
prior_beta = torch.distributions.MultivariateNormal(mu_prior, cov_prior)
cov_lik = torch.eye(2)*2

def simulator1(theta):
    return torch.distributions.MultivariateNormal(theta, cov_lik).sample() 

cov_post = torch.linalg.inv(torch.linalg.inv(cov_prior)+torch.linalg.inv(cov_lik))

def mu_post(x):
    "mean of the individual posterior p(beta|x)"
    return cov_post@(torch.linalg.inv(cov_lik)@x.reshape(2,1) + torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1))

def true_post_score(theta,x):
    "analytical score of the true individual posterior p(beta|x)"
    return -torch.linalg.inv(cov_post)@(theta.reshape(2,1)-mu_post(x))

def true_diff_post_score(theta,x,t, score_net):
    "analytical score of the true diffused posterior p_t(beta|x)"
    alpha_t = score_net.alpha(t).item()
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_post)@(theta.reshape(2,1)-alpha_t**0.5*mu_post(x))

def cov_tall_post(x):
    "covariance matrix of the tall true posterior p(beta|x0,...xn)"
    return torch.linalg.inv(torch.linalg.inv(cov_prior)+x.shape[0]*torch.linalg.inv(cov_lik))

def mu_tall_post(x):
    "mean of the true tall posterior p(beta|x0,...xn)"
    tmp = torch.linalg.inv(cov_prior)@mu_prior.reshape(2,1)
    for i in range(x.shape[0]):
        tmp += torch.linalg.inv(cov_lik)@x[i,:].reshape(2,1)
    return cov_tall_post(x)@tmp

def true_tall_post_score(theta,x):
    "analytical score of the true tall posterior p(beta|x0,...xn)"
    return -torch.linalg.inv(cov_tall_post(x))@(theta.reshape(2,1)-mu_tall_post(x))

def true_diff_tall_post_score(theta, x,t,score_net):
    "analytical score of the true diffused tall posterior p_t(beta|x0,...xn)"
    alpha_t = score_net.alpha(t)
    return -torch.linalg.inv((1-alpha_t)*torch.eye(2)+alpha_t*cov_tall_post(x))@(theta.reshape(2,1)-alpha_t**0.5*mu_tall_post(x))

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
    num_train = cfg.num_train

    simulator = simulator1

    beta_train = prior_beta.sample((num_train,)) # dataset for the simulator
    x_train = simulator(beta_train)

    print("Training dimensions : \n beta : ", beta_train.size(),
        "\n x : ",x_train.size())

    # diffusion model for beta parameters
    inference_beta = NPSE(prior=prior_beta, sde_type="vp")
    inference_beta.append_simulations(beta_train, x_train)
    score_estimator = inference_beta.train()
    posterior_beta = inference_beta.build_posterior(score_estimator, sample_with="sde")

    # inference
    num_samples = cfg.num_samples
    beta_o = torch.tensor([cfg.alpha,cfg.beta])
    x_o = simulator1(beta_o)
    if cfg.nextra==0:
        # sampling with 0 extra observation
        true_samples = torch.distributions.MultivariateNormal(mu_post(x_o).squeeze(1), cov_post).sample((num_samples,))
        samples_beta = posterior_beta.sample((num_samples,), x=x_o, predictor="euler_maruyama", corrector="langevin").detach()
        acc_beta = c2st(true_samples,samples_beta)
        mmd_beta = unbiased_mmd_squared(true_samples,samples_beta)
        cov_true = torch.cov(true_samples.T)
        cov_est = torch.cov(samples_beta.T)
        diff_cov = torch.mean((cov_est-cov_true)**2)
        wass = wasserstein_dist(torch.mean(true_samples,dim=0),torch.mean(samples_beta),cov_true,cov_est)
        logger.log_metrics({"diff_cov":diff_cov.item(),"acc": acc_beta.item(), "mmd": mmd_beta.item(),"wass":wass.item()}, log_name="0_obs")
        
        fig = plt.figure(figsize=(11,8))
        plt.subplot(121)
        sns.kdeplot(samples_beta[:,0], color="blue", label="estimated")
        sns.kdeplot(true_samples[:,0], color="red", label="true")
        plt.axvline(x=beta_o[0], ls="dashed", color="orange", label="ground truth")
        plt.xlabel("dim1")
        plt.legend()
        plt.title(r"$p(\beta_1|x_0)$")
        plt.subplot(122)
        sns.kdeplot(samples_beta[:,1], color="blue", label="estimated")
        sns.kdeplot(true_samples[:,1], color="red", label="true")
        plt.axvline(x=beta_o[1], ls="dashed", color="orange", label="ground truth")
        plt.xlabel("dim2")
        plt.title(r"$p(\beta_2|x_0)$")
        plt.legend()
        logger.log_artifacts(fig, artifact_name=f"marginals_1_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                            artifact_type='image')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_beta[:,0], y=samples_beta[:,1], cmap="Blues", ax=ax1, fill=True, alpha=0.5,label="estimated")
        ax1.axvline(x=beta_o[0], ls="dashed", color="orange")
        ax1.axhline(y=beta_o[1], ls="dashed", color="orange")
        ax1.scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        ax1.set_xlabel("dim1")
        ax1.set_ylabel("dim2")
        ax1.legend()
        ax1.set_title(r"$p_{\text{estim}}(\beta_1,\beta_2|x_0)$")

        sns.kdeplot(x=true_samples[:,0], y=true_samples[:,1], cmap="Reds", ax=ax2,fill=True, label="true", alpha=0.5)
        ax2.axvline(x=beta_o[0], ls="dashed", color="orange")
        ax2.axhline(y=beta_o[1], ls="dashed", color="orange")
        ax2.scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        ax2.set_xlabel("dim1")
        ax2.set_ylabel("dim2")
        ax2.legend()
        ax2.set_title(r"$p_{\text{true}}(\beta_1,\beta_2|x_0)$")

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        logger.log_artifacts(fig, artifact_name=f"2D_plot_1_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                            artifact_type='image')
        
    else:
        # inference with several additional observations
        n_obs = cfg.nextra
        extra_obs = prior_beta.sample((n_obs,))
        beta_o_long = torch.cat((beta_o.unsqueeze(0),extra_obs),dim=0)
        x_o_long = simulator1(beta_o_long)
        
        true_samples = torch.distributions.MultivariateNormal(mu_tall_post(x_o_long).squeeze(1), cov_tall_post(x_o_long)).sample((num_samples,))
        samples_long_gauss = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="gauss", predictor="euler_maruyama", corrector="langevin").detach()
        samples_long_auto = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="auto_gauss", predictor="euler_maruyama", corrector="langevin").detach()
        samples_long_fnpe = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="fnpe", predictor="euler_maruyama", corrector="langevin").detach()
        samples_long_jac = posterior_beta.sample((num_samples,), x=x_o_long, iid_method="jac_gauss", predictor="euler_maruyama", corrector="langevin").detach()
        print(samples_long_jac.size())
        acc_beta_gauss = c2st(true_samples,samples_long_gauss)
        acc_beta_auto = c2st(true_samples,samples_long_auto)
        acc_beta_jac = c2st(true_samples,samples_long_jac)
        acc_beta_fnpe = c2st(true_samples,samples_long_fnpe)
        logger.log_metrics({"acc_gauss":acc_beta_gauss.item(),"acc_auto": acc_beta_auto.item(),
                            "acc_jac":acc_beta_jac.item(),
                            "acc_fnpe":acc_beta_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        mmd_beta_gauss = unbiased_mmd_squared(true_samples,samples_long_gauss)
        mmd_beta_auto = unbiased_mmd_squared(true_samples,samples_long_auto)
        mmd_beta_jac = unbiased_mmd_squared(true_samples,samples_long_jac)
        mmd_beta_fnpe = unbiased_mmd_squared(true_samples,samples_long_fnpe)
        logger.log_metrics({"mmd_gauss":mmd_beta_gauss.item(),"mmd_auto": mmd_beta_auto.item(),
                            "mmd_jac":mmd_beta_jac.item(),
                            "mmd_fnpe":mmd_beta_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        cov_true = torch.cov(true_samples.T)
        cov_est_gauss = torch.cov(samples_long_gauss.T)
        cov_est_auto = torch.cov(samples_long_auto.T)
        cov_est_jac = torch.cov(samples_long_jac.T)
        cov_est_fnpe = torch.cov(samples_long_fnpe.T)
        diff_gauss = torch.mean((cov_true-cov_est_gauss)**2)
        diff_auto = torch.mean((cov_true-cov_est_auto)**2)
        diff_jac = torch.mean((cov_true-cov_est_jac)**2)
        diff_fnpe = torch.mean((cov_true-cov_est_fnpe)**2)

        logger.log_metrics({"diff_gauss":diff_gauss.item(),"diff_auto": diff_auto.item(),
                            "diff_jac":diff_jac.item(),
                            "diff_fnpe":diff_fnpe.item()}, log_name=f"{n_obs+1}_obs")
        wass_gauss = wasserstein_dist(torch.mean(true_samples,dim=0),torch.mean(samples_long_gauss),cov_true,cov_est_gauss)
        wass_auto = wasserstein_dist(torch.mean(true_samples,dim=0),torch.mean(samples_long_auto),cov_true,cov_est_auto)
        wass_jac = wasserstein_dist(torch.mean(true_samples,dim=0),torch.mean(samples_long_jac),cov_true,cov_est_jac)
        wass_fnpe = wasserstein_dist(torch.mean(true_samples,dim=0),torch.mean(samples_long_fnpe),cov_true,cov_est_fnpe)
        logger.log_metrics({"wass_gauss":wass_gauss.item(),"wass_auto": wass_auto.item(),
                            "wass_jac":wass_jac.item(),
                            "wass_fnpe":wass_fnpe.item()}, log_name=f"{n_obs+1}_obs")

        fig = plt.figure(figsize=(12,8))
        plt.subplot(121)
        sns.kdeplot(samples_long_gauss[:,0], color="orange", label="GAUSS")
        sns.kdeplot(samples_long_auto[:,0], color="green", label="auto")
        sns.kdeplot(samples_long_fnpe[:,0], color="grey", label="Geffner")
        sns.kdeplot(samples_long_jac[:,0], color="blue", label="JAC")
        sns.kdeplot(true_samples[:,0], color="red", label="true")
        plt.legend()
        plt.title(fr"$p(\beta_1|x_0,...,x_{{{n_obs}}})$")

        plt.subplot(122)
        sns.kdeplot(samples_long_gauss[:,1], color="orange", label="GAUSS")
        sns.kdeplot(samples_long_auto[:,1], color="green", label="auto")
        sns.kdeplot(samples_long_fnpe[:,1], color="grey", label="Geffner")
        sns.kdeplot(samples_long_jac[:,1], color="blue", label="JAC")
        sns.kdeplot(true_samples[:,1], color="red", label="true")
        plt.legend()
        plt.title(fr"$p(\beta_2|x_0,...,x_{{{n_obs}}})$")
        logger.log_artifacts(fig, artifact_name=f"marginals_{n_obs+1}_obs_{num_train}_train_{cfg.beta}_beta.png",
                            artifact_type='image')
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
        sns.kdeplot(x=samples_long_gauss[:,0], y=samples_long_gauss[:,1], cmap="Greens", ax=axes[0][0], fill=True, alpha=0.5)
        axes[0][0].axvline(x=beta_o[0], ls="dashed", color="orange")
        axes[0][0].axhline(y=beta_o[1], ls="dashed", color="orange")
        axes[0][0].scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        axes[0][0].set_xlabel("dim1")
        axes[0][0].set_ylabel("dim2")
        axes[0][0].legend()
        axes[0][0].set_title(fr"$p_{{\text{{GAUSS}}}}(\beta_1,\beta_2|x_0,...,x_{{{n_obs}}})$")
                            
        sns.kdeplot(x=true_samples[:,0], y=true_samples[:,1], cmap="Reds", ax=axes[0][1],fill=True, alpha=0.5)
        axes[0][1].axvline(x=beta_o[0], ls="dashed", color="orange")
        axes[0][1].axhline(y=beta_o[1], ls="dashed", color="orange")
        axes[0][1].scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        axes[0][1].set_xlabel("dim1")
        axes[0][1].set_ylabel("dim2")
        axes[0][1].legend()
        axes[0][1].set_title(fr"$p_{{\text{{true}}}}(\beta_1,\beta_2|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_long_jac[:,0], y=samples_long_jac[:,1], cmap="Blues", ax=axes[1][0], fill=True, alpha=0.5)
        axes[1][0].axvline(x=beta_o[0], ls="dashed", color="orange")
        axes[1][0].axhline(y=beta_o[1], ls="dashed", color="orange")
        axes[1][0].scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        axes[1][0].set_xlabel("dim1")
        axes[1][0].set_ylabel("dim2")
        axes[1][0].legend()
        axes[1][0].set_title(fr"$p_{{\text{{JAC}}}}(\beta_1,\beta_2|x_0,...,x_{{{n_obs}}})$")

        sns.kdeplot(x=samples_long_fnpe[:,0], y=samples_long_fnpe[:,1], cmap="Greys", ax=axes[1][1], fill=True, alpha=0.5)
        axes[1][1].axvline(x=beta_o[0], ls="dashed", color="orange")
        axes[1][1].axhline(y=beta_o[1], ls="dashed", color="orange")
        axes[1][1].scatter(beta_o[0], beta_o[1], color="orange", marker="o", label="ground truth")
        axes[1][1].set_xlabel("dim1")
        axes[1][1].set_ylabel("dim2")
        axes[1][1].legend()
        axes[1][1].set_title(fr"$p_{{\text{{FNPE}}}}(\beta_1,\beta_2|x_0,...,x_{{{n_obs}}})$")

        axes[0][0].set_aspect('equal')
        axes[0][1].set_aspect('equal')
        axes[1][0].set_aspect('equal')
        axes[1][1].set_aspect('equal')
        plt.tight_layout()
        logger.log_artifacts(fig, artifact_name=f"2D_plot_{n_obs+1}_obs_{num_train}_train_{cfg.alpha}_alpha_{cfg.beta}_beta.png",
                        artifact_type='image')
  

if __name__ == "__main__":
    main()
