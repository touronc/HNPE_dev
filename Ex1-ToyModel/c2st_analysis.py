import pandas as pd
import torch
from metrics.c2st import c2st


def c2st_score(df_true_theta_file, df_estimated_theta_file):
    """
    compute the C2ST score between true posterior samples and estimated ones
    the closer to 0.5 the better
    """

    true_alpha=torch.tensor(pd.read_csv(f"results/{df_true_theta_file}")["alpha"]).unsqueeze(1)
    true_beta=torch.tensor(pd.read_csv(f"results/{df_true_theta_file}")["beta"]).unsqueeze(1)
    true_theta=torch.cat((true_alpha,true_beta), dim=1)
    #print(true_theta)
    estimated_alpha=torch.tensor(pd.read_csv(f"results/{df_estimated_theta_file}")["alpha"]).unsqueeze(1)
    estimated_beta=torch.tensor(pd.read_csv(f"results/{df_estimated_theta_file}")["beta"]).unsqueeze(1)
    estimated_theta=torch.cat((estimated_alpha,estimated_beta), dim=1)
    #print(estimated_theta)
    accuracy = c2st(true_theta, estimated_theta)
    print("C2ST score :", accuracy.item())
    return accuracy

def c2st_score_df(df_true_theta, df_estimated_theta):
    """
    compute the C2ST score between true posterior samples and estimated ones
    the closer to 0.5 the better
    """

    true_alpha=torch.tensor(df_true_theta["alpha"]).unsqueeze(1)
    true_beta=torch.tensor(df_true_theta["beta"]).unsqueeze(1)
    true_theta=torch.cat((true_alpha,true_beta), dim=1)
    #print(true_theta)
    estimated_alpha=torch.tensor(df_estimated_theta["alpha"]).unsqueeze(1)
    estimated_beta=torch.tensor(df_estimated_theta["beta"]).unsqueeze(1)
    estimated_theta=torch.cat((estimated_alpha,estimated_beta), dim=1)
    #print(estimated_theta)
    accuracy = c2st(true_theta, estimated_theta)
    print("C2ST score :", accuracy.item())
    return accuracy


# ATTENTION les fichiers doivent être stockés dans le répertoire results
# Ne donner que le nom du fichier dans ce répertoire

#print("____ Computation running ____")


#c2st_score("true_samples_alpha_0.5_beta_0.5_nextra_100.csv","estimated_posterior_samples_naive_False_100_nextra_10000_sim.csv")
#c2st_score("true_posterior_samples_0.001.csv","estimated_posterior_samples_naive_True_0_nextra_10000_sim_mlxp.csv")


#c2st_score("true_posterior_samples_0.001.csv","estimated_posterior_samples_0_nextra_10000_sim.csv")
#c2st_score("true_posterior_samples_0.1.csv","estimated_posterior_samples_10_nextra_10000_sim.csv")


