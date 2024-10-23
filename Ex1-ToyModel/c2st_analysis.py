import pandas as pd
import torch
from metrics.c2st import c2st


def c2st_score(df_true_theta_file, df_estimated_theta_file):
    true_alpha=torch.tensor(pd.read_csv(df_true_theta_file)["alpha"]).unsqueeze(1)
    true_beta=torch.tensor(pd.read_csv(df_true_theta_file)["beta"]).unsqueeze(1)
    true_theta=torch.cat((true_alpha,true_beta), dim=1)
    #print(true_theta)
    estimated_alpha=torch.tensor(pd.read_csv(f"results/{df_estimated_theta_file}")["alpha"]).unsqueeze(1)
    estimated_beta=torch.tensor(pd.read_csv(f"results/{df_estimated_theta_file}")["beta"]).unsqueeze(1)
    estimated_theta=torch.cat((estimated_alpha,estimated_beta), dim=1)
    #print(estimated_theta)
    accuracy = c2st(true_theta, estimated_theta)
    print("C2ST score", accuracy)
    return accuracy


c2st_score("true_posterior_samples_0.001.csv","estimated_posterior_samples_0_nextra.csv")
c2st_score("true_posterior_samples_0.1.csv","estimated_posterior_samples_0_nextra.csv")


