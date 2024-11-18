from sbibm.metrics import mmd
import torch
import pandas as pd

def mmd_score(true_path, estimated_path):
    true_alpha=torch.tensor(pd.read_csv(true_path)["alpha"].values).unsqueeze(1)
    true_beta=torch.tensor(pd.read_csv(true_path)["beta"].values).unsqueeze(1)
    true_samples=torch.cat((true_alpha,true_beta),dim=1)[:10000,:]
    print(true_samples.size())
    estimated_samples=torch.tensor(pd.read_csv(estimated_path).values)[:10000,:]
    #print(estimated_samples)
    acc = mmd(true_samples, estimated_samples)
    return(acc)

if __name__ == "__main__":
    true_path="results/true_samples_alpha_0.5_beta_0.5_nextra_10.csv"
    estimated_path="results/estimated_posterior_samples_naive_True_10_nextra_10000_sim.csv"
    print(mmd_score(true_path, estimated_path))