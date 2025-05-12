import json
import matplotlib.pyplot as plt
import torch

beta = "2"
# with open(f"beta_0_{beta}_noise_0_01.json", "r") as f:
with open(f"alpha_0.5_beta_0.{beta}.json", "r") as f:
    data = json.load(f)
nobs = [3,6,11]
c2st_alpha_gauss=[data["0"]["c2st_alpha_diff"]]
c2st_alpha_auto=[data["0"]["c2st_alpha_diff"]]
c2st_alpha_jac=[data["0"]["c2st_alpha_diff"]]
c2st_alpha_fnpe=[data["0"]["c2st_alpha_diff"]]
c2st_alpha_hnpe=[data["0"]["c2st_alpha_hnpe"]]
c2st_beta_gauss=[data["0"]["c2st_beta_diff"]]
c2st_beta_auto=[data["0"]["c2st_beta_diff"]]
c2st_beta_jac=[data["0"]["c2st_beta_diff"]]
c2st_beta_fnpe=[data["0"]["c2st_beta_diff"]]
c2st_beta_hnpe=[data["0"]["c2st_beta_hnpe"]]
c2st_pair_gauss=[data["0"]["c2st_pair_diff"]]
c2st_pair_auto=[data["0"]["c2st_pair_diff"]]
c2st_pair_jac=[data["0"]["c2st_pair_diff"]]
c2st_pair_fnpe=[data["0"]["c2st_pair_diff"]]
c2st_pair_hnpe=[data["0"]["c2st_pair_hnpe"]]

mmd_alpha_gauss=[data["0"]["mmd_alpha_diff"]]
mmd_alpha_auto=[data["0"]["mmd_alpha_diff"]]
mmd_alpha_jac=[data["0"]["mmd_alpha_diff"]]
mmd_alpha_fnpe=[data["0"]["mmd_alpha_diff"]]
mmd_alpha_hnpe=[data["0"]["mmd_alpha_hnpe"]]
mmd_beta_gauss=[data["0"]["mmd_beta_diff"]]
mmd_beta_auto=[data["0"]["mmd_beta_diff"]]
mmd_beta_jac=[data["0"]["mmd_beta_diff"]]
mmd_beta_fnpe=[data["0"]["mmd_beta_diff"]]
mmd_beta_hnpe=[data["0"]["mmd_beta_hnpe"]]
mmd_pair_gauss=[data["0"]["mmd_pair_diff"]]
mmd_pair_auto=[data["0"]["mmd_pair_diff"]]
mmd_pair_jac=[data["0"]["mmd_pair_diff"]]
mmd_pair_fnpe=[data["0"]["mmd_pair_diff"]]
mmd_pair_hnpe=[data["0"]["mmd_pair_hnpe"]]

# cov_true = torch.tensor(data["0"]["cov_true"])
# cov_est =  torch.tensor(data["0"]["cov_est"])
# diff_cov_gauss = [torch.mean((cov_est-cov_true)**2)]
# diff_cov_auto = [torch.mean((cov_est-cov_true)**2)]
# diff_cov_jac = [torch.mean((cov_est-cov_true)**2)]
# diff_cov_fnpe = [torch.mean((cov_est-cov_true)**2)]

diff_cov_gauss = [data["0"]["diff_cov"]]
diff_cov_auto = [data["0"]["diff_cov"]]
diff_cov_jac = [data["0"]["diff_cov"]]
diff_cov_fnpe = [data["0"]["diff_cov"]]
diff_cov_hnpe = [data["0"]["diff_cov_hnpe"]]

wass_gauss = [data["0"]["wass_diff"]]
wass_auto = [data["0"]["wass_diff"]]
wass_jac = [data["0"]["wass_diff"]]
wass_fnpe = [data["0"]["wass_diff"]]
wass_hnpe = [data["0"]["wass_hnpe"]]

for n in nobs:
    c2st_alpha_gauss.append(data[f"{n}"]["c2st_alpha_gauss"])
    c2st_alpha_auto.append(data[f"{n}"]["c2st_alpha_auto"])
    c2st_alpha_jac.append(data[f"{n}"]["c2st_alpha_jac"])
    c2st_alpha_fnpe.append(data[f"{n}"]["c2st_alpha_fnpe"])
    c2st_alpha_hnpe.append(data[f"{n}"]["c2st_alpha_hnpe"])
    c2st_beta_gauss.append(data[f"{n}"]["c2st_beta_gauss"])
    c2st_beta_auto.append(data[f"{n}"]["c2st_beta_auto"])
    c2st_beta_jac.append(data[f"{n}"]["c2st_beta_jac"])
    c2st_beta_fnpe.append(data[f"{n}"]["c2st_beta_fnpe"])
    c2st_beta_hnpe.append(data[f"{n}"]["c2st_beta_hnpe"])
    c2st_pair_gauss.append(data[f"{n}"]["c2st_pair_gauss"])
    c2st_pair_auto.append(data[f"{n}"]["c2st_pair_auto"])
    c2st_pair_jac.append(data[f"{n}"]["c2st_pair_jac"])
    c2st_pair_fnpe.append(data[f"{n}"]["c2st_pair_fnpe"])
    c2st_pair_hnpe.append(data[f"{n}"]["c2st_pair_hnpe"])

    mmd_alpha_gauss.append(data[f"{n}"]["mmd_alpha_gauss"])
    mmd_alpha_auto.append(data[f"{n}"]["mmd_alpha_auto"])
    mmd_alpha_jac.append(data[f"{n}"]["mmd_alpha_jac"])
    mmd_alpha_fnpe.append(data[f"{n}"]["mmd_alpha_fnpe"])
    mmd_alpha_hnpe.append(data[f"{n}"]["mmd_alpha_hnpe"])
    mmd_beta_gauss.append(data[f"{n}"]["mmd_beta_gauss"])
    mmd_beta_auto.append(data[f"{n}"]["mmd_beta_auto"])
    mmd_beta_jac.append(data[f"{n}"]["mmd_beta_jac"])
    mmd_beta_fnpe.append(data[f"{n}"]["mmd_beta_fnpe"])
    mmd_beta_hnpe.append(data[f"{n}"]["mmd_beta_hnpe"])
    mmd_pair_gauss.append(data[f"{n}"]["mmd_pair_gauss"])
    mmd_pair_auto.append(data[f"{n}"]["mmd_pair_auto"])
    mmd_pair_jac.append(data[f"{n}"]["mmd_pair_jac"])
    mmd_pair_fnpe.append(data[f"{n}"]["mmd_pair_fnpe"])
    mmd_pair_hnpe.append(data[f"{n}"]["mmd_pair_hnpe"])

    diff_cov_gauss.append(data[f"{n}"]["diff_gauss"])
    diff_cov_auto.append(data[f"{n}"]["diff_auto"])
    diff_cov_jac.append(data[f"{n}"]["diff_jac"])
    diff_cov_fnpe.append(data[f"{n}"]["diff_fnpe"])
    diff_cov_hnpe.append(data[f"{n}"]["diff_hnpe"])
    # cov_true = torch.tensor(data[f"{n}"]["cov_true"])
    # cov_est = torch.tensor(data[f"{n}"]["cov_est_gauss"])
    # diff_cov_gauss.append(torch.mean((cov_est-cov_true)**2))
    # cov_est = torch.tensor(data[f"{n}"]["cov_est_auto"])
    # diff_cov_auto.append(torch.mean((cov_est-cov_true)**2))
    # cov_est = torch.tensor(data[f"{n}"]["cov_est_jac"])
    # diff_cov_jac.append(torch.mean((cov_est-cov_true)**2))
    # cov_est = torch.tensor(data[f"{n}"]["cov_est_fnpe"])
    # diff_cov_fnpe.append(torch.mean((cov_est-cov_true)**2))

    wass_gauss.append(data[f"{n}"]["wass_gauss"])
    wass_auto.append(data[f"{n}"]["wass_auto"])
    wass_jac.append(data[f"{n}"]["wass_jac"])
    wass_fnpe.append(data[f"{n}"]["wass_fnpe"])
    wass_hnpe.append(data[f"{n}"]["wass_hnpe"])

plt.figure(figsize=(15,10))
plt.subplot(331)
plt.plot([1]+nobs,c2st_alpha_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,c2st_alpha_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,c2st_alpha_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,c2st_alpha_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,c2st_alpha_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.5, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.505, 'optimal C2ST', horizontalalignment = 'center', color='black', fontweight="bold")
plt.ylim(0.495)
plt.legend()
plt.ylabel("C2ST")
plt.xlabel("nb of conditional obs")
plt.title(r"$\alpha$")
plt.subplot(332)
plt.plot([1]+nobs,c2st_beta_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,c2st_beta_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,c2st_beta_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,c2st_beta_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,c2st_beta_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.5, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.505, 'optimal C2ST', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.ylabel("C2ST")
plt.title(r"$\beta$")
plt.subplot(333)
plt.plot([1]+nobs,c2st_pair_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,c2st_pair_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,c2st_pair_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,c2st_pair_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,c2st_pair_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.5, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.505, 'optimal C2ST', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.ylabel("C2ST")
plt.title(r"$(\alpha,\beta)$")
plt.subplot(334)
plt.plot([1]+nobs,mmd_alpha_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,mmd_alpha_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,mmd_alpha_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,mmd_alpha_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,mmd_alpha_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.0, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.007, 'optimal MMD', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.ylabel("MMD")
plt.title(r"$\alpha$")
plt.subplot(335)
plt.plot([1]+nobs,mmd_beta_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,mmd_beta_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,mmd_beta_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,mmd_beta_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,mmd_beta_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.0, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.007, 'optimal MMD', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.ylabel("MMD")
plt.title(r"$\beta$")
plt.subplot(336)
plt.plot([1]+nobs,mmd_pair_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,mmd_pair_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,mmd_pair_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,mmd_pair_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,mmd_pair_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.xticks([1]+nobs)
plt.axhline(y=0.0, ls ="dotted", color='black', lw=0.5)
plt.text(9, 0.005, 'optimal MMD', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.ylabel("MMD")
plt.title(r"$(\alpha,\beta)$")
plt.subplot(338)
plt.plot([1]+nobs,diff_cov_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,diff_cov_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,diff_cov_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,diff_cov_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,diff_cov_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.axhline(y=0.0, ls ="dotted", color='black', lw=0.5)
plt.text(3, 0.0000007, 'optimal diff', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.xticks([1]+nobs)
plt.ylabel(r"$||\Sigma_{\text{true}}-\Sigma_{\text{est}}||^2$")
plt.subplot(339)
plt.plot([1]+nobs,wass_gauss, ls="dashed", marker="o", label="gauss", color="orange")
plt.plot([1]+nobs,wass_auto, ls="dashed", marker="o", label="auto", color="green")
plt.plot([1]+nobs,wass_jac, ls="dashed", marker="o", label="jac",color="red")
plt.plot([1]+nobs,wass_fnpe, ls="dashed", marker="o", label="fnpe",color="grey")
plt.plot([1]+nobs,wass_hnpe, ls="dashed", marker="o", label="HNPE",color="blue")
plt.axhline(y=0.0, ls ="dotted", color='black', lw=0.5)
plt.text(7.5, 0.001, 'optimal dist', horizontalalignment = 'center', color='black', fontweight="bold")
plt.legend()
plt.xlabel("nb of conditional obs")
plt.xticks([1]+nobs)
plt.ylabel(r"$\mathcal{W}_2(p_{\text{true}},p_{\text{estim}})$")
plt.suptitle(fr"$\alpha = 0.5 \ \ \beta=0.{beta}$")
plt.tight_layout()
plt.savefig(f"alpha_0.5_beta_0.{beta}_noise_0_01_mean_metrics.png")
plt.show()
