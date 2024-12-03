import json
import glob
import numpy as np
import matplotlib.pyplot as plt


files = glob.glob("*.json")

acc_dict = np.zeros(11)
true_var = np.zeros((11,2))
estim_var = np.zeros((11,2))

for f in files:
    # store the nb of extra obs
    nextra = int(f.split('.')[0])
    with open(f) as file:
        for line in file:
            data = json.loads(line)
            if "accuracy" in data.keys():
                acc_dict[nextra]=float(data["accuracy"])
            elif "true_variance_alpha" in data.keys():
                true_var[nextra,0] = data["true_variance_alpha"]
            elif "true_variance_beta" in data.keys():
                true_var[nextra,1] = data["true_variance_beta"]
            elif "estimated_variance_alpha" in data.keys():
                estim_var[nextra,0] = data["estimated_variance_alpha"]
            else:
                estim_var[nextra,1] = data["estimated_variance_beta"]

fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(range(11),acc_dict, "o--g")
plt.axhline(y=0.5, ls ="dotted", color='orange')
plt.text(8, 0.505, 'optimal score', horizontalalignment = 'center', color='orange', fontweight="bold")
plt.title("Mean C2ST scores")
plt.xlabel("nb of extra obs")
plt.subplot(2,2,2)
plt.plot(range(11),true_var[:,0], "o--r", label="true")
plt.plot(range(11),estim_var[:,0], "o--b", label="estimated")
plt.axhline(y=0.01, ls ="dotted", color='orange')
plt.text(1, 0.011, 'noise', horizontalalignment = 'center', color='orange', fontweight="bold")
plt.legend()
plt.title(r"Mean variance of $\alpha$ parameters")
plt.xlabel("nb of extra obs")
plt.subplot(2,2,4)
plt.plot(range(11),true_var[:,1], "o--r", label="true")
plt.plot(range(11),estim_var[:,1], "o--b", label="estimated")
plt.axhline(y=0.01, ls ="dotted", color='orange')
plt.text(1, 0.011, 'noise', horizontalalignment = 'center', color='orange', fontweight="bold")
plt.legend()
plt.title(r"Mean variance of $\beta$ parameters")
plt.xlabel("nb of extra obs")
plt.savefig("evolution_mean_metrics.pdf", format="pdf")
plt.show()



