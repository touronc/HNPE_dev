import json
import glob

# data = {}
# for i in [1,4,7]:
#     files = glob.glob(f"logs/{i}/metrics/*.json")
#     print(files)
#     for f in files:
#         nobs = f.split("/")[-1]
#         nobs = nobs.split("_")[0]
#         with open(f, "r") as f:
#             j=0
#             for line in f:
#                 obj = json.loads(line)
#                 if j==0:
#                     data[nobs] = obj
#                 else:
#                     data[nobs].update(obj)
#                 j=1
#     print(data)
# with open("beta_0_2.json", "w") as f:
#     json.dump(data, f, indent=2)

beta = 0.2
data = {}
files = glob.glob(f"results/compar/*beta_{beta}.json")
print(files)
for f in files:
    nobs = f.split("/")[-1]
    nobs = nobs.split("_")[0]
    print(nobs)
    with open(f, "r") as f:
        j=0
        for line in f:
            obj = json.loads(line)
            if j==0:
                data[nobs] = obj
            else:
                data[nobs].update(obj)
            j=1
print(data)
with open(f"alpha_0.5_beta_{beta}.json", "w") as f:
    json.dump(data, f, indent=2)

