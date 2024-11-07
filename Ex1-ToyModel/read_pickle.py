import pickle
import pandas as pd

def pickle_to_csv(path):
    with open(path, 'rb') as f:
        object = pickle.load(f)
    df = pd.DataFrame(object)
    path_csv = path.split(".")[0]+".csv"
    df.to_csv(path_csv, index=False)

def main():
    path = input("Name of the pickle path : \n")
    pickle_to_csv(path)

if __name__ == "__main__":
    main()
