# import os
# import numpy as np
import pandas as pd
# from typing import Any, List


# def geomean_speedup(baseline: List, x: List) -> Any:
#     return np.exp((np.log(np.array(baseline)) - np.log(np.array(x))).mean())


def extract_data():
    datasets = [
        "cora", "citeseer", "pubmed", "ppi", "arxiv", "proteins", "reddit"
    ]

    feat_32_runtimes = []
    feat_64_runtimes = []
    feat_128_runtimes = []
    feat_256_runtimes = []
    feat_512_runtimes = []
    
    for dataset in datasets:
        with open(F"sparsetir_{dataset}_hyb.log", "r") as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.startswith("feat_size ="):
                    feat_size = int(line.split()[-1])
                    line = fin.readline()
                    runtime = float(line.split()[-2])
                    if 32 == feat_size:
                        feat_32_runtimes.append(runtime)
                    elif 64 == feat_size:
                        feat_64_runtimes.append(runtime)
                    elif 128 == feat_size:
                        feat_128_runtimes.append(runtime)
                    elif 256 == feat_size:
                        feat_256_runtimes.append(runtime)
                    elif 512 == feat_size:
                        feat_512_runtimes.append(runtime)
                    else:
                        raise ValueError(F"Feature value {feat_size} is not supported.")
    

    pd.set_option("display.width", 800)
    pd.set_option("display.max_columns", None)

    columns = {
        "Datasets": datasets,
        "SparseTIR.Feat_32": feat_32_runtimes,
        "SparseTIR.Feat_64": feat_64_runtimes,
        "SparseTIR.Feat_128": feat_128_runtimes,
        "SparseTIR.Feat_256": feat_256_runtimes,
        "SparseTIR.Feat_512": feat_512_runtimes,
    }
    dataFrame = pd.DataFrame(data=columns)
    print(dataFrame)
    dataFrame.to_csv(F"spmm.sparsetir.csv")


if __name__ == "__main__":
    extract_data()
