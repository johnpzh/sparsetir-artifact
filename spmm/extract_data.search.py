# import os
# import numpy as np
import pandas as pd
# from typing import Any, List
import argparse
import sys
import os


# def geomean_speedup(baseline: List, x: List) -> Any:
#     return np.exp((np.log(np.array(baseline)) - np.log(np.array(x))).mean())


def extract_data(name: str):
    # datasets = [
    #     "cora", "citeseer", "pubmed", "ppi", "arxiv", "proteins", "reddit"
    # ]

    # feat_32_runtimes = []
    # feat_64_runtimes = []
    # feat_128_runtimes = []
    # feat_256_runtimes = []
    # feat_512_runtimes = []

    FEAT_SIZES = [32, 64, 128, 256, 512]
    HEAD_STR = "name,num_rows,num_cols,nnz,avg_nnz_per_row,min_nnz_per_row,max_nnz_per_row,std_dev_nnz_per_row,avg_nnz_densitry_per_row,min_nnz_densitry_per_row,max_nnz_densitry_per_row,std_dev_nnz_density_per_row,K,best_num_partitions,best_max_bucket_width,best_exe_time"
    output_dir = "output"
    output = os.path.join(output_dir, F"output_tune_{name}_hyb_collect.csv")
    with open(output, "w") as fout:
        fout.write(F"{HEAD_STR}\n")
        for feat_size in FEAT_SIZES:
            input = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
            with open(input, "r") as fin:
                while True:
                    line = fin.readline()
                    if line.startswith(HEAD_STR):
                        line = fin.readline()
                        fout.write(line)
                    if not line:
                        break
    
    print(F"\nSaved to {output} .")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parse searching configurations")
    parser.add_argument("--dataset", "-d", type=str, help="dataset name")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    name = args.dataset
    extract_data(name)
