# import os
# import numpy as np
import pandas as pd
# from typing import Any, List
import argparse
import sys
import os

OUTPUT_DIR = "output.correctness"

# def geomean_speedup(baseline: List, x: List) -> Any:
#     return np.exp((np.log(np.array(baseline)) - np.log(np.array(x))).mean())

def check_if_valid(name: str,
                   output_dir: str,
                   feat_sizes: list):
    for feat_size in feat_sizes:
        input = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
        if not os.path.exists(input):
            print(F"File {input} does not exist. Skip extract_data.py on matrix {name}.")
            return False
    
    return True


def extract_data(name: str):

    FEAT_SIZES = [32]
    # FEAT_SIZES = [32, 64, 128, 256, 512]
    output_dir = OUTPUT_DIR
    if not check_if_valid(name,
                          output_dir,
                          FEAT_SIZES):
        sys.exit(-1)
        
    output = os.path.join(output_dir, F"output_tune_{name}_hyb_collect.csv")
    with open(output, "w") as fout:
        # fout.write(F"{HEAD_STR}\n")
        is_first = True
        for feat_size in FEAT_SIZES:
            input = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
            with open(input, "r") as fin:
                while True:
                    line = fin.readline()
                    if line.startswith("name,num_rows,"):
                        if is_first:
                            fout.write(line)
                            is_first = False
                        line = fin.readline()
                        fout.write(line)
                        break
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
