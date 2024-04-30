# import os
# import numpy as np
import pandas as pd
# from typing import Any, List
import argparse
import sys
import os


def collect_hyb_and_naive(name: str):
    output_dir = "output"
    hyb_csv = os.path.join(output_dir, F"output_tune_{name}_hyb_collect.csv")
    naive_csv = os.path.join(output_dir, F"output_tune_{name}_naive_collect.csv")
    final_csv = os.path.join(output_dir, F"output_tune_{name}_hyb-naive_collect.csv")
    # if os.path.isfile(final_csv):
    #     print(F"{final_csv} already exits. Skipped it.")
    #     return

    hyb_df = pd.read_csv(hyb_csv)
    hyb_df.set_index("name", inplace=True)
    naive_df = pd.read_csv(naive_csv)
    hyb_exe_time = list(hyb_df["best_exe_time"])
    naive_exe_time = list(naive_df["exe_time"])
    speedup = ["{:.6}".format(nai / hyb) for nai, hyb in zip(naive_exe_time, hyb_exe_time)]
    # hyb_df = hyb_df.assign(exe_time_naive=list(naive_df["exe_time"]))
    hyb_df = hyb_df.assign(**{'exe_time_naive': list(naive_df["exe_time"]),
                              'speed_to_naive': speedup})
    hyb_df = hyb_df.rename(columns={"best_exe_time": "exe_time_hyb"})

    print(hyb_df)
    # final_csv = os.path.join(output_dir, F"output_tune_{name}_hyb-naive_collect.csv")
    hyb_df.to_csv(final_csv)

    print(F"\nSaved to {final_csv} .")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parse searching configurations")
    parser.add_argument("--dataset", "-d", type=str, help="dataset name")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    name = args.dataset
    collect_hyb_and_naive(name)
