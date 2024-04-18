import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def save_to_csv(table: dict,
                basename: str):
    output_file = F"{basename}.speedup_gt_130.csv"
    df = pd.DataFrame(data=table)
    print(df)
    df.to_csv(output_file, index=False)


def select(feature_file: str):
    """Select matrices whose hyb speedup is no less than 1.30x.

    Args:
        feature_file (str): _description_
    """
    output_dir = "output"
    df = pd.read_csv(feature_file)

    # Select speedup >= 1.30x
    # above_130 = df[df["speed_to_naive"] >= 2.0]

    # Drop duplicated matrices by names
    # above_130.drop_duplicates(subset=['name'], inplace=True)
    refined = df.drop_duplicates(subset=['name']).drop(columns=['K'])

    basename = os.path.join(output_dir, F"{os.path.splitext(os.path.basename(feature_file))[0]}")
    output_file = F"{basename}.binary_decision_train.csv"
    print(refined)
    refined.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pre-process the feature data")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    select(feature_file)