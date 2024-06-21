import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier



def drop_num_parts_only_1(feature_file: str):
    df = pd.read_csv(feature_file)

    # Drop duplicated matrices by names
    df = df[df["speed_to_naive"] >= 1.1] # Drop slowdown cases
    df = df[df["best_num_partitions"] > 1] # Drop slowdown cases
    # df.drop_duplicates(subset=['name'], inplace=True, keep='last') # Drop duplicate names
    df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names
    # df.drop(columns=['K'], inplace=True) # Drop column 'K'
    # df.dropna(inplace=True) # Drop rows with any NaN

    # Save
    basename = os.path.splitext(os.path.basename(feature_file))[0]
    output_file = F"{basename}.num_parts_gt_one.csv"
    df.to_csv(output_file, index=False)
    print(F"Saved to {output_file} .")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Drop duplicate names in csv")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    # split(feature_file, train_ratio)
    drop_num_parts_only_1(feature_file)