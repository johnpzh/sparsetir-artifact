import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier

OUTPUT_DIR = "data"


def split(feature_file: str,
         train_ratio: float):
    output_dir = OUTPUT_DIR
    df = pd.read_csv(feature_file)

    # Drop duplicated matrices by names
    # df = df[df["speed_to_naive"] >= 1.1] # Drop slowdown cases
    # df.drop_duplicates(subset=['name'], inplace=True, keep='last') # Drop duplicate names
    df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names
    # df.drop(columns=['K'], inplace=True) # Drop column 'K'
    # df.dropna(inplace=True) # Drop rows with any NaN

    # Label data according to speed_to_naive
    df.loc[df["speed_to_naive"] >= 1.1, "speed_to_naive"] = 2
    df.loc[df["speed_to_naive"] < 1.1, "speed_to_naive"] = 0

    # Drop density-related columns and K
    df.drop(columns=["avg_nnz_density_per_row",
                     "min_nnz_density_per_row",
                     "max_nnz_density_per_row",
                     "std_dev_nnz_density_per_row",
                     "K",
                     "best_num_partitions",
                     "best_max_bucket_width",
                     "autotuning_time(s)",
                     "exe_time_hyb(ms)",
                     "exe_time_naive(ms)",
                     ], inplace=True)

    # Divide data
    num_rows = df.shape[0]
    num_trains = int(num_rows * train_ratio)
    if num_trains == num_rows:
        num_trains = num_rows - 1
    num_tests = num_rows - num_trains
    random_indices = np.random.randint(low=0, high=num_rows, size=(num_tests))
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for row_ind in range(num_rows):
        if row_ind in random_indices:
            test_df = pd.concat([test_df, df.iloc[row_ind:row_ind + 1]])
        else:
            train_df = pd.concat([train_df, df.iloc[row_ind:row_ind + 1]])

    # Save
    basename = os.path.splitext(os.path.basename(feature_file))[0]
    train_file = os.path.join(output_dir, F"{basename}.for_selection.train_set.csv")
    test_file = os.path.join(output_dir, F"{basename}.for_selection.infer_set.csv")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    # # test
    # print(F"train_df: {train_df}")
    # print(F"test_df: {test_df}")
    # # end test

    print(F"Ratio {train_ratio} ({num_trains}/{num_rows}) rows saved to {train_file} .")
    print(F"Ratio {1 - train_ratio} ({num_rows - num_trains}/{num_rows}) rows saved to {test_file} .")
    ##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser("split data to training set and test set for selection")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    parser.add_argument("--ratio", "-r", type=float, default=0.8, help="ratio of data for training set")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    train_ratio = args.ratio
    split(feature_file, train_ratio)