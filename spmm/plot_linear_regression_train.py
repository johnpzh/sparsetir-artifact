import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def save_to_csv(table: dict,
                basename: str):
    output_file = F"{basename}.linear_regression.train_output.csv"
    df = pd.DataFrame(data=table)
    print(df)
    df.to_csv(output_file, index=False)

    # ##################
    # columns = {
    #     head: column
    # }
    # output_file = F"{basename}.linear_regression.{head}.csv"
    # dataFrame = pd.DataFrame(data=columns)
    # print(dataFrame)
    # dataFrame.to_csv(output_file, index=False)


def linear_regression(feature_file: str):
    output_dir = "output"
    dataFrame = pd.read_csv(feature_file)

    y_num_partitions = dataFrame['best_num_partitions'].to_numpy()
    y_max_bucket_width = dataFrame['best_max_bucket_width'].to_numpy()
    X = dataFrame.drop(columns=['name', 'best_num_partitions', 'best_max_bucket_width', 'exe_time_hyb', 'exe_time_naive', 'speed_to_naive']).to_numpy()

    # Linear regression
    b_num_partitions, _, _, _ = np.linalg.lstsq(X, y_num_partitions, rcond=None)
    b_max_bucket_width, _, _, _ = np.linalg.lstsq(X, y_max_bucket_width, rcond=None)

    # Save to csv
    basename = os.path.join(output_dir, F"{os.path.splitext(os.path.basename(feature_file))[0]}")
    table = {
        "b_num_partitions": b_num_partitions,
        "b_max_bucket_width": b_max_bucket_width
    }
    save_to_csv(table=table,
                basename=basename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parse searching configurations")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    linear_regression(feature_file)