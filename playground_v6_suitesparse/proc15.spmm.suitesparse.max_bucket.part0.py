# import tvm
import os
import sys
import argparse
import torch as th
import math
import numpy as np
import pandas as pd
import time
from format_matrix_market import MTX
from format_algos import (
    build_hyb_format,
    bench_hyb_with_config,
    search_bucket_config,
    CostModelSettings,
    bench_bsrmm,
    bench_naive,
)
from joblib import load


# NAMES = ['cora', 'citeseer', 'pubmed', 'ppi',  'arxiv', 'proteins', 'reddit']


def get_X_for_format_selection(g: MTX):
    mtx_features = g.matrix_features()
    features = [
        "num_rows",
        "num_cols",
        "nnz",
        "avg_nnz_per_row",
        "min_nnz_per_row",
        "max_nnz_per_row",
        "std_dev_nnz_per_row",
    ]
    # # test
    # print(f"mtx_features: {mtx_features}")
    # # end test
    # values = [mtx_features[f] for f in features]
    # # test
    # print(f"values: {values}")
    # # end test
    X = [np.array([mtx_features[f] for f in features])]

    return X


def get_X_for_num_partitions(g: MTX, num_features: int):
    mtx_features = g.matrix_features()
    features = [
        "num_rows", 
        "num_cols",
        "nnz",
        "avg_nnz_density_per_row",
        "min_nnz_density_per_row",
        "max_nnz_density_per_row",
        "std_dev_nnz_density_per_row",
    ]
    X = [np.array([mtx_features[f] for f in features] + [num_features])]

    return X


def predict_format_selection(g: MTX,
                             clf):
    # Get the matrix characteristics
    X_selection = get_X_for_format_selection(g)

    # Predict
    time_s_start = time.perf_counter()
    y_pred = clf.predict(X_selection)
    time_s_end = time.perf_counter()
    time_s = time_s_end - time_s_start

    format = "CELL" if y_pred[0] == 1 else "BCSR"
    print(f"name: {g.name} format: {format} time(s): {time_s}")

    return format, time_s


def predict_num_partitions(g: MTX,
                           clf: str,
                           num_features: int):
    # Get the matrix characteristics
    X_selection = get_X_for_num_partitions(g, num_features)

    # Predict
    time_s_start = time.perf_counter()
    y_pred = clf.predict(X_selection)
    time_s_end = time.perf_counter()
    time_s = time_s_end - time_s_start

    num_parts = y_pred[0]
    print(f"name: {g.name} num_parts: {num_parts} time(s): {time_s}")

    return num_parts, time_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser("predict the format selection")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--csv", "-c", type=str, help="CSV file of SuiteSparse matrices")
    parser.add_argument("--model-format-selection", "-s", type=str, help="model for format selection")
    parser.add_argument("--model-num-partitions", "-p", type=str, help="model for format selection")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    # name = args.dataset
    model_selection_name = args.model_format_selection
    model_num_partitions = args.model_num_partitions
    csv_filename = args.csv

    df_ref = pd.read_csv(csv_filename)

    SCRATCH_DIR = "/raid/peng599/scratch/spmm/data/suitesparse"
    NAME_FILE = "/home/peng599/pppp/vscode/sparsetir-artifact_mac/playground_v6_suitesparse/data/output.names.part0.txt"
    names_part = []
    with open(NAME_FILE) as fin:
        for line in fin:
            names_part.append(line.strip())


    # Load data
    clf_selection = load(model_selection_name)
    clf_partition = load(model_num_partitions)
    # df = pd.read_csv(ss_filename)

    # names_list = []
    # formats_list = []
    # num_features_list = []
    # num_parts_list = []
    # time_selet_list = []
    # time_num_parts_list = []
    # time_bucket_list = []
    # time_exe_list = []

    # for name in NAMES:
    # for name in df["name"]:
    output_filename = f"output.{sys.argv[0]}.csv"
    if not os.path.isfile(output_filename):
        with open(output_filename, "w") as fout:
            fout.write("name,density,format,feat_size,num_parts,autotune_max_bucket,cost_model_max_bucket\n")

    for name in names_part:
    # for name in ["Olivetti_norm_10NN"]:
        # Load the data
        # g = MTX(name)
        filename = os.path.join(SCRATCH_DIR, f"{name}/{name}.mtx")
        print("")
        print(f"Go to name {filename} ...")
        g = MTX(filename)

        # Predict format selection
        format, time_selet = predict_format_selection(g, clf=clf_selection)
        # formats_list.append(format)
        # time_selet_list.append(time_selet)

        # Retrieve the density
        num_rows = df_ref.loc[(df_ref['name'] == name) & (df_ref['K'] == 32), 'num_rows'].values[0]
        num_cols = df_ref.loc[(df_ref['name'] == name) & (df_ref['K'] == 32), 'num_cols'].values[0]
        nnz = df_ref.loc[(df_ref['name'] == name) & (df_ref['K'] == 32), 'nnz'].values[0]
        density = nnz / (num_rows * num_cols)

        if format == "CELL":
        # if True:
            # Use CELL format
            # Predict number of partitions
            # num_features = 32
            for num_features in [32, 64, 128, 256, 512]:

                # Retrieve the autotuning max bucket width
                autotune_max_bucket = df_ref.loc[(df_ref['name'] == name) & (df_ref['K'] == num_features), 'best_max_bucket_width'].values[0]

                num_parts, time_parts = predict_num_partitions(g, clf=clf_partition, num_features=num_features)

                # Build buckets
                time_bucket_start = time.perf_counter()
                cost_model_config = CostModelSettings(feat_size=num_features, num_parts=num_parts)
                bucket_config = search_bucket_config(g, num_parts, cost_model_config)
                time_bucket_end = time.perf_counter()
                time_bucket = time_bucket_end - time_bucket_start

                # Get the max bucket width
                max_buckets = []
                for buckets in bucket_config:
                    if buckets:
                        max_buckets.append(max(buckets))
                max_buckets_avg = np.mean(max_buckets)

                # Save
                with open(output_filename, "a") as fout:
                    fout.write(f"{name},{density},{format},{num_features},{num_parts},{autotune_max_bucket},{max_buckets_avg}\n")
