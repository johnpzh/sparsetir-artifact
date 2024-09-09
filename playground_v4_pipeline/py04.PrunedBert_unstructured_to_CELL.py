import tvm
import os
import sys
import argparse
import torch as th
import math
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
from utils import get_dataset
import numpy as np
from scipy import sparse as sp
from typing import Any
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
from sparsetir_artifact import profile_pytorch_ms, profile_tvm_ms
import dgl
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from joblib import load


# NAMES = ['cora', 'citeseer', 'pubmed', 'ppi', 'arxiv', 'proteins', 'reddit']
# NAMES = ['arxiv', 'proteins', 'reddit']
NAMES = ['proteins', 'reddit']


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
    print(f"name: {name} format: {format} time(s): {time_s}")

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
    print(f"name: {name} num_parts: {num_parts} time(s): {time_s}")

    return num_parts, time_s



if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    parser.add_argument("--model-format-selection", "-s", type=str, help="model for format selection")
    parser.add_argument("--model-num-partitions", "-p", type=str, help="model for format selection")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    name = args.dataset


    model_selection_name = args.model_format_selection
    model_num_partitions = args.model_num_partitions

    # Load models
    clf_selection = load(model_selection_name)
    clf_partition = load(model_num_partitions)

    model = AutoModelForQuestionAnswering.from_pretrained(
        "huggingface/prunebert-base-uncased-6-finepruned-w-distil-squad")

    names_list = []
    density_list = []
    formats_list = []
    num_features_list = []
    num_parts_list = []
    time_selet_list = []
    time_num_parts_list = []
    time_bucket_list = []
    time_exe_list = []

    for name, param in model.named_parameters():
        if (
            name.endswith("key.weight")
            or name.endswith("value.weight")
            or name.endswith("query.weight")
            or name.endswith("dense.weight")
        ):
            print("")
            print(f"Go to name {name} ...")
            # Load the matrix
            # csr_weight = sp.csr_matrix(param.detach().numpy())
            coo_weight = sp.coo_matrix(param.detach().numpy())
            density = coo_weight.nnz / param.numel()
            g = MTX(None)
            g.name = name
            num_rows, num_cols = coo_weight.shape
            print(f"origin_shape: ({num_rows}, {num_cols})")
            num_rows = max(num_rows, num_cols)
            num_cols = num_rows
            print(f"sqaured_shape: ({num_rows}, {num_cols})")
            # g.coo_mtx = csr_weight.tocoo()
            g.coo_mtx = sp.coo_matrix((coo_weight.data, (coo_weight.row, coo_weight.col)), shape=(num_rows, num_cols))

            # Predict format selection
            format, time_selet = predict_format_selection(g, clf=clf_selection)

            if format == "CELL":
                num_features = 512
                num_parts, time_parts = predict_num_partitions(g, clf=clf_partition, num_features=num_features)

                # Build buckets
                time_bucket_start = time.perf_counter()
                cost_model_config = CostModelSettings(feat_size=num_features, num_parts=num_parts)
                bucket_config = search_bucket_config(g, num_parts, cost_model_config)
                time_bucket_end = time.perf_counter()
                time_bucket = time_bucket_end - time_bucket_start

                # Run kernel
                features = g.matrix_features()
                features["K"] = num_features
                x = th.ones((g.num_dst_nodes(), num_features))
                # test
                print(f"mtx: ({g.num_src_nodes()}, {g.num_dst_nodes()})")
                print(f"x.shape: ({x.shape})")
                # end test
                y_ndarray = g.dot(x.numpy())
                hyb_format = build_hyb_format(g, bucket_config)
                print(f"name: {name} feat_size: {num_features} num_partitions: {num_parts} cost_model_config: {cost_model_config} bucket_config: {bucket_config}", flush=True)
                try:
                    exe_time = bench_hyb_with_config(g,
                                            x,
                                            # y_golden,
                                            y_ndarray,
                                            feat_size=num_features,
                                            bucket_widths=bucket_config,
                                            bucketing_format=hyb_format,
                                            coarsening_factor=2,
                                            num_col_parts=num_parts,
                                            use_implicit_unroll=True,
                                            check_correctness=False)
                    exe_time = float(f"{exe_time}")
                    print(f"name: {name} density: {density} exe_time: {exe_time}")
                except Exception as e:
                    exe_time = math.inf
                    print(e, file=sys.stderr)
                    format = "CRASHED"
            # # x = th.rand(csr_weight.shape[1], args.dim).half()
            # x = th.rand((g.num_dst_nodes(), num_features))
            # # test
            # print(f"mtx: ({mtx.num_src_nodes()}, {mtx.num_dst_nodes()})")
            # print(f"x.shape: ({x.shape})")
            # # end test
            # # y_golden = dgl.ops.copy_u_sum(g, x)
            # y_ndarray = mtx.dot(x.numpy())
            # exe_time = bench_naive(mtx,
            #                     x,
            #                     y_ndarray,
            #                     feat_size=num_features,
            #                     coarsening_factor=2)
            # print(f"name: {name} exe_time: {exe_time}")
            names_list.append(name)
            density_list.append(density)
            formats_list.append(format)
            num_features_list.append(num_features)
            num_parts_list.append(num_parts)
            time_selet_list.append(time_selet)
            time_num_parts_list.append(time_parts)
            time_bucket_list.append(time_bucket)
            time_exe_list.append(exe_time)

                
    table = {
        "name": names_list,
        "density": density_list,
        "format": formats_list,
        "feat_size": num_features_list,
        "num_parts": num_parts_list,
        "time_select(s)": time_selet_list,
        "time_num_parts(s)": time_num_parts_list,
        "time_bucket(s)": time_bucket_list,
        "time_exe_our(s)": time_exe_list,
    }
    df = pd.DataFrame(data=table)
    print(df)
    df.to_csv(f"output.{sys.argv[0]}.csv", index=False)
    # mtx_bsr = mtx.tobsr(blocksize=(32, 32), copy=True)
    # print(f"mtx_bsr: {mtx_bsr}")

    # mtx_csr = mtx.tocsr(copy=True)
    # print(f"mtx_csr: {mtx_csr}")



    # g = get_dataset(name)

    # print(g)
    # print(g.adj_tensors(fmt="coo"))
    # src, dst = g.adj_tensors(fmt="coo")
    # print(f"src: {src.numpy()} dst: {dst.numpy()}")
    # mtx = MTX(name)
    # print(f"mtx: {mtx}")
    # print(f"num_src_nodes: {mtx.num_src_nodes()} num_dst_nodes: {mtx.num_dst_nodes()} num_edges: {mtx.num_edges()}")
    # filename = args.dataset
    # g = MTX(filename)
    # num_parts = args.num_partitions
    # feat_size = args.feat_size
    # name = g.name