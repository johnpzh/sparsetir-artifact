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
    CostModelSettings
)
from utils import get_dataset

NAMES = ['cora', 'citeseer', 'pubmed', 'ppi',  'arxiv', 'proteins', 'reddit']

if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)

    print(g)
    print(g.adj_tensors(fmt="coo"))
    src, dst = g.adj_tensors(fmt="coo")
    print(f"src: {src.numpy()} dst: {dst.numpy()}")
    mtx = MTX(name)
    print(f"mtx: {mtx}")
    print(f"num_src_nodes: {mtx.num_src_nodes()} num_dst_nodes: {mtx.num_dst_nodes()} num_edges: {mtx.num_edges()}")
    # filename = args.dataset
    # g = MTX(filename)
    # num_parts = args.num_partitions
    # feat_size = args.feat_size
    # name = g.name