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
from typing import Any
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
from sparsetir_artifact import profile_pytorch_ms, profile_tvm_ms
import dgl

# NAMES = ['cora', 'citeseer', 'pubmed', 'ppi', 'arxiv', 'proteins', 'reddit']
# NAMES = ['arxiv', 'proteins', 'reddit']
NAMES = ['proteins', 'reddit']


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    name = args.dataset

    names_list = []
    formats_list = []
    num_features_list = []
    blocksizes_list = []
    time_exe_list = []

    for name in NAMES:
        print("")
        print(f"Go to matrix {name} ... ")
        feat_size = 32
        mtx = MTX(name)
        num_src = mtx.num_src_nodes()
        num_dst = mtx.num_dst_nodes()
        assert num_src == num_dst, f"Error: matrix {name} is not sqaure ({num_src} x {num_dst})."

        bsize = 32
        # Padding the size
        print(f"original_size: ({num_src}, {num_dst})")
        num_block_row = (num_src + bsize - 1) // bsize
        num_src = num_block_row * bsize
        num_dst = num_src

        names_list.append(name)
        num_features_list.append(feat_size)
        print(f"padded_size: ({num_src}, {num_dst}) bsize: {bsize}")
        try:
            if name not in ['proteins', 'reddit']:
            # if True:
                bsr_weight = mtx.tobsr_with_padding(shape=(num_src, num_dst), blocksize=(bsize, bsize))
                # print(f"mtx: {mtx.coo_mtx.toarray()}")
                # print(f"bsr_weight: {bsr_weight.toarray()}")
                print(f"mtx.shape: ({mtx.num_dst_nodes(), mtx.num_src_nodes()}) nnz: {mtx.num_edges()}")
                print(f"bsr_weight.shape: ({bsr_weight.shape[0], bsr_weight.shape[1]}) nnz: {bsr_weight.nnz} size: {bsr_weight.size}")

                x = th.rand(bsr_weight.shape[1], feat_size).half()
                # print(f"mtx_bsr: {mtx_bsr}")
                # csr_weight = mtx.tocsr()
                exe_time = bench_bsrmm(bsr_weight, x, block_size=bsize)
                print(f"name: {name} exe_time: {exe_time}")
                # print(f"mtx_csr: {mtx_csr}")
                formats_list.append("BSR")
                blocksizes_list.append(bsize)
                time_exe_list.append(exe_time)
            else:
                raise Exception(f"name {name} cannot do BSR (scipy.sparse.bsr_matrix() will cause Segmentation Fault).")
        except Exception as e:
            print(e, file=sys.stderr)
            print(f"name {name} in BSR is out-of-memory. Do CSR then.")
            try:
                # BSR is too large to be handled.
                # Try CSR instead.
                x = th.rand((mtx.num_src_nodes(), feat_size))
                # y_golden = dgl.ops.copy_u_sum(mtx, x)
                y_ndarray = mtx.dot(x.numpy())
                exe_time = bench_naive(mtx,
                                       x,
                                       y_ndarray,
                                       feat_size=feat_size,
                                       coarsening_factor=2)
                print(f"name: {name} exe_time: {exe_time}")
                formats_list.append("CSR")
                blocksizes_list.append(0)
                time_exe_list.append(exe_time)
            except Exception as e2:
                print(e2, file=sys.stderr)
                print(f"name: {name} in CSR is also out-of-memory. Exit.")
                formats_list.append("Crashed")
                blocksizes_list.append(0)
                time_exe_list.append(0)

    table = {
        "name": NAMES,
        "feat_size": num_features_list,
        "format": formats_list,
        "blocksize": blocksizes_list,
        "time_exe(s)": time_exe_list,
    }
    df = pd.DataFrame(data=table)
    print(df)
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