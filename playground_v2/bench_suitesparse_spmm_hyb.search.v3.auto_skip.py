# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import dgl
import tvm
import os
import sys
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
import tvm.sparse
# from utils import get_dataset
from sparsetir_artifact import profile_tvm_ms
import math
import pandas as pd
from matrix_market import MTX
import time


@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    coarsening_factor: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    # I, J, J_detach, K1, K2, K3 are Axes.
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(coarsening_factor)
    K3 = T.dense_fixed(32)
    # A, B, C are Sparse Buffers.
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    # Sparse Iterations.
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


# cached_bucketing_format = None
OUTPUT_DIR="output"


def bench_hyb(
    g,
    x,
    # y_golden,
    y_ndarray,
    feat_size=128,
    bucket_sizes=[],
    coarsening_factor=2,
    num_col_parts=1,
    use_implicit_unroll=False,
):
    num_buckets = len(bucket_sizes)
    coarsening_factor = min(coarsening_factor, feat_size // 32)
    # indptr, indices, _ = g.adj_tensors("csc")
    indptr, indices, _ = g.adj_tensors("csr")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    # Changed by Zhen Peng on 3/21/2024
    # global cached_bucketing_format
    # if cached_bucketing_format is None:
    #     indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    #     indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    #     cached_bucketing_format = column_part_hyb(
    #         m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
    #     )

    #
    indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
    indices_nd = tvm.nd.array(indices, device=tvm.cpu())
    # indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    # indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    cached_bucketing_format = column_part_hyb(
        m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
    )
    row_indices, col_indices, mask = cached_bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            rewrites.append(
                FormatRewriteRule(
                    str(part_id) + "_" + str(bucket_id),
                    ell.specialize({nnz_cols_symbol: bucket_size}),
                    ["A"],
                    ["I", "J"],
                    ["O", "I", "J"],
                    {"I": ["O", "I"], "J": ["J"]},
                    csr2ell_index_map,
                    csr2ell_inv_index_map,
                )
            )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coarsening_factor,  # coersening_factor
    }
    for part_id in range(num_col_parts):
        for bucket_id in range(num_buckets):
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
                part_id
            ][bucket_id].shape[0]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in [
        "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    ]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            sch.reorder(foo, fi, j, foi)
            if is_atomic:
                sch.annotate(blk, "atomic", True)
                write_blk = sch.cache_write(blk, 0, "local")
                sch.reverse_compute_at(write_blk, fi, True)
                # sch.unroll(sch.get_loops(write_blk)[-2])
            sch.bind(fi, "threadIdx.x")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            if use_implicit_unroll:
                sch.annotate(foi, "pragma_unroll_explicit", 0)
            sch.unroll(j)
            if use_implicit_unroll:
                sch.annotate(j, "pragma_unroll_explicit", 0)
            io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            init_blk = sch.decompose_reduction(blk, fi)
            ax0, ax1 = sch.get_loops(init_blk)[-2:]
            sch.bind(ax0, "threadIdx.x")
            sch.unroll(ax1)
            if use_implicit_unroll:
                sch.annotate(ax1, "pragma_unroll_explicit", 0)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    # prepare args
    args = [b_nd, c_nd]

    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            weight = tvm.nd.array(
                mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]

    # test accuracy
    # f(*args)
    # Turned off correctness check
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    print("tir hyb time: {:.6f} ms".format(dur))

    return dur


def get_bucket_config(max_bucket_width: int):
    power = math.ceil(math.log2(max_bucket_width))
    bucket_config = [int(2**i) for i in range(power + 1)]
    return bucket_config


def save_statistics(features: dict,
                    execution_times: list,
                    partitions: list,
                    max_bucket_sizes: list):
    name = features["name"]
    feat_size = features["K"]
    # columns = {
    #     "num_partitions": PARTITIONS,
    # }
    # min_exe_time = execution_times[0][0]
    # best_width_ind = 0
    # best_part_ind = 0
    # for i, max_bucket_width in enumerate(MAX_BUCKET_SIZES):
    #     columns[F"max_bucket_width:{max_bucket_width}"] = execution_times[i]
    #     lowest = min(execution_times[i])
    #     if lowest < min_exe_time:
    #         min_exe_time = lowest
    #         best_width_ind = i
    #         best_part_ind = execution_times[i].index(lowest)

    # # Print statistics first
    # features["best_num_partitions"] = [PARTITIONS[best_part_ind]]
    # features["best_max_bucket_width"] = [MAX_BUCKET_SIZES[best_width_ind]]
    # features["best_exe_time"] = min_exe_time

    # Pandas settings
    pd.set_option("display.width", 800)
    pd.set_option("display.max_columns", None)

    dataFrame = pd.DataFrame(data=features)
    dataFrame.set_index("name", inplace=True)
    log = OUTPUT_DIR
    if not os.path.exists(F"{log}"):
        os.mkdir(F"{log}")
    log = os.path.join(log, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    print(dataFrame)
    dataFrame.to_csv(log)

    # Put prints under output/ directory
    columns = {
        "num_partitions": partitions,
        "max_bucket_size": max_bucket_sizes,
        "exe_time": execution_times
    }
    dataFrame = pd.DataFrame(data=columns)
    dataFrame.set_index("num_partitions", inplace=True)
    print(dataFrame)
    dataFrame.to_csv(log, mode='a')

    print(F"#### Saved to {log} .")



# col_part_config = {
#     "cora": 1,
#     "citeseer": 1,
#     "pubmed": 1,
#     "ppi": 16,
#     "arxiv": 1,
#     "proteins": 8,
#     "reddit": 8,
#     "products": 16,
# }

# bucketing_config = {
#     "cora": [1, 2, 4],
#     "citeseer": [1, 2, 4],
#     "pubmed": [1, 2, 4, 8], # changed, better
#     "ppi": [1, 2, 4, 8, 16, 32, 64], # changed
#     "arxiv": [1, 2, 4, 8, 16], # changed
#     "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], # changed
#     "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], # changed
#     "products": [1, 2, 4, 8, 16, 32, 64], # not used
# }
# # bucketing_config = {
# #     "cora": [1, 2, 4],
# #     "citeseer": [1, 2, 4],
# #     "pubmed": [1, 2, 4, 8, 16, 32],
# #     "ppi": [1, 2, 4, 8, 16, 32],
# #     "arxiv": [1, 2, 4, 8, 16, 32],
# #     "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
# #     "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
# #     "products": [1, 2, 4, 8, 16, 32],
# # }

# PARTITIONS_SET = {
#     "cora": [1, 2, 4, 8],
#     "citeseer": [1, 2, 4, 8],
#     "pubmed": [1, 2, 4, 8],
#     "ppi": [1, 2, 4, 8, 16, 32, 64],
#     "arxiv": [1, 2, 4, 8],
#     "proteins": [1, 2, 4, 8, 16, 32, 64, 128],
#     "reddit": [1, 2, 4, 8, 16, 32, 64, 128],
# }

# MAX_BUCKET_SIZES_SET = {
#     "cora": [1, 2, 4, 8, 16],
#     "citeseer": [1, 2, 4, 8, 16],
#     "pubmed": [4, 8, 16, 32, 64, 128, 256],
#     "ppi": [4, 8, 16, 32, 64, 128],
#     "arxiv": [4, 8, 16, 32, 64, 128],
#     "proteins": [32, 64, 128, 256, 512],
#     "reddit": [16, 32, 64, 128, 256, 512, 1024],
# }


class Config:
    """Config class to store hyb configurations 1) the number of column partitions, and 2) the maximum bucket width. It also store the execution time of the configuration.
    """
    def __init__(self):
        self.num_parts = 0
        self.max_bucket_width = 0
        self.exe_time = None
    def __init__(self, num_parts, max_bucket_width):
        self.num_parts = num_parts
        self.max_bucket_width = max_bucket_width
        self.exe_time = None
    def __eq__(self, other): 
        return self.num_parts == other.num_parts and \
                self.max_bucket_width == other.max_bucket_width
    def __str__(self):
        return F"(num_parts: {self.num_parts} max_bucket_width: {self.max_bucket_width} exe_time: {self.exe_time})"

DIRECTIONS = [
    # num_parts, max_bucket_width
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1]
]


def next_step_config(config: Config,
                     step: list):
    """Get a new config according to the step.

    Args:
        config (Config): current config
        step (list): one element in DIRECTIONS

    Returns:
        Config: a new config after proceeding the step
    """
    # Get old config
    pow_num_parts = math.log2(config.num_parts)
    pow_max_bucket_width = math.log2(config.max_bucket_width)
    # Proceed the step
    new_pow_num_parts = pow_num_parts + step[0]
    new_pow_max_bucket_width = pow_max_bucket_width + step[1]
    # Get new config
    new_num_parts = int(2 ** new_pow_num_parts)
    new_max_bucket_width = int(2 ** new_pow_max_bucket_width)

    return Config(new_num_parts, new_max_bucket_width)


def start_max_bucket_width(avg_nnz_per_row: float):
    # pow_floor = math.floor(math.log2(avg_nnz_per_row))
    # val_floor = int(2 ** pow_floor)
    pow_ceil = math.ceil(math.log2(avg_nnz_per_row))
    width = int(2 ** pow_ceil / 32)
    if width < 1:
        width = 1
    
    return width


def check_if_skip(overhead: float,
                  name: str):
    # if overhead >= 0.001:
    if overhead >= 3600:
        # longer than 1 hour
        output_dir = OUTPUT_DIR
        error_file = os.path.join(output_dir, F"skipped_{name}.csv")
        df = pd.DataFrame(data=features, index=[0])
        df.to_csv(error_file, index=False)
        print(F"Overhead is {overhead} (s). Matrix {name} is skipped. Its features are written to {error_file} .")
        print(df)
        return True
    
    return False


def search_best_config(features,
                       g,
                       x,
                       y_ndarray,
                       feat_size,
                       coarsening_factor,
                       use_implicit_unroll,
                       partitions: list, # output
                       max_bucket_sizes: list, # output
                       execution_times: list): # output
    name = features["name"]
    num_parts = 1
    max_bucket_width = start_max_bucket_width(features["avg_nnz_per_row"])
    curr = Config(num_parts, max_bucket_width)
    bucket_sizes = get_bucket_config(max_bucket_width)
    start_time = time.perf_counter()
    try:
        print(F"#### data: {name} feat_size: {feat_size} num_partitions: {num_parts} max_bucket_width: {max_bucket_width} bucket_config: {bucket_sizes}", flush=True)
        exe_time = bench_hyb(g,
                             x,
                             y_ndarray,
                             feat_size=feat_size,
                             bucket_sizes=bucket_sizes,
                             coarsening_factor=coarsening_factor,
                             num_col_parts=num_parts,
                             use_implicit_unroll=use_implicit_unroll)
    except Exception as e:
        exe_time = math.inf
        print(e, file=sys.stderr)
    exe_time = float("{:.6}".format(exe_time))
    curr.exe_time = exe_time
    end_time = time.perf_counter()
    # check if skip this input matrix
    if check_if_skip(overhead=(end_time - start_time),
                     name=name):
        sys.exit(-1)

    # # test
    # print(F"453 curr: {curr}")
    # # end test
    partitions.append(num_parts)
    max_bucket_sizes.append(max_bucket_width)
    execution_times.append(exe_time)
    is_visited = [curr]
    min_config = curr
    queue = [curr]

    while queue:
        curr = queue.pop()
        # # test
        # print(F"465 curr: {curr}")
        # # end test
        for step in DIRECTIONS:
            adj_config = next_step_config(curr, step) # adjacent config from current config
            # # test
            # print(F"470 adj_config: {adj_config}")
            # # end test
            if adj_config.num_parts < 1 or adj_config.max_bucket_width < 1:
                # Check if not valid
                continue
            if adj_config in is_visited:
                # Check if visited
                continue
            is_visited.append(adj_config) # mark as visited
            bucket_sizes = get_bucket_config(adj_config.max_bucket_width)
            # Measure execution time
            start_time = time.perf_counter()
            try:
                print(F"#### data: {name} feat_size: {feat_size} num_partitions: {adj_config.num_parts} max_bucket_width: {adj_config.max_bucket_width} bucket_config: {bucket_sizes}", flush=True)
                exe_time = bench_hyb(g,
                                     x,
                                     y_ndarray,
                                     feat_size=feat_size,
                                     bucket_sizes=bucket_sizes,
                                     coarsening_factor=coarsening_factor,
                                     num_col_parts=adj_config.num_parts,
                                     use_implicit_unroll=use_implicit_unroll)
                exe_time = float("{:.6}".format(exe_time))
            except Exception as e:
                exe_time = math.inf
                print(e, file=sys.stderr)
            partitions.append(adj_config.num_parts)
            max_bucket_sizes.append(adj_config.max_bucket_width)
            execution_times.append(exe_time)
            exe_time = float("{:.6}".format(exe_time))
            # check if skip this input matrix
            if check_if_skip(overhead=(end_time - start_time),
                            name=name):
                sys.exit(-1)
            if exe_time == math.inf or exe_time > curr.exe_time:
                # Pruning adjacent config if its execution time is longer
                continue
            adj_config.exe_time = exe_time
            # # test
            # print(F"exe_time: {exe_time}")
            # # end test
            queue.append(adj_config)
            if adj_config.exe_time < min_config.exe_time:
                # Record the best config so far with the shortest execution time
                min_config = adj_config
    
    return min_config


def check_if_done_before(name: str,
                         feat_size: int):
    output_dir = OUTPUT_DIR
    filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    if os.path.isfile(filename):
        print(F"{filename} already exits. Skipped it.")
        return True
    else:
        return False


def check_if_too_risky(g: MTX):
    output_dir = OUTPUT_DIR
    name = g.name
    num_rows = g.num_src_nodes()
    num_cols = g.num_dst_nodes()
    num_edges = g.num_edges()
    skipped_filename = os.path.join(output_dir, F"skipped_{name}.csv")
    if num_rows != num_cols:
        print(F"{name} has num_rows={num_rows} num_cols={num_cols}, not equal. Skip it.")
        return True
    elif num_rows >= 3997962: # com-LiveJournal
        print(F"{name} has num_nodes={num_rows}, too large. Skip it.")
        return True
    elif num_edges >= 329499284: # Queen_4147
        print(F"{name} has num_edges={num_edges}, too large. Skip it.")
        return True
    elif os.path.isfile(skipped_filename):
        print(F"{skipped_filename} exits, meaning {name} should be skipped. Skipped it.")
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", default=True, help="use implicit unroll")
    # parser.add_argument("--gpu", "-g", type=int, default=0, help="select the GPU device by index")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    # name = args.dataset
    # g = get_dataset(name)
    filename = args.dataset
    g = MTX(filename)

    if check_if_too_risky(g):
        sys.exit(-1)
    # # Feasibility check
    # if g.num_dst_nodes() >= 5558326 or g.num_edges() >= 59524291:
    #     print(F"\nMatrix {filename} is too large to be handled. num_cols: {g.num_dst_nodes}. nnz: {g.num_edges}. Passed.")
    #     sys.exit(-1)

    # features = g.matrix_features()
    # # test
    # print(F"features: {features}")
    # exit(-1)
    # # end test
    # GPU_DEVICE = args.gpu
    # print(F"GPU_DEVICE: {GPU_DEVICE}")
    print(F"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', default=0)}")


    # Tuning
    # PARTITIONS          = [1, 2]
    # MAX_BUCKET_SIZES    = [1, 2]
    # PARTITIONS          = PARTITIONS_SET[name]
    # MAX_BUCKET_SIZES    = MAX_BUCKET_SIZES_SET[name]
    # for feat_size in [32]:
    for feat_size in [32, 64, 128, 256, 512]:
        features = g.matrix_features()
        features["K"] = feat_size

        # If done before, skipped
        if check_if_done_before(features["name"], feat_size):
            continue

        x = th.rand((g.num_dst_nodes(), feat_size))
        # # test
        # # x = th.zeros((g.num_dst_nodes(), feat_size))
        # # for i in range(min(g.num_dst_nodes(), feat_size)):
        # #     x[i][i] = 1
        # x = th.ones((g.num_dst_nodes(), feat_size))
        # # end test
        # y_golden = dgl.ops.copy_u_sum(g, x)
        # y_ndarray = g.dot(x.numpy())
        y_ndarray = []
        # # test
        # print(F"y_ndarray: {y_ndarray}")
        # # end test

        execution_times = []
        partitions = []
        max_bucket_sizes = []
        start_time = time.perf_counter()
        best_config = search_best_config(features,
                                         g,
                                         x,
                                         y_ndarray,
                                         feat_size=feat_size,
                                         coarsening_factor=2,
                                         use_implicit_unroll=args.implicit_unroll,
                                         partitions=partitions,
                                         max_bucket_sizes=max_bucket_sizes,
                                         execution_times=execution_times)
        end_time = time.perf_counter()
        features["best_num_partitions"] = best_config.num_parts
        features["best_max_bucket_width"] = best_config.max_bucket_width
        features["autotuning_time(s)"] = end_time - start_time
        features["best_exe_time"] = [best_config.exe_time]
        
        # Statistics
        save_statistics(features,
                        execution_times,
                        partitions,
                        max_bucket_sizes)
