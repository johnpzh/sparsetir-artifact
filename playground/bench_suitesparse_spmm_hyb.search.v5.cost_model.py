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

# import dgl
import tvm
import os
import sys
import tvm.testing
# import tvm.tir as tir
# import scipy.sparse as sp
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
import time
# from matrix_market import MTX
# from matrix_market import Bucket
# from algos import build_hyb_format
from format_matrix_market import MTX
from format_algos import (
    build_hyb_format,
    bench_hyb_with_config,
    search_bucket_config
)

# @T.prim_func
# def ell(
#     a: T.handle,
#     indptr_i: T.handle,
#     indices_i: T.handle,
#     indices_j: T.handle,
#     m: T.int32,
#     n: T.int32,
#     num_rows: T.int32,
#     nnz_cols: T.int32,
# ) -> None:
#     O = T.dense_fixed(1)
#     I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
#     J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
#     A = T.match_sparse_buffer(a, (O, I, J), "float32")
#     T.evaluate(0)


# @T.prim_func
# def csrmm(
#     a: T.handle,
#     b: T.handle,
#     c: T.handle,
#     indptr: T.handle,
#     indices: T.handle,
#     m: T.int32,
#     n: T.int32,
#     num_tiles: T.int32,
#     nnz: T.int32,
#     coarsening_factor: T.int32,
# ) -> None:
#     T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
#     # I, J, J_detach, K1, K2, K3 are Axes.
#     I = T.dense_fixed(m)
#     J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
#     J_detach = T.dense_fixed(n)
#     K1 = T.dense_fixed(num_tiles)
#     K2 = T.dense_fixed(coarsening_factor)
#     K3 = T.dense_fixed(32)
#     # A, B, C are Sparse Buffers.
#     A = T.match_sparse_buffer(a, (I, J), "float32")
#     B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
#     C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
#     # Sparse Iterations.
#     with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
#         with T.init():
#             C[i, k1, k2, k3] = 0.0
#         C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


# def csr2ell_inv_index_map(o, i, j):
#     return i, j


# def csr2ell_index_map(i, j):
#     return 0, i, j


# cached_bucketing_format = None
OUTPUT_DIR="output.cost_model"


# def bench_hyb_given_buckets(g,
#                             x,
#                             # y_golden,
#                             y_ndarray,
#                             feat_size=128,
#                             bucket_widths=[],
#                             bucketing_format=[],
#                             coarsening_factor=2,
#                             num_col_parts=1,
#                             use_implicit_unroll=False):
#     # num_buckets = len(bucket_sizes)
#     coarsening_factor = min(coarsening_factor, feat_size // 32)
#     # indptr, indices, _ = g.adj_tensors("csc")
#     m = g.num_dst_nodes()
#     n = g.num_src_nodes()
#     nnz = g.num_edges()
#     # Changed by Zhen Peng on 3/21/2024
#     # global cached_bucketing_format
#     # if cached_bucketing_format is None:
#     #     indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
#     #     indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
#     #     cached_bucketing_format = column_part_hyb(
#     #         m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
#     #     )

#     #
#     # indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
#     # indices_nd = tvm.nd.array(indices, device=tvm.cpu())
#     # indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
#     # indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
#     # cached_bucketing_format = column_part_hyb(
#     #     m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
#     # )
#     cached_bucketing_format = bucketing_format
#     row_indices, col_indices, mask = cached_bucketing_format

#     # rewrite csrmm
#     nnz_cols_symbol = ell.params[-1]
#     rewrites = []
#     for part_id in range(num_col_parts):
#         b_widths = bucket_widths[part_id]
#         for bucket_id, bucket_size in enumerate(b_widths):
#             rewrites.append(
#                 FormatRewriteRule(
#                     str(part_id) + "_" + str(bucket_id),
#                     ell.specialize({nnz_cols_symbol: bucket_size}),
#                     ["A"],
#                     ["I", "J"],
#                     ["O", "I", "J"],
#                     {"I": ["O", "I"], "J": ["J"]},
#                     csr2ell_index_map,
#                     csr2ell_inv_index_map,
#                 )
#             )
#     mod = tvm.IRModule.from_expr(csrmm)
#     mod = format_decompose(mod, rewrites)
#     mod = tvm.tir.transform.RemovePreprocess()(mod)

#     # specialize
#     params = mod["main"].params
#     # # test
#     # print(F"params: {params} len(params): {len(params)}")
#     # # end test
#     param_map = {
#         params[5]: m,  # m
#         params[6]: n,  # n
#         params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
#         params[8]: nnz,  # nnz
#         params[9]: coarsening_factor,  # coersening_factor
#     }
    
#     loc_base = 14
#     loc_step = 7
#     loc = loc_base
#     for part_id in range(num_col_parts):
#         b_widths = bucket_widths[part_id]
#         num_buckets = len(b_widths)
#         # # test
#         # print(F"part_id: {part_id} num_buckets: {num_buckets}")
#         # # end test
#         for bucket_id in range(num_buckets):
#             # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
#             # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
#             # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
#             #     part_id
#             # ][bucket_id].shape[0]

#             param_map[params[loc]] = m
#             param_map[params[loc + 1]] = n
#             param_map[params[loc + 2]] = row_indices[
#                 part_id
#             ][bucket_id].shape[0]
#             # # test
#             # print(F"loc = {loc}")
#             # print(F"loc + 1 = {loc + 1}")
#             # print(F"loc + 2 = {loc + 2}")
#             # # end test
#             loc += loc_step

#     mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

#     # schedule
#     iter_names = []
#     for part_id in range(num_col_parts):
#         b_widths = bucket_widths[part_id]
#         num_buckets = len(b_widths)
#         for bucket_id in range(num_buckets):
#             iter_names.append(F"csrmm_{part_id}_{bucket_id}")
    
#     # # test
#     # tmp = [
#     #     "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
#     # ]
#     # print(F"tmp: {tmp}")
#     # print(F"iter_names: {iter_names}")
#     # sys.exit(-1)
#     # # end test


#     sch = tvm.tir.Schedule(mod)
#     # for sp_iter_name in [
#     #     "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
#     # ]:
#     for sp_iter_name in iter_names:
#         sp_iteration = sch.get_sparse_iteration(sp_iter_name)
#         o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
#         sch.sparse_fuse(sp_iteration, [o, i])

#     mod = sch.mod
#     mod = tvm.sparse.lower_sparse_iter(mod)
#     sch = tvm.tir.Schedule(mod)
#     for part_id in range(num_col_parts):
#         b_widths = bucket_widths[part_id]
#         num_buckets = len(b_widths)
#         # # test
#         # print(F"part_id: {part_id} len(b_widths): {len(b_widths)}")
#         # # end test
#         for bucket_id, bucket_size in enumerate(b_widths):
#             is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
#             blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
#             i, j, foo, foi, fi = sch.get_loops(blk)
#             sch.reorder(foo, fi, j, foi)
#             if is_atomic:
#                 sch.annotate(blk, "atomic", True)
#                 write_blk = sch.cache_write(blk, 0, "local")
#                 sch.reverse_compute_at(write_blk, fi, True)
#                 # sch.unroll(sch.get_loops(write_blk)[-2])
#             sch.bind(fi, "threadIdx.x")
#             sch.bind(foo, "blockIdx.y")
#             sch.unroll(foi)
#             if use_implicit_unroll:
#                 sch.annotate(foi, "pragma_unroll_explicit", 0)
#             sch.unroll(j)
#             if use_implicit_unroll:
#                 sch.annotate(j, "pragma_unroll_explicit", 0)
#             io, ioi, ii = sch.split(i, [None, b_widths[-1] // bucket_size, 8])
#             sch.bind(io, "blockIdx.x")
#             sch.bind(ii, "threadIdx.y")
#             init_blk = sch.decompose_reduction(blk, fi)
#             ax0, ax1 = sch.get_loops(init_blk)[-2:]
#             sch.bind(ax0, "threadIdx.x")
#             sch.unroll(ax1)
#             if use_implicit_unroll:
#                 sch.annotate(ax1, "pragma_unroll_explicit", 0)

#     mod = tvm.sparse.lower_sparse_buffer(sch.mod)
#     mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
#     f = tvm.build(mod, target="cuda")

#     # prepare nd array
#     b_nd = tvm.nd.array(
#         x.numpy().reshape(-1).astype("float32"),
#         device=tvm.cuda(0),
#     )
#     c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
#     # prepare args
#     args = [b_nd, c_nd]

#     for part_id in range(num_col_parts):
#         b_widths = bucket_widths[part_id]
#         for bucket_id, _ in enumerate(b_widths):
#             weight = tvm.nd.array(
#                 mask[part_id][bucket_id].reshape(-1).astype("float32"), device=tvm.cuda(0)
#             )
#             rows = tvm.nd.array(
#                 row_indices[part_id][bucket_id].astype("int32"), device=tvm.cuda(0)
#             )
#             cols = tvm.nd.array(
#                 col_indices[part_id][bucket_id].reshape(-1).astype("int32"),
#                 device=tvm.cuda(0),
#             )
#             # weight = tvm.nd.array(
#             #     mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
#             # )
#             # rows = tvm.nd.array(
#             #     row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
#             # )
#             # cols = tvm.nd.array(
#             #     col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
#             #     device=tvm.cuda(0),
#             # )
#             args += [weight, rows, cols]

#     # test accuracy
#     f(*args)
#     # Turned off correctness check
#     tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
#     # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

#     # evaluate time
#     dur = profile_tvm_ms(f, args)
#     print("tir hyb time: {:.6f} ms".format(dur))

#     return dur


# def get_bucket_config(max_bucket_width: int):
#     power = math.ceil(math.log2(max_bucket_width))
#     bucket_config = [int(2**i) for i in range(power + 1)]
#     return bucket_config


# def save_statistics(features: dict,
#                     execution_times: list,
#                     partitions: list,
#                     max_bucket_sizes: list):
def save_statistics(features: dict):
    name = features["name"]
    feat_size = features["K"]

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

    # # Put prints under output/ directory
    # columns = {
    #     "num_partitions": partitions,
    #     "max_bucket_size": max_bucket_sizes,
    #     "exe_time": execution_times
    # }
    # dataFrame = pd.DataFrame(data=columns)
    # dataFrame.set_index("num_partitions", inplace=True)
    # print(dataFrame)
    # dataFrame.to_csv(log, mode='a')

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


def check_if_done_before(name: str,
                         feat_size: int):
    output_dir = OUTPUT_DIR
    filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    if os.path.isfile(filename):
        print(F"{filename} already exits. Skipped it.")
        return True
    else:
        return False
    

# def get_bucket_widths_list(buckets: dict):
#     bucket_pool = list(buckets.keys())
#     bucket_pool.sort()

#     return bucket_pool


# def move_largest_bucket(buckets: dict,
#                         pre_max_width: int,
#                         new_max_width: int):
#     larger_bucket = buckets[pre_max_width]
#     ratio = pre_max_width // new_max_width
#     new_num_rows = larger_bucket.num_rows * ratio
#     if new_max_width not in buckets:
#         buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz)
#     else:
#         buckets[new_max_width].num_rows += new_num_rows
#         buckets[new_max_width].nnz += larger_bucket.nnz

#     del buckets[pre_max_width]


# def get_bubble_overhead(buckets: dict):
#     """Get the bubble overhead of all buckets

#     Args:
#         buckets (dict): Dictionary of bucket width to Bucket object
#     """
#     origin_max_width = max(buckets.keys())
#     overhead = 0

#     for width, bucket in buckets.items():
#         num_rows = bucket.num_rows
#         shape_rows = origin_max_width / width
#         remaining = num_rows % shape_rows  # num_rows % shape_rows
#         # remaining = num_rows & (shape_rows - 1)  # num_rows % shape_rows
#         if remaining != 0:
#             bubble_rows = shape_rows - remaining
#         else:
#             bubble_rows = 0
#         # # test
#         # print(F"bucket: {bucket} max_width: {origin_max_width} bubble_rows: {bubble_rows}")
#         # # end test
#         # overhead += bubble_rows
#         overhead += bubble_rows * width

#     return overhead


# # def get_bubble_overhead(buckets: dict,
# #                         pre_max_width: int,
# #                         new_max_width: int):
# #     pre_overhead = curr_bubble_overhead(buckets)
# #     move_largest_bucket(buckets, pre_max_width, new_max_width)
# #     new_overhead = curr_bubble_overhead(buckets)
# #     # origin_max_width = max(buckets.keys())
# #     # overhead = 0

# #     # for width, bucket in buckets.items():
# #     #     num_rows = bucket.num_rows
# #     #     shape_rows = origin_max_width / width
# #     #     remaining = num_rows % shape_rows  # num_rows % shape_rows
# #     #     # remaining = num_rows & (shape_rows - 1)  # num_rows % shape_rows
# #     #     if remaining != 0:
# #     #         bubble_rows = shape_rows - remaining
# #     #     else:
# #     #         bubble_rows = 0
# #     #     # test
# #     #     print(F"bucket: {bucket} max_width: {origin_max_width} bubble_rows: {bubble_rows}")
# #     #     # end test
# #     #     overhead += bubble_rows

# #     # return overhead
# #     return new_overhead


# def get_memory_overhead(buckets: dict,
#                     pre_max_width: int,
#                     new_max_width: int):
#     larger_bucket = buckets[pre_max_width]
#     ratio = pre_max_width / new_max_width
#     cost = (ratio - 1) * larger_bucket.num_rows

#     # # Update the buckets
#     # new_num_rows = larger_bucket.num_rows * ratio
#     # if new_max_width not in buckets:
#     #     buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz)
#     # else:
#     #     buckets[new_max_width].num_rows += new_num_rows
#     #     buckets[new_max_width].nnz += larger_bucket.nnz

#     # del buckets[pre_max_width]

#     return cost    


# def modify_bucket_widths(buckets: dict):
#     """Modify initial bucket settings by using a cost model.

#     Args:
#         buckets (Dict{bucket_width: Bucket}): Dictionary of bucket width to Bucket object.
#     """
#     # # test
#     # for width, bk in buckets.items():
#     #     print(F"504 width: {width} bucket: {bk}")
#     # # end test
#     # Get original overhead as baseline
#     base_overhead = get_bubble_overhead(buckets)
#     base_buckets_list = get_bucket_widths_list(buckets)
#     min_overhead = base_overhead
#     best_setting = base_buckets_list
#     # # test
#     # print(F"base_overhead: {base_overhead} base_buckets_list: {base_buckets_list}")
#     # # end test

#     origin_max_width = max(buckets.keys())
#     pre_max_width = origin_max_width
#     new_max_width = pre_max_width // 2
#     while new_max_width >= 1:
#         # # test
#         # print(F"pre_max_width: {pre_max_width} new_max_width: {new_max_width}")
#         # # end test
#         # Calculate the overhead if change the max bucket width
#         mem_overhead = get_memory_overhead(buckets, pre_max_width, new_max_width)
#         move_largest_bucket(buckets, pre_max_width, new_max_width)
#         # # test
#         # for width, bk in buckets.items():
#         #     print(F"539 width: {width} bucket: {bk}")
#         # # end test
#         bb_overhead = get_bubble_overhead(buckets)
#         total_overhead = mem_overhead + 4 * bb_overhead
#         # # test
#         # print(F"mem_overhead: {mem_overhead} bb_overhead: {bb_overhead} total_overhead: {total_overhead}")
#         # # end test
#         if total_overhead < min_overhead:
#             best_setting = get_bucket_widths_list(buckets)
#             min_overhead = total_overhead
#         pre_max_width = new_max_width
#         new_max_width //= 2
    
#     return best_setting


# def get_bucket_config(init_buckets):
#     num_parts = len(init_buckets)
#     bucket_config = []

#     for part_ind in range(num_parts):
#         # # test
#         # print(F"Go to partition {part_ind}...")
#         # # end test
#         buckets = init_buckets[part_ind]
#         # # test
#         # print(F"533 buckets: {buckets}")
#         # # end test
#         # # test
#         # print(F"old buckets setting: {get_bucket_widths_list(buckets)}")
#         # # end test
#         bucket_widths = modify_bucket_widths(buckets)
#         # # test
#         # print(F"537 bucket_widths: {bucket_widths}")
#         # # end test
#         # # test
#         # print(F"new buckets setting: {bucket_widths}")
#         # # end test
#         bucket_config.append(bucket_widths)

#     return bucket_config


# def search_bucket_sizes(g: MTX,
#                         # x,
#                         # y_ndarray,
#                         # feat_size,
#                         num_partitions):
#                         # coarsening_factor,
#                         # use_implicit_unroll):
#     # init_buckets()
#     init_buckets = g.init_buckets(num_partitions)
#     # # test
#     # for part_ind in range(num_partitions):
#     #     buckets = init_buckets[part_ind]
#     #     print(F"part_id: {part_ind} num_buckets: {len(buckets)}")
#     #     for width, bucket in buckets.items():
#     #         print(F"bucekt: width={width} num_rows={bucket.num_rows} nnz={bucket.nnz}")
#     # # end test
            
#     bucket_config = get_bucket_config(init_buckets)

#     return bucket_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--num-partitions", "-p", type=int, help="number of column partitions")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", default=True, help="use implicit unroll")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    # name = args.dataset
    # g = get_dataset(name)
    filename = args.dataset
    g = MTX(filename)
    num_parts = args.num_partitions

    # GPU device
    print(F"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', default=0)}")

    start_time = time.perf_counter()
    # bucket_config = search_bucket_sizes(g, num_parts)
    bucket_config = search_bucket_config(g, num_parts)
    end_time = time.perf_counter()
    search_overhead = end_time - start_time

    # test
    # bucket_config = [[1, 2, 4, 8, 16, 32, 64]]
    # bucket_config = [[8, 128, 1024]]
    # end test

    for feat_size in [32]:
    # for feat_size in [32, 64, 128, 256, 512]:
        features = g.matrix_features()
        features["K"] = feat_size

        # # If done before, skipped
        # if check_if_done_before(features["name"], feat_size):
        #     continue

        # x = th.rand((g.num_dst_nodes(), feat_size))
        x = th.ones((g.num_dst_nodes(), feat_size))
        # y_golden = dgl.ops.copy_u_sum(g, x)
        y_ndarray = g.dot(x.numpy())

        hyb_format = build_hyb_format(g,
                                    bucket_config)
        name = features["name"]
        print(F"#### data: {name} feat_size: {feat_size} num_partitions: {num_parts} bucket_config: {bucket_config}", flush=True)
        try:
            exe_time = bench_hyb_with_config(g,
                                    x,
                                    # y_golden,
                                    y_ndarray,
                                    feat_size=feat_size,
                                    bucket_widths=bucket_config,
                                    bucketing_format=hyb_format,
                                    coarsening_factor=2,
                                    num_col_parts=num_parts,
                                    use_implicit_unroll=args.implicit_unroll)
            exe_time = float(F"{exe_time:.6}")
        except Exception as e:
            exe_time = math.inf
            print(e, file=sys.stderr)
        features["best_num_partitions"] = num_parts
        features["bucket_widths"] = [str(bucket_config)]
        features["search_time(s)"] = search_overhead
        features["best_exe_time"] = [exe_time]

        # # execution_times = []
        # # partitions = []
        # # max_bucket_sizes = []
        # search_bucket_sizes(features,
        #                     g,
        #                     x,
        #                     y_ndarray,
        #                     feat_size=feat_size,
        #                     num_partitions=num_parts,
        #                     coarsening_factor=2,
        #                     use_implicit_unroll=args.implicit_unroll)
        # # features["best_num_partitions"] = best_config.num_parts
        # # features["best_max_bucket_width"] = best_config.max_bucket_width
        # # features["best_exe_time"] = [best_config.exe_time]
        
        # # Statistics
        # save_statistics(features,
        #                 execution_times,
        #                 partitions,
        #                 max_bucket_sizes)
        save_statistics(features)
