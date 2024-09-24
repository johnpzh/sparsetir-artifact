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
from tvm.script import tir as T
from format_algos import (
    build_hyb_format,
    bench_hyb_with_config,
    search_bucket_config,
    CostModelSettings,
    bench_bsrmm,
    bench_naive,
    bench_tensorcores_with_config,
)
from joblib import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from scipy import sparse as sp

import tvm
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
from sparsetir_artifact import profile_tvm_ms


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


cached_bucketing_format = None


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
    check_correctness=False
):
    num_buckets = len(bucket_sizes)
    coarsening_factor = min(coarsening_factor, feat_size // 32)
    indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
        indices_nd = tvm.nd.array(indices, device=tvm.cpu())
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
    f(*args)
    if check_correctness:
        # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
        tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    print("tir hyb time: {:.5f} ms".format(dur))

    return dur




if __name__ == "__main__":
    parser = argparse.ArgumentParser("predict the format selection")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--model-format-selection", "-s", type=str, help="model for format selection")
    parser.add_argument("--model-num-partitions", "-p", type=str, help="model for format selection")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    name = args.dataset
    model_selection_name = args.model_format_selection
    model_num_partitions = args.model_num_partitions

    # Load data
    clf_selection = load(model_selection_name)
    clf_partition = load(model_num_partitions)
    model = AutoModelForQuestionAnswering.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )

    names_list = []
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


            # Try hyb


            # end

            # Load the matrix
            # csr_weight = sp.csr_matrix(param.detach().numpy())

            # Padding matrix
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

            # # normal matrix
            # coo_weight = sp.coo_matrix(param.detach().numpy())
            # density = coo_weight.nnz / param.numel()
            # g = MTX(None)
            # g.name = name
            # g.coo_mtx = coo_weight

            # # test
            # np.set_printoptions(threshold=np.inf)
            # print(g.coo_mtx.toarray())
            # # end test

            # # Try original hyb
            # num_features = 512
            # x = th.rand((g.num_dst_nodes(), num_features))
            # # y_golden = dgl.ops.copy_u_sum(g, x)
            # y_ndarray = g.dot(x.numpy())
            # exe_time = bench_hyb(g,
            #           x,
            #           y_ndarray,
            #           feat_size=num_features,
            #           bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            #           coarsening_factor=2,
            #           num_col_parts=2,
            #           use_implicit_unroll=True
            #     )
            # print(f"density: {density} exe_time(ms): {exe_time}")
            # sys.exit(-1)

            # Predict format selection
            format, time_selet = predict_format_selection(g, clf=clf_selection)

            if format == "CELL":
            # if False:
                # Use CELL format
                # Predict number of partitions
                num_features = 512
                num_parts, time_parts = predict_num_partitions(g, clf=clf_partition, num_features=num_features)

                # test
                # num_parts = num_rows // 256
                num_parts = 1
                # end test

                # Build buckets
                time_bucket_start = time.perf_counter()
                cost_model_config = CostModelSettings(feat_size=num_features, num_parts=num_parts)
                bucket_config = search_bucket_config(g, num_parts, cost_model_config)
                time_bucket_end = time.perf_counter()
                time_bucket = time_bucket_end - time_bucket_start

                # # test
                # num_parts = 768 // 32
                # bucket_config = []
                # for _ in range(num_parts):
                #     bucket_config.append([32])
                bucket_config = [[32]]
                # # end test

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
                    exe_time = bench_tensorcores_with_config(g,
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
                    
                # Record statistics
                names_list.append(name)
                formats_list.append(format)
                num_features_list.append(num_features)
                num_parts_list.append(num_parts)
                time_selet_list.append(time_selet)
                time_num_parts_list.append(time_parts)
                time_bucket_list.append(time_bucket)
                time_exe_list.append(exe_time)
            else:
                format = "BSR"
                # Not use CELL format
                num_features = 512
                # for num_features in [32, 64, 128, 256, 512]:
                # mtx = MTX(name)
                num_src = g.num_src_nodes()
                num_dst = g.num_dst_nodes()
                # assert num_src == num_dst, f"Error: matrix {name} is not sqaure ({num_src} x {num_dst})."

                # Block size
                bsize = 32

                # # Padding the size
                # print(f"original_size: ({num_src}, {num_dst})")
                # num_block_row = (num_src + bsize - 1) // bsize
                # num_src = num_block_row * bsize
                # num_dst = num_src
                # print(f"padded_size: ({num_src}, {num_dst}) bsize: {bsize}")

                try:
                    # raise Exception("test CSR")
                    if name not in ['proteins', 'reddit']:
                        bsr_weight = g.tobsr_with_padding(shape=(num_src, num_dst), blocksize=(bsize, bsize))
                        # print(f"g: {g.coo_mtx.toarray()}")
                        # print(f"bsr_weight: {bsr_weight.toarray()}")
                        print(f"g.shape: ({g.num_src_nodes(), g.num_dst_nodes()}) nnz: {g.num_edges()}")
                        print(f"bsr_weight.shape: ({bsr_weight.shape[0], bsr_weight.shape[1]}) nnz: {bsr_weight.nnz} size: {bsr_weight.size}")

                        x = th.rand(bsr_weight.shape[1], num_features).half()
                        # print(f"mtx_bsr: {mtx_bsr}")
                        # csr_weight = mtx.tocsr()
                        exe_time = bench_bsrmm(bsr_weight, x, block_size=bsize)
                        print(f"name: {name} exe_time: {exe_time}")
                        # print(f"mtx_csr: {mtx_csr}")
                        # formats_list.append("BSR")
                        # format = "BSR"
                        # blocksizes_list.append(bsize)
                        # time_exe_list.append(exe_time)
                    else:
                        # name is in ['proteins', 'reddit']
                        raise Exception(f"name {name} cannot do BSR (scipy.sparse.bsr_matrix() will cause Segmentation Fault).")
                except Exception as e:
                    print(e, file=sys.stderr)
                    print(f"name {name} in BSR is out-of-memory. Do CSR then.")
                    format = "CSR"
                    try:
                        # BSR is too large to be handled.
                        # Try CSR instead.
                        x = th.rand((g.num_dst_nodes(), num_features))
                        # y_golden = dgl.ops.copy_u_sum(g, x)
                        y_ndarray = g.dot(x.numpy())
                        exe_time = bench_naive(g,
                                            x,
                                            y_ndarray,
                                            feat_size=num_features,
                                            coarsening_factor=2)
                        print(f"name: {name} exe_time: {exe_time}")
                        # formats_list.append("CSR")
                        # format = "CSR"
                        # blocksizes_list.append(0)
                        # time_exe_list.append(exe_time)
                    except Exception as e2:
                        print(e2, file=sys.stderr)
                        print(f"name: {name} in CSR is also out-of-memory. Exit.")
                        # formats_list.append("Crashed")
                        exe_time = 0
                        format = "CRASHED"
                        # blocksizes_list.append(0)
                        # time_exe_list.append(0)
                
                # Record statistics
                names_list.append(name)
                formats_list.append(format)
                num_features_list.append(num_features)
                num_parts_list.append(0)
                time_selet_list.append(time_selet)
                time_num_parts_list.append(0)
                time_bucket_list.append(0)
                time_exe_list.append(exe_time)

            # test
            sys.exit(-1)
            # end test
    
    table = {
        "name": names_list,
        "format": formats_list,
        "feat_size": num_features_list,
        "num_parts": num_parts_list,
        "time_select(s)": time_selet_list,
        "time_num_parts(s)": time_num_parts_list,
        "time_bucket(s)": time_bucket_list,
        "time_exe_our(ms)": time_exe_list,
    }

    df = pd.DataFrame(data=table)
    print(df.to_string())
    df.to_csv(f"output.{sys.argv[0]}.csv", index=False)
