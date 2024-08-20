import dgl
import sys
import tvm
import argparse
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.script import parser as P
# from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from ogb.nodeproppred import DglNodePropPredDataset
from sparsetir_artifact import profile_tvm_ms
from utils import get_dataset
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)

def create_prim_func(num_col_parts, bucket_size, feat_size):
    prim_func = \
"""
@T.prim_func
def func(
    a: T.handle,
    b: T.handle,
    m: T.int32,
    n: T.int32,
"""

    for i in range(num_col_parts):
        for j in range(bucket_size):
            prim_func += 4*" "+ "c_{}_{}: T.handle,\n".format(i,j)
            prim_func += 4*" "+ "indptr_i_{}_{}: T.handle,\n".format(i,j)
            prim_func += 4*" "+ "indices_i_{}_{}: T.handle,\n".format(i,j)
            prim_func += 4*" "+ "indices_j_{}_{}: T.handle,\n".format(i,j)
            prim_func += 4*" "+ "num_rows_{}_{}:T.int32,\n".format(i,j)
            prim_func += 4*" "+ "nnz_cols_{}_{}:T.int32,\n".format(i,j)
    prim_func += ") -> None:\n"
    prim_func += 4*" " + "K = T.dense_fixed({})\n".format(feat_size)
    prim_func += \
"""    
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I_detach = T.dense_fixed(m)    
    J_detach = T.dense_fixed(n)
    A = T.match_sparse_buffer(a, (I_detach, K), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
"""
    for i in range(num_col_parts):
        for j in range(bucket_size):
            prim_func+= 4*" " +  "O_{0}_{1} = T.dense_fixed(1)\n".format(i,j)
            prim_func+= 4*" " +  "I_{0}_{1} = T.sparse_variable(O_{0}_{1}, (m, num_rows_{0}_{1}), (indptr_i_{0}_{1}, indices_i_{0}_{1}))\n".format(i,j)
            prim_func+= 4*" " +  "J_{0}_{1} = T.sparse_fixed(I_{0}_{1}, (n, nnz_cols_{0}_{1}), indices_j_{0}_{1})\n".format(i,j)
    
    for i in range(num_col_parts):
        for j in range(bucket_size):
            prim_func+= 4*" " +  'C_{0}_{1} = T.match_sparse_buffer(c_{0}_{1}, (O_{0}_{1}, I_{0}_{1}, J_{0}_{1}), "float32")\n'.format(i,j)
    
    for i in range(num_col_parts):
        for j in range(bucket_size):
            prim_func+= 4*" " +'with T.iter([O_{0}_{1}, I_{0}_{1}, J_{0}_{1}, K], "SSSR", "sddmm_{0}_{1}") as [o, i, j, k]:\n'.format(i,j)
            prim_func+= 8*" " +'with T.init():\n'
            prim_func+= 12*" " + "C_{0}_{1}[o, i, j] = 0.0\n".format(i,j)
            prim_func+= 8*" " + "C_{0}_{1}[o, i, j] = C_{0}_{1}[o, i, j] + A[i, k] * B[j, k]\n".format(i,j)
    return prim_func

# @T.prim_func
# def ell(
#     a: T.handle,
#     b: T.handle,
#     c_0: T.handle,
#     c_1: T.handle,
#     m: T.int32,
#     n: T.int32,
#     indptr_i_0: T.handle,
#     indices_i_0: T.handle,
#     indices_j_0: T.handle,
#     indptr_i_1: T.handle,
#     indices_i_1: T.handle,
#     indices_j_1: T.handle,
#     num_rows_0: T.int32,
#     num_rows_1: T.int32,
#     nnz_cols_0: T.int32,
#     nnz_cols_1: T.int32,
# ) -> None:
#     T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
#     I_detach = T.dense_fixed(m)    
#     J_detach = T.dense_fixed(n)
#     K = T.dense_fixed(32)

#     O_0 = T.dense_fixed(1)
#     I_0 = T.sparse_variable(O_0, (m, num_rows_0), (indptr_i_0, indices_i_0))
#     J_0 = T.sparse_fixed(I_0, (n, nnz_cols_0), indices_j_0)
#     O_1 = T.dense_fixed(1)
#     I_1 = T.sparse_variable(O_1, (m, num_rows_1), (indptr_i_1, indices_i_1))
#     J_1 = T.sparse_fixed(I_1, (n, nnz_cols_1), indices_j_1)
    

#     A = T.match_sparse_buffer(a, (I_detach, K), "float32")
#     B = T.match_sparse_buffer(b, (J_detach, K), "float32")
#     C_0 = T.match_sparse_buffer(c_0, (O_0, I_0, J_0), "float32")
#     C_1 = T.match_sparse_buffer(c_1, (O_1, I_1, J_1), "float32")

#     with T.iter([O_0, I_0, J_0, K], "SSSR", "sddmm0") as [o, i, j, k]:
#         with T.init():
#             C_0[o, i, j] = 0.0
#         C_0[o, i, j] = C_0[o, i, j] + A[i, k] * B[j, k]

#     with T.iter([O_1, I_1, J_1, K], "SSSR", "sddmm1") as [o, i, j, k]:
#         with T.init():
#             C_1[o, i, j] = 0.0
#         C_1[o, i, j] = C_1[o, i, j] + A[i, k] * B[j, k]

cached_bucketing_format = None


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    num_col_parts = 1
    bucket_sizes = [1, 2]
    num_buckets = len(bucket_sizes)
    global sddmm
    indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
        cached_bucketing_format = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )
    row_indices, col_indices, mask = cached_bucketing_format

    a = th.rand(m, feat_size).to(th.float32)
    b = th.rand(n, feat_size).to(th.float32)
    cs = []
    for part_id in range(num_col_parts):
        cs.append([])
        for bucket_id in range(num_buckets):
            cs[part_id].append(tvm.nd.array(th.zeros(row_indices[part_id][bucket_id].shape[0]*bucket_sizes[bucket_id]).to(th.float32).numpy()))

    ell_sddmm = create_prim_func(num_col_parts, num_buckets, feat_size)
    ell_sddmm = tvm.IRModule.from_expr(P.from_source(ell_sddmm))
    
    parsed_params = ell_sddmm["main"].params
    parsed_param_map = {
        parsed_params[2]: m,  # m
        parsed_params[3]: n,  # n
    }

    for part_id in range(num_col_parts):
        for bucket_id in range(num_buckets):
                parsed_param_map[parsed_params[4 + 6 * (part_id * num_buckets + bucket_id) + 4]] = row_indices[
                    part_id
                ][bucket_id].shape[0]
                parsed_param_map[parsed_params[4 + 6 * (part_id * num_buckets + bucket_id) + 5]] = bucket_sizes[bucket_id]

    ell_sddmm["main"] = ell_sddmm["main"].specialize(parsed_param_map)#.with_attr("horizontal_fuse", True)

    schell = tir.Schedule(ell_sddmm)
    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            sp_iteration = schell.get_sparse_iteration("sddmm_{}_{}".format(part_id, bucket_id))
            o, i, j, k = schell.get_sp_iters(sp_iteration)
            schell.sparse_fuse(sp_iteration, [o, i])

    ell_sddmm = lower_sparse_iter(schell.mod)
    ell_sddmm = lower_sparse_buffer(ell_sddmm)
    ell_sddmm = tvm.tir.transform.RemoveUnusedArgs()(ell_sddmm)
    
    indptr_nd = tvm.nd.array(np.zeros(1, dtype=np.int32))
    indices_nd = tvm.nd.array(indices.numpy())
    a_nd = tvm.nd.array(a.numpy().reshape(-1))
    b_nd = tvm.nd.array(b.numpy().reshape(-1))
    # c_0_nd = tvm.nd.array(c_0.numpy())
    # c_1_nd = tvm.nd.array(c_1.numpy())
    args = [a_nd, b_nd]
    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            c_nd = cs[part_id][bucket_id]
            rows = row_indices[part_id][bucket_id]
            cols = tvm.nd.array(col_indices[part_id][bucket_id].numpy().reshape(-1))
            args.append(c_nd)
            args.append(rows)
            args.append(cols)


    ell_sddmm = tvm.build(ell_sddmm["main"])
    ell_sddmm(*args)

    # tvm.testing.assert_allclose(
    #     c_0_nd.numpy(), c_0_nd_.numpy(), rtol=1e-5
    # )
    # tvm.testing.assert_allclose(
    #     c_1_nd.numpy(), c_1_nd_.numpy(), rtol=1e-5
    # )


    # dgl
    a_gpu = a.to(0)
    b_gpu = b.to(0)
    g = g.to(0)
    c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)


    # tvm.testing.assert_allclose(
    #     c_1_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-5
    # )

    print(cs[0][1].numpy())
    # print(c_golden.shape)
    print(c_golden.cpu().numpy())
    exit(0)

    # split preprocess and compute
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

    # schedule preprocess
    sch = tir.Schedule(mod_preprocess)
    blk = sch.get_block("binary_search_block_0_0")
    (i,) = sch.get_loops(blk)
    io, ii = sch.split(i, [None, 32])
    sch.bind(ii, "threadIdx.x")
    sch.bind(io, "blockIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    print(mod)
    preproc = tvm.build(mod["main"], target="cuda")
    print(preproc)

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)
    print(mid_nd.shape)
    ty_candidates = [1, 2, 4, 8]
    tx_candidates = [8, 16, 32]
    vecsize_candidates = [1, 2, 4]
    groupsize_candidates = [1, 2, 4]

    best = 1e9
    for ty in ty_candidates:
        for tx in tx_candidates:
            for vec_size in vecsize_candidates:
                for group_size in groupsize_candidates:
                    if tx * vec_size > feat_size:
                        continue
                    # schedule compute
                    sch = tir.Schedule(mod_sddmm)
                    blk = sch.get_block("sddmm0")
                    j, k = sch.get_loops(blk)
                    ko, kio, kii = sch.split(k, [None, tx, vec_size])
                    rf_blk = sch.rfactor(kio, 2)
                    j = sch.get_loops(rf_blk)[0]
                    joo, joi, ji = sch.split(j, [None, ty, group_size])
                    sch.bind(joo, "blockIdx.x")
                    sch.bind(joi, "threadIdx.y")
                    sch.unroll(ji)
                    sch.reverse_compute_at(blk, joi, True)
                    sch.set_scope(rf_blk, 0, "local")
                    read_A = sch.cache_read(rf_blk, 0, "local")
                    read_B = sch.cache_read(rf_blk, 2, "local")
                    write_C = sch.cache_write(blk, 0, "local")
                    ko, kio, kii = sch.get_loops(rf_blk)[-3:]
                    sch.reorder(ko, ji)
                    # schedule read A
                    sch.compute_at(read_A, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_A)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule read B
                    sch.compute_at(read_B, ji, True)
                    ax0, ax1 = sch.split(sch.get_loops(read_B)[-1], [tx, vec_size])
                    sch.bind(ax0, "threadIdx.x")
                    sch.vectorize(ax1)
                    # schedule write C
                    sch.reverse_compute_at(write_C, joi, True)
                    ax0, ax1 = sch.get_loops(write_C)[-2:]
                    sch.vectorize(ax1)
                    # schedule rf
                    sch.bind(kio, "threadIdx.x")
                    sch.unroll(kii)
                    sch.unroll(ko)
                    # schedule write back
                    ax0, ax1, ax2 = sch.get_loops(blk)[-3:]
                    sch.reorder(ax1, ax2, ax0)
                    sch.bind(ax0, "threadIdx.x")
                    sch.unroll(ax2)
                    sch.unroll(ax1)
                    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
                    print(mod)
                    f = tvm.build(mod["main"], target="cuda")

                    # check result
                    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd]
                    f(*args)
                    tvm.testing.assert_allclose(
                        c_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-5
                    )

                    # evaluate time
                    mean_time = profile_tvm_ms(f, args)

                    if mean_time < best:
                        best = mean_time
                        best_config = (tx, ty, vec_size, group_size)
    print("sparse tir:\t{:.5f} ms".format(best))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sddmm in sparse-tir")
    parser.add_argument(
        "--dataset", "-d", type=str, default="pubmed", help="dataset name"
    )
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)
    for feat_size in [32, 64, 128, 256, 512]:
    # for feat_size in [32]:
        print("feat_size = ", feat_size)
        try:
            bench_sddmm(g, feat_size)
        except Exception as e:
            print("OOM")
            print(e, file=sys.stderr)
