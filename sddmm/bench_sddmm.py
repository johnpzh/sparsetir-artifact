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
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from ogb.nodeproppred import DglNodePropPredDataset
from sparsetir_artifact import profile_tvm_ms
from utils import get_dataset


def sddmm(m: int, n: int, feat_size: int, nnz: int):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        indptr: T.handle,
        indices: T.handle,
    ) -> None:
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        I = T.dense_fixed(m)
        J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
        J_detach = T.dense_fixed(n)
        K = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (I, K), "float32")
        B = T.match_sparse_buffer(b, (J_detach, K), "float32")
        C = T.match_sparse_buffer(c, (I, J), "float32")

        with T.iter([I, J, K], "SSR", "sddmm") as [i, j, k]:
            with T.init():
                C[i, j] = 0.0
            C[i, j] = C[i, j] + A[i, k] * B[j, k]

    return func


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    global sddmm
    indptr, indices, _ = g.adj_tensors("csr")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()

    a = th.rand(m, feat_size).to(th.float32)
    b = th.rand(n, feat_size).to(th.float32)
    c = th.zeros(nnz).to(th.float32)

    # dgl
    a_gpu = a.to(0)
    b_gpu = b.to(0)
    g = g.to(0)
    c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)

    # tvm
    mod = tvm.IRModule.from_expr(sddmm(m, n, feat_size, nnz))
    sch = tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("sddmm")
    i, j, k = sch.get_sp_iters(sp_iteration)
    sch.sparse_fuse(sp_iteration, [i, j])
    mod = lower_sparse_iter(sch.mod)

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
    preproc = tvm.build(mod["main"], target="cuda")

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)
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
        print("feat_size = ", feat_size)
        try:
            bench_sddmm(g, feat_size)
        except Exception as e:
            print("OOM")
            print(e, file=sys.stderr)
