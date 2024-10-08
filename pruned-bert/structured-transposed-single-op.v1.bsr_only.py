import argparse
from typing import Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertModel
from scipy import sparse as sp
import pytest
import numpy as np
import torch as th
import pandas as pd
import triton
import tvm
import os
import sys
from tvm.sparse import column_part_hyb
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter
from sparsetir_artifact import profile_pytorch_ms, profile_tvm_ms


def bsrmm(mb, nb, nnz, blk, feat_size):
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
        I = T.dense_fixed(mb)
        J = T.sparse_variable(I, (nb, nnz), (indptr, indices), "int32")
        J_detach = T.dense_fixed(nb)
        BI = T.dense_fixed(blk)
        BJ = T.dense_fixed(blk)
        F = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float16")
        B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
        C = T.match_sparse_buffer(c, (I, BI, F), "float16")

        with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
            i,
            bi,
            bj,
            f,
            j,
        ]:
            with T.init():
                C[i, bi, f] = T.float16(0.0)
            C[i, bi, f] = C[i, bi, f] + A[i, j, bi, bj] * B[j, bj, f]

    return func


def dbsrmm(mb, nb, nnz0, nnz1, blk, feat_size):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        c: T.handle,
        indptr_0: T.handle,
        indices_0: T.handle,
        indptr_1: T.handle,
        indices_1: T.handle,
    ) -> None:
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        O = T.dense_fixed(1)
        I = T.sparse_variable(O, (mb, nnz0), (indptr_0, indices_0), "int32")
        I_detach = T.dense_fixed(mb)
        J = T.sparse_variable(I, (nb, nnz1), (indptr_1, indices_1), "int32")
        J_detach = T.dense_fixed(nb)
        BI = T.dense_fixed(blk)
        BJ = T.dense_fixed(blk)
        F = T.dense_fixed(feat_size)
        A = T.match_sparse_buffer(a, (O, I, J, BI, BJ), "float16")
        B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
        C = T.match_sparse_buffer(c, (I_detach, BI, F), "float16")

        with T.iter([O, I, BI, BJ, F, J], "SSSRSR", "bsrmm") as [
            o,
            i,
            bi,
            bj,
            f,
            j,
        ]:
            with T.init():
                C[i, bi, f] = T.float16(0.0)
            C[i, bi, f] = C[i, bi, f] + A[o, i, j, bi, bj] * B[j, bj, f]

    return func


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = (
                    C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]
                )


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:16, 0:16],
                A_frag[0:16, 0:16],
                B_frag[0:16, 0:16],
            ]
        )
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256
                    + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256
                    + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )


def wmma_load_a(scope: str):
    @T.prim_func
    def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 16), "float16", align=128, offset_factor=1, scope=scope
        )
        A_frag = T.match_buffer(
            a_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(A_frag[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    A_frag[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        A_frag = T.match_buffer(
            a_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(A_frag[0:16, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_load_matrix_sync(
                        A_frag.data,
                        16,
                        16,
                        16,
                        A_frag.elem_offset // 256
                        + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                        A.access_ptr("r"),
                        A.strides[0],
                        "row_major",
                        dtype="handle",
                    )
                )

    return wmma_load_a_desc, wmma_load_a_impl


def wmma_load_b(scope: str):
    @T.prim_func
    def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
        B = T.match_buffer(
            b, (16, 16), "float16", align=128, offset_factor=1, scope=scope
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        with T.block("root"):
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    B_frag[vii, vjj] = B[vii, vjj]

    @T.prim_func
    def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        B = T.match_buffer(
            b,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope=scope,
            strides=[s0, s1],
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=128,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        with T.block("root"):
            T.reads(B[0:16, 0:16])
            T.writes(B_frag[0:16, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_load_matrix_sync(
                        B_frag.data,
                        16,
                        16,
                        16,
                        B_frag.elem_offset // 256
                        + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                        B.access_ptr("r"),
                        B.strides[0],
                        "row_major",
                        dtype="handle",
                    )
                )

    return wmma_load_b_desc, wmma_load_b_impl


@T.prim_func
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=1, scope="global"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="wmma.accumulator",
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float16",
        align=128,
        offset_factor=1,
        scope="global",
        strides=[s0, s1],
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256
                    + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )


WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A_SHARED = tir.TensorIntrin.register(
    "wmma_load_a_shared", *wmma_load_a("shared")
)

WMMA_LOAD_A_GLOBAL = tir.TensorIntrin.register(
    "wmma_load_a_global", *wmma_load_a("global")
)

WMMA_LOAD_B_SHARED = tir.TensorIntrin.register(
    "wmma_load_b_shared", *wmma_load_b("shared")
)

WMMA_LOAD_B_GLOBAL = tir.TensorIntrin.register(
    "wmma_load_b_global", *wmma_load_b("global")
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)

WMMA_STORE = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)


def bench_bsrmm(bsr_mat: Any, x: th.Tensor, block_size: int):
    global bsrmm
    mb = bsr_mat.shape[0] // bsr_mat.blocksize[0]
    nb = bsr_mat.shape[1] // bsr_mat.blocksize[1]
    nnzb = bsr_mat.nnz // (block_size**2)
    feat_size = x.shape[1]
    ind = (bsr_mat.indptr[1:] - bsr_mat.indptr[:-1]).nonzero()[0]
    print(bsr_mat.indptr[ind + 1] - bsr_mat.indptr[ind])

    mod = tvm.IRModule.from_expr(bsrmm(mb, nb, nnzb, block_size, feat_size))
    sch = tvm.tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
    sch.sparse_reorder(sp_iteration, [i, j, bi, f, bj])
    mod = lower_sparse_iter(sch.mod)
    sch = tir.Schedule(mod)
    blk_inner = sch.get_block("bsrmm1")
    blk_outer = sch.get_block("bsrmm0")
    j, bi, f, bj = sch.get_loops(blk_inner)
    bio, bii = sch.split(bi, [block_size // 16, 16])
    bjo, bji = sch.split(bj, [block_size // 16, 16])
    foo, foi, fi = sch.split(f, [None, 2, 16])
    sch.reorder(foo, j, bio, foi, bjo, bii, fi, bji)
    sch.unroll(foi)
    (i,) = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(bio, "threadIdx.y")
    sch.bind(foo, "blockIdx.y")
    C_local = sch.cache_write(blk_inner, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_local, foo, True)
    ax0, ax1 = sch.get_loops(C_local)[-2:]
    ax2, ax3 = sch.split(ax1, [None, 16])
    ax0, ax1 = sch.split(ax0, [None, 16])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    init_blk = sch.decompose_reduction(blk_inner, j)
    A_local = sch.cache_read(blk_inner, 1, "wmma.matrix_a")
    sch.compute_at(A_local, bio)
    ax0, ax1 = sch.get_loops(A_local)[-2:]
    ax1, ax2 = sch.split(ax1, [None, 16])
    sch.reorder(ax1, ax0, ax2)
    sch.unroll(ax1)
    B_shared = sch.cache_read(blk_inner, 2, "shared")
    sch.compute_at(B_shared, foi)
    B_local = sch.cache_read(blk_inner, 2, "wmma.matrix_b")
    sch.compute_at(B_local, bjo)
    sch.hide_buffer_access(blk_inner, "read", [3])
    sch.tensorize(sch.get_loops(blk_inner)[-3], "wmma_sync")
    sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b_shared")
    sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a_global")
    sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
    sch.tensorize(sch.get_loops(init_blk)[-2], "wmma_fill")
    # schedule B_shared
    ax0, ax1 = sch.get_loops(B_shared)[-2:]
    fused_ax = sch.fuse(ax0, ax1)
    ax0, ax1, ax2, ax3 = sch.split(fused_ax, [None, 2, 32, 4])
    sch.vectorize(ax3)
    sch.bind(ax2, "threadIdx.x")
    sch.bind(ax1, "threadIdx.y")
    sch.unroll(ax0)

    mod = lower_sparse_buffer(sch.mod)

    f = tvm.build(mod["main"], target="cuda")

    ctx = tvm.cuda(0)
    A_indptr = tvm.nd.array(np.copy(bsr_mat.indptr).astype("int32"), device=ctx)
    A_indices = tvm.nd.array(np.copy(bsr_mat.indices).astype("int32"), device=ctx)
    A_data = tvm.nd.array(
        np.copy(bsr_mat.data).reshape(-1).astype("float16"), device=ctx
    )
    X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(
        np.zeros((mb * block_size * feat_size), dtype="float16"), device=ctx
    )
    args = [A_data, X_nd, Y_nd, A_indptr, A_indices]
    f(*args)

    avg_time = profile_tvm_ms(f, args)
    print("bsrmm time: \t{:.5f}ms".format(avg_time))
    return avg_time


def bench_dbsrmm(bsr_mat: Any, x: th.Tensor, block_size: int):
    global bsrmm
    mb = bsr_mat.shape[0] // bsr_mat.blocksize[0]
    nb = bsr_mat.shape[1] // bsr_mat.blocksize[1]
    nnzb = bsr_mat.nnz // (block_size**2)
    feat_size = x.shape[1]
    row_indices = np.nonzero(bsr_mat.indptr[:-1] != bsr_mat.indptr[1:])[0]
    unique_indptr = np.concatenate([np.array([0]), bsr_mat.indptr[row_indices + 1]])
    nnz0 = len(row_indices)

    mod = tvm.IRModule.from_expr(dbsrmm(mb, nb, nnz0, nnzb, block_size, feat_size))
    sch = tvm.tir.Schedule(mod)
    sp_iteration = sch.get_sparse_iteration("bsrmm")
    o, i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
    sch.sparse_reorder(sp_iteration, [o, i, j, bi, f, bj])
    sch.sparse_fuse(sp_iteration, [o, i])
    mod = lower_sparse_iter(sch.mod)
    sch = tvm.tir.Schedule(mod)
    blk_inner = sch.get_block("bsrmm1")
    blk_outer = sch.get_block("bsrmm0")
    j, bi, f, bj = sch.get_loops(blk_inner)
    bio, bii = sch.split(bi, [block_size // 16, 16])
    bjo, bji = sch.split(bj, [block_size // 16, 16])
    foo, foi, fi = sch.split(f, [None, 2, 16])
    sch.reorder(foo, j, bio, foi, bjo, bii, fi, bji)
    sch.unroll(foi)
    (i,) = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(bio, "threadIdx.y")
    sch.bind(foo, "blockIdx.y")
    C_local = sch.cache_write(blk_inner, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_local, foo, True)
    ax0, ax1 = sch.get_loops(C_local)[-2:]
    ax2, ax3 = sch.split(ax1, [None, 16])
    ax0, ax1 = sch.split(ax0, [None, 16])
    sch.reorder(ax0, ax2, ax1, ax3)
    sch.unroll(ax2)
    sch.bind(ax0, "threadIdx.y")
    init_blk = sch.decompose_reduction(blk_inner, j)
    A_local = sch.cache_read(blk_inner, 2, "wmma.matrix_a")
    sch.compute_at(A_local, bio)
    ax0, ax1 = sch.get_loops(A_local)[-2:]
    ax1, ax2 = sch.split(ax1, [None, 16])
    sch.reorder(ax1, ax0, ax2)
    sch.unroll(ax1)
    B_shared = sch.cache_read(blk_inner, 3, "shared")
    sch.compute_at(B_shared, foi)
    B_local = sch.cache_read(blk_inner, 3, "wmma.matrix_b")
    sch.compute_at(B_local, bjo)
    sch.hide_buffer_access(blk_inner, "read", [1, 4])
    sch.tensorize(sch.get_loops(blk_inner)[-3], "wmma_sync")
    sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b_shared")
    sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a_global")
    sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
    sch.tensorize(sch.get_loops(init_blk)[-2], "wmma_fill")
    # schedule B_shared
    ax0, ax1 = sch.get_loops(B_shared)[-2:]
    fused_ax = sch.fuse(ax0, ax1)
    ax0, ax1, ax2, ax3 = sch.split(fused_ax, [None, 2, 32, 4])
    sch.vectorize(ax3)
    sch.bind(ax2, "threadIdx.x")
    sch.bind(ax1, "threadIdx.y")
    sch.unroll(ax0)

    mod = lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)

    f = tvm.build(mod["main"], target="cuda")

    ctx = tvm.cuda(0)
    row_indices_nd = tvm.nd.array(row_indices.astype("int32"), device=ctx)
    A_indptr = tvm.nd.array(unique_indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(bsr_mat.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(bsr_mat.data.reshape(-1).astype("float16"), device=ctx)
    X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(
        np.zeros((mb * block_size * feat_size), dtype="float16"), device=ctx
    )
    args = [A_data, X_nd, Y_nd, row_indices_nd, A_indptr, A_indices]
    f(*args)

    avg_time = profile_tvm_ms(f, args)
    print("dbsrmm time: \t{:.5f}ms".format(avg_time))
    return avg_time


def bench_cublas(W: th.Tensor, X: th.Tensor):
    with th.no_grad():
        W = W.half().to(0)
        X = X.to(0)

        measure = profile_pytorch_ms(lambda: W @ X)

        print("cublas time: \t{:.5f}ms".format(measure))
        return measure


def bench_cusparse(csr: Any, X: th.Tensor):
    W = th.sparse_csr_tensor(
        csr.indptr, csr.indices, csr.data, size=csr.shape, dtype=th.float16
    ).to(0)
    with th.no_grad():
        W = W.half().to(0)
        X = X.to(0)
        measure = profile_pytorch_ms(lambda: W @ X)

        print("cusparse time: \t{:.5f}ms".format(measure))
        return measure


def bench_triton(bsr: Any, X: th.Tensor, block_size: int):
    indptr = bsr.indptr
    indices = bsr.indices
    M, K = bsr.shape
    mb = M // bsr.blocksize[0]
    kb = K // bsr.blocksize[1]
    N = X.shape[1]
    rows = []
    cols = []
    for i in range(mb):
        row = i
        for j in range(indptr[i], indptr[i + 1]):
            col = indices[j]
            rows.append(row)
            cols.append(col)

    rows = th.tensor(rows, dtype=th.long)
    cols = th.tensor(cols, dtype=th.long)

    mask = th.zeros(1, mb, kb, dtype=th.int32)
    mask[0, rows, cols] = 1
    mask = mask.to(0)

    a_tri = triton.testing.sparsify_tensor(
        th.rand(1, 1, M, K, dtype=th.float16).to(0), mask, block_size
    )
    b_tri = X.view(1, 1, K, N).to(0)
    op = triton.ops.blocksparse.matmul(
        mask, block_size, "dsd", trans_a=False, trans_b=False, device="cuda"
    )
    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)

    measure = profile_pytorch_ms(lambda: op(a_tri, b_tri))
    print("triton time: \t{:.5f}ms".format(measure))
    return measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser("structured prunned bert")
    parser.add_argument("--dim", "-d", type=int, default=128, help="feature size")
    parser.add_argument(
        "--csv", "-c", action="store_true", help="whether to dump csv file or not"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )

    # block_size = 32
    sparsetir_durs = []
    sparsetir_dbsrmm_durs = []
    cublas_durs = []
    cusparse_durs = []
    triton_durs = []
    densities = []
    for name, param in model.named_parameters():
        if (
            name.endswith("key.weight")
            or name.endswith("value.weight")
            or name.endswith("query.weight")
            or name.endswith("dense.weight")
        ):
            # # test
            # print(f"name: {name} param: {param}")
            # print(f"param.detach().numpy(): {param.detach().numpy()}")
            # sys.exit(-1)
            # # end test

            bsr_weight = sp.bsr_matrix(
                param.detach().numpy(), shape=param.shape, blocksize=(32, 32)
            )

            csr_weight = sp.csr_matrix(param.detach().numpy())
            x = th.rand(csr_weight.shape[1], args.dim).half()
            densities.append(csr_weight.nnz / param.numel())
            print("--------------------")
            print("density:\t{:.3f}".format(densities[-1]))
            sparsetir_durs.append(bench_bsrmm(bsr_weight, x, 32))
            # sparsetir_dbsrmm_durs.append(bench_dbsrmm(bsr_weight, x, 32))
            # cublas_durs.append(bench_cublas(param.data, x))
            # cusparse_durs.append(bench_cusparse(csr_weight, x))
            # triton_durs.append(bench_triton(bsr_weight, x, 32))

    print(
        sum(sparsetir_durs),
        # sum(sparsetir_dbsrmm_durs),
        # sum(cublas_durs),
        # sum(cusparse_durs),
        # sum(triton_durs),
    )
    if args.csv:
        pd = pd.DataFrame(
            data={
                "density": densities,
                "sparsetir_dur": sparsetir_durs,
                # "sparsetir_dbsrmm_dur": sparsetir_dbsrmm_durs,
                # "cublas_dur": cublas_durs,
                # "cusparse_dur": cusparse_durs,
                # "triton_dur": triton_durs,
            }
        )
        pd.to_csv("structured_single_op.csv", index=False)
