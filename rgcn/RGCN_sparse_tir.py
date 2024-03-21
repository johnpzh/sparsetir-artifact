import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tvm
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import (
    lower_sparse_iter,
    lower_sparse_buffer,
    FormatRewriteRule,
    format_decompose,
    csf_to_ell3d,
)
from torch.utils.dlpack import to_dlpack as th_to_dlpack
from torch.utils.dlpack import from_dlpack as th_from_dlpack

cached_kernel = None


def rgcn_hetero_forward(
    m: int, n: int, num_rels: int, feat_in: int, feat_out: int, nnz_i: int, nnz_j: int
):
    @T.prim_func
    def func(
        a: T.handle,
        w: T.handle,
        x: T.handle,
        y: T.handle,
        indptr_i: T.handle,
        indices_i: T.handle,
        indptr_j: T.handle,
        indices_j: T.handle,
    ):
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        R = T.dense_fixed(num_rels)
        I = T.sparse_variable(R, (m, nnz_i), (indptr_i, indices_i), "int32")
        J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
        I_detach = T.dense_fixed(m)
        J_detach = T.dense_fixed(n)
        F_in = T.dense_fixed(feat_in)
        F_out = T.dense_fixed(feat_out)
        A = T.match_sparse_buffer(a, (R, I, J), "float32")
        W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
        X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
        Y = T.match_sparse_buffer(y, (I_detach, F_out), "float32")
        with T.iter([F_out, R, I, J, F_in], "SSSRR", "rgcn-hetero-forward") as [
            fo,
            r,
            i,
            j,
            fi,
        ]:
            with T.init():
                Y[i, fo] = 0.0
            Y[i, fo] = Y[i, fo] + A[r, i, j] * W[r, fo, fi] * X[j, fi]

    return func


def ell3d(d0: int, d1: int, d2: int, nnz: int, nnz_rows: int, nnz_cols: int):
    @T.prim_func
    def func(
        a: T.handle,
        indptr_io: T.handle,
        indices_ii: T.handle,
        indices_j: T.handle,
    ) -> None:
        R = T.dense_fixed(d0, idtype="int32")
        IO = T.dense_variable(R, (d1, nnz), indptr_io, idtype="int32")
        II = T.sparse_fixed(IO, (d2, nnz_rows), indices_ii, idtype="int32")
        J = T.sparse_fixed(II, (d2, nnz_cols), indices_j, idtype="int32")
        A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float32")
        T.evaluate(0)

    return func


def rgcn_hetero_forward_tc(m, n, num_rels, feat_in, feat_out, dtype):
    @T.prim_func
    def func(
        a: T.handle,
        w: T.handle,
        x: T.handle,
        y: T.handle,
        wx: T.handle,
        indptr_i: T.handle,
        indices_i: T.handle,
        indptr_j: T.handle,
        indices_j: T.handle,
        nnz_i: T.int32,
        nnz_j: T.int32,
    ):
        T.func_attr(
            {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2}
        )
        R = T.dense_fixed(num_rels)
        I = T.sparse_variable(R, (m, nnz_i), (indptr_i, indices_i), "int32")
        J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
        I_detach = T.dense_fixed(m)
        J_detach = T.dense_fixed(n)
        F_in = T.dense_fixed(feat_in)
        F_out = T.dense_fixed(feat_out)
        A = T.match_sparse_buffer(a, (R, I, J), "float16")
        W = T.match_sparse_buffer(w, (R, F_in, F_out), "float16")
        X = T.match_sparse_buffer(x, (J_detach, F_in), "float16")
        Y = T.match_sparse_buffer(y, (I_detach, F_out), dtype)
        WX = T.match_sparse_buffer(wx, (R, I, J, F_out), dtype)

        with T.iter([R, I, J, F_out, F_in], "SSSSR", "rgcn-hetero-forward_wx") as [
            r,
            i,
            j,
            fo,
            fi,
        ]:
            with T.init():
                WX[r, i, j, fo] = T.cast(0, dtype)
            WX[r, i, j, fo] += T.cast(X[j, fi] * W[r, fi, fo], dtype)

        with T.iter([R, I, J, F_out], "SSRS", "rgcn-hetero-forward") as [r, i, j, fo]:
            with T.init():
                Y[i, fo] = T.cast(0, dtype)
            Y[i, fo] = Y[i, fo] + T.cast(A[r, i, j], dtype) * WX[r, i, j, fo]

    return func


def ell3d_fp16(
    d0: int,
    d1: int,
    d2: int,
    nnz: int,
    nnz_rows: int,
    nnz_cols: int,
    feat_out: int,
    dtype: str,
):
    @T.prim_func
    def func(
        a: T.handle,
        wx: T.handle,
        indptr_io: T.handle,
        indices_ii: T.handle,
        indices_j: T.handle,
    ) -> None:
        R = T.dense_fixed(d0, idtype="int32")
        IO = T.dense_variable(R, (d1, nnz), indptr_io, idtype="int32")
        II = T.sparse_fixed(IO, (d2, nnz_rows), indices_ii, idtype="int32")
        J = T.sparse_fixed(II, (d2, nnz_cols), indices_j, idtype="int32")
        FO = T.dense_fixed(feat_out, idtype="int32")
        A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float16")
        WX = T.match_sparse_buffer(wx, (R, IO, II, J, FO), dtype=dtype)
        T.evaluate(0)

    return func


def prepare_hetero_graph_simplified(g):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": torch.IntTensor(ntype_pointer),
        "etype_edge_pointer": torch.IntTensor(etype_pointer),
    }


# V100
config_dict = {
    "aifb": (2, 32),
    "mutag": (8, 256),
    "bgs": (4, 256),
    "biokg": (4, 512),
    "am": (8, 512),
}

# GTX 3070
# config_dict = {
#     'aifb': (2, 32),
#     'mutag': (4, 512),
#     'bgs': (4, 256),
#     'biokg': (4, 512),
#     'am': (4, 1024),
# }


def simplify(script: tvm.tir.PrimFunc) -> tvm.tir.PrimFunc:
    return tvm.tir.transform.Simplify()(tvm.IRModule.from_expr(script))["main"]


def wmma_sync(d0: int, d1: int, dtype: str):
    @T.prim_func
    def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_a",
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=1,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(d0, d1, 16, 16):
                with T.block("update"):
                    vio, vii, vj, vk = T.axis.remap("SSSR", [io, ii, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[vio, vii, vj] = C_frag[vio, vii, vj] + T.cast(
                        A_frag[vio, vii, vk] * B_frag[vk, vj], dtype
                    )

    @T.prim_func
    def wmma_sync_16_1_desc(
        a_frag: T.handle, b_frag: T.handle, c_frag: T.handle
    ) -> None:
        A_frag = T.match_buffer(
            a_frag,
            (16, 1, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_a",
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        C_frag = T.match_buffer(
            c_frag,
            (16, 1, 16),
            dtype,
            align=64,
            offset_factor=1,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(16, 1, 16, 16):
                with T.block("update"):
                    vio, vj, vk = T.axis.remap("SSR", [io, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[vio, 0, vj] = C_frag[vio, 0, vj] + T.cast(
                        A_frag[vio, 0, vk] * B_frag[vk, vj], dtype
                    )

    @T.prim_func
    def wmma_sync_1_16_desc(
        a_frag: T.handle, b_frag: T.handle, c_frag: T.handle
    ) -> None:
        A_frag = T.match_buffer(
            a_frag,
            (1, 16, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_a",
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=1,
            scope="wmma.matrix_b",
        )
        C_frag = T.match_buffer(
            c_frag,
            (1, 16, 16),
            dtype,
            align=64,
            offset_factor=1,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(1, 16, 16, 16):
                with T.block("update"):
                    vii, vj, vk = T.axis.remap("SSR", [ii, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[0, vii, vj] = C_frag[0, vii, vj] + T.cast(
                        A_frag[0, vii, vk] * B_frag[vk, vj], dtype
                    )

    @T.prim_func
    def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
        )
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )

        with T.block("root"):
            T.reads(
                [
                    C_frag[0:d0, 0:d1, 0:16],
                    A_frag[0:d0, 0:d1, 0:16],
                    B_frag[0:16, 0:16],
                ]
            )
            T.writes(C_frag[0:d0, 0:d1, 0:16])
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

    if d0 == 1:
        return simplify(wmma_sync_1_16_desc), wmma_sync_impl
    elif d1 == 1:
        return simplify(wmma_sync_16_1_desc), wmma_sync_impl
    else:
        return simplify(wmma_sync_desc), wmma_sync_impl


def wmma_load_a(d0: int, d1: int, scope: str):
    @T.prim_func
    def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(
            a, (d0, d1, 16), "float16", align=64, offset_factor=16, scope=scope
        )
        A_frag = T.match_buffer(
            a_frag,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("load"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    A_frag[vio, vii, vj] = A[vio, vii, vj]

    @T.prim_func
    def wmma_load_a_16_1_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(
            a, (16, 1, 16), "float16", align=64, offset_factor=16, scope=scope
        )
        A_frag = T.match_buffer(
            a_frag,
            (16, 1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("load"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    A_frag[vio, 0, vj] = A[vio, 0, vj]

    @T.prim_func
    def wmma_load_a_1_16_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(
            a, (1, 16, 16), "float16", align=64, offset_factor=16, scope=scope
        )
        A_frag = T.match_buffer(
            a_frag,
            (1, 16, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("load"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    A_frag[0, vii, vj] = A[0, vii, vj]

    @T.prim_func
    def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        A = T.match_buffer(
            a,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope=scope,
            strides=[s0, s1, s2],
        )
        A_frag = T.match_buffer(
            a_frag,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_a",
        )

        with T.block("root"):
            T.reads(A[0:d0, 0:d1, 0:16])
            T.writes(A_frag[0:d0, 0:d1, 0:16])
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
                        s1,
                        "row_major",
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_load_a_1_16_desc, wmma_load_a_impl
    elif d1 == 1:
        return wmma_load_a_16_1_desc, wmma_load_a_impl
    else:
        return wmma_load_a_desc, wmma_load_a_impl


def wmma_load_b(scope: str):
    @T.prim_func
    def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
        B = T.match_buffer(
            b, (16, 16), "float16", align=64, offset_factor=16, scope=scope
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope="wmma.matrix_b",
        )
        with T.block("root"):
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_frag[vi, vj] = B[vi, vj]

    @T.prim_func
    def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        B = T.match_buffer(
            b,
            (16, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope=scope,
            strides=[s0, s1],
        )
        B_frag = T.match_buffer(
            b_frag,
            (16, 16),
            "float16",
            align=64,
            offset_factor=16,
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


def wmma_fill(d0: int, d1: int, dtype: str):
    @T.prim_func
    def wmma_fill_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("init"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    C_frag[vio, vii, vj] = T.cast(0, dtype)

    @T.prim_func
    def wmma_fill_16_1_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (16, 1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("init"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    C_frag[vio, 0, vj] = T.cast(0, dtype)

    @T.prim_func
    def wmma_fill_1_16_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (1, 16, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("init"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    C_frag[0, vii, vj] = T.cast(0, dtype)

    @T.prim_func
    def wmma_fill_impl(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        with T.block("root"):
            T.reads([])
            T.writes(C_frag[0:d0, 0:d1, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_fill_fragment(
                        C_frag.data,
                        16,
                        16,
                        16,
                        C_frag.elem_offset // 256
                        + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                        T.cast(0, dtype),
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_fill_1_16_desc, wmma_fill_impl
    elif d1 == 1:
        return wmma_fill_16_1_desc, wmma_fill_impl
    else:
        return wmma_fill_desc, wmma_fill_impl


def wmma_store(d0: int, d1: int, scope: str, dtype: str):
    @T.prim_func
    def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (d0, d1, 16), dtype, align=64, offset_factor=16, scope=scope
        )
        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("store"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    C[vio, vii, vj] = C_frag[vio, vii, vj]

    @T.prim_func
    def wmma_store_desc_16_1(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (16, 1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (16, 1, 16), dtype, align=64, offset_factor=16, scope=scope
        )
        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("store"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    C[vio, 0, vj] = C_frag[vio, 0, vj]

    @T.prim_func
    def wmma_store_desc_1_16(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag,
            (1, 16, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c, (1, 16, 16), dtype, align=64, offset_factor=16, scope=scope
        )
        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("store"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    C[0, vii, vj] = C_frag[0, vii, vj]

    @T.prim_func
    def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        C_frag = T.match_buffer(
            c_frag,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope="wmma.accumulator",
        )
        C = T.match_buffer(
            c,
            (d0, d1, 16),
            dtype,
            align=64,
            offset_factor=16,
            scope=scope,
            strides=[s0, s1, s2],
        )
        with T.block("root"):
            T.reads(C_frag[0:d0, 0:d1, 0:16])
            T.writes(C[0:d0, 0:d1, 0:16])
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
                        s1,
                        "row_major",
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_store_desc_1_16, wmma_store_impl
    elif d1 == 1:
        return wmma_store_desc_16_1, wmma_store_impl
    else:
        return wmma_store_desc, wmma_store_impl


def convert_indptr_to_mid_array(indptr):
    indptr_numpy = indptr.numpy()
    ret = []
    for i in range(len(indptr_numpy) - 1):
        ret.append(
            np.zeros((indptr_numpy[i + 1] - indptr_numpy[i],), dtype=np.int32) + i
        )
    return np.concatenate(ret, axis=-1)


result_dtype = "float32"
# register wmma instructions
tir.TensorIntrin.register("wmma_{}_load_b".format("shared"), *wmma_load_b("shared"))

for bucket_size in [1, 2, 4, 8, 16]:
    d0 = 16 // bucket_size
    d1 = bucket_size
    tir.TensorIntrin.register(
        "wmma_{}_{}_{}_store".format(d0, d1, "shared"),
        *wmma_store(d0, d1, "shared", result_dtype),
    )
    tir.TensorIntrin.register(
        "wmma_{}_{}_{}_load_a".format(d0, d1, "shared"), *wmma_load_a(d0, d1, "shared")
    )
    tir.TensorIntrin.register(
        "wmma_{}_{}_init".format(d0, d1), *wmma_fill(d0, d1, result_dtype)
    )
    tir.TensorIntrin.register(
        "wmma_{}_{}_sync".format(d0, d1), *wmma_sync(d0, d1, result_dtype)
    )


def tc_csf_to_ell3d_inv_idx_map(r, io, ii, j, fo):
    return r, ii, j, fo


def tc_csf_to_ell3d_idx_map(r, i, j, fo):
    return r, 0, i, j, fo


def create_tensorcores_kernel(g, dataset, W, in_feat, out_feat):
    fo_factor = 2
    buckets = [1, 2, 4, 8, 16]
    g = g.cpu()
    type_pointers = prepare_hetero_graph_simplified(g)

    ty = 4
    num_workloads_per_thread = 4
    fi_factor = in_feat // 16
    group_size = ty * num_workloads_per_thread * 16
    # preprocess data
    ntype_node_pointer = type_pointers["ntype_node_pointer"]
    etype_edge_pointer = type_pointers["etype_edge_pointer"]
    csf_indptr_0 = [0]
    csf_indices_0 = []
    csf_indptr_1 = [torch.tensor([0], dtype=torch.int32)]
    csf_indices_1 = []
    num_rels = len(g.canonical_etypes)
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        m_sub, n_sub = g_sub.num_dst_nodes(), g_sub.num_src_nodes()
        indptr, indices, _ = g_sub.adj_tensors(fmt="csc")
        csf_indptr_0.append(csf_indptr_0[-1] + m_sub)
        csf_indices_0.append(
            ntype_node_pointer[dst_type_id] + torch.arange(m_sub, dtype=torch.int32)
        )
        csf_indptr_1.append(csf_indptr_1[-1][-1] + indptr[1:])
        csf_indices_1.append(ntype_node_pointer[src_type_id] + indices)

    csf_indptr_0 = torch.tensor(csf_indptr_0, dtype=torch.int32)
    csf_indices_0 = torch.cat(csf_indices_0, dim=-1)
    csf_indptr_1 = torch.cat(csf_indptr_1, dim=-1)
    csf_indices_1 = torch.cat(csf_indices_1, dim=-1)

    dev = tvm.cpu(0)
    csf_indptr_0_nd = tvm.nd.array(csf_indptr_0.int(), device=dev)
    csf_indices_0_nd = tvm.nd.array(csf_indices_0.int(), device=dev)
    csf_indptr_1_nd = tvm.nd.array(csf_indptr_1.int(), device=dev)
    csf_indices_1_nd = tvm.nd.array(csf_indices_1.int(), device=dev)
    buckets_row = [group_size // _ for _ in buckets]

    indptr, row_indices, col_indices, mask = csf_to_ell3d(
        csf_indptr_0_nd,
        csf_indices_0_nd,
        csf_indptr_1_nd,
        csf_indices_1_nd,
        buckets_row,
        buckets,
    )
    mids = list(map(convert_indptr_to_mid_array, indptr))

    rewrites = []
    for bucket_id, bucket_size in enumerate(buckets):
        rewrites.append(
            FormatRewriteRule(
                str(bucket_id),
                ell3d_fp16(
                    num_rels,
                    m,
                    n,
                    row_indices[bucket_id].shape[0],
                    group_size // bucket_size,
                    bucket_size,
                    out_feat,
                    result_dtype,
                ),
                ["A", "WX"],
                ["R", "I", "J", "F_out"],
                ["R", "IO", "II", "J", "FO"],
                {"R": ["R"], "I": ["IO", "II"], "J": ["J"], "F_out": ["FO"]},
                tc_csf_to_ell3d_idx_map,
                tc_csf_to_ell3d_inv_idx_map,
            )
        )
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward_tc(
            m, n, num_rels, in_feat, out_feat, result_dtype
        ).with_attr("horizontal_fuse", True)
    )
    mod = format_decompose(mod, rewrites, include_format_rewrite_blks=False)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, _ in enumerate(buckets):
        sp_iteration = sch.get_sparse_iteration(
            "rgcn-hetero-forward_wx_{}".format(bucket_id)
        )
        r, io, ii, j, fo, fi = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [r, io])
        sp_iteration = sch.get_sparse_iteration(
            "rgcn-hetero-forward_{}".format(bucket_id)
        )
        r, io, ii, j, fo = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [r, io])
    mod = lower_sparse_iter(sch.mod)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, bucket_size in enumerate(buckets):
        sch.set_block_filter("group_{}".format(bucket_id))
        d0 = 16 // bucket_size
        d1 = bucket_size
        blk_wx = sch.get_block("rgcn-hetero-forward_wx_{}0".format(bucket_id))
        blk = sch.get_block("rgcn-hetero-forward_{}0".format(bucket_id))
        sch.annotate(blk_wx, "group_{}".format(bucket_id), 1)
        sch.annotate(blk, "group_{}".format(bucket_id), 1)
        sch.match_to_alloc(blk_wx, 0)
        sch.set_scope(blk_wx, 0, "shared")
        i, j, k, fo, fi = sch.get_loops(blk_wx)[-5:]
        fooo, fooi, foi = sch.split(fo, [None, fo_factor, 16])
        fio, fii = sch.split(fi, [fi_factor, 16])
        sch.reorder(i, fooo, fooi, fio, j, k, foi, fii)
        sch.annotate(fooo, "pragma_unroll_explicit", 0)
        sch.unroll(fooo)
        sch.reverse_compute_at(blk, fooo, True)
        W_shared = sch.reverse_cache_read(blk_wx, 2, "shared", [0, 1, 5, 4])
        sch.compute_at(W_shared, fooo, True)
        j_unroll, j_ty, j_inner = sch.split(j, [num_workloads_per_thread, ty, None])
        sch.reorder(j_unroll, j_ty, fooi, fio, j_inner, k, foi, fii)
        sch.unroll(fooi)
        sch.unroll(fio)
        X_shared = sch.reverse_cache_read(blk_wx, 0, "shared")
        sch.compute_at(X_shared, fio, True)
        WX_accum = sch.reverse_cache_write(blk_wx, 0, "wmma.accumulator")
        sch.reverse_compute_at(WX_accum, fio, True)
        W_wmma = sch.reverse_cache_read(blk_wx, 2, "wmma.matrix_b", [0, 1, 5, 4])
        sch.compute_at(W_wmma, fio, True)
        X_wmma = sch.reverse_cache_read(blk_wx, 0, "wmma.matrix_a")
        sch.bind(sch.get_loops(blk)[0], "blockIdx.x")
        sch.bind(j_ty, "threadIdx.y")
        sch.unroll(j_unroll)
        init_blk = sch.decompose_reduction(blk_wx, fio)
        sch.reverse_compute_at(blk, j_ty, True)
        Y_local = sch.reverse_cache_write(blk, 0, "local")
        sch.annotate(Y_local, "atomic", True)
        sch.reverse_compute_at(Y_local, sch.get_loops(blk)[-3], True)

        # tensorize
        sch.tensorize(
            sch.get_loops(WX_accum)[-3], "wmma_{}_{}_{}_store".format(d0, d1, "shared")
        )
        sch.tensorize(
            sch.get_loops(X_wmma)[-3], "wmma_{}_{}_{}_load_a".format(d0, d1, "shared")
        )
        sch.tensorize(sch.get_loops(W_wmma)[-2], "wmma_{}_load_b".format("shared"))
        sch.tensorize(
            sch.get_loops(init_blk)[-3],
            "wmma_{}_{}_init".format(d0, d1),
        )
        sch.hide_buffer_access(blk_wx, "read", [2, 4])
        sch.tensorize(sch.get_loops(blk_wx)[-4], "wmma_{}_{}_sync".format(d0, d1))

        # schedule W_shared
        fi, fo = sch.get_loops(W_shared)[-2:]
        fused_ax = sch.fuse(fi, fo)
        j, k, fi, fo = sch.split(fused_ax, [None, ty, 32, 8])
        sch.vectorize(fo)
        sch.bind(fi, "threadIdx.x")
        sch.bind(k, "threadIdx.y")
        sch.unroll(j)

        # schedule X_shared
        ax0, ax1, ax2 = sch.get_loops(X_shared)[-3:]
        fused_ax = sch.fuse(ax0, ax1, ax2)
        ax0, ax1, ax2 = sch.split(fused_ax, [None, 32, 8])
        sch.vectorize(ax2)
        sch.bind(ax1, "threadIdx.x")
        sch.unroll(ax0)

        # schedule for the write block
        j_i, k, fo = sch.get_loops(blk)[-3:]
        sch.unroll(j_i)
        ax0, ax1 = sch.split(fo, [None, 32])
        sch.unroll(ax0)
        sch.bind(ax1, "threadIdx.x")
        sch.unroll(k)
        ax0, ax1 = sch.split(sch.get_loops(Y_local)[-1], [None, 32])
        sch.unroll(ax0)
        sch.bind(ax1, "threadIdx.x")

        # Unset block filter
        sch.unset_block_filter()

    mod = lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)

    f = tvm.build(mod["main"], target="cuda")

    W_nd = tvm.nd.from_dlpack(th_to_dlpack(W.half().view(-1).contiguous()))
    double_buffer = [
        torch.zeros((g.num_dst_nodes(), out_feat), device=0) for _ in range(2)
    ]
    counter = 0
    post_args = []
    dev = tvm.cuda(0)
    for bucket_id, _ in enumerate(buckets):
        post_args.append(
            tvm.nd.array(
                mask[bucket_id].numpy().reshape(-1).astype(np.float16), device=dev
            )
        )
        post_args.append(
            tvm.nd.array(row_indices[bucket_id].numpy().reshape(-1), device=dev)
        )
        post_args.append(
            tvm.nd.array(col_indices[bucket_id].numpy().reshape(-1), device=dev)
        )
    for bucket_id, _ in enumerate(buckets):
        post_args.append(tvm.nd.array(mids[bucket_id], device=dev))

    def foo(X):
        nonlocal counter
        nonlocal double_buffer
        Y = double_buffer[counter % 2]
        Y.zero_()
        counter += 1
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        args = [W_nd, X_nd, Y_nd] + post_args
        f(*args)
        return Y

    return foo


def csf_to_ell3d_inv_idx_map(r, io, ii, j):
    return r, ii, j


def csf_to_ell3d_idx_map(r, i, j):
    return r, 0, i, j


def create_composable_kernel(g, dataset, W, in_feat, out_feat):
    split_factor_f = 16
    group_size = 32
    buckets = [1, 2, 4, 8, 16]
    g = g.cpu()
    type_pointers = prepare_hetero_graph_simplified(g)
    ntype_node_pointer = type_pointers["ntype_node_pointer"]
    etype_edge_pointer = type_pointers["etype_edge_pointer"]
    csf_indptr_0 = [0]
    csf_indices_0 = []
    csf_indptr_1 = [torch.tensor([0], dtype=torch.int32)]
    csf_indices_1 = []
    num_rels = len(g.canonical_etypes)
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        m_sub, n_sub = g_sub.num_dst_nodes(), g_sub.num_src_nodes()
        indptr, indices, _ = g_sub.adj_tensors(fmt="csc")
        csf_indptr_0.append(csf_indptr_0[-1] + m_sub)
        csf_indices_0.append(
            ntype_node_pointer[dst_type_id] + torch.arange(m_sub, dtype=torch.int32)
        )
        csf_indptr_1.append(csf_indptr_1[-1][-1] + indptr[1:])
        csf_indices_1.append(ntype_node_pointer[src_type_id] + indices)

    csf_indptr_0 = torch.tensor(csf_indptr_0, dtype=torch.int32)
    csf_indices_0 = torch.cat(csf_indices_0, dim=-1)
    csf_indptr_1 = torch.cat(csf_indptr_1, dim=-1)
    csf_indices_1 = torch.cat(csf_indices_1, dim=-1)

    dev = tvm.cpu(0)
    csf_indptr_0_nd = tvm.nd.array(csf_indptr_0.int(), device=dev)
    csf_indices_0_nd = tvm.nd.array(csf_indices_0.int(), device=dev)
    csf_indptr_1_nd = tvm.nd.array(csf_indptr_1.int(), device=dev)
    csf_indices_1_nd = tvm.nd.array(csf_indices_1.int(), device=dev)
    buckets_row = [group_size // _ for _ in buckets]

    indptr, row_indices, col_indices, mask = csf_to_ell3d(
        csf_indptr_0_nd,
        csf_indices_0_nd,
        csf_indptr_1_nd,
        csf_indices_1_nd,
        buckets_row,
        buckets,
    )
    mids = list(map(convert_indptr_to_mid_array, indptr))

    rewrites = []
    for bucket_id, bucket_size in enumerate(buckets):
        rewrites.append(
            FormatRewriteRule(
                str(bucket_id),
                ell3d(
                    num_rels,
                    m,
                    n,
                    row_indices[bucket_id].shape[0],
                    group_size // bucket_size,
                    bucket_size,
                ),
                ["A"],
                ["R", "I", "J"],
                ["R", "IO", "II", "J"],
                {"R": ["R"], "I": ["IO", "II"], "J": ["J"]},
                csf_to_ell3d_idx_map,
                csf_to_ell3d_inv_idx_map,
            )
        )
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward(m, n, num_rels, in_feat, out_feat, 0, 0).with_attr(
            "horizontal_fuse", True
        )
    )
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, _ in enumerate(buckets):
        sp_iteration = sch.get_sparse_iteration(
            "rgcn-hetero-forward_{}".format(bucket_id)
        )
        fo, r, io, ii, j, fi = sch.get_sp_iters(sp_iteration)
        sch.sparse_reorder(sp_iteration, [r, io, ii, j, fo, fi])
        sch.sparse_fuse(sp_iteration, [r, io])
    mod = lower_sparse_iter(sch.mod)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, bucket_size in enumerate(buckets):
        blk = sch.get_block("rgcn-hetero-forward_{}0".format(bucket_id))
        io, ii, j, fo, fi = sch.get_loops(blk)
        foo, foi = sch.split(fo, [split_factor_f, None])
        sch.reorder(io, ii, foo, foi, j, fi)
        blk_outer, blk_inner = sch.blockize(j, True), blk
        read_W = sch.cache_read(blk_inner, 2, "local")
        write_Y = sch.cache_write(blk_inner, 0, "local")
        sch.annotate(write_Y, "atomic", True)
        sch.bind(fi, "threadIdx.x")
        sch.bind(sch.get_loops(read_W)[-1], "threadIdx.x")
        sch.unroll(j)
        sch.bind(foi, "threadIdx.y")
        sch.bind(io, "blockIdx.x")
        sch.unroll(ii)
        sch.unroll(foo)

    mod = lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod["main"], target="cuda")

    W_nd = tvm.nd.from_dlpack(th_to_dlpack(W.view(-1).contiguous()))
    double_buffer = [
        torch.zeros((g.num_dst_nodes(), out_feat), device=0) for _ in range(2)
    ]
    counter = 0
    post_args = []
    dev = tvm.cuda(0)
    for bucket_id, _ in enumerate(buckets):
        post_args.append(
            tvm.nd.array(
                mask[bucket_id].numpy().reshape(-1).astype(np.float32), device=dev
            )
        )
        post_args.append(
            tvm.nd.array(row_indices[bucket_id].numpy().reshape(-1), device=dev)
        )
        post_args.append(
            tvm.nd.array(col_indices[bucket_id].numpy().reshape(-1), device=dev)
        )
    for bucket_id, _ in enumerate(buckets):
        post_args.append(tvm.nd.array(mids[bucket_id], device=dev))

    def foo(X):
        nonlocal counter
        nonlocal double_buffer
        Y = double_buffer[counter % 2]
        Y.zero_()
        counter += 1
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        args = [W_nd, X_nd, Y_nd] + post_args
        f(*args)
        return Y

    return foo


def create_naive_kernel(g, dataset, W, in_feat, out_feat):
    g = g.cpu()
    type_pointers = prepare_hetero_graph_simplified(g)
    g.ntype_pointer = type_pointers["ntype_node_pointer"]
    g.etype_pointer = type_pointers["etype_edge_pointer"]

    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    out = np.zeros((m * out_feat))
    A = tvm.nd.array(np.ones((nnz_j,), dtype=np.float32), device=tvm.cuda(0))

    indptr_i = [torch.LongTensor([0])]
    indices_i = []
    indptr_j = [torch.LongTensor([0])]
    indices_j = []
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_tensors(fmt="csc")

        unique_nodes = torch.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        indptr_i.append(torch.LongTensor([len(unique_nodes) + indptr_i[-1].item()]))
        indices_i.append(unique_nodes + g.ntype_pointer[dst_type_id])
        indptr_j.append(indptr[unique_nodes + 1] + g.etype_pointer[etype_id])
        indices_j.append(indices + g.ntype_pointer[src_type_id])

    indptr_i = tvm.nd.array(
        torch.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0)
    )
    indices_i = tvm.nd.array(
        torch.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0)
    )
    indptr_j = tvm.nd.array(
        torch.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0)
    )
    indices_j = tvm.nd.array(
        torch.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0)
    )

    nnz_i = indices_i.shape[0]
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward(m, n, r, in_feat, out_feat, nnz_i, nnz_j)
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)
    # NOTE(Zihao): We might want to tune
    split_factor_f = 2

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 3, "local")
    write_blk = sch.cache_write(blk2, 0, "local")
    sch.annotate(write_blk, "atomic", True)
    fo, r = sch.get_loops(blk0)
    foo, foi = sch.split(fo, [split_factor_f, None])
    (i,) = sch.get_loops(blk1)
    (io, ii) = sch.split(i, [256, None])
    j, f_in = sch.get_loops(blk2)
    sch.reorder(f_in, j)
    sch.bind(r, "blockIdx.y")
    sch.bind(io, "blockIdx.x")
    sch.bind(foo, "blockIdx.z")
    sch.bind(f_in, "threadIdx.x")
    sch.bind(foi, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")
    # print(f.imported_modules[0].get_source())

    W_nd = tvm.nd.from_dlpack(th_to_dlpack(W.view(-1).contiguous()))
    double_buffer = [
        torch.zeros((g.num_dst_nodes(), out_feat), device=0) for _ in range(2)
    ]
    counter = 0

    def foo(X):
        nonlocal counter
        nonlocal double_buffer
        Y = double_buffer[counter % 2]
        Y.zero_()
        counter += 1
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        args = [A, W_nd, X_nd, Y_nd, indptr_i, indices_i, indptr_j, indices_j]
        f(*args)
        return Y

    return foo


class RGCNSparseTIRNaiveLayer(nn.Module):
    def __init__(self, dataset, in_feat, out_feat, num_rels) -> None:
        super().__init__()
        self.dataset = dataset
        self.in_feat = in_feat
        self.out_feat = out_feat
        dropout = 0.0

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.Tensor(num_rels, out_feat, in_feat))
        self.cached_kernel = None

    def forward(self, g, feat):
        if self.cached_kernel is None:
            self.cached_kernel = create_naive_kernel(
                g, self.dataset, self.W.data, self.in_feat, self.out_feat
            )
        h = self.cached_kernel(feat)
        return h


class RGCNSparseTIRComposableLayer(nn.Module):
    def __init__(self, dataset, in_feat, out_feat, num_rels) -> None:
        super().__init__()
        self.dataset = dataset
        self.in_feat = in_feat
        self.out_feat = out_feat
        dropout = 0.0

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.Tensor(num_rels, out_feat, in_feat))
        self.cached_kernel = None

    def forward(self, g, feat):
        if self.cached_kernel is None:
            self.cached_kernel = create_composable_kernel(
                g, self.dataset, self.W.data, self.in_feat, self.out_feat
            )
        h = self.cached_kernel(feat)
        return h


class RGCNSparseTIRTensorCoresLayer(nn.Module):
    def __init__(self, dataset, in_feat, out_feat, num_rels) -> None:
        super().__init__()
        self.dataset = dataset
        self.in_feat = in_feat
        self.out_feat = out_feat
        dropout = 0.0

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.Tensor(num_rels, out_feat, in_feat))
        self.cached_kernel = None

    def forward(self, g, feat):
        if self.cached_kernel is None:
            self.cached_kernel = create_tensorcores_kernel(
                g, self.dataset, self.W.data, self.in_feat, self.out_feat
            )
        h = self.cached_kernel(feat)
        return h


class RGCNSparseTIRNaive(nn.Module):
    def __init__(self, dataset, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNSparseTIRNaive, self).__init__()
        self.layer1 = RGCNSparseTIRNaiveLayer(dataset, in_dim, hidden_dim, num_rels)
        self.layer2 = RGCNSparseTIRNaiveLayer(dataset, hidden_dim, out_dim, num_rels)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class RGCNSparseTIRComposable(nn.Module):
    def __init__(self, dataset, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNSparseTIRComposable, self).__init__()
        self.layer1 = RGCNSparseTIRComposableLayer(
            dataset, in_dim, hidden_dim, num_rels
        )
        self.layer2 = RGCNSparseTIRComposableLayer(
            dataset, hidden_dim, out_dim, num_rels
        )

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class RGCNSparseTIRTensorCores(nn.Module):
    def __init__(self, dataset, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCNSparseTIRTensorCores, self).__init__()
        self.layer1 = RGCNSparseTIRTensorCoresLayer(
            dataset, in_dim, hidden_dim, num_rels
        )
        self.layer2 = RGCNSparseTIRTensorCoresLayer(
            dataset, hidden_dim, out_dim, num_rels
        )

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features.half()))
        x = self.layer2(g, x.half())
        return x
