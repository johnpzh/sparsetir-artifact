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
import sys
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset
from utils import get_dataset
from sparsetir_artifact import profile_tvm_ms


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
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(coarsening_factor)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = T.float32(0)
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


def bench_naive(
    g,
    x,
    y_golden,
    feat_size=128,
    coarsening_factor=2,
):
    indptr, indices, _ = g.adj_tensors("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    if feat_size < 64:
        coarsening_factor = 1
    mod = tvm.IRModule.from_expr(csrmm)
    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coarsening_factor,  # coarsening factor
    }

    mod["main"] = mod["main"].specialize(param_map)

    # schedule
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    outer_blk = sch.get_block("csrmm0")
    inner_blk = sch.get_block("csrmm1")
    (i,) = sch.get_loops(outer_blk)
    j, foo, foi, fi = sch.get_loops(inner_blk)
    sch.reorder(foo, fi, j, foi)
    sch.bind(fi, "threadIdx.x")
    sch.bind(foo, "blockIdx.y")
    sch.unroll(foi)
    io, ii = sch.split(i, [None, 8])
    sch.bind(io, "blockIdx.x")
    sch.bind(ii, "threadIdx.y")
    init_blk = sch.decompose_reduction(inner_blk, fi)
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    sch.bind(ax0, "threadIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")
    # prepare nd array
    indptr_nd = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    indices_nd = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    a_nd = tvm.nd.array(np.ones((nnz,)).astype("float32"), device=tvm.cuda(0))
    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd]
    f(*args)
    tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
    dur = profile_tvm_ms(f, args)
    print("tir naive time: {:.5f} ms".format(dur))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default="arxiv", help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)

    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size = ", feat_size)
        try:
            x = th.rand((g.num_src_nodes(), feat_size))
            y_golden = dgl.ops.copy_u_sum(g, x)
            
            bench_naive(
                g,
                x,
                y_golden,
                feat_size=feat_size,
                coarsening_factor=2,
            )
        except Exception as e:
            print("OOM")
            print(e, file=sys.stderr)