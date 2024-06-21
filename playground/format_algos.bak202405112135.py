from format_matrix_market import MTX, Bucket
import numpy as np
import bisect
import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
import tvm.sparse
from sparsetir_artifact import profile_tvm_ms
import time
import sys

__all__ = ["build_hyb_format", "bench_hyb_with_config", "search_bucket_config"]

#####################
# SparseTIR settings
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

# End SparseTIR settings
#########################


# def get_max_bucket_width(buckets: dict):
#     return max(buckets.keys)


# def modify_max_bucket_width(buckets: dict,
#                            num_parts: int):
#     curr_max_width = get_max_bucket_width(buckets)
#     # get_bubble_overhead(buckets)
#     while curr_max_width < 1:
#         # Halve
#         curr_max_width /= 2


def build_list_of_list(bucket_widths):
    """Return an empty list of lists with shape (num_partitions, num_buckets), and the inner element is a list.

    Args:
        bucket_widths (list of lists): All bucket widths with shape (num_partitions, num_buckets). [ [b_0_0, b_0_1, b_0_2, ...], [b_1_0, b_1_1, b_1_2, ...], ... ], b_i_j the bucket width of bucket j in partition i.

    Returns:
        list of list: an empty list of lists with shape (num_partitions, num_buckets), and the inner element is a list.
    """
    # shape is [num_buckets, num_partitions]
    num_parts = len(bucket_widths)
    res = []
    for part_ind in range(num_parts):
        num_buckets = len(bucket_widths[part_ind])
        inner = [ [] for _ in range(num_buckets) ]
        res.append(inner)
    
    return res


def build_hyb_format(g: MTX,
                    #  num_parts,
                     bucket_widths):
    """build the hyb format based on the given bucket widths.

    Args:
        g (MTX): the matrix
        bucket_widths (list of lists): All bucket widths with shape (num_partitions, num_buckets). [ [b_0_0, b_0_1, b_0_2, ...], [b_1_0, b_1_1, b_1_2, ...], ... ], b_i_j the bucket width of bucket j in partition i.

    Returns:
        Tuple: a tuple of (row_indices_nd, col_indices_nd, mask_nd). row_indices_nd is with shape (num_partitions, num_buckets), and the inner element is a numpy ndarray. col_indices_nd and mask_nd are in the same shape.
    """
    num_parts = len(bucket_widths)
    num_rows = g.num_src_nodes()
    num_cols = g.num_dst_nodes()
    partition_size = (num_cols + num_parts - 1) // num_parts

    # # test
    # print(F"\npartition_size: {partition_size}")
    # # end test

    # Count the degree of each row in each partition
    # degree_counter shape is partition_size * num_rows
    degree_counter = [ [0] * num_rows for i in range(num_parts) ]
    for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
        part_ind = col_ind // partition_size
        degree_counter[part_ind][row_ind] += 1
    
    indptr, indices, _ = g.adj_tensors("csr")

    # num_bucket = len(bucket_widths)
    # 3-dimentional list
    # It is a list of list with shape [num_parts, num_bucket], where the innermost element is a list
    # The inner most element needs to be a numpy ndarray in the end.
    # row_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # col_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # mask = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    row_indices = build_list_of_list(bucket_widths)
    col_indices = build_list_of_list(bucket_widths)
    mask = build_list_of_list(bucket_widths)

    # for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
    for row_ind in range(num_rows):
        for col_loc in range(indptr[row_ind], indptr[row_ind + 1]):
            col_ind = indices[col_loc]
            part_ind = col_ind // partition_size
            b_widths = bucket_widths[part_ind]
            num_buckets = len(b_widths)
            degree = degree_counter[part_ind][row_ind]
            bucket_ind = bisect.bisect_left(b_widths, degree)
            if bucket_ind == num_buckets:
                # The very long row goes to the largest bucket
                bucket_ind -= 1
            bucket_width = b_widths[bucket_ind]
            need_new_row = False
            # # # test
            # print("")
            # print(F"row_ind: {row_ind} col_ind: {col_ind} degree: {degree} part_ind: {part_ind} bucket_ind: {bucket_ind} bucket_width: {bucket_width}")
            # # print(F"len(col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]): {len(col_indices[part_ind][bucket_ind])} (bucket_width - 1): {(bucket_width - 1)}")
            # # # end test
            remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
            # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
            if remainder == 0:
                # Current row is full, so need a new row
                need_new_row = True
            else:
                assert row_indices[part_ind][bucket_ind], F"row_indices[{part_ind}][{bucket_ind}] is empty"
                if row_ind != row_indices[part_ind][bucket_ind][-1]:
                    # row_ind is changed, so need a new row
                    need_new_row = True
                    # Padding current row if not full
                    for _ in range(remainder, bucket_width):
                        col_indices[part_ind][bucket_ind].append(0)
                        mask[part_ind][bucket_ind].append(0)
            
            if need_new_row:
                # Current row is full, or the row_ind changed
                assert len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1) == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
                # assert len(col_indices[part_ind][bucket_ind]) % bucket_width == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
                row_indices[part_ind][bucket_ind].append(row_ind)
                # # test
                # print(F"row_indices[{part_ind}][{bucket_ind}]: {row_indices[part_ind][bucket_ind]}")
                # # end test
            
            col_indices[part_ind][bucket_ind].append(col_ind)
            mask[part_ind][bucket_ind].append(1)

    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     print(F"\npart_ind: {part_ind} num_buckets: {num_buckets}")
    #     for bucket_ind in range(num_buckets):
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"\nbucket_ind: {bucket_ind} bucket_width: {bucket_width}")
    #         print(F"row_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices[part_ind][bucket_ind]}")
    #         print(F"col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices[part_ind][bucket_ind]}")
    #         print(F"mask[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask[part_ind][bucket_ind]}")
    # # end test

    # Padding the last rows and convert to numpy ndarray required by SparseTIR
    row_indices_nd = []
    col_indices_nd = []
    mask_nd = []
    for part_ind in range(num_parts):
        row_indices_part_local = []
        col_indices_part_local = []
        mask_part_local = []
        b_widths = bucket_widths[part_ind]
        num_buckets = len(b_widths)
        for bucket_ind in range(num_buckets):
            bucket_width = b_widths[bucket_ind]
            remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
            # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
            # Padding the last row
            if remainder:
                for _ in range(remainder, bucket_width):
                    col_indices[part_ind][bucket_ind].append(0)
                    mask[part_ind][bucket_ind].append(0)

            num_nz_rows = len(row_indices[part_ind][bucket_ind])
            # # test
            # print(F"len(col_indices[{part_ind}][{bucket_ind}]): {len(col_indices[part_ind][bucket_ind])}")
            # print(F"num_nz_rows: {num_nz_rows} bucket_width: {bucket_width} num_nz_rows * bucket_width: {num_nz_rows * bucket_width}")
            # # end test
            assert len(col_indices[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(col_indices[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            assert len(mask[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(mask[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            
            # Convert to numpy ndarray
            if num_nz_rows:
                row_indices_part_local.append(np.array(row_indices[part_ind][bucket_ind]))
                col_indices_part_local.append(np.array(col_indices[part_ind][bucket_ind]).reshape(num_nz_rows, bucket_width))
                mask_part_local.append(np.array(mask[part_ind][bucket_ind]).reshape(num_nz_rows, bucket_width))
            else:
                row_indices_part_local.append(np.empty(num_nz_rows, dtype=int))
                col_indices_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int).reshape(num_nz_rows, bucket_width))
                mask_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int).reshape(num_nz_rows, bucket_width))
        row_indices_nd.append(row_indices_part_local)
        col_indices_nd.append(col_indices_part_local)
        mask_nd.append(mask_part_local)

    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     for bucket_ind in range(num_buckets):
    #         print(F"\npart_ind: {part_ind} num_buckets: {num_buckets} bucket_ind: {bucket_ind} bucket_width: {b_widths[bucket_ind]}")
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"row_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices_nd[part_ind][bucket_ind]}")
    #         print(F"col_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices_nd[part_ind][bucket_ind]}")
    #         print(F"mask_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask_nd[part_ind][bucket_ind]}")
    # # end test
            
    return (row_indices_nd, col_indices_nd, mask_nd)
        

def bench_hyb_with_config(g,
                          x,
                          # y_golden,
                          y_ndarray,
                          feat_size=128,
                          # bucket_sizes=[],
                          bucket_widths=[],
                          bucketing_format=[],
                          coarsening_factor=2,
                          num_col_parts=1,
                          use_implicit_unroll=False):
    """Build and benchmark the SpMM kernel using hyb format

    """
    tvm_start_time = time.perf_counter()
    # num_buckets = len(bucket_sizes)
    coarsening_factor = min(coarsening_factor, feat_size // 32)
    # indptr, indices, _ = g.adj_tensors("csc")
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
    # indptr_nd = tvm.nd.array(indptr, device=tvm.cpu())
    # indices_nd = tvm.nd.array(indices, device=tvm.cpu())
    # indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
    # indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
    # cached_bucketing_format = column_part_hyb(
    #     m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
    # )
    # row_indices, col_indices, mask = cached_bucketing_format
    row_indices, col_indices, mask = bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        # for bucket_id, bucket_size in enumerate(bucket_sizes):
        for bucket_id, bucket_size in enumerate(b_widths):
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
    loc_base = 14
    loc_step = 7
    loc = loc_base
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        for bucket_id in range(num_buckets):
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            # param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
            #     part_id
            # ][bucket_id].shape[0]
            param_map[params[loc]] = m
            param_map[params[loc + 1]] = n
            param_map[params[loc + 2]] = row_indices[
                part_id
            ][bucket_id].shape[0]
            loc += loc_step

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    iter_names = []
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        for bucket_id in range(num_buckets):
            iter_names.append(F"csrmm_{part_id}_{bucket_id}")

    sch = tvm.tir.Schedule(mod)
    # for sp_iter_name in [
    #     "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    # ]:
    for sp_iter_name in iter_names:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        b_widths = bucket_widths[part_id]
        num_buckets = len(b_widths)
        # for bucket_id, bucket_size in enumerate(bucket_sizes):
        for bucket_id, bucket_size in enumerate(b_widths):
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
            # # test
            # print(F"part_id: {part_id} bucket_id: {bucket_id} bucket_size: {bucket_size} b_widths: {b_widths}")
            # # end test
            # io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            io, ioi, ii = sch.split(i, [None, b_widths[-1] // bucket_size, 8])
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
        b_widths = bucket_widths[part_id]
        # for bucket_id, _ in enumerate(bucket_sizes):
        for bucket_id, _ in enumerate(b_widths):
            # weight = tvm.nd.array(
            #     mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
            # )
            # rows = tvm.nd.array(
            #     row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            # )
            # cols = tvm.nd.array(
            #     col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
            #     device=tvm.cuda(0),
            # )
            weight = tvm.nd.array(
                mask[part_id][bucket_id].reshape(-1).astype("float32"), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]

    tvm_end_time = time.perf_counter()
    tvm_exe_time = tvm_end_time - tvm_start_time
    print(F"tvm_schedule_time(s): {tvm_exe_time:.6}")

    # test accuracy
    f(*args)
    tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
    # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    print("tir hyb time: {:.6f} ms".format(dur))

    return dur


def get_bucket_widths_list(buckets: dict):
    bucket_pool = list(buckets.keys())
    bucket_pool.sort()

    return bucket_pool


def move_largest_bucket(buckets: dict,
                        pre_max_width: int,
                        new_max_width: int):
    larger_bucket = buckets[pre_max_width]
    ratio = pre_max_width // new_max_width
    new_num_rows = larger_bucket.num_rows * ratio
    if new_max_width not in buckets:
        buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz)
    else:
        buckets[new_max_width].num_rows += new_num_rows
        buckets[new_max_width].nnz += larger_bucket.nnz

    del buckets[pre_max_width]


def get_bubble_overhead(buckets: dict):
    """Get the bubble overhead of all buckets

    Args:
        buckets (dict): Dictionary of bucket width to Bucket object
    """
    origin_max_width = max(buckets.keys())
    overhead = 0

    for width, bucket in buckets.items():
        num_rows = bucket.num_rows
        shape_rows = origin_max_width / width
        remaining = num_rows % shape_rows  # num_rows % shape_rows
        # remaining = num_rows & (shape_rows - 1)  # num_rows % shape_rows
        if remaining != 0:
            bubble_rows = shape_rows - remaining
        else:
            bubble_rows = 0
        # # test
        # print(F"bucket: {bucket} max_width: {origin_max_width} bubble_rows: {bubble_rows}")
        # # end test
        # overhead += bubble_rows
        overhead += bubble_rows * width

    return overhead


def get_memory_overhead(buckets: dict,
                    pre_max_width: int,
                    new_max_width: int):
    larger_bucket = buckets[pre_max_width]
    ratio = pre_max_width / new_max_width
    cost = (ratio - 1) * larger_bucket.num_rows

    # # Update the buckets
    # new_num_rows = larger_bucket.num_rows * ratio
    # if new_max_width not in buckets:
    #     buckets[new_max_width] = Bucket(new_max_width, new_num_rows, larger_bucket.nnz)
    # else:
    #     buckets[new_max_width].num_rows += new_num_rows
    #     buckets[new_max_width].nnz += larger_bucket.nnz

    # del buckets[pre_max_width]

    return cost    


def modify_bucket_widths(buckets: dict):
    """Modify initial bucket settings by using a cost model.

    Args:
        buckets (Dict{bucket_width: Bucket}): Dictionary of bucket width to Bucket object.
    """
    # # test
    # for width, bk in buckets.items():
    #     print(F"504 width: {width} bucket: {bk}")
    # # end test
    # Get original overhead as baseline
    base_overhead = get_bubble_overhead(buckets)
    base_buckets_list = get_bucket_widths_list(buckets)
    min_overhead = base_overhead
    best_setting = base_buckets_list
    # # test
    # print(F"base_overhead: {base_overhead} base_buckets_list: {base_buckets_list}")
    # # end test

    origin_max_width = max(buckets.keys())
    pre_max_width = origin_max_width
    new_max_width = pre_max_width // 2
    while new_max_width >= 1:
        # # test
        # print(F"pre_max_width: {pre_max_width} new_max_width: {new_max_width}")
        # # end test
        # Calculate the overhead if change the max bucket width
        mem_overhead = get_memory_overhead(buckets, pre_max_width, new_max_width)
        move_largest_bucket(buckets, pre_max_width, new_max_width)
        # # test
        # for width, bk in buckets.items():
        #     print(F"539 width: {width} bucket: {bk}")
        # # end test
        bb_overhead = get_bubble_overhead(buckets)
        total_overhead = mem_overhead + 4 * bb_overhead
        # # test
        # print(F"mem_overhead: {mem_overhead} bb_overhead: {bb_overhead} total_overhead: {total_overhead}")
        # # end test
        if total_overhead < min_overhead:
            best_setting = get_bucket_widths_list(buckets)
            min_overhead = total_overhead
        pre_max_width = new_max_width
        new_max_width //= 2
    
    return best_setting


def get_bucket_config(init_buckets):
    num_parts = len(init_buckets)
    bucket_config = []

    for part_ind in range(num_parts):
        # # test
        # print(F"Go to partition {part_ind}...")
        # # end test
        buckets = init_buckets[part_ind]
        # # test
        # print(F"533 buckets: {buckets}")
        # # end test
        # # test
        # print(F"old buckets setting: {get_bucket_widths_list(buckets)}")
        # # end test
        bucket_widths = modify_bucket_widths(buckets)
        # # test
        # print(F"537 bucket_widths: {bucket_widths}")
        # # end test
        # # test
        # print(F"new buckets setting: {bucket_widths}")
        # # end test
        bucket_config.append(bucket_widths)

    return bucket_config


def search_bucket_config(g: MTX,
                        # x,
                        # y_ndarray,
                        # feat_size,
                        num_partitions):
                        # coarsening_factor,
                        # use_implicit_unroll):
    # init_buckets()
    init_buckets = g.init_buckets(num_partitions)
    # # test
    # for part_ind in range(num_partitions):
    #     buckets = init_buckets[part_ind]
    #     print(F"part_id: {part_ind} num_buckets: {len(buckets)}")
    #     for width, bucket in buckets.items():
    #         print(F"bucekt: width={width} num_rows={bucket.num_rows} nnz={bucket.nnz}")
    # # end test
            
    bucket_config = get_bucket_config(init_buckets)

    return bucket_config