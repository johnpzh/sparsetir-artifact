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
                          use_implicit_unroll=False,
                          check_correctness=False):
    """Build and benchmark the SpMM kernel using hyb format basedon SparseTIR and TVM.

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
            # io, ioi, ii = sch.split(i, [b_widths[-1] // bucket_size, 8])
            # io, ii = sch.split(i, [1, 8])
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

    # Dump code generated
    # test
    # dev_module = func.imported_modules[0]
    # print(dev_module.get_source())

    # dev_module = f.imported_modules[0]
    # print(dev_module.get_source())
    # sys.exit(-1)
    # end test

    # # test accuracy
    f(*args)
    if check_correctness:
        tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_ndarray, rtol=1e-4)
    # # tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    dur = profile_tvm_ms(f, args)
    # dur = profile_100_runs_ms(f, args)
    print("cell exe time: {:.6f} ms".format(dur))

    return dur