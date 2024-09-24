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
