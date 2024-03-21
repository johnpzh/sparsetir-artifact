# dump_npz.py
# dump each DGLGraph into a Numpy npz file.
# dgSPARSE, Sputnik, and TACO can read this npz file by using cnpy (https://github.com/rogersce/cnpy).


# sparsetir
# SparseTIR reads DGLGraph directly
python3 bench_spmm_hyb.py -d ${dataset} -i > sparsetir_${dataset}_hyb.log 2> sparsetir_${dataset}_hyb.err
python3 bench_spmm_naive.py -d ${dataset} > sparsetir_${dataset}_naive.log 2> sparsetir_${dataset}_naive.err



            bench_hyb(
                g,
                x,
                y_golden,
                feat_size=feat_size,
                bucket_sizes=bucketing_config[name],
                coarsening_factor=2,
                num_col_parts=col_part_config[name],
                use_implicit_unroll=args.implicit_unroll,
            )


# 3rdparty/SparseTIR/src/sparse/format.cc
/*!
 * \brief Partition input CSR matrix by columns and collect rows into buckets according to non zero
 * elements per row.
 * \param num_rows Number of rows in the CSR matrix.
 * \param num_cols Number of columns in the CSR matrix.
 * \param indptr The indptr array of CSR matrix.
 * \param indices The indices array of CSR matrix.
 * \param num_col_parts Number of column partitions.
 * \param buckets The bucket sizes array.
 * \return {row_indices, col_indices, mask}, each one of them is a [num_col_parts, num_buckets *]
 * array.
 */
Array<Array<Array<NDArray>>> ColumnPartHyb(int num_rows, int num_cols, NDArray indptr,
                                           NDArray indices, int num_col_parts,
                                           Array<Integer> buckets) {
    //# The length of a row. How many elements a row has.
    //# Notes: partitions are divided evenly by column id, not by non-zeros.
    //# int part_id = col_id / partition_size;
    std::vector<std::unordered_multiset<int>> degree_counter(num_col_parts);

    //# Notes: bucketing_config is the bucket width.
    //# [1, 2, 4, 8, ...] means rows whose non-zero elements are less or equal to 1, 2, 4, 8, ...
    //# A very long row will be put into the last bucket.
    //# A bucket has row_indices, col_indices, mask, and bucket_size (width);
    //# One element in row_indices has <width> elements in col_indices and <width> elements in mask.
    //# ???How about the number of rows in a bucket? totally 2^k elements?
    int bucket_id = std::upper_bound(buckets_vec.begin(), buckets_vec.end(), degree - 1) - buckets_vec.begin();
    if (bucket_id == num_bkts) {
        bucket_id--;
    }
    int bucket_size = buckets_vec[bucket_id];

}

# 3rdparty/SparseTIR/python/tvm/sparse/format.py
def column_part_hyb(num_rows, num_cols, indptr_nd, indices_nd, num_col_parts, buckets):
    """Partition input CSR matrix by columns and collect rows into buckets according to non zero elements per row.

    Parameters
    ----------
    num_rows : int
        Number of rows in the CSR matrix.
    num_cols : int
        Number of columns in the CSR matrix.
    indptr : NDArray
        The indptr array of CSR matrix.
    indices : NDArray
        The indices array of CSR matrix.
    num_col_parts : int
        Number of column partitions.
    buckets : List
        The bucket sizes array.

    Returns
    -------
    Tuple[List[List[NDArray]]]
        The pair of (row_indices, col_indices, mask).
        row_indices is stored as a list of lists with shape (num_col_parts, len(buckets)), where the innermost element is an NDArray.
        col_indices and mask are stored in the same way.
    """
    return _ffi_api.ColumnPartHyb(
        num_rows, num_cols, indptr_nd, indices_nd, num_col_parts, buckets  # type: ignore
    )



//# 3rdparty/SparseTIR/src/tir/transforms/sparse_format_decompose.cc
PrimFunc SparseFormatDecompose(Array<FormatRewriteRule> composable_formats, PrimFunc f,
                               bool include_format_rewrite_blks = true) {
    for (const FormatRewriteRule& rule : composable_formats) {
      format_descs.push_back(AddSuffix(rule->new_format_desc, "_" + rule->name));
    }
    fptr->params = UpdateParams(format_descs, f->params);
    fptr->buffer_map = UpdateBufferMap(format_descs, f->buffer_map);
    fptr->sp_axes = UpdateSparseAxes(format_descs, f->sp_axes);
    Array<Stmt> format_rewrite_blks, compute_blks;
    // generate format rewrite blocks and compute blocks for each rule
    for (size_t i = 0; i < composable_formats.size(); ++i) {
      SparseFormatDecomposer rewriter(composable_formats[i], format_descs[i], old_sp_axes,
                                      old_buffers);
      rewriter(f->body);
      for (const Stmt& sp_iter : rewriter.format_rewrites_blks) {
        format_rewrite_blks.push_back(sp_iter);
      }
      for (const Stmt& sp_iter : rewriter.compute_blks) {
        compute_blks.push_back(sp_iter);
      }
    }
}

namespace transform {

Pass SparseFormatDecompose(Array<FormatRewriteRule> composable_formats,
                           bool include_format_rewrite_blks) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return SparseFormatDecompose(std::move(composable_formats), std::move(f),
                                 include_format_rewrite_blks);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SparseFormatDecompose", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SparseFormatDecompose").set_body_typed(SparseFormatDecompose);

}  // namespace transform


//# 3rdparty/SparseTIR/python/tvm/tir/transform/transform.py
def SparseFormatDecompose(
    composable_formats: List["FormatRewriteRule"], include_format_rewrite_blks: bool = True
):
    """Rewrite the sparse format of sparse buffers in the TIR scripts.

    Parameters
    ----------
    composable_formats : List[FormatRewriteRule]
        Composable formats is a list of rewrite rules.
    include_format_rewrite_blks : bool
        Whether to include format rewrite blocks in the output.

    Returns
    ------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SparseFormatDecompose(composable_formats, include_format_rewrite_blks)  # type: ignore


//# 3rdparty/SparseTIR/python/tvm/sparse/format.py
def format_decompose(
    mod: IRModule,
    composable_formats: List["FormatRewriteRule"],
    include_format_rewrite_blks: bool = True,
):
    """Rewrite the sparse format of sparse buffers in the TIR scripts.

    Parameters
    ----------
    mod : IRModule
        The IRModule to lower.
    composable_formats : List[FormatRewriteRule]
        Composable formats is a list of rewrite rules.
    include_format_rewrite_blks : bool
        Whether to include format rewrite blocks in the output.
    """
    if not isinstance(mod, IRModule):
        raise TypeError("Expected IRModule, but got {}".format(type(mod)))
    return SparseFormatDecompose(composable_formats, include_format_rewrite_blks)(mod)


from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)



cached_bucketing_format = None

def bench_hyb(
    g,
    x,
    y_golden,
    feat_size=128,
    bucket_sizes=[],
    coarsening_factor=2,
    num_col_parts=1,
    use_implicit_unroll=False,
):
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
        cached_bucketing_format = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )
    row_indices, col_indices, mask = cached_bucketing_format
    
    mod = format_decompose(mod, rewrites)

    # rewrite csrmm
    # rewrite CSR to ELL. The sparse matrix A is rewritten. Axis I is to be O, I.
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


# dgsparse runs dgsparse and also cuSPARSE
dgsparse-gespmm data/${dataset}.npz ${feat_size} > dgsparse_${dataset}_${feat_size}.log 2> dgsparse_${dataset}_${feat_size}.log


# get performance numbers
# For a given dataset, a geo-mean speedup is generated for different feature size which is the number of columns of matrix B in C = A * B.
python3 extract_data.py

with open("spmm.dat", "w") as fout:

# Plot figures
python3 plot.py


# References:
references = {
    "PackedFunc": "Let Python call a function that is from C++ codebase. https://tvm.apache.org/docs/arch/runtime.html",
    "TVM Codebase": "Code structure. https://tvm.apache.org/docs/dev/tutorial/codebase_walkthrough.html",
    "TVM Overview": "Compilation pipeline. https://tvm.apache.org/docs/tutorial/introduction.html#an-overview-of-tvm-and-model-optimization"
}





mod = tvm.IRModule.from_expr(csrmm)
sch = tvm.tir.Schedule(mod)
for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8]) # Here it split axis i into nested loops with factor=[None, bucket_sizes[-1] // bucket_size, 8] 