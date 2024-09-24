@main = primfn(a: handle, b: handle, c: handle, indptr: handle, indices: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 1}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [24, 24, 32, 32], []),
             B: Buffer(B_1: Pointer(global float16), float16, [24, 32, 512], []),
             C: Buffer(C_1: Pointer(global float16), float16, [24, 32, 512], []),
             J_indptr: Buffer(J_indptr.data: Pointer(global int32), int32, [24], []),
             J_indices: Buffer(J_indices.data: Pointer(global int32), int32, [24, 24], [])}
  buffer_map = {a: A, b: B, c: C, indptr: J_indptr, indices: J_indices} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i: int32, 0, 24) "thread_binding" {
      block([24], "bsrmm0") as [vi] {
        bind(vi, i)
        tir.reads([J_indptr[vi:(vi + 2)], A[vi, 0:24, 0:32, 0:32], B[0:24, 0:32, 0:512], J_indices[vi, 0:24]])
        tir.writes([C[vi, 0:32, 0:512]])
        tir.attrs({"sparse": True})
        C_wmma.accumulator = alloc_buffer(float16[24, 32, 512])
        A_wmma.matrix_a = alloc_buffer(float16[24, 24, 32, 32])
        B_shared = alloc_buffer(float16[24, 32, 512])
        B_shared_wmma.matrix_b = alloc_buffer(float16[24, 32, 512])
        for (f_0: int32, 0, 16) "thread_binding" {
          for (bi_0_init: int32, 0, 2) "thread_binding" {
            for (f_1_init: int32, 0, 2) "unroll" {
              block([2, 32], "bsrmm1_init_o") as [vbi_o, vf_o] {
                bind(vbi_o, bi_0_init)
                bind(vf_o, ((f_0*2) + f_1_init))
                tir.reads([])
                tir.writes([C_wmma.accumulator[vi, (vbi_o*16):((vbi_o*16) + 16), (vf_o*16):((vf_o*16) + 16)]])
                C_frag = match_buffer(C_wmma.accumulator[vi, (vbi_o*16):((vbi_o*16) + 16), (vf_o*16):((vf_o*16) + 16)])
                for (tx: int32, 0, 32) "thread_binding" {
                  @tir.tvm_fill_fragment(C_frag_1: Pointer(wmma.accumulator float16), 16, 16, 16, (floordiv(C_frag_elem_offset: int32, 256) + floordiv(floormod(C_frag_elem_offset, 256), 16)), 0f16, dtype=handle)
                }
            }
          }
          for (j: int32, 0, (J_indptr[(vi + 1)] - J_indptr[vi])) {
            for (bi_0: int32, 0, 2) "thread_binding" {
              for (ax1_0: int32, 0, 2) "unroll" {
                block([24, 24, 2, 2], "A_wmma.matrix_a_o") as [v0, v1, v2_o, v3_o] {
                  bind(v0, vi)
                  bind(v1, j)
                  bind(v2_o, bi_0)
                  bind(v3_o, ax1_0)
                  tir.reads([A[v0, v1, (v2_o*16):((v2_o*16) + 16), (v3_o*16):((v3_o*16) + 16)]])
                  tir.writes([A_wmma.matrix_a[v0, v1, (v2_o*16):((v2_o*16) + 16), (v3_o*16):((v3_o*16) + 16)]])
                  A_2 = match_buffer(A[v0, v1, (v2_o*16):((v2_o*16) + 16), (v3_o*16):((v3_o*16) + 16)])
                  A_frag = match_buffer(A_wmma.matrix_a[v0, v1, (v2_o*16):((v2_o*16) + 16), (v3_o*16):((v3_o*16) + 16)])
                  for (tx_1: int32, 0, 32) "thread_binding" {
                    @tir.tvm_load_matrix_sync(A_frag_1: Pointer(wmma.matrix_a float16), 16, 16, 16, (floordiv(A_frag_elem_offset: int32, 256) + floordiv(floormod(A_frag_elem_offset, 256), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A_3: Pointer(global float16), A_elem_offset: int32, (s0: int32*16), 1, dtype=handle), s0, "row_major", dtype=handle)
                  }
              }
              for (f_1: int32, 0, 2) "unroll" {
                for (ax0_ax1_fused_0: int32, 0, 2) "unroll" {
                  for (ax0_ax1_fused_1: int32, 0, 2) "thread_binding" {
                    for (ax0_ax1_fused_2: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_3: int32, 0, 4) "vectorized" {
                        block([24, 32, 512], "B_shared") as [v0_1, v1_1, v2] {
                          bind(v0_1, J_indices[vi, j])
                          bind(v1_1, floordiv(((((ax0_ax1_fused_0*256) + (ax0_ax1_fused_1*128)) + (ax0_ax1_fused_2*4)) + ax0_ax1_fused_3), 16))
                          bind(v2, (((f_0*32) + (f_1*16)) + floormod(((((ax0_ax1_fused_0*256) + (ax0_ax1_fused_1*128)) + (ax0_ax1_fused_2*4)) + ax0_ax1_fused_3), 16)))
                          tir.reads([B[v0_1, v1_1, v2]])
                          tir.writes([B_shared[v0_1, v1_1, v2]])
                          tir.attrs({"sparse": True})
                          B_shared[v0_1, v1_1, v2] = B[v0_1, v1_1, v2]
                      }
                    }
                  }
                }
                for (bj_0: int32, 0, 2) {
                  block([24, 2, 32], "B_shared_wmma.matrix_b_o") as [v0_2, v1_o, v2_o_1] {
                    bind(v0_2, J_indices[vi, j])
                    bind(v1_o, bj_0)
                    bind(v2_o_1, ((f_0*2) + f_1))
                    tir.reads([B_shared[v0_2, (v1_o*16):((v1_o*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)]])
                    tir.writes([B_shared_wmma.matrix_b[v0_2, (v1_o*16):((v1_o*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)]])
                    B_2 = match_buffer(B_shared[v0_2, (v1_o*16):((v1_o*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)])
                    B_frag = match_buffer(B_shared_wmma.matrix_b[v0_2, (v1_o*16):((v1_o*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)])
                    for (tx_2: int32, 0, 32) "thread_binding" {
                      @tir.tvm_load_matrix_sync(B_frag_1: Pointer(wmma.matrix_b float16), 16, 16, 16, (floordiv(B_frag_elem_offset: int32, 256) + floordiv(floormod(B_frag_elem_offset, 256), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B_3: Pointer(shared float16), B_elem_offset: int32, (s0_1: int32*16), 1, dtype=handle), s0_1, "row_major", dtype=handle)
                    }
                  block([tir.reduce_axis(0, 24), 2, 32, tir.reduce_axis(0, 2)], "bsrmm1_update_o") as [vj, vbi_o_1, vf_o_1, vbj_o] {
                    bind(vj, j)
                    bind(vbi_o_1, bi_0)
                    bind(vf_o_1, ((f_0*2) + f_1))
                    bind(vbj_o, bj_0)
                    tir.reads([C_wmma.accumulator[vi, (vbi_o_1*16):((vbi_o_1*16) + 16), (vf_o_1*16):((vf_o_1*16) + 16)], A_wmma.matrix_a[vi, vj, (vbi_o_1*16):((vbi_o_1*16) + 16), (vbj_o*16):((vbj_o*16) + 16)], B_shared_wmma.matrix_b[J_indices[vi, vj], (vbj_o*16):((vbj_o*16) + 16), (vf_o_1*16):((vf_o_1*16) + 16)]])
                    tir.writes([C_wmma.accumulator[vi, (vbi_o_1*16):((vbi_o_1*16) + 16), (vf_o_1*16):((vf_o_1*16) + 16)]])
                    A_frag_2 = match_buffer(A_wmma.matrix_a[vi, vj, (vbi_o_1*16):((vbi_o_1*16) + 16), (vbj_o*16):((vbj_o*16) + 16)])
                    B_frag_2 = match_buffer(B_shared_wmma.matrix_b[J_indices[vi, vj], (vbj_o*16):((vbj_o*16) + 16), (vf_o_1*16):((vf_o_1*16) + 16)])
                    C_frag_2 = match_buffer(C_wmma.accumulator[vi, (vbi_o_1*16):((vbi_o_1*16) + 16), (vf_o_1*16):((vf_o_1*16) + 16)])
                    for (tx_3: int32, 0, 32) "thread_binding" {
                      @tir.tvm_mma_sync(C_frag_3: Pointer(wmma.accumulator float16), (floordiv(C_frag_elem_offset_1: int32, 256) + floordiv(floormod(C_frag_elem_offset_1, 256), 16)), A_frag_3: Pointer(wmma.matrix_a float16), (floordiv(A_frag_elem_offset_1: int32, 256) + floordiv(floormod(A_frag_elem_offset_1, 256), 16)), B_frag_3: Pointer(wmma.matrix_b float16), (floordiv(B_frag_elem_offset_1: int32, 256) + floordiv(floormod(B_frag_elem_offset_1, 256), 16)), C_frag_3, (floordiv(C_frag_elem_offset_1, 256) + floordiv(floormod(C_frag_elem_offset_1, 256), 16)), dtype=handle)
                    }
                }
              }
            }
          }
          for (ax0: int32, 0, 1) {
            for (ax1_0_1: int32, 0, 2) "thread_binding" {
              for (ax2_0: int32, 0, 2) "unroll" {
                block([24, 2, 32], "C_wmma.accumulator_o") as [v0_3, v1_o_1, v2_o_2] {
                  bind(v0_3, vi)
                  bind(v1_o_1, ax1_0_1)
                  bind(v2_o_2, ((f_0*2) + ax2_0))
                  tir.reads([C_wmma.accumulator[v0_3, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)]])
                  tir.writes([C[v0_3, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)]])
                  C_frag_4 = match_buffer(C_wmma.accumulator[v0_3, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)])
                  C_2 = match_buffer(C[v0_3, (v1_o_1*16):((v1_o_1*16) + 16), (v2_o_2*16):((v2_o_2*16) + 16)])
                  for (tx_4: int32, 0, 32) "thread_binding" {
                    @tir.tvm_store_matrix_sync(C_frag_5: Pointer(wmma.accumulator float16), 16, 16, 16, (floordiv(C_frag_elem_offset_2: int32, 256) + floordiv(floormod(C_frag_elem_offset_2, 256), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_3: Pointer(global float16), C_elem_offset: int32, (s0_2: int32*16), 2, dtype=handle), s0_2, "row_major", dtype=handle)
                  }
              }
            }
          }
        }
    }
}