@main = primfn(a: handle, b: handle, c: handle, indptr: handle, indices: handle, a_0_0: handle, indptr_i_0_0: handle, indices_i_0_0: handle, indices_j_0_0: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "horizontal_fuse": True, "sparse_tir_level": 1}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [768, 768], []),
             B: Buffer(B_1: Pointer(global float16), float16, [768, 8, 2, 32], []),
             C: Buffer(C_1: Pointer(global float16), float16, [768, 8, 2, 32], []),
             J_indptr: Buffer(J_indptr.data: Pointer(global int32), int32, [768], []),
             J_indices: Buffer(J_indices.data: Pointer(global int32), int32, [768, 768], []),
             A_0_0: Buffer(A_0_0_1: Pointer(global float16), float16, [1, 768, 768], []),
             I_0_0_indptr: Buffer(I_0_0_indptr.data: Pointer(global int32), int32, [1], []),
             I_0_0_indices: Buffer(I_0_0_indices.data: Pointer(global int32), int32, [1, 768], []),
             J_0_0_indices: Buffer(J_0_0_indices.data: Pointer(global int32), int32, [1, 768, 32], [])}
  buffer_map = {a: A, b: B, c: C, indptr: J_indptr, indices: J_indices, a_0_0: A_0_0, indptr_i_0_0: I_0_0_indptr, indices_i_0_0: I_0_0_indices, indices_j_0_0: J_0_0_indices} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    C_local = alloc_buffer(float16[768, 8, 2, 32])
    for (i_0_0_0: int32, 0, 248) "thread_binding" {
      for (i_0_0_1: int32, 0, 1) {
        for (i_0_0_2: int32, 0, 8) "thread_binding" {
          for (k1: int32, 0, 8) "thread_binding" {
            for (k3_init: int32, 0, 32) "thread_binding" {
              for (k2_init: int32, 0, 2) "unroll" {
                block([1, 1984, 8, 2, 32], "csrmm_0_00_init") as [vo_0_0, vi_0_0, vk1, vk2, vk3] {
                  bind(vo_0_0, 0)
                  bind(vi_0_0, (((i_0_0_0 + i_0_0_1)*8) + i_0_0_2))
                  bind(vk1, k1)
                  bind(vk2, k2_init)
                  bind(vk3, k3_init)
                  tir.reads([])
                  tir.writes([C_local[I_0_0_indices[vo_0_0, vi_0_0], vk1, vk2, vk3]])
                  tir.attrs({"sparse": True})
                  C_local[I_0_0_indices[vo_0_0, vi_0_0], vk1, vk2, vk3] = 0f16
              }
            }
            for (k3: int32, 0, 32) "thread_binding" {
              for (j_0_0: int32, 0, 32) "unroll" {
                for (k2: int32, 0, 2) "unroll" {
                  block([1, 1984, tir.reduce_axis(0, 32), 8, 2, 32], "csrmm_0_00_update") as [vo_0_0_1, vi_0_0_1, vj_0_0, vk1_1, vk2_1, vk3_1] {
                    bind(vo_0_0_1, 0)
                    bind(vi_0_0_1, (((i_0_0_0 + i_0_0_1)*8) + i_0_0_2))
                    bind(vj_0_0, j_0_0)
                    bind(vk1_1, k1)
                    bind(vk2_1, k2)
                    bind(vk3_1, k3)
                    tir.reads([C_local[I_0_0_indices[vo_0_0_1, vi_0_0_1], vk1_1, vk2_1, vk3_1], I_0_0_indices[vo_0_0_1, vi_0_0_1], A_0_0[vo_0_0_1, vi_0_0_1, vj_0_0], B[J_0_0_indices[vo_0_0_1, vi_0_0_1, vj_0_0], vk1_1, vk2_1, vk3_1], J_0_0_indices[vo_0_0_1, vi_0_0_1, vj_0_0]])
                    tir.writes([C_local[I_0_0_indices[vo_0_0_1, vi_0_0_1], vk1_1, vk2_1, vk3_1]])
                    tir.attrs({"sparse": True})
                    C_local[I_0_0_indices[vo_0_0_1, vi_0_0_1], vk1_1, vk2_1, vk3_1] = (C_local[I_0_0_indices[vo_0_0_1, vi_0_0_1], vk1_1, vk2_1, vk3_1] + (A_0_0[vo_0_0_1, vi_0_0_1, vj_0_0]*B[J_0_0_indices[vo_0_0_1, vi_0_0_1, vj_0_0], vk1_1, vk2_1, vk3_1]))
                }
              }
              for (ax0: int32, 0, 1) {
                for (ax1: int32, 0, 1) {
                  for (ax2: int32, 0, 2) {
                    for (ax3: int32, 0, 1) {
                      block([768, 8, 2, 32], "C_local") as [v0, v1, v2, v3] {
                        bind(v0, (I_0_0_indices[0, (((i_0_0_0 + i_0_0_1)*8) + i_0_0_2)] + ax0))
                        bind(v1, (k1 + ax1))
                        bind(v2, ax2)
                        bind(v3, (k3 + ax3))
                        tir.reads([C_local[v0, v1, v2, v3]])
                        tir.writes([C[v0, v1, v2, v3]])
                        tir.attrs({"sparse": True, "atomic": 1})
                        C[v0, v1, v2, v3] = C_local[v0, v1, v2, v3]
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
}