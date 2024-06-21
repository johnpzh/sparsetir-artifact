
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) main_kernel0(float* __restrict__ A_0_0, float* __restrict__ B, int* __restrict__ J_0_0_indices_data, float* __restrict__ C, int* __restrict__ I_0_0_indices_data) {
  float C_local[2];
  #pragma unroll
  for (int k2_init = 0; k2_init < 2; ++k2_init) {
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < 2451) {
      C_local[k2_init] = 0.000000e+00f;
    }
  }
  #pragma unroll
  for (int j_0_0 = 0; j_0_0 < 16; ++j_0_0) {
    #pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < 2451) {
        C_local[k2] = (C_local[k2] + (A_0_0[(((((int)blockIdx.x) * 128) + (((int)threadIdx.y) * 16)) + j_0_0)] * B[((((J_0_0_indices_data[(((((int)blockIdx.x) * 128) + (((int)threadIdx.y) * 16)) + j_0_0)] * 512) + (((int)blockIdx.y) * 64)) + (k2 * 32)) + ((int)threadIdx.x))]));
      }
    }
  }
  for (int ax2 = 0; ax2 < 2; ++ax2) {
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < 2451) {
      atomicAdd(C + ((((I_0_0_indices_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] * 512) + (((int)blockIdx.y) * 64)) + (ax2 * 32)) + ((int)threadIdx.x)), C_local[ax2]);
    }
  }
}

/**/
/* Switch blockIdx.x and blockIdx.y, because ge-spmm used swapped blockIdx.x and blockIdx.y */
extern "C" __global__ void __launch_bounds__(256) main_kernel0(float* __restrict__ A_0_0, float* __restrict__ B, int* __restrict__ J_0_0_indices_data, float* __restrict__ C, int* __restrict__ I_0_0_indices_data) {
  float C_local[2];
  #pragma unroll
  for (int k2_init = 0; k2_init < 2; ++k2_init) {
    if (((((int)blockIdx.y) * 8) + ((int)threadIdx.y)) < 2451) {
      C_local[k2_init] = 0.000000e+00f;
    }
  }
  #pragma unroll
  for (int j_0_0 = 0; j_0_0 < 16; ++j_0_0) {
    #pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
      if (((((int)blockIdx.y) * 8) + ((int)threadIdx.y)) < 2451) {
        C_local[k2] = (C_local[k2] + (A_0_0[(((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 16)) + j_0_0)] * B[((((J_0_0_indices_data[(((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 16)) + j_0_0)] * 512) + (((int)blockIdx.x) * 64)) + (k2 * 32)) + ((int)threadIdx.x))]));
      }
    }
  }
  for (int ax2 = 0; ax2 < 2; ++ax2) {
    if (((((int)blockIdx.y) * 8) + ((int)threadIdx.y)) < 2451) {
      atomicAdd(C + ((((I_0_0_indices_data[((((int)blockIdx.y) * 8) + ((int)threadIdx.y))] * 512) + (((int)blockIdx.x) * 64)) + (ax2 * 32)) + ((int)threadIdx.x)), C_local[ax2]);
    }
  }
}

/**/
/* Simplify the code */
extern "C" __global__ void __launch_bounds__(256) main_kernel0(float* __restrict__ A_0_0, float* __restrict__ B, int* __restrict__ J_0_0_indices_data, float* __restrict__ C, int* __restrict__ I_0_0_indices_data) {
  float C_local;
  #pragma unroll
  // for (int k2_init = 0; k2_init < 2; ++k2_init) {
  
  int i = ((((int)blockIdx.y) * 8) + ((int)threadIdx.y));
  int j = (int)blockIdx.x * 64 + (int)threadIdx.x;

  if (i < 2451) {
    C_local = 0.000000e+00f;
  }
  // }
  #pragma unroll
  for (int j_0_0 = 0; j_0_0 < 16; ++j_0_0) {
    int loc = (((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 16)) + j_0_0);
    /// loc = blockIdx.y * 8 * WIDTH + threadIdx.y * WIDTH + j_0_0;
    int k = J_0_0_indices_data[loc];
    // #pragma unroll
    // for (int k2 = 0; k2 < 2; ++k2) {
    if (i < 2451) {
      C_local = (C_local + (A_0_0[loc] * B[k * 512 + j]));  // N == 512
    }
    // }
  }
  // for (int ax2 = 0; ax2 < 2; ++ax2) {
  if (i < 2451) {
    atomicAdd(C + ((I_0_0_indices_data[i] * 512) + j), C_local);
  }
  // }
}

/**/
/* Simplify further to CUDA kernel style */
__global__
void main_kernel0(float *A, float *B, int *J_indices, float *C, int *I_indices) {
  float C_local = 0;
  int i = blockIdx.y * 8 + threadIdx.y;  /// blockDim.y == 8
  int j = blockIdx.x * 64 + threadIdx.x;  /// blockDim.x == 64

  for (int w_i = 0; w_i < WIDTH; ++w_i) {
    int loc = blckIdx.y * 8 * MAX_width + threadIdx.y * WIDTH + w_i - OFFSET;
    int k = J_indices[loc];
    C_local += A[loc] * B[k * N + j];
  }
  atomicAdd(C[I_indices[i] * N + j], C_local);
}

/**/
/* Simplify further to pseudocode C[M,W] = A[M,W] * B[W,N] */
void main_kernel0(float *A, float *B, int *J_indices, float *C, int *I_indices) {
  for (int i = 0; i < M; ++i) {   /// Number of total rows
    for (int j = 0; j < N; ++j) {  /// Feature size
      float C_local = 0;
      for (int w_i = 0; w_i < WIDTH; ++w_i) {
        int loc = affine(w_i);
        int k = J_indices[loc];
        C_local += A[loc] * B[k * N + j];
      }
      if (!need_atomic) {
        C[I_indices[i] * N + j] += C_local
      } else {
        atomicAdd(C[I_indices[i] * N + j], C_local);
      }
    }
  }
}
