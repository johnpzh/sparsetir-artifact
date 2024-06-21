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
extern "C" __global__ void __launch_bounds__(256) main_kernel0(float* __restrict__ C, int* __restrict__ I_0_0_indices_data, float* __restrict__ A_0_0, float* __restrict__ B, int* __restrict__ J_0_0_indices_data, float* __restrict__ A_0_1, int* __restrict__ J_0_1_indices_data, int* __restrict__ I_0_1_indices_data) {
  float C_local[2];
  if (((int)blockIdx.x) < 1551) {
    for (int i_0_0_1 = 0; i_0_0_1 < 16; ++i_0_0_1) {
      #pragma unroll
      for (int k2_init = 0; k2_init < 2; ++k2_init) {
        if ((((((int)blockIdx.x) * 128) + (i_0_0_1 * 8)) + ((int)threadIdx.y)) < 198405) {
          C[((((I_0_0_indices_data[(((((int)blockIdx.x) * 128) + (i_0_0_1 * 8)) + ((int)threadIdx.y))] * 512) + (((int)blockIdx.y) * 64)) + (k2_init * 32)) + ((int)threadIdx.x))] = 0.000000e+00f;
        }
      }
      #pragma unroll
      for (int j_0_0 = 0; j_0_0 < 16; ++j_0_0) {
        #pragma unroll
        for (int k2 = 0; k2 < 2; ++k2) {
          if ((((((int)blockIdx.x) * 128) + (i_0_0_1 * 8)) + ((int)threadIdx.y)) < 198405) {
            C[((((I_0_0_indices_data[(((((int)blockIdx.x) * 128) + (i_0_0_1 * 8)) + ((int)threadIdx.y))] * 512) + (((int)blockIdx.y) * 64)) + (k2 * 32)) + ((int)threadIdx.x))] = (C[((((I_0_0_indices_data[(((((int)blockIdx.x) * 128) + (i_0_0_1 * 8)) + ((int)threadIdx.y))] * 512) + (((int)blockIdx.y) * 64)) + (k2 * 32)) + ((int)threadIdx.x))] + (A_0_0[((((((int)blockIdx.x) * 2048) + (i_0_0_1 * 128)) + (((int)threadIdx.y) * 16)) + j_0_0)] * B[((((J_0_0_indices_data[((((((int)blockIdx.x) * 2048) + (i_0_0_1 * 128)) + (((int)threadIdx.y) * 16)) + j_0_0)] * 512) + (((int)blockIdx.y) * 64)) + (k2 * 32)) + ((int)threadIdx.x))]));
          }
        }
      }
    }
  } else {
    #pragma unroll
    for (int k2_init1 = 0; k2_init1 < 2; ++k2_init1) {
      if (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) >> 1)) < 9609) {
        C_local[k2_init1] = 0.000000e+00f;
      }
    }
    #pragma unroll
    for (int j_0_1 = 0; j_0_1 < 256; ++j_0_1) {
      #pragma unroll
      for (int k21 = 0; k21 < 2; ++k21) {
        if (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) >> 1)) < 9609) {
          C_local[k21] = (C_local[k21] + (A_0_1[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 256)) + j_0_1) - 3176448)] * B[((((J_0_1_indices_data[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 256)) + j_0_1) - 3176448)] * 512) + (((int)blockIdx.y) * 64)) + (k21 * 32)) + ((int)threadIdx.x))]));
        }
      }
    }
    for (int ax2 = 0; ax2 < 2; ++ax2) {
      if (((((int)blockIdx.x) * 4) + (((int)threadIdx.y) >> 1)) < 9609) {
        atomicAdd(C + ((((I_0_1_indices_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) - 12408)] * 512) + (((int)blockIdx.y) * 64)) + (ax2 * 32)) + ((int)threadIdx.x)), C_local[ax2]);
      }
    }
  }
}


