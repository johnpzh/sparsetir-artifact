#include <iostream>

/** 
 * C[i,k] = A[i,j] * B[j,k]
*/
void spmm_ell(int NI,
              int NJ,
              int NK,
              int *A_col_ind,
              double *A_values,
              int A_ell_size,
              double *B_values,
              double *C_values) {
    for (int i_ind = 0; i_ind < NI; ++i_ind) {
        int j_loc_start = i_ind * A_ell_size;
        int j_loc_bound = j_loc_start + A_ell_size;
        for (int j_loc = j_loc_start; j_loc < j_loc_bound; ++j_loc) {
            int j_ind = A_col_ind[i_ind * A_ell_size + j_loc];
            double A_val = A_values[i_ind * A_ell_size + j_loc];
            for (int k_ind = 0; k_ind < NK; ++k_ind) {
                double B_val = B_values[j_ind * NK + k_ind];
                C_values[i_ind * NK + k_ind] += A_val * B_val;
            }
        }
    }
}