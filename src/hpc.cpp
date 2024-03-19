//
// Created by geraltigas on 3/19/24.
//
#include "hpc.h"

void mul_mat(int64_t block,
             int64_t ne01, int64_t ne11, int64_t ne12,
             int64_t nb01,
             int64_t ne1,
             int64_t nb1, int64_t nb2,
             size_t row_size,
             int64_t shared_edge,
             const char *src0, const char *src1, const char *dst,
             ggml_vec_dot_t vec_dot) {



    for (int64_t col1_i = 0; col1_i < ne11 * ne12; col1_i += block) { // every col of src1 matrix, block by block
        for (int64_t row0_i = 0; row0_i < ne01; row0_i += block) { // every row of src0 matrix, block by block
            for (int64_t col_i = col1_i; col_i < col1_i + block && col_i < ne11 * ne12; col_i += 1) { // every col of dst matrix
                const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
                const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice

                const char *src0_row = src0;
                const char *src1_col = src1 + col_i * row_size;
                auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));

                for (int64_t row0 = row0_i; row0 < row0_i + block && row0 < ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
                    vec_dot(shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
                            src1_col, 0, 1);
                }
            }
        }
    }

}
