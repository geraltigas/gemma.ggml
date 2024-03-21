//
// Created by geraltigas on 3/19/24.
//
#include <cstdio>
#include <cmath>
#include "hpc.h"
#include "macro.h"
#include "profiling.h"
#include "thread_pool.h"
#include "opencl.h"

void mul_mat_sub(
        int64_t start_row, int64_t end_row,
        int64_t block,
        int64_t shared_edge, int64_t col_num, int64_t ne01, int64_t ne1, int64_t nb01, int64_t nb1, int64_t nb2, int64_t row_size,
        const char *src0, const char *src1, const char *dst,
        ggml_vec_dot_t vec_dot) {
    for (int64_t row0_i = start_row; row0_i < end_row; row0_i += block) {
        for (int64_t col1_i = 0; col1_i < col_num; col1_i += block) { // every col of src1 matrix, block by block
            for (int64_t col_i = col1_i; col_i < col1_i + block && col_i < col_num; col_i += 1) { // every col of dst matrix
                const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
                const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice

                const char *src0_row = src0;
                const char *src1_col = src1 + col_i * row_size;
                auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));

                for (int64_t row0 = row0_i;
                     row0 < row0_i + block && row0 < ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
                    vec_dot((int)shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
                            src1_col, 0, 1);
                }
            }
        }
    }
}

//__kernel void matrix_multiply(
//    __global const char* src0,
//    __global const char* src1,
//    __global float* dst,
//    const int src0_row_num,
//    const int col_num,
//    const int block,
//    const int ne1,
//    const int nb1,
//    const int nb2,
//    const int nb01,
//    const int row_size,
//    const int shared_edge)

void start_gpu_compute(
        const char *src0,
        const char *src1,
        const char *dst,
        const int src0_row_num,
        const int col_num,
        const int block,
        const int ne1,
        const int nb1,
        const int nb2,
        const int nb01,
        const int row_size,
        const int shared_edge,
        const int gpu_row_num
        ) {
    cl_kernel kernel = get_kernel("matrix_multiply");
    if (kernel == nullptr) {
        printf("kernel is null\n");
        exit(1);
    }
//    add_kernel();

}

void wait_gpu_compute() {

}


void mul_mat(int64_t ne01, int64_t ne11, int64_t ne12,
             int64_t nb01,
             int64_t ne1,
             int64_t nb1, int64_t nb2,
             size_t row_size,
             int64_t shared_edge,
             const char *src0, const char *src1, const char *dst,
             ggml_vec_dot_t vec_dot) {
    static char name[256];

    static const int64_t block = MUL_MAT_BLOCK_SIZE;
    static const double split_ratio = CPU_SPLIT_RATIO;
    const int64_t col_num = ne11 * ne12;
    // shared_edge
    sprintf(name, "shared_edge=%ld", shared_edge);
    add_count(name);
    // count col_num
    sprintf(name, "col_num=%ld", col_num);
    add_count(name);
    const int64_t src0_row_num = ne01;
    sprintf(name, "src0_row_num=%ld", src0_row_num);
    add_count(name);
    const int64_t cpu_row_num =
            ((int64_t) std::ceil(((double) src0_row_num * split_ratio) / (double) N_THREADS_MUL_MAT_CPU)) *
            N_THREADS_MUL_MAT_CPU;
    const int64_t gpu_row_num = src0_row_num - cpu_row_num;

    const int64_t row_per_core = cpu_row_num / N_THREADS_MUL_MAT_CPU;

    std::vector<std::future<void>> futures;

    start_gpu_compute(src0, src1, dst, (int)src0_row_num, (int)col_num, block, (int)ne1, (int)nb1, (int)nb2, (int)nb01, (int)row_size, (int)shared_edge, (int)gpu_row_num);

    for (int64_t i_thread = 0; i_thread < N_THREADS_MUL_MAT_CPU; i_thread++) {
        int64_t start_row = i_thread * row_per_core;
        int64_t end_row = (i_thread + 1) * row_per_core;
        futures.push_back(g_thread_pool->enqueue(mul_mat_sub, start_row, end_row, block, shared_edge, col_num, ne01, ne1, nb01, nb1, nb2, row_size, src0, src1, dst, vec_dot));
    }

    for (auto &future : futures) {
        future.get();
    }

    wait_gpu_compute();

//    for (int64_t row0_i = 0; row0_i < cpu_row_num; row0_i += block) { // every row of src0 matrix, block by block
//        for (int64_t col1_i = 0; col1_i < col_num; col1_i += block) { // every col of src1 matrix, block by block
//            for (int64_t col_i = col1_i;
//                 col_i < col1_i + block && col_i < ne11 * ne12; col_i += 1) { // every col of dst matrix
//                const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
//                const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice
//
//                const char *src0_row = src0;
//                const char *src1_col = src1 + col_i * row_size;
//                auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));
//
//                for (int64_t row0 = row0_i;
//                     row0 < row0_i + block && row0 < ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
//                    vec_dot(shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
//                            src1_col, 0, 1);
//                }
//            }
//        }
//    }

    sprintf(name, "gpu_row_num=%ld", gpu_row_num);
    add_count(name);

    for (int64_t row0_i = cpu_row_num; row0_i < src0_row_num; row0_i += block) {
        for (int64_t col1_i = 0; col1_i < col_num; col1_i += block) { // every col of src1 matrix, block by block
            for (int64_t col_i = col1_i; col_i < col1_i + block && col_i < ne11 * ne12; col_i += 1) { // every col of dst matrix
                const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
                const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice

                const char *src0_row = src0;
                const char *src1_col = src1 + col_i * row_size;
                auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));

                for (int64_t row0 = row0_i; row0 < row0_i + block && row0 < ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
                    vec_dot((int)shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
                            src1_col, 0, 1);
                }
            }
        }
    }

}
