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
        int64_t shared_edge, int64_t col_num, int64_t ne01, int64_t ne1, int64_t nb01, int64_t nb1, int64_t nb2,
        int64_t row_size,
        const char *src0, const char *src1, const char *dst,
        ggml_vec_dot_t vec_dot) {
    for (int64_t row0_i = start_row; row0_i < end_row; row0_i += block) {
        for (int64_t col1_i = 0; col1_i < col_num; col1_i += block) { // every col of src1 matrix, block by block
            for (int64_t col_i = col1_i;
                 col_i < col1_i + block && col_i < col_num; col_i += 1) { // every col of dst matrix
                const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
                const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice

                const char *src0_row = src0;
                const char *src1_col = src1 + col_i * row_size;
                auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));

                for (int64_t row0 = row0_i;
                     row0 < row0_i + block && row0 < ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
                    vec_dot((int) shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
                            src1_col, 0, 1);
                }
            }
        }
    }
}

//__kernel void matrix_multiply(
//    __global const char* src0,
//    __global const char* src1,
//    __global const char* dst,
//    const int src0_row_num,
//    const int col_num,
//    const int block,
//    const int ne1,
//    const int nb1,
//    const int nb2,
//    const int nb01,
//    const int row_size,
//    const int shared_edge,
//    const int cpu_row_num)

//static void *ptr = nullptr;
//static cl_mem buffer = nullptr;
//static size_t buffer_size = 0;
//
//cl_mem get_dequantized_buffer(size_t size) {
//    cl_int err;
//    if (buffer == nullptr) {
//        ptr = malloc(size);
//        buffer = clCreateBuffer(_global_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, ptr, &err);
//        if (err != CL_SUCCESS) {
//            LOG(ERROR) << "clCreateBuffer failed, err: " << err;
//            exit(1);
//        }
//        buffer_size = size;
//        return buffer;
//    }
//    if (buffer_size < size) {
//        clReleaseMemObject(buffer);
//        LOG(INFO) << "resize buffer to " << size;
//        free(ptr);
//        ptr = malloc(size);
//        buffer = clCreateBuffer(_global_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, ptr, &err);
//        if (err != CL_SUCCESS) {
//            LOG(ERROR) << "clCreateBuffer failed, err: " << err;
//            exit(1);
//        }
//        buffer_size = size;
//    }
//    return buffer;
//}

//cl_kernel get_dequantize_kernel(ggml_type type) {
//    switch (type) {
//        case GGML_TYPE_Q4_K:
//            return get_kernel("dequantize_block_q4_K");
//        case GGML_TYPE_Q6_K:
//            return get_kernel("dequantize_block_q6_K");
//        case GGML_TYPE_F16:
//            return get_kernel("convert_fp16_to_fp32");
//        default:
//            return nullptr;
//    }
//}

//void start_gpu_compute(
//        const char *src0,
//        const char *src1,
//        const char *dst,
//        const int src0_row_num,
//        const int col_num,
//        const int block,
//        const int ne1,
//        const int nb1,
//        const int nb2,
//        const int nb01,
//        const int row_size,
//        const int shared_edge,
//        const int cpu_row_num,
//        ggml_type src0_type
//) {
//
//    cl_kernel kernel = get_kernel("matrix_multiply");
//    if (kernel == nullptr) {
//        printf("kernel is null\n");
//        exit(1);
//    }
//
//    cl_mem dequantized_buffer = get_dequantized_buffer(src0_row_num * shared_edge * sizeof(float));
//    if (dequantized_buffer == nullptr) {
//        printf("dequantized_buffer is null\n");
//        exit(1);
//    }
//
//    cl_mem src0_buf = create_buffer(buffer_type::READ_ONLY, src0_row_num * nb01 * sizeof(char), (void *) src0);
//    cl_mem src1_buf = create_buffer(buffer_type::READ_ONLY, col_num * row_size, (void *) src1);
//    cl_mem dst_buf = create_buffer(buffer_type::WRITE_ONLY, col_num * nb1 * sizeof(char), (void *) dst);
//
//    cl_kernel dequantize_kernel = get_dequantize_kernel(src0_type);
//    if (dequantize_kernel == nullptr) {
//        printf("dequantize_kernel is null\n");
//        exit(1);
//    }
//
//    clSetKernelArg(dequantize_kernel, 0, sizeof(cl_mem), &src0_buf);
//    clSetKernelArg(dequantize_kernel, 1, sizeof(cl_mem), &dequantized_buffer);
//
//    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dequantized_buffer);
//    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src1_buf);
//    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_buf);
//    clSetKernelArg(kernel, 3, sizeof(int), &src0_row_num);
//    clSetKernelArg(kernel, 4, sizeof(int), &col_num);
//    clSetKernelArg(kernel, 5, sizeof(int), &block);
//    clSetKernelArg(kernel, 6, sizeof(int), &ne1);
//    clSetKernelArg(kernel, 7, sizeof(int), &nb1);
//    clSetKernelArg(kernel, 8, sizeof(int), &nb2);
//    clSetKernelArg(kernel, 9, sizeof(int), &nb01);
//    clSetKernelArg(kernel, 10, sizeof(int), &row_size);
//    clSetKernelArg(kernel, 11, sizeof(int), &shared_edge);
//    clSetKernelArg(kernel, 12, sizeof(int), &cpu_row_num);
//
////    size_t global_work_size[2] = {static_cast<size_t>(nb2), 0};
//
//    switch (src0_type) {
//        case GGML_TYPE_Q4_K:
//            printf("q4_block_num: %d\n", src0_row_num * nb01 / 144);
//            GGML_ASSERT(src0_row_num * nb01 / 144 == nb2);
//            break;
//        case GGML_TYPE_Q6_K:
//            printf("q6_block_num: %d\n", src0_row_num * nb01 / 210);
//            GGML_ASSERT(src0_row_num * nb01 / 210 == nb2);
//            break;
//        case GGML_TYPE_F16:
//            break;
//        default:
//            break;
//    }
//    size_t global_work_size[2] = {static_cast<size_t>(2), 0};
//    size_t local_work_size[2] = {static_cast<size_t>(1), 0};
//
//    if (src0_type != GGML_TYPE_F16) {
//        enqueue_kernel(dequantize_kernel, 1, global_work_size, local_work_size);
//    }else {
//        f16_to_f32();
//    }
//
//    wait_queue_finish();
//
//    global_work_size[0] = static_cast<size_t>(col_num);
//    global_work_size[1] = static_cast<size_t>(cpu_row_num);
//    local_work_size[0] = static_cast<size_t>(block);
//    local_work_size[1] = static_cast<size_t>(block);
//
//    enqueue_kernel(kernel, 2, global_work_size, local_work_size);
//
//    // create cl buffer from host memory
////    cl_mem src0_buf = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src0_row_num * nb01 * sizeof(char), (void *)src0, NULL);
//
//}
//
//void wait_gpu_compute() {
//    wait_queue_finish();
//}

void mul_mat(int64_t ne01, int64_t ne11, int64_t ne12,
             int64_t nb01,
             int64_t ne1,
             int64_t nb1, int64_t nb2,
             size_t row_size,
             int64_t shared_edge,
             const char *src0, const char *src1, const char *dst,
             ggml_vec_dot_t vec_dot,
             ggml_type src0_type
) {
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

//    if (gpu_row_num > 0) {
//        start_gpu_compute(src0, src1, dst, (int)src0_row_num, (int)col_num, block, (int)ne1, (int)nb1, (int)nb2, (int)nb01, (int)row_size, (int)shared_edge, (int)cpu_row_num,src0_type);
//    }

    for (int64_t i_thread = 0; i_thread < N_THREADS_MUL_MAT_CPU; i_thread++) {
        int64_t start_row = i_thread * row_per_core;
        int64_t end_row = (i_thread + 1) * row_per_core;
        futures.push_back(
                g_thread_pool->enqueue(mul_mat_sub, start_row, end_row, block, shared_edge, col_num, ne01, ne1, nb01,
                                       nb1, nb2, row_size, src0, src1, dst, vec_dot));
    }

    for (auto &future: futures) {
        future.get();
    }

//    if (gpu_row_num > 0) {
//        wait_gpu_compute();
//    }

    return;

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
    if (gpu_row_num > 0) {
        for (int64_t row0_i = cpu_row_num; row0_i < src0_row_num; row0_i += block) {
            for (int64_t col1_i = 0; col1_i < col_num; col1_i += block) { // every col of src1 matrix, block by block
                for (int64_t col_i = col1_i;
                     col_i < col1_i + block && col_i < ne11 * ne12; col_i += 1) { // every col of dst matrix
                    const int64_t mat_i = col_i / ne1; // index of matrix slice (col)
                    const int64_t mat_col_i = col_i % ne1; // index of col in matrix slice

                    const char *src0_row = src0;
                    const char *src1_col = src1 + col_i * row_size;
                    auto *dst_col = (float *) (dst + (mat_col_i * nb1 + mat_i * nb2));

                    for (int64_t row0 = row0_i; row0 < row0_i + block && row0 <
                                                                         ne01; row0 += 1) { // iterate over rows of src0 matrix in the block
                        vec_dot((int) shared_edge, &dst_col[row0], 0, src0_row + row0 * nb01, 0,
                                src1_col, 0, 1);
                    }
                }
            }
        }

    }

}
