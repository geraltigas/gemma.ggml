//
// Created by geraltigas on 3/19/24.
//
#include <cstdio>
#include <cmath>
#include <utility>
#include <gmp.h>
#include "hpc.h"
#include "macro.h"
#include "profiling.h"
#include "thread_pool.h"
#include "opencl.h"
#include "ggml-quants.h"

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

//static cl_mem dequantized_buffer = nullptr;
//static size_t size = 0;
//static void *ptr = nullptr;
//static bool is_lastest = false;
//
//void set_quantized_buffer_size(size_t s) {
//    if (s > size) {
//        if (ptr != nullptr) {
//            free(ptr);
//        }
//        ptr = malloc(s);
//        size = s;
//        is_lastest = false;
//    }
//}
//
//cl_mem set_dequantized_buffer() {
//    cl_int err;
//    if (dequantized_buffer == nullptr) {
//        dequantized_buffer = clCreateBuffer(_global_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, ptr, &err);
//        if (err != CL_SUCCESS) {
//            printf("create buffer failed\n");
//            exit(1);
//        }
//        is_lastest = true;
//    }
//    if (!is_lastest) {
//        dequantized_buffer = clCreateBuffer(_global_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, ptr, &err);
//        if (err != CL_SUCCESS) {
//            printf("create buffer failed\n");
//            exit(1);
//        }
//        is_lastest = true;
//    }
//    return dequantized_buffer;
//}

//cl_kernel get_dequantize_kernel(ggml_type type) {
//    switch (type) {
//        case GGML_TYPE_Q4_K:
//            add_count("q4_K");
//            return get_kernel("dequantize_block_q4_K");
//        case GGML_TYPE_Q6_K:
//            add_count("q6_K");
//            return get_kernel("dequantize_block_q6_K");
//        default:
//            return nullptr;
//    }
//}

int scale_factor(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return 8;
        case GGML_TYPE_Q6_K:
            return 4;
        case GGML_TYPE_F16:
        default:
            return 1;
    }
}

size_t local_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return 32;
        case GGML_TYPE_Q6_K:
            return 64;
        case GGML_TYPE_F16:
        default:
            return 0;
    }
}

cl_kernel get_mul_mat_kernel(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            return get_kernel("matrix_multiply_q4_K");
        case GGML_TYPE_Q6_K:
            return get_kernel("matrix_multiply_q6_K");
        case GGML_TYPE_F16:
            return get_kernel("matrix_multiply_f16");
        default:
            return nullptr;
    }
}

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
        const int cpu_row_num,
        ggml_type src0_type
) {

    cl_kernel kernel = get_mul_mat_kernel(src0_type);
    if (kernel == nullptr) {
        printf("kernel is null\n");
        exit(1);
    }

    cl_mem src0_buf = create_buffer(buffer_type::READ_ONLY, src0_row_num * nb01 * sizeof(char), (void *) src0);
    cl_mem src1_buf = create_buffer(buffer_type::READ_ONLY, col_num * row_size, (void *) src1);
    cl_mem dst_buf = create_buffer(buffer_type::WRITE_ONLY, col_num * nb1 * sizeof(char), (void *) dst);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src0_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src1_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(kernel, 3, sizeof(int), &src0_row_num);
    clSetKernelArg(kernel, 4, sizeof(int), &col_num);
    clSetKernelArg(kernel, 5, sizeof(int), &block);
    clSetKernelArg(kernel, 6, sizeof(int), &ne1);
    clSetKernelArg(kernel, 7, sizeof(int), &nb1);
    clSetKernelArg(kernel, 8, sizeof(int), &nb2);
    clSetKernelArg(kernel, 9, sizeof(int), &nb01);
    clSetKernelArg(kernel, 10, sizeof(int), &row_size);
    clSetKernelArg(kernel, 11, sizeof(int), &shared_edge);
    clSetKernelArg(kernel, 12, sizeof(int), &cpu_row_num);

    size_t global_work_size[2] = {static_cast<size_t>(src0_row_num / block), static_cast<size_t>(col_num) / block};
    size_t global_offset[2] = {static_cast<size_t>(cpu_row_num / block), 0};
    size_t local_work_size[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};

    enqueue_kernel(kernel, 2, global_work_size, global_offset , local_work_size);

}

void wait_gpu_compute() {
    wait_queue_finish();
    release_buffer_in_set();
}

bool is_continuous(ggml_tensor *tensor) {
    return tensor->nb[3] / tensor->nb[2] == tensor->ne[2] && tensor->nb[2] / tensor->nb[1] == tensor->ne[1];
}

struct big_int {
    mpz_t val{};
    std::string name;
    big_int(int val, std::string name) : name(std::move(name)) {
        mpz_init_set_si(this->val, val);
    }
    ~big_int() {
        // print val and name
        gmp_printf("%s: %Zd\n", name.c_str(), val);
        mpz_clear(val);
    }
};

void mul_mat(int64_t ne01, int64_t ne11, int64_t ne12,
             int64_t nb01,
             int64_t ne1,
             int64_t nb1, int64_t nb2,
             size_t row_size,
             int64_t shared_edge,
             ggml_tensor *src0,ggml_tensor *src1, ggml_tensor *dst,
             ggml_vec_dot_t vec_dot,
             ggml_type src0_type,
             const char *wdata
) {
    static char name[256];
    auto * src0_p = (const char *)src0->data;
    auto * dst_p = (const char *)dst->data;

    static const int64_t block = MUL_MAT_BLOCK_SIZE;
    static const double split_ratio = CPU_SPLIT_RATIO;
    const int64_t col_num = ne11 * ne12;
    const int64_t src0_row_num = ne01;
    MASK(
    // shared_edge
    sprintf(name, "shared_edge=%ld", shared_edge);
    add_count(name);
    // count col_num
    sprintf(name, "col_num=%ld", col_num);
    add_count(name);
    sprintf(name, "src0_row_num=%ld", src0_row_num);
    add_count(name);
    )
    int64_t cpu_row_num =
            ((int64_t) std::ceil(((double) src0_row_num * split_ratio) / (double) N_THREADS_MUL_MAT_CPU)) *
            N_THREADS_MUL_MAT_CPU;
    int64_t gpu_row_num = src0_row_num - cpu_row_num;

    const int64_t row_per_core = cpu_row_num / N_THREADS_MUL_MAT_CPU;

    std::vector<std::future<void>> futures;

    if (src0_type == GGML_TYPE_Q6_K) {
        gpu_row_num = 0;
        cpu_row_num = src0_row_num;
    }

    if (gpu_row_num > 0) {
        start_gpu_compute(src0_p, (char *)wdata, dst_p, (int)src0_row_num, (int)col_num, block, (int)ne1, (int)nb1, (int)nb2, (int)nb01, (int)row_size, (int)shared_edge, (int)cpu_row_num,src0_type);
    }

    for (int64_t i_thread = 0; i_thread < N_THREADS_MUL_MAT_CPU; i_thread++) {
        int64_t start_row = i_thread * row_per_core;
        int64_t end_row = (i_thread + 1) * row_per_core;
        futures.push_back(
                g_thread_pool->enqueue(mul_mat_sub, start_row, end_row, block, shared_edge, col_num, ne01, ne1, nb01,
                                       nb1, nb2, row_size, src0_p, wdata, dst_p, vec_dot));
    }

    for (auto &future: futures) {
        future.get();
    }

    if (gpu_row_num > 0) {
        wait_gpu_compute();
    }

MASK(
    // gmp int and init with 0
    static big_int sum_q4(0, "q4_K");
    static big_int sum_q6(0, "q6_K");
    static big_int sum_f16(0, "f16");

    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            sprintf(name, "q4_K - [%ld,%ld] - [%ld,%ld, %ld]", src0->ne[0], src0->ne[1], src0->ne[2], src1->ne[0], src1->ne[1]);
            // add matrix multiply complexity to sum_q4
            mpz_addmul_ui(sum_q4.val, sum_q4.val, src0->ne[0] * src0->ne[1] * src0->ne[2] * src1->ne[0] * src1->ne[1]);
            add_count(name);
            break;
        case GGML_TYPE_Q6_K:
            sprintf(name, "q6_K - [%ld,%ld] - [%ld,%ld, %ld]", src0->ne[0], src0->ne[1], src0->ne[2], src1->ne[0], src1->ne[1]);
            // add matrix multiply complexity to sum_q6
            mpz_addmul_ui(sum_q6.val, sum_q6.val, src0->ne[0] * src0->ne[1] * src0->ne[2] * src1->ne[0] * src1->ne[1]);
            add_count(name);
            break;
        case GGML_TYPE_F16:
            sprintf(name, "f16 - [%ld,%ld] - [%ld,%ld, %ld]", src0->ne[0], src0->ne[1], src0->ne[2], src1->ne[0], src1->ne[1]);
            // add matrix multiply complexity to sum_f16
            mpz_addmul_ui(sum_f16.val, sum_f16.val, src0->ne[0] * src0->ne[1] * src0->ne[2] * src1->ne[0] * src1->ne[1]);
            add_count(name);
            break;
        default:
            LOG(ERROR) << "unknown type";
            assert(false);
    }
)
//    cpu_row_num = 0;
//    gpu_row_num = src0_row_num;
//
//    static int i = 0;

//    if (strcmp(src0->name, "blk.0.attn_v.weight") == 0) {
//        printf("tensor name %s\n", src0->name);
//        printf("src0_row_num %ld\n", src0_row_num);
//        printf("col_num %ld\n", col_num);
//        printf("block %ld\n", block);
//        printf("ne1 %ld\n", ne1);
//        printf("nb1 %ld\n", nb1);
//        printf("nb2 %ld\n", nb2);
//        printf("nb01 %ld\n", nb01);
//        printf("row_size %ld\n", row_size);
//        printf("shared_edge %ld\n", shared_edge);
//        printf("cpu_row_num %ld\n", cpu_row_num);
//        printf("src0_type %d\n", src0_type);
//    }

//    if (src0_type == GGML_TYPE_Q6_K) {
//        return;
//    }
//
//    if (gpu_row_num > 0) {
//        start_gpu_compute(src0_p, (char *)wdata, dst_p, (int)src0_row_num, (int)col_num, block, (int)ne1, (int)nb1, (int)nb2, (int)nb01, (int)row_size, (int)shared_edge, (int)cpu_row_num,src0_type);
//    }
//
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

                    const char *src0_row = src0_p;
                    const char *src1_col = wdata + col_i * row_size;
                    auto *dst_col = (float *) (dst_p + (mat_col_i * nb1 + mat_i * nb2));

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
