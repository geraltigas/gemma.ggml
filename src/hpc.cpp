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

static cl_mem dequantized_buffer = nullptr;
static size_t size = 0;
static void *ptr = nullptr;
static bool is_lastest = false;

void set_quantized_buffer_size(size_t s) {
    if (s > size) {
        if (ptr != nullptr) {
            free(ptr);
        }
        ptr = malloc(s);
        size = s;
        is_lastest = false;
    }
}

cl_mem set_dequantized_buffer() {
    cl_int err;
    if (dequantized_buffer == nullptr) {
        dequantized_buffer = clCreateBuffer(_global_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, ptr, &err);
        if (err != CL_SUCCESS) {
            printf("create buffer failed\n");
            exit(1);
        }
        is_lastest = true;
    }
    if (!is_lastest) {
        dequantized_buffer = clCreateBuffer(_global_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, ptr, &err);
        if (err != CL_SUCCESS) {
            printf("create buffer failed\n");
            exit(1);
        }
        is_lastest = true;
    }
    return dequantized_buffer;
}

cl_kernel get_dequantize_kernel(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            add_count("q4_K");
            return get_kernel("dequantize_block_q4_K");
        case GGML_TYPE_Q6_K:
            add_count("q6_K");
            return get_kernel("dequantize_block_q6_K");
        default:
            return nullptr;
    }
}

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

    cl_kernel kernel = get_kernel("matrix_multiply");
    if (kernel == nullptr) {
        printf("kernel is null\n");
        exit(1);
    }

    set_quantized_buffer_size(src0_row_num * shared_edge * sizeof(float));

    // dequantize
    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            for (int i = cpu_row_num; i < src0_row_num; i++) {
                dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(src0 + i * nb01), (float *) ((char *)ptr + i * shared_edge * sizeof(float)), shared_edge);
            }
            break;
        case GGML_TYPE_Q6_K:
            for (int i = cpu_row_num; i < src0_row_num; i++) {
                dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(src0 + i * nb01), (float *) ((char *)ptr + i * shared_edge * sizeof(float)), shared_edge);
            }
            break;
        case GGML_TYPE_F16:
            LOG(ERROR) << "src0 is f16";
            exit(1);
            break;
        default:
            break;
    }

    // ptr
    printf("\n");
    for (int i = 0; i < 4; i++) {
            printf("%f ", ((float *)ptr)[i]);
    }

    // src1
    printf("\n");
    for (int i = 0; i < 4; i++) {
            printf("%f ", ((float *)src1)[i]);
    }

//    char temp[1024];
//    quantize_row_q4_K(reinterpret_cast<const float *>(src0 + 0 * nb01), (float *) ((char *)ptr + 0 * shared_edge * sizeof(float)), shared_edge);
//
//    bool same = true;
////   nb01 byte
//    for (int i = 0; i < nb01; i++) {
//        if (src0[i] != ((char *)ptr)[i]) {
//            same = false;
//            break;
//        }
//    }
//
//    if (same) {
//        printf("same\n");
//    } else {
//        printf("not same\n");
//    }

    printf("ptr=%p\n", ptr);

    set_dequantized_buffer();
    cl_mem src1_buf = create_buffer(buffer_type::READ_ONLY, col_num * row_size, (void *) src1);
    cl_mem dst_buf = create_buffer(buffer_type::WRITE_ONLY, col_num * nb1 * sizeof(char), (void *) dst);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dequantized_buffer);
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

//    size_t global_work_size[2] = {static_cast<size_t>(shared_edge * src0_row_num / (size_t)scale_factor(src0_type)), 0};
//    size_t offset[2] = {(size_t)(shared_edge * src0_row_num / ggml_blck_size(src0_type)),0};
//    size_t local_work_size[2] = {local_size(src0_type), 0};
//
//    enqueue_kernel(dequantize_kernel, 1, global_work_size,offset, local_work_size);
//
//    wait_queue_finish();
//    global_work_size[0] = static_cast<size_t>(col_num);
//    global_work_size[1] = static_cast<size_t>(cpu_row_num);
//    local_work_size[0] = static_cast<size_t>(block);
//    local_work_size[1] = static_cast<size_t>(block);
//    size_t global_work_size[2] = {static_cast<size_t>(src0_row_num - cpu_row_num) / block, static_cast<size_t>(col_num) / block};
    size_t global_work_size[2] = {1,1};
    size_t local_work_size[2] = {static_cast<size_t>(1), static_cast<size_t>(1)};

    enqueue_kernel(kernel, 2, global_work_size, nullptr, local_work_size);

    // create cl buffer from host memory
//    cl_mem src0_buf = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src0_row_num * nb01 * sizeof(char), (void *)src0, NULL);

}

void wait_gpu_compute() {
    wait_queue_finish();
    release_buffer_in_set();
    exit(0);
}

bool is_continuous(ggml_tensor *tensor) {
    return tensor->nb[3] / tensor->nb[2] == tensor->ne[2] && tensor->nb[2] / tensor->nb[1] == tensor->ne[1];
}

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
    // shared_edge
    sprintf(name, "shared_edge=%ld", shared_edge);
    add_count(name);
    // count col_num
    sprintf(name, "col_num=%ld", col_num);
    add_count(name);
    const int64_t src0_row_num = ne01;
    sprintf(name, "src0_row_num=%ld", src0_row_num);
    add_count(name);
    int64_t cpu_row_num =
            ((int64_t) std::ceil(((double) src0_row_num * split_ratio) / (double) N_THREADS_MUL_MAT_CPU)) *
            N_THREADS_MUL_MAT_CPU;
    int64_t gpu_row_num = src0_row_num - cpu_row_num;

    if (src0_type == GGML_TYPE_F16) {
        gpu_row_num = 0;
        cpu_row_num = src0_row_num;
        add_count("f16");
    }

    const int64_t row_per_core = cpu_row_num / N_THREADS_MUL_MAT_CPU;

    std::vector<std::future<void>> futures;

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

    if (src0_type == GGML_TYPE_F16) {
        printf("src0 is f16\n");
        return;
    }

    if (!is_continuous(src0)) {
        printf("src0 is not continuous\n");
        return;
    }

    cpu_row_num = 0;
    gpu_row_num = src0_row_num;

    if (gpu_row_num > 0) {
        start_gpu_compute(src0_p, (char *)src1->data, dst_p, (int)src0_row_num, (int)col_num, block, (int)ne1, (int)nb1, (int)nb2, (int)nb01, (int)row_size, (int)shared_edge, (int)cpu_row_num,src0_type);
    }

    if (gpu_row_num > 0) {
        wait_gpu_compute();
    }

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
