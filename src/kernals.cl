__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, const unsigned int numElements) {

    int i = get_global_id(0);

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__kernel void matrix_multiply(
    __global const char* src0,
    __global const char* src1,
    __global const char* dst,
    const int src0_row_num,
    const int col_num,
    const int block,
    const int ne1,
    const int nb1,
    const int nb2,
    const int nb01,
    const int row_size,
    const int shared_edge) {

    const int shared_edge_float = shared_edge / 4;

    int start_row = get_global_id(0) * block;
    int start_col = get_global_id(1) * block;

    for (int row0_i = start_row; row0_i < start_row + block && row0_i < src0_row_num; ++row0_i) {
        for (int col1_i = start_col; col1_i < start_col + block && col1_i < col_num; ++col1_i) {
            const int mat_i = col1_i / ne1; // index of matrix slice (col)
            const int mat_col_i = col1_i % ne1; // index of col in matrix slice

            __global const char *src0_row = src0 + row0_i * nb01;
            __global const char *src1_col = src1 + col1_i * row_size;
            __global const char *dst_col = dst + (col1_i * nb1 + mat_i * nb2);

            for (int i = 0; i < shared_edge_float; ++i) {
                ((__global float *)dst_col)[i] += (((__global float*)src0_row)[i]) * (((__global float*)src1_col)[i]);
            }
        }
    }
}
