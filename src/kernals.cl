typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

__constant const int QK_K = 256;

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
    const int shared_edge,
    const int cpu_row_num
    ) {

    const int shared_edge_float = shared_edge / 4;

    int start_row = get_global_id(0) * block;
    int start_col = get_global_id(1) * block;

    if (start_row == 0 && start_col == 0) {
        bool all_zero = true;
        for (int i = 0; i< shared_edge_float * col_num; ++i) {
            if ((*(__global float *)(src0 + i * 4)) != 0) {
                all_zero = false;
                break;
            }
        }
        printf("ele num %d\n", shared_edge_float * col_num);
        printf("all_zero %d\n", all_zero);
    }

    printf("start_row %d start_col %d\n", start_row, start_col);
    printf("block %d\n", block);
    printf("shared_edge_float %d\n", shared_edge_float);

    for (int row0_i = start_row; row0_i < start_row + block && row0_i < src0_row_num; ++row0_i) {
        for (int col1_i = start_col; col1_i < start_col + block && col1_i < col_num; ++col1_i) {

            printf("row0_i %d col1_i %d\n", row0_i, col1_i);

            __global const char *src0_row = src0 + row0_i * nb01;
            __global const char *src1_col = src1 + col1_i * row_size;
            __global const char *dst_col = dst + col1_i * nb1;

            printf("src0_row %p\n", src0_row);

            float temp = 0;

            int l = 0;

            for (int i = 0; i < shared_edge_float; ++i) {
                temp += (*(__global float *)(src0_row + i * 4)) * (*(__global float *)(src1_col + i * 4));
                if (l < 5) {
                    printf("1: %f\n", (*(__global float *)(src0_row + i * 4)));
                    printf("2: %f\n", (*(__global float *)(src1_col + i * 4)));
                    l++;
                }
            }

            printf("temp %f\n", temp);
            printf("dst_col %f\n", (*(__global float *)dst_col));

            if ((*(__global float *)dst_col) != temp) {
                printf("Error at %d %d %f %f\n", row0_i, col1_i, (*(__global float *)dst_col), temp);
            }
        }
    }
}