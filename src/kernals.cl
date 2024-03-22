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

inline void get_scale_min_k4(int j, const __global uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4)
    {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

struct __attribute__((packed)) block_q4_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

__kernel void dequantize_block_q4_K(__global const struct block_q4_K *x, __global float *yy)
{
    const int i = get_group_id(0);
    printf("i: %d\n", i);
    const int tid = get_local_id(0);
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const int n = 4;

    __global float *y = yy + get_group_id(0) * QK_K + 64 * il + n * ir;

    const float dall = vload_half(0, &x[i].d);
    const float dmin = vload_half(0, &x[i].dmin);

    __global const uint8_t *q = x[i].qs + 32 * il + n * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
    float d1 = dall * sc;
    float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
    float d2 = dall * sc;
    float m2 = dmin * m;
    for (int l = 0; l < n; ++l)
    {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
}

struct __attribute__((packed)) block_q6_K
{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

__kernel void dequantize_block_q6_K(__global const struct block_q6_K *x, __global float *yy)
{
    const int i = get_group_id(0);
    printf("i: %d\n", i);
    const int tid = get_local_id(0);
    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;

    __global float *y = yy + get_group_id(0) * QK_K + 128 * ip + il;

    const float d = vload_half(0, &x[i].d);

    __global const uint8_t *ql = x[i].ql + 64 * ip + il;
    const uint8_t qh = x[i].qh[32 * ip + il];
    __global const int8_t *sc = x[i].scales + is;

    y[0] = d * sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

__kernel void convert_fp16_to_fp32(__global half* x, __global float* y) {
    const uint i = get_global_id(0);

    y[i] = vload_half(0, &x[i]);
}