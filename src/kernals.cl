typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

__constant const int QK_K = 256;
__constant const uint32_t kmask1 = 0x3f3f3f3f;
__constant const uint32_t kmask2 = 0x0f0f0f0f;
__constant const uint32_t kmask3 = 0x03030303;

struct __attribute__((packed)) block_q4_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct __attribute__((packed)) block_q6_K
{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

struct __attribute__((packed)) block_q8_K
{
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
};

void memset(char *ptr, char value, int size) {
    for (int i = 0; i < size; ++i) {
        ptr[i] = value;
    }
}

void memcpy(char *dst,__global const char *src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

void vec_dot_q4_K_q8_K(int n, float * restrict s, size_t bs,__global const void * restrict vx, size_t bx,__global const void * restrict vy, size_t by) {

    __global const struct block_q4_K * restrict x = vx;
    __global const struct block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset((char *)sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        __global const uint8_t * restrict q4 = x[i].qs;
        __global const  int8_t * restrict q8 = y[i].qs;
        memset((char *)aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy((char *)utmp, (__global char *)x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = (float)(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = (float)(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}


void vec_dot_q6_K_q8_K(int n, float * restrict s, size_t bs,__global const void * restrict vx, size_t bx,__global const void * restrict vy, size_t by) {

    __global const struct block_q6_K * restrict x = vx;
    __global const struct block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset((char *)sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        __global const uint8_t * restrict q4 = x[i].ql;
        __global const uint8_t * restrict qh = x[i].qh;
        __global const  int8_t * restrict q8 = y[i].qs;
        memset((char *)aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int l = 0; l < 16; ++l) {
            a[l+ 0] = (int8_t)((q4[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            a[l+16] = (int8_t)((q4[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            a[l+32] = (int8_t)((q4[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            a[l+48] = (int8_t)((q4[l+16] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = ((float)(x[i].d)) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

}

void vec_dot_f16(int n, float *restrict s, size_t bs,__global const char *restrict x, size_t bx,__global const char *restrict y, size_t by) {
    double sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += ((float)(((__global half *)x)[i])*(float)(((__global half *)y)[i]));
    }
    *s = sumf;
}

__kernel void matrix_multiply_q4_K(
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
    int start_row = get_global_id(0) * block;
    int start_col = get_global_id(1) * block;

    for (int row0_i = start_row; row0_i < start_row + block && row0_i < src0_row_num; ++row0_i) {
        for (int col1_i = start_col; col1_i < start_col + block && col1_i < col_num; ++col1_i) {

            __global const char *src0_row = src0 + row0_i * nb01;
            __global const char *src1_col = src1 + col1_i * row_size;
            __global const char *dst_col = dst + col1_i * nb1 + row0_i * sizeof(float);

            float temp = 0;

            vec_dot_q4_K_q8_K(
                shared_edge,
                &temp,
                0,
                src0_row,
                0,
                src1_col,
                0
            );

            if ((*(__global float *)dst_col) != temp) {
                float diff = (*(__global float *)dst_col) - temp;
                if (diff < 0) {
                    diff = -diff;
                }
                if (diff > 0.0001) {
                    printf("Error at %d %d %f %f\n", row0_i, col1_i, (*(__global float *)dst_col), temp);
                }
            }
        }
    }
}

__kernel void matrix_multiply_q6_K(
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

    int start_row = get_global_id(0) * block;
    int start_col = get_global_id(1) * block;

    for (int row0_i = start_row; row0_i < start_row + block && row0_i < src0_row_num; ++row0_i) {
        for (int col1_i = start_col; col1_i < start_col + block && col1_i < col_num; ++col1_i) {

            __global const char *src0_row = src0 + row0_i * nb01;
            __global const char *src1_col = src1 + col1_i * row_size;
            __global const char *dst_col = dst + col1_i * nb1 + row0_i * sizeof(float);

            float temp = 0;

            vec_dot_q6_K_q8_K(
                shared_edge,
                &temp,
                0,
                src0_row,
                0,
                src1_col,
                0
            );

            if ((*(__global float *)dst_col) != temp) {
                float diff = (*(__global float *)dst_col) - temp;
                if (diff < 0) {
                    diff = -diff;
                }
                if (diff > 0.0001) {
                    printf("Error at %d %d %f %f\n", row0_i, col1_i, (*(__global float *)dst_col), temp);
                }
            }
        }
    }
}

__kernel void matrix_multiply_f16(
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

    int start_row = get_global_id(0) * block;
    int start_col = get_global_id(1) * block;

    for (int row0_i = start_row; row0_i < start_row + block && row0_i < src0_row_num; ++row0_i) {
        for (int col1_i = start_col; col1_i < start_col + block && col1_i < col_num; ++col1_i) {

            __global const char *src0_row = src0 + row0_i * nb01;
            __global const char *src1_col = src1 + col1_i * row_size;
            __global const char *dst_col = dst + col1_i * nb1 + row0_i * sizeof(float);

            float temp = 0;

            vec_dot_f16(
                shared_edge,
                &temp,
                0,
                src0_row,
                0,
                src1_col,
                0
            );

            if ((*(__global float *)dst_col) != temp) {
                float diff = (*(__global float *)dst_col) - temp;
                if (diff < 0) {
                    diff = -diff;
                }
                if (diff > 0.0001) {
                    printf("Error at %d %d %f %f\n", row0_i, col1_i, (*(__global float *)dst_col), temp);
                }
            }
        }
    }
}

