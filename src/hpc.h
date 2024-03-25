//
// Created by geraltigas on 3/19/24.
//

#ifndef GEMMA_GGML_HPC_H
#define GEMMA_GGML_HPC_H

#ifdef __cplusplus

#include <cstdint>

#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml.h"

void mul_mat(int64_t ne01, int64_t ne11, int64_t ne12,
             int64_t nb01,
             int64_t ne1,
             int64_t nb1, int64_t nb2,
             size_t row_size,
             int64_t shared_edge,
             struct ggml_tensor *src0,struct ggml_tensor *src1,struct ggml_tensor *dst,
             ggml_vec_dot_t vec_dot,
             enum ggml_type src0_type,
                     const char *wdata
                     );

#ifdef __cplusplus
};
#endif

#endif //GEMMA_GGML_HPC_H
