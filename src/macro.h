#ifndef MACROS_H
#define MACROS_H

#include <glog/logging.h>
#include <cassert>

#define CGRAPH_MAX_NODE_NUM 1024
#define DEFAULT_BATCH_SIZE 512
#define DEFAULT_CTX_NUM 512
#define DEFAULT_TOKEN_NUM std::min(DEFAULT_CTX_NUM, DEFAULT_BATCH_SIZE)
#define DEFAULT_KV_CACHE_TYPE GGML_TYPE_F16
#define DEFAULT_ROPE_TYPE 2
#define DEFAULT_FREQ_BASE 10000
#define DEFAULT_FREQ_SCALE 1
#define DEFAULT_EXT_FACTOR 0
#define DEFAULT_ATTN_FACTOR 1
#define DEFAULT_BETA_FAST 32
#define DEFAULT_BETA_SLOW 1
#define DEFAULT_F_MAX_ALIBI_BIAS 0
#define N_THREADS_FOR_GGML 1
#define N_THREADS_MUL_MAT_CPU 4
#define CPU_SPLIT_RATIO 1
#define MUL_MAT_BLOCK_SIZE 2
#define COMPUTE_MID_NODE_DATA_BUFFER_SIZE (128u * 1024 * 1024)
#define OPENCL_PLATFORM_NAME "Intel(R) OpenCL HD Graphics"
#define OPENCL_DEVICE_NAME "Intel(R) UHD Graphics [0x9bc4]"
#define OPENCL_KERNELS_FILE "/home/geraltigas/Desktop/gemma.ggml/src/kernals.cl"

#define ON 1
#define OFF 0
#define SHOW 1
#define HIDE 0

#define CODE_MASK SHOW

#if CODE_MASK == SHOW
#define MASK(code) code
#else
#define MASK(code)
#endif

// define a macro to check return value and print error message, assert
#define CHECK_RT_MSG(rt, msg) if (rt != 0) { \
    LOG(ERROR) << msg; \
    assert(false); \
}


#define CHECK_RT(rt) if (rt != 0) { \
    assert(false); \
}

#define CHECK_PTR(ptr) if ((ptr) == nullptr) { \
    assert(false); \
}

#define ASSERT_MSG(expr, msg) if (!(expr)) { \
    LOG(ERROR) << msg;                       \
    assert(false);                           \
}

#define CHECK_BOOL(expr) if (!(expr)) { \
    assert(false); \
}


#endif //MACROS_H