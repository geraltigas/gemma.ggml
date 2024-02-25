//
// Created by geraltigas on 2/25/24.
//

#include "gemma_model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ggml-quants.h>
#include <ggml.h>
#include <mm_malloc.h>
#include <string>
#include <vector>
#include <asm-generic/errno-base.h>


enum ne_type {
    NE_TYPE_F32 = 0,
    NE_TYPE_F16 = 1,
    NE_TYPE_Q4_0 = 2,
    NE_TYPE_Q4_1 = 3,
    // NE_TYPE_Q4_2 = 4, support has been removed
    // NE_TYPE_Q4_3 (5) support has been removed
    NE_TYPE_Q5_0 = 6,
    NE_TYPE_Q5_1 = 7,
    NE_TYPE_Q8_0 = 8,
    NE_TYPE_Q8_1 = 9,
    // k-quantizations
    // GGML_TYPE_Q2_K = 10,
    // GGML_TYPE_Q3_K = 11,
    // GGML_TYPE_Q4_K = 12,
    // GGML_TYPE_Q5_K = 13,
    NE_TYPE_Q6_K = 14,
    NE_TYPE_Q8_K = 15,
    NE_TYPE_I8,
    NE_TYPE_I16,
    NE_TYPE_I32,
    NE_TYPE_BTLA,
    NE_TYPE_COUNT,
};

typedef uint16_t ne_fp16_t;
static constexpr size_t NE_TYPE_SIZE[NE_TYPE_COUNT] = {
    // [NE_TYPE_F32] = sizeof(float),       [NE_TYPE_F16] = sizeof(ne_fp16_t),   [NE_TYPE_Q4_0] = sizeof(block_q4_0),
    // [NE_TYPE_Q4_1] = sizeof(block_q4_1), [NE_TYPE_Q5_0] = sizeof(block_q5_0), [NE_TYPE_Q5_1] = sizeof(block_q5_1),
    // [NE_TYPE_Q8_0] = sizeof(block_q8_0), [NE_TYPE_Q8_1] = sizeof(block_q8_1), [NE_TYPE_Q6_K] = sizeof(block_q6_K),
    // [NE_TYPE_Q8_K] = sizeof(block_q8_K), [NE_TYPE_I8] = sizeof(int8_t),       [NE_TYPE_I16] = sizeof(int16_t),
    // [NE_TYPE_I32] = sizeof(int32_t),
};

// normal array init, dont use [index] = value for static constexpr, use type name = {value, value, value, ...}
// static constexpr size_t NE_TYPE_SIZE[NE_TYPE_COUNT] = {
//   sizeof(float),       sizeof(ne_fp16_t),   sizeof(block_q4_0), sizeof(block_q4_1), 0, 0, sizeof(block_q5_0), sizeof(block_q5_1),
//   sizeof(block_q8_0), sizeof(block_q8_1), 0, 0, 0, 0, sizeof(block_q6_K), sizeof(block_q8_K), sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
// };

static constexpr int NE_BLCK_SIZE[NE_TYPE_COUNT] = {
    // [NE_TYPE_F32] = 1,      [NE_TYPE_F16] = 1,      [NE_TYPE_Q4_0] = QK4_0, [NE_TYPE_Q4_1] = QK4_1,
    // [NE_TYPE_Q5_0] = QK5_0, [NE_TYPE_Q5_1] = QK5_1, [NE_TYPE_Q8_0] = QK8_0, [NE_TYPE_Q8_1] = QK8_1,
    // [NE_TYPE_Q6_K] = QK_K,  [NE_TYPE_Q8_K] = QK_K,  [NE_TYPE_I8] = 1,       [NE_TYPE_I16] = 1,
    // [NE_TYPE_I32] = 1,
};

size_t ne_type_size(enum ne_type type) { return NE_TYPE_SIZE[type]; }

template<typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        // throw format("overflow multiplying %llu * %llu", (unsigned long long)a, (unsigned long long)b);
        // use std lib
        throw std::string("overflow multiplying ") + std::to_string(static_cast<unsigned long long>(a)) + " * " +
              std::to_string(static_cast<unsigned long long>(b));
    }
    return ret;
}

int ne_blck_size(enum ne_type type) { return NE_BLCK_SIZE[type]; }

static size_t model_calc_tensor_size(const std::vector<uint32_t> &ne, enum ne_type type) {
    size_t size = ne_type_size(type);
    for (uint32_t dim: ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ne_blck_size(type);
}

struct model_load_tensor_shard {
    std::vector<uint32_t> ne;
    size_t size;
    enum ne_type type;
    size_t file_idx;
    size_t file_off;

    void calc_size() { size = model_calc_tensor_size(ne, type); }
};

struct gguf_str {
    uint64_t n; // GGUFv2
    char *data;
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void *data;
    size_t size;
};

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    sizeof(uint8_t),
    sizeof(int8_t),
    sizeof(uint16_t),
    sizeof(int16_t),
    sizeof(uint32_t),
    sizeof(int32_t),
    sizeof(float),
    sizeof(bool),
    sizeof(struct gguf_str),
    0, // undefined
    sizeof(uint64_t),
    sizeof(int64_t),
    sizeof(double),
};

static bool gguf_fread_el(FILE *file, void *dst, size_t size, size_t *offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE *file, struct gguf_str *p, size_t *offset) {
    p->n = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);
    p->data = static_cast<char *>(calloc(p->n + 1, 1));
    ok = ok && gguf_fread_el(file, p->data, p->n, offset);

    return ok;
}


union gguf_value {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n; // GGUFv2
        void *data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum gguf_type type;
    union gguf_value value;
};

struct gguf_header {
    char magic[4];
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv; // GGUFv2
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv *kv;
    struct gguf_tensor_info *infos;

    size_t alignment;
    size_t offset; // offset of `data` from beginning of file
    size_t size; // size of `data` in bytes

    // uint8_t * padding;
    void *data;
};

inline static void *ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        printf("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void *aligned_memory = NULL;

    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);

    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        printf("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
        return NULL;
    }
    return aligned_memory;
}

#define GGML_ALIGNED_MALLOC(size) ggml_aligned_malloc(size)

int GemmaModel::load_model_from_file(const char *file_path) {
    if (file_path == nullptr) {
        return -1;
    }

    gguf_ctx = gguf_init_from_file(file_path,{
        .no_alloc = false,
        .ctx = nullptr,
    });

    size_t offset = 0;
    char magic[4];

    FILE *gguf_file = fopen(file_path, "rb");

    gguf_fread_el(gguf_file, &magic, sizeof(magic), &offset);
    std::printf("magic: %c%c%c%c\n", magic[0], magic[1], magic[2], magic[3]);
    const auto ctx = static_cast<struct gguf_context *>(GGML_ALIGNED_MALLOC(sizeof(struct gguf_context)));
    ctx->offset = 0;
    // read the header
    strncpy(ctx->header.magic, magic, 4);

    bool ok = true;
    ctx->kv = nullptr;
    ctx->infos = nullptr;
    ctx->data = nullptr;

    ok = ok && gguf_fread_el(gguf_file, &ctx->header.version, sizeof(ctx->header.version), &offset);
    ok = ok && gguf_fread_el(gguf_file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
    ok = ok && gguf_fread_el(gguf_file, &ctx->header.n_kv, sizeof(ctx->header.n_kv), &offset);

    std::printf("version: %d\n", ctx->header.version);
    std::printf("n_tensors: %lu\n", ctx->header.n_tensors);
    std::printf("n_kv: %lu\n", ctx->header.n_kv);

    if (ctx->header.version == 1) {
        fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
        fclose(gguf_file);
        gguf_free(ctx);
        return -1;
    }

    if (!ok) {
        fprintf(stderr, "%s: failed to read header\n", __func__);
        fclose(gguf_file);
        gguf_free(ctx);
        return -1;
    }

    ctx->kv = static_cast<gguf_kv *>(malloc(ctx->header.n_kv * sizeof(gguf_kv)));


    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        struct gguf_kv *kv = &ctx->kv[i];

        ok = ok && gguf_fread_str(gguf_file, &kv->key, &offset);
        ok = ok && gguf_fread_el(gguf_file, &kv->type, sizeof(kv->type), &offset);

        switch (kv->type) {
            case GGUF_TYPE_UINT8:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.uint8, sizeof(kv->value.uint8), &offset);
            break;
            case GGUF_TYPE_INT8:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.int8, sizeof(kv->value.int8), &offset);
            break;
            case GGUF_TYPE_UINT16:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.uint16, sizeof(kv->value.uint16), &offset);
            break;
            case GGUF_TYPE_INT16:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.int16, sizeof(kv->value.int16), &offset);
            break;
            case GGUF_TYPE_UINT32:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.uint32, sizeof(kv->value.uint32), &offset);
            break;
            case GGUF_TYPE_INT32:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.int32, sizeof(kv->value.int32), &offset);
            break;
            case GGUF_TYPE_FLOAT32:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.float32, sizeof(kv->value.float32), &offset);
            break;
            case GGUF_TYPE_UINT64:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.uint64, sizeof(kv->value.uint64), &offset);
            break;
            case GGUF_TYPE_INT64:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.int64, sizeof(kv->value.int64), &offset);
            break;
            case GGUF_TYPE_FLOAT64:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.float64, sizeof(kv->value.float64), &offset);
            break;
            case GGUF_TYPE_BOOL:
                ok = ok && gguf_fread_el(gguf_file, &kv->value.bool_, sizeof(kv->value.bool_), &offset);
            break;
            case GGUF_TYPE_STRING:
                ok = ok && gguf_fread_str(gguf_file, &kv->value.str, &offset);
            break;
            case GGUF_TYPE_ARRAY: {
                ok = ok && gguf_fread_el(gguf_file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                ok = ok && gguf_fread_el(gguf_file, &kv->value.arr.n, sizeof(kv->value.arr.n), &offset);

                switch (kv->value.arr.type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32:
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64:
                    case GGUF_TYPE_BOOL: {
                        kv->value.arr.data = malloc(kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type]);
                        ok = ok && gguf_fread_el(gguf_file, kv->value.arr.data,
                                                 kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type], &offset);
                    }
                    break;
                    case GGUF_TYPE_STRING: {
                        kv->value.arr.data = malloc(kv->value.arr.n * sizeof(struct gguf_str));
                        for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                            ok = ok && gguf_fread_str(gguf_file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
                        }
                    }
                    break;
                    case GGUF_TYPE_ARRAY:
                    case GGUF_TYPE_COUNT:
                        printf("False && invalid type");
                    break; // NE_ASSERT(false && "invalid type"); break;
                }
            }
            break;
            case GGUF_TYPE_COUNT:
                printf("False && invalid type"); // NE_ASSERT(false && "invalid type");
        }

        if (!ok) {
            break;
        }
    }

    // print the key-value pairs
    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        gguf_kv *kv = &ctx->kv[i];

        printf("key: %s\n", kv->key.data);
        printf("type: %d\n", kv->type);
        printf("value: %d\n", kv->value.uint64);
    }

    // read the tensor infos
    ctx->infos =
            static_cast<gguf_tensor_info *>(malloc(ctx->header.n_tensors * sizeof(gguf_tensor_info)));

    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        gguf_tensor_info *info = &ctx->infos[i];

        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            info->ne[j] = 1;
        }

        ok = ok && gguf_fread_str(gguf_file, &info->name, &offset);
        ok = ok && gguf_fread_el(gguf_file, &info->n_dims, sizeof(info->n_dims), &offset);
        for (uint32_t j = 0; j < info->n_dims; ++j) {
            ok = ok && gguf_fread_el(gguf_file, &info->ne[j], sizeof(info->ne[j]), &offset);
        }
        ok = ok && gguf_fread_el(gguf_file, &info->type, sizeof(info->type), &offset);
        ok = ok && gguf_fread_el(gguf_file, &info->offset, sizeof(info->offset), &offset);

        if (!ok) {
            fprintf(stderr, "%s: failed to read tensor info\n", __func__);
            fclose(gguf_file);
            gguf_free(ctx);
            return -1;
        }

        model_load_tensor_shard shard;
        std::string name = gguf_get_tensor_name(ctx, i);
        uint32_t name_len = name.length();
        shard.type = (enum ne_type) info->type;

        uint32_t n_dims = info->n_dims;
        shard.ne.resize(n_dims);
        for (uint32_t j = 0; j < info->n_dims; ++j) {
            shard.ne[j] = info->ne[j];
        }

        if (n_dims < 1 || n_dims > 2) {
            // throw format("model.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
            throw std::string("model.cpp: tensor '") + name + "' should not be " + std::to_string(n_dims) +
                  "-dimensional";
        }
        switch (shard.type) {
            case NE_TYPE_F32:
            case NE_TYPE_F16:
            case NE_TYPE_Q4_0:
            case NE_TYPE_Q4_1:
            case NE_TYPE_Q5_0:
            case NE_TYPE_Q5_1:
            case NE_TYPE_Q8_0:
            case NE_TYPE_Q6_K:
            case NE_TYPE_BTLA:
                break;
            default: {
                // throw format("unrecognized tensor type %u\n", shard.type);
                throw std::string("unrecognized tensor type ") + std::to_string(shard.type);
            }
        }

        //   shard.file_idx = 0;
        //   const size_t offs = file_offset(ctx, name.c_str());
        //   int length = info->ne[0] * info->ne[1] * info->ne[2] * info->ne[3] * 4;
        //
        //   shard.file_off = offs;
        //
        //   auto it = tensors_map.name_to_idx.find(name);
        //   size_t idx;
        //   if (it != tensors_map.name_to_idx.end()) {
        //     idx = it->second;
        //   } else {
        //     tensors_map.tensors.emplace_back(name);
        //     idx = tensors_map.tensors.size() - 1;
        //     tensors_map.name_to_idx.emplace(name, idx);
        //   }
        //   tensors_map.tensors.at(idx).shards.push_back(shard);
        // }
        //
        // ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
        //
        // int alignment_idx = gguf_find_key(ctx, "general.alignment");
        // if (alignment_idx != -1) {
        //   ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
        // }
        //
        // const size_t offset_pad = offset % ctx->alignment;
        //
        // if (offset_pad != 0) {
        //   offset += ctx->alignment - offset_pad;
        //   // fseek(file, offset, SEEK_SET);
        // }
        //
        // ctx->offset = offset;
        // gguf_data_offset = offset;
        //
        // return ctx;

        return 0;
    }
}
