//
// Created by geraltigas on 2/25/24.
//

#ifndef GEMMA_MODEL_H
#define GEMMA_MODEL_H

#include <ggml.h>
#include <map>
#include <vector>
#include <string>
#include <set>
#include "type.h"
#include "ggml-backend.h"
#include "ggml-backend.h"

enum class InferenceStage {
    PREFILL,
    DECODE
};

struct GemmaLayer {
    ggml_tensor *attn_output;
    ggml_tensor *attn_k;
    ggml_tensor *attn_v;
    ggml_tensor *attn_q;
    ggml_tensor *ffn_gate;
    ggml_tensor *ffn_up;
    ggml_tensor *ffn_down;
    ggml_tensor *attn_norm;
    ggml_tensor *ffn_norm;
};

struct GemmaInputTensorHolder {
    ggml_tensor *inp_tokens;
//    ggml_tensor *inp_embd;
    ggml_tensor *inp_pos;
    ggml_tensor *inp_KQ_mask;
//    ggml_tensor *inp_KV_mask;
//    ggml_tensor *inp_K_shift;
//    ggml_tensor *inp_mean;
//    ggml_tensor *inp_cls;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

struct GemmaTensorHolder {
    u32 layer_num;

    ggml_tensor *token_embd;
    ggml_tensor *output_norm;
    ggml_tensor *output;

    std::vector<GemmaLayer> layers;
};

struct GemmaMidTensorHolder {
    ggml_tensor *input;
    ggml_tensor *output;
};

struct GemmaTokenizer {
    std::vector<str> tokens;
    std::vector<f32> scores;
    std::map<str, token_type> token_type_map;
public:
    token_id special_bos_id;
    token_id special_eos_id;
    token_id special_unk_id;
    token_id special_sep_id;
    token_id special_pad_id;
    std::string find_token(token_id id);
    void print_tokens(std::vector<token_id> &input);
    void print_token(token_id id);
};


struct kv_cell {
    i32 pos = -1;
    i32 delta = 0;

    std::set<i32> seq_id;

    bool has_seq_id(const i32 &id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const kv_cell &other) const {
        return seq_id == other.seq_id;
    }
};


struct kv_cache {

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
//    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

//    size_t total_size() const {
//        size_t size = 0;
//        for (ggml_backend_buffer_t buf: bufs) {
//            size += ggml_backend_buffer_get_size(buf);
//        }
//        return size;
//    }

//    ~kv_cache() {
//        for (struct ggml_context *ctx: ctxs) {
//            ggml_free(ctx);
//        }
//        for (ggml_backend_buffer_t buf: bufs) {
//            ggml_backend_buffer_free(buf);
//        }
//    }
};

struct HyperParam {
    u32 n_layer = 0;
    u32 n_kv_pair = 0;
    u32 n_tensors = 0;
    u32 n_embd = 0;
    u32 n_head = 0;
    u32 n_embd_heads = 0;
    f32 f_norm_rms_eps = 0;
    u32 origin_ctx_len = 0;
    u32 n_head_kv;
//    u32 n_kv_cache = 0;
};


class GemmaModel {
    ggml_context *input_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
    ggml_context *compute_ctx = nullptr;
    ggml_context *kv_ctx = nullptr;
    std::vector<u8> compute_meta_buffer;

    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_buffer_type_t backend_buffer_type = nullptr;


    HyperParam hyper_param;

    kv_cache _kv_cache;
    std::map<ggml_type, u32> n_type;
    std::map<str, ggml_tensor *> tensors;
    std::map<str, ggml_type> tensor_types;
    std::map<str, gguf_type> kv_types;
    std::map<str, i32> kv_index;

    GemmaInputTensorHolder input_tensor_holder;
    GemmaTensorHolder tensor_holder;
    GemmaMidTensorHolder mid_tensor_holder;

    GemmaTokenizer tokenizer;

public:
    int load_model_from_file(const char *file_path);

    int load_tokenizer(gguf_context *gguf_ctx);

    int model_warmup();

    int init_input_tensor();

    ggml_tensor *get_tensor(const char *name);

    void inference(std::vector<token_id> &input, InferenceStage stage);

    void begin_one_round_inference();

private:
    u32 get_u32_from_kv(gguf_context *gguf_ctx, const char *key);

    f32 get_f32_from_kv(gguf_context *gguf_ctx, const char *key);

    str get_str_from_kv(gguf_context *gguf_ctx, const char *key);

    gguf_type get_arr_elem_type(gguf_context *gguf_ctx, const char *key);

    std::vector<str> get_str_arr_from_kv(gguf_context *gguf_ctx, const char *key);

    std::vector<f32> get_f32_array_from_kv(gguf_context *gguf_ctx, const char *key);

    std::vector<i32> get_i32_array_from_kv(gguf_context *gguf_ctx, const char *key);

    int composite_model(gguf_context *gguf_ctx);

    int load_input_tokens_to_tensor(std::vector<token_id> &input, InferenceStage stage);

    int init_hyper_param(gguf_context *gguf_ctx);

    int init_kv_cache(gguf_context *gguf_ctx);

    int refresh_kv_cache();

    int update_kv_cache(std::vector<token_id> &input, InferenceStage stage);

    ggml_tensor *cgraph_build_norm(ggml_context *pContext, ggml_tensor *pTensor, ggml_tensor *norm);

    ggml_tensor *cgraph_build_kqv(
            struct ggml_context *ctx,
            struct ggml_cgraph *graph,
            struct ggml_tensor *attn_output,
            struct ggml_tensor *q_tensor,
            struct ggml_tensor *kq_mask,
            i64 n_ctx,
            i32 n_tokens,
            i32 n_kv,
            float kq_scale,
            int index_layer);

    void cgraph_build_kv_store(
            ggml_context *ctx,
            ggml_cgraph *graph,
            ggml_tensor *k_tensor,
            ggml_tensor *v_tensor,
            i64 n_ctx,
            i32 n_tokens,
            i32 kv_head,
            i64 index_layer);

    ggml_tensor *cgraph_build_kv(
            ggml_context *ctx,
            ggml_cgraph *cgraph,
            ggml_tensor *attn_output,
            ggml_tensor *k_tensor,
            ggml_tensor *v_tensor,
            ggml_tensor *q_tensor,
            ggml_tensor *kq_mask,
            i64 n_ctx,
            i32 n_tokens,
            i32 kv_head,
            i32 n_kv,
            float kq_scale,
            int index_layer);

    ggml_tensor *
    cgraph_build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *up, ggml_tensor *gate, ggml_tensor *down);

    token_id greedy_sample(const ggml_tensor *model_output);
};

#endif //GEMMA_MODEL_H
