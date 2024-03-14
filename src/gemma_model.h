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
#include <type.h>
#include <ggml-backend.h>
#include <ggml-backend.h>

enum class inference_stage {
    PREFILL,
    DECODE
};

struct gemma_layer {
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

struct gemma_input_tensor_holder {
    ggml_tensor *inp_tokens;
    ggml_tensor *inp_pos;
    ggml_tensor *inp_KQ_mask;
    ggml_backend_buffer_type_t input_tensor_buffer_type = nullptr;
    ggml_backend_buffer_t input_tensor_buffer = nullptr;
};

struct gemma_tensor_holder {
    u32 layer_num;

    ggml_tensor *token_embd;
    ggml_tensor *output_norm;
    ggml_tensor *output;

    std::vector<gemma_layer> layers;
};

struct gemma_tokenizer {
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
};


struct kv_cache {

    u32 head = 0;
    u32 size = 0;

    u32 n = 0;

    std::vector<kv_cell> cells;

    std::vector<ggml_tensor *> k_layer;
    std::vector<ggml_tensor *> v_layer;

    ggml_backend_buffer_type_t kv_cache_buffer_type = nullptr;
    ggml_backend_buffer_t kv_cache_buffer = nullptr;
};

struct hyper_param {
    u32 n_layer = 0;
    u32 n_header_kv_pair = 0;
    u32 n_tensors = 0;
    u32 n_embed = 0;
    u32 n_head = 0;
    u32 n_embed_heads = 0;
    f32 f_norm_rms_eps = 0;
    u32 origin_ctx_len = 0;
    u32 n_head_kv;
};

class gemma_model {
    // hyper param
    hyper_param _hyper_param;

    // input tensor context
    ggml_context *input_ctx = nullptr;
    gemma_input_tensor_holder _input_tensor_holder;

    // model weight context
    ggml_context *weight_ctx = nullptr;
    std::map<str, ggml_tensor *> tensors;
    gemma_tensor_holder _tensor_holder;

    // compute graph context
    ggml_context *compute_ctx = nullptr;

    // kv cache context
    ggml_context *kv_ctx = nullptr;
    kv_cache _kv_cache;
    std::map<str, i32> kv_index;

    // tokenizer
    gemma_tokenizer _tokenizer;

public:

    // inference
    void inference(std::vector<token_id> &input, inference_stage stage);

    void begin_one_round_inference();

    // load
    int load_model_from_file(const char *file_path);

    // init
    int init_input_tensor();
    int init_kv_cache();

private:
    // load
    int load_tensors_from_ctx(gguf_context *gguf_ctx, int n_tensors);

    int load_header_kv_pair_from_ctx(gguf_context *gguf_ctx, int n_kv_pair);

    int load_tokenizer(gguf_context *gguf_ctx);

    int init_hyper_param(gguf_context *gguf_ctx);

    int composite_model(gguf_context *gguf_ctx);

    // inference
    int model_warmup();

    ggml_cgraph *build_compute_graph(std::vector<token_id> &input);

    int load_input_tokens_to_tensor(std::vector<token_id> &input, inference_stage stage) const;

    int refresh_kv_cache();

    int update_kv_cache(std::vector<token_id> &input, [[maybe_unused]] inference_stage stage);

    int reset_compute_context();

    static token_id greedy_sample(const ggml_tensor *model_output);

    // tensor
    ggml_tensor *get_tensor(const char *name);

    // header kv pair
    u32 get_u32_from_kv(gguf_context *gguf_ctx, const char *key);

    f32 get_f32_from_kv(gguf_context *gguf_ctx, const char *key);

    str get_str_from_kv(gguf_context *gguf_ctx, const char *key);

    gguf_type get_arr_elem_type(gguf_context *gguf_ctx, const char *key);

    std::vector<str> get_str_arr_from_kv(gguf_context *gguf_ctx, const char *key);

    std::vector<f32> get_f32_array_from_kv(gguf_context *gguf_ctx, const char *key);

    std::vector<i32> get_i32_array_from_kv(gguf_context *gguf_ctx, const char *key);

    // build compute graph
    ggml_tensor *graph_build_norm(ggml_context *pContext, ggml_tensor *pTensor, ggml_tensor *norm) const;

    ggml_tensor *graph_build_kqv(
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

    void graph_build_kv_store(
            ggml_context *ctx,
            ggml_cgraph *graph,
            ggml_tensor *k_tensor,
            ggml_tensor *v_tensor,
            i64 n_ctx,
            i32 n_tokens,
            i32 kv_head,
            i64 index_layer);

    ggml_tensor *graph_build_kv(
            ggml_context *ctx,
            ggml_cgraph *graph,
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

    static ggml_tensor *
    graph_build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *up, ggml_tensor *gate, ggml_tensor *down);
};

#endif //GEMMA_MODEL_H
