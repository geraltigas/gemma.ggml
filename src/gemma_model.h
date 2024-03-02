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
    ggml_tensor *inp_embd;
    ggml_tensor *inp_pos;
    ggml_tensor *inp_KQ_mask;
    ggml_tensor *inp_KV_mask;
    ggml_tensor *inp_K_shift;
    ggml_tensor *inp_mean;
    ggml_tensor *inp_cls;
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
    std::map<str, token_id> token_id_map;
    std::map<str, token_type> token_type_map;
public:
    token_id special_bos_id;
    token_id special_eos_id;
    token_id special_unk_id;
    token_id special_sep_id;
    token_id special_pad_id;
};


struct kv_cell {
    i32 pos   = -1;
    i32 delta = 0;

    std::set<i32> seq_id;

    bool has_seq_id(const i32 & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};


struct kv_cache {
    bool has_shift = false;
    bool do_defrag = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    ~kv_cache() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};


class GemmaModel {
    ggml_context *input_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
    ggml_context *compute_ctx = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<u8> compute_meta_buffer;
    int n_kv_pair = 0;
    int n_tensors = 0;
    u32 n_embd = 0;
    u32 n_embd_heads = 0;
    u32 n_kv_cache = 0;

    kv_cache kv_cache;
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
    int load_model_from_file(const char * file_path);
    int load_tokenizer(gguf_context *gguf_ctx);
    int model_warmup();
    int init_input_tensor();
    ggml_tensor *get_tensor(const char * name);
    std::vector<token_id> inference(std::vector<token_id> &input);
private:
    u32 get_u32_from_kv(gguf_context *gguf_ctx, const char * key);
    f32 get_f32_from_kv(gguf_context *gguf_ctx, const char * key);
    str get_str_from_kv(gguf_context *gguf_ctx, const char * key);
    gguf_type get_arr_elem_type(gguf_context *gguf_ctx, const char * key);
    std::vector<str> get_str_arr_from_kv(gguf_context *gguf_ctx, const char * key);
    std::vector<f32> get_f32_array_from_kv(gguf_context *gguf_ctx, const char * key);
    std::vector<i32> get_i32_array_from_kv(gguf_context *gguf_ctx, const char * key);
    int composite_model(gguf_context *gguf_ctx);
    int load_input_tokens_to_tensor(std::vector<token_id> &input);
    int init_kv_cache(gguf_context *gguf_ctx);
    ggml_tensor * cgraph_build_inp_embd(std::vector<token_id> &input, ggml_tensor *tok_embd, ggml_tensor *inp_tokens,
                                        ggml_tensor *inp_embd);
};

#endif //GEMMA_MODEL_H
