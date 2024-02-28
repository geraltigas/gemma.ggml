//
// Created by geraltigas on 2/25/24.
//

#ifndef GEMMA_MODEL_H
#define GEMMA_MODEL_H

#include <ggml.h>
#include <map>
#include <vector>
#include <string>
#include "type.h"

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

struct GemmaTensorHolder {
    u32 layer_num;

    ggml_tensor *token_embd;
    ggml_tensor *output_norm;
    ggml_tensor *output;

    std::vector<GemmaLayer> layers;
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

class GemmaModel {
    gguf_context *gguf_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
    int n_kv = 0;
    int n_tensors = 0;
//    int gguf_version = 0;
//    int64_t n_elements = 0;
//    size_t n_bytes = 0;
    std::map<ggml_type, u32> n_type;
    std::map<str, ggml_tensor *> tensors;
    std::map<str, ggml_type> tensor_types;
    std::map<str, gguf_type> kv_types;
    std::map<str, i32> kv_index;

    GemmaTensorHolder tensor_holder;

    GemmaTokenizer tokenizer;

public:
    int load_model_from_file(const char * file_path);
    int load_tokenizer();
    int model_warmup();
    ggml_tensor *get_tensor(const char * name);
private:
    u32 get_u32_from_kv(const char * key);
    f32 get_f32_from_kv(const char * key);
    str get_str_from_kv(const char * key);
    std::vector<token_id> inference(std::vector<token_id> &input);
    gguf_type get_arr_elem_type(const char * key);
    std::vector<str> get_str_arr_from_kv(const char * key);
    std::vector<f32> get_f32_array_from_kv(const char * key);
    std::vector<i32> get_i32_array_from_kv(const char * key);
    int composite_model();
};

#endif //GEMMA_MODEL_H
