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
    int layer_num;

    ggml_tensor *token_embd;
    ggml_tensor *output_norm;

    std::vector<GemmaLayer> layers;
};

class GemmaModel {
    gguf_context *gguf_ctx = nullptr;
    ggml_context *ggml_ctx = nullptr;
    int n_kv = 0;
    int n_tensors = 0;
//    int gguf_version = 0;
//    int64_t n_elements = 0;
//    size_t n_bytes = 0;
    ggml_type ftype = GGML_TYPE_F32;



    //useless but may be useful in the future
    std::map<ggml_type, uint32_t> n_type;
    std::map<std::string, ggml_tensor *> tensors;
    std::map<std::string, ggml_type> tensor_types;
    std::map<std::string, gguf_type> kv_types;
    std::map<std::string, int> kv_index;

public:
    int load_model_from_file(const char * file_path);
    u32 get_u32_from_kv(const char * key);
    f32 get_f32_from_kv(const char * key);
    std::string get_str_from_kv(const char * key);
    std::vector<std::string> get_str_arr_from_kv(const char * key);
    std::vector<f32> get_f32_array_from_kv(const char * key);
    std::vector<i32> get_i32_array_from_kv(const char * key);
};

#endif //GEMMA_MODEL_H
