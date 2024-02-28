//
// Created by geraltigas on 2/25/24.
//

#include "gemma_model.h"
#include "macro.h"

#include <map>
#include <glog/logging.h>

int GemmaModel::load_model_from_file(const char *file_path) {
    if (file_path == nullptr) {
        return -1;
    }

    gguf_ctx = gguf_init_from_file(file_path, {
            .no_alloc = false,
            .ctx = &ggml_ctx
    });

    n_kv = gguf_get_n_kv(gguf_ctx);
    n_tensors = gguf_get_n_tensors(gguf_ctx);

    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
        MASK(
                LOG(INFO) << "Loading tensor: " << name;
        )
        ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
        tensors[name] = t;
        ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
        tensor_types[name] = type;
        n_type[type]++;
    }

    LOG(INFO) << tensors.size() << " tensors loaded";

    std::map<gguf_type, int> type_count;

    for (int i = 0; i < n_kv; i++) {
        const char *name = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);
        kv_types[name] = type;
        kv_index[name] = i;
        type_count[type]++;
        MASK(
                switch (type) {
                    case GGUF_TYPE_UINT32:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_u32_from_kv(name);
                        break;
                    case GGUF_TYPE_FLOAT32:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_f32_from_kv(name);
                        break;
                    case GGUF_TYPE_STRING:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_str_from_kv(name);
                        break;
                    case GGUF_TYPE_ARRAY:
                        switch (get_arr_elem_type(name)) {
                            case GGUF_TYPE_STRING:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(name)) << "] = "
                                          << get_str_arr_from_kv(name).size();
                                break;
                            case GGUF_TYPE_FLOAT32:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(name)) << "] = "
                                          << get_f32_array_from_kv(name).size();
                                break;
                            case GGUF_TYPE_INT32:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(name)) << "] = "
                                          << get_i32_array_from_kv(name).size();
                                break;
                            default:
                                LOG(ERROR) << "Unknown array type";
                                CHECK_RT(-1);
                        }
                        break;
                    default:
                        LOG(ERROR) << "Unknown type";
                        CHECK_RT(-1);
                }
        )

    }

    LOG(INFO) << kv_index.size() << " kv pairs loaded";

    CHECK_RT(composite_model());

    CHECK_RT(load_tokenizer())

    return 0;
}

u32 GemmaModel::get_u32_from_kv(const char *key) {
    static std::map<str, u32> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        u32 temp = gguf_get_val_u32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

f32 GemmaModel::get_f32_from_kv(const char *key) {
    static std::map<str, float> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        float temp = gguf_get_val_f32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

str GemmaModel::get_str_from_kv(const char *key) {
    static std::map<str, str> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        str temp = gguf_get_val_str(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

std::vector<str> GemmaModel::get_str_arr_from_kv(const char *key) {
    static std::map<str, std::vector<str>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        std::vector<str> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_STRING) {
                str val = gguf_get_arr_str(gguf_ctx, kv_index[key], j);
                temp.push_back(val);
            } else {
                LOG(ERROR) << "Not a string array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<f32> GemmaModel::get_f32_array_from_kv(const char *key) {
    static std::map<str, std::vector<float>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        std::vector<float> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        const void *data = gguf_get_arr_data(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_FLOAT32) {
                float val = static_cast<const float *>(data)[j];
                temp.push_back(val);
            } else {
                LOG(ERROR) << "Not a float array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<i32> GemmaModel::get_i32_array_from_kv(const char *key) {
    static std::map<str, std::vector<i32>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        std::vector<i32> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        const void *data = gguf_get_arr_data(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_INT32) {
                i32 val = static_cast<const i32 *>(data)[j];
                temp.push_back(val);
            } else {
                LOG(ERROR) << "Not a int32 array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

gguf_type GemmaModel::get_arr_elem_type(const char *key) {
    return gguf_get_arr_type(gguf_ctx, kv_index[key]);
}

int GemmaModel::composite_model() {
    int composited_tensor_count = 0;
    CHECK_PTR(tensor_holder.token_embd = get_tensor("token_embd.weight"));
    CHECK_PTR(tensor_holder.output_norm = get_tensor("output_norm.weight"));
    CHECK_PTR(tensor_holder.output = get_tensor("output_norm.weight"));
    composited_tensor_count += 2;
    tensor_holder.layer_num = get_u32_from_kv("gemma.block_count");
    tensor_holder.layers.resize(tensor_holder.layer_num);

    char * name = new char[64];
    for (int i = 0; i < tensor_holder.layer_num; i++) {
        GemmaLayer& layer = tensor_holder.layers.emplace_back();
        snprintf(name, 64, "blk.%d.attn_output.weight", i);
        CHECK_PTR(layer.attn_output = get_tensor(name));
        snprintf(name, 64, "blk.%d.attn_k.weight", i);
        CHECK_PTR(layer.attn_k = get_tensor(name));
        snprintf(name, 64, "blk.%d.attn_v.weight", i);
        CHECK_PTR(layer.attn_v = get_tensor(name));
        snprintf(name, 64, "blk.%d.attn_q.weight", i);
        CHECK_PTR(layer.attn_q = get_tensor(name));
        snprintf(name, 64, "blk.%d.ffn_gate.weight", i);
        CHECK_PTR(layer.ffn_gate = get_tensor(name));
        snprintf(name, 64, "blk.%d.ffn_up.weight", i);
        CHECK_PTR(layer.ffn_up = get_tensor(name));
        snprintf(name, 64, "blk.%d.ffn_down.weight", i);
        CHECK_PTR(layer.ffn_down = get_tensor(name));
        snprintf(name, 64, "blk.%d.attn_norm.weight", i);
        CHECK_PTR(layer.attn_norm = get_tensor(name));
        snprintf(name, 64, "blk.%d.ffn_norm.weight", i);
        CHECK_PTR(layer.ffn_norm = get_tensor(name));
        composited_tensor_count += 9;
    }

    LOG(INFO) << "Composited " << composited_tensor_count << " tensors";

    return 0;
}

ggml_tensor *GemmaModel::get_tensor(const char *name) {
    if (tensors.find(name) != tensors.end()) {
        return tensors[name];
    } else {
        return nullptr;
    }
}

int GemmaModel::model_warmup() {
    std::vector<token_id> warmup_prompt = {tokenizer.special_bos_id, tokenizer.special_eos_id};
    std::vector<token_id> warmup_output = inference(warmup_prompt);
    if (warmup_output.empty()) {
        LOG(ERROR) << "Warmup failed";
        return -1;
    }
    return 0;
}

int GemmaModel::load_tokenizer() {
    tokenizer.special_bos_id = 2;
    tokenizer.special_eos_id = 1;
    tokenizer.special_unk_id = 3;
    tokenizer.special_sep_id = -1;
    tokenizer.special_pad_id = 0;

    tokenizer.tokens = get_str_arr_from_kv("tokenizer.ggml.tokens");
    auto ids = get_f32_array_from_kv("tokenizer.ggml.scores");
    for (int i = 0; i < ids.size(); i++) {
        tokenizer.token_id_map[tokenizer.tokens[i]] = ids[i];
    }
    auto types = get_i32_array_from_kv("tokenizer.ggml.token_type");
    for (int i = 0; i < types.size(); i++) {
        tokenizer.token_type_map[tokenizer.tokens[i]] = types[i];
    }

    // find token with bos in it
    for (const auto & token : tokenizer.tokens) {
        if (token.find("<bos>") != std::string::npos) {
            LOG(INFO) << "Found <bos> token: " << token << " id: " << tokenizer.token_id_map[token];
        }
    }

    return 0;
}

std::vector<token_id> GemmaModel::inference(std::vector<token_id> &input) {

    return std::vector<token_id>();
}
