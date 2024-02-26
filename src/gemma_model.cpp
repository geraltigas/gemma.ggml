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

