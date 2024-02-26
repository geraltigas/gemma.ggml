//
// Created by geraltigas on 2/25/24.
//

#include "gemma_model.h"
#include "macro.h"

#include <ggml_addon.h>
#include <map>
#include <glog/logging.h>
#include <type.h>

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
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        // if (i < 9) {
//        LOG(INFO) << "Loading tensor: " << name;
        // }else if (i == 9) {
        //     LOG(INFO) << "Loading tensors: .........";
        // }
        ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
        tensors[name] = t;
//        n_elements += ggml_nelements(t);
//        n_bytes    += ggml_nbytes(t);
        ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
        tensor_types[name] = type;
        n_type[type]++;
    }

    LOG(INFO) << tensors.size() << " tensors loaded";
//    {
//        const int kid = gguf_find_key(gguf_ctx, "general.file_type");
//        if (kid >= 0) {
//            ftype = static_cast<ggml_type>(gguf_get_val_u32(gguf_ctx, kid));
//        }
//    }
    std::map<gguf_type, int> type_count;

    for (int i = 0; i < n_kv; i++) {
        const char * name           = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type   = gguf_get_kv_type(gguf_ctx, i);
        kv_types[name]              = type;
        kv_index[name]              = i;
        type_count[type]++;
        const std::string type_name =
            type == GGUF_TYPE_ARRAY
            // ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(ctx_gguf, i)), gguf_get_arr_n(ctx_gguf, i))
            // use std lib
        ? std::string(gguf_type_name(type)) + "[" + gguf_type_name(gguf_get_arr_type(gguf_ctx, i)) + "," + std::to_string(gguf_get_arr_n(gguf_ctx, i)) + "]"
        : gguf_type_name(type);

        std::string value          = gguf_kv_to_str(gguf_ctx, i);
        if (constexpr size_t MAX_VALUE_LEN = 40; value.size() > MAX_VALUE_LEN) {
            // value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            // use std lib
            value = value.substr(0, MAX_VALUE_LEN - 3) + "...";
        }
        replace_all(value, "\n", "\\n");
         LOG(INFO) << name << " - kv " << i << " " << type_name << " = " << value;
    }

    for (auto &kv : type_count) {
        LOG(INFO) << "Type: " << gguf_type_name(kv.first) << " count: " << kv.second;
    }

    // for (auto &kv : n_type) {
    //     LOG(INFO) << "Type: " << ggml_type_name(kv.first) << " count: " << kv.second;
    // }

    return 0;
}

u32 GemmaModel::get_u32_from_kv(const char *key) {
    static std::map<std::string, u32> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        u32 temp = gguf_get_val_u32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

f32 GemmaModel::get_f32_from_kv(const char *key) {
    static std::map<std::string, float> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }else {
        float temp = gguf_get_val_f32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

std::string GemmaModel::get_str_from_kv(const char *key) {
    static std::map<std::string, std::string> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }else {
        std::string temp = gguf_get_val_str(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

std::vector<std::string> GemmaModel::get_str_arr_from_kv(const char *key) {
    static std::map<std::string, std::vector<std::string>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }else {
        std::vector<std::string> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_STRING) {
                std::string val = gguf_get_arr_str(gguf_ctx, kv_index[key], j);
                temp.push_back(val);
            }else {
                LOG(ERROR) << "Not a string array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<f32> GemmaModel::get_f32_array_from_kv(const char *key) {
    static std::map<std::string, std::vector<float>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }else {
        std::vector<float> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        const void * data = gguf_get_arr_data(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_FLOAT32) {
                float val = static_cast<const float *>(data)[j];
                temp.push_back(val);
            }else {
                LOG(ERROR) << "Not a float array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<i32> GemmaModel::get_i32_array_from_kv(const char *key) {
    static std::map<std::string, std::vector<i32>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    }else {
        std::vector<i32> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        const void * data = gguf_get_arr_data(gguf_ctx, kv_index[key]);
        for (int j = 0; j < arr_n; j++) {
            if (arr_type == GGUF_TYPE_INT32) {
                i32 val = static_cast<const i32 *>(data)[j];
                temp.push_back(val);
            }else {
                LOG(ERROR) << "Not a int32 array";
                CHECK_RT(-1);
            }
        }
        cache[key] = temp;
        return temp;
    }
}

