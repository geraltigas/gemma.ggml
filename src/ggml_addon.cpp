//
// Created by geraltigas on 2/26/24.
//

#include "ggml_addon.h"

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(static_cast<const uint8_t *>(data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(static_cast<const int8_t *>(data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(static_cast<const uint16_t *>(data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(static_cast<const int16_t *>(data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(static_cast<const uint32_t *>(data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(static_cast<const int32_t *>(data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(static_cast<const uint64_t *>(data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(static_cast<const int64_t *>(data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(static_cast<const float *>(data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(static_cast<const double *>(data)[i]);
        case GGUF_TYPE_BOOL:    return static_cast<const bool *>(data)[i] ? "true" : "false";
        // default:                return format("unknown type %d", type);
        default:                return "unknown type " + std::to_string(type);
    }
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
        {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
            int arr_n = gguf_get_arr_n(ctx_gguf, i);
            const void * data = gguf_get_arr_data(ctx_gguf, i);
            std::stringstream ss;
            ss << "[";
            for (int j = 0; j < arr_n; j++) {
                if (arr_type == GGUF_TYPE_STRING) {
                    std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                    // escape quotes
                    replace_all(val, "\\", "\\\\");
                    replace_all(val, "\"", "\\\"");
                    ss << '"' << val << '"';
                } else if (arr_type == GGUF_TYPE_ARRAY) {
                    ss << "???";
                } else {
                    ss << gguf_data_to_str(arr_type, data, j);
                }
                if (j < arr_n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
            return ss.str();
        }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}
