//
// Created by geraltigas on 2/25/24.
//

#include "gemma_model.h"
#include "macro.h"
#include <tensor_dump.h>

#include <map>
#include <algorithm>
#include <glog/logging.h>
#include <cmath>
#include <fstream>

int GemmaModel::load_model_from_file(const char *file_path) {
    if (file_path == nullptr) {
        return -1;
    }

    gguf_context *gguf_ctx = nullptr;

    gguf_ctx = gguf_init_from_file(file_path, {
            .no_alloc = false,
            .ctx = &ggml_ctx
    });

    // hyper parameters
    int n_kv_pair = gguf_get_n_kv(gguf_ctx);
    int n_tensors = gguf_get_n_tensors(gguf_ctx);

    CHECK_PTR(gguf_ctx);

//    buffer_type = ggml_backend_cpu_buffer_type();
//    CHECK_PTR(buffer_type);
//    buffer = ggml_backend_alloc_ctx_tensors_from_buft(ggml_ctx, buffer_type);
//    CHECK_PTR(buffer);
//    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    LOG(INFO) << "model weight tensor ctx size: " << ggml_get_mem_size(ggml_ctx) / 1024.0 / 1024.0 << " MiB";

    // allocate cgraph mem
    compute_meta_buffer.resize(ggml_tensor_overhead() * CGRAPH_MAX_NODE_NUM + ggml_graph_overhead());

    // init compute ctx
    struct ggml_init_params params = {
            /*.mem_size   =*/ compute_meta_buffer.size(),
            /*.mem_buffer =*/ compute_meta_buffer.data(),
            /*.no_alloc   =*/ true,
    };
    compute_ctx = ggml_init(params);

    //    LLAMA_LOG_INFO("%s: %10s buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0)
    // USE LOG


    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
//        MASK(
//                LOG(INFO) << "Loading tensor: " << name;
//        )
        ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
//        CHECK_PTR(t->buffer);
        tensors[name] = t;
        ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
        tensor_types[name] = type;
        n_type[type]++;
    }

    LOG(INFO) << tensors.size() << " tensors loaded";

    std::map<gguf_type, int> type_count;

    for (int i = 0; i < n_kv_pair; i++) {
        const char *name = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);
        kv_types[name] = type;
        kv_index[name] = i;
        type_count[type]++;
        MASK(
                switch (type) {
                    case GGUF_TYPE_UINT32:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_u32_from_kv(gguf_ctx, name);
                        break;
                    case GGUF_TYPE_FLOAT32:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_f32_from_kv(gguf_ctx, name);
                        break;
                    case GGUF_TYPE_STRING:
                        LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << " = "
                                  << get_str_from_kv(gguf_ctx, name);
                        break;
                    case GGUF_TYPE_ARRAY:
                        switch (get_arr_elem_type(gguf_ctx, name)) {
                            case GGUF_TYPE_STRING:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(gguf_ctx, name)) << "] = "
                                          << get_str_arr_from_kv(gguf_ctx, name).size();
                                break;
                            case GGUF_TYPE_FLOAT32:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(gguf_ctx, name)) << "] = "
                                          << get_f32_array_from_kv(gguf_ctx, name).size();
                                break;
                            case GGUF_TYPE_INT32:
                                LOG(INFO) << name << " - kv " << i << " " << gguf_type_name(type) << "["
                                          << gguf_type_name(get_arr_elem_type(gguf_ctx, name)) << "] = "
                                          << get_i32_array_from_kv(gguf_ctx, name).size();
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

    CHECK_RT(init_hyper_param(gguf_ctx));

    CHECK_RT(init_kv_cache(gguf_ctx))

    CHECK_RT(composite_model(gguf_ctx));

    CHECK_RT(load_tokenizer(gguf_ctx))

    return 0;
}

u32 GemmaModel::get_u32_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, u32> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        u32 temp = gguf_get_val_u32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

f32 GemmaModel::get_f32_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, float> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        float temp = gguf_get_val_f32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

str GemmaModel::get_str_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, str> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        str temp = gguf_get_val_str(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

std::vector<str> GemmaModel::get_str_arr_from_kv(gguf_context *gguf_ctx, const char *key) {
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

std::vector<f32> GemmaModel::get_f32_array_from_kv(gguf_context *gguf_ctx, const char *key) {
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

std::vector<i32> GemmaModel::get_i32_array_from_kv(gguf_context *gguf_ctx, const char *key) {
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

gguf_type GemmaModel::get_arr_elem_type(gguf_context *gguf_ctx, const char *key) {
    return gguf_get_arr_type(gguf_ctx, kv_index[key]);
}

int GemmaModel::composite_model(gguf_context *gguf_ctx) {
    int composited_tensor_count = 0;
    CHECK_PTR(tensor_holder.token_embd = get_tensor("token_embd.weight"));
    CHECK_PTR(tensor_holder.output_norm = get_tensor("output_norm.weight"));
    CHECK_PTR(tensor_holder.output = get_tensor("output_norm.weight"));
    composited_tensor_count += 2;
    tensor_holder.layer_num = get_u32_from_kv(gguf_ctx, "gemma.block_count");

    char *name = new char[64];
    for (int i = 0; i < tensor_holder.layer_num; i++) {
        GemmaLayer &layer = tensor_holder.layers.emplace_back();
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

    LOG(INFO) << "composited " << composited_tensor_count << " tensors";

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
    std::vector<token_id> warmup_output = inference(warmup_prompt, InferenceStage::PREFILL);
    if (warmup_output.empty()) {
        LOG(ERROR) << "Warmup failed";
        return -1;
    }
    return 0;
}

int GemmaModel::load_tokenizer(gguf_context *gguf_ctx) {
    tokenizer.special_bos_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.bos_token_id");
    tokenizer.special_eos_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.eos_token_id");
    tokenizer.special_unk_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.unknown_token_id");
    tokenizer.special_pad_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.padding_token_id");
    tokenizer.special_sep_id = -1;

    tokenizer.tokens = get_str_arr_from_kv(gguf_ctx, "tokenizer.ggml.tokens");
    auto ids = get_f32_array_from_kv(gguf_ctx, "tokenizer.ggml.scores");
    for (int i = 0; i < ids.size(); i++) {
        tokenizer.token_id_map[tokenizer.tokens[i]] = ids[i];
    }
    auto types = get_i32_array_from_kv(gguf_ctx, "tokenizer.ggml.token_type");
    for (int i = 0; i < types.size(); i++) {
        tokenizer.token_type_map[tokenizer.tokens[i]] = types[i];
    }

    // find token with bos in it
    for (const auto &token: tokenizer.tokens) {
        if (token.find("<bos>") != std::string::npos) {
            LOG(INFO) << "found <bos> token: " << token << " id: " << tokenizer.token_id_map[token];
        }
    }

    // print id 2
    LOG(INFO) << "token id 2: " << tokenizer.tokens[2] << " type: " << tokenizer.token_type_map[tokenizer.tokens[2]];
    // print id 25612
    LOG(INFO) << "token id 25612: " << tokenizer.tokens[25612] << " type: "
              << tokenizer.token_type_map[tokenizer.tokens[25612]];

    return 0;
}

std::vector<token_id> GemmaModel::inference(std::vector<token_id> &input, InferenceStage stage) {
    update_kv_cache();
    CHECK_RT(load_input_tokens_to_tensor(input, stage));
    ggml_cgraph *cgraph = ggml_new_graph(compute_ctx);
    CHECK_PTR(cgraph);

    struct ggml_tensor *cur;
    ASSERT_MSG(input.size() > 0, "input size must be greater than 0");

    ggml_tensor *inp_tokens_v = ggml_view_1d(compute_ctx, input_tensor_holder.inp_tokens, input.size(), 0);
    ggml_set_name(inp_tokens_v, "inp_tokens (view)");
    ggml_tensor *inpL = ggml_get_rows(compute_ctx, tensor_holder.token_embd, inp_tokens_v);

    inpL = ggml_scale(compute_ctx, inpL, sqrtf(hyper_param.n_embd));
    ggml_tensor *inp_pos = ggml_view_1d(compute_ctx, input_tensor_holder.inp_pos, input.size(), 0);
    ggml_set_name(inp_pos, "inp_pos (view)");

    ggml_tensor *KQ_mask = ggml_view_2d(compute_ctx, input_tensor_holder.inp_KQ_mask, _kv_cache.n, input.size(),
                                        _kv_cache.n * ggml_type_size(input_tensor_holder.inp_KQ_mask->type), 0);
    ggml_set_name(KQ_mask, "inp_KQ_mask (view)");

    LOG(INFO) << "kv_cache.n: " << _kv_cache.n;

    for (int il = 0; il < hyper_param.n_layer; ++il) {

        cur = cgraph_build_norm(compute_ctx, inpL, tensor_holder.layers[il].attn_norm);
        {
            ggml_tensor *Qcur = ggml_mul_mat(compute_ctx, tensor_holder.layers[il].attn_q, cur);

            ggml_tensor *Kcur = ggml_mul_mat(compute_ctx, tensor_holder.layers[il].attn_k, cur);

            ggml_tensor *Vcur = ggml_mul_mat(compute_ctx, tensor_holder.layers[il].attn_v, cur);

            Qcur = ggml_rope_custom(
                    compute_ctx,
                    ggml_reshape_3d(compute_ctx, Qcur, hyper_param.n_embd_heads, hyper_param.n_head, input.size()),
                    inp_pos,
                    hyper_param.n_embd_heads, DEFAULT_ROPE_TYPE, 0, hyper_param.origin_ctx_len, DEFAULT_FREQ_BASE,
                    DEFAULT_FREQ_SCALE,
                    DEFAULT_EXT_FACTOR, DEFAULT_ATTN_FACTOR, DEFAULT_BETA_FAST, DEFAULT_BETA_SLOW);

            Qcur = ggml_scale(compute_ctx, Qcur, 1.0f / sqrtf(float(hyper_param.n_embd_heads)));


            Kcur = ggml_rope_custom(
                    compute_ctx, ggml_reshape_3d(compute_ctx, Kcur, hyper_param.n_embd_heads, hyper_param.n_head_kv,
                                                 input.size()), inp_pos,
                    hyper_param.n_embd_heads, DEFAULT_ROPE_TYPE, 0, hyper_param.origin_ctx_len, DEFAULT_FREQ_BASE,
                    DEFAULT_FREQ_SCALE,
                    DEFAULT_EXT_FACTOR, DEFAULT_ATTN_FACTOR, DEFAULT_BETA_FAST, DEFAULT_BETA_SLOW);

            cur = llm_build_kv(compute_ctx, cgraph, tensor_holder.layers[il].attn_output, Kcur, Vcur, Qcur, KQ_mask,
                               DEFAULT_CTX_NUM,
                               input.size(), _kv_cache.head, _kv_cache.n, 1.0f, il);
        }

        ggml_tensor *sa_out = ggml_add(compute_ctx, cur, inpL);
        cur = cgraph_build_norm(compute_ctx, sa_out, tensor_holder.layers[il].ffn_norm);

        {
            cur = cgraph_build_ffn(compute_ctx, cur, tensor_holder.layers[il].ffn_up, tensor_holder.layers[il].ffn_gate,
                                   tensor_holder.layers[il].ffn_down);
        }
        cur = ggml_add(compute_ctx, cur, sa_out);

        inpL = cur;
    }

    cur = inpL;

    cur = cgraph_build_norm(compute_ctx, cur, tensor_holder.output_norm);

    cur = ggml_mul_mat(compute_ctx, tensor_holder.output, cur);

    ggml_build_forward_expand(cgraph, cur);

    auto print_all = [](ggml_cgraph *gf) {
        auto to_file = "/home/geraltigas/Desktop/gemma.ggml/tensor_dump/tensor_in_source_cgraph";
        std::ofstream file(to_file);
        for (int i = 0; i < gf->n_nodes; i++) {
            file << "node[" << i << "]: " << gf->nodes[i]->name << "\n";
        }
    };

    print_all(cgraph);

    auto check_tensor_value = [&]() {
        ggml_tensor **tensor_in_cgraph = cgraph->nodes;
        int num_tensor_in_cgraph = cgraph->n_nodes;
        std::map<std::string, std::string> tensor_dump_list = get_tensor_dump_list();

        for (const auto &item: tensor_dump_list) {
            const char *name = item.first.c_str();
            const char *tensor_name = item.second.c_str();
            ggml_tensor *tensor = nullptr;
            for (int i = 0; i < num_tensor_in_cgraph; i++) {
                if (strcmp(tensor_in_cgraph[i]->name, tensor_name) == 0) {
                    tensor = tensor_in_cgraph[i];
                }
            }
            if (tensor == nullptr) {
                LOG(ERROR) << "tensor " << tensor_name << " not found in cgraph";
                continue;
            }
            dump_tensor(name, tensor);
            compare_tensors(name);
        }
    };

    check_tensor_value();

    return std::vector<token_id>();
}

int GemmaModel::load_input_tokens_to_tensor(std::vector<token_id> &input, InferenceStage stage) {
    CHECK_PTR(input_tensor_holder.inp_tokens);

    input_tensor_holder.buffer_type = ggml_backend_cpu_buffer_type();
    input_tensor_holder.buffer = ggml_backend_alloc_ctx_tensors_from_buft(input_ctx, input_tensor_holder.buffer_type);
    ggml_backend_buffer_clear(input_tensor_holder.buffer, 0);

    // log buffer size
    LOG(INFO) << "input buffer size: " << ggml_backend_buffer_name(input_tensor_holder.buffer) << " = "
              << ggml_backend_buffer_get_size(input_tensor_holder.buffer) / 1024.0 / 1024.0 << " MiB";

    ggml_backend_tensor_set(input_tensor_holder.inp_tokens, input.data(), 0,
                            input.size() * ggml_element_size(input_tensor_holder.inp_tokens));

    std::vector<i32> pos;

    switch (stage) {
        case InferenceStage::PREFILL:
            pos.resize(input.size());
            for (int i = 0; i < input.size(); i++) {
                pos[i] = i;
            }
            break;
        case InferenceStage::DECODE:
            pos.resize(1);
            pos[0] = input.size();
            break;
    }

    CHECK_PTR(input_tensor_holder.inp_pos);
    ggml_backend_tensor_set(input_tensor_holder.inp_pos, pos.data(), 0,
                            pos.size() * ggml_element_size(input_tensor_holder.inp_pos));

    return 0;
}

int GemmaModel::init_input_tensor() {
    ggml_init_params init_params = {
            /* .mem_size   */ ggml_tensor_overhead() * 8,
            /* .mem_buffer */ nullptr,
            /* .no_alloc   */ true,
    };
    CHECK_PTR(input_ctx = ggml_init(init_params));
    CHECK_PTR(input_tensor_holder.inp_tokens = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE));
//    CHECK_PTR(input_tensor_holder.inp_embd = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, hyper_param.n_embd,
//                                                                DEFAULT_BATCH_SIZE));
    CHECK_PTR(input_tensor_holder.inp_pos = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE));
    CHECK_PTR(input_tensor_holder.inp_KQ_mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, DEFAULT_CTX_NUM,
                                                                   DEFAULT_BATCH_SIZE));
//    CHECK_PTR(input_tensor_holder.inp_KV_mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, DEFAULT_CTX_NUM,
//                                                                   DEFAULT_BATCH_SIZE));
//    CHECK_PTR(input_tensor_holder.inp_K_shift = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_CTX_NUM));
//    CHECK_PTR(input_tensor_holder.inp_mean = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, DEFAULT_BATCH_SIZE,
//                                                                DEFAULT_BATCH_SIZE));
//    CHECK_PTR(input_tensor_holder.inp_cls = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE));


    return 0;
}

int GemmaModel::init_kv_cache(gguf_context *gguf_ctx) {
    _kv_cache.size = DEFAULT_CTX_NUM;
    u32 n_embd_k_gqa = hyper_param.n_embd_heads;
    u32 n_embd_v_gqa = hyper_param.n_embd_heads;
    u32 n_layer = hyper_param.n_layer;

    _kv_cache.has_shift = false;

    _kv_cache.head = 0;
    _kv_cache.size = DEFAULT_CTX_NUM;
    _kv_cache.used = 0;

    _kv_cache.type_k = DEFAULT_KV_CACHE_TYPE;
    _kv_cache.type_v = DEFAULT_KV_CACHE_TYPE;

    _kv_cache.cells.clear();
    _kv_cache.cells.resize(DEFAULT_CTX_NUM);

    struct ggml_init_params params = {
            /*.mem_size   =*/ 2u * n_layer * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
    };

    CHECK_PTR(kv_ctx = ggml_init(params))

    _kv_cache.k_l.reserve(n_layer);
    _kv_cache.v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor *k = ggml_new_tensor_1d(kv_ctx, _kv_cache.type_k, n_embd_k_gqa * DEFAULT_CTX_NUM);
        ggml_tensor *v = ggml_new_tensor_1d(kv_ctx, _kv_cache.type_v, n_embd_v_gqa * DEFAULT_CTX_NUM);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        _kv_cache.k_l.push_back(k);
        _kv_cache.v_l.push_back(v);
    }

    _kv_cache.buffer_type = ggml_backend_cpu_buffer_type();
    _kv_cache.buffer = ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx, _kv_cache.buffer_type);
//    ggml_backend_buffer_clear(buffer, 0);


    CHECK_PTR(_kv_cache.buffer);
    LOG(INFO) << "KV buffer size: " << ggml_backend_buffer_name(_kv_cache.buffer) << " = "
              << ggml_backend_buffer_get_size(_kv_cache.buffer) / 1024.0 / 1024.0 << " MiB";

    return 0;
}

int GemmaModel::init_hyper_param(gguf_context *gguf_ctx) {
    hyper_param.n_layer = get_u32_from_kv(gguf_ctx, "gemma.block_count");
    hyper_param.n_kv_pair = gguf_get_n_kv(gguf_ctx);
    hyper_param.n_tensors = gguf_get_n_tensors(gguf_ctx);
    hyper_param.n_embd = get_u32_from_kv(gguf_ctx, "gemma.embedding_length");
    hyper_param.n_head = get_u32_from_kv(gguf_ctx, "gemma.attention.head_count");
    hyper_param.n_embd_heads = get_u32_from_kv(gguf_ctx, "gemma.embedding_length") /
                               get_u32_from_kv(gguf_ctx, "gemma.attention.head_count");
    hyper_param.f_norm_rms_eps = get_f32_from_kv(gguf_ctx, "gemma.attention.layer_norm_rms_epsilon");
    hyper_param.origin_ctx_len = get_u32_from_kv(gguf_ctx, "gemma.context_length");
    hyper_param.n_head_kv = get_u32_from_kv(gguf_ctx, "gemma.attention.head_count_kv");
    return 0;
}

static int32_t kv_cache_cell_max(const struct kv_cache &cache) {
    for (uint32_t i = cache.size - 1; i > 0; --i) {
        if (cache.cells[i].pos >= 0 && !cache.cells[i].is_empty()) {
            return i + 1;
        }
    }

    return 0;
}

int GemmaModel::update_kv_cache() {
    _kv_cache.n = std::min((int32_t) DEFAULT_CTX_NUM, std::max(32, GGML_PAD(kv_cache_cell_max(_kv_cache), 32)));
    return 0;
}

ggml_tensor *GemmaModel::cgraph_build_norm(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *norm) {
    cur = ggml_rms_norm(ctx, cur, hyper_param.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, norm);
    return cur;
}

ggml_tensor *GemmaModel::cgraph_build_ffn(
        ggml_context *ctx,
        ggml_tensor *cur,
        ggml_tensor *up,
        ggml_tensor *gate,
        ggml_tensor *down) {
    ggml_tensor *tmp = ggml_mul_mat(ctx, up, cur);
    cur = ggml_mul_mat(ctx, gate, cur);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_mul(ctx, cur, tmp);
    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

ggml_tensor *
GemmaModel::cgraph_build_kv(ggml_context *ctx, ggml_cgraph *cgraph, ggml_tensor *q_tensor, ggml_tensor *k_tensor,
                            ggml_tensor *v_tensor,
                            int index_layer) {
    ggml_build_forward_expand(cgraph, q_tensor);
    ggml_build_forward_expand(cgraph, k_tensor);
    ggml_build_forward_expand(cgraph, v_tensor);

    ggml_tensor *v_cur_t = ggml_transpose(ctx,
                                          ggml_reshape_2d(ctx, v_tensor, hyper_param.n_embd_heads, DEFAULT_TOKEN_NUM));

    ggml_tensor *k_cache_view = ggml_view_1d(ctx, _kv_cache.k_l[index_layer],
                                             DEFAULT_TOKEN_NUM * hyper_param.n_embd_heads,
                                             (ggml_row_size(_kv_cache.k_l[index_layer]->type,
                                                            hyper_param.n_embd_heads)) * _kv_cache.head);

    ggml_tensor *v_cache_view = ggml_view_2d(ctx, _kv_cache.v_l[index_layer], DEFAULT_TOKEN_NUM,
                                             hyper_param.n_embd_heads,
                                             (DEFAULT_CTX_NUM) * ggml_element_size(_kv_cache.v_l[index_layer]),
                                             (_kv_cache.head) * ggml_element_size(_kv_cache.v_l[index_layer]));

    ggml_build_forward_expand(cgraph, ggml_cpy(ctx, k_tensor, k_cache_view));
    ggml_build_forward_expand(cgraph, ggml_cpy(ctx, v_cur_t, v_cache_view));

    ggml_tensor *q = ggml_permute(ctx, q_tensor, 0, 2, 1, 3);

//    ggml_tensor *k =
//            ggml_view_3d(ctx, _kv_cache.k_l[index_layer],
//                         hyper_param.n_embd_heads, _kv_cache.n, n_head_kv,
//                         ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
//                         ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
//                         0);
//
//    return cur;
    return nullptr;
}

ggml_tensor *GemmaModel::llm_build_kqv(
        struct ggml_context *ctx,
        struct ggml_cgraph *graph,
        struct ggml_tensor *attn_output,
        struct ggml_tensor *q_tensor,
        struct ggml_tensor *kq_mask,
        i64 n_ctx,
        i32 n_tokens,
        i32 n_kv,
        float kq_scale,
        int index_layer) {

    const int64_t n_head = hyper_param.n_head;
    const int64_t n_head_kv = hyper_param.n_head_kv;
    const int64_t n_embd_head_k = hyper_param.n_embd_heads;
    const int64_t n_embd_k_gqa = hyper_param.n_embd_heads;
    const int64_t n_embd_head_v = hyper_param.n_embd_heads;

    ggml_tensor *q = ggml_permute(ctx, q_tensor, 0, 2, 1, 3);

    ggml_tensor *k =
            ggml_view_3d(ctx, _kv_cache.k_l[index_layer],
                         n_embd_head_k, n_kv, n_head_kv,
                         ggml_row_size(_kv_cache.k_l[index_layer]->type, n_embd_k_gqa),
                         ggml_row_size(_kv_cache.k_l[index_layer]->type, n_embd_head_k),
                         0);

    ggml_tensor *kq = ggml_mul_mat(ctx, k, q);

    kq = ggml_soft_max_ext(ctx, kq, kq_mask, nullptr, kq_scale, DEFAULT_F_MAX_ALIBI_BIAS);

    ggml_tensor *v =
            ggml_view_3d(ctx, _kv_cache.v_l[index_layer],
                         n_kv, n_embd_head_v, n_head_kv,
                         ggml_element_size(_kv_cache.v_l[index_layer]) * n_ctx,
                         ggml_element_size(_kv_cache.v_l[index_layer]) * n_ctx * n_embd_head_v,
                         0);

    ggml_tensor *kqv = ggml_mul_mat(ctx, v, kq);

    ggml_tensor *kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

    ggml_tensor *cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_k * n_head, n_tokens);

    ggml_build_forward_expand(graph, cur);

    cur = ggml_mul_mat(ctx, attn_output, cur);

    return cur;
}


void GemmaModel::llm_build_kv_store(
        ggml_context *ctx,
        ggml_cgraph *graph,
        ggml_tensor *k_tensor,
        ggml_tensor *v_tensor,
        i64 n_ctx,
        i32 n_tokens,
        i32 kv_head,
        i64 index_layer) {

    const i64 n_embd_v_gqa = hyper_param.n_embd_heads;
    const i64 n_embd_k_gqa = hyper_param.n_embd_heads;

    ggml_tensor *v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_tensor, n_embd_v_gqa, n_tokens));
    ggml_tensor *k_cache_view = ggml_view_1d(ctx, _kv_cache.k_l[index_layer], n_tokens * n_embd_k_gqa,
                                             (ggml_row_size(_kv_cache.k_l[index_layer]->type, n_embd_k_gqa)) * kv_head);

    ggml_tensor *v_cache_view = ggml_view_2d(ctx, _kv_cache.v_l[index_layer], n_tokens, n_embd_v_gqa,
                                             (n_ctx) * ggml_element_size(_kv_cache.v_l[index_layer]),
                                             (kv_head) * ggml_element_size(_kv_cache.v_l[index_layer]));

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_tensor, k_cache_view));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur_t, v_cache_view));
}

ggml_tensor *GemmaModel::llm_build_kv(
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
        int index_layer) {
    ggml_build_forward_expand(cgraph, q_tensor);
    ggml_build_forward_expand(cgraph, k_tensor);
    ggml_build_forward_expand(cgraph, v_tensor);

    llm_build_kv_store(ctx, cgraph, k_tensor, v_tensor, n_ctx, n_tokens, kv_head, index_layer);
    return llm_build_kqv(ctx, cgraph, attn_output, q_tensor, kq_mask, n_ctx, n_tokens, n_kv, kq_scale, index_layer);
}

ggml_tensor *GemmaModel::get_tensor_from_meta(ggml_context *ctx, ggml_tensor *tensor) {
    ggml_tensor *t = ggml_dup_tensor(ctx, tensor);
    ggml_set_name(t, ggml_get_name(tensor));
    return t;
}

