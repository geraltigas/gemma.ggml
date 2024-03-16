//
// Created by geraltigas on 2/25/24.
//

#include <gemma_model.h>
#include <macro.h>
#include <tensor_dump.h>
#include <map>
#include <algorithm>
#include <glog/logging.h>
#include <cmath>
#include <fstream>
#include <chrono>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantParameter"

int gemma_model::load_model_from_file(const char *file_path) {
    if (file_path == nullptr) {
        return -1;
    }

    gguf_context *gguf_ctx = gguf_init_from_file(file_path, {
            .no_alloc = false,
            .ctx = &weight_ctx
    });
    CHECK_PTR(gguf_ctx)
    LOG(INFO) << "model weight tensor ctx size: " << (double) ggml_get_mem_size(weight_ctx) / 1024.0 / 1024.0 << " MiB";

    {
        int n_kv_pair = gguf_get_n_kv(gguf_ctx);
        CHECK_RT(load_header_kv_pair_from_ctx(gguf_ctx, n_kv_pair))
    }

    {
        int n_tensors = gguf_get_n_tensors(gguf_ctx);
        CHECK_RT(load_tensors_from_ctx(gguf_ctx, n_tensors))
        CHECK_RT(composite_model(gguf_ctx))
    }

    {
        CHECK_RT(init_hyper_param(gguf_ctx))
    }

    {
        CHECK_RT(load_tokenizer(gguf_ctx))
    }

    return 0;
}

u32 gemma_model::get_u32_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, u32> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        u32 temp = gguf_get_val_u32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

f32 gemma_model::get_f32_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, float> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        float temp = gguf_get_val_f32(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

str gemma_model::get_str_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, str> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        str temp = gguf_get_val_str(gguf_ctx, kv_index[key]);
        cache[key] = temp;
        return temp;
    }
}

std::vector<str> gemma_model::get_str_arr_from_kv(gguf_context *gguf_ctx, const char *key) {
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
                CHECK_RT(-1)
            }
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<f32> gemma_model::get_f32_array_from_kv(gguf_context *gguf_ctx, const char *key) {
    static std::map<str, std::vector<float>> cache;
    if (cache.find(key) != cache.end()) {
        return cache[key];
    } else {
        std::vector<float> temp;
        const enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, kv_index[key]);
        int arr_n = gguf_get_arr_n(gguf_ctx, kv_index[key]);
        const void *data = gguf_get_arr_data(gguf_ctx, kv_index[key]);
        dump_ptr_data(key, data, arr_n * ggml_type_size(GGML_TYPE_F32));
        if (arr_type == GGUF_TYPE_FLOAT32) {
            for (int j = 0; j < arr_n; j++) {
                float val = static_cast<const float *>(data)[j];
                temp.push_back(val);
            }
        } else {
            LOG(ERROR) << "Not a float array";
            CHECK_RT(-1)
        }
        cache[key] = temp;
        return temp;
    }
}

std::vector<i32> gemma_model::get_i32_array_from_kv(gguf_context *gguf_ctx, const char *key) {
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
                CHECK_RT(-1)
            }
        }
        cache[key] = temp;
        return temp;
    }
}

gguf_type gemma_model::get_arr_elem_type(gguf_context *gguf_ctx, const char *key) {
    return gguf_get_arr_type(gguf_ctx, kv_index[key]);
}

int gemma_model::composite_model(gguf_context *gguf_ctx) {
    int composited_tensor_count = 0;
    CHECK_PTR(_tensor_holder.token_embd = get_tensor("token_embd.weight"))
    CHECK_PTR(_tensor_holder.output_norm = get_tensor("output_norm.weight"))
    CHECK_PTR(_tensor_holder.output = get_tensor("token_embd.weight"))
    composited_tensor_count += 3;
    _tensor_holder.layer_num = get_u32_from_kv(gguf_ctx, "gemma.block_count");

    char *name = new char[64];
    for (int i = 0; i < _tensor_holder.layer_num; i++) {
        gemma_layer &layer = _tensor_holder.layers.emplace_back();
        snprintf(name, 64, "blk.%d.attn_output.weight", i);
        CHECK_PTR(layer.attn_output = get_tensor(name))
        snprintf(name, 64, "blk.%d.attn_k.weight", i);
        CHECK_PTR(layer.attn_k = get_tensor(name))
        snprintf(name, 64, "blk.%d.attn_v.weight", i);
        CHECK_PTR(layer.attn_v = get_tensor(name))
        snprintf(name, 64, "blk.%d.attn_q.weight", i);
        CHECK_PTR(layer.attn_q = get_tensor(name))
        snprintf(name, 64, "blk.%d.ffn_gate.weight", i);
        CHECK_PTR(layer.ffn_gate = get_tensor(name))
        snprintf(name, 64, "blk.%d.ffn_up.weight", i);
        CHECK_PTR(layer.ffn_up = get_tensor(name))
        snprintf(name, 64, "blk.%d.ffn_down.weight", i);
        CHECK_PTR(layer.ffn_down = get_tensor(name))
        snprintf(name, 64, "blk.%d.attn_norm.weight", i);
        CHECK_PTR(layer.attn_norm = get_tensor(name))
        snprintf(name, 64, "blk.%d.ffn_norm.weight", i);
        CHECK_PTR(layer.ffn_norm = get_tensor(name))
        composited_tensor_count += 9;
    }

    LOG(INFO) << "composited " << composited_tensor_count << " tensors";

    return 0;
}

ggml_tensor *gemma_model::get_tensor(const char *name) {
    if (tensors.find(name) != tensors.end()) {
        return tensors[name];
    } else {
        return nullptr;
    }
}

int gemma_model::model_warmup() {
    std::vector<token_id> warmup_prompt = {_tokenizer.special_bos_id, _tokenizer.special_eos_id};
    update_kv_cache(warmup_prompt, inference_stage::PREFILL);
    ggml_cgraph *graph = build_compute_graph(warmup_prompt, inference_stage::PREFILL);
    CHECK_PTR(graph)
    return 0;
}

int gemma_model::load_tokenizer(gguf_context *gguf_ctx) {
    _tokenizer.special_bos_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.bos_token_id");
    _tokenizer.special_eos_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.eos_token_id");
    _tokenizer.special_unk_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.unknown_token_id");
    _tokenizer.special_pad_id = get_u32_from_kv(gguf_ctx, "tokenizer.ggml.padding_token_id");
    _tokenizer.special_sep_id = -1;

    _tokenizer.tokens = get_str_arr_from_kv(gguf_ctx, "tokenizer.ggml.tokens");
    _tokenizer.scores = get_f32_array_from_kv(gguf_ctx, "tokenizer.ggml.scores");
    auto types = get_i32_array_from_kv(gguf_ctx, "tokenizer.ggml.token_type");
    for (int i = 0; i < types.size(); i++) {
        _tokenizer.token_type_map[_tokenizer.tokens[i]] = types[i];
    }

    return 0;
}

void gemma_model::inference(std::vector<token_id> &input, inference_stage stage) {
//    static u32 round = 0;

    update_kv_cache(input, stage);
    CHECK_RT(load_input_tokens_to_tensor(input, stage))
    ggml_cgraph *graph = build_compute_graph(input, stage);
    ggml_graph_compute_with_ctx(compute_ctx, graph, N_THREADS);

//
//    auto print_all = [](ggml_cgraph *gf) {
//        auto to_file = "/home/geraltigas/Desktop/gemma.ggml/tensor_dump/tensor_in_source_cgraph";
//        std::ofstream file(to_file);
//        for (int i = 0; i < gf->n_nodes; i++) {
//            file << "node[" << i << "]: " << gf->nodes[i]->name << "\n";
//        }
//    };
//
//    print_all(graph);
//
//    auto check_tensor_value = [&]() {
//        ggml_tensor **tensor_in_graph = graph->nodes;
//        int num_tensor_in_graph = graph->n_nodes;
//        static std::map<std::string, std::string> tensor_dump_list = get_tensor_dump_list();
//
//        LOG(INFO) << "checking round " << round << " tensors...";
//
//        for (const auto &item: tensor_dump_list) {
//            const char *name = item.first.c_str();
//            const char *tensor_name = item.second.c_str();
//            ggml_tensor *tensor = nullptr;
//            for (int i = 0; i < num_tensor_in_graph; i++) {
//                if (strcmp(tensor_in_graph[i]->name, tensor_name) == 0) {
//                    tensor = tensor_in_graph[i];
//                }
//            }
//            if (tensor == nullptr) {
//                LOG(ERROR) << "tensor " << tensor_name << " not found in graph";
//                continue;
//            }
//            // u32 round to str
//            dump_tensor(str(name) + "(" + std::to_string(round) + ")", tensor);
//            if (!compare_tensors(str(name) + "(" + std::to_string(round) + ")")) {
//                LOG(ERROR) << "tensor " << tensor_name << " failed the test in" << " round " << round;
//            }
//        }
//    };
//
//    check_tensor_value();
//    round++;
//

    ggml_tensor *result_output = graph->nodes[graph->n_nodes - 1];
    token_id _token_id = greedy_sample(result_output);

    input.push_back(_token_id);
}

int gemma_model::load_input_tokens_to_tensor(std::vector<token_id> &input, inference_stage stage) const {
    CHECK_PTR(_input_tensor_holder.inp_tokens)

//    // log kv_cache_buffer size
//    LOG(INFO) << "input kv_cache_buffer size: " << ggml_backend_buffer_name(_input_tensor_holder.input_tensor_buffer) << " = "
//              << (double)ggml_backend_buffer_get_size(_input_tensor_holder.input_tensor_buffer) / 1024.0 / 1024.0 << " MiB";

    ggml_backend_tensor_set(_input_tensor_holder.inp_tokens,
                            input.data() + (stage == inference_stage::PREFILL ? 0 : input.size() - 1), 0,
                            (stage == inference_stage::PREFILL ? input.size() : 1) *
                            ggml_element_size(_input_tensor_holder.inp_tokens));

    std::vector<i32> pos;

    switch (stage) {
        case inference_stage::PREFILL:
            pos.resize(input.size());
            for (int i = 0; i < input.size(); i++) {
                pos[i] = i;
            }
            break;
        case inference_stage::DECODE:
            pos.resize(1);
            pos[0] = (i32) input.size() - 1;
            break;
    }

    CHECK_PTR(_input_tensor_holder.inp_pos)
    ggml_backend_tensor_set(_input_tensor_holder.inp_pos, pos.data(), 0,
                            pos.size() * ggml_element_size(_input_tensor_holder.inp_pos));

    // init kq mask
    auto *kq_mask_data = (float *) _input_tensor_holder.inp_KQ_mask->data;
    CHECK_PTR(kq_mask_data)
    size_t x = _kv_cache.n;
    size_t y = (stage == inference_stage::PREFILL ? input.size() : 1);
    size_t begin = (stage == inference_stage::PREFILL ? 0 : input.size() - 1);

    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            if (j > begin) {
                kq_mask_data[i * x + j] = -INFINITY;
            } else {
                kq_mask_data[i * x + j] = 0;
            }
        }
        begin++;
    }

    return 0;
}

int gemma_model::init_input_tensor() {
    ggml_init_params init_params = {
            ggml_tensor_overhead() * 4,
            nullptr,
            true,
    };

    CHECK_PTR(input_ctx = ggml_init(init_params))
    CHECK_PTR(_input_tensor_holder.inp_tokens = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE))
    CHECK_PTR(_input_tensor_holder.inp_pos = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE))
    CHECK_PTR(_input_tensor_holder.inp_KQ_mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, DEFAULT_CTX_NUM,
                                                                    DEFAULT_BATCH_SIZE))

    _input_tensor_holder.input_tensor_buffer_type = ggml_backend_cpu_buffer_type();
    _input_tensor_holder.input_tensor_buffer = ggml_backend_alloc_ctx_tensors_from_buft(input_ctx,
                                                                                        _input_tensor_holder.input_tensor_buffer_type);
    ggml_backend_buffer_clear(_input_tensor_holder.input_tensor_buffer, 0);

    return 0;
}

int gemma_model::init_kv_cache() {
    u32 n_embed_k_gqa = _hyper_param.n_embed_heads;
    u32 n_embed_v_gqa = _hyper_param.n_embed_heads;
    u32 n_layer = _hyper_param.n_layer;

    _kv_cache.head = 0;
    _kv_cache.size = DEFAULT_CTX_NUM;

    _kv_cache.cells.clear();
    _kv_cache.cells.resize(DEFAULT_CTX_NUM);

    struct ggml_init_params params = {
            2u * n_layer * ggml_tensor_overhead(),
            nullptr,
            true,
    };

    CHECK_PTR(kv_ctx = ggml_init(params))

    _kv_cache.k_layer.reserve(n_layer);
    _kv_cache.v_layer.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        ggml_tensor *k = ggml_new_tensor_1d(kv_ctx, DEFAULT_KV_CACHE_TYPE, n_embed_k_gqa * DEFAULT_CTX_NUM);
        ggml_tensor *v = ggml_new_tensor_1d(kv_ctx, DEFAULT_KV_CACHE_TYPE, n_embed_v_gqa * DEFAULT_CTX_NUM);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        _kv_cache.k_layer.push_back(k);
        _kv_cache.v_layer.push_back(v);
    }

    _kv_cache.kv_cache_buffer_type = ggml_backend_cpu_buffer_type();
    _kv_cache.kv_cache_buffer = ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx, _kv_cache.kv_cache_buffer_type);
    ggml_backend_buffer_clear(_kv_cache.kv_cache_buffer, 0);

    CHECK_PTR(_kv_cache.kv_cache_buffer)
    LOG(INFO) << "KV kv_cache_buffer size: " << ggml_backend_buffer_name(_kv_cache.kv_cache_buffer) << " = "
              << (double) ggml_backend_buffer_get_size(_kv_cache.kv_cache_buffer) / 1024.0 / 1024.0 << " MiB";

    return 0;
}

int gemma_model::init_hyper_param(gguf_context *gguf_ctx) {
    _hyper_param.n_layer = get_u32_from_kv(gguf_ctx, "gemma.block_count");
    _hyper_param.n_header_kv_pair = gguf_get_n_kv(gguf_ctx);
    _hyper_param.n_tensors = gguf_get_n_tensors(gguf_ctx);
    _hyper_param.n_embed = get_u32_from_kv(gguf_ctx, "gemma.embedding_length");
    _hyper_param.n_head = get_u32_from_kv(gguf_ctx, "gemma.attention.head_count");
    _hyper_param.n_embed_heads = get_u32_from_kv(gguf_ctx, "gemma.embedding_length") /
                                 get_u32_from_kv(gguf_ctx, "gemma.attention.head_count");
    _hyper_param.f_norm_rms_eps = get_f32_from_kv(gguf_ctx, "gemma.attention.layer_norm_rms_epsilon");
    _hyper_param.origin_ctx_len = get_u32_from_kv(gguf_ctx, "gemma.context_length");
    _hyper_param.n_head_kv = get_u32_from_kv(gguf_ctx, "gemma.attention.head_count_kv");
    return 0;
}

static int32_t kv_cache_cell_max(const struct kv_cache &cache) {
    for (uint32_t i = cache.size - 1; i > 0; --i) {
        if (cache.cells[i].pos >= 0 && !cache.cells[i].seq_id.empty()) {
            printf("kv_cache_cell_max: %d\n", i);
            fflush(stdout);
            return (i32) i + 1;
        }
    }
    return 0;
}

int gemma_model::update_kv_cache(std::vector<token_id> &input, [[maybe_unused]] inference_stage stage) {
    _kv_cache.n = std::min((int32_t) DEFAULT_CTX_NUM, ((i32)((input.size()) / 32) + 1) * 32);
    if (stage == inference_stage::PREFILL) {
        _kv_cache.head = 0;
    } else {
        _kv_cache.head = input.size() - 1;
    }
    return 0;
}

ggml_tensor *gemma_model::graph_build_norm(ggml_context *pContext, ggml_tensor *pTensor, ggml_tensor *norm) const {
    pTensor = ggml_rms_norm(pContext, pTensor, _hyper_param.f_norm_rms_eps);
    pTensor = ggml_mul(pContext, pTensor, norm);
    return pTensor;
}

ggml_tensor *gemma_model::graph_build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *up, ggml_tensor *gate,
                                          ggml_tensor *down) {
    ggml_tensor *tmp = ggml_mul_mat(ctx, up, cur);
    cur = ggml_mul_mat(ctx, gate, cur);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_mul(ctx, cur, tmp);
    cur = ggml_mul_mat(ctx, down, cur);
    return cur;
}

ggml_tensor *
gemma_model::graph_build_kqv(struct ggml_context *ctx, struct ggml_cgraph *graph, struct ggml_tensor *attn_output,
                             struct ggml_tensor *q_tensor, struct ggml_tensor *kq_mask, i64 n_ctx, i32 n_tokens,
                             i32 n_kv,
                             float kq_scale, int index_layer) {
    const int64_t n_head = _hyper_param.n_head;
    const int64_t n_head_kv = _hyper_param.n_head_kv;
    const int64_t n_embd_head_k = _hyper_param.n_embed_heads;
    const int64_t n_embd_k_gqa = _hyper_param.n_embed_heads;
    const int64_t n_embd_head_v = _hyper_param.n_embed_heads;

    ggml_tensor *q = ggml_permute(ctx, q_tensor, 0, 2, 1, 3);

    ggml_tensor *k =
            ggml_view_3d(ctx, _kv_cache.k_layer[index_layer],
                         n_embd_head_k, n_kv, n_head_kv,
                         ggml_row_size(_kv_cache.k_layer[index_layer]->type, n_embd_k_gqa),
                         ggml_row_size(_kv_cache.k_layer[index_layer]->type, n_embd_head_k),
                         0);

    ggml_tensor *kq = ggml_mul_mat(ctx, k, q);

    kq = ggml_soft_max_ext(ctx, kq, kq_mask, nullptr, kq_scale, DEFAULT_F_MAX_ALIBI_BIAS);

    ggml_tensor *v =
            ggml_view_3d(ctx, _kv_cache.v_layer[index_layer],
                         n_kv, n_embd_head_v, n_head_kv,
                         ggml_element_size(_kv_cache.v_layer[index_layer]) * n_ctx,
                         ggml_element_size(_kv_cache.v_layer[index_layer]) * n_ctx * n_embd_head_v,
                         0);

    ggml_tensor *kqv = ggml_mul_mat(ctx, v, kq);

    ggml_tensor *kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

    ggml_tensor *cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_k * n_head, n_tokens);

    ggml_build_forward_expand(graph, cur);

    cur = ggml_mul_mat(ctx, attn_output, cur);

    return cur;
}


void
gemma_model::graph_build_kv_store(ggml_context *ctx, ggml_cgraph *graph, ggml_tensor *k_tensor, ggml_tensor *v_tensor,
                                  i64 n_ctx, i32 n_tokens, i32 kv_head, i64 index_layer) {
    const i64 n_embd_v_gqa = _hyper_param.n_embed_heads;
    const i64 n_embd_k_gqa = _hyper_param.n_embed_heads;

    ggml_tensor *v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_tensor, n_embd_v_gqa, n_tokens));
    ggml_tensor *k_cache_view = ggml_view_1d(ctx, _kv_cache.k_layer[index_layer], n_tokens * n_embd_k_gqa,
                                             (ggml_row_size(_kv_cache.k_layer[index_layer]->type, n_embd_k_gqa)) *
                                             kv_head);

    ggml_tensor *v_cache_view = ggml_view_2d(ctx, _kv_cache.v_layer[index_layer], n_tokens, n_embd_v_gqa,
                                             (n_ctx) * ggml_element_size(_kv_cache.v_layer[index_layer]),
                                             (kv_head) * ggml_element_size(_kv_cache.v_layer[index_layer]));

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_tensor, k_cache_view));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur_t, v_cache_view));
}

ggml_tensor *
gemma_model::graph_build_kv(ggml_context *ctx, ggml_cgraph *graph, ggml_tensor *attn_output, ggml_tensor *k_tensor,
                            ggml_tensor *v_tensor, ggml_tensor *q_tensor, ggml_tensor *kq_mask, i64 n_ctx, i32 n_tokens,
                            i32 kv_head, i32 n_kv, float kq_scale, int index_layer) {
    ggml_build_forward_expand(graph, q_tensor);
    ggml_build_forward_expand(graph, k_tensor);
    ggml_build_forward_expand(graph, v_tensor);

    graph_build_kv_store(ctx, graph, k_tensor, v_tensor, n_ctx, n_tokens, kv_head, index_layer);
    return graph_build_kqv(ctx, graph, attn_output, q_tensor, kq_mask, n_ctx, n_tokens, n_kv, kq_scale, index_layer);
}

token_id gemma_model::greedy_sample(const ggml_tensor *model_output) {
    const auto *logits = (const float *) model_output->data + model_output->ne[0] * (model_output->ne[1] - 1);
    const size_t n = model_output->ne[0];
    float max_val = -INFINITY;
    int max_idx = -1;

    for (int i = 0; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

//    auto print_top_k_logits = [&](size_t k) {
//        std::vector<std::pair<float, int>> logits_;
//        for (int i = 0; i < n; i++) {
//            logits_.emplace_back(logits[i], i);
//        }
////        std::sort(logits.begin(), logits.end(), std::greater<>());
//        for (int i = 0; i < k; i++) {
//            LOG(INFO) << "logit " << i << ": " << logits_[i].first << " token id: " << logits_[i].second;
//        }
//    };
//
//    print_top_k_logits(10);

    return max_idx;
}

void gemma_model::begin_one_round_inference() {
    std::vector<token_id> input = {_tokenizer.special_bos_id, 25612};
    _tokenizer.print_tokens(input);
    u32 size = input.size();
    auto prefill_start = std::chrono::high_resolution_clock::now();
    inference(input, inference_stage::PREFILL);
    auto prefill_end = std::chrono::high_resolution_clock::now();

    if (input.size() <= size) {
        LOG(ERROR) << "inference failed";
    }
    auto decode_start = std::chrono::high_resolution_clock::now();
    while (input[input.size() - 1] != _tokenizer.special_eos_id && input.size() < DEFAULT_TOKEN_NUM){
        _tokenizer.print_token(input, input.size() - 1);
        inference(input, inference_stage::DECODE);
    }
    auto decode_end = std::chrono::high_resolution_clock::now();
    _tokenizer.print_token(input, input.size() - 1);

    auto prefill_time = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
    auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();

    LOG(INFO) << "prefill time: " << prefill_time << "ms.";
    LOG(INFO) << (double) size / prefill_time * 1000 << " tokens/s";
    LOG(INFO) << "decode time: " << decode_time << "ms.";
    LOG(INFO) << (double) (input.size() - size) / decode_time * 1000 << " tokens/s";

    LOG(INFO) << "inference finished";
}

int gemma_model::refresh_kv_cache() {
    _kv_cache.n =
    _kv_cache.head = 0;
    return 0;
}

int gemma_model::load_tensors_from_ctx(gguf_context *gguf_ctx, int n_tensors) {
    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
        MASK(
                LOG(INFO) << "Loading tensor: " << name;
        )
        ggml_tensor *t = ggml_get_tensor(weight_ctx, name);
        tensors[name] = t;
    }
    LOG(INFO) << tensors.size() << " tensors loaded";
    return 0;
}

int gemma_model::load_header_kv_pair_from_ctx(gguf_context *gguf_ctx, int n_kv_pair) {
    std::map<gguf_type, int> type_count;

    for (int i = 0; i < n_kv_pair; i++) {
        const char *name = gguf_get_key(gguf_ctx, i);
        const enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);
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
    return 0;
}

int gemma_model::reset_compute_context() {
    static u32 mem_size =
            ggml_tensor_overhead() * CGRAPH_MAX_NODE_NUM + ggml_graph_overhead() + COMPUTE_MID_NODE_DATA_BUFFER_SIZE;
    static void *compute_context_buffer = malloc(mem_size);

    ggml_free(compute_ctx);
    ggml_init_params params = {
            mem_size,
            compute_context_buffer,
            false,
    };
    compute_ctx = ggml_init(params);
    return 0;
}

ggml_cgraph *gemma_model::build_compute_graph(std::vector<token_id> &input, inference_stage stage) {
    reset_compute_context(); // only for tensor node reference, not for tensor data
    ggml_cgraph *graph = ggml_new_graph(compute_ctx);
    CHECK_PTR(graph)

    i64 input_token_tensor_size = (stage == inference_stage::PREFILL ? (i64) input.size() : 1);

    ggml_tensor *cur;
    ASSERT_MSG(!input.empty(), "input size must be greater than 0")

    ggml_tensor *inp_tokens_v = ggml_view_1d(compute_ctx, _input_tensor_holder.inp_tokens, input_token_tensor_size, 0);
    ggml_set_name(inp_tokens_v, "inp_tokens (view)");
    ggml_tensor *inpL = ggml_get_rows(compute_ctx, _tensor_holder.token_embd, inp_tokens_v);

    inpL = ggml_scale(compute_ctx, inpL, sqrtf((float) _hyper_param.n_embed));
    ggml_tensor *inp_pos = ggml_view_1d(compute_ctx, _input_tensor_holder.inp_pos, input_token_tensor_size, 0);
    ggml_set_name(inp_pos, "inp_pos (view)");

    ggml_tensor *KQ_mask = ggml_view_2d(compute_ctx, _input_tensor_holder.inp_KQ_mask, _kv_cache.n,
                                        input_token_tensor_size,
                                        _kv_cache.n * ggml_type_size(_input_tensor_holder.inp_KQ_mask->type), 0);
    ggml_set_name(KQ_mask, "inp_KQ_mask (view)");

    for (int il = 0; il < _hyper_param.n_layer; ++il) {

        cur = graph_build_norm(compute_ctx, inpL, _tensor_holder.layers[il].attn_norm);
        {
            ggml_tensor *Qcur = ggml_mul_mat(compute_ctx, _tensor_holder.layers[il].attn_q, cur);

            ggml_tensor *Kcur = ggml_mul_mat(compute_ctx, _tensor_holder.layers[il].attn_k, cur);

            ggml_tensor *Vcur = ggml_mul_mat(compute_ctx, _tensor_holder.layers[il].attn_v, cur);

            Qcur = ggml_rope_custom(
                    compute_ctx,
                    ggml_reshape_3d(compute_ctx, Qcur, _hyper_param.n_embed_heads, _hyper_param.n_head,
                                    input_token_tensor_size),
                    inp_pos,
                    (i32) _hyper_param.n_embed_heads, DEFAULT_ROPE_TYPE, 0, (i32) _hyper_param.origin_ctx_len,
                    DEFAULT_FREQ_BASE,
                    DEFAULT_FREQ_SCALE,
                    DEFAULT_EXT_FACTOR, DEFAULT_ATTN_FACTOR, DEFAULT_BETA_FAST, DEFAULT_BETA_SLOW);

            Qcur = ggml_scale(compute_ctx, Qcur, 1.0f / sqrtf(float(_hyper_param.n_embed_heads)));


            Kcur = ggml_rope_custom(
                    compute_ctx, ggml_reshape_3d(compute_ctx, Kcur, _hyper_param.n_embed_heads, _hyper_param.n_head_kv,
                                                 input_token_tensor_size), inp_pos,
                    (i32) _hyper_param.n_embed_heads, DEFAULT_ROPE_TYPE, 0, (i32) _hyper_param.origin_ctx_len,
                    DEFAULT_FREQ_BASE,
                    DEFAULT_FREQ_SCALE,
                    DEFAULT_EXT_FACTOR, DEFAULT_ATTN_FACTOR, DEFAULT_BETA_FAST, DEFAULT_BETA_SLOW);

            cur = graph_build_kv(compute_ctx, graph, _tensor_holder.layers[il].attn_output, Kcur, Vcur, Qcur, KQ_mask,
                                 DEFAULT_CTX_NUM,
                                 (i32) input_token_tensor_size, (i32) _kv_cache.head, (i32) _kv_cache.n, 1.0f, il);
        }

        ggml_tensor *sa_out = ggml_add(compute_ctx, cur, inpL);
        cur = graph_build_norm(compute_ctx, sa_out, _tensor_holder.layers[il].ffn_norm);

        {
            cur = graph_build_ffn(compute_ctx, cur, _tensor_holder.layers[il].ffn_up,
                                  _tensor_holder.layers[il].ffn_gate,
                                  _tensor_holder.layers[il].ffn_down);
        }
        cur = ggml_add(compute_ctx, cur, sa_out);

        inpL = cur;
    }

    cur = inpL;

    cur = graph_build_norm(compute_ctx, cur, _tensor_holder.output_norm);

    cur = ggml_mul_mat(compute_ctx, _tensor_holder.output, cur);

    ggml_build_forward_expand(graph, cur);

    ggml_tensor *result_output = graph->nodes[graph->n_nodes - 1];
    ggml_set_name(result_output, "result_output");
    return graph;
}

std::string gemma_tokenizer::find_token(token_id id) {
    return tokens[id];
}

void gemma_tokenizer::print_tokens(std::vector<token_id> &input) {
    for (int i = 0; i < input.size(); i++) {
        printf("%s", tokens[input[i]].c_str());
    }
    fflush(stdout);
}

void gemma_tokenizer::print_token(token_id id) {
    printf("%s", tokens[id].c_str());
    fflush(stdout);
}

void gemma_tokenizer::print_token(std::vector<token_id> &input, u32 index) {
    printf("%s", tokens[input[index]].c_str());
    fflush(stdout);
}

#pragma clang diagnostic pop