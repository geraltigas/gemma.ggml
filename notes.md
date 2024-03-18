### from tokens array (vector<token_id>) to input embedding

llama.cpp:7735
```cpp
ggml_backend_tensor_set(lctx.inp_tokens, batch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
```
from tokens array (vector<token_id>) to tokens array tensor

llama.cpp:7458
```cpp
inpL = llm_build_inp_embd(ctx0, hparams, batch, model.tok_embd, lctx.inp_tokens, lctx.inp_embd, cb);
```
from tokens array tensor to input embedding

### load tensor data from file

llama.cpp:4565

### kv_cache init

llama.cpp:1972

### kv_cache n and mem update

llama.cpp:7960

### tensor, context and buffer

tensor: **reference** and **data**
context: store the reference
buffer: store the data

allocate tensor references in context first, then allocate buffer(actual data) for tensor references

### all tensors related and their classification

- **model weight tensors**: load from file, no buffer. once allocated, no change. context: **weight_ctx**
- **input tensors**: create manually, init with 0, with buffer. once allocated, no reallocate. context: **input_ctx**
- **kv cache tensors**: create manually, with buffer. once allocated, no reallocate. context: **kv_ctx**
- **mid-inference tensors**: create automatically, with buffer. reallocate during each inference. context: none

### how to allocate tensor

1. create a ctx with enough tensor overhead space: tensor_num * ggml_tensor_overhead()
2. create a buffer and allocate tensor data in the buffer
3. (optional) init buffer data with 0

```cpp
    ggml_init_params init_params = {
            ggml_tensor_overhead() * 4,
            nullptr,
            true,
    };

    CHECK_PTR(input_ctx = ggml_init(init_params))
    CHECK_PTR(_input_tensor_holder.inp_tokens = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE))
    CHECK_PTR(_input_tensor_holder.inp_pos = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, DEFAULT_BATCH_SIZE))
    CHECK_PTR(_input_tensor_holder.inp_KQ_mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, DEFAULT_CTX_NUM, DEFAULT_BATCH_SIZE))

    _input_tensor_holder.input_tensor_buffer_type = ggml_backend_cpu_buffer_type();
    _input_tensor_holder.input_tensor_buffer = ggml_backend_alloc_ctx_tensors_from_buft(input_ctx, _input_tensor_holder.input_tensor_buffer_type);
    ggml_backend_buffer_clear(_input_tensor_holder.input_tensor_buffer, 0);
```


### profile data

----------------- op -----------------
op_add: 20672 μs (0.0197446%), called 3456 times, overlapping with: none
op_cont: 1903 μs (0.00181763%), called 1728 times, overlapping with: none
op_cpy: 6991 μs (0.00667737%), called 3456 times, overlapping with: none
op_get_rows: 709 μs (0.000677193%), called 96 times, overlapping with: none
op_mul: 84797 μs (0.0809929%), called 5280 times, overlapping with: none
op_mul_mat: 104164107 μs (99.4912%), called 31296 times, overlapping with: none
op_permute: 127 μs (0.000121303%), called 3456 times, overlapping with: none
op_reshape: 931 μs (0.000889234%), called 5184 times, overlapping with: none
op_rms_norm: 23627 μs (0.0225671%), called 3552 times, overlapping with: none
op_rope: 43399 μs (0.0414521%), called 3456 times, overlapping with: none
op_scale: 4085 μs (0.00390174%), called 1824 times, overlapping with: none
op_soft_max: 25793 μs (0.0246359%), called 1728 times, overlapping with: none
op_transpose: 25 μs (2.38785e-05%), called 1728 times, overlapping with: none
op_unary: 319275 μs (0.304952%), called 1728 times, overlapping with: none
op_view: 400 μs (0.000382055%), called 7200 times, overlapping with: none

----------------- mul_mat -----------------
mul_mat_compute_before_loop: 17805 μs (0.0171826%), called 15648 times, overlapping with: mmstage_compute op_mul_mat 
mul_mat_compute_loop: 103604331 μs (99.9828%), called 15648 times, overlapping with: mmstage_compute op_mul_mat 

----------------- mmstage -----------------
mmstage_compute: 103682683 μs (99.5858%), called 15648 times, overlapping with: op_mul_mat 
mmstage_init: 431268 μs (0.414227%), called 15648 times, overlapping with: op_mul_mat 
