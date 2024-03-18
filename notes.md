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
