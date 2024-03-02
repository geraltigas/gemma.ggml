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

