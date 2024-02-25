//
// Created by geraltigas on 2/25/24.
//

#ifndef GEMMA_MODEL_H
#define GEMMA_MODEL_H

#include <ggml.h>

class GemmaModel {
    gguf_context * gguf_ctx;

public:
    int load_model_from_file(const char * file_path);
};

#endif //GEMMA_MODEL_H
