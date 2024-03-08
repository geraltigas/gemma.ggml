//
// Created by geraltigas on 3/5/24.
//

#ifndef GEMMA_GGML_TENSOR_DUMP_H
#define GEMMA_GGML_TENSOR_DUMP_H

#include <ggml.h>

#define DEFAULT_TENSOR_DUMP_DIR "/home/geraltigas/Desktop/gemma.ggml/tensor_dump"
#define TARGET 1
#define SOURCE 0
#define MODE SOURCE

void dump_tensor(const char *name, const ggml_tensor *tensor);
bool compare_tensors(const char *name);

#endif //GEMMA_GGML_TENSOR_DUMP_H
