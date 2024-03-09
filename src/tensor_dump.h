//
// Created by geraltigas on 3/5/24.
//

#ifndef GEMMA_GGML_TENSOR_DUMP_H
#define GEMMA_GGML_TENSOR_DUMP_H

#include <ggml.h>
#include <map>
#include <string>

#define DEFAULT_TENSOR_DUMP_DIR "/home/geraltigas/Desktop/gemma.ggml/tensor_dump"
#define TENSOR_DUMP_LIST "/home/geraltigas/Desktop/gemma.ggml/tensor_dump/dump_tensor_list"
#define TARGET 1
#define SOURCE 0
#define MODE SOURCE

void dump_tensor(const char *name, const ggml_tensor *tensor);
bool compare_tensors(const char *name);
std::map<std::string, std::string> get_tensor_dump_list();

#endif //GEMMA_GGML_TENSOR_DUMP_H
