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
#define VOCAB_DUMP_DIR "/home/geraltigas/Desktop/gemma.ggml/vocab_dump"
#define TARGET 1
#define SOURCE 0
#define MODE SOURCE

enum class tensor_dump_mode {
    target = 1,
    source = 0
};

void dump_tensor(const char *name, const ggml_tensor *tensor);
void dump_tensor(std::string name, const ggml_tensor *tensor);
bool compare_tensors(const char *name);
bool compare_tensors(std::string name);
std::map<std::string, std::string> get_tensor_dump_list();
void *load_tensor(const char *name, tensor_dump_mode mode);
void dump_ptr_data(const char *name, const void *ptr, size_t size);

#endif //GEMMA_GGML_TENSOR_DUMP_H
