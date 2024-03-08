//
// Created by geraltigas on 3/5/24.
//

#include <tensor_dump.h>
#include <glog/logging.h>

void dump_tensor(const char *name, const ggml_tensor *tensor) {
    LOG(INFO) << "Dumping tensor: " << name;
    size_t byte_size = tensor->nb[3];
    // create a file with name+suffix
#if MODE == TARGET
    // write tensor data to file with name_target
    // create file using std lib
    std::string file_name = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_target";
#elif MODE == SOURCE
    // write tensor data to file with name_source
    std::string file_name = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_source";
#endif
    // if exists, overwrite
    FILE *file = fopen(file_name.c_str(), "wb");
    fwrite(tensor->data, byte_size, 1, file);
    fclose(file);
}

bool compare_tensors(const char *name) {
    // read file with name_target
    std::string file_name_target = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_target";
    // read file with name_source
    std::string file_name_source = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_source";

    FILE *file_target = fopen(file_name_target.c_str(), "rb");
    FILE *file_source = fopen(file_name_source.c_str(), "rb");

    // compare the two files
    // if the two files are the same, return true
    // else return false
    // begin compare
    if (file_target == nullptr || file_source == nullptr) {
        LOG(ERROR) << "Failed to open file " << file_name_target << " or " << file_name_source;
        return false;
    }

    size_t file_size_target = 0;
    size_t file_size_source = 0;

    fseek(file_target, 0, SEEK_END);
    file_size_target = ftell(file_target);
    fseek(file_target, 0, SEEK_SET);
    fseek(file_source, 0, SEEK_END);
    file_size_source = ftell(file_source);
    fseek(file_source, 0, SEEK_SET);

    if (file_size_target != file_size_source) {
        LOG(ERROR) << "File size mismatch: " << file_size_target << " vs " << file_size_source;
        return false;
    }

    char *buffer_target = new char[file_size_target];
    char *buffer_source = new char[file_size_source];

    fread(buffer_target, file_size_target, 1, file_target);
    fread(buffer_source, file_size_source, 1, file_source);

    for (size_t i = 0; i < file_size_target; i++) {
        if (buffer_target[i] != buffer_source[i]) {
            delete[] buffer_target;
            delete[] buffer_source;
            return false;
        }
    }

    // pass the test
    LOG(INFO) << "Tensor: " << name << " passed the test";

    return true;
}

