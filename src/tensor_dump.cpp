//
// Created by geraltigas on 3/5/24.
//

#include <tensor_dump.h>
#include <glog/logging.h>
// gtest
#include <gtest/gtest.h>
#include <fstream>

void dump_tensor(const char *name, const ggml_tensor *tensor) {
    LOG(INFO) << "dumping tensor: " << name;
    size_t byte_size = tensor->nb[3];
    // create a file with name+suffix
#if MODE == TARGET
    // write tensor data to file with name_target
    // create file using std lib
    std::string file_name = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "target";
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
    std::string file_name_target = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "target";
    // read file with name_source
    std::string file_name_source = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_source";

    FILE *file_target = fopen(file_name_target.c_str(), "rb");
    FILE *file_source = fopen(file_name_source.c_str(), "rb");

    // compare the two files
    // if the two files are the same, return true
    // else return false
    // begin compare
    if (file_target == nullptr || file_source == nullptr) {
        LOG(ERROR) << "failed to open file " << file_name_target << " or " << file_name_source;
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
        LOG(ERROR) << "file size mismatch: " << file_size_target << " vs " << file_size_source;
        return false;
    }

    LOG(INFO) << "file size: " << file_size_target;

    char *buffer_target = new char[file_size_target];
    char *buffer_source = new char[file_size_source];

    fread(buffer_target, file_size_target, 1, file_target);
    fread(buffer_source, file_size_source, 1, file_source);

    for (size_t i = 0; i < file_size_target; i++) {
        if (buffer_target[i] != buffer_source[i]) {
            delete[] buffer_target;
            delete[] buffer_source;
            LOG(ERROR) << "data mismatch at index " << i;
            LOG(ERROR) << "target: " << (int)buffer_target[i];
            LOG(ERROR) << "source: " << (int)buffer_source[i];
            return false;
        }
    }

    // pass the test
    LOG(INFO) << "tensor: " << name << " passed the test";

    return true;
}

std::map<std::string, std::string> get_tensor_dump_list() {
    std::map<std::string, std::string> tensor_dump_list;
    std::ifstream file(TENSOR_DUMP_LIST);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // inp_pos:inp_pos (view)
            // split by :
            // the first part is the name
            // the second part is the view
            // and put it in map
            // begin with // is a comment, skip
            if (line[0] == '/' && line[1] == '/') {
                continue;
            }
            size_t pos = line.find(':');
            std::string name = line.substr(0, pos);
            std::string view = line.substr(pos + 1);
            tensor_dump_list[name] = view;
        }
    }
    return tensor_dump_list;
}

void *load_tensor(const char *name, tensor_dump_mode mode) {
    std::string file_name;
    if (mode == tensor_dump_mode::target) {
        file_name = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "target";
    } else {
        file_name = std::string(DEFAULT_TENSOR_DUMP_DIR) + "/" + name + "_source";
    }
    FILE *file = fopen(file_name.c_str(), "rb");
    if (file == nullptr) {
        LOG(ERROR) << "failed to open file " << file_name;
        return nullptr;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *buffer = malloc(file_size);
    fread(buffer, file_size, 1, file);
    fclose(file);
    return buffer;
}

void dump_ptr_data(const char *name, const void *ptr, size_t size) {
    std::string file_name = std::string(VOCAB_DUMP_DIR) + "/" + name;
    FILE *file = fopen(file_name.c_str(), "wb");
    fwrite(ptr, size, 1, file);
    fclose(file);
}

TEST(tensor_dump, get_list) {
    std::map<std::string, std::string> tensor_dump_list = get_tensor_dump_list();
    for (auto &item : tensor_dump_list) {
        LOG(INFO) << item.first << " " << item.second;
    }
}

TEST(tensor_dump, load_tensor) {
    float *tensor = (float *)load_tensor("inp_KQ_mask", tensor_dump_mode::target);
    for (size_t i = 0; i < 10; i++) {
        LOG(INFO) << tensor[i];
    }
}