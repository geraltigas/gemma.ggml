//
// Created by geraltigas on 2/25/24.
//

#include "app.h"

#include <gemma_model.h>
#include <iostream>
#include <memory>

#include "macro.h"
#include "ggml.h"

int init_glog(int /*argc*/, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter([](const char* data, int size) {
        std::cerr.write(data, size);
    });
    // set log level
    FLAGS_stderrthreshold = google::INFO;
    return 0;
}

int app::run(int argc, char* argv[])
{
    // init glog
    CHECK_RT_MSG(init_glog(argc, argv), "Failed to init glog");
    const char * gguf_file_path = "../models/gemma-2b-it-q4_k_m.gguf";
    // struct ggml_model * model = ggml_load_gguf(gguf_file_path);
    // if (model == nullptr) {
    //     LOG(ERROR) << "Failed to load model from " << gguf_file_path;
    //     return -1;
    // }
    auto model = std::make_unique<GemmaModel>();
    model->load_model_from_file(gguf_file_path);
    model->model_warmup();
    LOG(INFO) << "app::run";
    return 0;
}