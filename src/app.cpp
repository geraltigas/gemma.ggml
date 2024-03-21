//
// Created by geraltigas on 2/25/24.
//

#include <app.h>
#include <gemma_model.h>
#include <iostream>
#include <memory>
#include <macro.h>
#include "profiling.h"
#include "opencl.h"
#include "thread_pool.h"

#define LOG_DIR "/home/geraltigas/Desktop/gemma.ggml/log"

int app::init_glog(int argc, char* argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_log_dir = LOG_DIR;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::InstallFailureWriter([](const char* data, int size) {
        std::cerr.write(data, size);
    });
    // set log level
    return 0;
}

int app::run(int argc, char* argv[])
{
    CHECK_RT_MSG(init_glog(argc, argv), "Failed to init glog")
    init_profiling();
    CHECK_RT_MSG(init_opencl(), "Failed to init opencl")
    CHECK_RT_MSG(init_thread_pool(N_THREADS_MUL_MAT_CPU), "Failed to init ggml")
    const char * gguf_file_path = "../models/gemma-2b-it-q4_k_m.gguf";
    auto model = std::make_unique<gemma_model>();
    model->load_model_from_file(gguf_file_path); // init weight etc
    model->init_input_tensor(); //  init input tensor
    model->init_kv_cache(); // init kv cache tensor
    model->begin_one_round_inference();
    print_profiling_result();
    LOG(INFO) << "app::run";
    return 0;
}