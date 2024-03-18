//
// Created by geraltigas on 3/17/24.
//

#ifndef GEMMA_GGML_PROFILING_H
#define GEMMA_GGML_PROFILING_H

#define profile(str, code) { \
    start_recording(str); \
    code; \
    stop_recording(str); \
}

#ifdef __cplusplus
extern "C" {
#endif

int init_profiling(void);

void start_recording(const char *name);

void stop_recording(const char *name);

void print_profiling_result(void);

#ifdef __cplusplus
}
#endif

#endif //GEMMA_GGML_PROFILING_H
