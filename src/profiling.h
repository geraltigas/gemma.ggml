//
// Created by geraltigas on 3/17/24.
//

#ifndef GEMMA_GGML_PROFILING_H
#define GEMMA_GGML_PROFILING_H

#define ENABLE_PROFILING 1

#ifdef __cplusplus
extern "C" {
#endif

#if ENABLE_PROFILING == 1
#define profile(str, code) start_recording(str); code; stop_recording(str);
#else
#define profile(str, code) code;
#endif

#if ENABLE_PROFILING == 1
#define init_profiling() _init_profiling()
#else
#define init_profiling()
#endif
int _init_profiling(void);

#if ENABLE_PROFILING == 1
#define start_recording(name) _start_recording(name)
#else
#define start_recording(name)
#endif
void _start_recording(const char *name);

#if ENABLE_PROFILING == 1
#define stop_recording(name) _stop_recording(name)
#else
#define stop_recording(name)
#endif
void _stop_recording(const char *name);

#if ENABLE_PROFILING == 1
#define add_count(name) _add_count(name)
#else
#define add_count(name)
#endif
void _add_count(const char *name);

#if ENABLE_PROFILING == 1
#define print_profiling_result() _print_profiling_result()
#else
#define print_profiling_result()
#endif
void _print_profiling_result(void);

#ifdef __cplusplus
}
#endif

#endif //GEMMA_GGML_PROFILING_H
