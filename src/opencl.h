//
// Created by geraltigas on 3/19/24.
//

#ifndef GEMMA_GGML_OPENCL_H
#define GEMMA_GGML_OPENCL_H

#include <tuple>
#include <CL/cl.h>

extern size_t max_work_group_size;

int init_opencl();
void print_opencl_info();
void print_now_using(cl_platform_id platform, cl_device_id device);
void print_now_using();
// return multiple values: platform and device, function name is get_platform_and_device(const char *platform_name, const char *device_name)
std::tuple<cl_platform_id, cl_device_id> get_platform_and_device(const char *platform_name, const char *device_name);
void set_platform_and_device();
void setup_opencl_env();
void build_all_kernels();
cl_kernel get_kernel(const char* kernel_name);
void add_kernel(const char* kernel_name, cl_kernel kernel);
void release_opencl_env();
void check_unified_memory_support();

#endif //GEMMA_GGML_OPENCL_H
