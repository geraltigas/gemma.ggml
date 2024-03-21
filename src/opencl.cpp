//
// Created by geraltigas on 3/19/24.
//

#include <set>
#include <string>
#include <map>
// include glog
#include <glog/logging.h>
#include "opencl.h"
#include "macro.h"

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        LOG(ERROR) << msg << std::endl; \
        exit(EXIT_FAILURE); \
    }

bool fp16_support;
bool unified_memory_support;
cl_platform_id _global_platform;
cl_device_id _global_device;
cl_context _global_context;
cl_command_queue _global_queue;
cl_program _global_program;
// buffers set
static std::set<cl_mem> _global_buffers;
// kernel map, key is kernel name, value is kernel
static std::map<std::string, cl_kernel> _global_kernels;
const char* platform_name = OPENCL_PLATFORM_NAME;
//"NVIDIA CUDA";
const char* device_name = OPENCL_DEVICE_NAME;
const char* kernels_file = OPENCL_KERNELS_FILE;
size_t max_work_group_size;
void print_platforms() {
    LOG(INFO) << "OpenCL Platforms:" << std::endl;
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id* platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_ERROR(err, "Failed to get platform IDs")
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
    CHECK_ERROR(err, "Failed to get platform IDs")
    for (unsigned int i = 0; i < num_platforms; i++) {
        char _platform_name[128];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, _platform_name, nullptr);
        CHECK_ERROR(err, "Failed to get platform info")
        LOG(INFO) << "Platform " << i << ": " << _platform_name << std::endl;
    }
    free(platforms);
}
void print_gpus()  {
    LOG(INFO) << "OpenCL GPUs:" << std::endl;
    cl_int err;
    cl_uint num_platforms, num_devices;
    cl_platform_id* platforms;
    cl_device_id* devices;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_ERROR(err, "Failed to get platform IDs")
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
    CHECK_ERROR(err, "Failed to get platform IDs")
    for (unsigned int i = 0; i < num_platforms; i++) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS) {
            continue;
        }
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
        CHECK_ERROR(err, "Failed to get device IDs")
        for (unsigned int j = 0; j < num_devices; j++) {
            char deviceName[128];
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
            CHECK_ERROR(err, "Failed to get device info")
            LOG(INFO) << "Platform " << i << ", GPU " << j << ": " << deviceName << std::endl;
        }
        free(devices);
    }
    free(platforms);
}

void print_opencl_info() {
    print_platforms();
    print_gpus();
}

void print_now_using(cl_platform_id platform, cl_device_id device) {
    LOG(INFO) << "Now Using:" << std::endl;
    char _platform_name[128];
    char _device_name[128];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, _platform_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, _device_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    LOG(INFO) << "Platform: " << _platform_name << std::endl;
    LOG(INFO) << "Device: " << _device_name << std::endl;
    LOG(INFO) << "Max work group size: " << max_work_group_size << std::endl;
}

void print_now_using() {
    print_now_using(_global_platform, _global_device);
}
// return multiple values: platform and device, function name is get_platform_and_device(const char *platform_name, const char *device_name)

std::tuple<cl_platform_id, cl_device_id> get_platform_and_device(const char* platform_name, const char* device_name) {
    LOG(INFO) << "Get Platform and Device:" << std::endl;
    cl_int err;
    cl_uint num_platforms, num_devices;
    cl_platform_id* platforms;
    cl_device_id* devices;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_ERROR(err, "Failed to get platform IDs")
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
    CHECK_ERROR(err, "Failed to get platform IDs")
    for (unsigned int i = 0; i < num_platforms; i++) {
        char _platform_name[128];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, _platform_name, nullptr);
        CHECK_ERROR(err, "Failed to get platform info")
        if (strcmp(_platform_name, platform_name) == 0) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
            CHECK_ERROR(err, "Failed to get device IDs")
            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
            CHECK_ERROR(err, "Failed to get device IDs")
            for (unsigned int j = 0; j < num_devices; j++) {
                char _device_name[128];
                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, _device_name, nullptr);
                CHECK_ERROR(err, "Failed to get device info")
                if (strcmp(_device_name, device_name) == 0) {
                    LOG(INFO) << "Platform: " << _platform_name << std::endl;
                    LOG(INFO) << "Device: " << _device_name << std::endl;
                    return std::make_tuple(platforms[i], devices[j]);
                }
            }
            free(devices);
        }
    }
    free(platforms);
    return std::make_tuple(nullptr, nullptr);
}
void set_platform_and_device() {
    // use get_platform_and_device to get platform and device, and set them to _global_platform and _global_device
    auto tuple = get_platform_and_device(platform_name, device_name);
    _global_platform = std::get<0>(tuple);
    _global_device = std::get<1>(tuple);
    if (_global_platform == nullptr || _global_device == nullptr) {
        LOG(ERROR) << "Platform or device not found" << std::endl;
        exit(EXIT_FAILURE);
    }

    // check unified memory support
    check_unified_memory_support();
    LOG(INFO) << "Device unified memory support: " << (unified_memory_support ? "true" : "false") << std::endl;
}
void setup_opencl_env() {
    // check _global_platform and _global_device
    if (_global_platform == nullptr || _global_device == nullptr) {
        LOG(ERROR) << "Platform or device is not initialized" << std::endl;
        exit(EXIT_FAILURE);
    }
    cl_context_properties properties[] = {
        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)_global_platform, 0
    };
    // init context and queue
    cl_int err;
    _global_context = clCreateContext(properties, 1, &_global_device, NULL, NULL, &err);
        // clCreateContext(nullptr, 1, &_global_device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create context")
    _global_queue = clCreateCommandQueueWithProperties(_global_context, _global_device, nullptr, &err);
    CHECK_ERROR(err, "Failed to create command queue")
}

void build_all_kernels() {
    cl_int err;
    // read kernels.cl
    FILE* fp = fopen(kernels_file, "rb");
    if (fp == nullptr) {
        LOG(ERROR) << "Failed to open file " << kernels_file << std::endl;
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = (char *)malloc(size + 1);
    source[size] = '\0';
    // read file
    fread(source, sizeof(char), size, fp);
    fclose(fp);
    // create program
    _global_program = clCreateProgramWithSource(_global_context, 1, (const char **)&source, nullptr, &err);
    CHECK_ERROR(err, "Failed to create program")
    err = clBuildProgram(_global_program, 1, &_global_device, nullptr, nullptr, nullptr);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(_global_program, _global_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        char* log = (char *)malloc(log_size);
        clGetProgramBuildInfo(_global_program, _global_device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

        LOG(ERROR) << "Build log:" << std::endl << log << std::endl;

        free(log);
    }
    CHECK_ERROR(err, "Failed to build program")
    // get kernel count
    cl_uint num_kernels;
    err = clCreateKernelsInProgram(_global_program, 0, nullptr, &num_kernels);
    CHECK_ERROR(err, "Failed to get kernel count")

    // get kernel names
    std::vector<cl_kernel> kernels(num_kernels);
    err = clCreateKernelsInProgram(_global_program, num_kernels, kernels.data(), nullptr);
    CHECK_ERROR(err, "Failed to get kernel names")

    for (unsigned int i = 0; i < num_kernels; i++) {
        char kernel_name[128];
        err = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 128, kernel_name, nullptr);
        CHECK_ERROR(err, "Failed to get kernel name")
        _global_kernels[kernel_name] = kernels[i];
    }
}

cl_kernel get_kernel(const char* kernel_name) {
    // check _global_kernels
    if (_global_kernels.empty()) {
        LOG(ERROR) << "Kernels are not initialized" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto kernel = _global_kernels.find(kernel_name);
    if (kernel == _global_kernels.end()) {
        LOG(ERROR) << "Kernel " << kernel_name << " not found" << std::endl;
        exit(EXIT_FAILURE);
    }
    return kernel->second;
}

void add_kernel(const char* kernel_name, cl_kernel kernel) {
    if (_global_kernels.find(kernel_name) != _global_kernels.end()) {
        LOG(ERROR) << "Kernel " << kernel_name << " already exists" << std::endl;
        exit(EXIT_FAILURE);
    }
    _global_kernels[kernel_name] = kernel;
}

void release_opencl_env() {
    // release buffers
    for (auto buffer: _global_buffers) {
        clReleaseMemObject(buffer);
    }
    // release kernels
    for (const auto& kernel: _global_kernels) {
        clReleaseKernel(kernel.second);
    }
    // release program
    clReleaseProgram(_global_program);
    // release command queue
    clReleaseCommandQueue(_global_queue);
    // release context
    clReleaseContext(_global_context);
}

void check_unified_memory_support() {
    cl_bool unified_memory;
    cl_int err;

    err = clGetDeviceInfo(_global_device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL);
    if (err != CL_SUCCESS) {
        // 错误处理
        printf("Error getting device info: %d\n", err);
        unified_memory_support = false;
    }else {
        unified_memory_support = (unified_memory == CL_TRUE);
    }
}

int init_opencl() {
    set_platform_and_device();
    print_now_using();
    setup_opencl_env();
    build_all_kernels();
    return 0;
}
