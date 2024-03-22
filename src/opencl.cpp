////
//// Created by geraltigas on 3/19/24.
////
//
//#include <set>
//#include <string>
//#include <map>
//// include glog
//#include <glog/logging.h>
//#include "opencl.h"
//#include "macro.h"
//
//#define CHECK_ERROR(err, msg) \
//    if (err != CL_SUCCESS) { \
//        LOG(ERROR) << msg << std::endl; \
//        exit(EXIT_FAILURE); \
//    }
//
//bool fp16_support;
//bool unified_memory_support;
//cl_platform_id _global_platform;
//cl_device_id _global_device;
//cl_context _global_context;
//cl_command_queue _global_queue;
//cl_program _global_program;
//// buffers set
//static std::set<cl_mem> _global_buffers;
//// kernel map, key is kernel name, value is kernel
//static std::map<std::string, cl_kernel> _global_kernels;
//const char* platform_name = OPENCL_PLATFORM_NAME;
////"NVIDIA CUDA";
//const char* device_name = OPENCL_DEVICE_NAME;
//const char* kernels_file = OPENCL_KERNELS_FILE;
//size_t max_work_group_size;
//void print_platforms() {
//    LOG(INFO) << "openCL Platforms:" << std::endl;
//    cl_int err;
//    cl_uint num_platforms;
//    cl_platform_id* platforms;
//    err = clGetPlatformIDs(0, nullptr, &num_platforms);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
//    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    for (unsigned int i = 0; i < num_platforms; i++) {
//        char _platform_name[128];
//        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, _platform_name, nullptr);
//        CHECK_ERROR(err, "failed to get platform info")
//        LOG(INFO) << "platform " << i << ": " << _platform_name << std::endl;
//    }
//    free(platforms);
//}
//void print_gpus()  {
//    LOG(INFO) << "openCL GPUs:" << std::endl;
//    cl_int err;
//    cl_uint num_platforms, num_devices;
//    cl_platform_id* platforms;
//    cl_device_id* devices;
//    err = clGetPlatformIDs(0, nullptr, &num_platforms);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
//    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    for (unsigned int i = 0; i < num_platforms; i++) {
//        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
//        if (err != CL_SUCCESS) {
//            continue;
//        }
//        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
//        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
//        CHECK_ERROR(err, "failed to get device IDs")
//        for (unsigned int j = 0; j < num_devices; j++) {
//            char deviceName[128];
//            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
//            CHECK_ERROR(err, "failed to get device info")
//            LOG(INFO) << "platform " << i << ", GPU " << j << ": " << deviceName << std::endl;
//        }
//        free(devices);
//    }
//    free(platforms);
//}
//
//void print_opencl_info() {
//    print_platforms();
//    print_gpus();
//}
//
//void print_now_using(cl_platform_id platform, cl_device_id device) {
//    LOG(INFO) << "now Using:" << std::endl;
//    char _platform_name[128];
//    char _device_name[128];
//    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, _platform_name, nullptr);
//    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, _device_name, nullptr);
//    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
//    LOG(INFO) << "platform: " << _platform_name << std::endl;
//    LOG(INFO) << "device: " << _device_name << std::endl;
//    LOG(INFO) << "max work group size: " << max_work_group_size << std::endl;
//}
//
//void print_now_using() {
//    print_now_using(_global_platform, _global_device);
//    char version[128]; // 存储版本信息的字符串
//    cl_int err = clGetPlatformInfo(_global_platform, CL_PLATFORM_VERSION, 128, version, NULL);
//    if(err != CL_SUCCESS) {
//    } else {
//        LOG(INFO) << "platform opencl version: " << version << std::endl;
//    }
//
//    err = clGetDeviceInfo(_global_device, CL_DEVICE_VERSION, 128, version, NULL);
//    if(err != CL_SUCCESS) {
//    } else {
//        LOG(INFO) << "device opencl version: " << version << std::endl;
//    }
//
//    err = clGetDeviceInfo(_global_device, CL_DRIVER_VERSION, 128, &version, NULL);
//    if(err != CL_SUCCESS) {
//        LOG(ERROR) << "failed to get driver version" << std::endl;
//    } else {
//        LOG(INFO) << "driver version: " << version << std::endl;
//    }
//}
//// return multiple values: platform and device, function name is get_platform_and_device(const char *platform_name, const char *device_name)
//
//std::tuple<cl_platform_id, cl_device_id> get_platform_and_device(const char* platform_name, const char* device_name) {
//    LOG(INFO) << "get Platform and Device:" << std::endl;
//    cl_int err;
//    cl_uint num_platforms, num_devices;
//    cl_platform_id* platforms;
//    cl_device_id* devices;
//    err = clGetPlatformIDs(0, nullptr, &num_platforms);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
//    err = clGetPlatformIDs(num_platforms, platforms, nullptr);
//    CHECK_ERROR(err, "failed to get platform IDs")
//    for (unsigned int i = 0; i < num_platforms; i++) {
//        char _platform_name[128];
//        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, _platform_name, nullptr);
//        CHECK_ERROR(err, "failed to get platform info")
//        if (strcmp(_platform_name, platform_name) == 0) {
//            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
//            CHECK_ERROR(err, "failed to get device IDs")
//            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
//            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
//            CHECK_ERROR(err, "failed to get device IDs")
//            for (unsigned int j = 0; j < num_devices; j++) {
//                char _device_name[128];
//                err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, _device_name, nullptr);
//                CHECK_ERROR(err, "failed to get device info")
//                if (strcmp(_device_name, device_name) == 0) {
//                    LOG(INFO) << "platform: " << _platform_name << std::endl;
//                    LOG(INFO) << "device: " << _device_name << std::endl;
//                    return std::make_tuple(platforms[i], devices[j]);
//                }
//            }
//            free(devices);
//        }
//    }
//    free(platforms);
//    return std::make_tuple(nullptr, nullptr);
//}
//void set_platform_and_device() {
//    // use get_platform_and_device to get platform and device, and set them to _global_platform and _global_device
//    auto tuple = get_platform_and_device(platform_name, device_name);
//    _global_platform = std::get<0>(tuple);
//    _global_device = std::get<1>(tuple);
//    if (_global_platform == nullptr || _global_device == nullptr) {
//        LOG(ERROR) << "platform or device not found" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//
//    // check unified memory support
//    check_unified_memory_support();
//    LOG(INFO) << "device unified memory support: " << (unified_memory_support ? "true" : "false") << std::endl;
//}
//void setup_opencl_env() {
//    // check _global_platform and _global_device
//    if (_global_platform == nullptr || _global_device == nullptr) {
//        LOG(ERROR) << "platform or device is not initialized" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    cl_context_properties properties[] = {
//        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)_global_platform, 0
//    };
//    // init context and queue
//    cl_int err;
//    _global_context = clCreateContext(properties, 1, &_global_device, NULL, NULL, &err);
//        // clCreateContext(nullptr, 1, &_global_device, nullptr, nullptr, &err);
//    CHECK_ERROR(err, "failed to create context")
//    _global_queue = clCreateCommandQueueWithProperties(_global_context, _global_device, nullptr, &err);
//    CHECK_ERROR(err, "failed to create command queue")
//}
//
//void build_all_kernels() {
//    cl_int err;
//    // read kernels.cl
//    FILE* fp = fopen(kernels_file, "rb");
//    if (fp == nullptr) {
//        LOG(ERROR) << "failed to open file " << kernels_file << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    fseek(fp, 0, SEEK_END);
//    size_t size = ftell(fp);
//    rewind(fp);
//    char* source = (char *)malloc(size + 1);
//    source[size] = '\0';
//    // read file
//    auto read_size = fread(source, sizeof(char), size, fp);
//    if (read_size != size) {
//        LOG(ERROR) << "failed to read file " << kernels_file << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    fclose(fp);
//    // create program
//    _global_program = clCreateProgramWithSource(_global_context, 1, (const char **)&source, nullptr, &err);
//    CHECK_ERROR(err, "failed to create program")
//    err = clBuildProgram(_global_program, 1, &_global_device, nullptr, nullptr, nullptr);
//    if (err == CL_BUILD_PROGRAM_FAILURE) {
//        size_t log_size;
//        clGetProgramBuildInfo(_global_program, _global_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
//
//        char* log = (char *)malloc(log_size);
//        clGetProgramBuildInfo(_global_program, _global_device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
//
//        LOG(ERROR) << "build log:" << std::endl << log << std::endl;
//
//        free(log);
//    }
//    CHECK_ERROR(err, "failed to build program")
//    // get kernel count
//    cl_uint num_kernels;
//    err = clCreateKernelsInProgram(_global_program, 0, nullptr, &num_kernels);
//    CHECK_ERROR(err, "failed to get kernel count")
//
//    // get kernel names
//    std::vector<cl_kernel> kernels(num_kernels);
//    err = clCreateKernelsInProgram(_global_program, num_kernels, kernels.data(), nullptr);
//    CHECK_ERROR(err, "failed to get kernel names")
//
//    for (unsigned int i = 0; i < num_kernels; i++) {
//        char kernel_name[128];
//        err = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 128, kernel_name, nullptr);
//        CHECK_ERROR(err, "failed to get kernel name")
//        _global_kernels[kernel_name] = kernels[i];
//    }
//}
//
//cl_kernel get_kernel(const char* kernel_name) {
//    // check _global_kernels
//    if (_global_kernels.empty()) {
//        LOG(ERROR) << "kernels are not initialized" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    auto kernel = _global_kernels.find(kernel_name);
//    if (kernel == _global_kernels.end()) {
//        LOG(ERROR) << "kernel " << kernel_name << " not found" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    return kernel->second;
//}
//
//void add_kernel(const char* kernel_name, cl_kernel kernel) {
//    if (_global_kernels.find(kernel_name) != _global_kernels.end()) {
//        LOG(ERROR) << "kernel " << kernel_name << " already exists" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    _global_kernels[kernel_name] = kernel;
//}
//
//void release_opencl_env() {
//    // release buffers
//    for (auto buffer: _global_buffers) {
//        clReleaseMemObject(buffer);
//    }
//    // release kernels
//    for (const auto& kernel: _global_kernels) {
//        clReleaseKernel(kernel.second);
//    }
//    // release program
//    clReleaseProgram(_global_program);
//    // release command queue
//    clReleaseCommandQueue(_global_queue);
//    // release context
//    clReleaseContext(_global_context);
//}
//
//void check_unified_memory_support() {
//    cl_bool unified_memory;
//    cl_int err;
//
//    err = clGetDeviceInfo(_global_device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL);
//    if (err != CL_SUCCESS) {
//        // 错误处理
//        printf("error getting device info: %d\n", err);
//        unified_memory_support = false;
//    }else {
//        unified_memory_support = (unified_memory == CL_TRUE);
//    }
//}
//
//void enqueue_kernel(cl_kernel kernel,cl_uint work_dim ,const size_t* global_work_size, const size_t* local_work_size) {
//    cl_int err = clEnqueueNDRangeKernel(_global_queue, kernel, work_dim, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
//    if (err != CL_SUCCESS) {
//        LOG(ERROR) << "failed to enqueue kernel. err: " << err << std::endl;
//        exit(EXIT_FAILURE);
//    }
//}
//
//void wait_queue_finish() {
//    cl_int err = clFinish(_global_queue);
//    CHECK_ERROR(err, "failed to finish queue")
//}
//
//cl_mem create_buffer(const buffer_type buffer_type, const size_t size, void* host_ptr) {
//    // check _global_context
//    if (_global_context == nullptr) {
//        LOG(ERROR) << "context is not initialized" << std::endl;
//        exit(EXIT_FAILURE);
//    }
//    cl_mem buffer;
//    switch (buffer_type) {
//        case buffer_type::READ_ONLY:
//            buffer = clCreateBuffer(_global_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, size, host_ptr, nullptr);
//            break;
//        case buffer_type::WRITE_ONLY:
//            buffer = clCreateBuffer(_global_context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, size, host_ptr, nullptr);
//            break;
//        case buffer_type::READ_WRITE:
//            buffer = clCreateBuffer(_global_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, host_ptr, nullptr);
//            break;
//        default:
//            LOG(ERROR) << "unknown buffer type" << std::endl;
//            exit(EXIT_FAILURE);
//    }
//    _global_buffers.insert(buffer);
//    return buffer;
//}
//
//int init_opencl() {
//    set_platform_and_device();
//    print_now_using();
//    setup_opencl_env();
//    build_all_kernels();
//    return 0;
//}
