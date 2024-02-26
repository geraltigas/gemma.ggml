cmake_minimum_required(VERSION 3.25)
project(gemma_ggml)

add_subdirectory(lib)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "src/*.cpp")
include_directories(src)

add_executable(gemma_ggml main.cpp ${SOURCES})

# link glog
find_package(glog REQUIRED)
target_link_libraries(gemma_ggml glog::glog ggml)
