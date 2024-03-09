cmake_minimum_required(VERSION 3.25)
project(gemma_ggml)

add_subdirectory(lib)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "src/*.cpp")
include_directories(src)

add_executable(gemma_ggml main.cpp ${SOURCES})
add_executable(run_tests test_main.cpp ${SOURCES})


find_package(glog REQUIRED)
find_package(GTest REQUIRED)
# create a string to hold all lib
set(LIBS ggml glog::glog GTest::GTest GTest::Main)

target_link_libraries(gemma_ggml ${LIBS})
target_link_libraries(run_tests ${LIBS})
