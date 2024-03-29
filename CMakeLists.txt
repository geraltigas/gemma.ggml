cmake_minimum_required(VERSION 3.25)
project(gemma_ggml)

#include_directories("/opt/intel/oneapi/2024.0/include")
#link_directories("/opt/intel/oneapi/2024.0/lib")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Rno-debug-disables-optimization")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories(src)


add_subdirectory(lib)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(gemma_ggml main.cpp ${SOURCES})
add_executable(run_tests test_main.cpp ${SOURCES})

find_package(glog REQUIRED)
find_package(GTest REQUIRED)
# find opencl
find_package(OpenCL REQUIRED)

find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
find_library(GMPXX_LIBRARY NAMES gmpxx)
if(GMP_INCLUDE_DIR AND GMP_LIBRARY AND GMPXX_LIBRARY)
    message(STATUS "Found GMP: ${GMP_LIBRARY}, ${GMPXX_LIBRARY}")
else()
    message(FATAL_ERROR "Cannot find GMP")
endif()

target_include_directories(gemma_ggml PRIVATE ${GMP_INCLUDE_DIR})

# create a string to hold all lib
set(LIBS ggml glog::glog GTest::GTest GTest::Main OpenCL::OpenCL ${GMP_LIBRARY} ${GMPXX_LIBRARY})
#        "/opt/intel/oneapi/2024.0/lib/libimf.so"
#        "/opt/intel/oneapi/2024.0/lib/libsvml.so"
#        "/opt/intel/oneapi/2024.0/lib/libirng.so"
#        "/opt/intel/oneapi/2024.0/lib/libintlc.so.5"

target_link_libraries(gemma_ggml ${LIBS})
target_link_libraries(run_tests ${LIBS})
