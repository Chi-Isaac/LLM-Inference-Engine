cmake_minimum_required(VERSION 3.16)

project(LLMInferenceEngine VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. Change the default build type to Release for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 2. Properly find the OpenMP package
find_package(OpenMP REQUIRED)

# Find the BLAS package
find_package(BLAS REQUIRED)

add_executable(llm_inference
    src/model.cpp
    src/tokenizer.cpp
    src/transformer.cpp
)

target_include_directories(llm_inference PRIVATE include)

# 3. Add -O3 and -march=native here (removed -fopenmp from here)
target_compile_options(llm_inference PRIVATE
    -O3
    -march=native
    -ffast-math
    -Wall
    -Wextra
    -Wpedantic
)

# 4. Correctly link OpenMP to your executable for both compiling and linking
target_link_libraries(llm_inference PRIVATE 
    OpenMP::OpenMP_CXX
    BLAS::BLAS
)