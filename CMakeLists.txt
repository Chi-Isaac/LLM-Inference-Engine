cmake_minimum_required(VERSION 3.16)

project(LLMInferenceEngine VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(llm_inference
    src/model.cpp
    src/tokenizer.cpp
    src/transformer.cpp
)

target_include_directories(llm_inference PRIVATE include)

target_compile_options(llm_inference PRIVATE
    -ffast-math
    -Wall
    -Wextra
    -Wpedantic
)

# Optional: nicer build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()