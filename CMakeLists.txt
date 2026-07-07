cmake_minimum_required(VERSION 3.16)
project(llm_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(llm_inference
    src/main.cpp
    src/tokenizer.cpp
)