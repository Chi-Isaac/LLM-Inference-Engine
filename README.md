# LLM Inference Engine
A custom-built, lightweight Large Language Model (LLM) inference engine written from scratch in C++.  
This project focuses on understanding transformer architectures at a low level by implementing the core inference loop, tokenisation, and model weight loading without relying on machine learning frameworks such as PyTorch or TensorFlow.

---
## Contributors
- Advay Goel
- Isaac Chi

---
## Project Structure
```text
├── CMakeLists.txt
├── data
│   ├── stories15M.bin
│   ├── tinyllama_chat.bin
│   └── tokenizer.bin
├── include
│   ├── model.hpp
│   ├── tokenizer.hpp
│   └── transformer.hpp
├── scripts
│   ├── download_tinyllama.py
└── src
    ├── model.cpp
    ├── tokenizer.cpp
    └── transformer.cpp
```

---
## Prerequisites
To build and run this engine, you will need:
- A C++17 compatible compiler (e.g. GCC)
- [CMake](https://cmake.org/) (minimum version 3.16)
- **OpenMP**
- A **BLAS** implementation (e.g. CBLAS)

---
## Building the Project
This project is built using CMake.
```bash
# Clone the repository
git clone https://github.com/Chi-Isaac/LLM-Inference-Engine.git
cd LLM-Inference-Engine

# Create a build directory
mkdir build

# Configure and compile
cmake -S . -B build

# Build the project (replace 8 with the number of cores)
cmake --build build -j 8
```

---
## Running Inference
1. Download the required `.bin` model weights (e.g. `tinyllama-1.1b-chat`) and the corresponding tokeniser file into the `data/` directory.
    - `tokenizer.bin` and weights for the `stories15M` model are provided in the `data/` directory
    - A Python script to download the model weights for a TinyLlama model is provided in the `scripts/` directory
2. Execute the compiled binary from the root directory, passing the model path, tokeniser path, and formatted prompt.

Here is an example using the TinyLlama prompt format:
```bash
./build/llm_inference "./data/tinyllama_chat.bin" "./data/tokenizer.bin" $'<|system|>\nYou are a helpful and concise AI assistant.</s>\n<|user|>\nGive me a bulleted list of 5 fun indoor hobbies to try on a rainy day.</s>\n<|assistant|>\n'
```

---
## Acknowledgements
This project makes use of several open-source libraries and resources. Thanks to:
- **[llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy**: For the necessary .bin files (stories15M.bin and tokenizer.bin)
- **Meta and The Open-Source AI Community**: For the Llama architecture innovations implemented in this engine (RoPE, RMSNorm, SwiGLU, and Grouped-Query Attention), as well as the creators of the **TinyLlama** project for providing an accessible 1.1B parameter model.
- **OpenMP**: For allowing seamless CPU multithreading cross the attention mechanism and activation functions.
- **CBLAS (Basic Linear Algebra Subprograms)**: For providing optimised matrix multiplication routines.
