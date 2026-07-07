#include "tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {
    template <typename T>
    void read_binary(std::ifstream& input_file, T& value) {
        input_file.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!input_file) {
            throw std::runtime_error("tokenizer: failed to read binary data");
        }
    }
}
