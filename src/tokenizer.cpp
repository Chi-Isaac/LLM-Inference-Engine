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

bool Tokenizer::load_from_file(const std::string& file_path) {
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        return false;
    }

    vocab_.clear();
    max_token_length_ = 0;

    read_binary(input_file, max_token_length_);

    while (true) {
        float score = 0.0f;
        std::int32_t len = 0;

        input_file.read(reinterpret_cast<char*>(&score), sizeof(score));
        if (!input_file) {
            break;
        }

        read_binary(input_file, len);

        if (len < 0) {
            throw std::runtime_error("tokenizer: negative token length in input file");
        }

        std::string text(static_cast<std::size_t>(len), '\0');
        if (len > 0) {
            input_file.read(&text[0], len);
            if (!input_file) {
                throw std::runtime_error("tokenizer: failed to read token bytes");
            }
        }
        vocab_.push_back(Token{std::move(text), len, score});
    }

    return true;
}

const Token& Tokenizer::token_at(std::size_t id) const {
    if (id >= vocab_.size()) {
        throw std::out_of_range("tokenizer: token id out of range");
    }
    return vocab_[id];
}

std::string Tokenizer::decode(std::size_t id) const {
    return token_at(id).text;
}

std::size_t Tokenizer::size() const {
    return vocab_.size();
}

int Tokenizer::max_token_length() const {
    return max_token_length_;
}