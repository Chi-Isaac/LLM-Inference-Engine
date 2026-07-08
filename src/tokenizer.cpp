#include "tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

// Helper function to read binary data from bianry stream safely
// Throws std::runtime_error if reading fails
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
    // Open file in binary mode
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        return false;
    }

    // Reset existing tokenizer state
    vocab_.clear();
    max_token_length_ = 0;

    // Read the maximum token length from the file
    read_binary(input_file, max_token_length_);

    // While loop until EOF or failure
    while (true) {
        float score = 0.0f;
        std::uint32_t len = 0;

        // Attempt to read score, breaks if EOF is reached
        input_file.read(reinterpret_cast<char*>(&score), sizeof(score));
        if (!input_file) {
            break;
        }

        // Read the length of the token
        read_binary(input_file, len);
        if (len < 0) {
            throw std::runtime_error("tokenizer: negative token length in input file");
        }

        // Allocate str buffer and read
        std::string text(static_cast<std::size_t>(len), '\0');
        // Check token bytes are non-empty
        if (len > 0) {
            input_file.read(&text[0], len);
            if (!input_file) {
                throw std::runtime_error("tokenizer: failed to read token bytes");
            }
        }
        // Stores token in vocabulary
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

// Linear search for the token in the vocabulary
std::optional<std::size_t> Tokenizer::find_token(const std::string& text) const {
    for (std::size_t i = 0; i < vocab_.size(); ++i) {
        if (vocab_[i].text == text) {
            return i;
        }
    }
    return std::nullopt;
}