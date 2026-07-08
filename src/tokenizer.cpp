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
        // Insert token into trie
        insert_token(vocab_.back().text, vocab_.size() - 1);
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

std::optional<std::size_t> Tokenizer::find_token(const std::string& text) const {
    const TrieNode* curr = &root_;
    for (char c : text) {
        auto it = curr->children.find(c);
        if (it == curr->children.end()) {
            return std::nullopt; // Token not found
        }
        curr = it->second.get();
    }
    if (curr->token_id.has_value()) {
        return curr->token_id; // Token found
    }
    return std::nullopt; // Token not found
}

// Walks along trie from start pos until no more matches can be made and returns if any match made
std::optional<std::pair<std::size_t, std::size_t>> Tokenizer::longest_match(const std::string& text, std::size_t start) const {
    const TrieNode* curr = &root_;
    std::optional<std::size_t> last_token_id;
    std::size_t last_token_length = 0;
    
    for (std::size_t i = start; i < text.size(); ++i) {
        char c = text[i];
        auto it = curr->children.find(c);
        if (it == curr->children.end()) {
            break; // No further match
        }
        curr = it->second.get();
        if (curr->token_id.has_value()) {
            last_token_id = curr->token_id;
            last_token_length = i - start + 1; // Update length of the last matched token
        }
    }

    if (last_token_id.has_value()) {
        return std::make_pair(last_token_id.value(), last_token_length);
    }
    return std::nullopt; // No match found
}

// Walks through input text, finds longest matching tokens and returns IDs
std::vector<std::size_t> Tokenizer::encode(const std::string& text) const {
    std::vector<std::size_t> result;
    std::size_t pos = 0;
    while (pos < text.size()) {
        auto match = longest_match(text, pos);
        if (!match.has_value()) {
            throw std::runtime_error("tokenizer: no matching token at position " + std::to_string(pos));
        }
        auto [token_id, token_length] = match.value();
        result.push_back(token_id);
        pos += token_length; // Move position forward by the length of the matched token
    }
    return result;
}

std::size_t Tokenizer::size() const {
    return vocab_.size();
}

int Tokenizer::max_token_length() const {
    return max_token_length_;
}

void Tokenizer::insert_token(const std::string& text, std::size_t id) {
    TrieNode* curr = &root_;
    for (char c : text) {
        auto& child = curr->children[c];
        if (!child) {
            child = std::make_unique<TrieNode>();
        }
        curr = child.get();
    }
    curr->token_id = id;
}

bool Tokenizer::merge_best_pair(std::vector<int>& tokens) const {
    if (tokens.size() < 2) {
        return false; // Not enough tokens to merge
    }

    float best_score = -std::numeric_limits<float>::infinity();
    std::size_t best_id = -1;
    std::size_t best_pos = -1;
    bool found = false;

    for (std::size_t i = 0; i < tokens.size() - 1; ++i) {
        std::string merged = vocab_[tokens[i]].text + vocab_[tokens[i + 1]].text;
        auto id = find_token(merged);
        if (!id) {
            continue; // Merged token not found
        }
        
        float score = vocab_[id.value()].score;
        if (score > best_score || !found) {
            found = true;
            best_score = score;
            best_id = id.value();
            best_pos = i;
        }
    }

    if (!found) {
        return false; // No mergeable pair found
    }

    tokens[best_pos] = best_id; // Replace first token with merged
    tokens.erase(tokens.begin() + best_pos + 1); // Remove the second original token
    return true; // Merge successful
}