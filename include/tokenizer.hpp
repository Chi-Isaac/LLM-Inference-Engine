#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <memory>

struct Token {
    std::string text;
    std::uint32_t len;
    float score;
};

struct TrieNode {
    // Children nodes stored in hashmap
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    std::optional<std::size_t> token_id;
};

class Tokenizer {
public:
    bool load_from_file(const std::string& file_path);

    const Token& token_at(std::size_t id) const;
    std::string decode(std::size_t id) const;

    std::optional<std::size_t> find_token(const std::string& text) const;
    std::optional<std::pair<std::size_t, std::size_t>> longest_match(const std::string& text, std::size_t start) const;
    std::vector<std::size_t> encode(const std::string& text) const;

    std::size_t size() const;
    int max_token_length() const;

private:
    int max_token_length_ = 0;
    std::vector<Token> vocab_;
    TrieNode root_;

    void insert_token(const std::string& text, std::size_t id);
    bool Tokenizer::merge_best_pair(std::vector<int>& tokens) const;
}
;