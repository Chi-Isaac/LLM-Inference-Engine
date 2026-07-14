#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <memory>

const std::uint8_t UTF8_ASCII_MASK = 0x80; // 1000 0000
const std::uint8_t UTF8_LEAD_2_BYTE_MASK = 0xE0; // 1110 0000
const std::uint8_t UTF8_LEAD_3_BYTE_MASK = 0xF0; // 1111 0000
const std::uint8_t UTF8_LEAD_4_BYTE_MASK = 0xF8; // 1111 1000

const std::uint8_t ASCII_PREFIX = 0x00; // 0000 0000
const std::uint8_t UTF8_LEAD_2_BYTE_PREFIX = 0xC0; // 1100 0000
const std::uint8_t UTF8_LEAD_3_BYTE_PREFIX = 0xE0; // 1110 0000
const std::uint8_t UTF8_LEAD_4_BYTE_PREFIX = 0xF0; // 1111 0000

const std::uint8_t UTF8_CONTINUATION_MASK = 0xC0; // 1100 0000
const std::uint8_t UTF8_CONTINUATION_PREFIX = 0x80; // 1000 0000

const int start_id = 1;
const int end_id = 2;

struct Token {
    std::string text;
    std::uint32_t len;
    float score;
};

struct TrieNode {
    // Children nodes stored in hashmap
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    std::optional<int> token_id;
};

class Tokenizer {
public:
    bool load_from_file(const std::string& file_path);

    const Token& token_at(int id) const;
    std::string decode(int id) const;

    std::optional<int> find_token(const std::string& text) const;
    std::optional<std::pair<int, int>> longest_match(const std::string& text, int start) const;
    std::vector<int> encode(const std::string& text) const;

    int size() const;
    int max_token_length() const;

private:
    int max_token_length_ = 0;
    std::vector<Token> vocab_;
    TrieNode root_;

    void insert_token(const std::string& text, int id);
    bool merge_best_pair(std::vector<int>& tokens) const;
}
;