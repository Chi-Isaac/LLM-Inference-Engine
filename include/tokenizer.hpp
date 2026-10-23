#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

inline constexpr int START_ID = 1;
inline constexpr int END_ID = 2;

struct Token {
    std::string text;
    std::uint32_t len;
    float score;
};

class Tokenizer {
public:
    bool load_from_file(const std::string& file_path);

    const Token& token_at(int id) const;
    std::string decode(int prev_id, int id) const;

    std::optional<int> find_token(std::string_view text) const;
    std::vector<int> encode(std::string_view text, bool add_bos = true, bool add_eos = false) const;

    int size() const;
    int max_token_length() const;

private:
    struct TrieNode {
        // Children nodes stored in hashmap
        std::unordered_map<char, std::unique_ptr<TrieNode>> children;
        std::optional<int> token_id;
    };

    int max_token_length_ = 0;
    std::vector<Token> vocab_;
    TrieNode root_;

    void insert_token(const std::string& text, int id);
    bool merge_best_pair(std::vector<int>& tokens) const;
};