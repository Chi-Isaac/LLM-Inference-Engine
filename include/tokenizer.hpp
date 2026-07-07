#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

struct Token {
    std::string text;
    std::uint32_t len;
    float score;
};

class Tokenizer {
public:
    bool load_from_file(const std::string& file_path);

    const Token& token_at(std::size_t id) const;
    std::string decode(std::size_t id) const;

    std::size_t size() const;
    int max_token_length() const;

    std::optional<std::size_t> find_token(const std::string& text) const;

private:
    int max_token_length_ = 0;
    std::vector<Token> vocab_;
}
;