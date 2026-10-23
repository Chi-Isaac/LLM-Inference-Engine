#include "tokenizer.hpp"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <limits>
#include <algorithm>
#include <iostream>
#include <string>
#include <optional>

namespace {
    constexpr std::uint8_t UTF8_ASCII_MASK = 0x80; // 1000 0000
    constexpr std::uint8_t UTF8_LEAD_2_BYTE_MASK = 0xE0; // 1110 0000
    constexpr std::uint8_t UTF8_LEAD_3_BYTE_MASK = 0xF0; // 1111 0000
    constexpr std::uint8_t UTF8_LEAD_4_BYTE_MASK = 0xF8; // 1111 1000

    constexpr std::uint8_t ASCII_PREFIX = 0x00; // 0000 0000
    constexpr std::uint8_t UTF8_LEAD_2_BYTE_PREFIX = 0xC0; // 1100 0000
    constexpr std::uint8_t UTF8_LEAD_3_BYTE_PREFIX = 0xE0; // 1110 0000
    constexpr std::uint8_t UTF8_LEAD_4_BYTE_PREFIX = 0xF0; // 1111 0000

    constexpr std::uint8_t UTF8_CONTINUATION_MASK = 0xC0; // 1100 0000
    constexpr std::uint8_t UTF8_CONTINUATION_PREFIX = 0x80; // 1000 0000

    constexpr std::string_view BYTE_FALLBACK_TEMPLATE = "<0x00>";
    constexpr std::string_view BYTE_FALLBACK_PREFIX = "<0x";
    constexpr char BYTE_FALLBACK_SUFFIX = '>';
    
    constexpr int BYTE_FALLBACK_LENGTH = BYTE_FALLBACK_TEMPLATE.size();
    constexpr std::size_t BYTE_FALLBACK_HIGH_POS = BYTE_FALLBACK_PREFIX.size();
    constexpr std::size_t BYTE_FALLBACK_LOW_POS = BYTE_FALLBACK_HIGH_POS + 1;
    constexpr std::string_view HEX_DIGITS = "0123456789ABCDEF";
    constexpr int NIBBLE_SIZE = 4; // 1 nibble = 4 bits
    constexpr std::uint8_t BOTTOM_NIBBLE_MASK = 0xF; // To keep lowest hex digit

    constexpr int HEX_DIGIT_OFFSET_VALUE = 10; // 'A' = 10

    constexpr int NO_TOKEN_ID = -1; // No merge found yet

    /* Reads sizeof(T) raw bytes from input_file and copies them into the memory
       of value. */

    template <typename T>
    void read_binary(std::ifstream& input_file, T& value) {
        static_assert(std::is_trivially_copyable_v<T>, 
            "tokenizer: read_binary requires a trivially copyable type.");
        input_file.read(reinterpret_cast<char*>(&value), sizeof(T));
        if (!input_file) {
            throw std::runtime_error("tokenizer: failed to read binary data");
        }
    }

    /* Returns the byte fallback token for a single byte.
       For example, 0x4F returns "<0x4F>".
       Used when a UTF-8 char has no token, so each byte is encoded as a 
       fallback token. */
    std::string byte_fallback_token(unsigned char b) {
        std::string s(BYTE_FALLBACK_TEMPLATE);
        s[BYTE_FALLBACK_HIGH_POS] = HEX_DIGITS[(b >> NIBBLE_SIZE)];
        s[BYTE_FALLBACK_LOW_POS] = HEX_DIGITS[b & BOTTOM_NIBBLE_MASK];
        return s;
    }

    int hex_value(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return HEX_DIGIT_OFFSET_VALUE + (c - 'A');
        if (c >= 'a' && c <= 'f') return HEX_DIGIT_OFFSET_VALUE + (c - 'a');
        return -1;
    }

    /* Return byte encoded by a byte fallback token.
       For example, "<0x1F>" returns 0x1F.
       Returns nullopt if the text is not a valid byte fallback token. */
    std::optional<unsigned char> byte_fallback_parser(std::string_view text) {
        if (text.size() != BYTE_FALLBACK_LENGTH ||
            text.substr(0, BYTE_FALLBACK_PREFIX.size()) != BYTE_FALLBACK_PREFIX ||
            text.back() != BYTE_FALLBACK_SUFFIX) {
            return std::nullopt;
        }

        int hi = hex_value(text[BYTE_FALLBACK_HIGH_POS]);
        int lo = hex_value(text[BYTE_FALLBACK_LOW_POS]);
        if (hi < 0 || lo < 0) {
            return std::nullopt;
        }
        return static_cast<unsigned char>((hi << NIBBLE_SIZE) | lo);
    }

    /* Returns the number of bytes in the UTF-8 sequence that starts with a 
       lead byte b. If b cannot be the start of a sequence, return 0. */
    int utf8_sequence_length(unsigned char b) {
        if ((b & UTF8_ASCII_MASK) == ASCII_PREFIX) return 1;
        if ((b & UTF8_LEAD_2_BYTE_MASK) == UTF8_LEAD_2_BYTE_PREFIX) return 2;
        if ((b & UTF8_LEAD_3_BYTE_MASK) == UTF8_LEAD_3_BYTE_PREFIX) return 3;
        if ((b & UTF8_LEAD_4_BYTE_MASK) == UTF8_LEAD_4_BYTE_PREFIX) return 4;
        return 0;
    }

    bool is_utf8_continuation(unsigned char b) {
        return (b & UTF8_CONTINUATION_MASK) == UTF8_CONTINUATION_PREFIX;
    }
}

bool Tokenizer::load_from_file(const std::string& file_path) {
    std::ifstream input_file(file_path, std::ios::binary);
    if (!input_file) {
        return false;
    }

    vocab_.clear();
    max_token_length_ = 0;
    root_ = TrieNode();

    read_binary(input_file, max_token_length_);

    while (input_file.peek() != std::char_traits<char>::eof()) {
        float score = 0.0f;
        read_binary(input_file, score);
        std::int32_t len = 0;
        read_binary(input_file, len);

        if (len <= 0) {
            throw std::runtime_error("tokenizer: non-positive token length in input file");
        }

        std::string text(static_cast<size_t>(len), '\0');
        input_file.read(&text[0], len);
        if (!input_file) {
            throw std::runtime_error("tokenizer: failed to read token bytes");
        }

        vocab_.push_back(Token{std::move(text), static_cast<uint32_t>(len), score});
        insert_token(vocab_.back().text, static_cast<int>(vocab_.size()) - 1);
    }

    return true;
}

const Token& Tokenizer::token_at(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
        throw std::out_of_range("tokenizer: token id out of range");
    }
    return vocab_[id];
}

// Converts token id back into text
std::string Tokenizer::decode(int prev_id, int id) const {
    const std::string& text = token_at(id).text;

    if (id == START_ID || id == END_ID) {
        return "";
    }

    if (auto byte = byte_fallback_parser(text)) {
        return std::string(1, static_cast<char>(*byte));
    }

    /* Since Llama model trained with tokens having leading spaces, we need to
       undo the leading space from encode in the first token after BOS. */
    if (prev_id == START_ID && !text.empty() && text.front() == ' ') {
        return text.substr(1);
    }

    return text;
}

std::optional<int> Tokenizer::find_token(std::string_view text) const {
    const TrieNode* curr = &root_;
    for (unsigned char c : text) {
        auto it = curr->children.find(static_cast<char>(c));
        if (it == curr->children.end()) {
            return std::nullopt;
        }
        curr = it->second.get();
    }
    if (curr->token_id.has_value()) {
        return curr->token_id;
    }
    return std::nullopt;
}

std::optional<std::pair<int, int>> Tokenizer::longest_match(const std::string& text, int start) const {
    const TrieNode* curr = &root_;
    std::optional<int> last_token_id;
    int last_token_length = 0;

    for (size_t i = static_cast<size_t>(start); i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        auto it = curr->children.find(static_cast<char>(c));
        if (it == curr->children.end()) {
            break;
        }
        curr = it->second.get();
        if (curr->token_id.has_value()) {
            last_token_id = curr->token_id;
            last_token_length = static_cast<int>(i - static_cast<size_t>(start) + 1);
        }
    }

    if (last_token_id.has_value()) {
        return std::make_pair(last_token_id.value(), last_token_length);
    }
    return std::nullopt;
}

// Defaults to true add_bos and false add_eos to exhibit old behaviour
std::vector<int> Tokenizer::encode(std::string_view text, bool add_bos, bool add_eos) const {
    std::vector<int> tokens;
    // Reserve to save allocations later. +2 for BOS and EOS.
    tokens.reserve(text.size() + 2);

    if (!text.empty()) {
        tokens.push_back(find_token(" ").value());
    }

    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        std::size_t len = utf8_sequence_length(static_cast<unsigned char>(text[start]));
        /* When the UTF-8 is invalid, take one byte as a fallback byte.
           Advance by one so next byte can be considered alone (may be a valid
           character). */
        if (len == 0 || start + len > text.size()) {
            len = 1;
        }
        for (std::size_t j = 1; j < len; ++j) {
            if (!is_utf8_continuation(static_cast<unsigned char>(text[start + j]))) {
                len = 1;
            }
        }
        i = start + len;

        std::string_view piece = text.substr(start, i - start);

        // First try full UTF-8 chunk.
        if (auto id = find_token(piece); id.has_value()) {
            tokens.push_back(*id);
            continue;
        }

        // Otherwise byte-fallback as <0xNN> pieces.
        for (unsigned char byte : piece) {
            std::string fallback = byte_fallback_token(byte);
            auto byte_id = find_token(fallback);
            if (!byte_id) {
                throw std::runtime_error("tokenizer: no fallback token for piece: " + fallback);
            }
            tokens.push_back(*byte_id);
        }
    }

    while (merge_best_pair(tokens)) {
    }

    // Insert BOS and EOS after merging so not involved in merge step.
    if (add_bos) tokens.insert(tokens.begin(), START_ID);
    if (add_eos) tokens.push_back(END_ID);
    return tokens;
}

int Tokenizer::size() const {
    return static_cast<int>(vocab_.size());
}

int Tokenizer::max_token_length() const {
    return max_token_length_;
}

void Tokenizer::insert_token(const std::string& text, int id) {
    TrieNode* curr = &root_;
    for (unsigned char c : text) {
        auto& child = curr->children[static_cast<char>(c)];
        if (!child) {
            child = std::make_unique<TrieNode>();
        }
        curr = child.get();
    }
    curr->token_id = id;
}

bool Tokenizer::merge_best_pair(std::vector<int>& tokens) const {
    float best_score = -std::numeric_limits<float>::infinity();
    int best_id = NO_TOKEN_ID;
    size_t best_pos = 0;

    // Avoid reallocation by reusing buffer
    std::string merged;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        merged.assign(vocab_[tokens[i]].text);
        merged += vocab_[tokens[i + 1]].text;

        auto id = find_token(merged);
        if (!id.has_value()) {
            continue;
        }

        float score = vocab_[id.value()].score;
        if (best_id == NO_TOKEN_ID || score > best_score) {
            best_score = score;
            best_id = id.value();
            best_pos = i;
        }
    }

    if (best_id == NO_TOKEN_ID) {
        return false;
    }

    tokens[best_pos] = best_id;
    // Remove token that got merged into best_pos position
    tokens.erase(tokens.begin() + best_pos + 1);
    return true;
}
