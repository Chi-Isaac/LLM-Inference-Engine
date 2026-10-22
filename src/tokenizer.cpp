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

    bool is_byte_fallback_piece(const std::string& text) {
        return text.size() == BYTE_FALLBACK_LENGTH &&
               text.compare(0, 
                            BYTE_FALLBACK_PREFIX.size(), 
                            BYTE_FALLBACK_PREFIX) == 0 &&
               text.back() == BYTE_FALLBACK_SUFFIX;
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

        std::string text(static_cast<size_t>(len), '\0');
        if (len > 0) {
            input_file.read(&text[0], len);
            if (!input_file) {
                throw std::runtime_error("tokenizer: failed to read token bytes");
            }
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
std::string Tokenizer::decode(int id) const {
    const std::string& text = token_at(id).text;

    if (id == START_ID || id == END_ID) {
        return "";
    }

    if (is_byte_fallback_piece(text)) {
        int hi = hex_value(text[BYTE_FALLBACK_HIGH_POS]);
        int lo = hex_value(text[BYTE_FALLBACK_LOW_POS]);
        if (hi >= 0 && lo >= 0) {
            char byte = static_cast<char>((hi << NIBBLE_SIZE) | lo);
            return std::string(1, byte);
        }
    }

    return text;
}

std::optional<int> Tokenizer::find_token(const std::string& text) const {
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

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    result.push_back(START_ID); // BOS inserted here only

    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        unsigned char b = static_cast<unsigned char>(text[i++]);

        int num_cont_bytes = 0;
        if ((b & UTF8_ASCII_MASK) == ASCII_PREFIX) {
            num_cont_bytes = 0;
        } else if ((b & UTF8_LEAD_2_BYTE_MASK) == UTF8_LEAD_2_BYTE_PREFIX) {
            num_cont_bytes = 1;
        } else if ((b & UTF8_LEAD_3_BYTE_MASK) == UTF8_LEAD_3_BYTE_PREFIX) {
            num_cont_bytes = 2;
        } else if ((b & UTF8_LEAD_4_BYTE_MASK) == UTF8_LEAD_4_BYTE_PREFIX) {
            num_cont_bytes = 3;
        } else {
            throw std::runtime_error("tokenizer: invalid UTF-8 start byte encountered");
        }

        while (num_cont_bytes > 0) {
            if (i >= text.size()) {
                throw std::runtime_error("tokenizer: truncated UTF-8 sequence");
            }
            unsigned char c = static_cast<unsigned char>(text[i]);
            if ((c & UTF8_CONTINUATION_MASK) != UTF8_CONTINUATION_PREFIX) {
                throw std::runtime_error("tokenizer: invalid UTF-8 continuation byte encountered");
            }
            ++i;
            --num_cont_bytes;
        }

        std::string_view piece = std::string_view(text).substr(start, i - start);

        // First try full UTF-8 chunk.
        if (auto id = find_token(piece); id.has_value()) {
            result.push_back(id.value());
            continue;
        }

        // Otherwise byte-fallback as <0xNN> pieces.
        for (size_t j = start; j < i; ++j) {
            unsigned char byte = static_cast<unsigned char>(text[j]);
            std::string fallback = byte_fallback_token(byte);
            auto byte_id = find_token(fallback);
            if (!byte_id.has_value()) {
                throw std::runtime_error("tokenizer: no fallback token for piece: " + fallback);
            }
            result.push_back(byte_id.value());
        }
    }

    while (merge_best_pair(result)) {
    }

    result.erase(std::remove(result.begin(), result.end(), -1), result.end());
    return result;
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
    if (tokens.size() < 2) {
        return false;
    }

    float best_score = -std::numeric_limits<float>::infinity();
    int best_id = -1;
    size_t best_pos = 0;
    size_t best_right_pos = 0;
    bool found = false;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        if (tokens[i] == -1) continue;

        size_t right = i + 1;
        while (right < tokens.size() && tokens[right] == -1) {
            ++right;
        }
        if (right >= tokens.size()) {
            break;
        }

        if (tokens[i] < 0 || tokens[right] < 0) {
            continue;
        }

        std::string merged = vocab_[tokens[i]].text + vocab_[tokens[right]].text;
        auto id = find_token(merged);
        if (!id.has_value()) {
            continue;
        }

        float score = vocab_[id.value()].score;
        if (!found || score > best_score) {
            found = true;
            best_score = score;
            best_id = id.value();
            best_pos = i;
            best_right_pos = right;
        }
    }

    if (!found) {
        return false;
    }

    tokens[best_pos] = best_id;
    tokens[best_right_pos] = -1;
    return true;
}
