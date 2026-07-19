#include "transformer.hpp"
#include "tokenizer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdexcept>
#include <random>

struct TokenProb {
    int id;
    float prob;
};

// Load token embedding for ids[position] into x
void lookup(const Model& model, int id, std::vector<float>& x) {
    for (int i = 0; i < model.config.dim; i++) {
        x[i] = model.weights.token_embedding_table[id * model.config.dim + i];
    }
}

std::vector<float> apply_rope_q(
    const Model& model,
    const std::vector<float>& x,
    int pos)
{
    int head_size = model.config.dim / model.config.n_heads;
    std::vector<float> out(x.size());

    for (int h = 0; h < model.config.n_heads; h++) {
        int base = h * head_size;
        for (int i = 0; i < head_size; i += 2) {
            int pair_idx = i / 2;
            float cosv = model.rpe_cache.cos_cache[pos][pair_idx];
            float sinv = model.rpe_cache.sin_cache[pos][pair_idx];

            float a = x[base + i];
            float b = x[base + i + 1];

            out[base + i]     = a * cosv - b * sinv;
            out[base + i + 1] = a * sinv + b * cosv;
        }
    }
    return out;
}

std::vector<float> apply_rope_k(
    const Model& model,
    const std::vector<float>& x,
    int pos)
{
    int head_size = model.config.dim / model.config.n_heads;
    int kv_dim = model.config.n_kv_heads * head_size;
    std::vector<float> out(kv_dim);

    for (int h = 0; h < model.config.n_kv_heads; h++) {
        int base = h * head_size;
        for (int i = 0; i < head_size; i += 2) {
            int pair_idx = i / 2;
            float cosv = model.rpe_cache.cos_cache[pos][pair_idx];
            float sinv = model.rpe_cache.sin_cache[pos][pair_idx];

            float a = x[base + i];
            float b = x[base + i + 1];

            out[base + i]     = a * cosv - b * sinv;
            out[base + i + 1] = a * sinv + b * cosv;
        }
    }
    return out;
}

std::vector<float> rms_norm(const Model& model, const std::vector<float>& vector, int curr_layer, float* norm_weights) {
    float epsilon = 1e-5f;
    float sq_sum = 0.0f;

    for (int i = 0; i < model.config.dim; i++) {
        sq_sum += vector[i] * vector[i];
    }

    float rms = std::sqrt(sq_sum / model.config.dim + epsilon);
    float* gamma = norm_weights + curr_layer * model.config.dim;

    std::vector<float> norm(model.config.dim);
    for (int i = 0; i < model.config.dim; i++) {
        norm[i] = vector[i] * gamma[i] / rms;
    }

    return norm;
}

std::vector<float> pre_att_norm(const Model& model, const std::vector<float>& vector, int curr_layer) {
    return rms_norm(model, vector, curr_layer, model.weights.rms_att_weight);
}

std::vector<float> pre_ffn_norm(const Model& model, const std::vector<float>& vector, int curr_layer) {
    return rms_norm(model, vector, curr_layer, model.weights.rms_ffn_weight);
}

std::vector<float> final_norm(const Model& model, const std::vector<float>& vector) {
    return rms_norm(model, vector, 0, model.weights.rms_final_weight);
}

void mult_matrix(std::vector<float>& result, const std::vector<float>& x, const float* w, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float value = 0.0f;
        for (int j = 0; j < cols; j++) {
            value += x[j] * w[i * cols + j];
        }
        result[i] = value;
    }
}

// Assumes:
// key_cache[layer * seq_len + pos][kv_dim]
// value_cache[layer * seq_len + pos][kv_dim]
void compute_attention(const Model& model,
                       std::vector<float>& attn_out,
                       const std::vector<float>& q,
                       int curr_pos,
                       int layer_index) {
    int head_size = model.config.dim / model.config.n_heads;
    int kv_dim = model.config.n_kv_heads * head_size;
    int queries_per_group = model.config.n_heads / model.config.n_kv_heads;

    for (int head = 0; head < model.config.n_heads; head++) {
        const float* q_head = q.data() + head * head_size;
        int kv_head = head / queries_per_group;

        std::vector<float> attn_scores(curr_pos + 1);

        for (int pos = 0; pos <= curr_pos; pos++) {
            int row = layer_index * model.config.seq_len + pos;

            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                int col = kv_head * head_size + i;
                score += q_head[i] * model.kv_cache.key_cache[row][col];
            }

            score /= std::sqrt(static_cast<float>(head_size));
            attn_scores[pos] = score;
        }

        // Softmax inline for attention scores
        float max_score = attn_scores[0];
        for (int i = 1; i <= curr_pos; i++) {
            if (attn_scores[i] > max_score) {
                max_score = attn_scores[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i <= curr_pos; i++) {
            attn_scores[i] = std::exp(attn_scores[i] - max_score);
            sum += attn_scores[i];
        }
        for (int i = 0; i <= curr_pos; i++) {
            attn_scores[i] /= sum;
        }

        float* out_head = attn_out.data() + head * head_size;
        for (int i = 0; i < head_size; i++) {
            out_head[i] = 0.0f;
        }

        for (int pos = 0; pos <= curr_pos; pos++) {
            int row = layer_index * model.config.seq_len + pos;

            for (int i = 0; i < head_size; i++) {
                int col = kv_head * head_size + i;
                out_head[i] += attn_scores[pos] * model.kv_cache.value_cache[row][col];
            }
        }
    }
}

void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int>& history, float penalty = 1.05f) {
    if (penalty <= 1.0f || history.empty()) return;

    // Sliding window: only penalize tokens from the last 64 generated tokens
    int window_size = 16;
    int start_idx = std::max(0, static_cast<int>(history.size()) - window_size);

    for (int i = start_idx; i < history.size(); i++) {
        int token_id = history[i];
        float& logit = logits[token_id];
        
        if (logit > 0.0f) {
            logit /= penalty;
        } else {
            logit *= penalty;
        }
    }
}

float calculate_entropy(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_val);
        sum_exp += probs[i];
    }
    float entropy = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        float p = probs[i] / sum_exp;
        if (p > 1e-10f) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

void apply_top_p(std::vector<float>& probs, float top_p) {
    std::vector<TokenProb> sorted_probs;
    sorted_probs.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); i++) {
        sorted_probs.push_back({static_cast<int>(i), probs[i]});
    }

    std::sort(sorted_probs.begin(), sorted_probs.end(), 
              [](const TokenProb& a, const TokenProb& b) { return a.prob > b.prob; });

    float cumulative_prob = 0.0f;
    for (size_t i = 0; i < sorted_probs.size(); i++) {
        cumulative_prob += sorted_probs[i].prob;
        if (cumulative_prob > top_p) {
            for (size_t j = i + 1; j < sorted_probs.size(); j++) {
                probs[sorted_probs[j].id] = 0.0f;
            }
            break; 
        }
    }
}

int sample_dynamic_temperature_top_p(std::vector<float>& logits, float dynatemp_min, float dynatemp_max, float dynatemp_exponent, float top_p) {
    float entropy = calculate_entropy(logits);
    float max_entropy = std::log(static_cast<float>(logits.size()));
    float normalized_entropy = entropy / max_entropy;
    float curved_entropy = std::pow(normalized_entropy, dynatemp_exponent);
    float current_temp = dynatemp_min + (1.0f - curved_entropy) * (dynatemp_max - dynatemp_min);
    
    float max_val = -INFINITY;
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] /= current_temp;
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] /= sum;
    }

    // Apply Top-P masking
    apply_top_p(logits, top_p);

    // Re-normalize probabilities after Top-P masking
    sum = 0.0f;
    for (float p : logits) sum += p;
    for (float& p : logits) p /= sum;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float coin = dis(gen);
    float cumulative_prob = 0.0f;
    
    for (size_t i = 0; i < logits.size(); i++) {
        cumulative_prob += logits[i];
        if (coin < cumulative_prob) {
            return i;
        }
    }
    return logits.size() - 1;
}

int forward(Model& model, int token_id, int current_position, const std::vector<int>& history) {
    int dim = model.config.dim;
    int head_size = dim / model.config.n_heads;
    int kv_dim = model.config.n_kv_heads * head_size;
    int hidden_dim = model.config.hidden_dim;

    std::vector<float> x(dim);
    lookup(model, token_id, x);

    std::vector<float> q(dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);

    for (int layer = 0; layer < model.config.n_layers; layer++) {
        std::vector<float> xb = pre_att_norm(model, x, layer);

        float* wq_layer = model.weights.wq + layer * dim * dim;
        float* wk_layer = model.weights.wk + layer * kv_dim * dim;
        float* wv_layer = model.weights.wv + layer * kv_dim * dim;
        float* wo_layer = model.weights.wo + layer * dim * dim;

        mult_matrix(q, xb, wq_layer, dim, dim);
        mult_matrix(k, xb, wk_layer, kv_dim, dim);
        mult_matrix(v, xb, wv_layer, kv_dim, dim);

        q = apply_rope_q(model, q, current_position);
        k = apply_rope_k(model, k, current_position);

        int cache_row = layer * model.config.seq_len + current_position;
        for (int j = 0; j < kv_dim; j++) {
            model.kv_cache.key_cache[cache_row][j] = k[j];
            model.kv_cache.value_cache[cache_row][j] = v[j];
        }

        std::vector<float> attn_out(dim);
        compute_attention(model, attn_out, q, current_position, layer);

        std::vector<float> wo_out(dim);
        mult_matrix(wo_out, attn_out, wo_layer, dim, dim);

        for (int j = 0; j < dim; j++) {
            x[j] += wo_out[j];
        }

        std::vector<float> xb_ffn = pre_ffn_norm(model, x, layer);

        std::vector<float> hb(hidden_dim);
        std::vector<float> hb2(hidden_dim);

        float* w1_layer = model.weights.w1 + layer * hidden_dim * dim;
        float* w2_layer = model.weights.w2 + layer * dim * hidden_dim;
        float* w3_layer = model.weights.w3 + layer * hidden_dim * dim;

        mult_matrix(hb,  xb_ffn, w1_layer, hidden_dim, dim);
        mult_matrix(hb2, xb_ffn, w3_layer, hidden_dim, dim);

        for (int j = 0; j < hidden_dim; j++) {
            float silu = hb[j] / (1.0f + std::exp(-hb[j]));
            hb[j] = silu * hb2[j];
        }

        std::vector<float> w2_out(dim);
        mult_matrix(w2_out, hb, w2_layer, dim, hidden_dim);

        for (int j = 0; j < dim; j++) {
            x[j] += w2_out[j];
        }
    }
    x = final_norm(model, x);
    
    std::vector<float> logits(model.config.vocab_size);
    mult_matrix(logits, x, model.weights.wcls, model.config.vocab_size, dim);
    // logits[start_id] = -1e30f;   // forbid BOS during generation
    apply_repetition_penalty(logits, history, 1.2f);
    
    // Applying Dynamic Temperature and Top-P (0.9f)
    int id = sample_dynamic_temperature_top_p(logits, 0.6f, 0.8f, 1.0f, 0.9f);
    return id;
}

void generate(Model* model, Tokenizer* tokenizer, const std::string& prompt) {
    std::vector<int> prompt_tokens = tokenizer->encode(prompt);

    
    int prompt_len = prompt_tokens.size();
    int next_token = prompt_tokens[0]; 
    std::vector<int> history;
    
    for (int pos = 0; pos < model->config.seq_len; pos++) {
        history.push_back(next_token);
        
        int predicted = forward(*model, next_token, pos, history);
        
        if (pos < prompt_len - 1) {
            next_token = prompt_tokens[pos + 1];
        } else {
            next_token = predicted;
            
            if (next_token == start_id || next_token == end_id) break; 
            
            std::string word = tokenizer->decode(next_token);
            std::cout << word << std::flush;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file.bin> <tokenizer.bin> \"Prompt text\"" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];
    std::string prompt = argv[3];

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error: Could not open model file " << model_path << std::endl;
        return 1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error: Could not stat model file" << std::endl;
        close(fd);
        return 1;
    }
    size_t file_size = sb.st_size;

    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Error: mmap failed" << std::endl;
        close(fd);
        return 1;
    }
    close(fd);

    Model model;
    
    std::cout << "Loading model..." << std::endl;
    load_model(&model, data, file_size);
    
    std::cout << "--- Model Configuration ---" << std::endl;
    std::cout << "dim: " << model.config.dim << std::endl;
    std::cout << "hidden_dim: " << model.config.hidden_dim << std::endl;
    std::cout << "n_layers: " << model.config.n_layers << std::endl;
    std::cout << "n_heads: " << model.config.n_heads << std::endl;
    std::cout << "n_kv_heads: " << model.config.n_kv_heads << std::endl;
    std::cout << "vocab_size: " << model.config.vocab_size << std::endl;
    std::cout << "seq_len: " << model.config.seq_len << std::endl;

    Tokenizer tokenizer;
    std::cout << "\nLoading tokenizer..." << std::endl;
    if (!tokenizer.load_from_file(tokenizer_path)) {
        std::cerr << "Error: Failed to load tokenizer from " << tokenizer_path << std::endl;
        munmap(data, file_size);
        return 1;
    }

    std::cout << "\n--- Starting Generation ---\n" << std::endl;
    std::cout << prompt;
    
    generate(&model, &tokenizer, prompt);

    munmap(data, file_size);
    return 0;
}
