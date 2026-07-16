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
// Load token embedding for ids[position] into x
void lookup(const Model& model, int id, std::vector<float>& x) {
    for (int i = 0; i < model.config.dim; i++) {
        x[i] = model.weights.token_embedding_table[id * model.config.dim + i];
    }
}

// Apply RoPE to any even-length vector using its actual size
std::vector<float> apply_rpe(const Model& model, const std::vector<float>& orig, int current_position) {
    std::vector<float> rotated(orig.size());

    int num_pairs = static_cast<int>(orig.size()) / 2;
    for (int pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
        int i = 2 * pair_idx;
        float sinv = model.rpe_cache.sin_cache[current_position][pair_idx];
        float cosv = model.rpe_cache.cos_cache[current_position][pair_idx];

        rotated[i]     = orig[i] * cosv - orig[i + 1] * sinv;
        rotated[i + 1] = orig[i] * sinv + orig[i + 1] * cosv;
    }

    return rotated;
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

void softmax(std::vector<float>& x, int length) {
    float max_score = x[0];
    for (int i = 1; i < length; i++) {
        if (x[i] > max_score) {
            max_score = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        x[i] = std::exp(x[i] - max_score);
        sum += x[i];
    }

    for (int i = 0; i < length; i++) {
        x[i] /= sum;
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

        softmax(attn_scores, curr_pos + 1);

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

int sample(std::vector<float> v) {
    return std::max_element(v.begin(), v.end()) - v.begin();
}

int forward(Model& model, int token_id, int current_position) {
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

        q = apply_rpe(model, q, current_position);
        k = apply_rpe(model, k, current_position);

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
    int id = sample(logits);
    return id;
}

void generate(Model* model, Tokenizer* tokenizer, const std::string& prompt) {
    std::vector<int> prompt_tokens = tokenizer->encode(prompt);
    int prompt_len = prompt_tokens.size();
    
    // (Optional but recommended) LLaMA often expects a BOS token (ID 1) at the very start
    // prompt_tokens.insert(prompt_tokens.begin(), 1); 
    // prompt_len++;

    int next_token = prompt_tokens[0]; 
    
    for (int pos = 0; pos < model->config.seq_len; pos++) {
        
        int predicted = forward(*model, next_token, pos);
        
        if (pos < prompt_len - 1) {
            // Still reading the prompt
            next_token = prompt_tokens[pos + 1];
        } else {
            // Generating new tokens
            next_token = predicted;
            
            // --- EOS CHECK ---
            // If the model generates the End-Of-Sequence token (usually ID 2), stop generating!
            if (next_token == 2) {
                break; 
            }
            
            // Decode and print the generated token
            std::string word = tokenizer->decode(next_token);
            std::cout << word << std::flush;
        }
    }
    std::cout << std::endl; // Print a final newline when generation finishes
}

#include <iostream>
#include <stdexcept>

// Ensure Tokenizer is included so we can instantiate it
// #include "tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file.bin> <tokenizer.bin> \"Prompt text\"" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];
    std::string prompt = argv[3];

    // 1. Open the model file
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

    // 2. Memory-map the weights file directly into RAM
    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Error: mmap failed" << std::endl;
        close(fd);
        return 1;
    }
    
    // We can close the file descriptor safely; mmap keeps the mapping active
    close(fd);

    Model model;
    
    std::cout << "Loading model..." << std::endl;
    // 3. Point our structs to the memory-mapped data
    load_model(&model, data, file_size);
    
    std::cout << "--- Model Configuration ---" << std::endl;
    std::cout << "dim: " << model.config.dim << std::endl;
    std::cout << "hidden_dim: " << model.config.hidden_dim << std::endl;
    std::cout << "n_layers: " << model.config.n_layers << std::endl;
    std::cout << "n_heads: " << model.config.n_heads << std::endl;
    std::cout << "n_kv_heads: " << model.config.n_kv_heads << std::endl;
    std::cout << "vocab_size: " << model.config.vocab_size << std::endl;
    std::cout << "seq_len: " << model.config.seq_len << std::endl;

    // 4. Load the Tokenizer
    Tokenizer tokenizer;
    std::cout << "\nLoading tokenizer..." << std::endl;
    if (!tokenizer.load_from_file(tokenizer_path)) {
        std::cerr << "Error: Failed to load tokenizer from " << tokenizer_path << std::endl;
        munmap(data, file_size); // Cleanup
        return 1;
    }

    // 5. Generate Text
    std::cout << "\n--- Starting Generation ---\n" << std::endl;
    std::cout << prompt; // Print prompt so it flows visually into generation
    
    generate(&model, &tokenizer, prompt);

    // 6. Cleanup memory map
    munmap(data, file_size);
    return 0;
}
