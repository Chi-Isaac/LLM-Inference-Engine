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
#include <chrono>
#include <cblas.h>
#include <openblas_config.h>

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

// CBLAS mult_matrix implementation
void mult_matrix(std::vector<float>& result, const std::vector<float>& x, const float* w, int rows, int cols) {
    cblas_sgemv(
        CblasRowMajor, // Rows stored contiguously in memory
        CblasNoTrans, // Do not transpose matrix
        rows, // Num rows in w
        cols, // Num cols in w
        1.0f, // Scalar multiplier for product
        w, // Ptr to matrix w
        cols, // Number of columns in w for lda
        x.data(), // Ptr to vector x
        1, // Stride of vector x
        0.0f, // Scalar multiplier for existing contents in result
        result.data(), // Ptr to result vector
        1 // Stride of result vector
    );
    // result = 1.0 * w * x + 0.0 * result
}

// CBLAS mult_matrices implementation
void mult_matrices(std::vector<float>& result_matrix, 
                       const std::vector<float>& input_matrix, 
                       const float* weight_matrix, 
                       int batch_size, 
                       int out_dim, 
                       int in_dim) {
    // result_matrix size: [batch_size, out_dim]
    // input_matrix size: [batch_size, in_dim]
    // weight_matrix size: [out_dim, in_dim] because its transposed in memory

    cblas_sgemm(CblasRowMajor, // Use row-major order
                CblasNoTrans, // Use input_matrix as is
                CblasTrans, // Transpose weight_matrix
                batch_size, // Number of rows in input_matrix
                out_dim, // Number of columns in weight_matrix
                in_dim, // Number of columns in input_matrix and rows in weight_matrix)
                1.0f, // Scalar multiplier for the product
                input_matrix.data(), // Pointer to input_matrix
                in_dim, // Leading dimension of input_matrix (number of columns)
                weight_matrix, // Pointer to weight_matrix
                in_dim, // Leading dimension of weight_matrix (number of columns)
                0.0f, // Scalar multiplier for the existing contents of result_matrix
                result_matrix.data(), // Pointer to result_matrix
                out_dim // Leading dimension of result_matrix (number of columns)
            );
    // result = 1.0 * input_matrix * weight_matrix^T + 0.0 * result_matrix
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
    int window_size = 64;
    int start_idx = std::max(0, static_cast<int>(history.size()) - window_size);

    for (int i = start_idx; i < history.size(); i++) {
        int token_id = history[i];
        
        // Do not penalize EOS, BOS, or special ChatML tokens
        if (token_id == 1 || token_id == 2 || token_id >= 32000) {
            continue; 
        }

        float& logit = logits[token_id];
        if (logit > 0.0f) {
            logit /= penalty;
        } else {
            logit *= penalty;
        }
    }
}

// Calculates how 'certain' the model is about what the next word should be
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
    // Sorts based on an anonymous ordering function
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

int prefill(Model &model, const std::vector<int>& prompt_ids, std::vector<int>& history) {
    int num_ids = prompt_ids.size();
    int dim = model.config.dim;
    int head_size = dim / model.config.n_heads;
    int kv_dim = model.config.n_kv_heads * head_size;
    int hidden_dim = model.config.hidden_dim;
    std::vector<float> X_matrix(num_ids * dim);

    for(int i = 0; i < num_ids; i++) {
        const float *emb_row = model.weights.token_embedding_table + prompt_ids[i] * dim;
        for (int j = 0; j < dim; j++) {
            X_matrix[i * dim + j] = emb_row[j];
        }
        history.push_back(prompt_ids[i]);
    }

    std::vector<float> Q_matrix(num_ids * dim);
    std::vector<float> K_matrix(num_ids * kv_dim);
    std::vector<float> V_matrix(num_ids * kv_dim);
    std::vector<float> Attn_matrix(num_ids * dim);
    std::vector<float> hb_matrix(num_ids * hidden_dim);
    std::vector<float> hb2_matrix(num_ids * hidden_dim);
    for (int layer = 0; layer < model.config.n_layers; layer++) {
        std::vector<float> xb_matrix(num_ids * dim);
        for (int i = 0; i < num_ids; i++) {
            std::vector<float> row(X_matrix.begin() + i * dim, X_matrix.begin() + (i + 1) * dim);
            std::vector<float> norm = pre_att_norm(model, row, layer);
            for (int j = 0; j < dim; j++) {
                xb_matrix[i * dim + j] = norm[j];
            }
        }
        mult_matrices(Q_matrix, xb_matrix, model.weights.wq + layer * dim * dim, num_ids, dim, dim);
        mult_matrices(K_matrix, xb_matrix, model.weights.wk + layer * kv_dim * dim, num_ids, kv_dim, dim);
        mult_matrices(V_matrix, xb_matrix, model.weights.wv + layer * kv_dim * dim, num_ids, kv_dim, dim);
        openblas_set_num_threads(1); 
        #pragma omp parallel for
        for (int i = 0; i < num_ids; i++) {
            std::vector<float> q_row(Q_matrix.begin() + i * dim, Q_matrix.begin() + (i + 1) * dim);
            std::vector<float> k_row(K_matrix.begin() + i * kv_dim, K_matrix.begin() + (i + 1) * kv_dim);

            q_row = apply_rope_q(model, q_row, i);
            k_row = apply_rope_k(model, k_row, i);

            for (int j = 0; j < dim; j++) {
                Q_matrix[i * dim + j] = q_row[j];
            }

            int cache_row = layer * model.config.seq_len + i; 
            for (int j = 0; j < kv_dim; j++) {
                model.kv_cache.key_cache[cache_row][j] = k_row[j];
                model.kv_cache.value_cache[cache_row][j] = V_matrix[i * kv_dim + j];
            }
        }
        openblas_set_num_threads(1); 
        #pragma omp parallel for
        for (int i = 0; i < num_ids; i++) {
            std::vector<float> q_row(Q_matrix.begin() + i * dim, Q_matrix.begin() + (i + 1) * dim);
            std::vector<float> attn_out(dim);
            compute_attention(model, attn_out, q_row, i, layer);
            for (int j = 0; j < dim; j++) {
                Attn_matrix[i * dim + j] = attn_out[j];
            }
        }

        std::vector<float> wo_out_matrix(num_ids * dim);
        mult_matrices(wo_out_matrix, Attn_matrix, model.weights.wo + layer * dim * dim, num_ids, dim, dim);
        openblas_set_num_threads(1); 
        #pragma omp parallel for
        for (int i = 0; i < num_ids; i++) {
            for (int j = 0; j < dim; j++) {
                X_matrix[i * dim + j] += wo_out_matrix[i * dim + j];
            }
        }
        std::vector<float> xb_ffn_matrix(num_ids * dim);
        openblas_set_num_threads(1); 
        #pragma omp parallel for
        for (int i = 0; i < num_ids; i++) {
            std::vector<float> row(X_matrix.begin() + i * dim, X_matrix.begin() + (i + 1) * dim);
            std::vector<float> normed = pre_ffn_norm(model, row, layer);
            for (int j = 0; j < dim; j++) {
                xb_ffn_matrix[i * dim + j] = normed[j];
            }
        }
        mult_matrices(hb_matrix, xb_ffn_matrix, model.weights.w1 + layer * hidden_dim * dim, num_ids, hidden_dim, dim);
        mult_matrices(hb2_matrix, xb_ffn_matrix, model.weights.w3 + layer * hidden_dim * dim, num_ids, hidden_dim, dim);      
        openblas_set_num_threads(1); 
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < num_ids; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                float val = hb_matrix[i * hidden_dim + j];
                float silu = val / (1.0f + std::exp(-val));
                hb_matrix[i * hidden_dim + j] = silu * hb2_matrix[i * hidden_dim + j];
            }
        }
        std::vector<float> w2_out_matrix(num_ids * dim);
        mult_matrices(w2_out_matrix, hb_matrix, model.weights.w2 + layer * dim * hidden_dim, num_ids, dim, hidden_dim);

        // --- Final FFN Residual Add ---
        openblas_set_num_threads(1); 
        #pragma omp parallel for
        for (int i = 0; i < num_ids; i++) {
            for (int j = 0; j < dim; j++) {
                X_matrix[i * dim + j] += w2_out_matrix[i * dim + j];
            }
        }
    }

    // 3. Classifier (We only need the logits for the very last token in the prompt)
    std::vector<float> final_x(dim);
    for (int i = 0; i < dim; i++) {
        final_x[i] = X_matrix[(num_ids - 1) * dim + i]; 
    }
    
    final_x = final_norm(model, final_x);
    
    std::vector<float> logits(model.config.vocab_size);
    mult_matrix(logits, final_x, model.weights.wcls, model.config.vocab_size, dim);
    
    apply_repetition_penalty(logits, history, 1.05f);

    return sample_dynamic_temperature_top_p(logits, 0.6f, 0.8f, 1.0f, 0.9f);
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
    apply_repetition_penalty(logits, history, 1.05f);
    

    int id = sample_dynamic_temperature_top_p(logits, 0.1f, 0.3f, 1.0f, 0.9f);
    // int id = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    return id;    
}

int generate(Model* model, Tokenizer* tokenizer, const std::string& prompt) {
    int generated_tokens = 0;

    std::vector<int> prompt_tokens = tokenizer->encode(prompt);
    
    int prompt_len = prompt_tokens.size(); 
    std::vector<int> history;
    int next_token = prefill(*model, prompt_tokens, history);
    if (next_token == start_id || next_token == end_id) return generated_tokens;
    std::cout << tokenizer->decode(next_token) << std::flush;
    generated_tokens++;
    
    for (int pos = prompt_len; pos < model->config.seq_len; pos++) {
        history.push_back(next_token);
        
        int predicted = forward(*model, next_token, pos, history);
        
        next_token = predicted;
        
        if (next_token == start_id || next_token == end_id) break; 
        
        std::string word = tokenizer->decode(next_token);
        std::cout << word << std::flush;
        generated_tokens++;
    }
    std::cout << std::endl;
    return generated_tokens;
}

int inference(int argc, char** argv, int &generated_tokens) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file.bin> <tokenizer.bin>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    const char* tokenizer_path = argv[2];

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

std::string configuration = "";
std::string userInput = "";
while (true) {
    std::cout << "\n--- What type of assistant would you like? ---\n" << std::endl;
    if (std::getline(std::cin, configuration)) {
        break;
    }
}

while (true) {
    std::cout << "User: ";
    if (!std::getline(std::cin, userInput)) {
        break;
    }
    
    if (userInput == "quit" || userInput == "exit") {
        break;
    }
    if (userInput.empty()) {
        continue;
    }
    userInput = "<|system|>\n" + configuration + "</s>\n<|user|>\n" + userInput + "</s>\n<|assistant|>\n";
    // Add a label so you know the model is speaking
    std::cout << "TinyLlama: ";
    
    // This function should now print tokens with std::flush
    generated_tokens = generate(&model, &tokenizer, userInput);
    
    // Print a newline after generation is completely finished
    // This separates the response from the next User prompt
    std::cout << std::endl << std::endl; 
}
    munmap(data, file_size);
    return 0;
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    int generated_tokens = 0;
    int exit_status = inference(argc, argv, generated_tokens);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\n--- Summary ---\n";
    std::cout << "Total Time (ms): " << elapsed.count() << "\n";
    std::cout << "Number of Tokens Generated: " << generated_tokens << "\n";
    std::cout << "Avg Time per Token (ms): " << (generated_tokens ? elapsed.count() / generated_tokens : 0) << "\n";
    return exit_status;
}