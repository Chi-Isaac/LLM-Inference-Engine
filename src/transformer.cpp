#include "model.hpp"
#include "transformer.hpp"
#include <vector>
#include <cmath>


// Takes a loaded model, a list of ids and the position of the current id then finds the corresponding vector in the embedded lookup table and loads its values into a predefined vector x
void lookup(const Model& model, int position, const std::vector<int>& ids, std::vector<float>& x) {
    int row = ids[position];
    for (int i = 0; i < model.config.dim; i++) {
        x[i] = model.weights.token_embedding_table[row * model.config.dim + i]; 
    }
}


std::vector<float> apply_rpe(const Model& model, const std::vector<float>& orig, int current_position) {
    std::vector<float> rotated(model.config.dim);
    for (int pair_idx = 0; pair_idx < model.config.dim / 2; pair_idx++) {
        int i = 2 * pair_idx;
        float sin = model.rpe_cache.sin_cache[current_position][pair_idx];
        float cos = model.rpe_cache.cos_cache[current_position][pair_idx];
        rotated[i] = orig[i] * cos - orig[i+1] * sin;
        rotated[i+1] = orig[i] * sin + orig[i+1] * cos;     
    }
    return rotated; 
}

std::vector<float> rms_norm(const Model& model, const std::vector<float>& vector, int curr_layer, float *norm_weights) {
    float epsilon = 1e-5f;
    float sq_sum = 0;
    for (int i = 0; i < model.config.dim; i++) {
        sq_sum += vector[i] * vector[i];
    }
    float rms = std::sqrt(sq_sum / model.config.dim + epsilon);
    float *gamma = norm_weights + curr_layer * model.config.dim;
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

std::vector<float> final_norm(const Model& model, const std::vector<float>& vector, int curr_layer) {
    return rms_norm(model, vector, curr_layer, model.weights.rms_final_weight);
}

void mult_matrix(std::vector<float>& result, const std::vector<float>& x, const float *w, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float value = 0.0f;
        for (int j = 0; j < cols; j++) {
            value += x[j] * w[i * cols + j];
        }
        result[i] = value;
    }
}

void forward(const Model& model, const std::vector<int>& ids, int current_position) {
    std::vector<float> x(model.config.dim);
    lookup(model, current_position, ids, x);

    int dim = model.config.dim;
    int head_size = dim / model.config.n_heads;

    // Allocating QKV projection vectors
    std::vector<float> q(dim);
    std::vector<float> k(model.config.n_kv_heads * head_size);
    std::vector<float> v(model.config.n_kv_heads * head_size);

    for (int i = 0; i < model.config.n_layers; i++) {
        // perform logic for a single layer
    }
}