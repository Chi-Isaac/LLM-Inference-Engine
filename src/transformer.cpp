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

void forward(const Model& model, const std::vector<int>& ids, int current_position) {
    std::vector<float> x(model.config.dim);
    lookup(model, current_position, ids, x);
    for (int i = 0; i < model.config.n_layers; i++) {
        // perform logic for a single layer
    }

}