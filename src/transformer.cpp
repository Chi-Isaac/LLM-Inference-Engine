#include "model.hpp"
#include "transformer.hpp"
#include <vector>

// Takes a loaded model, a list of ids and the position of the current id then finds the corresponding vector in the embedded lookup table and loads its values into a predefined vector x
void lookup(const Model& model, int position, const std::vector<int>& ids, std::vector<float>& x) {
    int row = ids[position];
    for (int i = 0; i < model.config.dim; i++) {
        x[i] = model.weights.token_embedding_table[row * model.config.dim + i]; 
    }
}

void forward(const Model& model, const std::vector<int>& ids, int current_position) {
    std::vector<float> x(model.config.dim);
    lookup(model, current_position, ids, x);
    // perform rotary positional embeddings
    for (int i = 0; i < model.config.n_layers; i++) {
        // perform logic for a single layer
    }

}