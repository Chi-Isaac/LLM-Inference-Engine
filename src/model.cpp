#include "model.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <cmath>

void build_rpe_cache(Model* model) {
    int head_size = model->config.dim / model->config.n_heads;
    int num_pairs = head_size / 2;

    model->rpe_cache.sin_cache.assign(
        model->config.seq_len, std::vector<float>(num_pairs, 0.0f));
    model->rpe_cache.cos_cache.assign(
        model->config.seq_len, std::vector<float>(num_pairs, 0.0f));

    for (int pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
        float exponent = -2.0f * pair_idx / static_cast<float>(head_size);
        float theta = std::pow(10000.0f, exponent);

        for (int pos = 0; pos < model->config.seq_len; pos++) {
            float angle = pos * theta;
            model->rpe_cache.sin_cache[pos][pair_idx] = std::sin(angle);
            model->rpe_cache.cos_cache[pos][pair_idx] = std::cos(angle);
        }
    }
}
void load_model(Model *model, void *data, size_t file_size) {
    memcpy(&model->config, data, sizeof(struct Config));
    float *ptr = (float *)((char *)data + sizeof(struct Config));

    // 1. Check if weights are shared based on the negative sign
    bool shared_weights = (model->config.vocab_size < 0) ? true : false;

    // 2. Take the absolute value so the math operations use the real number (32000)
    model->config.vocab_size = std::abs(model->config.vocab_size);
    int head_size = model->config.dim / model->config.n_heads;

    // Embeddings
    model->weights.token_embedding_table = ptr;
    ptr += model->config.vocab_size * model->config.dim;

    // Attention RMSNorm
    model->weights.rms_att_weight = ptr;
    ptr += model->config.n_layers * model->config.dim;

    // Attention Query, Key, Value weights
    model->weights.wq = ptr;
    ptr += model->config.n_layers * model->config.dim * (model->config.n_heads * head_size);
    model->weights.wk = ptr;
    ptr += model->config.n_layers * model->config.dim * (model->config.n_kv_heads * head_size);
    model->weights.wv = ptr;
    ptr += model->config.n_layers * model->config.dim * (model->config.n_kv_heads * head_size);

    // Attention Output weights
    model->weights.wo = ptr;
    ptr += model->config.n_layers * (model->config.n_heads * head_size) * model->config.dim;

    // Feed Forward Network RMSNorm
    model->weights.rms_ffn_weight = ptr;
    ptr += model->config.n_layers * model->config.dim;

    // Feed Forward Network (MLP) weights
    model->weights.w1 = ptr;
    ptr += model->config.n_layers * model->config.dim * model->config.hidden_dim;
    model->weights.w2 = ptr;
    ptr += model->config.n_layers * model->config.hidden_dim * model->config.dim;
    model->weights.w3 = ptr;
    ptr += model->config.n_layers * model->config.dim * model->config.hidden_dim;

    // Final RMSNorm
    model->weights.rms_final_weight = ptr;
    ptr += model->config.dim;

    // Calculate how many bytes ptr has advanced from the start of data
    size_t ptr_offset = (char*)ptr - (char*)data;
    
    // The wcls matrix requires vocab_size * dim * sizeof(float) bytes
    size_t wcls_size = model->config.vocab_size * model->config.dim * sizeof(float);
    
    // Check if the file actually has enough space for the full matrix
    if (!shared_weights) {
        model->weights.wcls = ptr;
    } else {
        // Tied weights: point wcls back to the start of the file
        model->weights.wcls = model->weights.token_embedding_table;
    }

    // Instantiate KV cache
    int head_size_local = model->config.dim / model->config.n_heads;
    int kv_dim = model->config.n_kv_heads * head_size_local;
    int num_rows = model->config.n_layers * model->config.seq_len;

    model->kv_cache.key_cache.resize(num_rows, std::vector<float>(kv_dim, 0.0f));
    model->kv_cache.value_cache.resize(num_rows, std::vector<float>(kv_dim, 0.0f));

    build_rpe_cache(model);
}
/*
int main(void) {
    int fd = open("data/stories15M.bin", O_RDONLY);
    if (fd < 0) {
        // handle error
        return 1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        // handle error
        return 1;
    }
    size_t file_size = sb.st_size;

    struct Config config;
    struct TransformerWeights weights;
    
    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        // Handle error
        return 1;
    }
    
    // Close the file descriptor (mmap keeps the mapping active)
    close(fd);
    
    load_model(&config, &weights, data, file_size);
    printf("--- Model Configuration ---\n");
    printf("dim: %d\n", config.dim);
    printf("hidden_dim: %d\n", config.hidden_dim);
    printf("n_layers: %d\n", config.n_layers);
    printf("n_heads: %d\n", config.n_heads);
    printf("n_kv_heads: %d\n", config.n_kv_heads);
    printf("vocab_size: %d\n", config.vocab_size);
    printf("seq_len: %d\n", config.seq_len);

    // Also test that the wcls tied weights logic worked:
    printf("\n--- Weight Pointers ---\n");
    if (weights.wcls == weights.token_embedding_table) {
        printf("Classifier weights are TIED to the embedding table.\n");
    } else {
        printf("Classifier weights are SEPARATE.\n");
    }
    return 0;
}
    */