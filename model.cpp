#include "model.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

void load_model(struct Config *config, struct TransformerWeights *weights, void *data, size_t file_size) {
    memcpy(config, data, sizeof(struct Config));
    float *ptr = (float *)((char *)data + sizeof(struct Config));
    int head_size = config->dim / config->n_heads;

    // Embeddings
    weights->token_embedding_table = ptr;
    ptr += config->vocab_size * config->dim;

    // Attention RMSNorm
    weights->rms_att_Weight = ptr;
    ptr += config->n_layers * config->dim;

    // Attention Query, Key, Value weights
    weights->wq = ptr;
    ptr += config->n_layers * config->dim * (config->n_heads * head_size);
    weights->wk = ptr;
    ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);
    weights->wv = ptr;
    ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);

    // Attention Output weights
    weights->wo = ptr;
    ptr += config->n_layers * (config->n_heads * head_size) * config->dim;

    // Feed Forward Network RMSNorm
    weights->rms_ffn_weight = ptr;
    ptr += config->n_layers * config->dim;

    // Feed Forward Network (MLP) weights
    weights->w1 = ptr;
    ptr += config->n_layers * config->dim * config->hidden_dim;
    weights->w2 = ptr;
    ptr += config->n_layers * config->hidden_dim * config->dim;
    weights->w3 = ptr;
    ptr += config->n_layers * config->dim * config->hidden_dim;

    // Final RMSNorm
    weights->rms_final_weight = ptr;
    ptr += config->dim;

    // Calculate how many bytes ptr has advanced from the start of data
    size_t ptr_offset = (char*)ptr - (char*)data;
    
    // If ptr hasn't hit the end, wcls is at ptr. If it has, wcls is tied to embeddings.
    // We add a small buffer (e.g. sizeof(float)) to account for potential padding
    if (ptr_offset + sizeof(float) <= file_size) {
        weights->wcls = ptr;
    } else {
        weights->wcls = weights->token_embedding_table;
    }
}

int main(void) {
    int fd = open("stories15M.bin", O_RDONLY);
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

    return 0;
}