#include <vector>

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct TransformerWeights {
    float *token_embedding_table;
    float *rms_att_weight;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *rms_ffn_weight;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weight;
    float *wcls;
};

struct RpeCache {
    std::vector<std::vector<float>> sin_cache;
    std::vector<std::vector<float>> cos_cache;
};

struct KVCache {
    std::vector<std::vector<float>> key_cache;
    std::vector<std::vector<float>> value_cache;
};

typedef struct {
    struct Config config;
    struct TransformerWeights weights;
    struct RpeCache rpe_cache;
    struct KVCache kv_cache;
} Model;
