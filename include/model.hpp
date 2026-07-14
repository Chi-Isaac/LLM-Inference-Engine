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
    float *rms_att_Weight;
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

typedef struct Model {
    struct Config config;
    struct TransformerWeights weights;
} Model;