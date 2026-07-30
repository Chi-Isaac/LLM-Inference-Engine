import torch
import struct
from transformers import AutoModelForCausalLM

def permute(w, n_heads):
    dim1, dim2 = w.shape
    head_dim = dim1 // n_heads
    return w.view(n_heads, 2, head_dim // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def export_tinyllama(hf_model_id, filepath):
    print(f"Downloading and loading {hf_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype=torch.float32)
    hf_config = model.config

    with open(filepath, 'wb') as f:
        tied_weights = False
        if model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
            tied_weights = True

        vocab_size = hf_config.vocab_size
        if tied_weights: vocab_size = -vocab_size

        header = struct.pack('iiiiiii',
            hf_config.hidden_size, hf_config.intermediate_size, hf_config.num_hidden_layers,
            hf_config.num_attention_heads, hf_config.num_key_value_heads, vocab_size, hf_config.max_position_embeddings
        )
        f.write(header)

        def write_tensor(t):
            t_fp32 = t.detach().cpu().to(torch.float32).contiguous()
            f.write(t_fp32.numpy().tobytes())

        print("Writing tensors...")
        write_tensor(model.model.embed_tokens.weight)

        for layer in model.model.layers: write_tensor(layer.input_layernorm.weight)

        for layer in model.model.layers:
            write_tensor(permute(layer.self_attn.q_proj.weight, hf_config.num_attention_heads))
        for layer in model.model.layers:
            write_tensor(permute(layer.self_attn.k_proj.weight, hf_config.num_key_value_heads))

        for layer in model.model.layers: write_tensor(layer.self_attn.v_proj.weight)
        for layer in model.model.layers: write_tensor(layer.self_attn.o_proj.weight)

        for layer in model.model.layers: write_tensor(layer.post_attention_layernorm.weight)

        for layer in model.model.layers: write_tensor(layer.mlp.gate_proj.weight)
        for layer in model.model.layers: write_tensor(layer.mlp.down_proj.weight)
        for layer in model.model.layers: write_tensor(layer.mlp.up_proj.weight)
            
        write_tensor(model.model.norm.weight)
        if not tied_weights: write_tensor(model.lm_head.weight)

    print("Success! Exported fixed weights.")

if __name__ == "__main__":
    export_tinyllama("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_chat.bin")