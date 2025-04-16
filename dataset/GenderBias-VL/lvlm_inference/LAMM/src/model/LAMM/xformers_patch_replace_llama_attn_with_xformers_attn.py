def replace_llama_attn_with_xformers_attn():
    LlamaAttention.forward = xformers_forward
