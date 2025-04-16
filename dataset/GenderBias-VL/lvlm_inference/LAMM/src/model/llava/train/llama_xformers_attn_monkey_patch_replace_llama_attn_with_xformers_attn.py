def replace_llama_attn_with_xformers_attn():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = (
        xformers_forward)
