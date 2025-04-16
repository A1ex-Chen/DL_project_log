def replace_llama_attn_with_flash_attn():
    (transformers.models.llama.modeling_llama.LlamaModel.
        _prepare_decoder_attention_mask) = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
