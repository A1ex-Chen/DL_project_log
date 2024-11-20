def replace_llama_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            'Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward.ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593'
            )
    LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask)
    LlamaAttention.forward = llama_flash_attn_forward
