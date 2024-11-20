def deepview_model_provider():
    enable_flash_attention = False
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=
        dropout, enable_flash_attention=enable_flash_attention)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    return model.to(device)
