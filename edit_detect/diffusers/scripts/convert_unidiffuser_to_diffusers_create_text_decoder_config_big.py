def create_text_decoder_config_big():
    text_decoder_config = {'prefix_length': 77, 'prefix_inner_dim': 768,
        'prefix_hidden_dim': 64, 'vocab_size': 50258, 'n_positions': 1024,
        'n_embd': 768, 'n_layer': 12, 'n_head': 12, 'n_inner': 3072,
        'activation_function': 'gelu', 'resid_pdrop': 0.1, 'embd_pdrop': 
        0.1, 'attn_pdrop': 0.1, 'layer_norm_epsilon': 1e-05,
        'initializer_range': 0.02}
    return text_decoder_config
