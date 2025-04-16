def create_text_decoder_config_test():
    text_decoder_config = {'prefix_length': 77, 'prefix_inner_dim': 32,
        'prefix_hidden_dim': 32, 'vocab_size': 1025, 'n_positions': 1024,
        'n_embd': 32, 'n_layer': 5, 'n_head': 4, 'n_inner': 37,
        'activation_function': 'gelu', 'resid_pdrop': 0.1, 'embd_pdrop': 
        0.1, 'attn_pdrop': 0.1, 'layer_norm_epsilon': 1e-05,
        'initializer_range': 0.02}
    return text_decoder_config
