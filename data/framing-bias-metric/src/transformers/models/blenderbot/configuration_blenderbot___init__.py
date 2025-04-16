def __init__(self, activation_dropout=0.0, extra_pos_embeddings=0,
    activation_function='gelu', vocab_size=54944, d_model=512,
    encoder_ffn_dim=2048, encoder_layers=8, encoder_attention_heads=16,
    decoder_ffn_dim=2048, decoder_layers=8, decoder_attention_heads=16,
    encoder_layerdrop=0.0, decoder_layerdrop=0.0, attention_dropout=0.0,
    dropout=0.1, max_position_embeddings=512, classifier_dropout=0.0,
    is_encoder_decoder=True, pad_token_id=1, bos_token_id=0, eos_token_id=2,
    normalize_before=False, add_final_layer_norm=False,
    do_blenderbot_90_layernorm=True, scale_embedding=False,
    normalize_embedding=True, static_position_embeddings=False,
    add_bias_logits=False, force_bos_token_to_be_generated=False, **
    common_kwargs):
    """
        Examples::

            >>> from transformers import BlenderbotConfig
            >>> config = BlenderbotConfig.from_pretrained('facebook/blenderbot-90M')

        """
    if 'hidden_size' in common_kwargs:
        raise ValueError('hidden size is called d_model')
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder,
        vocab_size=vocab_size, d_model=d_model, encoder_ffn_dim=
        encoder_ffn_dim, encoder_layers=encoder_layers, encoder_layerdrop=
        encoder_layerdrop, encoder_attention_heads=encoder_attention_heads,
        decoder_layerdrop=decoder_layerdrop, decoder_ffn_dim=
        decoder_ffn_dim, decoder_layers=decoder_layers, normalize_before=
        normalize_before, normalize_embedding=normalize_embedding,
        static_position_embeddings=static_position_embeddings,
        add_bias_logits=add_bias_logits, force_bos_token_to_be_generated=
        force_bos_token_to_be_generated, do_blenderbot_90_layernorm=
        do_blenderbot_90_layernorm, add_final_layer_norm=
        add_final_layer_norm, scale_embedding=scale_embedding,
        attention_dropout=attention_dropout, dropout=dropout,
        classifier_dropout=classifier_dropout, activation_dropout=
        activation_dropout, max_position_embeddings=max_position_embeddings,
        extra_pos_embeddings=extra_pos_embeddings, activation_function=
        activation_function, decoder_attention_heads=
        decoder_attention_heads, **common_kwargs)
