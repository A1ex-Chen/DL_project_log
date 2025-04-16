def __init__(self, activation_dropout=0.0, extra_pos_embeddings=2,
    activation_function='gelu', vocab_size=50265, d_model=1024,
    encoder_ffn_dim=4096, encoder_layers=12, encoder_attention_heads=16,
    decoder_ffn_dim=4096, decoder_layers=12, decoder_attention_heads=16,
    encoder_layerdrop=0.0, decoder_layerdrop=0.0, attention_dropout=0.0,
    dropout=0.1, max_position_embeddings=1024, init_std=0.02,
    classifier_dropout=0.0, num_labels=3, is_encoder_decoder=True,
    normalize_before=False, add_final_layer_norm=False,
    do_blenderbot_90_layernorm=False, scale_embedding=False,
    normalize_embedding=True, static_position_embeddings=False,
    add_bias_logits=False, force_bos_token_to_be_generated=False, use_cache
    =True, pad_token_id=1, bos_token_id=0, eos_token_id=2, **common_kwargs):
    """
        :class:`~transformers.BartConfig` is the configuration class for `BartModel`.

        Examples::

            >>> from transformers import BartConfig, BartModel

            >>> config = BartConfig.from_pretrained('facebook/bart-large')
            >>> model = BartModel(config)

        """
    if 'hidden_size' in common_kwargs:
        raise ValueError('hidden size is called d_model')
    super().__init__(num_labels=num_labels, pad_token_id=pad_token_id,
        bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        is_encoder_decoder=is_encoder_decoder, **common_kwargs)
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.encoder_ffn_dim = encoder_ffn_dim
    self.encoder_layers = self.num_hidden_layers = encoder_layers
    self.encoder_attention_heads = encoder_attention_heads
    self.encoder_layerdrop = encoder_layerdrop
    self.decoder_layerdrop = decoder_layerdrop
    self.decoder_ffn_dim = decoder_ffn_dim
    self.decoder_layers = decoder_layers
    self.decoder_attention_heads = decoder_attention_heads
    self.max_position_embeddings = max_position_embeddings
    self.init_std = init_std
    self.activation_function = activation_function
    self.scale_embedding = scale_embedding
    self.normalize_embedding = normalize_embedding
    self.normalize_before = normalize_before
    self.add_final_layer_norm = add_final_layer_norm
    self.add_bias_logits = add_bias_logits
    self.static_position_embeddings = static_position_embeddings
    self.attention_dropout = attention_dropout
    self.activation_dropout = activation_dropout
    self.dropout = dropout
    self.classifier_dropout = classifier_dropout
    self.extra_pos_embeddings = extra_pos_embeddings
    self.force_bos_token_to_be_generated = force_bos_token_to_be_generated
    self.do_blenderbot_90_layernorm = do_blenderbot_90_layernorm
    self.use_cache = use_cache
