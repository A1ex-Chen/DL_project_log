def __init__(self, langs=['en', 'de'], src_vocab_size=42024, tgt_vocab_size
    =42024, activation_function='relu', d_model=1024, max_length=200,
    max_position_embeddings=1024, encoder_ffn_dim=4096, encoder_layers=12,
    encoder_attention_heads=16, encoder_layerdrop=0.0, decoder_ffn_dim=4096,
    decoder_layers=12, decoder_attention_heads=16, decoder_layerdrop=0.0,
    attention_dropout=0.0, dropout=0.1, activation_dropout=0.0, init_std=
    0.02, decoder_start_token_id=2, is_encoder_decoder=True,
    scale_embedding=True, tie_word_embeddings=False, num_beams=5,
    length_penalty=1.0, early_stopping=False, use_cache=True, pad_token_id=
    1, bos_token_id=0, eos_token_id=2, **common_kwargs):
    if 'hidden_size' in common_kwargs:
        raise ValueError('hidden size is called d_model')
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        eos_token_id=eos_token_id, decoder_start_token_id=
        decoder_start_token_id, is_encoder_decoder=is_encoder_decoder,
        tie_word_embeddings=tie_word_embeddings, **common_kwargs)
    self.langs = langs
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.d_model = d_model
    self.max_length = max_length
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
    self.num_beams = num_beams
    self.length_penalty = length_penalty
    self.early_stopping = early_stopping
    self.decoder = DecoderConfig(vocab_size=tgt_vocab_size, bos_token_id=
        eos_token_id)
    self.scale_embedding = scale_embedding
    self.attention_dropout = attention_dropout
    self.activation_dropout = activation_dropout
    self.dropout = dropout
    self.use_cache = use_cache
