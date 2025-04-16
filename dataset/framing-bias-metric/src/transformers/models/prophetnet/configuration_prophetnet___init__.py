def __init__(self, activation_dropout=0.1, activation_function='gelu',
    vocab_size=30522, hidden_size=1024, encoder_ffn_dim=4096,
    num_encoder_layers=12, num_encoder_attention_heads=16, decoder_ffn_dim=
    4096, num_decoder_layers=12, num_decoder_attention_heads=16,
    attention_dropout=0.1, dropout=0.1, max_position_embeddings=512,
    init_std=0.02, is_encoder_decoder=True, add_cross_attention=True,
    decoder_start_token_id=0, ngram=2, num_buckets=32,
    relative_max_distance=128, disable_ngram_loss=False, eps=0.0, use_cache
    =True, pad_token_id=0, bos_token_id=1, eos_token_id=2, **kwargs):
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder,
        add_cross_attention=add_cross_attention, decoder_start_token_id=
        decoder_start_token_id, **kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.encoder_ffn_dim = encoder_ffn_dim
    self.num_encoder_layers = num_encoder_layers
    self.num_encoder_attention_heads = num_encoder_attention_heads
    self.decoder_ffn_dim = decoder_ffn_dim
    self.num_decoder_layers = num_decoder_layers
    self.num_decoder_attention_heads = num_decoder_attention_heads
    self.max_position_embeddings = max_position_embeddings
    self.init_std = init_std
    self.activation_function = activation_function
    self.ngram = ngram
    self.num_buckets = num_buckets
    self.relative_max_distance = relative_max_distance
    self.disable_ngram_loss = disable_ngram_loss
    self.eps = eps
    self.attention_dropout = attention_dropout
    self.activation_dropout = activation_dropout
    self.dropout = dropout
    self.use_cache = use_cache
