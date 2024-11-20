def __init__(self, vocab_size=250112, d_model=512, d_kv=64, d_ff=1024,
    num_layers=8, num_decoder_layers=None, num_heads=6,
    relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon
    =1e-06, initializer_factor=1.0, feed_forward_proj='gated-gelu',
    is_encoder_decoder=True, use_cache=True, tokenizer_class='T5Tokenizer',
    tie_word_embeddings=False, pad_token_id=0, eos_token_id=1,
    decoder_start_token_id=0, **kwargs):
    super().__init__(is_encoder_decoder=is_encoder_decoder, tokenizer_class
        =tokenizer_class, tie_word_embeddings=tie_word_embeddings,
        pad_token_id=pad_token_id, eos_token_id=eos_token_id,
        decoder_start_token_id=decoder_start_token_id, **kwargs)
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.d_kv = d_kv
    self.d_ff = d_ff
    self.num_layers = num_layers
    self.num_decoder_layers = (num_decoder_layers if num_decoder_layers is not
        None else self.num_layers)
    self.num_heads = num_heads
    self.relative_attention_num_buckets = relative_attention_num_buckets
    self.dropout_rate = dropout_rate
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_factor = initializer_factor
    self.feed_forward_proj = feed_forward_proj
    self.use_cache = use_cache
