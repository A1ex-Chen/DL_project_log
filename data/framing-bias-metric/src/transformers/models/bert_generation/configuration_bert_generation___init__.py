def __init__(self, vocab_size=50358, hidden_size=1024, num_hidden_layers=24,
    num_attention_heads=16, intermediate_size=4096, hidden_act='gelu',
    hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    max_position_embeddings=512, initializer_range=0.02, layer_norm_eps=
    1e-12, pad_token_id=0, bos_token_id=2, eos_token_id=1,
    gradient_checkpointing=False, position_embedding_type='absolute', **kwargs
    ):
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        eos_token_id=eos_token_id, **kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.gradient_checkpointing = gradient_checkpointing
    self.position_embedding_type = position_embedding_type
