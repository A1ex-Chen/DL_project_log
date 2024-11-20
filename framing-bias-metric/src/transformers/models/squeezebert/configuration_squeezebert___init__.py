def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12,
    num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
    hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
    layer_norm_eps=1e-12, pad_token_id=0, embedding_size=768, q_groups=4,
    k_groups=4, v_groups=4, post_attention_groups=1, intermediate_groups=4,
    output_groups=4, **kwargs):
    super().__init__(pad_token_id=pad_token_id, **kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.embedding_size = embedding_size
    self.q_groups = q_groups
    self.k_groups = k_groups
    self.v_groups = v_groups
    self.post_attention_groups = post_attention_groups
    self.intermediate_groups = intermediate_groups
    self.output_groups = output_groups
