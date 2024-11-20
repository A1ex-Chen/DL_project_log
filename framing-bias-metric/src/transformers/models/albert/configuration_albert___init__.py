def __init__(self, vocab_size=30000, embedding_size=128, hidden_size=4096,
    num_hidden_layers=12, num_hidden_groups=1, num_attention_heads=64,
    intermediate_size=16384, inner_group_num=1, hidden_act='gelu_new',
    hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
    layer_norm_eps=1e-12, classifier_dropout_prob=0.1,
    position_embedding_type='absolute', pad_token_id=0, bos_token_id=2,
    eos_token_id=3, **kwargs):
    super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
        eos_token_id=eos_token_id, **kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_groups = num_hidden_groups
    self.num_attention_heads = num_attention_heads
    self.inner_group_num = inner_group_num
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.classifier_dropout_prob = classifier_dropout_prob
    self.position_embedding_type = position_embedding_type
