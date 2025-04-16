def __init__(self, vocab_size=30522, block_sizes=[4, 4, 4], block_repeats=
    None, num_decoder_layers=2, d_model=768, n_head=12, d_head=64, d_inner=
    3072, hidden_act='gelu_new', hidden_dropout=0.1, attention_dropout=0.1,
    activation_dropout=0.0, max_position_embeddings=512, type_vocab_size=3,
    initializer_range=0.1, initializer_std=None, layer_norm_eps=1e-09,
    pooling_type='mean', attention_type='relative_shift', separate_cls=True,
    truncate_seq=True, pool_q_only=True, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.block_sizes = block_sizes
    self.block_repeats = [1] * len(block_sizes
        ) if block_repeats is None else block_repeats
    assert len(block_sizes) == len(self.block_repeats
        ), '`block_sizes` and `block_repeats` should have the same length.'
    self.num_decoder_layers = num_decoder_layers
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.d_inner = d_inner
    self.hidden_act = hidden_act
    self.hidden_dropout = hidden_dropout
    self.attention_dropout = attention_dropout
    self.activation_dropout = activation_dropout
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.initializer_std = initializer_std
    self.layer_norm_eps = layer_norm_eps
    assert pooling_type in ['mean', 'max'
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
    self.pooling_type = pooling_type
    assert attention_type in ['relative_shift', 'factorized'
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
    self.attention_type = attention_type
    self.separate_cls = separate_cls
    self.truncate_seq = truncate_seq
    self.pool_q_only = pool_q_only
