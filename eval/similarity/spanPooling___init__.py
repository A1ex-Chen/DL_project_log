def __init__(self, word_embedding_dimension: int, pooling_mode_cls_token:
    bool=False, pooling_mode_max_tokens: bool=False,
    pooling_mode_mean_tokens: bool=True, pooling_mode_mean_sqrt_len_tokens:
    bool=False, pooling_mode_span: bool=False):
    super(spanPooling, self).__init__()
    self.config_keys = ['word_embedding_dimension',
        'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
        'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']
    self.word_embedding_dimension = word_embedding_dimension
    self.pooling_mode_cls_token = pooling_mode_cls_token
    self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
    self.pooling_mode_max_tokens = pooling_mode_max_tokens
    self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
    self.pooling_mode_span = pooling_mode_span
    pooling_mode_multiplier = sum([pooling_mode_cls_token,
        pooling_mode_max_tokens, pooling_mode_mean_tokens,
        pooling_mode_mean_sqrt_len_tokens])
    self.pooling_output_dimension = (pooling_mode_multiplier *
        word_embedding_dimension)
