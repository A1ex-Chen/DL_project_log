def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,),
    scale=1.0):
    super().__init__()
    if not hasattr(F, 'scaled_dot_product_attention'):
        raise ImportError(
            f'{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.'
            )
    self.hidden_size = hidden_size
    self.cross_attention_dim = cross_attention_dim
    if not isinstance(num_tokens, (tuple, list)):
        num_tokens = [num_tokens]
    self.num_tokens = num_tokens
    if not isinstance(scale, list):
        scale = [scale] * len(num_tokens)
    if len(scale) != len(num_tokens):
        raise ValueError(
            '`scale` should be a list of integers with the same length as `num_tokens`.'
            )
    self.scale = scale
    self.to_k_ip = nn.ModuleList([nn.Linear(cross_attention_dim,
        hidden_size, bias=False) for _ in range(len(num_tokens))])
    self.to_v_ip = nn.ModuleList([nn.Linear(cross_attention_dim,
        hidden_size, bias=False) for _ in range(len(num_tokens))])
