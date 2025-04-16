def __init__(self, hidden_size, eps=config.layer_norm_eps):
    """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
    super(BertLayerNorm, self).__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.bias = nn.Parameter(torch.zeros(hidden_size))
    self.variance_epsilon = 1e-05
