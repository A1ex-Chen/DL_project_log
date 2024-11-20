def __init__(self, input_dim, inner_dim, num_classes, activation_fn,
    pooler_dropout, ft_type):
    super().__init__()
    self.dense = nn.Linear(input_dim, inner_dim)
    self.activation_fn = utils.get_activation_fn(activation_fn)
    self.dropout = nn.Dropout(p=pooler_dropout)
    self.out_proj = nn.Linear(inner_dim, num_classes)
    self.ft_type = ft_type
