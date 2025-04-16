def __init__(self, input_dim, hidden_dims, output_dim, norm_fn_name=None,
    activation='relu', use_conv=False, dropout=None, hidden_use_bias=False,
    output_use_bias=True, output_use_activation=False, output_use_norm=
    False, weight_init_name=None):
    super().__init__()
    activation = ACTIVATION_DICT[activation]
    norm = None
    if norm_fn_name is not None:
        norm = NORM_DICT[norm_fn_name]
    if norm_fn_name == 'ln' and use_conv:
        norm = lambda x: nn.GroupNorm(1, x)
    if dropout is not None:
        if not isinstance(dropout, list):
            dropout = [dropout for _ in range(len(hidden_dims))]
    layers = []
    prev_dim = input_dim
    for idx, x in enumerate(hidden_dims):
        if use_conv:
            layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
        else:
            layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
        layers.append(layer)
        if norm:
            layers.append(norm(x))
        layers.append(activation())
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout[idx]))
        prev_dim = x
    if use_conv:
        layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
    else:
        layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
    layers.append(layer)
    if output_use_norm:
        layers.append(norm(output_dim))
    if output_use_activation:
        layers.append(activation())
    self.layers = nn.Sequential(*layers)
    if weight_init_name is not None:
        self.do_weight_init(weight_init_name)
