def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
    """Initialize the DeformableTransformerDecoder with the given parameters."""
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
