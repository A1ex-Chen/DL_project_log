def __init__(self, config):
    super().__init__()
    num_layers = (config.num_hidden_layers + 1) // 5
    hid_size = config.hidden_size
    num_heads = 1
    self.blocks = nn.ModuleList([BasicTransformerBlock(hid_size, num_heads,
        hid_size, activation_fn='gelu', attention_bias=True) for _ in range
        (num_layers)])
