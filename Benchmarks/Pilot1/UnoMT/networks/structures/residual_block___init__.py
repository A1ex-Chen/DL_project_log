def __init__(self, layer_dim: int, num_layers: int, dropout: float):
    super(ResBlock, self).__init__()
    self.block = nn.Sequential()
    for i in range(num_layers):
        self.block.add_module('res_dense_%d' % i, nn.Linear(layer_dim,
            layer_dim))
        if dropout > 0.0:
            self.block.add_module('res_dropout_%d' % i, nn.Dropout(dropout))
        if i != num_layers - 1:
            self.block.add_module('res_relu_%d' % i, nn.ReLU())
    self.activation = nn.ReLU()
    self.block.apply(basic_weight_init)
