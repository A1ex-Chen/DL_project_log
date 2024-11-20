def __init__(self, input_dim: int, layer_dim: int, num_layers: int,
    latent_dim: int, autoencoder: bool=True):
    super(EncNet, self).__init__()
    self.encoder = nn.Sequential()
    prev_dim = input_dim
    for i in range(num_layers):
        self.encoder.add_module('dense_%d' % i, nn.Linear(prev_dim, layer_dim))
        prev_dim = layer_dim
        self.encoder.add_module('relu_%d' % i, nn.ReLU())
    self.encoder.add_module('dense_%d' % num_layers, nn.Linear(prev_dim,
        latent_dim))
    if autoencoder:
        self.decoder = nn.Sequential()
        prev_dim = latent_dim
        for i in range(num_layers):
            self.decoder.add_module('dense_%d' % i, nn.Linear(prev_dim,
                layer_dim))
            prev_dim = layer_dim
            self.decoder.add_module('relu_%d' % i, nn.ReLU())
        self.decoder.add_module('dense_%d' % num_layers, nn.Linear(prev_dim,
            input_dim))
    else:
        self.decoder = None
    self.encoder.apply(basic_weight_init)
    if self.decoder is not None:
        self.decoder.apply(basic_weight_init)
