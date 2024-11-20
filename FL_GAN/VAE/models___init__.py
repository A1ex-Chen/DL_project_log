def __init__(self, dataset, device):
    self.device = device
    assert dataset in ['mnist', 'fashion-mnist', 'cifar', 'stl']
    super().__init__()
    self.n_latent_features = 128
    if dataset in ['mnist', 'fashion-mnist']:
        pooling_kernel = [2, 2]
        encoder_output_size = 7
    elif dataset == 'cifar':
        pooling_kernel = [4, 2]
        encoder_output_size = 4
    elif dataset == 'stl':
        pooling_kernel = [4, 4]
        encoder_output_size = 6
    if dataset in ['mnist', 'fashion-mnist']:
        color_channels = 1
    else:
        color_channels = 3
    n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size
    self.encoder = Encoder(color_channels, pooling_kernel,
        n_neurons_middle_layer)
    self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
    self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
    self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
    self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)
