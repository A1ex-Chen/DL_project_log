def __init__(self, in_channels: int, embedding_dim: int, num_embeddings:
    int, hidden_dims: List=None, beta: float=0.25, img_size: int=64, **kwargs
    ) ->None:
    super(VQVAE, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.img_size = img_size
    self.beta = beta
    self.out_channels = in_channels
    modules = []
    if hidden_dims is None:
        hidden_dims = [128, 256]
    for h_dim in hidden_dims:
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=
            h_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU()))
        in_channels = h_dim
    modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels,
        kernel_size=3, stride=1, padding=1), nn.LeakyReLU()))
    for _ in range(6):
        modules.append(ResidualLayer(in_channels, in_channels))
    modules.append(nn.LeakyReLU())
    modules.append(nn.Sequential(nn.Conv2d(in_channels, embedding_dim,
        kernel_size=1, stride=1), nn.LeakyReLU()))
    self.encoder = nn.Sequential(*modules)
    self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
    modules = []
    modules.append(nn.Sequential(nn.Conv2d(embedding_dim, hidden_dims[-1],
        kernel_size=3, stride=1, padding=1), nn.LeakyReLU()))
    for _ in range(6):
        modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
    modules.append(nn.LeakyReLU())
    hidden_dims.reverse()
    for i in range(len(hidden_dims) - 1):
        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i],
            hidden_dims[i + 1], kernel_size=4, stride=2, padding=1), nn.
            LeakyReLU()))
    modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1],
        out_channels=self.out_channels, kernel_size=4, stride=2, padding=1),
        nn.Tanh()))
    self.decoder = nn.Sequential(*modules)
