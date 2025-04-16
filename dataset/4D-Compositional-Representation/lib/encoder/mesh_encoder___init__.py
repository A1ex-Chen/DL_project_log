def __init__(self, filters_enc, latent_size, sizes, spiral_sizes, spirals,
    D, U, activation='elu', device=None):
    super(SpiralEncoder, self).__init__()
    self.spirals = spirals
    self.filters_enc = filters_enc
    self.spiral_sizes = spiral_sizes
    self.sizes = sizes
    self.D = D
    self.U = U
    self.activation = activation
    self.device = device
    self.conv = []
    input_size = filters_enc[0][0]
    for i in range(len(spiral_sizes) - 1):
        if filters_enc[1][i]:
            self.conv.append(SpiralConv(input_size, spiral_sizes[i],
                filters_enc[1][i], activation=self.activation, device=device))
            input_size = filters_enc[1][i]
        self.conv.append(SpiralConv(input_size, spiral_sizes[i],
            filters_enc[0][i + 1], activation=self.activation, device=device))
        input_size = filters_enc[0][i + 1]
    self.conv = nn.ModuleList(self.conv)
    self.fc_latent_enc = nn.Linear((self.sizes[-1] + 1) * input_size,
        latent_size)
