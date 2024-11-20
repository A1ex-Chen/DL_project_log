def __init__(self, n_channels_in, reduction_ratio):
    super(ChannelAttention, self).__init__()
    self.n_channels_in = n_channels_in
    self.reduction_ratio = reduction_ratio
    self.middle_layer_size = int(self.n_channels_in / float(self.
        reduction_ratio))
    self.bottleneck = nn.Sequential(nn.Linear(self.n_channels_in, self.
        middle_layer_size), nn.ReLU(), nn.Linear(self.middle_layer_size,
        self.n_channels_in))
