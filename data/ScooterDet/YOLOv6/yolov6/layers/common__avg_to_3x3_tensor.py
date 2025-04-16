def _avg_to_3x3_tensor(self, avgp):
    channels = self.in_channels
    groups = self.groups
    kernel_size = avgp.kernel_size
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :
        ] = 1.0 / kernel_size ** 2
    return k
