def _make_conv(self, weights, biases):
    nets = []
    for i in range(len(weights)):
        in_channel = weights[i].shape[0]
        out_channel = weights[i].shape[1]
        k_size = weights[i].shape[2]
        filter = torch.nn.Conv2d(in_channel, out_channel, k_size, 1,
            padding=k_size // 2)
        filter.weight.data = weights[i]
        filter.bias.data = biases[i]
        nets.append(filter)
        if i != len(weights) - 1:
            nets.append(torch.nn.ReLU())
    return torch.nn.Sequential(*nets)
