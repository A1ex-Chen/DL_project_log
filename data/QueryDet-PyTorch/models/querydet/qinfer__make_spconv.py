def _make_spconv(self, weights, biases):
    nets = []
    for i in range(len(weights)):
        in_channel = weights[i].shape[1]
        out_channel = weights[i].shape[0]
        k_size = weights[i].shape[2]
        filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1,
            padding=k_size // 2, indice_key='asd', algo=spconv.ConvAlgo.Native
            ).to(device=weights[i].device)
        filter.weight.data[:] = weights[i].permute(2, 3, 1, 0).contiguous()[:]
        filter.bias.data = biases[i]
        nets.append(filter)
        if i != len(weights) - 1:
            nets.append(torch.nn.ReLU(inplace=True))
    return spconv.SparseSequential(*nets)
