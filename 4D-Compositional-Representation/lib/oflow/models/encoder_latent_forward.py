def forward(self, inputs, c=None, **kwargs):
    """ Performs a forward pass through the network.

        Args:
            inputs (tensor): inputs
            c (tensor): latent conditioned code c
        """
    batch_size, n_t, T, _ = inputs.shape
    if self.dim == 3:
        inputs = inputs[:, 0]
    else:
        inputs = inputs.transpose(1, 2).contiguous().view(batch_size, T, -1)
    net = self.fc_pos(inputs)
    for i in range(self.n_blocks):
        if self.c_dim != 0:
            net_c = self.c_layers[i](c).unsqueeze(1)
            net = net + net_c
        net = self.blocks[i](net)
        if i < self.n_blocks - 1:
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2)
    net = self.pool(net, dim=1)
    mean = self.fc_mean(net)
    logstd = self.fc_logstd(net)
    return mean, logstd
