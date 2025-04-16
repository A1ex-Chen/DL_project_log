def _bn_convert(self):
    assert not self.training
    if self.bn_converted:
        return
    for i in range(self.num_convs):
        running_mean = self.bns[i].running_mean.data
        running_var = self.bns[i].running_var.data
        gamma = self.bns[i].weight.data
        beta = self.bns[i].bias.data
        bn_scale = gamma * torch.rsqrt(running_var + 1e-10)
        bn_bias = beta - bn_scale * running_mean
        self.subnet[i].weight.data = self.subnet[i
            ].weight.data * bn_scale.view(-1, 1, 1, 1)
        self.subnet[i].bias = torch.nn.Parameter(bn_bias)
    self.bn_converted = True
