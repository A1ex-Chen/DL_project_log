def forward(self, p, z, c, **kwargs):
    """ Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
    p = p.transpose(1, 2)
    batch_size, D, T = p.size()
    net = self.fc_p(p)
    if self.z_dim != 0:
        net_z = self.fc_z(z).unsqueeze(2)
        net = net + net_z
    net = self.block0(net, c)
    net = self.block1(net, c)
    net = self.block2(net, c)
    net = self.block3(net, c)
    net = self.block4(net, c)
    out = self.fc_out(self.actvn(self.bn(net, c)))
    out = out.squeeze(1)
    return out
