def forward(self, p, c, **kwargs):
    """ Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned temporal code c
        """
    p = p.transpose(1, 2)
    batch_size, D, T = p.size()
    net = self.fc_p(p)
    net = self.block0(net, c)
    net = self.block1(net, c)
    net = self.block2(net, c)
    net = self.block3(net, c)
    net = self.block4(net, c)
    out = self.fc_out(self.actvn(self.bn(net, c)))
    out = out.squeeze(1)
    return out
