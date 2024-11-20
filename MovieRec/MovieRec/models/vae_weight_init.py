def weight_init(self, m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
