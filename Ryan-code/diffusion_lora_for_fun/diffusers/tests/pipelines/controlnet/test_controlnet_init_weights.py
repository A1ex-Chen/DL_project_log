def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(1.0)
