def _style_pooling(self, x, eps=1e-05):
    N, C, _, _ = x.size()
    channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
    channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
    channel_std = channel_var.sqrt()
    t = torch.cat((channel_mean, channel_std), dim=2)
    return t
