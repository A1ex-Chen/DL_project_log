def agg_channel(self, x, pool='max'):
    b, c, h, w = x.size()
    x = x.view(b, c, h * w)
    x = x.permute(0, 2, 1)
    if pool == 'max':
        x = F.max_pool1d(x, int(c))
    elif pool == 'avg':
        x = F.avg_pool1d(x, int(c))
    x = x.permute(0, 2, 1)
    x = x.view(b, 1, h, w)
    return x
