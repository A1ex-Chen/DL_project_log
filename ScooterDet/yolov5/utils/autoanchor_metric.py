def metric(k, wh):
    r = wh[:, None] / k[None]
    x = torch.min(r, 1 / r).min(2)[0]
    return x, x.max(1)[0]
