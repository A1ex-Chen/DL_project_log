def test():
    b, n = 2, 100
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)
    in_channels = 3
    feats = torch.rand(b * n, in_channels)
    x = [coords, feats.cuda()]
    net = UNetSCN(in_channels).cuda()
    out_feats = net(x)
    print('out_feats', out_feats.shape)
