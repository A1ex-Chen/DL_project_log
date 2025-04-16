def test():
    b, c, h, w = 2, 20, 120, 160
    image = torch.randn(b, 3, h, w).cuda()
    net = UNetResNet34(pretrained=True)
    net.cuda()
    feats = net(image)
    print('feats', feats.shape)
