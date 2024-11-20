def test_Net2DSeg():
    batch_size = 2
    img_width = 400
    img_height = 225
    num_coords = 2000
    num_classes = 11
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords //
        batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords //
        batch_size, 1))
    img_indices = torch.cat([u, v], 2)
    img = img.cuda()
    img_indices = img_indices.cuda()
    net_2d = Net2DSeg(num_classes, backbone_2d='UNetResNet34',
        backbone_2d_kwargs={}, dual_head=True)
    net_2d.cuda()
    out_dict = net_2d({'img': img, 'img_indices': img_indices})
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)
