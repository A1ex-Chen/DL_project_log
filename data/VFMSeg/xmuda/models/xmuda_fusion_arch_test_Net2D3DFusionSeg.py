def test_Net2D3DFusionSeg():
    batch_size = 2
    img_width = 400
    img_height = 225
    num_coords = 2000
    num_classes = 11
    full_scale = 4096
    in_channels = 1
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords //
        batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords //
        batch_size, 1))
    img_indices = torch.cat([u, v], 2)
    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)
    feats = feats.cuda()
    img = img.cuda()
    img_indices = img_indices.cuda()
    net_fuse = Net2D3DFusionSeg(num_classes, backbone_2d='UNetResNet34',
        backbone_2d_kwargs={}, backbone_3d='SCN', backbone_3d_kwargs={
        'in_channels': in_channels}, dual_head=True)
    net_fuse.cuda()
    out_dict = net_fuse({'img': img, 'img_indices': img_indices, 'x': [
        coords, feats]})
    for k, v in out_dict.items():
        print('Net2D3DFusionSeg:', k, v.shape)
