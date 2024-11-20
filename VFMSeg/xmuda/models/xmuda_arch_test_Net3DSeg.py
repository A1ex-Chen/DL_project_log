def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11
    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)
    feats = feats.cuda()
    net_3d = Net3DSeg(num_seg_classes, dual_head=True, backbone_3d='SCN',
        backbone_3d_kwargs={'in_channels': in_channels})
    net_3d.cuda()
    out_dict = net_3d({'x': [coords, feats]})
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)
