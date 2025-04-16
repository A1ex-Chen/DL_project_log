def seprate_point_cloud(xyz, num_points, crop, fixed_points=None,
    padding_zeros=False):
    """
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    """
    _, n, c = xyz.shape
    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop
        points = points.unsqueeze(0)
        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()
        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze
            (1), p=2, dim=-1)
        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]
        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)
        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)
        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)
    input_data = torch.cat(INPUT, dim=0)
    crop_data = torch.cat(CROP, dim=0)
    return input_data.contiguous(), crop_data.contiguous()
