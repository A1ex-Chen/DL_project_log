def interpolate_func(inputs):
    idx = torch.from_numpy(np.array([[[0, 1, 2], [1, 2, 3]]])).int().cuda()
    weight = torch.from_numpy(np.array([[[1, 1, 1], [2, 2, 2]]])).float().cuda(
        )
    interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)
    return interpolated_feats
