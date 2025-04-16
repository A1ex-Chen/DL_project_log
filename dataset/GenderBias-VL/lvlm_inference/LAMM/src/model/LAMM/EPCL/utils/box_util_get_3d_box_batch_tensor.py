def get_3d_box_batch_tensor(box_size, angle, center):
    assert isinstance(box_size, torch.Tensor)
    assert isinstance(angle, torch.Tensor)
    assert isinstance(center, torch.Tensor)
    reshape_final = False
    if angle.ndim == 2:
        assert box_size.ndim == 3
        assert center.ndim == 3
        bsize = box_size.shape[0]
        nprop = box_size.shape[1]
        box_size = box_size.reshape(-1, box_size.shape[-1])
        angle = angle.reshape(-1)
        center = center.reshape(-1, 3)
        reshape_final = True
    input_shape = angle.shape
    R = roty_batch_tensor(angle)
    l = torch.unsqueeze(box_size[..., 0], -1)
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)
    corners_3d = torch.zeros(tuple(list(input_shape) + [8, 3]), device=
        box_size.device, dtype=torch.float32)
    corners_3d[..., :, 0] = torch.cat((l / 2, l / 2, -l / 2, -l / 2, l / 2,
        l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = torch.cat((h / 2, h / 2, h / 2, h / 2, -h / 2, 
        -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = torch.cat((w / 2, -w / 2, -w / 2, w / 2, w / 2,
        -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = torch.matmul(corners_3d, R.permute(tlist))
    corners_3d += torch.unsqueeze(center, -2)
    if reshape_final:
        corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
    return corners_3d
