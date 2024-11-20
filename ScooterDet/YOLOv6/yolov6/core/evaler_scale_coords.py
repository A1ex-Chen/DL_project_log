def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    gain = ratio_pad[0]
    pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [0, 2]] /= gain[1]
    coords[:, [1, 3]] -= pad[1]
    coords[:, [1, 3]] /= gain[0]
    if isinstance(coords, torch.Tensor):
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
    else:
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
    return coords
