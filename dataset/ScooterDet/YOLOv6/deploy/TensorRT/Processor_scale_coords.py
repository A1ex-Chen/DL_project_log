def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] /
            img0_shape[1])]
        if self.scale_exact:
            gain = [img1_shape[0] / img0_shape[0], img1_shape[1] /
                img0_shape[1]]
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - 
            img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    if self.scale_exact:
        coords[:, [0, 2]] /= gain[1]
    else:
        coords[:, [0, 2]] /= gain[0]
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
