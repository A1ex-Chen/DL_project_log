def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[..., top:bottom, left:right]
    masks = F.interpolate(masks, shape, mode='bilinear', align_corners=False)
    return masks
