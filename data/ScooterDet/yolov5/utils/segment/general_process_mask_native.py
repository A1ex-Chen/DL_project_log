def process_mask_native(protos, masks_in, bboxes, shape):
    """
    Crop after upsample.
    protos: [mask_dim, mask_h, mask_w]
    masks_in: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape: input_image_size, (h, w)

    return: h, w, n
    """
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2
    top, left = int(pad[1]), int(pad[0])
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]
    masks = F.interpolate(masks[None], shape, mode='bilinear',
        align_corners=False)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.5)
