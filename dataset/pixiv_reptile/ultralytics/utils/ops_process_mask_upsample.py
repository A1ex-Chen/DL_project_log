def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
    but is slower.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        (torch.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode='bilinear',
        align_corners=False)[0]
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.0)
