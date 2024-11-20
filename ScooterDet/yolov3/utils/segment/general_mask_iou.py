def mask_iou(mask1, mask2, eps=1e-07):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, [N, M]
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = mask1.sum(1)[:, None] + mask2.sum(1)[None] - intersection
    return intersection / (union + eps)
