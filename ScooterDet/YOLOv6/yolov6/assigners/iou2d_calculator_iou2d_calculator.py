def iou2d_calculator(bboxes1, bboxes2, mode='iou', is_aligned=False, scale=
    1.0, dtype=None):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""
    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
        bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
            empty. If ``is_aligned `` is ``True``, then m and n must be
            equal.
        mode (str): "iou" (intersection over union), "iof" (intersection
            over foreground), or "giou" (generalized intersection over
            union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert bboxes1.size(-1) in [0, 4, 5]
    assert bboxes2.size(-1) in [0, 4, 5]
    if bboxes2.size(-1) == 5:
        bboxes2 = bboxes2[..., :4]
    if bboxes1.size(-1) == 5:
        bboxes1 = bboxes1[..., :4]
    if dtype == 'fp16':
        bboxes1 = cast_tensor_type(bboxes1, scale, dtype)
        bboxes2 = cast_tensor_type(bboxes2, scale, dtype)
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        if not overlaps.is_cuda and overlaps.dtype == torch.float16:
            overlaps = overlaps.float()
        return overlaps
    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
