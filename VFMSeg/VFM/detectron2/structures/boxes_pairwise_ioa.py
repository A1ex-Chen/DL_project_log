def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) ->torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()
    inter = pairwise_intersection(boxes1, boxes2)
    ioa = torch.where(inter > 0, inter / area2, torch.zeros(1, dtype=inter.
        dtype, device=inter.device))
    return ioa
