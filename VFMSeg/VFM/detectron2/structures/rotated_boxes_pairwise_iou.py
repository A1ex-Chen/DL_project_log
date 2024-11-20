def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) ->None:
    """
    Given two lists of rotated boxes of size N and M,
    compute the IoU (intersection over union)
    between **all** N x M pairs of boxes.
    The box order must be (x_center, y_center, width, height, angle).

    Args:
        boxes1, boxes2 (RotatedBoxes):
            two `RotatedBoxes`. Contains N & M rotated boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)
