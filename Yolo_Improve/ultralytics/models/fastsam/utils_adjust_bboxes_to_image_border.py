def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """
    Adjust bounding boxes to stick to image border if they are within a certain threshold.

    Args:
        boxes (torch.Tensor): (n, 4)
        image_shape (tuple): (height, width)
        threshold (int): pixel threshold

    Returns:
        adjusted_boxes (torch.Tensor): adjusted bounding boxes
    """
    h, w = image_shape
    boxes[boxes[:, 0] < threshold, 0] = 0
    boxes[boxes[:, 1] < threshold, 1] = 0
    boxes[boxes[:, 2] > w - threshold, 2] = w
    boxes[boxes[:, 3] > h - threshold, 3] = h
    return boxes
