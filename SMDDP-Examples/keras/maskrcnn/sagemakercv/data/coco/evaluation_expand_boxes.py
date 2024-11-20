def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    w_half = boxes[:, 2] * 0.5
    h_half = boxes[:, 3] * 0.5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half
    w_half *= scale
    h_half *= scale
    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp
