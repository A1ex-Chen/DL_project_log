def bbox_ioa(box1, box2, eps=1e-07):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.array): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.array): A numpy array of shape (m, 4) representing m bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.array): A numpy array of shape (n, m) representing the intersection over box2 area.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:,
        None], b2_x1)).clip(0) * (np.minimum(b1_y2[:, None], b2_y2) - np.
        maximum(b1_y1[:, None], b2_y1)).clip(0)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter_area / box2_area
