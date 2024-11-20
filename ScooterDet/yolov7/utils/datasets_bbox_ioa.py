def bbox_ioa(box1, box2):
    box2 = box2.transpose()
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0
        ) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
    return inter_area / box2_area
