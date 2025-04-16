def cal_iou(bbox1, bbox2):
    ixmin = np.maximum(bbox1[0], bbox2[0])
    iymin = np.maximum(bbox1[1], bbox2[1])
    ixmax = np.minimum(bbox1[2], bbox2[2])
    iymax = np.minimum(bbox1[3], bbox2[3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inters = iw * ih
    uni = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) + (bbox1[2] - bbox1[0]
        ) * (bbox1[3] - bbox1[1]) - inters
    overlaps = inters / uni
    return overlaps
