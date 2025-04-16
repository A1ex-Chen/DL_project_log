def cal_iou_3d(bbox1, bbox2):
    """
        box [x1, y1, z1, l, w, h]
    """
    bbox1 = [round(bbox1[0] - abs(bbox1[3] / 2), 3), round(bbox1[1] - abs(
        bbox1[4] / 2), 3), round(bbox1[2] - abs(bbox1[5] / 2), 3), round(
        bbox1[0] + abs(bbox1[3] / 2), 3), round(bbox1[1] + abs(bbox1[4] / 2
        ), 3), round(bbox1[2] + abs(bbox1[5]) / 2, 3)]
    bbox2 = [round(bbox2[0] - abs(bbox2[3] / 2), 3), round(bbox2[1] - abs(
        bbox2[4] / 2), 3), round(bbox2[2] - abs(bbox2[5] / 2), 3), round(
        bbox2[0] + abs(bbox2[3] / 2), 3), round(bbox2[1] + abs(bbox2[4] / 2
        ), 3), round(bbox2[2] + abs(bbox2[5]) / 2, 3)]
    x1, y1, z1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), max(bbox1
        [2], bbox2[2])
    x2, y2, z2 = min(bbox1[3], bbox2[3]), min(bbox1[4], bbox2[4]), min(bbox1
        [5], bbox2[5])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    area1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] -
        bbox1[2])
    area2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] -
        bbox2[2])
    uni_area = area1 + area2 - inter_area
    iou = inter_area / uni_area
    return iou
