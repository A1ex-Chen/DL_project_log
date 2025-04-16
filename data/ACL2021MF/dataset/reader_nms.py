def nms(dets, classes, hierarchy, thresh=0.8):
    scores = [findall(hierarchy, filter_=lambda node: node.LabelName.lower(
        ) == cls)[0].height for cls in classes]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = np.array(scores)
    order = scores.argsort()
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        keep_condition = np.logical_or(scores[order[1:]] <= scores[i], 
            inter / (areas[i] + areas[order[1:]] - inter) <= thresh)
        inds = np.where(keep_condition)[0]
        order = order[inds + 1]
    return keep
