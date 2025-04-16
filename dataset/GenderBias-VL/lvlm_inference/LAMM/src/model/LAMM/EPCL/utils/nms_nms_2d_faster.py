def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]
    area = (x2 - x1) * (y2 - y1)
    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[I[:last - 1]])
        yy1 = np.maximum(y1[i], y1[I[:last - 1]])
        xx2 = np.minimum(x2[i], x2[I[:last - 1]])
        yy2 = np.minimum(y2[i], y2[I[:last - 1]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        if old_type:
            o = w * h / area[I[:last - 1]]
        else:
            inter = w * h
            o = inter / (area[i] + area[I[:last - 1]] - inter)
        I = np.delete(I, np.concatenate(([last - 1], np.where(o >
            overlap_threshold)[0])))
    return pick
