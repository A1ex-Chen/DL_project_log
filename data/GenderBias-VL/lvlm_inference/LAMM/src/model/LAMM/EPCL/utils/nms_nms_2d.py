def nms_2d(boxes, overlap_threshold):
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
        suppress = [last - 1]
        for pos in range(last - 1):
            j = I[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = xx2 - xx1
            h = yy2 - yy1
            if w > 0 and h > 0:
                o = w * h / area[j]
                print('Overlap is', o)
                if o > overlap_threshold:
                    suppress.append(pos)
        I = np.delete(I, suppress)
    return pick
