@numba.jit('float32[:, :], float32, float32, float32, uint32', nopython=True)
def soft_nms_jit(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i
        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        pos = i + 1
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts
        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        pos = i + 1
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = min(tx2, x2) - max(tx1, x1) + 1
            if iw > 0:
                ih = min(ty2, y2) - max(ty1, y1) + 1
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - 
                        iw * ih)
                    ov = iw * ih / ua
                    if method == 1:
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:
                        weight = np.exp(-(ov * ov) / sigma)
                    elif ov > Nt:
                        weight = 0
                    else:
                        weight = 1
                    boxes[pos, 4] = weight * boxes[pos, 4]
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1
            pos = pos + 1
    keep = [i for i in range(N)]
    return keep
