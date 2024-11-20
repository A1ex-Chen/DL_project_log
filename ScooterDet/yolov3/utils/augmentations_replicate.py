def replicate(im, labels):
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = (x2 - x1 + (y2 - y1)) / 2
    for i in s.argsort()[:round(s.size * 0.5)]:
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0
            )
    return im, labels
