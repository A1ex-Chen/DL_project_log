def random_perspective(im, targets=(), segments=(), degrees=10, translate=
    0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    M = T @ S @ R @ P @ C
    if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height),
                borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114))
    n = len(targets)
    new_segments = []
    if n:
        new = np.zeros((n, 4))
        segments = resample_segments(segments)
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T
            xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
            new[i] = segment2box(xy, width, height)
            new_segments.append(xy)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr
            =0.01)
        targets = targets[i]
        targets[:, 1:5] = new[i]
        new_segments = np.array(new_segments)[i]
    return im, targets, new_segments
