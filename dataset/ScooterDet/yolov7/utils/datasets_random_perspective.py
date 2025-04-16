def random_perspective(img, targets=(), segments=(), degrees=10, translate=
    0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1.1 + scale)
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
            img = cv2.warpPerspective(img, M, dsize=(width, height),
                borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114))
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                new[i] = segment2box(xy, width, height)
        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                ).reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))
                ).reshape(4, n).T
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr
            =0.01 if use_segments else 0.1)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return img, targets
