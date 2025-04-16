def random_affine(img, labels=(), degrees=10, translate=0.1, scale=0.1,
    shear=10, new_shape=(640, 640)):
    """Applies Random affine transformation."""
    n = len(labels)
    if isinstance(new_shape, int):
        height = width = new_shape
    else:
        height, width = new_shape
    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees,
        scale, shear, translate)
    if (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue
            =(114, 114, 114))
    if n:
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(
            4, n).T
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]
    return img, labels
