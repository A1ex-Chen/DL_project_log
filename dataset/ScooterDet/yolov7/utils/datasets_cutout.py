def cutout(image, labels):
    h, w = image.shape[:2]
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125
        ] * 16
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in
            range(3)]
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])
            labels = labels[ioa < 0.6]
    return labels
