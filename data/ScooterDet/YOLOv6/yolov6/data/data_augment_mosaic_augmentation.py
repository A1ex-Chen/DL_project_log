def mosaic_augmentation(shape, imgs, hs, ws, labels, hyp, specific_shape=
    False, target_height=640, target_width=640):
    """Applies Mosaic augmentation."""
    assert len(imgs
        ) == 4, 'Mosaic augmentation of current version only supports 4 images.'
    labels4 = []
    if not specific_shape:
        if isinstance(shape, list) or isinstance(shape, np.ndarray):
            target_height, target_width = shape
        else:
            target_height = target_width = shape
    yc, xc = (int(random.uniform(x // 2, 3 * x // 2)) for x in (
        target_height, target_width))
    for i in range(len(imgs)):
        img, h, w = imgs[i], hs[i], ws[i]
        if i == 0:
            img4 = np.full((target_height * 2, target_width * 2, img.shape[
                2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 
                target_width * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height *
                2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(
                target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b
        labels_per_img = labels[i].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2
                ) + padw
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2
                ) + padh
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2
                ) + padw
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2
                ) + padh
            labels_per_img[:, 1:] = boxes
        labels4.append(labels_per_img)
    labels4 = np.concatenate(labels4, 0)
    labels4[:, 1::2] = np.clip(labels4[:, 1::2], 0, 2 * target_width)
    labels4[:, 2::2] = np.clip(labels4[:, 2::2], 0, 2 * target_height)
    img4, labels4 = random_affine(img4, labels4, degrees=hyp['degrees'],
        translate=hyp['translate'], scale=hyp['scale'], shear=hyp['shear'],
        new_shape=(target_height, target_width))
    return img4, labels4
