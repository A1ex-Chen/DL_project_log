def pastein(image, labels, sample_labels, sample_images, sample_masks):
    h, w = image.shape[:2]
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])
        else:
            ioa = np.zeros(1)
        if (ioa < 0.3).all() and len(sample_labels
            ) and xmax > xmin + 20 and ymax > ymin + 20:
            sel_ind = random.randint(0, len(sample_labels) - 1)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
            r_w = int(ws * r_scale)
            r_h = int(hs * r_scale)
            if r_w > 10 and r_h > 10:
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    box = np.array([xmin, ymin, xmin + r_w, ymin + r_h],
                        dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[
                            sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])
                    image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop
    return labels
