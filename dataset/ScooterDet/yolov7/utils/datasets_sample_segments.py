def sample_segments(img, labels, segments, probability=0.5):
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0, w - 1), l[2].astype(int).clip(0,
                h - 1), l[3].astype(int).clip(0, w - 1), l[4].astype(int).clip(
                0, h - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            sample_labels.append(l[0])
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255,
                255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])
            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0
            mask[i] = result[i]
            sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])
    return sample_labels, sample_images, sample_masks
