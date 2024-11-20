def copy_paste(img, labels, segments, probability=0.5):
    n = len(segments)
    if probability and n:
        h, w, c = img.shape
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])
            if (ioa < 0.3).all():
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1,
                    (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)
        i = result > 0
        img[i] = result[i]
    return img, labels, segments
