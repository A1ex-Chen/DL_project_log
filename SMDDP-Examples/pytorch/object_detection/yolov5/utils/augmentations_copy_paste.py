def copy_paste(im, labels, segments, p=0.5):
    n = len(segments)
    if p and n:
        h, w, c = im.shape
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])
            if (ioa < 0.3).all():
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1,
                    (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)
        i = result > 0
        im[i] = result[i]
    return im, labels, segments
