def remove_background(img, labels, segments):
    n = len(segments)
    h, w, c = img.shape
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 
            255, 255), cv2.FILLED)
        result = cv2.bitwise_and(src1=img, src2=im_new)
        i = result > 0
        img_new[i] = result[i]
    return img_new, labels, segments
