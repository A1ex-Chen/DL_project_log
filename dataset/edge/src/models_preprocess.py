def preprocess(self, img):
    desired_h, desired_w = self.img_size
    img = cv2.resize(img, (desired_w, round(img.shape[0] * desired_w / img.
        shape[1])))
    if img.shape[0] >= desired_h:
        img = cv2.resize(img, (desired_w, desired_h))
    else:
        img = cv2.copyMakeBorder(img, 0, desired_h - img.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = img[np.newaxis, ...]
    return img.astype(np.float32)
