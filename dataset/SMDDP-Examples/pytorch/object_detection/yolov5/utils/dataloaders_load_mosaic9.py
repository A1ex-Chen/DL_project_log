def load_mosaic9(self, index):
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)
    random.shuffle(indices)
    hp, wp = -1, -1
    for i, index in enumerate(indices):
        img, _, (h, w) = self.load_image(index)
        if i == 0:
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
            h0, w0 = h, w
            c = s, s, s + w, s + h
        elif i == 1:
            c = s, s - h, s + w, s
        elif i == 2:
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:
            c = s - w, s + h0 - hp - h, s, s + h0 - hp
        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)
        labels, segments = self.labels[index].copy(), self.segments[index
            ].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]
        hp, wp = h, w
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])
    segments9 = [(x - c) for x in segments9]
    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)
    img9, labels9 = random_perspective(img9, labels9, segments9, degrees=
        self.hyp['degrees'], translate=self.hyp['translate'], scale=self.
        hyp['scale'], shear=self.hyp['shear'], perspective=self.hyp[
        'perspective'], border=self.mosaic_border)
    return img9, labels9
