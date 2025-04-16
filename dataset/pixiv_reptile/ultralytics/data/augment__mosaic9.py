def _mosaic9(self, labels):
    """Create a 3x3 image mosaic."""
    mosaic_labels = []
    s = self.imgsz
    hp, wp = -1, -1
    for i in range(9):
        labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
        img = labels_patch['img']
        h, w = labels_patch.pop('resized_shape')
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
        padw, padh = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)
        img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
        hp, wp = h, w
        labels_patch = self._update_labels(labels_patch, padw + self.border
            [0], padh + self.border[1])
        mosaic_labels.append(labels_patch)
    final_labels = self._cat_labels(mosaic_labels)
    final_labels['img'] = img9[-self.border[0]:self.border[0], -self.border
        [1]:self.border[1]]
    return final_labels
