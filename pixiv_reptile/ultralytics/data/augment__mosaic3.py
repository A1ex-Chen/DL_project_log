def _mosaic3(self, labels):
    """Create a 1x3 image mosaic."""
    mosaic_labels = []
    s = self.imgsz
    for i in range(3):
        labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
        img = labels_patch['img']
        h, w = labels_patch.pop('resized_shape')
        if i == 0:
            img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
            h0, w0 = h, w
            c = s, s, s + w, s + h
        elif i == 1:
            c = s + w0, s, s + w0 + w, s + h
        elif i == 2:
            c = s - w, s + h0 - h, s, s + h0
        padw, padh = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)
        img3[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
        labels_patch = self._update_labels(labels_patch, padw + self.border
            [0], padh + self.border[1])
        mosaic_labels.append(labels_patch)
    final_labels = self._cat_labels(mosaic_labels)
    final_labels['img'] = img3[-self.border[0]:self.border[0], -self.border
        [1]:self.border[1]]
    return final_labels
