def __call__(self, labels=None, image=None):
    """Return updated labels and image with added border."""
    if labels is None:
        labels = {}
    img = labels.get('img') if image is None else image
    shape = img.shape[:2]
    new_shape = labels.pop('rect_shape', self.new_shape)
    if isinstance(new_shape, int):
        new_shape = new_shape, new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not self.scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if self.auto:
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
    elif self.scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    if self.center:
        dw /= 2
        dh /= 2
    if labels.get('ratio_pad'):
        labels['ratio_pad'] = labels['ratio_pad'], (dw, dh)
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh +
        0.1))
    left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw +
        0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.
        BORDER_CONSTANT, value=(114, 114, 114))
    if len(labels):
        labels = self._update_labels(labels, ratio, dw, dh)
        labels['img'] = img
        labels['resized_shape'] = new_shape
        return labels
    else:
        return img
