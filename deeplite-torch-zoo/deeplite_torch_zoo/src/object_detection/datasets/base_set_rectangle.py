def set_rectangle(self):
    """Sets the shape of bounding boxes for YOLO detections as rectangles."""
    bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
    nb = bi[-1] + 1
    s = np.array([x.pop('shape') for x in self.labels])
    ar = s[:, 0] / s[:, 1]
    irect = ar.argsort()
    self.im_files = [self.im_files[i] for i in irect]
    self.labels = [self.labels[i] for i in irect]
    ar = ar[irect]
    shapes = [[1, 1]] * nb
    for i in range(nb):
        ari = ar[bi == i]
        mini, maxi = ari.min(), ari.max()
        if maxi < 1:
            shapes[i] = [maxi, 1]
        elif mini > 1:
            shapes[i] = [1, 1 / mini]
    self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride +
        self.pad).astype(int) * self.stride
    self.batch = bi
