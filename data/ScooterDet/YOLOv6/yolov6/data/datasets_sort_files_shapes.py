def sort_files_shapes(self):
    """Sort by aspect ratio."""
    batch_num = self.batch_indices[-1] + 1
    s = self.shapes
    ar = s[:, 1] / s[:, 0]
    irect = ar.argsort()
    self.img_paths = [self.img_paths[i] for i in irect]
    self.labels = [self.labels[i] for i in irect]
    self.shapes = s[irect]
    ar = ar[irect]
    shapes = [[1, 1]] * batch_num
    for i in range(batch_num):
        ari = ar[self.batch_indices == i]
        mini, maxi = ari.min(), ari.max()
        if maxi < 1:
            shapes[i] = [1, maxi]
        elif mini > 1:
            shapes[i] = [1 / mini, 1]
    self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.
        stride + self.pad).astype(np.int_) * self.stride
