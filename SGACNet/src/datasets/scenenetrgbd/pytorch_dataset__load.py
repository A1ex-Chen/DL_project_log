def _load(self, directory, filename):
    fp = os.path.join(self._data_dir, self.split, directory, filename)
    im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
