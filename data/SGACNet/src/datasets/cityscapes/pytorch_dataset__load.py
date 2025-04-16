def _load(self, directory, filename):
    fp = os.path.join(self._data_dir, self.split, directory, filename)
    if os.path.splitext(fp)[-1] == '.npy':
        return np.load(fp)
    else:
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im
