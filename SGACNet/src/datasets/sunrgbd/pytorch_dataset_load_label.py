def load_label(self, idx):
    if self.camera is None:
        label_dir = self.label_dir[self._split]['list']
    else:
        label_dir = self.label_dir[self._split]['dict'][self.camera]
    label = np.load(os.path.join(self._data_dir, label_dir[idx])).astype(np
        .uint8)
    return label
