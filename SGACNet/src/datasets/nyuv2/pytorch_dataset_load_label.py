def load_label(self, idx):
    return self._load(self.LABELS_DIR_FMT.format(self._n_classes), self.
        _filenames[idx])
