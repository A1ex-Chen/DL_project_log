def reset(self):
    self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes +
        1), dtype=np.int64)
    self._predictions = []
    if self.label_group:
        self._conf_matrix_reduced = np.zeros((self.n_merged_cls + 1, self.
            n_merged_cls + 1), dtype=np.int64)
