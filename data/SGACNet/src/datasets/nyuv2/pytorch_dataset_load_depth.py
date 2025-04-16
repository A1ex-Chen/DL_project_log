def load_depth(self, idx):
    if self._depth_mode == 'raw':
        return self._load(self.DEPTH_RAW_DIR, self._filenames[idx])
    else:
        return self._load(self.DEPTH_DIR, self._filenames[idx])
