def _download(self):
    raw = self.root.joinpath('raw')
    raw.mkdir(parents=True)
    raw_data = raw.joinpath('raw.pickle')
    shutil.copy(self.store, raw_data)
    self._preprocess(raw_data)
