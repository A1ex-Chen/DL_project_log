def __getitem__(self, index):
    frames = None
    if isinstance(index, int):
        ret, frames = self.read(index)
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    elif isinstance(index, slice):
        frames = np.stack([self[ii] for ii in range(*index.indices(len(self)))]
            )
    elif isinstance(index, range):
        frames = np.stack([self[ii] for ii in index])
    elif isinstance(index, tuple):
        if isinstance(index[0], slice):
            indices = range(*index[0].indices(len(self)))
        elif isinstance(index[0], (np.integer, int)):
            indices = int(index[0])
        else:
            indices = None
        if indices is not None:
            frames = self[indices]
            for cnt, idx in enumerate(index[1:]):
                if isinstance(idx, slice):
                    ix = range(*idx.indices(self.shape[cnt + 1]))
                elif isinstance(idx, int):
                    ix = range(idx - 1, idx)
                else:
                    continue
                if frames.ndim == 4:
                    cnt = cnt + 1
                frames = np.take(frames, ix, axis=cnt)
    if self.remove_leading_singleton and frames is not None:
        if frames.shape[0] == 1:
            frames = frames[0]
    return frames
