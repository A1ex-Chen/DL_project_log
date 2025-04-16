def prefetch(self, indices):
    if all(i in self.cache_index for i in indices):
        return
    indices = sorted(set(indices))
    total_size = 0
    for i in indices:
        total_size += self.data_offsets[i + 1] - self.data_offsets[i]
    self.cache = np.empty(total_size, dtype=self.dtype)
    ptx = 0
    self.cache_index.clear()
    for i in indices:
        self.cache_index[i] = ptx
        size = self.data_offsets[i + 1] - self.data_offsets[i]
        a = self.cache[ptx:ptx + size]
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        ptx += size
