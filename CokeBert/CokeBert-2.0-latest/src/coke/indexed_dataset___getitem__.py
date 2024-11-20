def __getitem__(self, i):
    self.check_index(i)
    tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
    a = np.empty(tensor_size, dtype=self.dtype)
    ptx = self.cache_index[i]
    np.copyto(a, self.cache[ptx:ptx + a.size])
    item = torch.from_numpy(a).long()
    if self.fix_lua_indexing:
        item -= 1
    return item
