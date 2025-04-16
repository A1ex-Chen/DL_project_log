def merge_file_(self, another_file):
    index = IndexedDataset(another_file, read_data=False)
    assert index.dtype == self.dtype
    begin = self.data_offsets[-1]
    for offset in index.data_offsets[1:]:
        self.data_offsets.append(begin + offset)
    self.sizes.extend(index.sizes)
    begin = self.dim_offsets[-1]
    for dim_offset in index.dim_offsets[1:]:
        self.dim_offsets.append(begin + dim_offset)
    with open(data_file_path(another_file), 'rb') as f:
        while True:
            data = f.read(1024)
            if data:
                self.out_file.write(data)
            else:
                break
