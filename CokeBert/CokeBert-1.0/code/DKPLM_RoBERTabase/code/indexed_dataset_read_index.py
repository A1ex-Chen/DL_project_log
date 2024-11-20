def read_index(self, path):
    with open(index_file_path(path), 'rb') as f:
        magic = f.read(8)
        assert magic == b'TNTIDX\x00\x00'
        version = f.read(8)
        assert struct.unpack('<Q', version) == (1,)
        code, self.element_size = struct.unpack('<QQ', f.read(16))
        self.dtype = dtypes[code]
        self.size, self.s = struct.unpack('<QQ', f.read(16))
        self.dim_offsets = read_longs(f, self.size + 1)
        self.data_offsets = read_longs(f, self.size + 1)
        self.sizes = read_longs(f, self.s)
