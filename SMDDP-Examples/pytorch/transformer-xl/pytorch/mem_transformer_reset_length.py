def reset_length(self, tgt_len, ext_len, mem_len):
    if tgt_len < 1:
        raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
    if ext_len < 0:
        raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
    if mem_len < 0:
        raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
    self.tgt_len = tgt_len
    self.mem_len = mem_len
    self.ext_len = ext_len
