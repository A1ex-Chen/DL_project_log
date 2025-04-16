def __init__(self, f: io.BufferedReader):
    self.f = f
    self.total_size = os.fstat(f.fileno()).st_size
    self.pbar = tqdm(total=self.total_size, leave=False)
    self.read = f.read
    f.read = self._read
