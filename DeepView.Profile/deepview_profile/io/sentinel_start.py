def start(self):
    self._read_pipe, self._write_pipe = os.pipe()
