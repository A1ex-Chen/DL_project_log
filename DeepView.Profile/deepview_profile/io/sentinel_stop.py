def stop(self):
    os.close(self._write_pipe)
    os.close(self._read_pipe)
    self._read_pipe = None
    self._write_pipe = None
