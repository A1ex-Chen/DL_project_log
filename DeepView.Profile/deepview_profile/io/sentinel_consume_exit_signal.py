def consume_exit_signal(self):
    os.read(self._read_pipe, 1)
