def signal_exit(self):
    os.write(self._write_pipe, b'\x00')
