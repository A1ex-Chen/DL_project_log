def stop(self):
    self._sentinel.signal_exit()
    self._reader.join()
    self._socket.close()
    self._sentinel.stop()
