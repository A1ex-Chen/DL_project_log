def send_bytes(self, raw_bytes):
    self._socket.sendall(struct.pack('!I', len(raw_bytes)))
    self._socket.sendall(raw_bytes)
