def _socket_read(self):
    buffer = b''
    message_length = -1
    try:
        while True:
            read_ready, _, _ = select.select([self._socket, self._sentinel.
                read_pipe], [], [])
            if self._sentinel.should_exit(read_ready):
                logger.debug('Connection (%s:%d) is being closed.', *self.
                    address)
                self._sentinel.consume_exit_signal()
                break
            data = self._socket.recv(4096)
            if len(data) == 0:
                logger.debug(
                    'Connection (%s:%d) has been closed by the client.', *
                    self.address)
                self._closed_handler(self.address)
                break
            buffer += data
            while True:
                if message_length <= 0:
                    if len(buffer) < 4:
                        break
                    message_length = struct.unpack('!I', buffer[:4])[0]
                    buffer = buffer[4:]
                if len(buffer) < message_length:
                    break
                try:
                    self._handler_function(buffer[:message_length], self.
                        address)
                finally:
                    buffer = buffer[message_length:]
                    message_length = -1
    except Exception:
        logger.exception('Connection unexpectedly stopping...')
