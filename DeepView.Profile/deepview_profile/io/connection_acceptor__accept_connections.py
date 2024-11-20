def _accept_connections(self):
    try:
        while True:
            read_ready, _, _ = select.select([self._server_socket, self.
                _sentinel.read_pipe], [], [])
            if self._sentinel.should_exit(read_ready):
                self._sentinel.consume_exit_signal()
                break
            socket, address = self._server_socket.accept()
            host, port = address
            logger.debug('Accepted a connection to (%s:%d).', host, port)
            self._handler_function(socket, address)
    except Exception:
        logging.exception(
            'DeepView has unexpectedly stopped accepting connections.')
