def start(self):
    self._server_socket.bind((self._host, self._port))
    self._port = self._server_socket.getsockname()[1]
    self._server_socket.listen()
    self._sentinel.start()
    self._acceptor.start()
    logger.debug('DeepView is listening for connections on (%s:%d).', self.
        _host, self._port)
