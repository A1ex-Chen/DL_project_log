def stop(self):
    self._sentinel.signal_exit()
    self._acceptor.join()
    self._server_socket.close()
    self._sentinel.stop()
    logging.debug('DeepView has stopped listening for connections on (%s:%d).',
        self._host, self._port)
