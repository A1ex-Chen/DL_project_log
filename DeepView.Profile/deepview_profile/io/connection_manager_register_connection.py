def register_connection(self, opened_socket, address):
    self._connections[address] = Connection(opened_socket, address, self.
        _message_handler, self._closed_handler), ConnectionState()
    self._connections[address][0].start()
