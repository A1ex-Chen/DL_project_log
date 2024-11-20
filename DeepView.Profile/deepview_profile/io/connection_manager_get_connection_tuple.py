def get_connection_tuple(self, address):
    if address not in self._connections:
        host, port = address
        raise NoConnectionError('Connection to ({}:{}) does not exist.'.
            format(host, port))
    return self._connections[address]
