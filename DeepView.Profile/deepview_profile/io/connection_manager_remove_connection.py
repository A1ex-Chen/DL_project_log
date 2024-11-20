def remove_connection(self, address):
    connection, state = self.get_connection_tuple(address)
    connection.stop()
    state.connected = False
    del self._connections[address]
    logger.debug('Removed connection to (%s:%d).', *address)
