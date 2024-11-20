def stop(self):
    for _, (connection, state) in self._connections.items():
        connection.stop()
        state.connected = False
    self._connections.clear()
