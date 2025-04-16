def broadcast(self, string_message):
    for _, (connection, _) in self._connections.items():
        connection.write_string_message(string_message)
