def __del__(self):
    self._connection.close()
