def build(self):
    self._connection.commit()
    return MemoryReport(self._connection)
