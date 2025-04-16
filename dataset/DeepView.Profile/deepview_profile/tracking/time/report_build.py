def build(self):
    self._connection.commit()
    return OperationRunTimeReport(self._connection)
