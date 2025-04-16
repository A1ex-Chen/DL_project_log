def add_misc_entry(self, size_type: MiscSizeType, size_bytes):
    cursor = self._connection.cursor()
    cursor.execute(queries.add_misc_entry, (size_type.value, size_bytes))
    return self
