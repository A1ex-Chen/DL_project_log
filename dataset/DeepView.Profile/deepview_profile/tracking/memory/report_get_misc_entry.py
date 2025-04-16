def get_misc_entry(self, misc_size_type: MiscSizeType):
    cursor = self._connection.cursor()
    cursor.execute(queries.get_misc_entry, (misc_size_type.value,))
    return cursor.fetchone()[0]
