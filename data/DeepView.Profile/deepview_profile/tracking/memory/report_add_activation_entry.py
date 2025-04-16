def add_activation_entry(self, operation_name, size_bytes, stack_context):
    cursor = self._connection.cursor()
    cursor.execute(queries.add_activation_entry, (operation_name, size_bytes))
    self._add_stack_frames(cursor=cursor, entry_id=cursor.lastrowid,
        entry_type=queries.EntryType.Activation, stack_context=stack_context)
    return self
