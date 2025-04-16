def add_weight_entry(self, weight_name, size_bytes, grad_size_bytes,
    stack_context):
    cursor = self._connection.cursor()
    cursor.execute(queries.add_weight_entry, (weight_name, size_bytes,
        grad_size_bytes))
    self._add_stack_frames(cursor=cursor, entry_id=cursor.lastrowid,
        entry_type=queries.EntryType.Weight, stack_context=stack_context)
    return self
