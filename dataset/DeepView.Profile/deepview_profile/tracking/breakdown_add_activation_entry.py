def add_activation_entry(self, operation_name, size_bytes, stack_context):
    if len(stack_context.frames) == 0:
        raise ValueError(
            'Adding activation entry with no context to the breakdown.')
    for entry in self._traverse_and_insert(self._operation_root,
        operation_name, stack_context):
        entry.add_activation_size(size_bytes)
    return self
