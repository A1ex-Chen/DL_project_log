def add_weight_entry(self, weight_name, size_bytes, grad_size_bytes,
    stack_context):
    if len(stack_context.frames) == 0:
        raise ValueError(
            'Adding weight entry with no context to the breakdown.')
    for entry in self._traverse_and_insert(self._weight_root, weight_name,
        stack_context):
        entry.add_weight_size(size_bytes, grad_size_bytes)
    return self
