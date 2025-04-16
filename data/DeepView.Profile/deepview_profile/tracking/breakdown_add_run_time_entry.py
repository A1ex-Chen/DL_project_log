def add_run_time_entry(self, operation_name, forward_ms, backward_ms,
    stack_context):
    if len(stack_context.frames) == 0:
        raise ValueError(
            'Adding run time entry with no context to the breakdown.')
    for entry in self._traverse_and_insert(self._operation_root,
        operation_name, stack_context):
        entry.add_run_time(forward_ms, backward_ms)
    return self
