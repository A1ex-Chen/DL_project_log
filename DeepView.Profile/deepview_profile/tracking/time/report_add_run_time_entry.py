def add_run_time_entry(self, operation_name, forward_ms, backward_ms,
    stack_context):
    cursor = self._connection.cursor()
    cursor.execute(queries.add_run_time_entry, (operation_name, forward_ms,
        backward_ms))
    entry_id = cursor.lastrowid

    def stack_frame_generator():
        for idx, frame in enumerate(stack_context.frames):
            yield idx, frame.file_path, frame.line_number, entry_id
    cursor.executemany(queries.add_stack_frame, stack_frame_generator())
