def _add_stack_frames(self, cursor, entry_id, entry_type: queries.EntryType,
    stack_context):
    cursor.execute(queries.add_correlation_entry, (entry_id, entry_type.value))
    correlation_id = cursor.lastrowid

    def stack_frame_generator():
        for idx, frame in enumerate(stack_context.frames):
            yield correlation_id, idx, frame.file_path, frame.line_number
    cursor.executemany(queries.add_stack_frame, stack_frame_generator())
