def stack_frame_generator():
    for idx, frame in enumerate(stack_context.frames):
        yield correlation_id, idx, frame.file_path, frame.line_number
