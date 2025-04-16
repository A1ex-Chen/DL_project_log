@staticmethod
def from_here(project_root, start_from=1):
    """
        Returns the current call stack when invoked.
        """
    stack = inspect.stack()
    context = []
    try:
        for frame_info in stack[start_from:]:
            if not (frame_info.filename.startswith(project_root) or
                find_pattern_match(frame_info.filename)):
                continue
            if 'self' not in frame_info.frame.f_locals:
                continue
            if not isinstance(frame_info.frame.f_locals['self'], torch.nn.
                Module):
                continue
            context.append(SourceLocation(file_path=os.path.relpath(
                frame_info.filename, start=project_root), line_number=
                frame_info.lineno, module_id=id(frame_info.frame.f_locals[
                'self'])))
        return CallStack(context)
    finally:
        del stack
