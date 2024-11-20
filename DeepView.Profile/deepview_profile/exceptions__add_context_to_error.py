def _add_context_to_error(error, project_root, file_path, line_number):
    error.with_file_context(file_path=os.path.relpath(file_path, start=
        project_root), line_number=line_number)
