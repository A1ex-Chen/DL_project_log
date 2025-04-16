def _serialize_file_ref(file_ref, context):
    file_path, line_number = context
    file_ref.line_number = line_number
    file_ref.file_path.components.extend(file_path.split(os.sep))
