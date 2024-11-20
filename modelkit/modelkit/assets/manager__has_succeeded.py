def _has_succeeded(local_path):
    return os.path.exists(_success_file_path(local_path))
