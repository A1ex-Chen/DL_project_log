def _success_file_path(local_path):
    if os.path.isdir(local_path):
        return os.path.join(local_path, '.SUCCESS')
    else:
        dirn, fn = os.path.split(local_path)
        return os.path.join(dirn, f'.{fn}.SUCCESS')
