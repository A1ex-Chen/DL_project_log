@contextmanager
def _resumable_file_manager() ->'io.BufferedWriter':
    with open(incomplete_path, 'ab') as f:
        yield f
